"""End-to-end build:
   ingest → SPADL → xT + VAEP → interactions → JOI per match → JOI90 per pair.
Writes intermediate parquets and the final chemistry.json with both metrics.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from chemistry import ingest, pipeline, joi, minutes
from chemistry.vaep_model import load_vaep_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build")

DATA = Path("data")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)


def run(use_heuristic_vaep: bool, fetch: bool = True, metric: str = "both") -> None:
    if fetch:
        log.info("Fetching StatsBomb open data for target competitions")
        ingest.fetch_all_targets()
    else:
        log.info("Skipping fetch (--no-fetch)")

    log.info("Converting events to SPADL")
    for comp_dir in sorted((DATA / "raw").iterdir()):
        if comp_dir.is_dir():
            try:
                pipeline.convert_competition(comp_dir)
            except Exception as exc:
                log.warning("SPADL conversion failed for %s: %s", comp_dir.name, exc)

    # Always score with xT (primary metric)
    if metric in ("xt", "both"):
        _score_with_xt()

    # Score with VAEP if requested
    if metric in ("vaep", "both"):
        _score_with_vaep(use_heuristic_vaep)

    # Build JOI from xT-scored parquets
    log.info("Computing JOI from xT-scored actions")
    all_per_match_xt, all_lineups = _collect_joi_and_lineups(DATA / "vaep")
    per_match_xt = pd.concat(all_per_match_xt, ignore_index=True) if all_per_match_xt else pd.DataFrame()
    lineups_df   = pd.concat(all_lineups,       ignore_index=True) if all_lineups       else pd.DataFrame()
    per_match_xt.to_parquet(OUT / "joi_per_match.parquet", index=False)
    lineups_df.to_parquet(OUT / "lineups.parquet", index=False)
    mins_provider = minutes.LineupsMinutes(lineups_df)
    joi90_xt = joi.joi90_window(per_match_xt, mins_provider)
    joi90_xt.to_parquet(OUT / "joi90_xt.parquet", index=False)
    log.info("Wrote %d xT pairs to outputs/joi90_xt.parquet", len(joi90_xt))
    # Keep legacy file for backward compat
    joi90_xt.to_parquet(OUT / "joi90.parquet", index=False)

    # Build JOI from VAEP-scored parquets (if they exist)
    vaep_scored_dir = DATA / "vaep_scored"
    joi90_vaep: pd.DataFrame | None = None
    if vaep_scored_dir.exists():
        log.info("Computing JOI from VAEP-scored actions")
        all_per_match_vaep, _ = _collect_joi_and_lineups(vaep_scored_dir)
        if all_per_match_vaep:
            per_match_vaep = pd.concat(all_per_match_vaep, ignore_index=True)
            joi90_vaep = joi.joi90_window(per_match_vaep, mins_provider)
            joi90_vaep.to_parquet(OUT / "joi90_vaep.parquet", index=False)
            log.info("Wrote %d VAEP pairs to outputs/joi90_vaep.parquet", len(joi90_vaep))

    log.info("Stitching squads + chemistry pairs into chemistry.json")
    path = export(joi90_vaep=joi90_vaep)
    log.info("Wrote %s", path)


def _collect_joi_and_lineups(
    scored_dir: Path,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Collect per-match JOI DataFrames and lineups from a scored-actions directory."""
    all_per_match: list[pd.DataFrame] = []
    all_lineups: list[pd.DataFrame] = []
    for comp_dir in sorted(scored_dir.iterdir()):
        if not comp_dir.is_dir():
            continue
        for path in sorted(comp_dir.glob("*.parquet")):
            match_id = int(path.stem)
            scored = pd.read_parquet(path)
            interactions = joi.enumerate_interactions(scored)
            if len(interactions) == 0:
                continue
            per_match = joi.joi_per_match(interactions)
            all_per_match.append(per_match)
            try:
                all_lineups.append(pipeline.load_lineups_for_match(match_id))
            except Exception as exc:
                log.warning("Lineups load failed for %s: %s", match_id, exc)
    return all_per_match, all_lineups


def _score_with_xt() -> None:
    """Fit xT on all SPADL and score every competition into data/vaep/."""
    from chemistry import xt_model as xt_mod

    log.info("Fitting xT on full SPADL set")
    all_spadl: list[pd.DataFrame] = []
    spadl_dir = DATA / "spadl"
    if spadl_dir.exists():
        for d in sorted(spadl_dir.iterdir()):
            if d.is_dir():
                for p in d.glob("*.parquet"):
                    try:
                        all_spadl.append(pd.read_parquet(p))
                    except Exception as exc:
                        log.warning("Could not read %s: %s", p, exc)

    if not all_spadl:
        raise RuntimeError("No SPADL data found to fit xT — run SPADL conversion first")

    combined = pd.concat(all_spadl, ignore_index=True)
    log.info("Fitting xT on %d total actions", len(combined))
    xt = xt_mod.fit_xt(combined)
    xt_mod.save(xt)
    log.info("Saved xT model to data/xt/xt.pkl")

    log.info("Scoring competitions with xT")
    for spadl_comp in sorted(spadl_dir.iterdir()):
        if spadl_comp.is_dir():
            try:
                pipeline.score_competition(spadl_comp, xt)
            except Exception as exc:
                log.warning("xT scoring failed for %s: %s", spadl_comp.name, exc)


def _score_with_vaep(use_heuristic: bool) -> None:
    """Score every SPADL competition with VAEP into data/vaep_scored/."""
    vaep_pkl = DATA / "vaep" / "vaep.pkl"
    if use_heuristic:
        kind = "heuristic"
    elif vaep_pkl.exists():
        kind = "trained"
    else:
        log.warning("No trained VAEP model at %s — run scripts/train_vaep.py first. "
                    "Skipping VAEP scoring.", vaep_pkl)
        return

    log.info("Scoring with %s VAEP", kind)
    model = load_vaep_model(kind=kind, path=vaep_pkl if kind == "trained" else None)
    vaep_out_dir = DATA / "vaep_scored"
    for spadl_comp in sorted((DATA / "spadl").iterdir()):
        if spadl_comp.is_dir():
            try:
                pipeline.score_competition(spadl_comp, model, out_dir=vaep_out_dir)
            except Exception as exc:
                log.warning("VAEP scoring failed for %s: %s", spadl_comp.name, exc)


def export(
    out_path: Path = OUT / "chemistry.json",
    metric: str = "xt",
    joi90_vaep: "pd.DataFrame | None" = None,
) -> Path:
    from chemistry import squads as squads_mod, export as export_mod

    joi90_path = OUT / "joi90_xt.parquet"
    if not joi90_path.exists():
        joi90_path = OUT / "joi90.parquet"
    joi90 = pd.read_parquet(joi90_path)
    lineups = pd.read_parquet(OUT / "lineups.parquet")
    squad_map = squads_mod.load_all_squads(Path("squads/wc2026"))

    # Try to load saved VAEP pairs if not passed in
    if joi90_vaep is None:
        vaep_path = OUT / "joi90_vaep.parquet"
        if vaep_path.exists():
            try:
                joi90_vaep = pd.read_parquet(vaep_path)
            except Exception as exc:
                log.warning("Could not load VAEP pairs: %s", exc)

    return export_mod.build_chemistry_json(
        joi90, lineups, squad_map, out_path,
        metric=metric, joi90_vaep=joi90_vaep,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--heuristic-vaep", action="store_true",
                   help="Use heuristic VAEP instead of trained model.")
    p.add_argument("--no-fetch", action="store_true",
                   help="Skip the StatsBomb fetch step (use already-cached data).")
    p.add_argument("--metric", choices=["xt", "vaep", "both"], default="both",
                   help="Which metric(s) to compute (default: both).")
    args = p.parse_args()
    run(
        use_heuristic_vaep=args.heuristic_vaep,
        fetch=not args.no_fetch,
        metric=args.metric,
    )
