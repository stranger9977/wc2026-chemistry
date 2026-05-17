"""End-to-end build:
   ingest → SPADL → VAEP/xT → interactions → JOI per match → JOI90 per pair.
Writes intermediate parquets and the final pair table.
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


def run(use_heuristic_vaep: bool, fetch: bool = True, metric: str = "xt") -> None:
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

    if metric == "xt":
        _score_with_xt()
    else:
        log.info("Scoring with %s VAEP", "heuristic" if use_heuristic_vaep else "trained")
        kind = "heuristic" if use_heuristic_vaep else "trained"
        model = load_vaep_model(kind=kind)
        for spadl_comp in sorted((DATA / "spadl").iterdir()):
            if spadl_comp.is_dir():
                try:
                    pipeline.score_competition(spadl_comp, model)
                except Exception as exc:
                    log.warning("VAEP scoring failed for %s: %s", spadl_comp.name, exc)

    all_per_match: list[pd.DataFrame] = []
    all_lineups: list[pd.DataFrame] = []
    for vaep_comp in sorted((DATA / "vaep").iterdir()):
        if not vaep_comp.is_dir():
            continue
        for path in sorted(vaep_comp.glob("*.parquet")):
            match_id = int(path.stem)
            spadl = pd.read_parquet(path)
            interactions = joi.enumerate_interactions(spadl)
            if len(interactions) == 0:
                continue
            per_match = joi.joi_per_match(interactions)
            all_per_match.append(per_match)
            try:
                all_lineups.append(pipeline.load_lineups_for_match(match_id))
            except Exception as exc:
                log.warning("Lineups load failed for %s: %s", match_id, exc)

    per_match_df = pd.concat(all_per_match, ignore_index=True) if all_per_match else pd.DataFrame()
    lineups_df   = pd.concat(all_lineups,   ignore_index=True) if all_lineups   else pd.DataFrame()
    per_match_df.to_parquet(OUT / "joi_per_match.parquet", index=False)
    lineups_df.to_parquet(OUT / "lineups.parquet", index=False)

    mins_provider = minutes.LineupsMinutes(lineups_df)
    joi90 = joi.joi90_window(per_match_df, mins_provider)
    joi90.to_parquet(OUT / "joi90.parquet", index=False)
    log.info("Wrote %d pairs to outputs/joi90.parquet", len(joi90))

    log.info("Stitching squads + chemistry pairs")
    path = export(metric=metric)
    log.info("Wrote %s", path)


def _score_with_xt() -> None:
    """Fit xT on all SPADL and score every competition."""
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


def export(out_path: Path = OUT / "chemistry.json", metric: str = "xt") -> Path:
    from chemistry import squads as squads_mod, export as export_mod
    joi90 = pd.read_parquet(OUT / "joi90.parquet")
    lineups = pd.read_parquet(OUT / "lineups.parquet")
    squad_map = squads_mod.load_all_squads(Path("squads/wc2026"))
    return export_mod.build_chemistry_json(joi90, lineups, squad_map, out_path, metric=metric)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--heuristic-vaep", action="store_true",
                   help="Use heuristic VAEP (ignored when --metric=xt).")
    p.add_argument("--no-fetch", action="store_true",
                   help="Skip the StatsBomb fetch step (use already-cached data).")
    p.add_argument("--metric", choices=["xt", "vaep"], default="xt",
                   help="Action-value metric to use (default: xt).")
    args = p.parse_args()
    run(
        use_heuristic_vaep=args.heuristic_vaep,
        fetch=not args.no_fetch,
        metric=args.metric,
    )
