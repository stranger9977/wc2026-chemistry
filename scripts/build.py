"""End-to-end build:
   ingest → SPADL → VAEP → interactions → JOI per match → JOI90 per pair.
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


def run(use_heuristic_vaep: bool, fetch: bool = True) -> None:
    if fetch:
        log.info("Fetching StatsBomb open data for target competitions")
        ingest.fetch_all_targets()
    else:
        log.info("Skipping fetch (--no-fetch)")

    log.info("Converting events to SPADL")
    for comp_dir in sorted((DATA / "raw").iterdir()):
        if comp_dir.is_dir():
            pipeline.convert_competition(comp_dir)

    log.info("Scoring VAEP")
    kind = "heuristic" if use_heuristic_vaep else "trained"
    model = load_vaep_model(kind=kind)
    for spadl_comp in sorted((DATA / "spadl").iterdir()):
        if spadl_comp.is_dir():
            pipeline.score_competition(spadl_comp, model)

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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--heuristic-vaep", action="store_true")
    p.add_argument("--no-fetch", action="store_true",
                   help="Skip the StatsBomb fetch step (use already-cached data).")
    args = p.parse_args()
    run(use_heuristic_vaep=args.heuristic_vaep, fetch=not args.no_fetch)
