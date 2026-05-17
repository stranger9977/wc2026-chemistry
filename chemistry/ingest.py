"""Pull and cache StatsBomb open data for international competitions."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from statsbombpy import sb

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "data" / "raw"

# Curated list — only international tournaments + qualifiers we care about
# for chemistry leading up to WC 2026. Identified by StatsBomb competition_id
# and season_id from sb.competitions().
# IDs verified against sb.competitions() on 2026-05-17.
INTERNATIONAL_COMPETITIONS: list[dict] = [
    {"competition_id": 43,   "season_id": 106, "label": "FIFA World Cup 2022"},
    {"competition_id": 55,   "season_id": 282, "label": "UEFA Euro 2024"},
    {"competition_id": 55,   "season_id": 43,  "label": "UEFA Euro 2020"},
    {"competition_id": 223,  "season_id": 282, "label": "Copa America 2024"},
]


def target_competitions() -> list[dict]:
    """Return the curated list of competitions to ingest."""
    return list(INTERNATIONAL_COMPETITIONS)


def list_available_competitions() -> list[dict]:
    """Return everything StatsBomb has in open data, as plain dicts."""
    df = sb.competitions()
    return df.to_dict(orient="records")


def cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def fetch_competition(competition_id: int, season_id: int) -> Path:
    """Download all matches + events for one competition-season into the cache.

    Returns the directory it wrote to.
    """
    base = cache_dir() / f"{competition_id}_{season_id}"
    base.mkdir(parents=True, exist_ok=True)

    matches_path = base / "matches.parquet"
    if matches_path.exists():
        matches = pd.read_parquet(matches_path)
    else:
        matches = sb.matches(competition_id=competition_id, season_id=season_id)
        matches.to_parquet(matches_path, index=False)
        logger.info("Wrote %s matches to %s", len(matches), matches_path)

    if matches.empty:
        logger.warning(
            "No matches returned for competition %s season %s — skipping",
            competition_id,
            season_id,
        )
        return base

    events_dir = base / "events"
    events_dir.mkdir(exist_ok=True)
    for match_id in matches["match_id"]:
        target = events_dir / f"{match_id}.parquet"
        if target.exists():
            continue
        events = sb.events(match_id=match_id)
        events.to_parquet(target, index=False)
    logger.info("Cached events for competition %s season %s", competition_id, season_id)
    return base


def fetch_all_targets() -> list[Path]:
    """Fetch every competition in INTERNATIONAL_COMPETITIONS."""
    paths = []
    for comp in INTERNATIONAL_COMPETITIONS:
        path = fetch_competition(comp["competition_id"], comp["season_id"])
        paths.append(path)
    return paths
