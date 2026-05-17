"""Stitch squads + JOI90 pair table into outputs/chemistry.json."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from chemistry import rank, squads, players


def _pair_record(row: pd.Series, id_to_name: dict[int, str]) -> dict:
    return {
        "player_a_id": int(row["player_a"]),
        "player_b_id": int(row["player_b"]),
        "player_a_name": id_to_name.get(int(row["player_a"]), str(int(row["player_a"]))),
        "player_b_name": id_to_name.get(int(row["player_b"]), str(int(row["player_b"]))),
        "joi90": float(row["joi90"]),
        "minutes": float(row["minutes"]),
        "matches": int(row["matches"]),
    }


def build_chemistry_json(
    joi90: pd.DataFrame,
    lineups: pd.DataFrame,
    squads_map: dict[str, squads.Squad],
    out_path: Path,
    min_minutes_global: float = 90.0,
    min_minutes_team: float = 90.0,
) -> Path:
    id_to_name = (
        lineups.dropna(subset=["player_id"])
               .drop_duplicates("player_id")
               .set_index("player_id")["player_name"]
               .to_dict()
    )
    id_to_name = {int(k): v for k, v in id_to_name.items()}

    per_nation: dict[str, dict] = {}
    for code, squad in squads_map.items():
        team_lineups = lineups[lineups["team_name"] == squad.nation]
        if team_lineups.empty:
            per_nation[code] = {
                "squad": squad.model_dump(),
                "pairs": [],
                "coverage": {"matches": 0, "warning": "no matches in open data"},
            }
            continue

        roster_names = [p.name for p in squad.players]
        matched, unmatched = players.resolve_squad_ids(squad.nation, roster_names, team_lineups)

        roster_ids = set(matched.values())
        squad_pairs = joi90[
            joi90["player_a"].isin(roster_ids) & joi90["player_b"].isin(roster_ids)
            & (joi90["minutes"] >= min_minutes_team)
        ].copy()
        squad_pairs = squad_pairs.sort_values("joi90", ascending=False)

        per_nation[code] = {
            "squad": squad.model_dump(),
            "pairs": [_pair_record(r, id_to_name) for _, r in squad_pairs.iterrows()],
            "coverage": {
                "matches": int(team_lineups["game_id"].nunique()),
                "unmatched_roster_names": unmatched,
            },
        }

    in_any_squad_ids: set[int] = set()
    for entry in per_nation.values():
        for p in entry["pairs"]:
            in_any_squad_ids.add(p["player_a_id"])
            in_any_squad_ids.add(p["player_b_id"])

    leaderboard_src = joi90[
        joi90["player_a"].isin(in_any_squad_ids)
        & joi90["player_b"].isin(in_any_squad_ids)
        & (joi90["minutes"] >= min_minutes_global)
    ].copy()
    leaderboard = rank.global_top_n(
        leaderboard_src.assign(nation_code="GLOBAL"), n=50,
    )

    doc = {
        "nations": per_nation,
        "leaderboard": [_pair_record(r, id_to_name) for _, r in leaderboard.iterrows()],
        "meta": {
            "min_minutes_global": min_minutes_global,
            "min_minutes_team": min_minutes_team,
            "total_pairs": int(len(joi90)),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    return out_path
