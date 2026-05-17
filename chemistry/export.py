"""Stitch squads + JOI90 pair table into outputs/chemistry.json."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from chemistry import rank, squads, players


VAEP_DIR = Path("data/vaep")


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


def _compute_avg_positions(team_lineups: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """Average start_x/start_y from VAEP-scored actions per player.

    Returns {player_id: (x, y)} in 105x68 SPADL coordinates.
    Players with no action data are omitted (caller falls back to default).
    """
    if not VAEP_DIR.exists():
        return {}

    game_ids = set(team_lineups["game_id"].unique())
    team_ids = set(team_lineups["team_id"].unique())

    actions_list = []
    for comp_dir in VAEP_DIR.iterdir():
        if not comp_dir.is_dir():
            continue
        for p in comp_dir.glob("*.parquet"):
            try:
                gid = int(p.stem)
            except ValueError:
                continue
            if gid in game_ids:
                actions_list.append(pd.read_parquet(p))

    if not actions_list:
        return {}

    actions = pd.concat(actions_list, ignore_index=True)
    actions = actions[actions["team_id"].isin(team_ids)]
    if actions.empty:
        return {}

    # Normalise player_id to int (may be float due to NaN rows elsewhere)
    actions = actions.dropna(subset=["player_id"])
    actions["player_id"] = actions["player_id"].astype(int)

    avg = actions.groupby("player_id")[["start_x", "start_y"]].mean()
    return {int(pid): (float(row.start_x), float(row.start_y)) for pid, row in avg.iterrows()}


def build_chemistry_json(
    joi90: pd.DataFrame,
    lineups: pd.DataFrame,
    squads_map: dict[str, squads.Squad],
    out_path: Path,
    min_minutes_global: float = 180.0,
    min_minutes_team: float = 180.0,
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
                "players_by_id": {},
                "coverage": {"matches": 0, "warning": "no matches in open data"},
            }
            continue

        # Build yaml roster IDs if squad has hand-curated players
        yaml_roster_ids: set[int] = set()
        matched: dict[str, int] = {}
        if squad.players:
            matched, unmatched = players.resolve_squad_ids(
                squad.nation, [p.name for p in squad.players], team_lineups
            )
            yaml_roster_ids = set(matched.values())

        # Auto-derived: top 26 by minutes played for this nation
        minutes_by_player = (
            team_lineups.assign(mins=team_lineups["to_minute"] - team_lineups["from_minute"])
                        .groupby("player_id")
                        .agg(
                            total_mins=("mins", "sum"),
                            player_name=("player_name", "first"),
                            matches=("game_id", "nunique"),
                        )
                        .sort_values("total_mins", ascending=False)
        )
        # Normalise index to int
        minutes_by_player.index = minutes_by_player.index.astype(int)
        auto_roster_ids = set(int(pid) for pid in minutes_by_player.head(26).index)

        # Effective roster: union of YAML and auto-derived
        effective_ids = yaml_roster_ids | auto_roster_ids

        # Compute empirical average positions from VAEP actions
        avg_pos = _compute_avg_positions(team_lineups)

        # Build reverse map: player_id -> yaml player entry
        yaml_by_id: dict[int, squads.Player] = {}
        for yp in squad.players:
            pid = matched.get(yp.name)
            if pid is not None:
                yaml_by_id[int(pid)] = yp

        # Build players_by_id dict
        players_by_id: dict[str, dict] = {}
        for pid in effective_ids:
            if pid not in minutes_by_player.index:
                continue
            row = minutes_by_player.loc[pid]
            yp = yaml_by_id.get(pid)
            default_xy = (50.0, 34.0)
            x, y = avg_pos.get(pid, default_xy)
            players_by_id[str(pid)] = {
                "name": yp.name if yp else row["player_name"],
                "position": yp.position if yp else None,
                "x": x,
                "y": y,
                "minutes": float(row["total_mins"]),
                "matches": int(row["matches"]),
            }

        # Filter chemistry pairs to effective roster
        effective_ids_int = {int(i) for i in effective_ids}
        squad_pairs = joi90[
            joi90["player_a"].isin(effective_ids_int)
            & joi90["player_b"].isin(effective_ids_int)
            & (joi90["minutes"] >= min_minutes_team)
        ].copy()
        squad_pairs = squad_pairs.sort_values("joi90", ascending=False)

        coverage: dict = {
            "matches": int(team_lineups["game_id"].nunique()),
            "auto_derived_roster": len(yaml_roster_ids) == 0,
            "roster_size": len(players_by_id),
        }
        if squad.players and unmatched:  # type: ignore[possibly-undefined]
            coverage["unmatched_roster_names"] = unmatched  # type: ignore[possibly-undefined]

        per_nation[code] = {
            "squad": squad.model_dump(),
            "players_by_id": players_by_id,
            "pairs": [_pair_record(r, id_to_name) for _, r in squad_pairs.iterrows()],
            "coverage": coverage,
        }

    in_any_squad_ids: set[int] = set()
    for entry in per_nation.values():
        for p in entry["pairs"]:
            in_any_squad_ids.add(p["player_a_id"])
            in_any_squad_ids.add(p["player_b_id"])

    # Build reverse index: player_id -> (nation_code, flag_iso)
    id_to_nation: dict[int, tuple[str, str]] = {}
    for code, entry in per_nation.items():
        for pid_str in entry.get("players_by_id", {}).keys():
            id_to_nation[int(pid_str)] = (code, entry["squad"]["flag_iso"])

    leaderboard_src = joi90[
        joi90["player_a"].isin(in_any_squad_ids)
        & joi90["player_b"].isin(in_any_squad_ids)
        & (joi90["minutes"] >= min_minutes_global)
    ].copy()
    leaderboard = rank.global_top_n(
        leaderboard_src.assign(nation_code="GLOBAL"), n=50,
    )

    def _enrich_leaderboard_row(row: pd.Series, id_to_name: dict[int, str], id_to_nation: dict[int, tuple[str, str]]) -> dict:
        rec = _pair_record(row, id_to_name)
        a, b = rec["player_a_id"], rec["player_b_id"]
        nation = id_to_nation.get(a) or id_to_nation.get(b)
        if nation:
            rec["nation_code"], rec["flag_iso"] = nation
        return rec

    doc = {
        "nations": per_nation,
        "leaderboard": [_enrich_leaderboard_row(r, id_to_name, id_to_nation) for _, r in leaderboard.iterrows()],
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
