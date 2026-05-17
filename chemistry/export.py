"""Stitch squads + JOI90 pair table into outputs/chemistry.json."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from chemistry import rank, squads, players
from chemistry.names import shorten_name
from chemistry.formation import position_xy, disambiguate_overlaps
from chemistry.pipeline import all_player_nicknames


VAEP_DIR = Path("data/vaep")


def _pair_record(
    row: pd.Series,
    id_to_name: dict[int, str],
    id_to_display: dict[int, str] | None = None,
) -> dict:
    a = int(row["player_a"])
    b = int(row["player_b"])
    disp = id_to_display or {}
    return {
        "player_a_id": a,
        "player_b_id": b,
        "player_a_name": id_to_name.get(a, str(a)),
        "player_b_name": id_to_name.get(b, str(b)),
        "player_a_display": disp.get(a, id_to_name.get(a, str(a))),
        "player_b_display": disp.get(b, id_to_name.get(b, str(b))),
        "joi90": float(row["joi90"]),
        "minutes": float(row["minutes"]),
        "matches": int(row["matches"]),
    }


def build_chemistry_json(
    joi90: pd.DataFrame,
    lineups: pd.DataFrame,
    squads_map: dict[str, squads.Squad],
    out_path: Path,
    min_minutes_global: float = 180.0,
    min_minutes_team: float = 180.0,
    metric: str = "xt",
) -> Path:
    id_to_name = (
        lineups.dropna(subset=["player_id"])
               .drop_duplicates("player_id")
               .set_index("player_id")["player_name"]
               .to_dict()
    )
    id_to_name = {int(k): v for k, v in id_to_name.items()}

    # Resolve nicknames from raw StatsBomb open data for all match_ids in scope
    all_match_ids = sorted(int(x) for x in lineups["game_id"].dropna().unique())
    nicknames_map = all_player_nicknames(all_match_ids)

    per_nation: dict[str, dict] = {}
    # Global display-name map: pid -> display_name (built up as we process nations)
    id_to_display: dict[int, str] = {}

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

        # Compute per-player minutes + dominant position
        team_lineups = team_lineups.copy()
        team_lineups["mins"] = team_lineups["to_minute"] - team_lineups["from_minute"]

        # Most-played StatsBomb position per player (mode)
        pos_mode = (
            team_lineups.dropna(subset=["position"])
                        .groupby("player_id")["position"]
                        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None)
        )

        minutes_by_player = (
            team_lineups.groupby("player_id")
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

        # Pitch shows just the top 11 by minutes
        pitch_ids = [int(pid) for pid in
                     minutes_by_player.head(11).index
                     if int(pid) in effective_ids]

        # Compute formation-based positions
        raw_positions: dict[int, tuple[float, float]] = {}
        for pid in effective_ids:
            if pid not in minutes_by_player.index:
                continue
            sb_pos = pos_mode.get(pid)
            xy = position_xy(sb_pos)
            if xy is None:
                xy = (50.0, 34.0)
            raw_positions[int(pid)] = xy

        resolved_positions = disambiguate_overlaps(raw_positions)

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
            sb_position = pos_mode.get(pid)
            full_name = id_to_name.get(int(pid), str(int(pid)))
            nickname = nicknames_map.get(int(pid))

            # Prefer YAML name as the base for display (often cleaner), but let
            # player_nickname override if available
            yaml_name = yp.name if yp else None
            display_name = shorten_name(yaml_name or full_name, nickname)

            x, y = resolved_positions.get(int(pid), (50.0, 34.0))
            players_by_id[str(pid)] = {
                "name": full_name,
                "display_name": display_name,
                "nickname": nickname,
                "sb_position": sb_position,
                "position": yp.position if yp else None,
                "x": float(x),
                "y": float(y),
                "minutes": float(row["total_mins"]),
                "matches": int(row["matches"]),
            }
            # Register in global display map
            id_to_display[int(pid)] = display_name

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
            "pitch_player_ids": pitch_ids,
            "pairs": [_pair_record(r, id_to_name, id_to_display) for _, r in squad_pairs.iterrows()],
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

    def _enrich_leaderboard_row(
        row: pd.Series,
        id_to_name: dict[int, str],
        id_to_nation: dict[int, tuple[str, str]],
        id_to_display: dict[int, str],
    ) -> dict:
        rec = _pair_record(row, id_to_name, id_to_display)
        a, b = rec["player_a_id"], rec["player_b_id"]
        nation = id_to_nation.get(a) or id_to_nation.get(b)
        if nation:
            rec["nation_code"], rec["flag_iso"] = nation
        return rec

    doc = {
        "nations": per_nation,
        "leaderboard": [
            _enrich_leaderboard_row(r, id_to_name, id_to_nation, id_to_display)
            for _, r in leaderboard.iterrows()
        ],
        "meta": {
            "metric": metric,
            "min_minutes_global": min_minutes_global,
            "min_minutes_team": min_minutes_team,
            "total_pairs": int(len(joi90)),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    return out_path
