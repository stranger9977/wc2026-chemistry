"""Per-player goals and assists from SPADL/VAEP-scored actions."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SHOT_TYPE_IDS = {11, 12, 13}   # shot, shot_penalty, shot_freekick
RESULT_SUCCESS = 1


def goals_and_assists(actions: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with columns: player_id, goals, assists.

    A goal: shot action with result_id == RESULT_SUCCESS.
    An assist: action immediately preceding a goal, by a different player on the same team.
        Restricted to pass-like actions (pass, cross, free-kick, corner, throw-in, dribble).
    """
    df = actions.sort_values(["game_id", "period_id", "time_seconds"]).reset_index(drop=True)

    # Goals
    is_shot = df["type_id"].isin(SHOT_TYPE_IDS)
    is_goal = is_shot & (df["result_id"] == RESULT_SUCCESS)
    goals = (
        df.loc[is_goal]
          .groupby("player_id")
          .size()
          .reset_index(name="goals")
          .astype({"player_id": "int64"})
    )

    # Assists: look at the action immediately before each goal, on the same team, by a different player.
    PASS_TYPES = {0, 1, 3, 4, 5, 6, 7, 21}   # pass, cross, freekick_crossed/short, corner_crossed/short, take_on, dribble
    goal_idx = df.index[is_goal].to_list()
    assist_rows = []
    for gi in goal_idx:
        if gi == 0:
            continue
        prev = df.iloc[gi - 1]
        goal_row = df.iloc[gi]
        if (
            prev["game_id"] == goal_row["game_id"]
            and prev["team_id"] == goal_row["team_id"]
            and prev["player_id"] != goal_row["player_id"]
            and prev["type_id"] in PASS_TYPES
            and prev["result_id"] == RESULT_SUCCESS
        ):
            assist_rows.append(int(prev["player_id"]))

    assists = (
        pd.DataFrame({"player_id": assist_rows})
          .groupby("player_id")
          .size()
          .reset_index(name="assists")
          .astype({"player_id": "int64"})
    )

    return goals.merge(assists, on="player_id", how="outer").fillna(0).astype({"goals": int, "assists": int})


def goals_assists_for_all_matches(spadl_dir: Path) -> pd.DataFrame:
    """Aggregate goals + assists across all cached VAEP-scored matches."""
    parts = []
    for d in sorted(spadl_dir.iterdir()):
        if not d.is_dir():
            continue
        for p in d.glob("*.parquet"):
            parts.append(pd.read_parquet(p))
    if not parts:
        return pd.DataFrame(columns=["player_id", "goals", "assists"])
    combined = pd.concat(parts, ignore_index=True)
    return goals_and_assists(combined)
