"""Joint Offensive Impact (JOI) — pair-wise chemistry from VAEP-scored actions."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

# SPADL type IDs that count as on-the-ball for JOI.
# Paper uses: pass, cross, dribble, take-on, shot.
# socceraction spadl type ids (verify against your installed version):
# 0=pass, 1=cross, 2=throw-in, 3=freekick_crossed, 4=freekick_short,
# 5=corner_crossed, 6=corner_short, 7=take_on, 8=foul, 9=tackle,
# 10=interception, 11=shot, 12=shot_penalty, 13=shot_freekick,
# 14=keeper_save, 15=keeper_claim, 16=keeper_punch, 17=keeper_pick_up,
# 18=clearance, 19=bad_touch, 20=non_action, 21=dribble, 22=goalkick
ELIGIBLE_TYPES: frozenset[int] = frozenset({0, 1, 7, 11, 21, 12, 13})


def enumerate_interactions(spadl: pd.DataFrame) -> pd.DataFrame:
    """Return one row per consecutive same-team action pair with different players.

    Columns:
        game_id, team_id, player_p, player_q,
        vaep_p, vaep_q, vaep_pair, time_p, time_q
    """
    df = spadl.copy()
    df = df[df["type_id"].isin(ELIGIBLE_TYPES)]
    df = df.sort_values(["game_id", "period_id", "time_seconds"]).reset_index(drop=True)

    next_df = df.shift(-1)

    consecutive = (
        (df["game_id"] == next_df["game_id"])
        & (df["team_id"] == next_df["team_id"])
        & (df["player_id"] != next_df["player_id"])
    )
    pair = pd.DataFrame({
        "game_id": df["game_id"],
        "team_id": df["team_id"],
        "player_p": df["player_id"],
        "player_q": next_df["player_id"],
        "vaep_p":   df["vaep_value"],
        "vaep_q":   next_df["vaep_value"],
        "time_p":   df["time_seconds"],
        "time_q":   next_df["time_seconds"],
    })[consecutive].reset_index(drop=True)
    pair["vaep_pair"] = pair["vaep_p"].fillna(0) + pair["vaep_q"].fillna(0)
    pair["player_q"] = pair["player_q"].astype("int64")
    return pair
