"""Joint Offensive Impact (JOI) — pair-wise chemistry from VAEP-scored actions."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Protocol

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


class MinutesProvider(Protocol):
    def minutes(self, game_id: int, player_p: int, player_q: int) -> float: ...


def _canonical_pair(p: pd.Series, q: pd.Series) -> tuple[pd.Series, pd.Series]:
    a = pd.concat([p, q], axis=1).min(axis=1)
    b = pd.concat([p, q], axis=1).max(axis=1)
    return a, b


def joi_per_match(interactions: pd.DataFrame) -> pd.DataFrame:
    """Sum vaep_pair per (game, unordered pair)."""
    a, b = _canonical_pair(interactions["player_p"], interactions["player_q"])
    df = interactions.assign(player_a=a.astype("int64"), player_b=b.astype("int64"))
    grouped = (
        df.groupby(["game_id", "team_id", "player_a", "player_b"], as_index=False)
          ["vaep_pair"].sum()
          .rename(columns={"vaep_pair": "joi"})
    )
    return grouped


def joi90_window(per_match: pd.DataFrame, minutes_provider: MinutesProvider) -> pd.DataFrame:
    """Aggregate per-match JOI to per-pair, per 90 minutes of shared play."""
    per_match = per_match.copy()
    per_match["minutes"] = per_match.apply(
        lambda r: minutes_provider.minutes(int(r["game_id"]), int(r["player_a"]), int(r["player_b"])),
        axis=1,
    )
    agg = (
        per_match.groupby(["team_id", "player_a", "player_b"], as_index=False)
                 .agg(joi=("joi", "sum"), minutes=("minutes", "sum"), matches=("game_id", "nunique"))
    )
    agg["joi90"] = (agg["joi"] * 90.0 / agg["minutes"]).where(agg["minutes"] > 0, 0.0)
    return agg
