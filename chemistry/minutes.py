"""Compute shared on-pitch minutes between players for a match."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def shared_minutes(lineups: pd.DataFrame, game_id: int, player_p: int, player_q: int) -> float:
    rows = lineups[(lineups["game_id"] == game_id)
                   & (lineups["player_id"].isin([player_p, player_q]))]
    if len(rows) < 2:
        return 0.0
    a = rows[rows["player_id"] == player_p].iloc[0]
    b = rows[rows["player_id"] == player_q].iloc[0]
    if a["team_id"] != b["team_id"]:
        return 0.0
    start = max(a["from_minute"], b["from_minute"])
    end = min(a["to_minute"], b["to_minute"])
    return float(max(0.0, end - start))


@dataclass
class LineupsMinutes:
    """MinutesProvider implementation backed by a lineups DataFrame."""
    lineups: pd.DataFrame

    def minutes(self, game_id: int, player_p: int, player_q: int) -> float:
        return shared_minutes(self.lineups, game_id, player_p, player_q)
