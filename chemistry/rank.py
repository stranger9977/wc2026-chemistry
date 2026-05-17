"""Ranking outputs — per-team chemistry tables and a global leaderboard."""
from __future__ import annotations

import pandas as pd


def per_team(pairs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """{nation_code: pairs sorted desc by joi90}."""
    out: dict[str, pd.DataFrame] = {}
    for code, grp in pairs.groupby("nation_code"):
        out[code] = grp.sort_values("joi90", ascending=False).reset_index(drop=True)
    return out


def global_top_n(pairs: pd.DataFrame, n: int = 50, min_minutes: float = 0.0) -> pd.DataFrame:
    filtered = pairs[pairs["minutes"] >= min_minutes].copy()
    return filtered.sort_values("joi90", ascending=False).head(n).reset_index(drop=True)
