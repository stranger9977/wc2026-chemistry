"""Map roster names to StatsBomb player_id values."""
from __future__ import annotations

import unicodedata
from typing import Iterable

import pandas as pd


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return s.lower().strip()


def build_player_index(lineups: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Return {team_name: {normalized_name: player_id}}."""
    out: dict[str, dict[str, int]] = {}
    for (team, name), grp in lineups.groupby(["team_name", "player_name"]):
        pid = int(grp["player_id"].mode().iloc[0])
        out.setdefault(team, {})[_norm(name)] = pid
    return out


def resolve_squad_ids(
    team_name: str,
    roster_names: Iterable[str],
    lineups: pd.DataFrame,
) -> tuple[dict[str, int], list[str]]:
    """Match roster names to StatsBomb player_ids; return (matched, unmatched)."""
    index = build_player_index(lineups).get(team_name, {})
    matched: dict[str, int] = {}
    unmatched: list[str] = []
    for name in roster_names:
        pid = index.get(_norm(name))
        if pid is None:
            unmatched.append(name)
        else:
            matched[name] = pid
    return matched, unmatched
