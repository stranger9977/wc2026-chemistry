"""WC 2026 squad definitions loaded from YAML."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class Player(BaseModel):
    name: str
    club: str
    position: str
    headshot: Optional[str] = None


class Squad(BaseModel):
    nation: str
    nation_code: str = Field(min_length=2, max_length=3)
    flag_iso: str
    manager: str
    formation: str
    team_color: str
    players: list[Player]


def load_squad(path: Path) -> Squad:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Squad(**data)


def load_all_squads(dir: Path) -> dict[str, Squad]:
    out: dict[str, Squad] = {}
    for p in sorted(Path(dir).glob("*.yaml")):
        squad = load_squad(p)
        out[squad.nation_code] = squad
    return out


import pandas as pd  # noqa: E402


def filter_pairs_by_squad(pairs: pd.DataFrame, player_ids: set[int]) -> pd.DataFrame:
    in_squad = pairs["player_a"].isin(player_ids) & pairs["player_b"].isin(player_ids)
    return pairs[in_squad].reset_index(drop=True)
