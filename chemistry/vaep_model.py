"""VAEP scoring — wraps socceraction's VAEP and offers a heuristic fallback."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).parent.parent / "data" / "vaep"


class VaepModel(Protocol):
    def score(self, spadl: pd.DataFrame) -> pd.Series: ...


@dataclass
class HeuristicVaep:
    """Quick & dirty heuristic for development and unit tests."""

    def score(self, spadl: pd.DataFrame) -> pd.Series:
        # SPADL pitch is 105x68; reward end_x closer to 105 (opponent goal)
        progress = spadl["end_x"] - spadl["start_x"]
        attacking_third = (spadl["end_x"] > 70).astype(float)
        shot_bonus = (spadl["type_id"] == 11).astype(float) * 0.4  # shot type
        successful = (spadl["result_id"] == 1).astype(float)
        val = (
            0.01 * progress
            + 0.05 * attacking_third
            + shot_bonus
        ) * successful
        return val.astype(float)


class TrainedVaep:
    """Loads a socceraction-trained VAEP model from disk."""

    def __init__(self, path: Path):
        with open(path, "rb") as f:
            self._model = pickle.load(f)

    def score(self, spadl: pd.DataFrame) -> pd.Series:
        from socceraction.vaep import features as fs
        from socceraction.vaep import formula

        gamestates = fs.gamestates(spadl)
        X = pd.concat(
            [fs.actiontype_onehot(gamestates),
             fs.result_onehot(gamestates),
             fs.startlocation(gamestates),
             fs.endlocation(gamestates)],
            axis=1,
        )
        p_scores = self._model.predict_proba(X)[:, 1]
        p_concedes = np.zeros(len(spadl))
        return formula.offensive_value(spadl, pd.Series(p_scores), pd.Series(p_concedes))


def load_vaep_model(kind: Literal["heuristic", "trained"] = "trained",
                    path: Path | None = None) -> VaepModel:
    if kind == "heuristic":
        return HeuristicVaep()
    target = path or (MODELS_DIR / "vaep.pkl")
    if not target.exists():
        raise FileNotFoundError(
            f"No trained VAEP model at {target}. Run scripts/train_vaep.py first, "
            f"or use kind='heuristic' for testing."
        )
    return TrainedVaep(target)
