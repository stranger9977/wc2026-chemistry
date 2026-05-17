"""xT (Expected Threat) scoring wrapper around socceraction.xthreat."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from socceraction.xthreat import ExpectedThreat

MODELS_DIR = Path(__file__).parent.parent / "data" / "xt"


@dataclass
class XtModel:
    """Wraps a fitted ExpectedThreat grid."""
    xt: ExpectedThreat

    def score(self, spadl: pd.DataFrame) -> pd.Series:
        """Return per-action xT delta. Returns 0 for action types xT doesn't value (NaN)."""
        try:
            raw = self.xt.rate(spadl)
            result = pd.Series(raw, index=spadl.index)
            # xT returns NaN for non-move actions (shots, fouls, etc.) — treat as 0
            return result.fillna(0.0)
        except Exception:
            # rate() can throw on edge-case actions; return zeros
            return pd.Series(0.0, index=spadl.index)


def fit_xt(all_actions: pd.DataFrame, l: int = 16, w: int = 12) -> XtModel:
    """Fit an xT grid on the combined SPADL action set."""
    xt = ExpectedThreat(l=l, w=w)
    xt.fit(all_actions)
    return XtModel(xt=xt)


def save(model: XtModel, path: Path = MODELS_DIR / "xt.pkl") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model.xt, f)
    return path


def load(path: Path = MODELS_DIR / "xt.pkl") -> XtModel:
    with open(path, "rb") as f:
        xt = pickle.load(f)
    return XtModel(xt=xt)
