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
    """Loads a socceraction-trained VAEP model bundle from disk."""

    def __init__(self, path: Path):
        with open(path, "rb") as f:
            self._bundle = pickle.load(f)

    def _build_features(self, spadl: pd.DataFrame) -> pd.DataFrame:
        from socceraction.vaep import features as fs
        gs = fs.gamestates(spadl)
        X = pd.concat(
            [fs.actiontype_onehot(gs),
             fs.result_onehot(gs),
             fs.startlocation(gs),
             fs.endlocation(gs)],
            axis=1,
        )
        # Align to training columns if recorded
        feature_cols = self._bundle.get("feature_cols")
        if feature_cols:
            # Add any missing cols as 0, drop extra cols
            for c in feature_cols:
                if c not in X.columns:
                    X[c] = 0
            X = X[feature_cols]
        return X

    @staticmethod
    def _enrich_spadl(spadl: pd.DataFrame) -> pd.DataFrame:
        """Add type_name and result_name columns required by socceraction formula."""
        import socceraction.spadl as spadl_mod
        at = spadl_mod.actiontypes_df().set_index("type_id")["type_name"]
        rt = spadl_mod.results_df().set_index("result_id")["result_name"]
        enriched = spadl.copy()
        if "type_name" not in enriched.columns:
            enriched["type_name"] = enriched["type_id"].map(at).fillna("unknown")
        if "result_name" not in enriched.columns:
            enriched["result_name"] = enriched["result_id"].map(rt).fillna("unknown")
        return enriched

    def score(self, spadl: pd.DataFrame) -> pd.Series:
        from socceraction.vaep import formula

        enriched = self._enrich_spadl(spadl)
        X = self._build_features(enriched)
        clf_scores = self._bundle["scores"]
        clf_concedes = self._bundle["concedes"]
        p_scores = pd.Series(clf_scores.predict_proba(X)[:, 1], index=enriched.index)
        p_concedes = pd.Series(clf_concedes.predict_proba(X)[:, 1], index=enriched.index)
        values_df = formula.value(enriched, p_scores, p_concedes)
        return values_df["vaep_value"]


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
