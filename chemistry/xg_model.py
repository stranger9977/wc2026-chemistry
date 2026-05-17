"""Lightweight xG model: logistic regression on shot location, type, body-part."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

MODELS_DIR = Path(__file__).parent.parent / "data" / "xg"

# SPADL type ids for shots
SHOT_TYPE_IDS = {11, 12, 13}   # shot, shot_penalty, shot_freekick


def _shot_features(actions: pd.DataFrame) -> pd.DataFrame:
    """For each shot, build features: distance to goal, angle, type, bodypart, result."""
    a = actions.copy()
    # Goal is at (105, 34) in SPADL
    dx = 105.0 - a["start_x"]
    dy = 34.0 - a["start_y"]
    distance = np.sqrt(dx ** 2 + dy ** 2)
    # Angle (rough): 0 when straight on, larger when wide. Use arctan ratio.
    angle = np.abs(np.arctan2(np.abs(dy), np.maximum(dx, 0.1)))
    feats = pd.DataFrame({
        "distance": distance,
        "angle": angle,
        "is_penalty": (a["type_id"] == 12).astype(int),
        "is_freekick": (a["type_id"] == 13).astype(int),
        "is_header": (a["bodypart_id"] == 3).astype(int) if "bodypart_id" in a.columns else 0,
    })
    return feats


@dataclass
class XgModel:
    clf: LogisticRegression
    scaler: StandardScaler

    def predict_xg(self, shots: pd.DataFrame) -> pd.Series:
        feats = _shot_features(shots)
        X = self.scaler.transform(feats.values)
        proba = self.clf.predict_proba(X)[:, 1]
        return pd.Series(proba, index=shots.index)


def fit_xg(actions: pd.DataFrame) -> XgModel:
    shots = actions[actions["type_id"].isin(SHOT_TYPE_IDS)].copy()
    # Result id 1 means a successful shot (which in SPADL corresponds to a goal for shots).
    y = (shots["result_id"] == 1).astype(int).values
    X_df = _shot_features(shots)
    scaler = StandardScaler().fit(X_df.values)
    X = scaler.transform(X_df.values)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
    return XgModel(clf=clf, scaler=scaler)


def save(model: XgModel, path: Path = MODELS_DIR / "xg.pkl") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"clf": model.clf, "scaler": model.scaler}, f)
    return path


def load(path: Path = MODELS_DIR / "xg.pkl") -> XgModel:
    with open(path, "rb") as f:
        d = pickle.load(f)
    return XgModel(clf=d["clf"], scaler=d["scaler"])
