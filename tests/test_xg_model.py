import pandas as pd
import numpy as np
from chemistry.xg_model import fit_xg, _shot_features, SHOT_TYPE_IDS


def test_shot_features_distance_and_angle():
    # Penalty spot at (94, 34) — distance to (105, 34) is 11, angle ~0
    shots = pd.DataFrame([{
        "start_x": 94.0, "start_y": 34.0, "end_x": 105, "end_y": 34,
        "type_id": 11, "result_id": 0, "bodypart_id": 1,
    }])
    feats = _shot_features(shots)
    assert abs(feats.iloc[0]["distance"] - 11.0) < 0.01
    assert feats.iloc[0]["angle"] < 0.01


def test_fit_xg_smoke():
    rng = np.random.default_rng(0)
    n = 200
    spadl = pd.DataFrame({
        "start_x": rng.uniform(60, 105, n),
        "start_y": rng.uniform(0, 68, n),
        "end_x": 105.0,
        "end_y": 34.0,
        "type_id": 11,
        "result_id": rng.integers(0, 2, n),
        "bodypart_id": 1,
    })
    model = fit_xg(spadl)
    proba = model.predict_xg(spadl.head(10))
    assert (proba >= 0).all() and (proba <= 1).all()
