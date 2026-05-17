"""Train the VAEP scoring model on cached SPADL data.

Pickles to data/vaep/vaep.pkl. Run after scripts/build.py has produced SPADL.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

SPADL_DIR = Path("data/spadl")
OUT = Path("data/vaep/vaep.pkl")


def load_all_spadl() -> pd.DataFrame:
    parts = []
    for comp_dir in sorted(SPADL_DIR.iterdir()):
        if not comp_dir.is_dir():
            continue
        for p in sorted(comp_dir.glob("*.parquet")):
            parts.append(pd.read_parquet(p))
    if not parts:
        raise RuntimeError(f"No SPADL data under {SPADL_DIR}. Run scripts/build.py first.")
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    try:
        from socceraction.vaep import features as fs
        from socceraction.vaep import labels as lb
        from xgboost import XGBClassifier
    except ImportError as e:
        raise SystemExit(
            f"Missing dependency for VAEP training: {e}\n"
            "Install with: pip install socceraction xgboost"
        ) from e

    spadl = load_all_spadl()

    # Build feature matrix using gamestates + feature functions
    gamestates = fs.gamestates(spadl)
    X = pd.concat(
        [fs.actiontype_onehot(gamestates),
         fs.result_onehot(gamestates),
         fs.startlocation(gamestates),
         fs.endlocation(gamestates)],
        axis=1,
    )

    # scores label: 1 if action leads to a goal within nb_prev_actions
    y_scores = lb.scores(spadl)
    y_concedes = lb.concedes(spadl)

    model_scores = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        n_jobs=-1, eval_metric="logloss",
    )
    model_scores.fit(X, y_scores)

    model_concedes = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        n_jobs=-1, eval_metric="logloss",
    )
    model_concedes.fit(X, y_concedes)

    bundle = {"scores": model_scores, "concedes": model_concedes}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Wrote VAEP model to {args.out}")


if __name__ == "__main__":
    main()
