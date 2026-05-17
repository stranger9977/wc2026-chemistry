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

# VAEP feature functions used for both training and inference
FEATURE_FUNCTIONS = ["actiontype_onehot", "result_onehot", "startlocation", "endlocation"]


def _add_type_name(spadl: pd.DataFrame) -> pd.DataFrame:
    """Add type_name column from type_id lookup (required by socceraction labels)."""
    import socceraction.spadl as spadl_mod
    at = spadl_mod.actiontypes_df().set_index("type_id")["type_name"]
    spadl = spadl.copy()
    spadl["type_name"] = spadl["type_id"].map(at).fillna("unknown")
    return spadl


def build_features(spadl: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix from SPADL actions using gamestates API."""
    from socceraction.vaep import features as fs
    gs = fs.gamestates(spadl)
    return pd.concat(
        [fs.actiontype_onehot(gs),
         fs.result_onehot(gs),
         fs.startlocation(gs),
         fs.endlocation(gs)],
        axis=1,
    )


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
    parser.add_argument("--sample", type=float, default=1.0,
                        help="Fraction of actions to use for training (0-1). Use 0.3 if OOM.")
    args = parser.parse_args()

    try:
        from socceraction.vaep import labels as lb
        from xgboost import XGBClassifier
    except ImportError as e:
        raise SystemExit(
            f"Missing dependency for VAEP training: {e}\n"
            "Install with: pip install socceraction xgboost"
        ) from e

    print("Loading SPADL data...")
    spadl = load_all_spadl()
    print(f"Loaded {len(spadl):,} actions from {SPADL_DIR}")

    if args.sample < 1.0:
        spadl = spadl.sample(frac=args.sample, random_state=42).reset_index(drop=True)
        print(f"Downsampled to {len(spadl):,} actions ({args.sample*100:.0f}%)")

    # Add type_name so labels functions can identify shots
    spadl = _add_type_name(spadl)

    print("Building feature matrix...")
    X = build_features(spadl)
    print(f"Feature matrix shape: {X.shape}")

    print("Building labels...")
    y_scores = lb.scores(spadl)["scores"]
    y_concedes = lb.concedes(spadl)["concedes"]
    print(f"  scores positives: {y_scores.sum():,}/{len(y_scores):,}")
    print(f"  concedes positives: {y_concedes.sum():,}/{len(y_concedes):,}")

    print("Training scores model...")
    model_scores = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        n_jobs=-1, eval_metric="logloss",
    )
    model_scores.fit(X, y_scores)

    print("Training concedes model...")
    model_concedes = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        n_jobs=-1, eval_metric="logloss",
    )
    model_concedes.fit(X, y_concedes)

    # Store the feature column names so inference can align exactly
    feature_cols = list(X.columns)
    bundle = {
        "scores": model_scores,
        "concedes": model_concedes,
        "feature_cols": feature_cols,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Wrote VAEP model to {args.out}")


if __name__ == "__main__":
    main()
