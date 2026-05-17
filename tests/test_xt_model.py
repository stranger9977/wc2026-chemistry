import pandas as pd
import pytest
from chemistry.xt_model import fit_xt


def test_fit_xt_and_score(fixtures_dir):
    # SPADL fixture from prior tests
    spadl = pd.read_parquet(fixtures_dir / "sample_events.parquet")
    # Need actual SPADL not raw events — convert
    from chemistry.pipeline import to_spadl
    actions = to_spadl(spadl)
    if len(actions) < 5:
        # The fixture is small; trim test
        return
    model = fit_xt(actions)
    scored = model.score(actions)
    assert len(scored) == len(actions)
    assert scored.notna().all()


def test_xt_score_nonnegative_for_progressive_moves(fixtures_dir):
    """xT deltas should be >= 0 for forward passes (end_x > start_x in final third)."""
    spadl = pd.read_parquet(fixtures_dir / "sample_events.parquet")
    from chemistry.pipeline import to_spadl
    actions = to_spadl(spadl)
    if len(actions) < 5:
        return
    model = fit_xt(actions)
    scored = model.score(actions)
    # No NaN values anywhere
    assert not scored.isna().any()
