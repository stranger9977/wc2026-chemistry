import pandas as pd
from chemistry.pipeline import to_spadl

SPADL_COLUMNS = {
    "game_id", "team_id", "player_id", "period_id", "time_seconds",
    "start_x", "start_y", "end_x", "end_y",
    "type_id", "result_id", "bodypart_id",
}


def test_to_spadl_returns_dataframe_with_expected_columns(fixtures_dir):
    events = pd.read_parquet(fixtures_dir / "sample_events.parquet")
    spadl = to_spadl(events)
    assert isinstance(spadl, pd.DataFrame)
    assert SPADL_COLUMNS.issubset(set(spadl.columns))
    assert len(spadl) > 0
