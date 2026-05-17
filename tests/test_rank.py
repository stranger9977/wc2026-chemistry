import pandas as pd
from chemistry.rank import per_team, global_top_n


def test_per_team_returns_pairs_per_nation_sorted_desc():
    pairs = pd.DataFrame([
        {"nation_code": "USA", "player_a": 1, "player_b": 2, "joi90": 0.2, "minutes": 100, "matches": 2},
        {"nation_code": "USA", "player_a": 1, "player_b": 3, "joi90": 0.5, "minutes": 100, "matches": 2},
        {"nation_code": "FRA", "player_a": 4, "player_b": 5, "joi90": 0.7, "minutes": 100, "matches": 2},
    ])
    out = per_team(pairs)
    assert set(out.keys()) == {"USA", "FRA"}
    usa = out["USA"]
    assert list(usa["joi90"]) == [0.5, 0.2]


def test_global_top_n_caps_to_n_rows():
    pairs = pd.DataFrame([
        {"nation_code": "USA", "player_a": i, "player_b": i+100, "joi90": 0.1 * i, "minutes": 100, "matches": 2}
        for i in range(1, 11)
    ])
    out = global_top_n(pairs, n=3)
    assert len(out) == 3
    assert out["joi90"].is_monotonic_decreasing


def test_global_top_n_filters_to_minimum_minutes():
    pairs = pd.DataFrame([
        {"nation_code": "USA", "player_a": 1, "player_b": 2, "joi90": 1.0, "minutes": 30, "matches": 1},
        {"nation_code": "FRA", "player_a": 3, "player_b": 4, "joi90": 0.5, "minutes": 180, "matches": 2},
    ])
    out = global_top_n(pairs, n=5, min_minutes=90)
    assert len(out) == 1
    assert out.iloc[0]["nation_code"] == "FRA"
