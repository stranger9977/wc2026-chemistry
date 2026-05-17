import pandas as pd
from chemistry.minutes import LineupsMinutes, shared_minutes


def test_two_starters_who_both_play_full_game_share_90_minutes():
    lineups = pd.DataFrame([
        {"game_id": 1, "team_id": 10, "player_id": 100, "from_minute": 0, "to_minute": 90},
        {"game_id": 1, "team_id": 10, "player_id": 101, "from_minute": 0, "to_minute": 90},
    ])
    out = shared_minutes(lineups, 1, 100, 101)
    assert out == 90.0


def test_substitute_at_60_shares_30_minutes_with_starter():
    lineups = pd.DataFrame([
        {"game_id": 1, "team_id": 10, "player_id": 100, "from_minute": 0,  "to_minute": 90},
        {"game_id": 1, "team_id": 10, "player_id": 101, "from_minute": 60, "to_minute": 90},
    ])
    out = shared_minutes(lineups, 1, 100, 101)
    assert out == 30.0


def test_players_who_never_overlap_share_zero_minutes():
    lineups = pd.DataFrame([
        {"game_id": 1, "team_id": 10, "player_id": 100, "from_minute": 0,  "to_minute": 60},
        {"game_id": 1, "team_id": 10, "player_id": 101, "from_minute": 60, "to_minute": 90},
    ])
    out = shared_minutes(lineups, 1, 100, 101)
    assert out == 0.0


def test_lineups_minutes_provider_uses_dataframe():
    lineups = pd.DataFrame([
        {"game_id": 1, "team_id": 10, "player_id": 100, "from_minute": 0, "to_minute": 90},
        {"game_id": 1, "team_id": 10, "player_id": 101, "from_minute": 0, "to_minute": 90},
    ])
    provider = LineupsMinutes(lineups)
    assert provider.minutes(1, 100, 101) == 90.0
