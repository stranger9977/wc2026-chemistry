import pandas as pd
from chemistry.players import build_player_index, resolve_squad_ids


def test_build_player_index_returns_name_to_id_map():
    lineups = pd.DataFrame([
        {"game_id": 1, "player_id": 100, "player_name": "Christian Pulisic", "team_name": "United States"},
        {"game_id": 1, "player_id": 101, "player_name": "Tyler Adams",       "team_name": "United States"},
        {"game_id": 2, "player_id": 100, "player_name": "Christian Pulisic", "team_name": "United States"},
    ])
    index = build_player_index(lineups)
    assert index["United States"]["christian pulisic"] == 100
    assert index["United States"]["tyler adams"] == 101


def test_resolve_squad_ids_returns_matched_and_unmatched():
    lineups = pd.DataFrame([
        {"game_id": 1, "player_id": 100, "player_name": "Christian Pulisic", "team_name": "United States"},
        {"game_id": 1, "player_id": 101, "player_name": "Tyler Adams",       "team_name": "United States"},
    ])
    names = ["Christian Pulisic", "Tyler Adams", "Phantom Player"]
    matched, unmatched = resolve_squad_ids("United States", names, lineups)
    assert matched == {"Christian Pulisic": 100, "Tyler Adams": 101}
    assert unmatched == ["Phantom Player"]
