import pandas as pd
import pytest
from chemistry.joi import enumerate_interactions


def _spadl_row(*, action_id, game_id, team_id, player_id, time, vaep, type_id=0):
    return {
        "action_id": action_id,
        "game_id": game_id,
        "team_id": team_id,
        "player_id": player_id,
        "period_id": 1,
        "time_seconds": time,
        "type_id": type_id,
        "result_id": 1,
        "vaep_value": vaep,
        "start_x": 50.0, "start_y": 34.0, "end_x": 60.0, "end_y": 34.0,
    }


def test_consecutive_actions_same_team_form_interaction():
    spadl = pd.DataFrame([
        _spadl_row(action_id=1, game_id=1, team_id=10, player_id=100, time=1.0, vaep=0.05),
        _spadl_row(action_id=2, game_id=1, team_id=10, player_id=101, time=2.0, vaep=0.10),
    ])
    interactions = enumerate_interactions(spadl)
    assert len(interactions) == 1
    row = interactions.iloc[0]
    assert row["player_p"] == 100
    assert row["player_q"] == 101
    assert row["vaep_pair"] == pytest.approx(0.15)


def test_actions_by_different_teams_do_not_form_interaction():
    spadl = pd.DataFrame([
        _spadl_row(action_id=1, game_id=1, team_id=10, player_id=100, time=1.0, vaep=0.05),
        _spadl_row(action_id=2, game_id=1, team_id=20, player_id=200, time=2.0, vaep=0.10),
    ])
    interactions = enumerate_interactions(spadl)
    assert len(interactions) == 0


def test_same_player_consecutive_does_not_form_pair():
    spadl = pd.DataFrame([
        _spadl_row(action_id=1, game_id=1, team_id=10, player_id=100, time=1.0, vaep=0.05),
        _spadl_row(action_id=2, game_id=1, team_id=10, player_id=100, time=2.0, vaep=0.10),
    ])
    interactions = enumerate_interactions(spadl)
    assert len(interactions) == 0


def test_only_eligible_action_types(monkeypatch):
    from chemistry import joi
    spadl = pd.DataFrame([
        _spadl_row(action_id=1, game_id=1, team_id=10, player_id=100, time=1.0, vaep=0.05, type_id=1),
        _spadl_row(action_id=2, game_id=1, team_id=10, player_id=101, time=2.0, vaep=0.10, type_id=1),
    ])
    monkeypatch.setattr(joi, "ELIGIBLE_TYPES", frozenset({99}))
    out = joi.enumerate_interactions(spadl)
    assert len(out) == 0
