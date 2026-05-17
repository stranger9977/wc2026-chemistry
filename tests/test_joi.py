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


from chemistry.joi import joi_per_match, joi90_window, MinutesProvider


class _StubMinutes:
    """Minutes provider that always returns the same shared minutes."""
    def __init__(self, table):
        self.table = table  # dict {(game_id, p, q): minutes}
    def minutes(self, game_id, player_p, player_q):
        a, b = sorted((player_p, player_q))
        return self.table.get((game_id, a, b), 0.0)


def test_joi_per_match_aggregates_both_orderings():
    interactions = pd.DataFrame([
        {"game_id": 1, "team_id": 10, "player_p": 100, "player_q": 101, "vaep_pair": 0.10},
        {"game_id": 1, "team_id": 10, "player_p": 101, "player_q": 100, "vaep_pair": 0.05},
        {"game_id": 2, "team_id": 10, "player_p": 100, "player_q": 101, "vaep_pair": 0.20},
    ])
    out = joi_per_match(interactions)
    row = out[(out["game_id"] == 1) & (out["player_a"] == 100) & (out["player_b"] == 101)].iloc[0]
    assert row["joi"] == pytest.approx(0.15)
    row2 = out[(out["game_id"] == 2) & (out["player_a"] == 100) & (out["player_b"] == 101)].iloc[0]
    assert row2["joi"] == pytest.approx(0.20)


def test_joi90_window_normalises_per_90_shared_minutes():
    per_match = pd.DataFrame([
        {"game_id": 1, "team_id": 10, "player_a": 100, "player_b": 101, "joi": 0.30},
        {"game_id": 2, "team_id": 10, "player_a": 100, "player_b": 101, "joi": 0.15},
    ])
    mins = _StubMinutes({(1, 100, 101): 45.0, (2, 100, 101): 90.0})
    out = joi90_window(per_match, mins)
    row = out[(out["player_a"] == 100) & (out["player_b"] == 101)].iloc[0]
    assert row["joi90"] == pytest.approx(0.30)
    assert row["minutes"] == pytest.approx(135.0)
    assert row["matches"] == 2
