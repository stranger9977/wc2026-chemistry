import pandas as pd
from chemistry.stats import goals_and_assists


def test_goal_with_assist():
    spadl = pd.DataFrame([
        {"game_id": 1, "period_id": 1, "time_seconds": 10.0, "team_id": 10,
         "player_id": 100, "type_id": 0, "result_id": 1},  # pass by player 100
        {"game_id": 1, "period_id": 1, "time_seconds": 11.0, "team_id": 10,
         "player_id": 101, "type_id": 11, "result_id": 1}, # shot/goal by 101
    ])
    out = goals_and_assists(spadl)
    goals = dict(zip(out["player_id"], out["goals"]))
    assists = dict(zip(out["player_id"], out["assists"]))
    assert goals.get(101) == 1
    assert assists.get(100) == 1


def test_solo_goal_no_assist():
    spadl = pd.DataFrame([
        {"game_id": 1, "period_id": 1, "time_seconds": 10.0, "team_id": 10,
         "player_id": 100, "type_id": 21, "result_id": 1},   # dribble (eligible)
        {"game_id": 1, "period_id": 1, "time_seconds": 11.0, "team_id": 10,
         "player_id": 100, "type_id": 11, "result_id": 1},   # same player shoots & scores
    ])
    out = goals_and_assists(spadl)
    goals = dict(zip(out["player_id"], out["goals"]))
    assists = dict(zip(out["player_id"], out["assists"]))
    assert goals.get(100) == 1
    assert 100 not in assists or assists[100] == 0
