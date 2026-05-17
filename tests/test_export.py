import json
import pandas as pd
from chemistry import export as ex
from chemistry.squads import Squad, Player


def test_build_chemistry_json_includes_every_nation(tmp_path):
    joi90 = pd.DataFrame([
        {"player_a": 100, "player_b": 101, "joi90": 0.5, "minutes": 180, "matches": 2, "team_id": 1},
    ])
    lineups = pd.DataFrame([
        {"game_id": 1, "team_id": 1, "team_name": "Testland",
         "player_id": 100, "player_name": "Alpha", "position": "Goalkeeper",
         "from_minute": 0, "to_minute": 90},
        {"game_id": 1, "team_id": 1, "team_name": "Testland",
         "player_id": 101, "player_name": "Beta", "position": "Center Back",
         "from_minute": 0, "to_minute": 90},
    ])
    squad = Squad(
        nation="Testland", nation_code="TST", flag_iso="tt",
        manager="X", formation="4-3-3", team_color="#000",
        players=[Player(name="Alpha", club="FC", position="GK"),
                 Player(name="Beta",  club="FC", position="CB")],
    )
    out = ex.build_chemistry_json(joi90, lineups, {"TST": squad}, tmp_path / "ch.json")
    doc = json.loads(out.read_text())
    assert "TST" in doc["nations"]
    assert len(doc["nations"]["TST"]["pairs"]) == 1
    assert doc["leaderboard"][0]["joi90"] == 0.5
