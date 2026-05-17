import pytest
from chemistry.squads import Squad, Player, load_squad, load_all_squads


def test_load_squad_parses_yaml(fixtures_dir):
    squad = load_squad(fixtures_dir / "squads" / "TST.yaml")
    assert isinstance(squad, Squad)
    assert squad.nation == "Testland"
    assert squad.nation_code == "TST"
    assert squad.flag_iso == "tt"
    assert squad.manager == "Jane Doe"
    assert squad.formation == "4-3-3"
    assert squad.team_color == "#112233"
    assert len(squad.players) == 3
    p = squad.players[0]
    assert isinstance(p, Player)
    assert p.name == "Alpha One"
    assert p.club == "FC Alpha"
    assert p.position == "GK"
    assert p.headshot is None
    assert squad.players[2].headshot == "gamma.jpg"


def test_load_all_squads_finds_every_yaml(fixtures_dir):
    squads = load_all_squads(fixtures_dir / "squads")
    assert "TST" in squads
    assert squads["TST"].nation == "Testland"


def test_invalid_yaml_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("nation: oops\n")  # missing required fields
    with pytest.raises(Exception):
        load_squad(bad)


import pandas as pd
from chemistry.squads import filter_pairs_by_squad


def test_filter_pairs_keeps_only_pairs_where_both_players_are_in_squad():
    pairs = pd.DataFrame([
        {"player_a": 100, "player_b": 101, "joi90": 0.5, "minutes": 90, "matches": 1},
        {"player_a": 101, "player_b": 999, "joi90": 0.3, "minutes": 90, "matches": 1},
        {"player_a": 800, "player_b": 999, "joi90": 0.4, "minutes": 90, "matches": 1},
    ])
    roster = {100: "Alpha", 101: "Beta", 102: "Gamma"}
    out = filter_pairs_by_squad(pairs, set(roster.keys()))
    assert len(out) == 1
    assert int(out.iloc[0]["player_a"]) == 100
