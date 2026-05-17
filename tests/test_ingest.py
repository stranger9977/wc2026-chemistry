from chemistry.ingest import INTERNATIONAL_COMPETITIONS, target_competitions


def test_target_competitions_returns_list_of_dicts():
    comps = target_competitions()
    assert isinstance(comps, list)
    assert len(comps) >= 4
    for c in comps:
        assert "competition_id" in c
        assert "season_id" in c
        assert "label" in c


def test_world_cup_2022_is_in_targets():
    labels = {c["label"] for c in target_competitions()}
    assert any("World Cup" in l and "2022" in l for l in labels)
