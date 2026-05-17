from chemistry.render import render_pitch_chemistry, fig_to_svg


def test_render_pitch_smoke(tmp_path):
    entry = {
        "squad": {
            "nation": "Testland", "nation_code": "TST", "flag_iso": "tt",
            "manager": "X", "formation": "4-3-3", "team_color": "#112233",
            "players": [
                {"name": "Alpha", "club": "FC", "position": "GK"},
                {"name": "Beta",  "club": "FC", "position": "CB"},
                {"name": "Gamma", "club": "FC", "position": "ST"},
            ],
        },
        "pairs": [
            {"player_a_id": 1, "player_b_id": 2,
             "player_a_name": "Alpha", "player_b_name": "Beta",
             "joi90": 0.4, "minutes": 180, "matches": 2},
            {"player_a_id": 2, "player_b_id": 3,
             "player_a_name": "Beta",  "player_b_name": "Gamma",
             "joi90": 0.6, "minutes": 270, "matches": 3},
        ],
        "coverage": {"matches": 3},
    }
    fig = render_pitch_chemistry(entry)
    out = fig_to_svg(fig, tmp_path / "pitch.svg")
    assert out.exists()
    assert out.stat().st_size > 0
