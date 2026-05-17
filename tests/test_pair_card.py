from chemistry.pair_card import render_pair_card


def test_pair_card_writes_file(tmp_path):
    pair = {
        "player_a_name": "Christian Pulisic",
        "player_b_name": "Tyler Adams",
        "joi90": 0.412, "minutes": 540, "matches": 8,
    }
    out = tmp_path / "card.png"
    render_pair_card(pair, nation="United States", flag_iso="us", out_path=out)
    assert out.exists() and out.stat().st_size > 0
