"""Render exports/ — video-ready asset bundle."""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

from chemistry.render import (
    PitchOptions, render_pitch_chemistry, fig_to_png, fig_to_svg, svg_to_4k_png,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("export")

OUT_JSON = Path("outputs/chemistry.json")
EXPORTS = Path("exports")


def export_nation(code: str, entry: dict) -> None:
    nat_dir = EXPORTS / "nations" / code
    nat_dir.mkdir(parents=True, exist_ok=True)

    svg_path = nat_dir / "pitch.svg"
    fig = render_pitch_chemistry(entry, PitchOptions(background="transparent"))
    fig_to_svg(fig, svg_path)
    plt.close(fig)

    svg_to_4k_png(svg_path, nat_dir / "pitch.png", width=3840)

    fig2 = render_pitch_chemistry(entry, PitchOptions(background=entry["squad"]["team_color"]))
    fig_to_png(fig2, nat_dir / "pitch_branded.png", transparent=False, dpi=200)
    plt.close(fig2)

    (nat_dir / "data.json").write_text(json.dumps(entry, indent=2, ensure_ascii=False))

    from chemistry.pair_card import render_pair_card, CardOptions
    cards_dir = nat_dir / "pair_cards"
    cards_dir.mkdir(exist_ok=True)
    opts = CardOptions(accent=entry["squad"]["team_color"])
    for i, p in enumerate(entry["pairs"][:5], start=1):
        out = cards_dir / f"{i:02d}_{p['player_a_name'].replace(' ', '_')}_{p['player_b_name'].replace(' ', '_')}.png"
        render_pair_card(p, nation=entry["squad"]["nation"],
                         flag_iso=entry["squad"]["flag_iso"], opts=opts, out_path=out)

    log.info("Exported %s", code)


def main() -> None:
    if not OUT_JSON.exists():
        raise RuntimeError(f"{OUT_JSON} missing — run scripts/build.py first")
    doc = json.loads(OUT_JSON.read_text())

    if EXPORTS.exists():
        shutil.rmtree(EXPORTS / "nations", ignore_errors=True)
    (EXPORTS / "nations").mkdir(parents=True, exist_ok=True)

    for code, entry in doc["nations"].items():
        try:
            export_nation(code, entry)
        except Exception as exc:
            log.exception("Failed exporting %s: %s", code, exc)

    from chemistry.render import render_leaderboard, render_landing_grid
    render_leaderboard(doc, EXPORTS / "leaderboard.png")
    render_landing_grid(doc, EXPORTS / "landing_grid.png")

    manifest = {
        "generated_from": str(OUT_JSON),
        "nations": list(doc["nations"].keys()),
        "assets_per_nation": ["pitch.svg", "pitch.png", "pitch_branded.png", "data.json"],
    }
    (EXPORTS / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Wrote %d nations to %s", len(doc["nations"]), EXPORTS)


if __name__ == "__main__":
    main()
