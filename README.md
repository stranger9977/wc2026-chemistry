# WC 2026 Chemistry

Player chemistry pipeline for the FIFA World Cup 2026 — computes offensive chemistry between every player pair within each squad from StatsBomb open international event data, and renders video-ready chart assets.

See [docs/superpowers/specs/2026-05-17-wc2026-chemistry-design.md](docs/superpowers/specs/2026-05-17-wc2026-chemistry-design.md) for the design.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"           # add this line
python -m scripts.build      # full pipeline
python -m scripts.export     # render exports/
```

## Layout

- `chemistry/` — pipeline package
- `squads/wc2026/` — hand-curated rosters (YAML)
- `outputs/chemistry.json` — pipeline output
- `exports/` — video-ready PNG/SVG assets
- `site/` — chart browser served on GitHub Pages

## Squad editing

`squads/wc2026/<CODE>.yaml` files are hand-curated. Edit the `players:` list as call-ups are announced. Re-run `python -m scripts.build` to refresh chemistry. The `manager` field reads `TBD` by default — replace once known.

To add a fresh stub: extend the `QUALIFIERS` list in `scripts/bootstrap_squads.py` and rerun.
