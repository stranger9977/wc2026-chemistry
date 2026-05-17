# WC 2026 Chemistry

Player chemistry pipeline for the FIFA World Cup 2026. Computes offensive chemistry (JOI90) between every player pair within each of the 32 squads from StatsBomb open international event data, and renders video-ready chart assets (4K PNG, SVG, lower-third pair cards, leaderboard graphics) plus a GitHub Pages chart browser.

**Method:** based on Bransen & Van Haaren, *Player Chemistry: Striving for a Perfectly Balanced Soccer Team*, MITSSAC 2020 ([PDF](https://www.janvanhaaren.be/assets/papers/mitssac-2020-chemistry.pdf)).

**Live site:** https://stranger9977.github.io/wc2026-chemistry/

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"

# 1. Stub squad YAMLs (already committed, but re-runnable)
python -m scripts.bootstrap_squads

# 2. Edit squads/wc2026/<CODE>.yaml as official rosters land

# 3. Run the build (uses heuristic VAEP; pass nothing to use trained)
python -m scripts.build --heuristic-vaep

# 4. Render every PNG/SVG/pair-card export
python -m scripts.export

# 5. Serve locally
python -m http.server 8080
```

## Layout

- `chemistry/` — Python package: ingest, SPADL+VAEP pipeline, JOI math, squad/roster handling, render
- `scripts/` — entry points: bootstrap_squads, train_vaep, build, export, fetch_headshots
- `squads/wc2026/` — hand-curated YAML rosters, one per nation
- `outputs/chemistry.json` — pipeline output the site reads
- `exports/` — video-ready assets bundle, per nation + leaderboard + landing grid
- `site/` — chart browser served on GitHub Pages
- `assets/flags/` — vendored MIT-licensed flag SVGs
- `assets/headshots/` — best-effort Wikipedia Commons headshots
- `docs/superpowers/specs/` — design spec
- `docs/superpowers/plans/` — this implementation plan

## Squad editing

Each `squads/wc2026/<CODE>.yaml` is hand-curated. Edit `players:` as call-ups are announced; replace `manager: TBD` once known. Re-run `scripts/build` + `scripts/export` to refresh.

## Data sources & credits

See [CREDITS.md](CREDITS.md).
