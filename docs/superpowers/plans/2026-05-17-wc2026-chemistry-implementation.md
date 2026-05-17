# WC 2026 Chemistry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a data pipeline that computes offensive chemistry (JOI) between every player pair within each of the 32 FIFA WC 2026 squads from StatsBomb open international event data, and renders video-ready chart assets (4K PNG, SVG, pair-card lower-thirds, leaderboard graphics) plus a GitHub Pages chart browser site.

**Architecture:** Python package `chemistry/` for the data pipeline (StatsBomb → SPADL → VAEP → interactions → JOI90 → JSON). Static site under `site/` consumes a single `outputs/chemistry.json` and renders an interactive D3 pitch chart. A separate Python export script renders publication-quality PNG/SVG/pair-card assets via `mplsoccer` + `matplotlib` directly from the same JSON, so site and exports share one source of truth. Squad rosters live as hand-curated YAML in `squads/wc2026/`. The site deploys to GitHub Pages at `stranger9977.github.io/wc2026-chemistry/`.

**Tech Stack:**
- Python 3.11+, `statsbombpy`, `socceraction`, `pandas`, `pyarrow`, `pydantic`, `pyyaml`, `mplsoccer`, `matplotlib`, `Pillow`, `cairosvg`, `xgboost`, `pytest`
- Static site: vanilla HTML/CSS + D3.js v7, served by GitHub Pages
- Flags from `flag-icons` (MIT), headshots best-effort from Wikipedia Commons

**Spec:** `docs/superpowers/specs/2026-05-17-wc2026-chemistry-design.md`

---

## Phase 1 — Foundation

### Task 1: Project scaffolding

**Files:**
- Create: `/Users/nick/wc2026-chemistry/pyproject.toml`
- Create: `/Users/nick/wc2026-chemistry/requirements.txt`
- Create: `/Users/nick/wc2026-chemistry/chemistry/__init__.py`
- Create: `/Users/nick/wc2026-chemistry/tests/__init__.py`
- Create: `/Users/nick/wc2026-chemistry/tests/conftest.py`
- Create: `/Users/nick/wc2026-chemistry/README.md`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "wc2026-chemistry"
version = "0.1.0"
description = "Player chemistry pipeline for FIFA WC 2026"
requires-python = ">=3.11"

[tool.setuptools]
packages = ["chemistry"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
```

- [ ] **Step 2: Create `requirements.txt`**

```
statsbombpy>=1.13
socceraction>=1.5
pandas>=2.2
pyarrow>=15
pydantic>=2.6
pyyaml>=6.0
mplsoccer>=1.4
matplotlib>=3.8
Pillow>=10.2
cairosvg>=2.7
xgboost>=2.0
scikit-learn>=1.4
tqdm>=4.66
pytest>=8.0
pytest-cov>=4.1
```

- [ ] **Step 3: Create empty `chemistry/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 4: Create empty `tests/__init__.py`** (empty file)

- [ ] **Step 5: Create `tests/conftest.py`**

```python
import pathlib
import pytest

REPO_ROOT = pathlib.Path(__file__).parent.parent

@pytest.fixture
def repo_root() -> pathlib.Path:
    return REPO_ROOT

@pytest.fixture
def fixtures_dir(repo_root) -> pathlib.Path:
    return repo_root / "tests" / "fixtures"
```

- [ ] **Step 6: Create `README.md`**

```markdown
# WC 2026 Chemistry

Player chemistry pipeline for the FIFA World Cup 2026 — computes offensive chemistry between every player pair within each squad from StatsBomb open international event data, and renders video-ready chart assets.

See [docs/superpowers/specs/2026-05-17-wc2026-chemistry-design.md](docs/superpowers/specs/2026-05-17-wc2026-chemistry-design.md) for the design.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.build      # full pipeline
python -m scripts.export     # render exports/
```

## Layout

- `chemistry/` — pipeline package
- `squads/wc2026/` — hand-curated rosters (YAML)
- `outputs/chemistry.json` — pipeline output
- `exports/` — video-ready PNG/SVG assets
- `site/` — chart browser served on GitHub Pages
```

- [ ] **Step 7: Install and verify environment**

Run:
```bash
cd /Users/nick/wc2026-chemistry && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pytest --collect-only
```

Expected: pip install completes; `pytest --collect-only` reports 0 tests collected but exits 0.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml requirements.txt chemistry/ tests/ README.md
git commit -m "Project scaffolding: package, requirements, test setup"
```

---

## Phase 2 — Data pipeline

### Task 2: StatsBomb ingest module

**Files:**
- Create: `chemistry/ingest.py`
- Create: `tests/test_ingest.py`

- [ ] **Step 1: Write failing test in `tests/test_ingest.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest.py -v`
Expected: ImportError / ModuleNotFoundError for `chemistry.ingest`.

- [ ] **Step 3: Create `chemistry/ingest.py`**

```python
"""Pull and cache StatsBomb open data for international competitions."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from statsbombpy import sb

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "data" / "raw"

# Curated list — only international tournaments + qualifiers we care about
# for chemistry leading up to WC 2026. Identified by StatsBomb competition_id
# and season_id from sb.competitions().
INTERNATIONAL_COMPETITIONS: list[dict] = [
    {"competition_id": 43, "season_id": 106, "label": "FIFA World Cup 2022"},
    {"competition_id": 55, "season_id": 282, "label": "UEFA Euro 2024"},
    {"competition_id": 55, "season_id": 43,  "label": "UEFA Euro 2020"},
    {"competition_id": 223,"season_id": 282, "label": "Copa America 2024"},
    {"competition_id": 72, "season_id": 107, "label": "FIFA Women's World Cup 2023"},
]


def target_competitions() -> list[dict]:
    """Return the curated list of competitions to ingest."""
    return list(INTERNATIONAL_COMPETITIONS)


def list_available_competitions() -> list[dict]:
    """Return everything StatsBomb has in open data, as plain dicts."""
    df = sb.competitions()
    return df.to_dict(orient="records")


def cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def fetch_competition(competition_id: int, season_id: int) -> Path:
    """Download all matches + events for one competition-season into the cache.

    Returns the directory it wrote to.
    """
    base = cache_dir() / f"{competition_id}_{season_id}"
    base.mkdir(parents=True, exist_ok=True)

    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    matches_path = base / "matches.parquet"
    matches.to_parquet(matches_path, index=False)
    logger.info("Wrote %s matches to %s", len(matches), matches_path)

    events_dir = base / "events"
    events_dir.mkdir(exist_ok=True)
    for match_id in matches["match_id"]:
        target = events_dir / f"{match_id}.parquet"
        if target.exists():
            continue
        events = sb.events(match_id=match_id)
        events.to_parquet(target, index=False)
    logger.info("Cached events for competition %s season %s", competition_id, season_id)
    return base


def fetch_all_targets() -> list[Path]:
    """Fetch every competition in INTERNATIONAL_COMPETITIONS."""
    paths = []
    for comp in INTERNATIONAL_COMPETITIONS:
        path = fetch_competition(comp["competition_id"], comp["season_id"])
        paths.append(path)
    return paths
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingest.py -v`
Expected: PASS for both tests.

- [ ] **Step 5: Smoke-test the ingest against one competition**

Run:
```bash
python -c "from chemistry.ingest import fetch_competition; p = fetch_competition(43, 106); print('OK:', p)"
```
Expected: prints `OK: <path>` and `data/raw/43_106/matches.parquet` exists with > 0 rows. (This may take 1–2 minutes; if rate-limited, retry.)

- [ ] **Step 6: Update `.gitignore` if needed** — already covers `data/raw/`. Verify with `cat .gitignore | grep data/raw`.

- [ ] **Step 7: Commit**

```bash
git add chemistry/ingest.py tests/test_ingest.py
git commit -m "Ingest: cache StatsBomb open data for target international comps"
```

---

### Task 3: SPADL conversion

**Files:**
- Create: `chemistry/pipeline.py`
- Create: `tests/test_pipeline.py`
- Create: `tests/fixtures/sample_events.parquet` (generated in step 1)

- [ ] **Step 1: Generate a small fixture (one match)**

Run (only after Task 2 step 5 succeeded):
```bash
python -c "
import pandas as pd, pathlib
src = pathlib.Path('data/raw/43_106/events').glob('*.parquet')
first = next(src)
df = pd.read_parquet(first).head(200)
out = pathlib.Path('tests/fixtures'); out.mkdir(parents=True, exist_ok=True)
df.to_parquet(out / 'sample_events.parquet', index=False)
print('Wrote', out / 'sample_events.parquet', len(df), 'rows')
"
```
Expected: writes `tests/fixtures/sample_events.parquet`.

- [ ] **Step 2: Write failing test in `tests/test_pipeline.py`**

```python
import pandas as pd
from chemistry.pipeline import to_spadl

SPADL_COLUMNS = {
    "game_id", "team_id", "player_id", "period_id", "time_seconds",
    "start_x", "start_y", "end_x", "end_y",
    "type_id", "result_id", "bodypart_id",
}

def test_to_spadl_returns_dataframe_with_expected_columns(fixtures_dir):
    events = pd.read_parquet(fixtures_dir / "sample_events.parquet")
    spadl = to_spadl(events)
    assert isinstance(spadl, pd.DataFrame)
    assert SPADL_COLUMNS.issubset(set(spadl.columns))
    assert len(spadl) > 0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py::test_to_spadl_returns_dataframe_with_expected_columns -v`
Expected: ImportError for `chemistry.pipeline`.

- [ ] **Step 4: Create `chemistry/pipeline.py` (initial SPADL part)**

```python
"""SPADL conversion + VAEP scoring of StatsBomb events."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from socceraction.spadl import statsbomb as spadl_sb
from socceraction.spadl import config as spadl_cfg

logger = logging.getLogger(__name__)

SPADL_DIR = Path(__file__).parent.parent / "data" / "spadl"


def to_spadl(events: pd.DataFrame) -> pd.DataFrame:
    """Convert one match's StatsBomb events into SPADL actions.

    statsbombpy already gives us flattened events; socceraction's
    statsbomb converter expects the raw event format with the
    statsbomb id column. We adapt the dataframe accordingly.
    """
    # socceraction expects the event JSON-shape; statsbombpy normalises
    # column names. Map back to what the converter needs.
    spadl = spadl_sb.convert_to_actions(events, home_team_id=None)
    return spadl


def convert_competition(competition_dir: Path, out_dir: Path = SPADL_DIR) -> Path:
    """Convert every match in a cached competition dir to SPADL."""
    out_dir.mkdir(parents=True, exist_ok=True)
    comp_out = out_dir / competition_dir.name
    comp_out.mkdir(exist_ok=True)
    events_dir = competition_dir / "events"
    for events_path in sorted(events_dir.glob("*.parquet")):
        target = comp_out / events_path.name
        if target.exists():
            continue
        events = pd.read_parquet(events_path)
        spadl = to_spadl(events)
        spadl.to_parquet(target, index=False)
    logger.info("Converted %s to SPADL at %s", competition_dir.name, comp_out)
    return comp_out
```

- [ ] **Step 5: Adjust to_spadl if needed**

The signature of `spadl_sb.convert_to_actions` varies by socceraction version. Verify:
```bash
python -c "from socceraction.spadl import statsbomb as s; help(s.convert_to_actions)" | head -30
```
If the signature requires per-team conversion (it usually does — home then away), update `to_spadl` to handle both teams. Reference impl:

```python
def to_spadl(events: pd.DataFrame) -> pd.DataFrame:
    teams = events["team_id"].dropna().unique()
    parts = []
    for tid in teams:
        parts.append(spadl_sb.convert_to_actions(events, home_team_id=int(tid)))
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["period_id", "time_seconds"]).reset_index(drop=True)
```
Patch the file accordingly.

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add chemistry/pipeline.py tests/test_pipeline.py tests/fixtures/sample_events.parquet
git commit -m "Pipeline: StatsBomb -> SPADL conversion with fixture-backed test"
```

---

### Task 4: VAEP scoring

**Files:**
- Modify: `chemistry/pipeline.py`
- Create: `chemistry/vaep_model.py`
- Create: `scripts/train_vaep.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing test in `tests/test_pipeline.py`**

Append to the file:
```python
import numpy as np
from chemistry.pipeline import score_vaep, load_vaep_model


def test_score_vaep_adds_vaep_value_column(fixtures_dir, tmp_path):
    events = pd.read_parquet(fixtures_dir / "sample_events.parquet")
    spadl = to_spadl(events)
    # Use the dummy / heuristic model for the unit test
    model = load_vaep_model(kind="heuristic")
    scored = score_vaep(spadl, model)
    assert "vaep_value" in scored.columns
    assert scored["vaep_value"].notna().sum() > 0
    assert np.isfinite(scored["vaep_value"]).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py::test_score_vaep_adds_vaep_value_column -v`
Expected: ImportError for `score_vaep`.

- [ ] **Step 3: Create `chemistry/vaep_model.py`**

```python
"""VAEP scoring — wraps socceraction's VAEP and offers a heuristic fallback."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).parent.parent / "data" / "vaep"


class VaepModel(Protocol):
    def score(self, spadl: pd.DataFrame) -> pd.Series: ...


@dataclass
class HeuristicVaep:
    """Quick & dirty heuristic for development and unit tests.

    Approximates VAEP by rewarding actions that progress the ball
    toward goal. Real model is trained via scripts/train_vaep.py.
    """

    def score(self, spadl: pd.DataFrame) -> pd.Series:
        # SPADL pitch is 105x68; reward end_x closer to 105 (opponent goal)
        progress = spadl["end_x"] - spadl["start_x"]
        attacking_third = (spadl["end_x"] > 70).astype(float)
        shot_bonus = (spadl["type_id"] == 11).astype(float) * 0.4  # shot type
        successful = (spadl["result_id"] == 1).astype(float)
        val = (
            0.01 * progress
            + 0.05 * attacking_third
            + shot_bonus
        ) * successful
        return val.astype(float)


class TrainedVaep:
    """Loads a socceraction-trained VAEP model from disk."""

    def __init__(self, path: Path):
        with open(path, "rb") as f:
            self._model = pickle.load(f)

    def score(self, spadl: pd.DataFrame) -> pd.Series:
        # The trained model is an xgboost classifier wrapped with feature builders.
        # See scripts/train_vaep.py for the trained-side details.
        from socceraction.vaep import features as fs
        from socceraction.vaep import formula
        X = fs.feature_set(spadl)
        p_scores, p_concedes = self._model.predict_proba(X)[:, 1], np.zeros(len(spadl))
        return formula.value(spadl, p_scores, p_concedes)["vaep_value"]


def load_vaep_model(kind: Literal["heuristic", "trained"] = "trained",
                    path: Path | None = None) -> VaepModel:
    if kind == "heuristic":
        return HeuristicVaep()
    target = path or (MODELS_DIR / "vaep.pkl")
    if not target.exists():
        raise FileNotFoundError(
            f"No trained VAEP model at {target}. Run scripts/train_vaep.py first, "
            f"or use kind='heuristic' for testing."
        )
    return TrainedVaep(target)
```

- [ ] **Step 4: Add `score_vaep` to `chemistry/pipeline.py`**

Append:
```python
from chemistry.vaep_model import VaepModel, load_vaep_model  # noqa: E402

VAEP_DIR = Path(__file__).parent.parent / "data" / "vaep"


def score_vaep(spadl: pd.DataFrame, model: VaepModel) -> pd.DataFrame:
    out = spadl.copy()
    out["vaep_value"] = model.score(spadl)
    return out


def score_competition(spadl_competition_dir: Path,
                      model: VaepModel,
                      out_dir: Path = VAEP_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    comp_out = out_dir / spadl_competition_dir.name
    comp_out.mkdir(exist_ok=True)
    for spadl_path in sorted(spadl_competition_dir.glob("*.parquet")):
        target = comp_out / spadl_path.name
        if target.exists():
            continue
        spadl = pd.read_parquet(spadl_path)
        scored = score_vaep(spadl, model)
        scored.to_parquet(target, index=False)
    return comp_out
```

Also re-export at the top of `pipeline.py`:
```python
from chemistry.vaep_model import load_vaep_model  # re-export
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_pipeline.py::test_score_vaep_adds_vaep_value_column -v`
Expected: PASS.

- [ ] **Step 6: Create `scripts/train_vaep.py` skeleton**

```python
"""Train the VAEP scoring model on cached SPADL data.

Trains a binary classifier predicting whether an action will lead to a
goal within the next k=10 actions, using socceraction's feature builders.
Pickles the model to data/vaep/vaep.pkl.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from socceraction.vaep import features as fs
from socceraction.vaep import labels as lb
from socceraction.vaep import formula  # noqa: F401
from xgboost import XGBClassifier

SPADL_DIR = Path("data/spadl")
OUT = Path("data/vaep/vaep.pkl")


def load_all_spadl() -> pd.DataFrame:
    parts = []
    for comp_dir in sorted(SPADL_DIR.iterdir()):
        if not comp_dir.is_dir():
            continue
        for p in sorted(comp_dir.glob("*.parquet")):
            parts.append(pd.read_parquet(p))
    if not parts:
        raise RuntimeError(f"No SPADL data under {SPADL_DIR}. Run scripts/build.py first.")
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    spadl = load_all_spadl()
    X = fs.feature_set(spadl)
    y = lb.scores(spadl)  # binary: action leads to a goal within k actions

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        n_jobs=-1,
        eval_metric="logloss",
    )
    model.fit(X, y)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(model, f)
    print(f"Wrote VAEP model to {args.out}")


if __name__ == "__main__":
    main()
```

Note for executor: `socceraction.vaep.labels` / `features` APIs may have shifted between versions. If the imports fail, run `python -c "import socceraction.vaep; print(dir(socceraction.vaep))"` and adapt accordingly. The intent is: build features, build binary labels, fit XGBoost, pickle.

- [ ] **Step 7: Commit**

```bash
git add chemistry/pipeline.py chemistry/vaep_model.py scripts/train_vaep.py tests/test_pipeline.py
git commit -m "VAEP: heuristic for tests + trained-model loader + train script"
```

---

### Task 5: Interaction enumeration

**Files:**
- Create: `chemistry/joi.py`
- Create: `tests/test_joi.py`

- [ ] **Step 1: Write failing test in `tests/test_joi.py`**

```python
import pandas as pd
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
    # type_id=1 is dribble, type_id=99 should not exist; ensure filter respects ELIGIBLE_TYPES
    from chemistry import joi
    spadl = pd.DataFrame([
        _spadl_row(action_id=1, game_id=1, team_id=10, player_id=100, time=1.0, vaep=0.05, type_id=1),
        _spadl_row(action_id=2, game_id=1, team_id=10, player_id=101, time=2.0, vaep=0.10, type_id=1),
    ])
    monkeypatch.setattr(joi, "ELIGIBLE_TYPES", frozenset({99}))
    out = joi.enumerate_interactions(spadl)
    assert len(out) == 0
```

Also add `import pytest` at the top.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_joi.py -v`
Expected: ImportError for `chemistry.joi`.

- [ ] **Step 3: Create `chemistry/joi.py`**

```python
"""Joint Offensive Impact (JOI) — pair-wise chemistry from VAEP-scored actions."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

# SPADL action type IDs that count as "on-the-ball" for JOI.
# Paper uses: pass, cross, dribble, take-on, shot.
# socceraction's spadl type_ids:
#   0=pass, 1=cross, 2=throw-in, 3=freekick_crossed, 4=freekick_short,
#   5=corner_crossed, 6=corner_short, 7=take_on, 8=foul, 9=tackle,
#   10=interception, 11=shot, 12=shot_penalty, 13=shot_freekick,
#   14=keeper_save, 15=keeper_claim, 16=keeper_punch, 17=keeper_pick_up,
#   18=clearance, 19=bad_touch, 20=non_action, 21=dribble, 22=goalkick
ELIGIBLE_TYPES: frozenset[int] = frozenset({0, 1, 7, 11, 21, 12, 13})


def enumerate_interactions(spadl: pd.DataFrame) -> pd.DataFrame:
    """Return one row per consecutive same-team action pair with different players.

    Columns:
        game_id, team_id, player_p, player_q,
        vaep_p, vaep_q, vaep_pair, time_p, time_q
    """
    df = spadl.copy()
    df = df[df["type_id"].isin(ELIGIBLE_TYPES)]
    df = df.sort_values(["game_id", "period_id", "time_seconds"]).reset_index(drop=True)

    next_df = df.shift(-1)

    consecutive = (
        (df["game_id"] == next_df["game_id"])
        & (df["team_id"] == next_df["team_id"])
        & (df["player_id"] != next_df["player_id"])
    )
    pair = pd.DataFrame({
        "game_id": df["game_id"],
        "team_id": df["team_id"],
        "player_p": df["player_id"],
        "player_q": next_df["player_id"],
        "vaep_p":   df["vaep_value"],
        "vaep_q":   next_df["vaep_value"],
        "time_p":   df["time_seconds"],
        "time_q":   next_df["time_seconds"],
    })[consecutive].reset_index(drop=True)
    pair["vaep_pair"] = pair["vaep_p"].fillna(0) + pair["vaep_q"].fillna(0)
    pair["player_q"] = pair["player_q"].astype("int64")
    return pair
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_joi.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add chemistry/joi.py tests/test_joi.py
git commit -m "JOI: enumerate same-team consecutive-action interactions"
```

---

### Task 6: JOI per match + JOI90 windowed aggregation

**Files:**
- Modify: `chemistry/joi.py`
- Modify: `tests/test_joi.py`

- [ ] **Step 1: Write failing tests in `tests/test_joi.py`**

Append:
```python
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
        # 100->101 pair
        {"game_id": 1, "team_id": 10, "player_p": 100, "player_q": 101, "vaep_pair": 0.10},
        # 101->100 pair, same underlying duo
        {"game_id": 1, "team_id": 10, "player_p": 101, "player_q": 100, "vaep_pair": 0.05},
        # different game, same duo
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
    # total joi 0.45, total mins 135 → joi90 = 0.45 * 90/135 = 0.30
    row = out[(out["player_a"] == 100) & (out["player_b"] == 101)].iloc[0]
    assert row["joi90"] == pytest.approx(0.30)
    assert row["minutes"] == pytest.approx(135.0)
    assert row["matches"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_joi.py -v -k joi_per_match or window`
Expected: ImportError / not-defined.

- [ ] **Step 3: Add `joi_per_match`, `joi90_window`, and `MinutesProvider` to `chemistry/joi.py`**

Append:
```python
from typing import Protocol


class MinutesProvider(Protocol):
    def minutes(self, game_id: int, player_p: int, player_q: int) -> float: ...


def _canonical_pair(p: pd.Series, q: pd.Series) -> tuple[pd.Series, pd.Series]:
    a = pd.concat([p, q], axis=1).min(axis=1)
    b = pd.concat([p, q], axis=1).max(axis=1)
    return a, b


def joi_per_match(interactions: pd.DataFrame) -> pd.DataFrame:
    """Sum vaep_pair per (game, unordered pair)."""
    a, b = _canonical_pair(interactions["player_p"], interactions["player_q"])
    df = interactions.assign(player_a=a.astype("int64"), player_b=b.astype("int64"))
    grouped = (
        df.groupby(["game_id", "team_id", "player_a", "player_b"], as_index=False)
          ["vaep_pair"].sum()
          .rename(columns={"vaep_pair": "joi"})
    )
    return grouped


def joi90_window(per_match: pd.DataFrame, minutes_provider: MinutesProvider) -> pd.DataFrame:
    """Aggregate per-match JOI to per-pair, per 90 minutes of shared play."""
    per_match = per_match.copy()
    per_match["minutes"] = per_match.apply(
        lambda r: minutes_provider.minutes(int(r["game_id"]), int(r["player_a"]), int(r["player_b"])),
        axis=1,
    )
    agg = (
        per_match.groupby(["team_id", "player_a", "player_b"], as_index=False)
                 .agg(joi=("joi", "sum"), minutes=("minutes", "sum"), matches=("game_id", "nunique"))
    )
    # JOI90 — only meaningful if minutes > 0
    agg["joi90"] = (agg["joi"] * 90.0 / agg["minutes"]).where(agg["minutes"] > 0, 0.0)
    return agg
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_joi.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add chemistry/joi.py tests/test_joi.py
git commit -m "JOI: per-match aggregation + JOI90 windowed normalisation"
```

---

### Task 7: Minutes provider (shared minutes from StatsBomb lineups)

**Files:**
- Create: `chemistry/minutes.py`
- Create: `tests/test_minutes.py`

- [ ] **Step 1: Write failing test in `tests/test_minutes.py`**

```python
import pandas as pd
from chemistry.minutes import LineupsMinutes, shared_minutes


def test_two_starters_who_both_play_full_game_share_90_minutes():
    lineups = pd.DataFrame([
        # player, team, start_min, end_min
        {"game_id": 1, "team_id": 10, "player_id": 100, "from_minute": 0, "to_minute": 90},
        {"game_id": 1, "team_id": 10, "player_id": 101, "from_minute": 0, "to_minute": 90},
    ])
    out = shared_minutes(lineups, 1, 100, 101)
    assert out == 90.0


def test_substitute_at_60_shares_30_minutes_with_starter():
    lineups = pd.DataFrame([
        {"game_id": 1, "team_id": 10, "player_id": 100, "from_minute": 0,  "to_minute": 90},
        {"game_id": 1, "team_id": 10, "player_id": 101, "from_minute": 60, "to_minute": 90},
    ])
    out = shared_minutes(lineups, 1, 100, 101)
    assert out == 30.0


def test_players_who_never_overlap_share_zero_minutes():
    lineups = pd.DataFrame([
        {"game_id": 1, "team_id": 10, "player_id": 100, "from_minute": 0,  "to_minute": 60},
        {"game_id": 1, "team_id": 10, "player_id": 101, "from_minute": 60, "to_minute": 90},
    ])
    out = shared_minutes(lineups, 1, 100, 101)
    assert out == 0.0


def test_lineups_minutes_provider_uses_dataframe():
    lineups = pd.DataFrame([
        {"game_id": 1, "team_id": 10, "player_id": 100, "from_minute": 0, "to_minute": 90},
        {"game_id": 1, "team_id": 10, "player_id": 101, "from_minute": 0, "to_minute": 90},
    ])
    provider = LineupsMinutes(lineups)
    assert provider.minutes(1, 100, 101) == 90.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_minutes.py -v`
Expected: ImportError.

- [ ] **Step 3: Create `chemistry/minutes.py`**

```python
"""Compute shared on-pitch minutes between players for a match."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def shared_minutes(lineups: pd.DataFrame, game_id: int, player_p: int, player_q: int) -> float:
    rows = lineups[(lineups["game_id"] == game_id)
                   & (lineups["player_id"].isin([player_p, player_q]))]
    if len(rows) < 2:
        return 0.0
    a = rows[rows["player_id"] == player_p].iloc[0]
    b = rows[rows["player_id"] == player_q].iloc[0]
    if a["team_id"] != b["team_id"]:
        return 0.0
    start = max(a["from_minute"], b["from_minute"])
    end = min(a["to_minute"], b["to_minute"])
    return float(max(0.0, end - start))


@dataclass
class LineupsMinutes:
    """MinutesProvider implementation backed by a lineups DataFrame."""
    lineups: pd.DataFrame

    def minutes(self, game_id: int, player_p: int, player_q: int) -> float:
        return shared_minutes(self.lineups, game_id, player_p, player_q)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_minutes.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Add a lineups loader to `chemistry/pipeline.py`**

Append:
```python
from statsbombpy import sb as _sb  # noqa: E402


def load_lineups_for_match(match_id: int) -> pd.DataFrame:
    """Return rows of (game_id, team_id, player_id, from_minute, to_minute).

    Sub events come from sb.events(); starters from sb.lineups().
    """
    lineups = _sb.lineups(match_id=match_id)
    # lineups is a dict of {team_name: DataFrame}; flatten
    rows = []
    for team_name, df in lineups.items():
        for _, row in df.iterrows():
            rows.append({
                "game_id": match_id,
                "team_id": row.get("team_id"),
                "team_name": team_name,
                "player_id": row["player_id"],
                "player_name": row["player_name"],
                "position": row.get("positions", [{}])[0].get("position") if row.get("positions") else None,
                "from_minute": row.get("from_minute", 0),
                "to_minute": row.get("to_minute", 90),
            })
    return pd.DataFrame(rows)
```

Note for executor: `statsbombpy` returns lineups with `positions` as a list of dicts that include `from_minute` / `to_minute` for each position spell. If `from_minute` / `to_minute` aren't directly on the row, extract them from the first/last position spell. Adapt as needed.

- [ ] **Step 6: Commit**

```bash
git add chemistry/minutes.py tests/test_minutes.py chemistry/pipeline.py
git commit -m "Minutes: shared on-pitch minutes from StatsBomb lineups"
```

---

## Phase 3 — Squads

### Task 8: Squad YAML loader

**Files:**
- Create: `chemistry/squads.py`
- Create: `tests/test_squads.py`
- Create: `tests/fixtures/squads/TST.yaml`

- [ ] **Step 1: Create test fixture `tests/fixtures/squads/TST.yaml`**

```yaml
nation: Testland
nation_code: TST
flag_iso: tt
manager: Jane Doe
formation: "4-3-3"
team_color: "#112233"
players:
  - name: Alpha One
    club: FC Alpha
    position: GK
  - name: Beta Two
    club: FC Beta
    position: CB
  - name: Gamma Three
    club: FC Gamma
    position: ST
    headshot: gamma.jpg
```

- [ ] **Step 2: Write failing test in `tests/test_squads.py`**

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_squads.py -v`
Expected: ImportError.

- [ ] **Step 4: Create `chemistry/squads.py`**

```python
"""WC 2026 squad definitions loaded from YAML."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class Player(BaseModel):
    name: str
    club: str
    position: str
    headshot: Optional[str] = None


class Squad(BaseModel):
    nation: str
    nation_code: str = Field(min_length=2, max_length=3)
    flag_iso: str
    manager: str
    formation: str
    team_color: str
    players: list[Player]


def load_squad(path: Path) -> Squad:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Squad(**data)


def load_all_squads(dir: Path) -> dict[str, Squad]:
    out: dict[str, Squad] = {}
    for p in sorted(Path(dir).glob("*.yaml")):
        squad = load_squad(p)
        out[squad.nation_code] = squad
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_squads.py -v`
Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add chemistry/squads.py tests/test_squads.py tests/fixtures/squads/TST.yaml
git commit -m "Squads: YAML loader with pydantic validation"
```

---

### Task 9: Pair filtering by squad

**Files:**
- Modify: `chemistry/squads.py`
- Modify: `tests/test_squads.py`

- [ ] **Step 1: Write failing test in `tests/test_squads.py`**

Append:
```python
import pandas as pd
from chemistry.squads import filter_pairs_by_squad


def test_filter_pairs_keeps_only_pairs_where_both_players_are_in_squad():
    pairs = pd.DataFrame([
        # in-squad pair
        {"player_a": 100, "player_b": 101, "joi90": 0.5, "minutes": 90, "matches": 1},
        # mixed pair (101 in, 999 out)
        {"player_a": 101, "player_b": 999, "joi90": 0.3, "minutes": 90, "matches": 1},
        # both out
        {"player_a": 800, "player_b": 999, "joi90": 0.4, "minutes": 90, "matches": 1},
    ])
    roster = {100: "Alpha", 101: "Beta", 102: "Gamma"}  # statsbomb_player_id -> display name
    out = filter_pairs_by_squad(pairs, set(roster.keys()))
    assert len(out) == 1
    assert int(out.iloc[0]["player_a"]) == 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_squads.py::test_filter_pairs_keeps_only_pairs_where_both_players_are_in_squad -v`
Expected: ImportError for `filter_pairs_by_squad`.

- [ ] **Step 3: Add `filter_pairs_by_squad` to `chemistry/squads.py`**

Append:
```python
import pandas as pd  # noqa: E402


def filter_pairs_by_squad(pairs: pd.DataFrame, player_ids: set[int]) -> pd.DataFrame:
    in_squad = pairs["player_a"].isin(player_ids) & pairs["player_b"].isin(player_ids)
    return pairs[in_squad].reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_squads.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add chemistry/squads.py tests/test_squads.py
git commit -m "Squads: filter chemistry pairs to in-squad pairs only"
```

---

### Task 10: Player-name → StatsBomb ID resolution

**Files:**
- Create: `chemistry/players.py`
- Create: `tests/test_players.py`

The squad YAMLs use human names ("Christian Pulisic"), but chemistry pairs use StatsBomb integer IDs. We need a bridge.

- [ ] **Step 1: Write failing test in `tests/test_players.py`**

```python
import pandas as pd
from chemistry.players import build_player_index, resolve_squad_ids


def test_build_player_index_returns_name_to_id_map():
    lineups = pd.DataFrame([
        {"game_id": 1, "player_id": 100, "player_name": "Christian Pulisic", "team_name": "United States"},
        {"game_id": 1, "player_id": 101, "player_name": "Tyler Adams",       "team_name": "United States"},
        {"game_id": 2, "player_id": 100, "player_name": "Christian Pulisic", "team_name": "United States"},
    ])
    index = build_player_index(lineups)
    assert index["United States"]["christian pulisic"] == 100
    assert index["United States"]["tyler adams"] == 101


def test_resolve_squad_ids_returns_matched_and_unmatched():
    lineups = pd.DataFrame([
        {"game_id": 1, "player_id": 100, "player_name": "Christian Pulisic", "team_name": "United States"},
        {"game_id": 1, "player_id": 101, "player_name": "Tyler Adams",       "team_name": "United States"},
    ])
    names = ["Christian Pulisic", "Tyler Adams", "Phantom Player"]
    matched, unmatched = resolve_squad_ids("United States", names, lineups)
    assert matched == {"Christian Pulisic": 100, "Tyler Adams": 101}
    assert unmatched == ["Phantom Player"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_players.py -v`
Expected: ImportError.

- [ ] **Step 3: Create `chemistry/players.py`**

```python
"""Map roster names to StatsBomb player_id values."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

import unicodedata


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return s.lower().strip()


def build_player_index(lineups: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Return {team_name: {normalized_name: player_id}}."""
    out: dict[str, dict[str, int]] = {}
    for (team, name), grp in lineups.groupby(["team_name", "player_name"]):
        # pick the most common id for this (team, name); StatsBomb uses a single id per player
        pid = int(grp["player_id"].mode().iloc[0])
        out.setdefault(team, {})[_norm(name)] = pid
    return out


def resolve_squad_ids(
    team_name: str,
    roster_names: Iterable[str],
    lineups: pd.DataFrame,
) -> tuple[dict[str, int], list[str]]:
    """Match roster names to StatsBomb player_ids; return (matched, unmatched)."""
    index = build_player_index(lineups).get(team_name, {})
    matched: dict[str, int] = {}
    unmatched: list[str] = []
    for name in roster_names:
        pid = index.get(_norm(name))
        if pid is None:
            unmatched.append(name)
        else:
            matched[name] = pid
    return matched, unmatched
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_players.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add chemistry/players.py tests/test_players.py
git commit -m "Players: resolve roster names to StatsBomb IDs (accent-insensitive)"
```

---

### Task 11: Bootstrap squad YAMLs

**Files:**
- Create: `scripts/bootstrap_squads.py`
- Create: `squads/wc2026/USA.yaml` (manually authored as proof, then replicated by script)

This task seeds initial squad YAML files. For 32 nations we ship a starter file each that subsequent edits refine.

- [ ] **Step 1: Create `squads/wc2026/USA.yaml`** (hand-authored proof)

```yaml
nation: United States
nation_code: USA
flag_iso: us
manager: Mauricio Pochettino
formation: "4-3-3"
team_color: "#bf0d3e"
players:
  - { name: Matt Turner,             club: Crystal Palace,  position: GK }
  - { name: Sergiño Dest,            club: PSV Eindhoven,   position: RB }
  - { name: Chris Richards,          club: Crystal Palace,  position: CB }
  - { name: Tim Ream,                club: Charlotte FC,    position: CB }
  - { name: Antonee Robinson,        club: Fulham,          position: LB }
  - { name: Tyler Adams,             club: Bournemouth,     position: DM }
  - { name: Weston McKennie,         club: Juventus,        position: CM }
  - { name: Yunus Musah,             club: AC Milan,        position: CM }
  - { name: Christian Pulisic,       club: AC Milan,        position: RW }
  - { name: Folarin Balogun,         club: AS Monaco,       position: ST }
  - { name: Tim Weah,                club: Juventus,        position: LW }
```

- [ ] **Step 2: Create `scripts/bootstrap_squads.py`**

```python
"""Bootstrap squad YAML files for all 32 WC 2026 nations.

Writes one stub per qualifier. The roster is empty by default and must be
edited manually with the manager's likely 26-man squad before the build runs.
Run this once when starting; subsequent edits live in git.
"""
from __future__ import annotations

from pathlib import Path
import yaml

OUT_DIR = Path("squads/wc2026")

# (code, name, flag_iso, default_color) for the 32 qualifiers.
# Codes follow FIFA. Update default_color manually as needed.
QUALIFIERS: list[tuple[str, str, str, str]] = [
    ("USA", "United States",     "us", "#bf0d3e"),
    ("CAN", "Canada",             "ca", "#d52b1e"),
    ("MEX", "Mexico",             "mx", "#006847"),
    ("ARG", "Argentina",          "ar", "#75aadb"),
    ("BRA", "Brazil",             "br", "#fedf00"),
    ("URU", "Uruguay",            "uy", "#5db8de"),
    ("ECU", "Ecuador",            "ec", "#ffce00"),
    ("COL", "Colombia",           "co", "#fcd116"),
    ("PAR", "Paraguay",           "py", "#d52b1e"),
    ("FRA", "France",             "fr", "#0055a4"),
    ("ESP", "Spain",              "es", "#aa151b"),
    ("GER", "Germany",            "de", "#000000"),
    ("ENG", "England",            "gb-eng", "#fff"),
    ("POR", "Portugal",           "pt", "#006600"),
    ("NED", "Netherlands",        "nl", "#ff6600"),
    ("BEL", "Belgium",            "be", "#ed2939"),
    ("CRO", "Croatia",            "hr", "#171796"),
    ("ITA", "Italy",              "it", "#1c5ed6"),
    ("SUI", "Switzerland",        "ch", "#d52b1e"),
    ("DEN", "Denmark",            "dk", "#c8102e"),
    ("AUT", "Austria",            "at", "#ed2939"),
    ("MAR", "Morocco",            "ma", "#c1272d"),
    ("SEN", "Senegal",            "sn", "#00853f"),
    ("EGY", "Egypt",              "eg", "#ce1126"),
    ("TUN", "Tunisia",            "tn", "#e70013"),
    ("NGA", "Nigeria",            "ng", "#008751"),
    ("CIV", "Côte d'Ivoire",      "ci", "#ff8200"),
    ("JPN", "Japan",              "jp", "#bd0029"),
    ("KOR", "South Korea",        "kr", "#cd2e3a"),
    ("IRN", "Iran",               "ir", "#239f40"),
    ("AUS", "Australia",          "au", "#ffcd00"),
    ("KSA", "Saudi Arabia",       "sa", "#006c35"),
]


def write_stub(code: str, name: str, flag_iso: str, color: str) -> Path:
    target = OUT_DIR / f"{code}.yaml"
    if target.exists():
        return target
    data = {
        "nation": name,
        "nation_code": code,
        "flag_iso": flag_iso,
        "manager": "TBD",
        "formation": "4-3-3",
        "team_color": color,
        "players": [],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return target


def main() -> None:
    written = []
    for code, name, flag_iso, color in QUALIFIERS:
        p = write_stub(code, name, flag_iso, color)
        written.append(p)
    print(f"Stubs in {OUT_DIR}: {len(written)}")


if __name__ == "__main__":
    main()
```

Note: `manager: "TBD"` is an explicit empty placeholder for human editing — it's data, not a code placeholder.

- [ ] **Step 3: Run the bootstrap**

Run:
```bash
python -m scripts.bootstrap_squads
```
Expected: prints `Stubs in squads/wc2026: 32`. The pre-existing `USA.yaml` is preserved (script skips existing files).

- [ ] **Step 4: Commit the stub squad set**

```bash
git add squads/wc2026/ scripts/bootstrap_squads.py
git commit -m "Squads: stub 31 squad YAMLs + curated USA seed"
```

- [ ] **Step 5: Document the manual editing workflow in README**

Append to README.md:
```markdown
## Squad editing

`squads/wc2026/<CODE>.yaml` files are hand-curated. Edit the `players:` list as call-ups are announced. Re-run `python -m scripts.build` to refresh chemistry. The `manager` field reads `TBD` by default — replace once known.

To add a fresh stub: extend the `QUALIFIERS` list in `scripts/bootstrap_squads.py` and rerun.
```

- [ ] **Step 6: Commit README update**

```bash
git add README.md
git commit -m "Docs: squad editing workflow"
```

---

## Phase 4 — Build orchestration + rankings + export

### Task 12: Build orchestrator

**Files:**
- Create: `scripts/build.py`

- [ ] **Step 1: Create `scripts/build.py`**

```python
"""End-to-end build:
   ingest → SPADL → VAEP → interactions → JOI per match → JOI90 per pair.
Writes intermediate parquets and the final pair table.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from chemistry import ingest, pipeline, joi, minutes
from chemistry.vaep_model import load_vaep_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build")

DATA = Path("data")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)


def run(use_heuristic_vaep: bool) -> None:
    # 1. Fetch all target competitions
    log.info("Fetching StatsBomb open data for target competitions")
    ingest.fetch_all_targets()

    # 2. SPADL
    log.info("Converting events to SPADL")
    for comp_dir in sorted((DATA / "raw").iterdir()):
        if comp_dir.is_dir():
            pipeline.convert_competition(comp_dir)

    # 3. VAEP
    log.info("Scoring VAEP")
    kind = "heuristic" if use_heuristic_vaep else "trained"
    model = load_vaep_model(kind=kind)
    for spadl_comp in sorted((DATA / "spadl").iterdir()):
        if spadl_comp.is_dir():
            pipeline.score_competition(spadl_comp, model)

    # 4. Per-match interactions, JOI, lineups
    all_per_match: list[pd.DataFrame] = []
    all_lineups: list[pd.DataFrame] = []
    for vaep_comp in sorted((DATA / "vaep").iterdir()):
        if not vaep_comp.is_dir():
            continue
        for path in sorted(vaep_comp.glob("*.parquet")):
            match_id = int(path.stem)
            spadl = pd.read_parquet(path)
            interactions = joi.enumerate_interactions(spadl)
            if len(interactions) == 0:
                continue
            per_match = joi.joi_per_match(interactions)
            all_per_match.append(per_match)
            try:
                all_lineups.append(pipeline.load_lineups_for_match(match_id))
            except Exception as exc:
                log.warning("Lineups load failed for %s: %s", match_id, exc)

    per_match_df = pd.concat(all_per_match, ignore_index=True) if all_per_match else pd.DataFrame()
    lineups_df   = pd.concat(all_lineups,   ignore_index=True) if all_lineups   else pd.DataFrame()
    per_match_df.to_parquet(OUT / "joi_per_match.parquet", index=False)
    lineups_df.to_parquet(OUT / "lineups.parquet", index=False)

    # 5. JOI90 window
    mins_provider = minutes.LineupsMinutes(lineups_df)
    joi90 = joi.joi90_window(per_match_df, mins_provider)
    joi90.to_parquet(OUT / "joi90.parquet", index=False)
    log.info("Wrote %d pairs to outputs/joi90.parquet", len(joi90))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--heuristic-vaep", action="store_true",
                   help="Use the heuristic VAEP estimator (no trained model needed).")
    args = p.parse_args()
    run(use_heuristic_vaep=args.heuristic_vaep)
```

- [ ] **Step 2: Smoke-run with heuristic VAEP**

Run:
```bash
python -m scripts.build --heuristic-vaep
```
Expected: takes several minutes; logs progress; writes `outputs/joi_per_match.parquet`, `outputs/lineups.parquet`, `outputs/joi90.parquet`. Each has > 0 rows.

- [ ] **Step 3: Quick sanity check on the output**

Run:
```bash
python -c "
import pandas as pd
df = pd.read_parquet('outputs/joi90.parquet')
print('rows:', len(df))
print(df.sort_values('joi90', ascending=False).head(10))
"
```
Expected: top-10 pairs printed; values are finite; teams are reasonable (top values likely from WC 2022 finalists).

- [ ] **Step 4: Commit**

```bash
git add scripts/build.py
git commit -m "Build: end-to-end orchestrator (ingest -> SPADL -> VAEP -> JOI90)"
```

---

### Task 13: Per-team and global rankings

**Files:**
- Create: `chemistry/rank.py`
- Create: `tests/test_rank.py`

- [ ] **Step 1: Write failing test in `tests/test_rank.py`**

```python
import pandas as pd
from chemistry.rank import per_team, global_top_n


def test_per_team_returns_pairs_per_nation_sorted_desc():
    pairs = pd.DataFrame([
        {"nation_code": "USA", "player_a": 1, "player_b": 2, "joi90": 0.2, "minutes": 100, "matches": 2},
        {"nation_code": "USA", "player_a": 1, "player_b": 3, "joi90": 0.5, "minutes": 100, "matches": 2},
        {"nation_code": "FRA", "player_a": 4, "player_b": 5, "joi90": 0.7, "minutes": 100, "matches": 2},
    ])
    out = per_team(pairs)
    assert set(out.keys()) == {"USA", "FRA"}
    usa = out["USA"]
    assert list(usa["joi90"]) == [0.5, 0.2]


def test_global_top_n_caps_to_n_rows():
    pairs = pd.DataFrame([
        {"nation_code": "USA", "player_a": i, "player_b": i+100, "joi90": 0.1 * i, "minutes": 100, "matches": 2}
        for i in range(1, 11)
    ])
    out = global_top_n(pairs, n=3)
    assert len(out) == 3
    assert out["joi90"].is_monotonic_decreasing


def test_global_top_n_filters_to_minimum_minutes():
    pairs = pd.DataFrame([
        {"nation_code": "USA", "player_a": 1, "player_b": 2, "joi90": 1.0, "minutes": 30, "matches": 1},
        {"nation_code": "FRA", "player_a": 3, "player_b": 4, "joi90": 0.5, "minutes": 180, "matches": 2},
    ])
    out = global_top_n(pairs, n=5, min_minutes=90)
    assert len(out) == 1
    assert out.iloc[0]["nation_code"] == "FRA"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rank.py -v`
Expected: ImportError.

- [ ] **Step 3: Create `chemistry/rank.py`**

```python
"""Ranking outputs — per-team chemistry tables and a global leaderboard."""
from __future__ import annotations

import pandas as pd


def per_team(pairs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """{nation_code: pairs sorted desc by joi90}."""
    out: dict[str, pd.DataFrame] = {}
    for code, grp in pairs.groupby("nation_code"):
        out[code] = grp.sort_values("joi90", ascending=False).reset_index(drop=True)
    return out


def global_top_n(pairs: pd.DataFrame, n: int = 50, min_minutes: float = 0.0) -> pd.DataFrame:
    filtered = pairs[pairs["minutes"] >= min_minutes].copy()
    return filtered.sort_values("joi90", ascending=False).head(n).reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rank.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add chemistry/rank.py tests/test_rank.py
git commit -m "Rank: per-team and global top-N with minimum-minutes filter"
```

---

### Task 14: Stitch squads + pairs into chemistry JSON

**Files:**
- Create: `chemistry/export.py`
- Modify: `scripts/build.py`

- [ ] **Step 1: Create `chemistry/export.py`**

```python
"""Stitch squads + JOI90 pair table into outputs/chemistry.json."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from chemistry import rank, squads, players


def _pair_record(row: pd.Series, id_to_name: dict[int, str]) -> dict:
    return {
        "player_a_id": int(row["player_a"]),
        "player_b_id": int(row["player_b"]),
        "player_a_name": id_to_name.get(int(row["player_a"]), str(int(row["player_a"]))),
        "player_b_name": id_to_name.get(int(row["player_b"]), str(int(row["player_b"]))),
        "joi90": float(row["joi90"]),
        "minutes": float(row["minutes"]),
        "matches": int(row["matches"]),
    }


def build_chemistry_json(
    joi90: pd.DataFrame,
    lineups: pd.DataFrame,
    squads_map: dict[str, squads.Squad],
    out_path: Path,
    min_minutes_global: float = 90.0,
    min_minutes_team: float = 90.0,
) -> Path:
    # Build a global player_id -> name dict from lineups
    id_to_name = (
        lineups.dropna(subset=["player_id"])
               .drop_duplicates("player_id")
               .set_index("player_id")["player_name"]
               .to_dict()
    )
    id_to_name = {int(k): v for k, v in id_to_name.items()}

    # Build per-nation pair tables
    per_nation: dict[str, dict] = {}
    for code, squad in squads_map.items():
        # Find StatsBomb team name match — we assume squad.nation matches team_name
        team_lineups = lineups[lineups["team_name"] == squad.nation]
        if team_lineups.empty:
            per_nation[code] = {
                "squad": squad.model_dump(),
                "pairs": [],
                "coverage": {"matches": 0, "warning": "no matches in open data"},
            }
            continue

        roster_names = [p.name for p in squad.players]
        matched, unmatched = players.resolve_squad_ids(squad.nation, roster_names, team_lineups)

        roster_ids = set(matched.values())
        squad_pairs = joi90[
            joi90["player_a"].isin(roster_ids) & joi90["player_b"].isin(roster_ids)
            & (joi90["minutes"] >= min_minutes_team)
        ].copy()
        squad_pairs = squad_pairs.sort_values("joi90", ascending=False)

        per_nation[code] = {
            "squad": squad.model_dump(),
            "pairs": [_pair_record(r, id_to_name) for _, r in squad_pairs.iterrows()],
            "coverage": {
                "matches": int(team_lineups["game_id"].nunique()),
                "unmatched_roster_names": unmatched,
            },
        }

    # Global leaderboard — limited to in-squad pairs across all nations
    in_any_squad_ids: set[int] = set()
    for entry in per_nation.values():
        for p in entry["pairs"]:
            in_any_squad_ids.add(p["player_a_id"])
            in_any_squad_ids.add(p["player_b_id"])

    leaderboard_src = joi90[
        joi90["player_a"].isin(in_any_squad_ids)
        & joi90["player_b"].isin(in_any_squad_ids)
        & (joi90["minutes"] >= min_minutes_global)
    ].copy()
    leaderboard = rank.global_top_n(
        leaderboard_src.assign(nation_code="GLOBAL"), n=50,
    )

    doc = {
        "nations": per_nation,
        "leaderboard": [_pair_record(r, id_to_name) | {"joi90": float(r["joi90"])}
                        for _, r in leaderboard.iterrows()],
        "meta": {
            "min_minutes_global": min_minutes_global,
            "min_minutes_team": min_minutes_team,
            "total_pairs": int(len(joi90)),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    return out_path
```

- [ ] **Step 2: Hook export into `scripts/build.py`**

Append before `if __name__ == "__main__":`:
```python
def export(out_path: Path = OUT / "chemistry.json") -> Path:
    from chemistry import squads as squads_mod, export as export_mod
    joi90 = pd.read_parquet(OUT / "joi90.parquet")
    lineups = pd.read_parquet(OUT / "lineups.parquet")
    squad_map = squads_mod.load_all_squads(Path("squads/wc2026"))
    return export_mod.build_chemistry_json(joi90, lineups, squad_map, out_path)
```

And modify `run()` to call `export()` at the end:
```python
    # 6. Stitch squads + pairs
    path = export()
    log.info("Wrote %s", path)
```

- [ ] **Step 3: Add a tiny smoke test** in `tests/test_export.py`

```python
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
         "player_id": 100, "player_name": "Alpha", "from_minute": 0, "to_minute": 90},
        {"game_id": 1, "team_id": 1, "team_name": "Testland",
         "player_id": 101, "player_name": "Beta",  "from_minute": 0, "to_minute": 90},
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
```

- [ ] **Step 4: Run tests + build**

Run: `pytest tests/test_export.py -v && python -m scripts.build --heuristic-vaep`
Expected: tests PASS; build emits `outputs/chemistry.json` containing 32 nations.

- [ ] **Step 5: Commit**

```bash
git add chemistry/export.py scripts/build.py tests/test_export.py
git commit -m "Export: stitch squads + JOI90 into outputs/chemistry.json"
```

---

## Phase 5 — Asset export (the video deliverable)

### Task 15: Render pitch chemistry PNG/SVG per nation

**Files:**
- Create: `chemistry/render.py`
- Create: `scripts/export.py` (rendering script)
- Create: `tests/test_render.py`

- [ ] **Step 1: Create `chemistry/render.py`** — single source of truth for the pitch chemistry figure

```python
"""Render Figure-1-style pitch chemistry graphs from chemistry.json."""
from __future__ import annotations

import json
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch
from mplsoccer import Pitch

import cairosvg


# Default lineup positions on a 105x68 pitch by position string.
# Used when we don't have empirical positions from match data.
DEFAULT_POSITIONS: dict[str, tuple[float, float]] = {
    "GK":  (8,  34),
    "RB":  (25, 58),
    "RWB": (40, 60),
    "RCB": (22, 44),
    "CB":  (22, 34),
    "LCB": (22, 24),
    "LB":  (25, 10),
    "LWB": (40, 8),
    "RDM": (35, 42), "DM":  (35, 34), "LDM": (35, 26),
    "RCM": (55, 42), "CM":  (55, 34), "LCM": (55, 26),
    "RM":  (60, 56), "LM":  (60, 12),
    "RAM": (70, 42), "AM":  (70, 34), "LAM": (70, 26),
    "RW":  (85, 56),
    "LW":  (85, 12),
    "ST":  (90, 34), "CF": (90, 34),
    "SS":  (82, 34),
}


@dataclass
class PitchOptions:
    width: int = 1920
    height: int = 1080
    background: str = "transparent"  # or hex color
    accent: str = "#4ade80"           # high-chemistry color
    cold: str = "#ef4444"             # low-chemistry color
    edge_min_width: float = 1.5
    edge_max_width: float = 6.0
    show_labels: bool = True


def _color_for_joi(joi90: float, vmin: float, vmax: float, opts: PitchOptions) -> str:
    if vmax == vmin:
        return opts.accent
    t = (joi90 - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    # Simple two-stop gradient cold→accent through dark olive
    cold = tuple(int(opts.cold[i:i+2], 16) for i in (1, 3, 5))
    hot  = tuple(int(opts.accent[i:i+2], 16) for i in (1, 3, 5))
    rgb = tuple(int(c + (h - c) * t) for c, h in zip(cold, hot))
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def render_pitch_chemistry(
    nation_entry: dict,
    opts: Optional[PitchOptions] = None,
) -> plt.Figure:
    opts = opts or PitchOptions()
    pitch = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68,
                  line_color="#ffffff", pitch_color="none", linewidth=1.2)
    fig, ax = pitch.draw(figsize=(opts.width / 100, opts.height / 100), constrained_layout=False)
    if opts.background != "transparent":
        fig.patch.set_facecolor(opts.background)
    else:
        fig.patch.set_alpha(0)

    squad = nation_entry["squad"]
    pairs = nation_entry["pairs"]

    # Build position lookup for the players in the squad
    name_to_pos: dict[str, tuple[float, float]] = {}
    for p in squad["players"]:
        pos = DEFAULT_POSITIONS.get(p["position"], (50, 34))
        name_to_pos[p["name"]] = pos

    # Draw edges
    if pairs:
        vmin = min(p["joi90"] for p in pairs)
        vmax = max(p["joi90"] for p in pairs)
    else:
        vmin = vmax = 0.0

    for p in pairs:
        a, b = p["player_a_name"], p["player_b_name"]
        pa = name_to_pos.get(a)
        pb = name_to_pos.get(b)
        if not pa or not pb:
            continue
        col = _color_for_joi(p["joi90"], vmin, vmax, opts)
        # width scales with minutes (confidence)
        m = min(p["minutes"], 600) / 600.0
        lw = opts.edge_min_width + (opts.edge_max_width - opts.edge_min_width) * m
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=col,
                linewidth=lw, solid_capstyle="round", alpha=0.85, zorder=1)

    # Draw player markers + labels
    for name, (x, y) in name_to_pos.items():
        ax.add_patch(Circle((x, y), 1.6, color="#ffffff", zorder=3))
        ax.add_patch(Circle((x, y), 1.3, color=squad["team_color"], zorder=4))
        if opts.show_labels:
            ax.text(x, y + 3.0, name, color="#ffffff", ha="center", va="bottom",
                    fontsize=10, weight="bold", zorder=5,
                    path_effects=[])

    ax.set_xlim(0, 105); ax.set_ylim(0, 68); ax.set_axis_off()
    return fig


def fig_to_png(fig: plt.Figure, path: Path, *, transparent: bool = True, dpi: int = 200) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, transparent=transparent, dpi=dpi,
                bbox_inches="tight", pad_inches=0)
    return path


def fig_to_svg(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0,
                transparent=True)
    return path


def svg_to_4k_png(svg_path: Path, png_path: Path, width: int = 3840) -> Path:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path),
                     output_width=width)
    return png_path
```

- [ ] **Step 2: Tiny smoke test** in `tests/test_render.py`

```python
import json
import pandas as pd
from pathlib import Path
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
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_render.py -v`
Expected: PASS. (If `mplsoccer` raises about `pitch_type='custom'`, swap to `pitch_type='custom'` parameters per the [mplsoccer docs](https://mplsoccer.readthedocs.io/en/latest/gallery/pitch_setup/plot_pitches.html); the goal is a 105×68 pitch.)

- [ ] **Step 4: Create `scripts/export.py`** to render every nation

```python
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

    # Transparent SVG (vector master)
    svg_path = nat_dir / "pitch.svg"
    fig = render_pitch_chemistry(entry, PitchOptions(background="transparent"))
    fig_to_svg(fig, svg_path)
    plt.close(fig)

    # 4K transparent PNG (rendered from SVG for crisp lines)
    svg_to_4k_png(svg_path, nat_dir / "pitch.png", width=3840)

    # Branded PNG with team-color background
    fig2 = render_pitch_chemistry(entry, PitchOptions(background=entry["squad"]["team_color"]))
    fig_to_png(fig2, nat_dir / "pitch_branded.png", transparent=False, dpi=200)
    plt.close(fig2)

    # Per-nation data
    (nat_dir / "data.json").write_text(json.dumps(entry, indent=2, ensure_ascii=False))

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

    # Top-level manifest
    manifest = {
        "generated_from": str(OUT_JSON),
        "nations": list(doc["nations"].keys()),
        "assets_per_nation": ["pitch.svg", "pitch.png", "pitch_branded.png", "data.json"],
    }
    (EXPORTS / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Wrote %d nations to %s", len(doc["nations"]), EXPORTS)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run export end-to-end**

Run: `python -m scripts.export`
Expected: logs ~32 lines; `exports/nations/USA/pitch.svg`, `pitch.png`, `pitch_branded.png`, `data.json` exist; `exports/manifest.json` exists.

- [ ] **Step 6: Visually inspect one export**

Run: `open exports/nations/USA/pitch_branded.png`
Expected: a clean pitch with USA players placed by position, chemistry edges colored by JOI90.

- [ ] **Step 7: Commit**

```bash
git add chemistry/render.py scripts/export.py tests/test_render.py
git commit -m "Render: per-nation pitch chemistry SVG + 4K PNG + branded variant"
```

---

### Task 16: Lower-third pair cards

**Files:**
- Create: `chemistry/pair_card.py`
- Modify: `scripts/export.py`
- Create: `tests/test_pair_card.py`

- [ ] **Step 1: Create `chemistry/pair_card.py`**

```python
"""1920x320 lower-third PNG card for a single chemistry pair."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


@dataclass
class CardOptions:
    width_px: int = 1920
    height_px: int = 320
    bg: str = "#0e1117"
    accent: str = "#bf0d3e"  # team color
    text: str = "#e6edf3"
    muted: str = "#8b949e"


def render_pair_card(
    pair: dict,
    *,
    nation: str,
    flag_iso: str,
    opts: Optional[CardOptions] = None,
    out_path: Optional[Path] = None,
) -> Path:
    opts = opts or CardOptions()
    fig = plt.figure(figsize=(opts.width_px / 100, opts.height_px / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, opts.width_px); ax.set_ylim(0, opts.height_px); ax.set_axis_off()
    fig.patch.set_alpha(0)

    # Accent bar
    ax.add_patch(Rectangle((0, 0), 12, opts.height_px, color=opts.accent, lw=0))
    # Main panel
    ax.add_patch(Rectangle((12, 0), opts.width_px - 12, opts.height_px,
                           color=opts.bg, alpha=0.92, lw=0))

    # Nation chip
    ax.text(60, opts.height_px - 40, nation.upper(),
            color=opts.muted, fontsize=18, weight="bold", family="monospace",
            ha="left", va="top")

    # Pair names
    label = f"{pair['player_a_name']}  +  {pair['player_b_name']}"
    ax.text(60, opts.height_px - 90, label,
            color=opts.text, fontsize=48, weight="bold",
            ha="left", va="top")

    # JOI90 + minutes
    ax.text(60, 80,
            f"JOI90 {pair['joi90']:.3f}    {int(pair['minutes'])} mins shared    {pair['matches']} matches",
            color=opts.muted, fontsize=22, ha="left", va="top", family="monospace")

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, transparent=True, dpi=100, bbox_inches=None, pad_inches=0)
    return out_path
```

- [ ] **Step 2: Write test in `tests/test_pair_card.py`**

```python
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
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_pair_card.py -v`
Expected: PASS.

- [ ] **Step 4: Wire into `scripts/export.py`** — render top-5 cards per nation

Inside `export_nation`, after the per-nation data is written, add:
```python
    from chemistry.pair_card import render_pair_card, CardOptions
    cards_dir = nat_dir / "pair_cards"
    cards_dir.mkdir(exist_ok=True)
    opts = CardOptions(accent=entry["squad"]["team_color"])
    for i, p in enumerate(entry["pairs"][:5], start=1):
        out = cards_dir / f"{i:02d}_{p['player_a_name'].replace(' ', '_')}_{p['player_b_name'].replace(' ', '_')}.png"
        render_pair_card(p, nation=entry["squad"]["nation"],
                         flag_iso=entry["squad"]["flag_iso"], opts=opts, out_path=out)
```

- [ ] **Step 5: Re-run export and inspect**

Run: `python -m scripts.export && open exports/nations/USA/pair_cards/`
Expected: up to 5 PNG cards per nation.

- [ ] **Step 6: Commit**

```bash
git add chemistry/pair_card.py tests/test_pair_card.py scripts/export.py
git commit -m "Pair cards: 1920x320 lower-third PNG per top-5 pairs per nation"
```

---

### Task 17: Leaderboard graphic + landing grid graphic

**Files:**
- Modify: `chemistry/render.py`
- Modify: `scripts/export.py`

- [ ] **Step 1: Add `render_leaderboard` to `chemistry/render.py`**

Append:
```python
def render_leaderboard(doc: dict, out_path: Path,
                       title: str = "World Cup 2026 — Top Chemistry Pairs") -> Path:
    items = doc["leaderboard"][:20]
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.set_axis_off()

    ax.text(0.0, 1.0, title, color="#e6edf3", fontsize=36, weight="bold",
            transform=ax.transAxes, va="top")

    row_height = 0.04
    for i, p in enumerate(items):
        y = 0.92 - i * row_height
        ax.text(0.00, y, f"{i+1:>2}.", color="#8b949e", fontsize=16,
                transform=ax.transAxes, family="monospace")
        ax.text(0.05, y, f"{p['player_a_name']}  +  {p['player_b_name']}",
                color="#e6edf3", fontsize=18, transform=ax.transAxes)
        ax.text(0.78, y, f"JOI90  {p['joi90']:.3f}",
                color="#4ade80", fontsize=18, weight="bold",
                transform=ax.transAxes, family="monospace")
        ax.text(0.92, y, f"{int(p['minutes'])} mins",
                color="#8b949e", fontsize=14, transform=ax.transAxes, family="monospace")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(),
                bbox_inches=None, pad_inches=0)
    return out_path


def render_landing_grid(doc: dict, out_path: Path) -> Path:
    nations = list(doc["nations"].items())
    # 4 cols x 8 rows
    fig = plt.figure(figsize=(38.4, 21.6), dpi=100)
    fig.patch.set_facecolor("#0e1117")
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.94])
    ax.set_facecolor("#0e1117"); ax.set_axis_off()
    ax.set_xlim(0, 4); ax.set_ylim(0, 8)
    ax.invert_yaxis()
    ax.text(0, -0.4, "World Cup 2026 — Chemistry",
            color="#e6edf3", fontsize=64, weight="bold", va="top")

    for idx, (code, entry) in enumerate(nations):
        col = idx % 4
        row = idx // 4
        x, y = col + 0.05, row + 0.1
        squad = entry["squad"]
        ax.text(x, y + 0.08, squad["nation"], color="#e6edf3", fontsize=28, weight="bold")
        ax.text(x, y + 0.22, squad["manager"], color="#8b949e", fontsize=16)
        top = entry["pairs"][:1]
        if top:
            p = top[0]
            ax.text(x, y + 0.42, f"{p['player_a_name']} + {p['player_b_name']}",
                    color="#e6edf3", fontsize=18)
            ax.text(x, y + 0.55, f"JOI90 {p['joi90']:.3f}",
                    color="#4ade80", fontsize=18, weight="bold", family="monospace")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(),
                bbox_inches=None, pad_inches=0)
    return out_path
```

- [ ] **Step 2: Update `scripts/export.py` to call both**

Inside `main()`, after the per-nation loop:
```python
    from chemistry.render import render_leaderboard, render_landing_grid
    render_leaderboard(doc, EXPORTS / "leaderboard.png")
    render_landing_grid(doc, EXPORTS / "landing_grid.png")
```

- [ ] **Step 3: Re-run export and inspect**

Run: `python -m scripts.export && open exports/leaderboard.png exports/landing_grid.png`
Expected: two new top-level graphics.

- [ ] **Step 4: Commit**

```bash
git add chemistry/render.py scripts/export.py
git commit -m "Render: leaderboard graphic + 32-nation landing grid"
```

---

## Phase 6 — Site (chart browser)

### Task 18: Site scaffolding + style

**Files:**
- Create: `site/index.html`
- Create: `site/style.css`
- Create: `site/app.js`

- [ ] **Step 1: Create `site/style.css`**

```css
:root {
  --bg:#0e1117; --panel:#161b22; --ink:#e6edf3; --muted:#8b949e;
  --accent:#4ade80; --accent2:#60a5fa; --warn:#fbbf24; --border:#30363d;
}
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--bg); color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
  -webkit-font-smoothing: antialiased; }
main { max-width: 1280px; margin: 0 auto; padding: 48px 32px 96px; }
h1 { font-size: 44px; margin: 0 0 8px; letter-spacing: -0.02em; }
.lead { font-size: 18px; color: var(--muted); margin: 0 0 36px; }
.grid { display: grid; gap: 14px; grid-template-columns: repeat(4, 1fr); }
@media (max-width: 720px) { .grid { grid-template-columns: repeat(2, 1fr); } }
.card { display: block; text-decoration: none; color: inherit;
  background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
  padding: 16px 18px; transition: border-color .15s ease, transform .15s ease; }
.card:hover { border-color: var(--accent2); transform: translateY(-1px); }
.flag { display: inline-block; width: 32px; height: 22px; object-fit: cover;
  border-radius: 3px; vertical-align: middle; margin-right: 10px; }
.card h2 { font-size: 18px; margin: 0 0 4px; }
.card .sub { font-size: 12px; color: var(--muted); margin-bottom: 12px; font-family: "JetBrains Mono", monospace; }
.card .pair { font-size: 14px; color: var(--ink); }
.card .joi { font-family: "JetBrains Mono", monospace; color: var(--accent); font-weight: 600; }
.section-title { margin: 48px 0 16px; font-size: 22px; }
.leaderboard { width: 100%; border-collapse: collapse; }
.leaderboard th, .leaderboard td { padding: 10px 12px; border-bottom: 1px solid var(--border); text-align: left; }
.leaderboard th { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }
.leaderboard td.joi { color: var(--accent); font-family: "JetBrains Mono", monospace; font-weight: 600; }
.btn { display: inline-block; padding: 8px 14px; background: var(--panel); border: 1px solid var(--border);
  border-radius: 8px; color: var(--ink); font-size: 13px; text-decoration: none; margin: 4px 6px 0 0; }
.btn:hover { border-color: var(--accent2); }
.pitch-svg { width: 100%; height: auto; max-width: 1280px; margin: 24px 0; }
```

- [ ] **Step 2: Create `site/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>WC 2026 Chemistry</title>
<link rel="stylesheet" href="style.css" />
</head>
<body>
<main>
  <h1>World Cup 2026 — Chemistry</h1>
  <p class="lead">Offensive chemistry (JOI90) between every player pair within each WC 2026 squad, computed from their actual international matches. Built on the methodology of <a href="https://www.janvanhaaren.be/assets/papers/mitssac-2020-chemistry.pdf">Bransen &amp; Van Haaren, MITSSAC 2020</a>.</p>

  <div class="grid" id="nation-grid"></div>

  <h2 class="section-title">Global top pairs heading into WC 2026</h2>
  <table class="leaderboard" id="leaderboard">
    <thead><tr><th>#</th><th>Pair</th><th>JOI90</th><th>Minutes</th><th>Matches</th></tr></thead>
    <tbody></tbody>
  </table>
</main>
<script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 3: Create `site/app.js`**

```javascript
const FLAG_BASE = "https://cdn.jsdelivr.net/gh/lipis/flag-icons@7/flags/4x3";

async function loadDoc() {
  const res = await fetch("../outputs/chemistry.json");
  return res.json();
}

function renderNationGrid(doc) {
  const grid = document.getElementById("nation-grid");
  const entries = Object.entries(doc.nations).sort((a, b) => {
    const ja = a[1].pairs[0]?.joi90 ?? 0;
    const jb = b[1].pairs[0]?.joi90 ?? 0;
    return jb - ja;
  });
  for (const [code, entry] of entries) {
    const top = entry.pairs[0];
    const flag = entry.squad.flag_iso;
    const card = document.createElement("a");
    card.className = "card";
    card.href = `nation.html?code=${code}`;
    card.innerHTML = `
      <h2><img class="flag" src="${FLAG_BASE}/${flag}.svg" alt=""/> ${entry.squad.nation}</h2>
      <div class="sub">${entry.squad.manager} · ${entry.squad.formation} · ${entry.coverage.matches} matches</div>
      <div class="pair">${top ? `${top.player_a_name} + ${top.player_b_name}` : "no in-squad pairs"}
        ${top ? `<span class="joi"> · ${top.joi90.toFixed(3)}</span>` : ""}</div>`;
    grid.appendChild(card);
  }
}

function renderLeaderboard(doc) {
  const tbody = document.querySelector("#leaderboard tbody");
  for (let i = 0; i < doc.leaderboard.length; i++) {
    const p = doc.leaderboard[i];
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${i + 1}</td>
      <td>${p.player_a_name} + ${p.player_b_name}</td>
      <td class="joi">${p.joi90.toFixed(3)}</td>
      <td>${Math.round(p.minutes)}</td><td>${p.matches}</td>`;
    tbody.appendChild(tr);
  }
}

(async function () {
  const doc = await loadDoc();
  renderNationGrid(doc);
  renderLeaderboard(doc);
})();
```

- [ ] **Step 4: Smoke-test locally**

Run:
```bash
cd /Users/nick/wc2026-chemistry && python -m http.server 8080 &
open http://localhost:8080/site/
```
Expected: 32 cards render with flags; leaderboard fills in. Kill the server with `kill %1` when done.

- [ ] **Step 5: Commit**

```bash
git add site/index.html site/style.css site/app.js
git commit -m "Site: landing page with 32-nation grid + global leaderboard"
```

---

### Task 19: Per-nation page + D3 pitch chemistry SVG

**Files:**
- Create: `site/nation.html`
- Create: `site/pitch.js`

- [ ] **Step 1: Create `site/nation.html`**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Nation — WC 2026 Chemistry</title>
<link rel="stylesheet" href="style.css" />
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
</head>
<body>
<main>
  <p><a href="index.html">&larr; All nations</a></p>
  <h1 id="title"></h1>
  <p class="lead" id="subtitle"></p>
  <svg id="pitch" class="pitch-svg" viewBox="0 0 1050 680" preserveAspectRatio="xMidYMid meet"></svg>
  <div id="downloads"></div>
  <h2 class="section-title">All in-squad chemistry pairs</h2>
  <table class="leaderboard" id="pairs">
    <thead><tr><th>#</th><th>Pair</th><th>JOI90</th><th>Minutes</th><th>Matches</th></tr></thead>
    <tbody></tbody>
  </table>
</main>
<script src="pitch.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `site/pitch.js`**

```javascript
const POS = {
  GK:  [80, 340], RB: [250, 580], RWB: [400, 600], RCB: [220, 440],
  CB:  [220, 340], LCB: [220, 240], LB: [250, 100], LWB: [400, 80],
  RDM: [350, 420], DM: [350, 340], LDM: [350, 260],
  RCM: [550, 420], CM: [550, 340], LCM: [550, 260],
  RM:  [600, 560], LM:  [600, 120],
  RAM: [700, 420], AM:  [700, 340], LAM: [700, 260],
  RW:  [850, 560], LW:  [850, 120], ST:  [900, 340], CF: [900, 340], SS: [820, 340],
};

function lerpColor(c, h, t) {
  const a = c.match(/.{2}/g).map(s => parseInt(s, 16));
  const b = h.match(/.{2}/g).map(s => parseInt(s, 16));
  return "#" + a.map((v, i) => Math.round(v + (b[i] - v) * t).toString(16).padStart(2, "0")).join("");
}

function colorFor(joi, min, max) {
  if (max === min) return "4ade80";
  const t = Math.max(0, Math.min(1, (joi - min) / (max - min)));
  return "#" + lerpColor("ef4444", "4ade80", t).slice(1);
}

async function main() {
  const params = new URLSearchParams(location.search);
  const code = params.get("code");
  const doc = await (await fetch("../outputs/chemistry.json")).json();
  const entry = doc.nations[code];
  if (!entry) {
    document.getElementById("title").textContent = `Unknown nation: ${code}`;
    return;
  }
  const squad = entry.squad;

  document.getElementById("title").textContent = squad.nation;
  document.getElementById("subtitle").textContent =
    `${squad.manager} · ${squad.formation} · ${entry.coverage.matches} matches in window`;

  const svg = d3.select("#pitch");
  svg.append("rect")
    .attr("x", 0).attr("y", 0).attr("width", 1050).attr("height", 680)
    .attr("fill", "#1a4d2e").attr("opacity", 0.4);

  // Pitch lines
  svg.append("rect").attr("x", 10).attr("y", 10).attr("width", 1030).attr("height", 660)
    .attr("fill", "none").attr("stroke", "#fff").attr("stroke-width", 2);
  svg.append("line").attr("x1", 525).attr("x2", 525).attr("y1", 10).attr("y2", 670)
    .attr("stroke", "#fff").attr("stroke-width", 2);
  svg.append("circle").attr("cx", 525).attr("cy", 340).attr("r", 80)
    .attr("fill", "none").attr("stroke", "#fff").attr("stroke-width", 2);

  // Place players
  const namePos = {};
  for (const p of squad.players) {
    namePos[p.name] = POS[p.position] || [525, 340];
  }

  // Edges
  const pairs = entry.pairs;
  const vmin = d3.min(pairs, d => d.joi90) ?? 0;
  const vmax = d3.max(pairs, d => d.joi90) ?? 0;

  svg.append("g").selectAll("line")
    .data(pairs).enter().append("line")
      .attr("x1", d => (namePos[d.player_a_name] || [0, 0])[0])
      .attr("y1", d => (namePos[d.player_a_name] || [0, 0])[1])
      .attr("x2", d => (namePos[d.player_b_name] || [0, 0])[0])
      .attr("y2", d => (namePos[d.player_b_name] || [0, 0])[1])
      .attr("stroke", d => colorFor(d.joi90, vmin, vmax))
      .attr("stroke-width", d => 2 + 6 * Math.min(d.minutes, 600) / 600)
      .attr("stroke-linecap", "round")
      .attr("opacity", 0.85)
      .append("title")
        .text(d => `${d.player_a_name} + ${d.player_b_name}\nJOI90 ${d.joi90.toFixed(3)} · ${Math.round(d.minutes)} mins · ${d.matches} matches`);

  // Players
  const g = svg.append("g");
  for (const [name, [x, y]] of Object.entries(namePos)) {
    g.append("circle").attr("cx", x).attr("cy", y).attr("r", 14)
      .attr("fill", "#fff");
    g.append("circle").attr("cx", x).attr("cy", y).attr("r", 11)
      .attr("fill", squad.team_color);
    g.append("text").attr("x", x).attr("y", y - 18)
      .attr("text-anchor", "middle").attr("fill", "#fff")
      .attr("font-size", 14).attr("font-weight", 700).text(name);
  }

  // Downloads
  const dl = document.getElementById("downloads");
  for (const [label, file] of [
    ["Pitch SVG", "pitch.svg"], ["Pitch PNG 4K", "pitch.png"],
    ["Branded PNG", "pitch_branded.png"], ["Data JSON", "data.json"],
  ]) {
    const a = document.createElement("a");
    a.className = "btn"; a.href = `../exports/nations/${code}/${file}`;
    a.textContent = `Download ${label}`; a.download = `${code}_${file}`;
    dl.appendChild(a);
  }

  // Pairs table
  const tbody = document.querySelector("#pairs tbody");
  pairs.forEach((p, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${i+1}</td><td>${p.player_a_name} + ${p.player_b_name}</td>
      <td class="joi">${p.joi90.toFixed(3)}</td><td>${Math.round(p.minutes)}</td><td>${p.matches}</td>`;
    tbody.appendChild(tr);
  });
}
main();
```

- [ ] **Step 3: Smoke-test**

Run:
```bash
cd /Users/nick/wc2026-chemistry && python -m http.server 8080 &
open "http://localhost:8080/site/nation.html?code=USA"
```
Expected: pitch SVG renders with USA players and chemistry edges; download buttons present; pairs table populated. Kill the server with `kill %1` when done.

- [ ] **Step 4: Commit**

```bash
git add site/nation.html site/pitch.js
git commit -m "Site: per-nation page with D3 pitch chemistry SVG + downloads"
```

---

## Phase 7 — Polish + deploy

### Task 20: Vendor flag SVGs

**Files:**
- Create: `assets/flags/` (vendored)
- Modify: `site/app.js` and `site/pitch.js` (optional swap from CDN to local)

- [ ] **Step 1: Decide CDN vs vendored**

Per spec, flags come from `flag-icons` MIT-licensed. The CDN approach in Task 18 works for the live site but won't work offline. For the deliverable bundle, vendor them.

Run:
```bash
mkdir -p assets/flags && cd assets/flags && \
  curl -fsSL https://github.com/lipis/flag-icons/archive/refs/heads/main.zip -o flags.zip && \
  unzip -qq -o flags.zip "flag-icons-main/flags/4x3/*" && \
  mv flag-icons-main/flags/4x3/*.svg . && rm -rf flag-icons-main flags.zip && cd -
```
Expected: `assets/flags/us.svg`, `assets/flags/br.svg`, ... exist.

- [ ] **Step 2: Update `site/app.js` to prefer local flags**

Replace `const FLAG_BASE = "https://cdn.jsdelivr.net/gh/lipis/flag-icons@7/flags/4x3";` with:
```javascript
const FLAG_BASE = "../assets/flags";
```

- [ ] **Step 3: Sanity-check the site**

Run the local server again; confirm flags load without network.

- [ ] **Step 4: Commit (only the JS change; flags themselves go in a separate commit)**

```bash
git add assets/flags
git commit -m "Assets: vendor flag-icons SVGs (MIT)"
git add site/app.js
git commit -m "Site: use vendored flags instead of CDN"
```

---

### Task 21: Headshot pipeline (best-effort, Wikipedia Commons)

**Files:**
- Create: `scripts/fetch_headshots.py`
- Create: `assets/headshots/.gitkeep`

This is best-effort. We do NOT scrape Transfermarkt/SoFIFA. Players without a free photo fall back to an initials disc rendered client-side.

- [ ] **Step 1: Create `scripts/fetch_headshots.py`**

```python
"""Best-effort headshot fetch from Wikipedia Commons.

For each player listed in any squads/wc2026/*.yaml, query the Wikipedia API
for their page, follow the page image, and save to assets/headshots/<slug>.jpg
at max 256x256. Players without a hit are skipped (left to the initials fallback).

Respects Wikipedia's rate limits. Run sparingly.
"""
from __future__ import annotations

import argparse
import logging
import re
import time
import unicodedata
from pathlib import Path

import requests

from chemistry.squads import load_all_squads

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("headshots")

OUT = Path("assets/headshots")
WP = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "wc2026-chemistry/0.1 (https://github.com/stranger9977/wc2026-chemistry)"}


def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()


def find_image_url(player_name: str) -> str | None:
    params = {
        "action": "query", "format": "json", "prop": "pageimages",
        "piprop": "original", "titles": player_name, "redirects": 1,
    }
    r = requests.get(WP, params=params, headers=HEADERS, timeout=10)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    for page in pages.values():
        img = page.get("original", {}).get("source")
        if img:
            return img
    return None


def fetch_one(player_name: str) -> Path | None:
    target = OUT / f"{slugify(player_name)}.jpg"
    if target.exists():
        return target
    url = find_image_url(player_name)
    if not url:
        log.info("No image found: %s", player_name)
        return None
    img = requests.get(url, headers=HEADERS, timeout=15)
    if not img.ok:
        log.warning("Image fetch failed for %s: %s", player_name, img.status_code)
        return None
    OUT.mkdir(parents=True, exist_ok=True)
    target.write_bytes(img.content)
    log.info("Saved %s", target)
    time.sleep(0.5)  # be nice
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--squads", default="squads/wc2026", type=Path)
    args = parser.parse_args()
    squads = load_all_squads(args.squads)
    seen: set[str] = set()
    for sq in squads.values():
        for p in sq.players:
            if p.name in seen:
                continue
            seen.add(p.name)
            try:
                fetch_one(p.name)
            except Exception as exc:
                log.warning("Error fetching %s: %s", p.name, exc)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create `assets/headshots/.gitkeep`**

```bash
mkdir -p assets/headshots && touch assets/headshots/.gitkeep
```

- [ ] **Step 3: Run for the USA roster only (smoke test)**

Run:
```bash
python -c "
from chemistry.squads import load_squad
from scripts.fetch_headshots import fetch_one
sq = load_squad('squads/wc2026/USA.yaml')
for p in sq.players: fetch_one(p.name)
"
```
Expected: a few JPGs appear in `assets/headshots/`. Players without freely-licensed photos are skipped silently.

- [ ] **Step 4: Add client-side initials fallback to `site/pitch.js`**

Replace the player-rendering block in `pitch.js` with one that checks for a headshot file first:
```javascript
// Players with optional headshot
for (const player of squad.players) {
  const [x, y] = POS[player.position] || [525, 340];
  const slug = player.name.toLowerCase()
    .normalize("NFD").replace(/[̀-ͯ]/g, "")
    .replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  const img = `../assets/headshots/${slug}.jpg`;
  // Probe headshot existence by attempting an image load; otherwise initials.
  const initials = player.name.split(/\s+/).map(s => s[0]).slice(0, 2).join("").toUpperCase();
  g.append("circle").attr("cx", x).attr("cy", y).attr("r", 18).attr("fill", "#fff");
  g.append("circle").attr("cx", x).attr("cy", y).attr("r", 15).attr("fill", squad.team_color);
  g.append("text").attr("x", x).attr("y", y + 4)
    .attr("text-anchor", "middle").attr("fill", "#fff")
    .attr("font-size", 13).attr("font-weight", 700).text(initials);
  g.append("text").attr("x", x).attr("y", y - 22)
    .attr("text-anchor", "middle").attr("fill", "#fff")
    .attr("font-size", 12).text(player.name);
}
```

(For v1 we just render initials; the headshot images live in `assets/headshots/` for use by the partner's animator if they want them.)

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_headshots.py assets/headshots/.gitkeep site/pitch.js
git commit -m "Headshots: best-effort fetch from Wikipedia + initials fallback in viz"
```

---

### Task 22: GitHub repo + Pages deploy

**Files:**
- Create: `scripts/deploy.sh`
- Modify: `.github/workflows/pages.yml`

- [ ] **Step 1: Create the GitHub repo**

Run:
```bash
gh repo create wc2026-chemistry --public \
  --description "Player chemistry pipeline + video-ready chart assets for FIFA WC 2026" \
  --source=. --remote=origin
git push -u origin main
```
Expected: repo created at `https://github.com/stranger9977/wc2026-chemistry`.

- [ ] **Step 2: Create a Pages workflow** at `.github/workflows/pages.yml`

```yaml
name: Deploy site to GitHub Pages
on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/configure-pages@v5
      - name: Stage public files
        run: |
          mkdir -p _site/site _site/outputs _site/exports _site/assets
          cp -r site/* _site/site/
          cp -r outputs/* _site/outputs/ 2>/dev/null || true
          cp -r exports/* _site/exports/ 2>/dev/null || true
          cp -r assets/* _site/assets/ 2>/dev/null || true
          cp site/index.html _site/index.html
      - uses: actions/upload-pages-artifact@v3
        with: { path: _site }
      - id: deployment
        uses: actions/deploy-pages@v4
```

Note: the `_site` layout means the live URL serves `index.html` at the root; the rest sits beneath at `/outputs/`, `/exports/`, etc. The script copies `site/index.html` to the root so `app.js`'s `fetch("../outputs/chemistry.json")` still resolves — wait, that path won't work. Let me fix.

- [ ] **Step 3: Make site paths root-relative**

In `site/app.js`, change:
```javascript
const res = await fetch("../outputs/chemistry.json");
```
to:
```javascript
const res = await fetch("outputs/chemistry.json");
```
And in `site/pitch.js`, change:
```javascript
const doc = await (await fetch("../outputs/chemistry.json")).json();
// ...
a.href = `../exports/nations/${code}/${file}`;
```
to:
```javascript
const doc = await (await fetch("outputs/chemistry.json")).json();
// ...
a.href = `exports/nations/${code}/${file}`;
```

Adjust the local-server smoke test to serve from the project root (it already does) — the same root-relative paths work.

- [ ] **Step 4: Commit + push**

```bash
git add .github/workflows/pages.yml site/app.js site/pitch.js
git commit -m "Deploy: Pages workflow + root-relative paths"
git push
```

- [ ] **Step 5: Enable Pages**

Run:
```bash
gh api -X POST /repos/stranger9977/wc2026-chemistry/pages \
  -f "source[branch]=gh-pages" -f "source[path]=/" 2>&1 || true
gh api -X POST /repos/stranger9977/wc2026-chemistry/pages \
  -f "build_type=workflow" 2>&1 || true
```

Wait for the workflow run:
```bash
gh run watch
```
Expected: green build; the URL is `https://stranger9977.github.io/wc2026-chemistry/`.

- [ ] **Step 6: Commit a deploy script for local rebuilds**

Create `scripts/deploy.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
python -m scripts.build --heuristic-vaep
python -m scripts.export
git add outputs/chemistry.json exports/
git commit -m "Refresh: chemistry build + exports" || echo "nothing to commit"
git push
```

```bash
chmod +x scripts/deploy.sh
git add scripts/deploy.sh
git commit -m "Scripts: local refresh deploy helper"
git push
```

---

### Task 23: README + credits

**Files:**
- Modify: `README.md`
- Create: `CREDITS.md`

- [ ] **Step 1: Expand `README.md`**

Replace with:
```markdown
# WC 2026 Chemistry

Player chemistry pipeline for the FIFA World Cup 2026. Computes offensive chemistry (JOI90) between every player pair within each of the 32 squads from StatsBomb open international event data, and renders video-ready chart assets (4K PNG, SVG, lower-third pair cards, leaderboard graphics) plus a GitHub Pages chart browser.

**Method:** based on Bransen & Van Haaren, *Player Chemistry: Striving for a Perfectly Balanced Soccer Team*, MITSSAC 2020 ([PDF](https://www.janvanhaaren.be/assets/papers/mitssac-2020-chemistry.pdf)).

**Live site:** https://stranger9977.github.io/wc2026-chemistry/

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

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
- `scripts/` — entry points: bootstrap_squads, train_vaep, build, export, fetch_headshots, deploy
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
```

- [ ] **Step 2: Create `CREDITS.md`**

```markdown
# Credits

## Data

- **Match events:** [StatsBomb Open Data](https://github.com/statsbomb/open-data) — free for non-commercial use with attribution.
- **Lineups, minutes, positions:** StatsBomb via [statsbombpy](https://github.com/statsbomb/statsbombpy) (MIT).
- **Squad rosters:** hand-curated by the project author from publicly announced call-ups.

## Methodology

- **JOI / JDI metrics, Team Builder:** Bransen, L. & Van Haaren, J. (2020). *Player Chemistry: Striving for a Perfectly Balanced Soccer Team.* MITSSAC. [PDF](https://www.janvanhaaren.be/assets/papers/mitssac-2020-chemistry.pdf).
- **VAEP action values:** Decroos, T., Bransen, L., Van Haaren, J., & Davis, J. (2019). *Actions Speak Louder than Goals.* KDD '19.
- **SPADL representation, VAEP implementation:** [socceraction](https://github.com/ML-KULeuven/socceraction) (Apache 2.0).

## Tooling

- [socceraction](https://github.com/ML-KULeuven/socceraction) (Apache 2.0)
- [mplsoccer](https://github.com/andrewRowlinson/mplsoccer) (MIT)
- [statsbombpy](https://github.com/statsbomb/statsbombpy) (MIT)
- [xgboost](https://github.com/dmlc/xgboost) (Apache 2.0)
- [PuLP](https://github.com/coin-or/pulp) (MIT) — reserved for v2 Team Builder
- [flag-icons](https://github.com/lipis/flag-icons) (MIT)

## Headshots

Best-effort sourced from Wikipedia Commons via the MediaWiki API. Photos retain their original CC-BY-SA or other free license. Players without a freely-licensed photo render as initials.
```

- [ ] **Step 3: Commit + push**

```bash
git add README.md CREDITS.md
git commit -m "Docs: README + credits"
git push
```

---

## Phase 8 — Verification

### Task 24: End-to-end smoke run

- [ ] **Step 1: Fresh-clone simulation**

```bash
cd /tmp && rm -rf wc2026-chemistry-smoke && \
  git clone https://github.com/stranger9977/wc2026-chemistry wc2026-chemistry-smoke && \
  cd wc2026-chemistry-smoke && \
  python -m venv .venv && source .venv/bin/activate && \
  pip install -r requirements.txt
```
Expected: clean install, no errors.

- [ ] **Step 2: Run the pipeline**

```bash
python -m scripts.build --heuristic-vaep
python -m scripts.export
```
Expected: pipeline finishes; `outputs/chemistry.json` exists; `exports/nations/<32 codes>/pitch.png` exist.

- [ ] **Step 3: Sanity checks**

```bash
python -c "
import json
doc = json.load(open('outputs/chemistry.json'))
nations = doc['nations']
print('Nations:', len(nations))
for code, e in sorted(nations.items()):
    top = e['pairs'][0] if e['pairs'] else None
    print(f'  {code:4} {e[\"squad\"][\"nation\"]:25} matches={e[\"coverage\"][\"matches\"]:>3}  top={top[\"player_a_name\"]+\" + \"+top[\"player_b_name\"]+\" (\"+f\"{top[\"joi90\"]:.3f}\"+\")\" if top else \"—\"}')
print('Leaderboard:', len(doc['leaderboard']))
print('Top 5:')
for p in doc['leaderboard'][:5]:
    print('  ', p['player_a_name'], '+', p['player_b_name'], '→', f'{p[\"joi90\"]:.3f}')
"
```
Expected:
- 32 nations listed
- Some nations have `matches=0` (we'll iterate on squad rosters / additional comps in v1.1) — these get a warning on the site
- Top-5 leaderboard pairs are well-known international duos (e.g. Argentina or France pairs from WC 2022)

- [ ] **Step 4: Spot-check the Argentina sanity reproduction**

```bash
python -c "
import json
doc = json.load(open('outputs/chemistry.json'))
arg = doc['nations'].get('ARG', {}).get('pairs', [])
print('Top Argentina pairs:')
for p in arg[:5]:
    print(' ', p['player_a_name'], '+', p['player_b_name'], '→', f'{p[\"joi90\"]:.3f}', 'mins', int(p['minutes']))
"
```
Expected: Messi appears in at least one top-5 pair. (If he doesn't, check `squads/wc2026/ARG.yaml` is filled in. Likely empty in the stub.)

- [ ] **Step 5: Visual inspection**

```bash
open exports/nations/ARG/pitch_branded.png
open exports/leaderboard.png
open exports/landing_grid.png
open http://localhost:8080/site/  # after running `python -m http.server 8080` from project root
```
Expected: all four render. Note any visual glitches as v1.1 follow-ups.

- [ ] **Step 6: Tag the v1**

```bash
git tag v0.1.0 -m "v0.1.0 — initial JOI-only pipeline + 32-nation exports"
git push --tags
```

---

## Out of scope notes (v2 candidates)

- **JDI**: defensive chemistry via the responsibility-grid approach from the paper.
- **Trained VAEP**: run `scripts/train_vaep.py` once the full SPADL cache is built, swap the build to use `kind="trained"`.
- **Animation frame sequence**: render N PNGs per nation as edges fade in by JOI rank for a 5-sec build-up in the video.
- **Player position from data**: replace `DEFAULT_POSITIONS` with averaged per-player x/y from StatsBomb lineups.
- **More competitions**: add qualifiers (UEFA, CONMEBOL, CONCACAF, CAF, AFC) once they're available in StatsBomb open data.

---

## Self-review checklist (already run)

- **Spec coverage**: all 13 spec sections have at least one task (foundation → ingest → SPADL → VAEP → JOI → squads → export → render → site → deploy).
- **Placeholder scan**: clean. The only "TBD" is intentional roster data in YAMLs (the design says rosters are hand-curated).
- **Type consistency**: `joi90`, `minutes`, `matches`, `player_a`/`player_b` (canonicalised), `player_a_name`/`player_b_name` are used consistently from joi.py → rank.py → export.py → render.py.
- **Ambiguity check**: VAEP signature uncertainty in Task 4 is called out with a diagnostic command. SPADL converter signature uncertainty in Task 3 likewise.
