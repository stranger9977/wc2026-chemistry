"""SPADL conversion + VAEP scoring of StatsBomb events.

Adapter notes
-------------
The cached events at data/raw/43_106/ were ingested via statsbombpy, which
returns *flat* DataFrames (e.g. ``pass_outcome``, ``pass_body_part`` columns)
rather than the nested-dict ``extra`` column that
``socceraction.spadl.statsbomb.convert_to_actions`` expects.

``_flatten_to_socceraction`` rebuilds the nested schema so the socceraction
converter can be used without modification.

Column mappings applied:
  id          -> event_id
  match_id    -> game_id
  period      -> period_id
  type        -> type_name  (plain string, e.g. "Pass")
  All flat pass_*/shot_*/carry_*/dribble_*/etc. columns -> extra dict

Locations are stored as numpy arrays; they are converted to plain Python lists
as expected by the converter.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SPADL_DIR = Path(__file__).parent.parent / "data" / "spadl"


# ---------------------------------------------------------------------------
# Flat-to-socceraction adapter
# ---------------------------------------------------------------------------

def _to_list(val: Any) -> Any:
    """Convert numpy arrays to Python lists for JSON-compatibility."""
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def _is_scalar_notna(val: Any) -> bool:
    """Return True if val is a non-null scalar (not a numpy array or None)."""
    if val is None:
        return False
    if isinstance(val, np.ndarray):
        # A numpy array means the field has a value (e.g. end_location)
        return True
    try:
        return bool(pd.notna(val))
    except (TypeError, ValueError):
        return False


def _build_extra(row: pd.Series) -> dict:
    """Build the ``extra`` nested-dict that socceraction expects from a flat row."""
    extra: dict = {}
    type_name = row.get("type_name", "")

    if type_name == "Pass":
        pass_dict: dict = {}
        if _is_scalar_notna(row.get("pass_type")):
            pass_dict["type"] = {"name": row["pass_type"]}
        if _is_scalar_notna(row.get("pass_height")):
            pass_dict["height"] = {"name": row["pass_height"]}
        if _is_scalar_notna(row.get("pass_cross")) and row["pass_cross"]:
            pass_dict["cross"] = True
        if _is_scalar_notna(row.get("pass_outcome")):
            pass_dict["outcome"] = {"name": row["pass_outcome"]}
        if _is_scalar_notna(row.get("pass_body_part")):
            pass_dict["body_part"] = {"name": row["pass_body_part"]}
        end_loc = row.get("pass_end_location")
        if end_loc is not None:
            pass_dict["end_location"] = _to_list(end_loc)
        extra["pass"] = pass_dict

    elif type_name == "Shot":
        shot_dict: dict = {}
        if _is_scalar_notna(row.get("shot_type")):
            shot_dict["type"] = {"name": row["shot_type"]}
        if _is_scalar_notna(row.get("shot_outcome")):
            shot_dict["outcome"] = {"name": row["shot_outcome"]}
        if _is_scalar_notna(row.get("shot_body_part")):
            shot_dict["body_part"] = {"name": row["shot_body_part"]}
        end_loc = row.get("shot_end_location")
        if end_loc is not None:
            shot_dict["end_location"] = _to_list(end_loc)
        extra["shot"] = shot_dict

    elif type_name == "Carry":
        carry_dict: dict = {}
        end_loc = row.get("carry_end_location")
        if end_loc is not None:
            carry_dict["end_location"] = _to_list(end_loc)
        extra["carry"] = carry_dict

    elif type_name == "Dribble":
        dribble_dict: dict = {}
        if _is_scalar_notna(row.get("dribble_outcome")):
            dribble_dict["outcome"] = {"name": row["dribble_outcome"]}
        extra["dribble"] = dribble_dict

    elif type_name == "Foul Committed":
        foul_dict: dict = {}
        if _is_scalar_notna(row.get("foul_committed_card")):
            foul_dict["card"] = {"name": row["foul_committed_card"]}
        extra["foul_committed"] = foul_dict

    elif type_name == "Duel":
        duel_dict: dict = {}
        if _is_scalar_notna(row.get("duel_type")):
            duel_dict["type"] = {"name": row["duel_type"]}
        if _is_scalar_notna(row.get("duel_outcome")):
            duel_dict["outcome"] = {"name": row["duel_outcome"]}
        extra["duel"] = duel_dict

    elif type_name == "Interception":
        interception_dict: dict = {}
        if _is_scalar_notna(row.get("interception_outcome")):
            interception_dict["outcome"] = {"name": row["interception_outcome"]}
        extra["interception"] = interception_dict

    elif type_name == "Goal Keeper":
        gk_dict: dict = {}
        if _is_scalar_notna(row.get("goalkeeper_type")):
            gk_dict["type"] = {"name": row["goalkeeper_type"]}
        if _is_scalar_notna(row.get("goalkeeper_outcome")):
            gk_dict["outcome"] = {"name": row["goalkeeper_outcome"]}
        if _is_scalar_notna(row.get("goalkeeper_body_part")):
            gk_dict["body_part"] = {"name": row["goalkeeper_body_part"]}
        extra["goalkeeper"] = gk_dict

    elif type_name == "Clearance":
        clearance_dict: dict = {}
        if _is_scalar_notna(row.get("clearance_body_part")):
            clearance_dict["body_part"] = {"name": row["clearance_body_part"]}
        extra["clearance"] = clearance_dict

    return extra


def _flatten_to_socceraction(events: pd.DataFrame) -> pd.DataFrame:
    """Reshape flat statsbombpy events into the schema socceraction expects."""
    df = events.copy()

    # Rename columns to match socceraction's expected schema
    df = df.rename(columns={
        "id": "event_id",
        "match_id": "game_id",
        "period": "period_id",
        "type": "type_name",
    })

    # Convert location arrays to Python lists
    df["location"] = df["location"].apply(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else x
    )

    # Build the nested extra dict per row
    df["extra"] = df.apply(_build_extra, axis=1)

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _mock_pandera() -> None:
    """Inject a lightweight pandera stub into sys.modules.

    The installed pandera version triggers an ImportError at startup because
    it depends on ``multimethod.overload`` which was removed in a newer release
    of multimethod.  We only need pandera's DataFrame/SchemaModel as type
    annotations so a minimal stub is sufficient.
    """
    import sys
    import types

    if "pandera" in sys.modules:
        return  # already loaded (real or mock)

    pandera_mock = types.ModuleType("pandera")
    pandera_typing_mock = types.ModuleType("pandera.typing")

    class _SchemaModel:
        pass

    class _DataFrameMeta(type):
        def __getitem__(cls, item: Any) -> type:
            return cls

    class _DataFrame(metaclass=_DataFrameMeta):
        pass

    class _Series:
        def __class_getitem__(cls, item: Any) -> type:
            return cls

    pandera_typing_mock.DataFrame = _DataFrame  # type: ignore[attr-defined]
    pandera_typing_mock.Series = _Series  # type: ignore[attr-defined]
    pandera_mock.typing = pandera_typing_mock  # type: ignore[attr-defined]
    pandera_mock.SchemaModel = _SchemaModel  # type: ignore[attr-defined]
    pandera_mock.Field = lambda *a, **kw: None  # type: ignore[attr-defined]

    sys.modules["pandera"] = pandera_mock
    sys.modules["pandera.typing"] = pandera_typing_mock


def to_spadl(events: pd.DataFrame) -> pd.DataFrame:
    """Convert one match's StatsBomb events (flat statsbombpy shape) into SPADL actions.

    Parameters
    ----------
    events:
        DataFrame as produced by the ingest pipeline (flat statsbombpy columns).

    Returns
    -------
    pd.DataFrame
        SPADL actions sorted by period and time.
    """
    import warnings

    _mock_pandera()

    from socceraction.spadl import statsbomb as spadl_sb

    adapted = _flatten_to_socceraction(events)
    home_team_id = int(adapted["team_id"].dropna().iloc[0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actions = spadl_sb.convert_to_actions(
            adapted,
            home_team_id=home_team_id,
            xy_fidelity_version=2,
            shot_fidelity_version=2,
        )

    return actions.sort_values(["period_id", "time_seconds"]).reset_index(drop=True)


def convert_competition(competition_dir: Path, out_dir: Path = SPADL_DIR) -> Path:
    """Convert every match in a cached competition dir to SPADL.

    Parameters
    ----------
    competition_dir:
        Path to a cached competition directory (e.g. ``data/raw/43_106``).
    out_dir:
        Output directory for SPADL parquet files.

    Returns
    -------
    Path
        Path to the output competition directory.
    """
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
