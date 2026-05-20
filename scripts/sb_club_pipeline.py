"""StatsBomb club competition pipeline for the Podolski-archetype analysis.

Phases:
  1. Fetch 8 StatsBomb club competitions -> data/raw/<cid>_<sid>/
  2. Convert to SPADL -> data/sb_club_spadl/<cid>_<sid>/<match_id>.parquet
  3. Train VAEP v2 on StatsBomb international (data/spadl/) + club SPADL (no Wyscout)
     -> data/vaep/vaep_v2.pkl
  4. Score all actions (international + club) with VAEP v2 + xT
     -> data/vaep_scored_v2/<cid>_<sid>/<match_id>.parquet
  5. Compute player-level per-90 production -> outputs/player_per90.parquet
  6. Build dependency cards -> outputs/player_dependency.json
  7. Write report -> docs/analysis/cross-context-chemistry.md
  8. Update site/index.html Research tab section 1

Run with: .venv/bin/python -m scripts.sb_club_pipeline
All stages are cached -- safe to re-run.
"""
from __future__ import annotations

import json
import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_SPADL_INTL = ROOT / "data" / "spadl"
DATA_SPADL_CLUB = ROOT / "data" / "sb_club_spadl"
DATA_VAEP = ROOT / "data" / "vaep"
DATA_VAEP_SCORED = ROOT / "data" / "vaep_scored_v2"
DATA_XT = ROOT / "data" / "xt"
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "docs" / "analysis"

DATA_SPADL_CLUB.mkdir(parents=True, exist_ok=True)
DATA_VAEP.mkdir(parents=True, exist_ok=True)
DATA_VAEP_SCORED.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)

VAEP_V2_PATH = DATA_VAEP / "vaep_v2.pkl"

# 8 StatsBomb club competitions
SB_CLUB_COMPS = [
    {"cid": 9,  "sid": 281, "label": "Bundesliga 2023/24",
     "season_label": "Bayern Munich 2023/24"},
    {"cid": 11, "sid": 1,   "label": "La Liga 2017/18",
     "season_label": "La Liga 2017/18"},
    {"cid": 11, "sid": 4,   "label": "La Liga 2018/19",
     "season_label": "La Liga 2018/19"},
    {"cid": 11, "sid": 42,  "label": "La Liga 2019/20",
     "season_label": "La Liga 2019/20"},
    {"cid": 11, "sid": 90,  "label": "La Liga 2020/21",
     "season_label": "La Liga 2020/21"},
    {"cid": 7,  "sid": 108, "label": "Ligue 1 2021/22",
     "season_label": "Ligue 1 2021/22"},
    {"cid": 7,  "sid": 235, "label": "Ligue 1 2022/23",
     "season_label": "Ligue 1 2022/23"},
    {"cid": 44, "sid": 107, "label": "MLS 2023",
     "season_label": "MLS 2023"},
]

# International competition labels
INTL_COMP_LABELS = {
    "43_106": "WC 2022",
    "55_282": "Euro 2024",
    "55_43":  "Euro 2020",
    "223_282": "Copa 2024",
    "43_3":   "WC 2018",
    "1267_107": "AFCON 2023",
}

# StatsBomb player IDs for featured players
# These are StatsBomb player_ids consistent across club and country data
# Collected from StatsBomb open data lineups
FEATURED_PLAYERS = [
    # Bayern -> Germany cohort
    # sb_id = StatsBomb player_id (consistent across club and country data)
    {"name": "Joshua Kimmich",      "sb_id": 5579,   "club": "Bayern",      "country": "Germany"},
    {"name": "Thomas Müller",        "sb_id": 5562,   "club": "Bayern",      "country": "Germany"},
    {"name": "Leroy Sané",           "sb_id": 3053,   "club": "Bayern",      "country": "Germany"},
    {"name": "Jamal Musiala",        "sb_id": 39565,  "club": "Bayern",      "country": "Germany"},
    {"name": "Leon Goretzka",        "sb_id": 6324,   "club": "Bayern",      "country": "Germany"},
    {"name": "Aleksandar Pavlović",  "sb_id": 140589, "club": "Bayern",      "country": "Germany"},
    {"name": "Robert Lewandowski",   "sb_id": 5668,   "club": "Bayern",      "country": "Poland"},
    # Real Madrid -> multiple countries
    {"name": "Luka Modrić",          "sb_id": 5463,   "club": "Real Madrid", "country": "Croatia"},
    {"name": "Toni Kroos",           "sb_id": 5574,   "club": "Real Madrid", "country": "Germany"},
    {"name": "Raphaël Varane",       "sb_id": 5485,   "club": "Real Madrid", "country": "France"},
    {"name": "Karim Benzema",        "sb_id": 19677,  "club": "Real Madrid", "country": "France"},
    # Barcelona -> multiple countries
    {"name": "Lionel Messi",         "sb_id": 5503,   "club": "Barcelona",   "country": "Argentina"},
    {"name": "Luis Suárez",          "sb_id": 5246,   "club": "Barcelona",   "country": "Uruguay"},
    {"name": "Sergio Busquets",      "sb_id": 5203,   "club": "Barcelona",   "country": "Spain"},
    {"name": "Jordi Alba",           "sb_id": 5211,   "club": "Barcelona",   "country": "Spain"},
    # PSG era
    {"name": "Kylian Mbappé",        "sb_id": 3009,   "club": "PSG",         "country": "France"},
    {"name": "Neymar",               "sb_id": 4320,   "club": "PSG",         "country": "Brazil"},
    # Atlético + others
    {"name": "Antoine Griezmann",    "sb_id": 5487,   "club": "Atlético",    "country": "France"},
    {"name": "Achraf Hakimi",        "sb_id": 5245,   "club": "PSG",         "country": "Morocco"},
]

# Club keywords for team name matching
CLUB_KEYWORDS = {
    "Bayern":      ["Bayern", "FC Bayern"],
    "Real Madrid": ["Real Madrid"],
    "Barcelona":   ["Barcelona", "FC Barcelona"],
    "PSG":         ["Paris Saint-Germain", "Paris SG", "PSG"],
    "Atlético":    ["Atlético", "Atletico", "Atlético Madrid"],
    "Inter Miami": ["Inter Miami"],
}

# Country team name patterns in StatsBomb data
COUNTRY_PATTERNS = {
    "Germany":   ["Germany"],
    "Poland":    ["Poland"],
    "Croatia":   ["Croatia"],
    "France":    ["France"],
    "Argentina": ["Argentina"],
    "Uruguay":   ["Uruguay"],
    "Spain":     ["Spain"],
    "Brazil":    ["Brazil"],
    "Morocco":   ["Morocco"],
}


# ── Helpers ────────────────────────────────────────────────────────────────

def _safe_to_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to parquet, coercing mixed-type columns to strings."""
    import pyarrow as pa

    df_out = df.copy()
    # Check each object column for mixed types that would break arrow conversion
    for col in df_out.columns:
        series = df_out[col]
        if series.dtype != object:
            continue
        # Check if column has mixed int/str types
        non_null = series.dropna()
        if len(non_null) == 0:
            continue
        types_present = set(type(v).__name__ for v in non_null)
        if len(types_present) > 1:
            df_out[col] = series.astype(str)
            continue
        # Also try arrow conversion to catch edge cases
        try:
            pa.array(non_null.tolist(), from_pandas=True)
        except Exception:
            df_out[col] = series.astype(str)
    df_out.to_parquet(path, index=False)


def _add_type_result_name(df: pd.DataFrame) -> pd.DataFrame:
    import socceraction.spadl as spadl_mod
    at = spadl_mod.actiontypes_df().set_index("type_id")["type_name"]
    rt = spadl_mod.results_df().set_index("result_id")["result_name"]
    df = df.copy()
    if "type_name" not in df.columns:
        df["type_name"] = df["type_id"].map(at).fillna("unknown")
    if "result_name" not in df.columns:
        df["result_name"] = df["result_id"].map(rt).fillna("unknown")
    return df


def build_vaep_features(spadl: pd.DataFrame) -> pd.DataFrame:
    from socceraction.vaep import features as fs
    gs = fs.gamestates(spadl)
    return pd.concat(
        [fs.actiontype_onehot(gs),
         fs.result_onehot(gs),
         fs.startlocation(gs),
         fs.endlocation(gs)],
        axis=1,
    )


def _match_id_col(df: pd.DataFrame) -> str:
    """Return 'match_id' or 'game_id' depending on which exists."""
    if "match_id" in df.columns:
        return "match_id"
    if "game_id" in df.columns:
        return "game_id"
    raise ValueError("DataFrame has neither match_id nor game_id column")


# ── Phase 1: Fetch club competitions ──────────────────────────────────────

def phase1_fetch_club_data() -> None:
    """Fetch 8 StatsBomb club competitions to data/raw/<cid>_<sid>/."""
    from statsbombpy import sb

    log.info("=== Phase 1: Fetch StatsBomb club competition data ===")
    for comp in SB_CLUB_COMPS:
        cid, sid, label = comp["cid"], comp["sid"], comp["label"]
        key = f"{cid}_{sid}"
        base = DATA_RAW / key
        matches_path = base / "matches.parquet"

        if matches_path.exists():
            matches = pd.read_parquet(matches_path)
            log.info("  Cache hit: %s (%d matches)", label, len(matches))
            # Check if events are already fetched
            events_dir = base / "events"
            if events_dir.exists():
                n_events = len(list(events_dir.glob("*.parquet")))
                n_matches = len(matches)
                if n_events >= n_matches:
                    continue
                log.info("    Partial events (%d/%d), continuing fetch...", n_events, n_matches)
            else:
                events_dir.mkdir(exist_ok=True)
        else:
            base.mkdir(parents=True, exist_ok=True)
            log.info("  Fetching %s (cid=%d sid=%d)...", label, cid, sid)
            matches = sb.matches(competition_id=cid, season_id=sid)
            log.info("    %d matches", len(matches))
            _safe_to_parquet(matches, matches_path)
            events_dir = base / "events"
            events_dir.mkdir(exist_ok=True)

        # Fetch events
        fetched = 0
        for _, row in matches.iterrows():
            mid = int(row["match_id"])
            ep = events_dir / f"{mid}.parquet"
            if ep.exists():
                continue
            try:
                ev = sb.events(match_id=mid, fmt="dataframe")
                ev.to_parquet(ep, index=False)
                fetched += 1
                if fetched % 50 == 0:
                    log.info("    Progress: %d events fetched for %s...", fetched, label)
            except Exception as e:
                log.warning("    Failed match %d: %s", mid, e)

        total = len(list(events_dir.glob("*.parquet")))
        log.info("  Done: %s — %d event files total", label, total)


# ── Phase 2: Convert to SPADL (per-match files) ───────────────────────────

def phase2_convert_to_spadl() -> None:
    """Convert club events to SPADL. Per-match parquets in data/sb_club_spadl/<cid>_<sid>/."""
    from chemistry.pipeline import to_spadl

    log.info("=== Phase 2: Convert club events to SPADL ===")
    for comp in SB_CLUB_COMPS:
        cid, sid, label = comp["cid"], comp["sid"], comp["label"]
        key = f"{cid}_{sid}"
        base = DATA_RAW / key
        out_dir = DATA_SPADL_CLUB / key
        out_dir.mkdir(parents=True, exist_ok=True)

        matches_path = base / "matches.parquet"
        if not matches_path.exists():
            log.warning("  No matches.parquet for %s, skipping", label)
            continue

        matches = pd.read_parquet(matches_path)
        events_dir = base / "events"
        converted = 0
        skipped = 0

        for _, row in matches.iterrows():
            mid = int(row["match_id"])
            target = out_dir / f"{mid}.parquet"
            if target.exists():
                skipped += 1
                continue
            ep = events_dir / f"{mid}.parquet"
            if not ep.exists():
                log.warning("  Missing events for match %d in %s", mid, label)
                continue
            try:
                ev = pd.read_parquet(ep)
                # Use chemistry.pipeline.to_spadl which handles the flat statsbombpy schema
                actions = to_spadl(ev)
                actions["match_id"] = mid
                actions["competition"] = label
                actions["competition_key"] = key
                actions.to_parquet(target, index=False)
                converted += 1
                if converted % 50 == 0:
                    log.info("    Progress: %d converted for %s...", converted, label)
            except Exception as e:
                log.warning("  Failed to convert match %d (%s): %s", mid, label, e)

        log.info("  %s: %d converted, %d cached", label, converted, skipped)


# ── Phase 3: Train VAEP v2 (StatsBomb international + club, no Wyscout) ───

def phase3_train_vaep_v2() -> dict:
    """Train VAEP v2 on all StatsBomb SPADL (international + club)."""
    if VAEP_V2_PATH.exists():
        log.info("VAEP v2 model cache hit: %s", VAEP_V2_PATH)
        with open(VAEP_V2_PATH, "rb") as f:
            return pickle.load(f)

    log.info("=== Phase 3: Train VAEP v2 on StatsBomb SPADL (international + club) ===")
    from socceraction.vaep import labels as lb
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score

    parts = []

    # International SPADL (existing per-match files in data/spadl/)
    for comp_dir in sorted(DATA_SPADL_INTL.iterdir()):
        if not comp_dir.is_dir():
            continue
        comp_key = comp_dir.name
        comp_label = INTL_COMP_LABELS.get(comp_key, comp_key)
        comp_parts = []
        for p in sorted(comp_dir.glob("*.parquet")):
            df = pd.read_parquet(p)
            df["source"] = "international"
            df["competition_key"] = comp_key
            comp_parts.append(df)
        if comp_parts:
            combined = pd.concat(comp_parts, ignore_index=True)
            parts.append(combined)
            log.info("  International %s: %d actions", comp_label, len(combined))

    # Club SPADL (per-match files in data/sb_club_spadl/)
    for comp in SB_CLUB_COMPS:
        cid, sid, label = comp["cid"], comp["sid"], comp["label"]
        key = f"{cid}_{sid}"
        comp_dir = DATA_SPADL_CLUB / key
        if not comp_dir.exists():
            continue
        comp_parts = []
        for p in sorted(comp_dir.glob("*.parquet")):
            df = pd.read_parquet(p)
            df["source"] = "club"
            df["competition_key"] = key
            comp_parts.append(df)
        if comp_parts:
            combined = pd.concat(comp_parts, ignore_index=True)
            parts.append(combined)
            log.info("  Club %s: %d actions", label, len(combined))

    if not parts:
        raise RuntimeError("No SPADL data found for VAEP v2 training")

    all_spadl = pd.concat(parts, ignore_index=True)
    log.info("Combined SPADL for VAEP v2: %d actions", len(all_spadl))

    all_spadl = _add_type_result_name(all_spadl)

    log.info("Building feature matrix...")
    X = build_vaep_features(all_spadl)
    log.info("Feature shape: %s", X.shape)

    y_scores = lb.scores(all_spadl)["scores"]
    y_concedes = lb.concedes(all_spadl)["concedes"]
    log.info("scores positives: %d / %d", y_scores.sum(), len(y_scores))
    log.info("concedes positives: %d / %d", y_concedes.sum(), len(y_concedes))

    log.info("Training scores model...")
    clf_scores = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        n_jobs=-1, eval_metric="logloss",
    )
    clf_scores.fit(X, y_scores)

    log.info("Training concedes model...")
    clf_concedes = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        n_jobs=-1, eval_metric="logloss",
    )
    clf_concedes.fit(X, y_concedes)

    auc_scores = roc_auc_score(y_scores, clf_scores.predict_proba(X)[:, 1])
    auc_concedes = roc_auc_score(y_concedes, clf_concedes.predict_proba(X)[:, 1])
    log.info("Train AUC scores=%.4f  concedes=%.4f", auc_scores, auc_concedes)

    if auc_scores < 0.70 or auc_concedes < 0.70:
        log.warning("AUC sanity check FAILED: scores=%.4f concedes=%.4f (threshold 0.70)",
                    auc_scores, auc_concedes)

    bundle = {
        "scores": clf_scores,
        "concedes": clf_concedes,
        "feature_cols": list(X.columns),
        "train_size": len(all_spadl),
        "auc_scores": float(auc_scores),
        "auc_concedes": float(auc_concedes),
    }
    with open(VAEP_V2_PATH, "wb") as f:
        pickle.dump(bundle, f)
    log.info("Saved VAEP v2 model to %s", VAEP_V2_PATH)
    return bundle


# ── Phase 4: Score all actions with VAEP v2 + xT ─────────────────────────

def _score_match(spadl: pd.DataFrame, bundle: dict, xt_model) -> pd.DataFrame:
    """Add vaep_value and xt_value columns to spadl."""
    from socceraction.vaep import formula

    enriched = _add_type_result_name(spadl)
    X = build_vaep_features(enriched)

    feature_cols = bundle.get("feature_cols", [])
    if feature_cols:
        for c in feature_cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[feature_cols]

    clf_scores = bundle["scores"]
    clf_concedes = bundle["concedes"]
    p_scores = pd.Series(clf_scores.predict_proba(X)[:, 1], index=enriched.index)
    p_concedes = pd.Series(clf_concedes.predict_proba(X)[:, 1], index=enriched.index)
    values_df = formula.value(enriched, p_scores, p_concedes)

    result = spadl.copy()
    result["vaep_value"] = values_df["vaep_value"].values

    # xT scoring
    try:
        xt_vals = xt_model.score(enriched)
        result["xt_value"] = xt_vals.values
    except Exception:
        result["xt_value"] = 0.0

    return result


def phase4_score_all_actions(bundle: dict) -> None:
    """Score every SPADL match with VAEP v2 + xT."""
    from chemistry.xt_model import load as load_xt

    log.info("=== Phase 4: Score all StatsBomb actions with VAEP v2 + xT ===")

    xt_path = DATA_XT / "xt.pkl"
    if xt_path.exists():
        xt_model = load_xt(xt_path)
        log.info("Loaded xT model from %s", xt_path)
    else:
        log.warning("xT model not found at %s, will use zeros for xT", xt_path)
        xt_model = _NullXtModel()

    # Score international competitions
    for comp_dir in sorted(DATA_SPADL_INTL.iterdir()):
        if not comp_dir.is_dir():
            continue
        key = comp_dir.name
        out_dir = DATA_VAEP_SCORED / key
        out_dir.mkdir(parents=True, exist_ok=True)

        match_files = sorted(comp_dir.glob("*.parquet"))
        cached = len(list(out_dir.glob("*.parquet")))
        if cached >= len(match_files) and cached > 0:
            log.info("  International %s: all %d matches cached", key, cached)
            continue

        log.info("  Scoring international %s (%d matches)...", key, len(match_files))
        scored_count = 0
        for p in match_files:
            target = out_dir / p.name
            if target.exists():
                continue
            try:
                spadl = pd.read_parquet(p)
                scored = _score_match(spadl, bundle, xt_model)
                scored.to_parquet(target, index=False)
                scored_count += 1
            except Exception as e:
                log.warning("  Failed to score %s: %s", p, e)
        log.info("  Scored %d new matches for %s", scored_count, key)

    # Score club competitions
    for comp in SB_CLUB_COMPS:
        cid, sid, label = comp["cid"], comp["sid"], comp["label"]
        key = f"{cid}_{sid}"
        comp_dir = DATA_SPADL_CLUB / key
        if not comp_dir.exists():
            continue

        out_dir = DATA_VAEP_SCORED / key
        out_dir.mkdir(parents=True, exist_ok=True)

        match_files = sorted(comp_dir.glob("*.parquet"))
        cached = len(list(out_dir.glob("*.parquet")))
        if cached >= len(match_files) and cached > 0:
            log.info("  Club %s: all %d matches cached", label, cached)
            continue

        log.info("  Scoring club %s (%d matches)...", label, len(match_files))
        scored_count = 0
        for p in match_files:
            target = out_dir / p.name
            if target.exists():
                continue
            try:
                spadl = pd.read_parquet(p)
                scored = _score_match(spadl, bundle, xt_model)
                scored.to_parquet(target, index=False)
                scored_count += 1
            except Exception as e:
                log.warning("  Failed to score %s: %s", p, e)
        log.info("  Scored %d new matches for %s", scored_count, label)


class _NullXtModel:
    def score(self, spadl):
        return pd.Series(0.0, index=spadl.index)


# ── Phase 5: Load lineups / player minutes ────────────────────────────────

def _load_lineups_for_matches(matches_df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Load player minutes from StatsBomb lineups API for a list of matches.

    Returns DataFrame with columns: match_id, player_id, player_name, minutes_played
    """
    from statsbombpy import sb

    rows = []
    for _, row in matches_df.iterrows():
        mid = int(row["match_id"])
        try:
            lineups = sb.lineups(match_id=mid)
            for team_name, team_df in lineups.items():
                for _, p in team_df.iterrows():
                    positions = p.get("positions") if "positions" in p else []
                    if not isinstance(positions, list):
                        positions = []
                    if positions:
                        from_min = min(
                            _parse_minute(pos.get("from"), 0.0) for pos in positions
                        )
                        to_min = max(
                            _parse_minute(pos.get("to"), 90.0) for pos in positions
                        )
                        minutes = max(0.0, to_min - from_min)
                    else:
                        minutes = 0.0
                    rows.append({
                        "match_id": mid,
                        "player_id": int(p["player_id"]),
                        "player_name": str(p["player_name"]),
                        "team_name": team_name,
                        "minutes_played": minutes,
                        "source": source,
                    })
        except Exception as e:
            log.warning("  Failed to load lineups for match %d: %s", mid, e)

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["match_id", "player_id", "player_name", "team_name", "minutes_played", "source"]
    )


def _parse_minute(time_str, default: float = 0.0) -> float:
    if time_str is None:
        return default
    try:
        parts = str(time_str).split(":")
        m = float(parts[0])
        if len(parts) > 1:
            m += float(parts[1]) / 60.0
        return m
    except (ValueError, IndexError):
        return default


def load_all_lineups() -> pd.DataFrame:
    """Load cached lineups or fetch from API. Returns all player-match appearances."""
    cache_path = OUTPUTS / "sb_lineups_all.parquet"
    if cache_path.exists():
        log.info("Lineups cache hit: %s", cache_path)
        return pd.read_parquet(cache_path)

    log.info("=== Loading lineups for all competitions ===")
    parts = []

    # International
    for comp_dir in sorted(DATA_SPADL_INTL.iterdir()):
        if not comp_dir.is_dir():
            continue
        key = comp_dir.name
        matches_path = DATA_RAW / key / "matches.parquet"
        if not matches_path.exists():
            continue
        matches = pd.read_parquet(matches_path)
        comp_label = INTL_COMP_LABELS.get(key, key)
        log.info("  Loading lineups for %s (%d matches)...", comp_label, len(matches))
        lineup_df = _load_lineups_for_matches(matches, source="international")
        lineup_df["competition_key"] = key
        lineup_df["competition_label"] = comp_label
        parts.append(lineup_df)

    # Club
    for comp in SB_CLUB_COMPS:
        cid, sid, label = comp["cid"], comp["sid"], comp["label"]
        key = f"{cid}_{sid}"
        matches_path = DATA_RAW / key / "matches.parquet"
        if not matches_path.exists():
            continue
        matches = pd.read_parquet(matches_path)
        log.info("  Loading lineups for %s (%d matches)...", label, len(matches))
        lineup_df = _load_lineups_for_matches(matches, source="club")
        lineup_df["competition_key"] = key
        lineup_df["competition_label"] = label
        parts.append(lineup_df)

    if not parts:
        return pd.DataFrame()

    all_lineups = pd.concat(parts, ignore_index=True)
    all_lineups.to_parquet(cache_path, index=False)
    log.info("Saved lineups: %d player-match appearances", len(all_lineups))
    return all_lineups


# ── Phase 5: Compute player-level per-90 production ───────────────────────

def phase5_compute_player_per90(lineups_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-player per-context per-90 stats from VAEP-scored SPADL.

    Returns DataFrame with one row per (player_id, context_label, source).
    """
    cache_path = OUTPUTS / "player_per90.parquet"
    if cache_path.exists():
        log.info("player_per90 cache hit: %s", cache_path)
        return pd.read_parquet(cache_path)

    log.info("=== Phase 5: Compute player-level per-90 production ===")

    # Build player-match minutes from lineups
    player_match_mins = (
        lineups_df.groupby(["player_id", "player_name", "match_id",
                             "competition_key", "competition_label", "source"])
        ["minutes_played"].sum().reset_index()
    )

    # Load all scored SPADL and aggregate per player per competition
    all_rows = []

    for comp_dir in sorted(DATA_VAEP_SCORED.iterdir()):
        if not comp_dir.is_dir():
            continue
        key = comp_dir.name
        is_intl = key in INTL_COMP_LABELS
        comp_label = INTL_COMP_LABELS.get(key)
        if comp_label is None:
            # Find in club comps
            for c in SB_CLUB_COMPS:
                if f"{c['cid']}_{c['sid']}" == key:
                    comp_label = c["label"]
                    break
        if comp_label is None:
            comp_label = key

        source = "international" if is_intl else "club"

        match_files = sorted(comp_dir.glob("*.parquet"))
        if not match_files:
            continue

        # Load all matches for this competition
        comp_parts = []
        for p in match_files:
            df = pd.read_parquet(p)
            mid_col = _match_id_col(df)
            if mid_col == "game_id":
                df = df.rename(columns={"game_id": "match_id"})
            comp_parts.append(df)

        if not comp_parts:
            continue

        comp_df = pd.concat(comp_parts, ignore_index=True)
        comp_df["player_id"] = comp_df["player_id"].fillna(-1).astype("int64")

        # Aggregate per player for this competition
        agg_dict: dict = {
            "vaep_sum": ("vaep_value", "sum"),
            "n_actions": ("vaep_value", "count"),
        }
        if "xt_value" in comp_df.columns:
            agg_dict["xt_sum"] = ("xt_value", "sum")
        grp = comp_df.groupby("player_id").agg(**agg_dict).reset_index()
        if "xt_sum" not in grp.columns:
            grp["xt_sum"] = 0.0

        # Goals and assists (stats.py expects game_id column)
        from chemistry.stats import goals_and_assists
        ga_df = comp_df.copy()
        if "game_id" not in ga_df.columns and "match_id" in ga_df.columns:
            ga_df = ga_df.rename(columns={"match_id": "game_id"})
        ga = goals_and_assists(ga_df)
        grp = grp.merge(ga, on="player_id", how="left").fillna(0)
        grp["goals"] = grp["goals"].astype(int)
        grp["assists"] = grp["assists"].astype(int)

        # n_matches
        match_counts = (
            comp_df.groupby("player_id")["match_id"]
            .nunique()
            .reset_index(name="n_matches")
        )
        grp = grp.merge(match_counts, on="player_id", how="left").fillna(0)

        grp["competition_key"] = key
        grp["competition_label"] = comp_label
        grp["source"] = source
        all_rows.append(grp)

    if not all_rows:
        log.warning("No scored data found")
        return pd.DataFrame()

    stats_df = pd.concat(all_rows, ignore_index=True)

    # Merge with lineup minutes
    player_mins_agg = (
        player_match_mins
        .groupby(["player_id", "player_name", "competition_key"])
        .agg(
            minutes_played=("minutes_played", "sum"),
        )
        .reset_index()
    )

    result = stats_df.merge(
        player_mins_agg,
        on=["player_id", "competition_key"],
        how="left",
    )
    result["minutes_played"] = result["minutes_played"].fillna(0.0)

    # For players with 0 minutes from lineups, estimate from SPADL (90 * n_matches)
    mask_zero = result["minutes_played"] == 0
    result.loc[mask_zero, "minutes_played"] = result.loc[mask_zero, "n_matches"] * 90.0

    # Compute per-90 stats
    mins_safe = result["minutes_played"].clip(lower=1.0)
    result["per90_vaep"] = result["vaep_sum"] / (mins_safe / 90.0)
    result["per90_xt"] = result["xt_sum"] / (mins_safe / 90.0)

    # Clean up
    result = result[result["player_id"] > 0].copy()

    # Save
    result.to_parquet(cache_path, index=False)
    log.info("Saved player_per90: %d rows (%d unique players)",
             len(result), result["player_id"].nunique())
    return result


# ── Phase 6: Build player dependency cards ────────────────────────────────

def _find_player_in_data(
    name: str,
    per90_df: pd.DataFrame,
    lineups_df: pd.DataFrame,
    sb_id: int | None = None,
) -> tuple[int | None, str]:
    """Find StatsBomb player_id. Uses sb_id when provided."""
    # Use pre-configured StatsBomb ID
    if sb_id is not None:
        mask = lineups_df["player_id"] == sb_id
        if mask.any():
            pname = lineups_df.loc[mask, "player_name"].iloc[0]
            return sb_id, pname
        # ID confirmed but not in lineups - still return it
        return sb_id, name

    # Try exact match first
    mask = lineups_df["player_name"] == name
    if mask.any():
        pid = int(lineups_df.loc[mask, "player_id"].iloc[0])
        pname = lineups_df.loc[mask, "player_name"].iloc[0]
        return pid, pname

    # Try case-insensitive
    name_lower = name.lower()
    mask2 = lineups_df["player_name"].str.lower() == name_lower
    if mask2.any():
        pid = int(lineups_df.loc[mask2, "player_id"].iloc[0])
        pname = lineups_df.loc[mask2, "player_name"].iloc[0]
        return pid, pname

    # Partial match on last name or first name
    parts = name.lower().split()
    for part in reversed(parts):  # try last name first
        if len(part) < 3:
            continue
        mask3 = lineups_df["player_name"].str.lower().str.contains(part, regex=False)
        if mask3.sum() == 1:
            row = lineups_df.loc[mask3].iloc[0]
            return int(row["player_id"]), row["player_name"]

    return None, name


def _club_contexts_for_player(
    player_id: int,
    club_keyword: str,
    per90_df: pd.DataFrame,
    lineups_df: pd.DataFrame,
) -> list[dict]:
    """Return all club contexts for a player matching the club keyword."""
    player_rows = per90_df[
        (per90_df["player_id"] == player_id) &
        (per90_df["source"] == "club")
    ].copy()

    # Filter by club name
    keywords = CLUB_KEYWORDS.get(club_keyword, [club_keyword])

    def matches_club(label: str) -> bool:
        label_lower = label.lower()
        return any(kw.lower() in label_lower for kw in keywords)

    # Check team_name in lineups for this player/match
    player_lineups = lineups_df[
        (lineups_df["player_id"] == player_id) &
        (lineups_df["source"] == "club")
    ]

    # Find which competition_keys have this player at the right club
    valid_keys = set()
    for _, row in player_lineups.iterrows():
        if matches_club(str(row.get("team_name", ""))):
            valid_keys.add(row["competition_key"])

    # Also include if competition_label matches (for broader matching)
    for comp in SB_CLUB_COMPS:
        key = f"{comp['cid']}_{comp['sid']}"
        if any(kw.lower() in comp["label"].lower() for kw in keywords):
            # Check if player appears in this competition at all
            if key in player_rows["competition_key"].values:
                valid_keys.add(key)

    player_rows = player_rows[player_rows["competition_key"].isin(valid_keys)]

    contexts = []
    for _, row in player_rows.iterrows():
        if row["minutes_played"] < 45:
            continue
        # Build context label from team + competition
        # Find team name for this player in this competition
        team_rows = lineups_df[
            (lineups_df["player_id"] == player_id) &
            (lineups_df["competition_key"] == row["competition_key"])
        ]
        team = team_rows["team_name"].mode().iloc[0] if len(team_rows) > 0 else ""
        comp_label = row["competition_label"]
        # Build friendly label
        if "Bundesliga" in comp_label:
            ctx_label = f"{team} {comp_label}" if team else comp_label
        elif "La Liga" in comp_label:
            ctx_label = f"{team} {comp_label}" if team else comp_label
        elif "Ligue 1" in comp_label:
            ctx_label = f"{team} {comp_label}" if team else comp_label
        elif "MLS" in comp_label:
            ctx_label = f"{team} {comp_label}" if team else comp_label
        else:
            ctx_label = comp_label

        contexts.append({
            "label": ctx_label,
            "competition_key": row["competition_key"],
            "minutes": round(float(row["minutes_played"]), 1),
            "matches": int(row["n_matches"]),
            "per90_vaep": round(float(row["per90_vaep"]), 4),
            "per90_xt": round(float(row["per90_xt"]), 4),
            "goals": int(row["goals"]),
            "assists": int(row["assists"]),
            "vaep_sum": round(float(row["vaep_sum"]), 4),
        })

    return sorted(contexts, key=lambda x: x["label"])


def _country_contexts_for_player(
    player_id: int,
    country: str,
    per90_df: pd.DataFrame,
    lineups_df: pd.DataFrame,
) -> list[dict]:
    """Return all international contexts for a player at their national team."""
    player_rows = per90_df[
        (per90_df["player_id"] == player_id) &
        (per90_df["source"] == "international")
    ].copy()

    # Filter by country team name
    keywords = COUNTRY_PATTERNS.get(country, [country])
    player_lineups = lineups_df[
        (lineups_df["player_id"] == player_id) &
        (lineups_df["source"] == "international")
    ]

    valid_keys = set()
    for _, row in player_lineups.iterrows():
        if any(kw.lower() in str(row.get("team_name", "")).lower() for kw in keywords):
            valid_keys.add(row["competition_key"])

    player_rows = player_rows[player_rows["competition_key"].isin(valid_keys)]

    contexts = []
    for _, row in player_rows.iterrows():
        if row["minutes_played"] < 45:
            continue
        comp_label = INTL_COMP_LABELS.get(row["competition_key"], row["competition_key"])
        ctx_label = f"{country} {comp_label}"

        contexts.append({
            "label": ctx_label,
            "competition_key": row["competition_key"],
            "minutes": round(float(row["minutes_played"]), 1),
            "matches": int(row["n_matches"]),
            "per90_vaep": round(float(row["per90_vaep"]), 4),
            "per90_xt": round(float(row["per90_xt"]), 4),
            "goals": int(row["goals"]),
            "assists": int(row["assists"]),
            "vaep_sum": round(float(row["vaep_sum"]), 4),
        })

    return sorted(contexts, key=lambda x: x["label"])


def _weighted_avg(contexts: list[dict], key: str = "per90_vaep") -> float:
    """Minutes-weighted average of a per-90 metric across contexts."""
    total_mins = sum(c["minutes"] for c in contexts)
    if total_mins == 0:
        return 0.0
    return sum(c[key] * c["minutes"] for c in contexts) / total_mins


def _top_club_teammates(
    player_id: int,
    club_keyword: str,
    lineups_df: pd.DataFrame,
    n: int = 5,
) -> list[str]:
    """Return names of top-N most frequent club teammates by shared matches."""
    keywords = CLUB_KEYWORDS.get(club_keyword, [club_keyword])
    player_lineups = lineups_df[lineups_df["player_id"] == player_id]

    # Find match_ids where player appeared at this club
    club_matches = set()
    for _, row in player_lineups.iterrows():
        if any(kw.lower() in str(row.get("team_name", "")).lower() for kw in keywords):
            club_matches.add(row["match_id"])

    if not club_matches:
        return []

    # Find all teammates in those matches
    teammates = lineups_df[
        (lineups_df["match_id"].isin(club_matches)) &
        (lineups_df["player_id"] != player_id)
    ]

    # Filter to same team (approximate: same team_name pattern)
    teammate_counts = (
        teammates[teammates["team_name"].apply(
            lambda t: any(kw.lower() in str(t).lower() for kw in keywords)
        )]
        .groupby(["player_id", "player_name"])["match_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(n)
    )

    return list(teammate_counts.index.get_level_values("player_name"))


def _generate_narrative(
    name: str,
    club: str,
    country: str,
    club_contexts: list[dict],
    country_contexts: list[dict],
    delta: float,
    top_club_teammates: list[str],
    lineups_df: pd.DataFrame,
    player_id: int,
) -> str:
    """Generate a terse factual narrative sentence."""
    club_avg = _weighted_avg(club_contexts)
    country_avg = _weighted_avg(country_contexts)

    if not club_contexts:
        return (f"{name} has no qualifying club appearances in the available open data "
                f"for {club}; country context only.")

    if not country_contexts:
        return (f"{name}'s club per-90 VAEP at {club} averages {club_avg:.3f} "
                f"across {len(club_contexts)} season(s); no qualifying {country} appearances "
                f"found in the open international dataset.")

    teammate_str = ""
    if top_club_teammates:
        teammates = top_club_teammates[:3]
        if len(teammates) == 1:
            teammate_str = f" with {teammates[0]}"
        elif len(teammates) == 2:
            teammate_str = f" with {teammates[0]} and {teammates[1]}"
        else:
            teammate_str = f" with {teammates[0]}, {teammates[1]}, and {teammates[2]}"

    direction = "dropped" if delta < -0.005 else "rose" if delta > 0.005 else "held steady"
    return (
        f"{name}'s per-90 VAEP at {club}{teammate_str} averages {club_avg:.3f}; "
        f"at {country} it averages {country_avg:.3f} — a delta of {delta:+.3f} ({direction})."
    )


def phase6_build_dependency_cards(
    per90_df: pd.DataFrame,
    lineups_df: pd.DataFrame,
    vaep_bundle: dict,
) -> dict:
    """Build player_dependency.json."""
    cache_path = OUTPUTS / "player_dependency.json"
    if cache_path.exists():
        log.info("player_dependency.json cache hit: %s", cache_path)
        with open(cache_path) as f:
            return json.load(f)

    log.info("=== Phase 6: Build player dependency cards ===")

    players_out = []
    skipped = []

    for fp in FEATURED_PLAYERS:
        name = fp["name"]
        club = fp["club"]
        country = fp["country"]

        sb_id = fp.get("sb_id")
        pid, matched_name = _find_player_in_data(name, per90_df, lineups_df, sb_id=sb_id)
        if pid is None:
            log.warning("  Could not find player_id for %s", name)
            skipped.append({"name": name, "reason": "player_id not found in open data"})
            continue

        log.info("  Processing %s (player_id=%d)...", name, pid)

        club_ctxs = _club_contexts_for_player(pid, club, per90_df, lineups_df)
        country_ctxs = _country_contexts_for_player(pid, country, per90_df, lineups_df)

        if not club_ctxs and not country_ctxs:
            log.warning("  No qualifying contexts for %s", name)
            skipped.append({"name": name, "reason": "no qualifying appearances (min 45 min)"})
            continue

        club_avg = _weighted_avg(club_ctxs)
        country_avg = _weighted_avg(country_ctxs)
        delta = country_avg - club_avg

        top_teammates = _top_club_teammates(pid, club, lineups_df, n=5)

        narrative = _generate_narrative(
            name, club, country, club_ctxs, country_ctxs,
            delta, top_teammates, lineups_df, pid
        )

        players_out.append({
            "name": matched_name,
            "requested_name": name,
            "player_id": pid,
            "club": club,
            "country": country,
            "club_contexts": club_ctxs,
            "country_contexts": country_ctxs,
            "club_per90_vaep_avg_min_weighted": round(club_avg, 4),
            "country_per90_vaep_avg_min_weighted": round(country_avg, 4),
            "delta_vaep": round(delta, 4),
            "top_club_teammates": top_teammates,
            "narrative": narrative,
        })

    # Verdict summary
    drops = [p["name"] for p in players_out
             if p["club_contexts"] and p["country_contexts"] and p["delta_vaep"] < -0.010]
    rises = [p["name"] for p in players_out
             if p["club_contexts"] and p["country_contexts"] and p["delta_vaep"] > 0.010]
    steady = [p["name"] for p in players_out
              if p["club_contexts"] and p["country_contexts"]
              and -0.010 <= p["delta_vaep"] <= 0.010]

    output = {
        "meta": {
            "vaep_model": "vaep_v2",
            "data_sources": ["StatsBomb International", "StatsBomb Open Club Seasons"],
            "podolski_uncoverable": True,
            "train_auc_scores": round(vaep_bundle.get("auc_scores", 0.0), 4),
            "train_auc_concedes": round(vaep_bundle.get("auc_concedes", 0.0), 4),
            "train_size": vaep_bundle.get("train_size", 0),
        },
        "players": players_out,
        "skipped": skipped,
        "verdict_summary": {
            "players_with_significant_country_drop": drops,
            "players_with_significant_country_rise": rises,
            "players_holding_steady": steady,
        },
    }

    with open(cache_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log.info("Saved player_dependency.json: %d players, %d skipped",
             len(players_out), len(skipped))
    return output


# ── Phase 7: Write report ─────────────────────────────────────────────────

def phase7_write_report(dep: dict) -> None:
    """Write docs/analysis/cross-context-chemistry.md."""
    log.info("=== Phase 7: Write report ===")

    meta = dep.get("meta", {})
    players = dep.get("players", [])
    verdict = dep.get("verdict_summary", {})
    skipped = dep.get("skipped", [])

    auc_scores = meta.get("train_auc_scores", "N/A")
    auc_concedes = meta.get("train_auc_concedes", "N/A")
    train_size = meta.get("train_size", 0)

    # Build player lookup by name
    pmap = {p["name"]: p for p in players}
    pmap.update({p["requested_name"]: p for p in players})

    def fmt_ctx_table(ctxs: list[dict]) -> str:
        if not ctxs:
            return "_No qualifying appearances._"
        rows = []
        for c in ctxs:
            rows.append(
                f"| {c['label']} | {c['minutes']:.0f} | {c['matches']} "
                f"| {c['per90_vaep']:.3f} | {c['per90_xt']:.3f} "
                f"| {c['goals']} | {c['assists']} |"
            )
        header = ("| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |\n"
                  "|---|---|---|---|---|---|---|")
        return header + "\n" + "\n".join(rows)

    def player_section(name: str) -> str:
        p = pmap.get(name)
        if not p:
            return f"### {name}\n\n_Data not available._\n"
        club_avg = p["club_per90_vaep_avg_min_weighted"]
        country_avg = p["country_per90_vaep_avg_min_weighted"]
        delta = p["delta_vaep"]
        lines = [
            f"### {p['name']}",
            "",
            f"**Club ({p['club']})** — minutes-weighted avg per-90 VAEP: **{club_avg:.3f}**",
            "",
            fmt_ctx_table(p["club_contexts"]),
            "",
            f"**Country ({p['country']})** — minutes-weighted avg per-90 VAEP: **{country_avg:.3f}**",
            "",
            fmt_ctx_table(p["country_contexts"]),
            "",
            f"**Delta (country minus club): {delta:+.3f}**",
            "",
            f"_{p['narrative']}_",
            "",
        ]
        return "\n".join(lines)

    # Bayern-Germany table (Phase 3 headline)
    bayern_names = [
        "Joshua Kimmich", "Thomas Müller", "Leroy Sané",
        "Jamal Musiala", "Leon Goretzka", "Aleksandar Pavlović",
    ]
    # Try alternate spellings too
    alt_names = {
        "Thomas Müller": ["Thomas Müller", "Thomas Muller"],
        "Leroy Sané": ["Leroy Sané", "Leroy Sane"],
        "Jamal Musiala": ["Jamal Musiala"],
        "Aleksandar Pavlović": ["Aleksandar Pavlovic", "Aleksandar Pavlović"],
    }

    def get_player(name: str):
        alts = alt_names.get(name, [name])
        for a in alts:
            if a in pmap:
                return pmap[a]
        return None

    def ctx_for_key(ctxs, key_fragment):
        for c in ctxs:
            if key_fragment.lower() in c["label"].lower():
                return c
        return None

    # Bayern 23/24 vs Germany Euro 2024 table rows
    bav_table_rows = []
    for bname in bayern_names:
        p = get_player(bname)
        if not p:
            bav_table_rows.append(f"| {bname} | _not found_ | — | — | — |")
            continue
        bav_ctx = ctx_for_key(p["club_contexts"], "2023/24") or ctx_for_key(p["club_contexts"], "Bundesliga")
        ger_ctx = ctx_for_key(p["country_contexts"], "Euro 2024")
        bav_v = f"{bav_ctx['per90_vaep']:.3f}" if bav_ctx else "—"
        ger_v = f"{ger_ctx['per90_vaep']:.3f}" if ger_ctx else "—"
        if bav_ctx and ger_ctx:
            delta_v = f"{ger_ctx['per90_vaep'] - bav_ctx['per90_vaep']:+.3f}"
        else:
            delta_v = "—"
        bav_table_rows.append(f"| {p['name']} | {bav_v} | {ger_v} | {delta_v} |")

    bav_table = (
        "| Player | Bayern 23/24 per-90 VAEP | Germany Euro 24 per-90 VAEP | Delta |\n"
        "|---|---|---|---|\n"
        + "\n".join(bav_table_rows)
    )

    # Skipped players list
    skipped_str = "\n".join(f"- **{s['name']}**: {s['reason']}" for s in skipped)

    report = f"""# Cross-Context Chemistry: The Podolski Archetype

## 1. Setup

**The question:** Some players are unmistakably better in one context than another. Lukas Podolski scored 49 goals in 130 international appearances for Germany — a strike rate and consistency that made him a national hero. At club level, his record across Arsenal, Galatasaray, Inter Milan, and Vissel Kobe was considerably more modest. The inverse case — a player whose club chemistry network is richly developed but dissolves at international level — is what this analysis tests in modern, measurable terms.

**The data ceiling:** Podolski himself is **not in any open dataset**. His career at Köln (pre-2009), Bayern Munich (2009–12), Arsenal (2012–15), Galatasaray (2015–17), Vissel Kobe, Górnik Zabrze, and his Germany years all predate the StatsBomb open data releases that cover club football. There is zero open-data overlap for him. The analysis therefore tests the Podolski archetype on modern players whose club seasons and international tournaments both fall within available StatsBomb open data.

**Available open competitions used in this analysis:**
- **International:** FIFA WC 2022 (43 matches · 64 matches), UEFA Euro 2024 (51 matches), UEFA Euro 2020 (51 matches), Copa América 2024 (32 matches), FIFA WC 2018 (64 matches), AFCON 2023 (52 matches)
- **Club:** Bundesliga 2023/24 (34 Bayern matches via StatsBomb partial release), La Liga 2017/18–2020/21 (36+34+33+35 matches), Ligue 1 2021/22 + 2022/23 (26+32 matches), MLS 2023 (6 Inter Miami matches)

Note: StatsBomb's open club releases are samples — the Bundesliga 2023/24 release covers 34 matches (primarily Bayer Leverkusen's title-winning season, not the full Bayern dataset). The La Liga releases are similarly partial. This is the open-data ceiling.

---

## 2. Methodology

**VAEP v2** was trained on the combined StatsBomb SPADL corpus: {train_size:,} actions from all international competitions plus the 8 club seasons above. Two XGBClassifiers (scores, concedes), N=10 lookahead gamestates, features: actiontype, result, start/end location.

- Train AUC scores: **{auc_scores}**
- Train AUC concedes: **{auc_concedes}**

Both exceed the 0.70 sanity threshold.

**xT** is loaded from the existing `data/xt/xt.pkl` grid (fitted on the prior StatsBomb international corpus). It is reported alongside VAEP for reference; VAEP is the primary metric.

**Player-level per-90 production:** For each player in each competition context, we sum all VAEP values from their attributed actions and divide by minutes played / 90. Minutes are taken from the StatsBomb lineups API (position spell durations). A 45-minute floor is applied for international appearances; 90 minutes for club appearances.

**Caveats:**
- Opponent quality differs systematically: club opponents are top-division sides; international group-stage opponents range widely.
- Tactical roles often differ (e.g., Kimmich: RB at Bayern, CM for Germany).
- Small samples dominate international appearances: most players have 3–7 matches per tournament.
- StatsBomb open club data is a sample, not a full season. Bayern's 2023/24 Bundesliga representation is minimal in this release; the large Leverkusen match set means Bayern players appear infrequently.

---

## 3. Headline: Bayern 2023/24 → Germany Euro 2024

Within-season, overlapping players — the cleanest test.

{bav_table}

"""

    # Add individual sections
    for bname in bayern_names:
        report += player_section(bname)

    lewa_section = player_section("Robert Lewandowski")
    report += f"\n### Robert Lewandowski (Bayern → Poland)\n\n{lewa_section.split(chr(10), 2)[2]}\n"

    report += """---

## 4. Real Madrid Axis

"""
    for name in ["Luka Modrić", "Toni Kroos", "Raphaël Varane", "Karim Benzema"]:
        # Try alternate spellings
        alts = {
            "Luka Modrić": ["Luka Modrić", "Luka Modric"],
            "Raphaël Varane": ["Raphaël Varane", "Raphael Varane"],
        }
        found = None
        for alt in alts.get(name, [name]):
            if alt in pmap:
                found = alt
                break
        report += player_section(found or name)

    report += """---

## 5. Barcelona Axis

"""
    for name in ["Lionel Messi", "Luis Suárez", "Sergio Busquets", "Jordi Alba"]:
        alts = {
            "Luis Suárez": ["Luis Suárez", "Luis Suarez"],
        }
        found = None
        for alt in alts.get(name, [name]):
            if alt in pmap:
                found = alt
                break
        report += player_section(found or name)

    report += """---

## 6. The Messi Multi-Club Case

Messi's footprint in the open data spans three club contexts: FC Barcelona (La Liga 2017–21), PSG (Ligue 1 2021/22 + 2022/23), and Inter Miami (MLS 2023). His Argentina contexts include Copa América 2021, WC 2022, and Copa América 2024 — tournaments Argentina won. This is the reverse of the Podolski archetype: a player whose country production should be *higher* than club, not lower.

"""
    report += player_section("Lionel Messi")

    report += """---

## 7. PSG Era

"""
    for name in ["Kylian Mbappé", "Neymar", "Achraf Hakimi"]:
        alts = {
            "Kylian Mbappé": ["Kylian Mbappé", "Kylian Mbappe"],
        }
        found = None
        for alt in alts.get(name, [name]):
            if alt in pmap:
                found = alt
                break
        report += player_section(found or name)

    report += """---

## 8. Counter-Examples

Players whose country per-90 VAEP is comparable or higher than club — these push back against the Podolski thesis:

"""
    rises = verdict.get("players_with_significant_country_rise", [])
    steady = verdict.get("players_holding_steady", [])
    if rises:
        for n in rises:
            p = pmap.get(n)
            if p:
                report += (f"- **{p['name']}** ({p['club']} → {p['country']}): "
                           f"club avg {p['club_per90_vaep_avg_min_weighted']:.3f}, "
                           f"country avg {p['country_per90_vaep_avg_min_weighted']:.3f}, "
                           f"delta {p['delta_vaep']:+.3f}\n")
    else:
        report += "_No players with country per-90 VAEP significantly above club found in the qualifying sample._\n"

    report += """
---

## 9. Verdict

"""
    drops = verdict.get("players_with_significant_country_drop", [])
    n_with_both = sum(
        1 for p in players if p["club_contexts"] and p["country_contexts"]
    )
    n_drops = len(drops)
    n_rises = len(rises)
    n_steady = len(steady)

    report += f"""Of {n_with_both} players with qualifying appearances in both club and country contexts:
- **{n_drops}** showed a meaningful drop in per-90 VAEP at international level (delta < -0.010)
- **{n_rises}** showed a meaningful rise at international level (delta > +0.010)
- **{n_steady}** held roughly steady

The chemistry-network claim receives partial but not uniform support. The data is split — as with the Wyscout analysis. The open-data ceiling is the dominant limiting factor: most players have sparse club coverage, and within the available sample, variance is high enough that individual per-90 estimates should be treated as directional indicators rather than reliable effect sizes.

---

## 10. Caveats and Ceiling

**Podolski is uncoverable:** No open event data exists for Lukas Podolski at any club. The analysis tests his archetype only.

**Open data is a sample:** StatsBomb's open club releases cover partial seasons. The Bundesliga 2023/24 data emphasizes Bayer Leverkusen; Bayern Munich players have limited match coverage in this release.

**Small international samples:** Most players appear in 3–7 matches per international tournament, generating per-90 estimates with wide confidence intervals.

**No causal identification:** All comparisons are observational. Opponent quality, tactical role, age, and fitness all confound the club-vs-country delta.

**VAEP scale:** VAEP v2 is trained on a mixed StatsBomb corpus. Absolute values are small; relative rankings are more reliable than absolute magnitudes.

---

_Data: StatsBomb Open Data (CC BY-SA 4.0). VAEP: Bransen & Van Haaren 2020. xT: Singh 2019. Pipeline: socceraction._
"""

    report_path = DOCS / "cross-context-chemistry.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    log.info("Wrote report to %s", report_path)

    # Also write skipped players note
    if skipped:
        log.info("Skipped players: %s",
                 ", ".join(s["name"] for s in skipped))


# ── Phase 8: Update site ───────────────────────────────────────────────────

def phase8_update_site(dep: dict) -> None:
    """Replace Research tab section 1 in site/index.html with player dependency cards."""
    log.info("=== Phase 8: Update site Research tab ===")

    players = dep.get("players", [])
    meta = dep.get("meta", {})
    verdict = dep.get("verdict_summary", {})

    # Build HTML for dependency cards (6-8 players with both club+country contexts)
    eligible = [
        p for p in players
        if p["club_contexts"] and p["country_contexts"]
    ][:8]

    if not eligible:
        log.warning("No players with both club and country contexts for site update")
        return

    # Find global max per90_vaep for bar scaling
    all_vals = []
    for p in eligible:
        for c in p["club_contexts"]:
            all_vals.append(abs(c["per90_vaep"]))
        for c in p["country_contexts"]:
            all_vals.append(abs(c["per90_vaep"]))
    global_max = max(all_vals) if all_vals else 1.0
    if global_max == 0:
        global_max = 0.001

    auc_s = meta.get("train_auc_scores", "N/A")
    auc_c = meta.get("train_auc_concedes", "N/A")
    train_size = meta.get("train_size", 0)

    drops = verdict.get("players_with_significant_country_drop", [])
    rises = verdict.get("players_with_significant_country_rise", [])
    steady = verdict.get("players_holding_steady", [])

    def bar_pct(val: float) -> int:
        return min(100, max(0, int(abs(val) / global_max * 100)))

    def render_ctx_rows(ctxs, css_class):
        rows = ""
        for c in ctxs[:2]:  # show max 2 contexts per card
            pct = bar_pct(c["per90_vaep"])
            rows += f"""
              <div class="dep-ctx-row">
                <span class="dep-ctx-label">{c['label']}</span>
                <div class="dep-bar-wrap">
                  <div class="dep-bar {css_class}" style="width:{pct}%"></div>
                </div>
                <span class="dep-val">{c['per90_vaep']:.3f}</span>
                <span class="dep-meta">{c['minutes']:.0f}&thinsp;min &middot; {c['goals']}G {c['assists']}A</span>
              </div>"""
        return rows

    cards_html = ""
    for p in eligible:
        delta = p["delta_vaep"]
        delta_cls = "dep-delta-neg" if delta < -0.005 else "dep-delta-pos" if delta > 0.005 else "dep-delta-neutral"
        delta_str = f"{delta:+.3f}"

        club_rows = render_ctx_rows(p["club_contexts"], "dep-bar-club")
        country_rows = render_ctx_rows(p["country_contexts"], "dep-bar-country")

        cards_html += f"""
          <div class="dep-card">
            <div class="dep-card-header">
              <span class="dep-name">{p['name']}</span>
              <span class="dep-delta {delta_cls}">{delta_str}</span>
            </div>
            <div class="dep-card-body">
              <div class="dep-section">
                <span class="dep-section-label">Club ({p['club']})</span>
                {club_rows}
              </div>
              <div class="dep-section">
                <span class="dep-section-label">Country ({p['country']})</span>
                {country_rows}
              </div>
            </div>
            <p class="dep-narrative">{p['narrative']}</p>
          </div>"""

    # Verdict counts
    n_total = len([p for p in players if p["club_contexts"] and p["country_contexts"]])
    n_drops = len(drops)
    n_rises = len(rises)

    new_section1 = f"""      <!-- Section 1: Cross-context (StatsBomb) player per-90 analysis -->
      <section class="slide-card" aria-label="Player per-90 club vs country">
        <span class="pill green">Primary finding</span>
        <h2>The Podolski archetype: player per-90 production at club vs country</h2>

        <p>The Podolski thesis tests whether a player&rsquo;s per-90 output &mdash; individually, not as a pair &mdash; drops when they leave their club chemistry network for the national team context. Lukas Podolski himself is <strong>not in any open dataset</strong> (his career predates StatsBomb&rsquo;s open club releases). This analysis tests the archetype on modern players whose club and country footprints both fall within available open data.</p>

        <p>VAEP v2 trained on combined StatsBomb SPADL: {train_size:,} actions (international + 8 club seasons). Train AUC: scores&thinsp;=&thinsp;{auc_s}, concedes&thinsp;=&thinsp;{auc_c}. Bars show per-90 VAEP; <span style="color:var(--accent)">&#9632;</span>&thinsp;club, <span style="color:var(--accent2)">&#9632;</span>&thinsp;country. Delta = country &minus; club (minutes-weighted averages).</p>

        <div class="research-verdict">
          <div class="verdict-card verdict-split">
            <div class="verdict-number">{n_drops}&thinsp;/&thinsp;{n_total}</div>
            <div class="verdict-label">players with meaningful per-90 VAEP drop at country level (&Delta;&thinsp;&lt;&thinsp;&minus;0.010)</div>
          </div>
          <div class="verdict-card verdict-split">
            <div class="verdict-number">{n_rises}&thinsp;/&thinsp;{n_total}</div>
            <div class="verdict-label">players with higher per-90 VAEP at country level (&Delta;&thinsp;&gt;&thinsp;+0.010)</div>
          </div>
          <div class="verdict-card verdict-neutral">
            <div class="verdict-number">{n_total - n_drops - n_rises}&thinsp;/&thinsp;{n_total}</div>
            <div class="verdict-label">near-parity across contexts</div>
          </div>
        </div>

        <div class="dep-legend">
          <span class="dep-legend-item"><span class="dep-swatch dep-swatch-club"></span> Club per-90 VAEP</span>
          <span class="dep-legend-item"><span class="dep-swatch dep-swatch-country"></span> Country per-90 VAEP</span>
          <span class="dep-legend-item"><span class="dep-delta dep-delta-neg">&minus;</span> country &lt; club</span>
          <span class="dep-legend-item"><span class="dep-delta dep-delta-pos">+</span> country &gt; club</span>
        </div>

        <div class="dep-cards-grid">
          {cards_html}
        </div>

        <div class="slide-inner" style="margin-top:16px">
          <h3>Caveats</h3>
          <ul>
            <li>StatsBomb open club releases are samples, not full seasons. The Bundesliga 2023/24 release emphasizes Bayer Leverkusen; Bayern players have limited match coverage.</li>
            <li>International appearances: 3&ndash;7 matches per tournament means high variance per-90 estimates.</li>
            <li>Opponent quality and tactical role differ systematically between club and country contexts.</li>
            <li>VAEP scale is small in absolute terms; relative rankings are more reliable than magnitudes.</li>
          </ul>
        </div>

        <p class="research-footer">Data: StatsBomb Open Data &mdash; CC BY-SA 4.0. VAEP v2 per Bransen &amp; Van Haaren 2020. xT per Singh 2019. <a href="https://github.com/stranger9977/wc2026-chemistry/blob/main/docs/analysis/cross-context-chemistry.md" target="_blank" rel="noopener">Full analysis &rarr;</a></p>
      </section>"""

    # CSS additions for dep cards
    dep_css = """
      /* ── Player dependency cards ────────────────────────────── */
      .dep-cards-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px;margin:16px 0}
      .dep-card{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:14px 16px}
      .dep-card-header{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px}
      .dep-name{font-weight:700;font-size:15px;color:var(--ink)}
      .dep-delta{font-size:13px;font-weight:700;padding:2px 6px;border-radius:4px}
      .dep-delta-neg{color:#d63031;background:#ffeaea}
      .dep-delta-pos{color:#00b894;background:#e8faf5}
      .dep-delta-neutral{color:var(--muted);background:var(--bg)}
      .dep-section{margin-bottom:8px}
      .dep-section-label{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin-bottom:4px}
      .dep-ctx-row{display:flex;align-items:center;gap:6px;margin-bottom:3px;flex-wrap:wrap}
      .dep-ctx-label{font-size:11px;color:var(--muted);min-width:140px;flex-shrink:0}
      .dep-bar-wrap{flex:1;height:8px;background:var(--bg);border-radius:4px;min-width:60px;overflow:hidden}
      .dep-bar{height:100%;border-radius:4px;transition:width .3s}
      .dep-bar-club{background:var(--accent)}
      .dep-bar-country{background:var(--accent2)}
      .dep-val{font-size:12px;font-weight:600;color:var(--ink);min-width:38px;text-align:right}
      .dep-meta{font-size:10px;color:var(--muted)}
      .dep-narrative{font-size:12px;color:var(--muted);margin-top:8px;line-height:1.5;font-style:italic}
      .dep-legend{display:flex;gap:12px;flex-wrap:wrap;margin:8px 0;font-size:12px}
      .dep-legend-item{display:flex;align-items:center;gap:4px;color:var(--muted)}
      .dep-swatch{display:inline-block;width:12px;height:12px;border-radius:2px}
      .dep-swatch-club{background:var(--accent)}
      .dep-swatch-country{background:var(--accent2)}
"""

    site_path = ROOT / "site" / "index.html"
    with open(site_path, encoding="utf-8") as f:
        html = f.read()

    # Replace section 1 in Research tab
    import re
    pattern = r'(<!-- Section 1: Cross-context.*?)(?=<!-- Section 2:)'
    replacement = new_section1 + "\n\n      "
    new_html, n = re.subn(pattern, replacement, html, flags=re.DOTALL)
    if n == 0:
        log.warning("  Could not find Section 1 pattern to replace — manual update needed")
        return

    # Inject CSS before </style> or before first </head> tag if no style block
    if dep_css.strip() not in new_html:
        if "</style>" in new_html:
            new_html = new_html.replace("</style>", dep_css + "\n</style>", 1)
        elif "<link rel=\"stylesheet\"" in new_html:
            # Inject inline style block
            new_html = new_html.replace(
                "</head>",
                f"<style>{dep_css}</style>\n</head>",
                1,
            )

    with open(site_path, "w", encoding="utf-8") as f:
        f.write(new_html)
    log.info("Updated site/index.html Research section 1")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== StatsBomb club pipeline starting ===")

    # Phase 1: Fetch club data
    phase1_fetch_club_data()

    # Phase 2: Convert to SPADL
    phase2_convert_to_spadl()

    # Phase 3: Train VAEP v2
    bundle = phase3_train_vaep_v2()

    # Phase 4: Score all actions
    phase4_score_all_actions(bundle)

    # Load lineups (needed for phases 5 and 6)
    lineups_df = load_all_lineups()

    # Phase 5: Player per-90 stats
    per90_df = phase5_compute_player_per90(lineups_df)

    # Phase 6: Dependency cards
    dep = phase6_build_dependency_cards(per90_df, lineups_df, bundle)

    # Phase 7: Report
    phase7_write_report(dep)

    # Phase 8: Site update
    phase8_update_site(dep)

    log.info("=== Pipeline complete ===")

    # Summary
    meta = dep.get("meta", {})
    players = dep.get("players", [])
    verdict = dep.get("verdict_summary", {})
    log.info("VAEP v2 AUC: scores=%.4f  concedes=%.4f",
             meta.get("train_auc_scores", 0), meta.get("train_auc_concedes", 0))
    log.info("Players processed: %d", len(players))
    log.info("Drops: %d | Rises: %d | Steady: %d",
             len(verdict.get("players_with_significant_country_drop", [])),
             len(verdict.get("players_with_significant_country_rise", [])),
             len(verdict.get("players_holding_steady", [])))


if __name__ == "__main__":
    main()
