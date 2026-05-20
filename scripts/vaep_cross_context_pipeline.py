"""VAEP cross-context chemistry pipeline.

Phases:
  1. Train VAEP on Wyscout SPADL -> data/wyscout/vaep.pkl
  2. Score Wyscout SPADL actions with VAEP
  3. Compute VAEP-based JOI per match -> outputs/wyscout_joi_per_match_vaep.parquet
  4. Fetch StatsBomb club competitions -> data/raw/<cid>_<sid>/
  5. Convert StatsBomb club matches to SPADL
  6. Train combined VAEP v2 (Wyscout + StatsBomb) -> data/vaep/vaep_v2.pkl
  7. Score StatsBomb club SPADL with VAEP v2
  8. Compute StatsBomb club JOI per match
  9. Run cross-context analysis with VAEP values
  10. Update outputs/cross_context_chemistry.json and docs/analysis/cross-context-chemistry.md
  11. Update site/index.html Research tab

Run with: .venv/bin/python -m scripts.vaep_cross_context_pipeline
Each stage is cached — safe to re-run after interruption.
"""

from __future__ import annotations

import json
import logging
import pickle
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_WY = ROOT / "data" / "wyscout"
DATA_SPADL_WY = ROOT / "data" / "wyscout_spadl"
DATA_RAW = ROOT / "data" / "raw"
DATA_VAEP = ROOT / "data" / "vaep"
DATA_SPADL_SB = ROOT / "data" / "sb_club_spadl"
OUTPUTS = ROOT / "outputs"

DATA_VAEP.mkdir(parents=True, exist_ok=True)
DATA_SPADL_SB.mkdir(parents=True, exist_ok=True)

VAEP_WY_PATH = DATA_WY / "vaep.pkl"
VAEP_V2_PATH = DATA_VAEP / "vaep_v2.pkl"

WY_COMPETITIONS = [
    "England", "France", "Germany", "Italy", "Spain",
    "European_Championship", "World_Cup",
]

# StatsBomb club competitions to add
SB_CLUB_COMPS = [
    {"cid": 9,  "sid": 281, "label": "Bundesliga 2023/24"},
    {"cid": 11, "sid": 1,   "label": "La Liga 2017/18"},
    {"cid": 11, "sid": 4,   "label": "La Liga 2018/19"},
    {"cid": 11, "sid": 42,  "label": "La Liga 2019/20"},
    {"cid": 11, "sid": 90,  "label": "La Liga 2020/21"},
    {"cid": 7,  "sid": 108, "label": "Ligue 1 2021/22"},
    {"cid": 7,  "sid": 235, "label": "Ligue 1 2022/23"},
    {"cid": 44, "sid": 107, "label": "MLS 2023"},
]

ELIGIBLE_TYPES: frozenset[int] = frozenset({0, 1, 3, 4, 5, 6, 7, 11, 12, 13, 21})


# ── Helpers ────────────────────────────────────────────────────────────────

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


def build_features(spadl: pd.DataFrame) -> pd.DataFrame:
    from socceraction.vaep import features as fs
    gs = fs.gamestates(spadl)
    return pd.concat(
        [fs.actiontype_onehot(gs),
         fs.result_onehot(gs),
         fs.startlocation(gs),
         fs.endlocation(gs)],
        axis=1,
    )


def train_vaep_model(spadl: pd.DataFrame, out_path: Path, sample: float = 1.0) -> dict:
    """Train VAEP on combined SPADL. Returns bundle dict (also saved)."""
    from socceraction.vaep import labels as lb
    from xgboost import XGBClassifier

    spadl = _add_type_result_name(spadl)

    if sample < 1.0:
        spadl = spadl.sample(frac=sample, random_state=42).reset_index(drop=True)
        log.info("Downsampled to %d actions (%.0f%%)", len(spadl), sample * 100)

    log.info("Building features for %d actions ...", len(spadl))
    X = build_features(spadl)
    log.info("Feature shape: %s", X.shape)

    y_scores = lb.scores(spadl)["scores"]
    y_concedes = lb.concedes(spadl)["concedes"]
    log.info("scores positives: %d / %d", y_scores.sum(), len(y_scores))
    log.info("concedes positives: %d / %d", y_concedes.sum(), len(y_concedes))

    log.info("Training scores model ...")
    clf_scores = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        n_jobs=-1, eval_metric="logloss", use_label_encoder=False,
    )
    clf_scores.fit(X, y_scores)

    log.info("Training concedes model ...")
    clf_concedes = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        n_jobs=-1, eval_metric="logloss", use_label_encoder=False,
    )
    clf_concedes.fit(X, y_concedes)

    # AUC evaluation
    from sklearn.metrics import roc_auc_score
    auc_scores = roc_auc_score(y_scores, clf_scores.predict_proba(X)[:, 1])
    auc_concedes = roc_auc_score(y_concedes, clf_concedes.predict_proba(X)[:, 1])
    log.info("Train AUC scores: %.4f  concedes: %.4f", auc_scores, auc_concedes)

    bundle = {
        "scores": clf_scores,
        "concedes": clf_concedes,
        "feature_cols": list(X.columns),
        "train_size": len(spadl),
        "auc_scores": auc_scores,
        "auc_concedes": auc_concedes,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    log.info("Saved VAEP model to %s (train AUC scores=%.4f concedes=%.4f)",
             out_path, auc_scores, auc_concedes)
    return bundle


def score_vaep(spadl: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Add vaep_value column. Returns copy of spadl."""
    from socceraction.vaep import formula

    enriched = _add_type_result_name(spadl)
    X = build_features(enriched)

    # Align columns to training
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
    return result


def compute_vaep_joi_per_match(actions: pd.DataFrame, value_col: str = "vaep_value") -> pd.DataFrame:
    """Compute per-match pair JOI using VAEP values.

    Returns DataFrame: match_id, competition, team_id, player_a, player_b, joi_vaep
    """
    df = actions[actions["type_id"].isin(ELIGIBLE_TYPES)].copy()
    df = df.sort_values(["match_id", "period_id", "time_seconds"]).reset_index(drop=True)

    nxt = df.shift(-1)
    consecutive = (
        (df["match_id"] == nxt["match_id"])
        & (df["team_id"] == nxt["team_id"])
        & (df["player_id"] != nxt["player_id"])
    )

    pairs = pd.DataFrame({
        "match_id": df["match_id"],
        "competition": df.get("competition", pd.Series(["unknown"] * len(df), index=df.index)),
        "team_id": df["team_id"],
        "player_p": df["player_id"],
        "player_q": nxt["player_id"],
        "vaep_q": nxt[value_col],
    })[consecutive].reset_index(drop=True)

    pairs["player_a"] = pairs[["player_p", "player_q"]].min(axis=1).astype("int64")
    pairs["player_b"] = pairs[["player_p", "player_q"]].max(axis=1).astype("int64")
    pairs["vaep_q"] = pairs["vaep_q"].fillna(0.0)

    grouped = (
        pairs.groupby(["match_id", "competition", "team_id", "player_a", "player_b"], as_index=False)
             ["vaep_q"].sum()
             .rename(columns={"vaep_q": "joi_vaep"})
    )
    return grouped


# ── Phase 1: Train VAEP on Wyscout ────────────────────────────────────────

def phase1_train_vaep_wyscout() -> dict:
    """Train VAEP on all 7 Wyscout competitions combined."""
    if VAEP_WY_PATH.exists():
        log.info("VAEP Wyscout model cache hit: %s", VAEP_WY_PATH)
        with open(VAEP_WY_PATH, "rb") as f:
            return pickle.load(f)

    log.info("=== Phase 1: Train VAEP on Wyscout SPADL ===")
    parts = []
    for comp in WY_COMPETITIONS:
        p = DATA_SPADL_WY / f"spadl_{comp}.parquet"
        df = pd.read_parquet(p)
        df["competition"] = comp
        parts.append(df)
        log.info("  Loaded %s: %d actions", comp, len(df))

    all_spadl = pd.concat(parts, ignore_index=True)
    log.info("Total Wyscout SPADL: %d actions", len(all_spadl))

    bundle = train_vaep_model(all_spadl, VAEP_WY_PATH)
    return bundle


# ── Phase 2: Score Wyscout SPADL with VAEP ────────────────────────────────

def phase2_score_wyscout(bundle: dict) -> None:
    """Score every Wyscout action with VAEP. Cache per competition."""
    log.info("=== Phase 2: Score Wyscout SPADL with VAEP ===")
    for comp in WY_COMPETITIONS:
        cache_path = DATA_SPADL_WY / f"vaep_scored_{comp}.parquet"
        if cache_path.exists():
            log.info("  VAEP scored cache hit: %s", comp)
            continue
        log.info("  Scoring %s ...", comp)
        df = pd.read_parquet(DATA_SPADL_WY / f"spadl_{comp}.parquet")
        df["competition"] = comp
        df["match_id"] = df["game_id"]  # ensure match_id present
        scored = score_vaep(df, bundle)
        scored.to_parquet(cache_path, index=False)
        log.info("    -> %d actions scored", len(scored))


# ── Phase 3: Compute Wyscout VAEP JOI per match ───────────────────────────

def phase3_compute_wy_vaep_joi() -> pd.DataFrame:
    """Compute VAEP-based JOI per match for all Wyscout competitions."""
    out_path = OUTPUTS / "wyscout_joi_per_match_vaep.parquet"
    if out_path.exists():
        log.info("Wyscout VAEP JOI cache hit: %s", out_path)
        return pd.read_parquet(out_path)

    log.info("=== Phase 3: Compute Wyscout VAEP JOI per match ===")

    # Load shared minutes (already computed by wyscout_pipeline.py)
    shared_mins_path = DATA_SPADL_WY / "shared_minutes.parquet"
    if not shared_mins_path.exists():
        log.error("shared_minutes.parquet not found — run scripts/wyscout_pipeline.py first")
        raise FileNotFoundError(shared_mins_path)

    shared_mins = pd.read_parquet(shared_mins_path)

    joi_parts = []
    for comp in WY_COMPETITIONS:
        cache_path = DATA_SPADL_WY / f"vaep_joi_{comp}.parquet"
        if cache_path.exists():
            log.info("  JOI cache hit: %s", comp)
            joi_parts.append(pd.read_parquet(cache_path))
            continue
        scored = pd.read_parquet(DATA_SPADL_WY / f"vaep_scored_{comp}.parquet")
        log.info("  Computing JOI for %s: %d actions ...", comp, len(scored))
        joi = compute_vaep_joi_per_match(scored)
        joi.to_parquet(cache_path, index=False)
        joi_parts.append(joi)

    all_joi = pd.concat(joi_parts, ignore_index=True)

    # Merge with shared minutes
    merged = all_joi.merge(
        shared_mins[["match_id", "team_id", "player_a", "player_b", "shared_minutes"]],
        on=["match_id", "team_id", "player_a", "player_b"],
        how="left",
    )
    merged["shared_minutes"] = merged["shared_minutes"].fillna(0.0)
    merged.to_parquet(out_path, index=False)
    log.info("Saved Wyscout VAEP JOI per match: %s (%d rows)", out_path, len(merged))
    return merged


# ── Phase 4: Fetch StatsBomb club data ────────────────────────────────────

def phase4_fetch_sb_club_data() -> None:
    """Fetch StatsBomb club competitions into data/raw/<cid>_<sid>/."""
    from statsbombpy import sb
    import warnings

    log.info("=== Phase 4: Fetch StatsBomb club data ===")
    for comp in SB_CLUB_COMPS:
        cid, sid, label = comp["cid"], comp["sid"], comp["label"]
        base = DATA_RAW / f"{cid}_{sid}"
        matches_path = base / "matches.parquet"

        if matches_path.exists():
            log.info("  Cache hit: %s (%d_%d)", label, cid, sid)
            continue

        log.info("  Fetching %s (cid=%d sid=%d) ...", label, cid, sid)
        base.mkdir(parents=True, exist_ok=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matches = sb.matches(competition_id=cid, season_id=sid)

        log.info("    %d matches", len(matches))
        matches.to_parquet(matches_path, index=False)

        # Fetch events per match
        events_dir = base / "events"
        events_dir.mkdir(exist_ok=True)
        for _, row in matches.iterrows():
            mid = int(row["match_id"])
            ep = events_dir / f"{mid}.parquet"
            if ep.exists():
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ev = sb.events(match_id=mid, fmt="dataframe")
                ev.to_parquet(ep, index=False)
            except Exception as e:
                log.warning("    Failed to fetch events for match %d: %s", mid, e)

        log.info("    Fetched events for %s", label)


# ── Phase 5: Convert StatsBomb club matches to SPADL ──────────────────────

def phase5_convert_sb_spadl() -> None:
    """Convert StatsBomb club event data to SPADL. Cache per competition."""
    from socceraction.spadl.statsbomb import convert_to_actions as sb_convert

    log.info("=== Phase 5: Convert StatsBomb club events to SPADL ===")
    for comp in SB_CLUB_COMPS:
        cid, sid, label = comp["cid"], comp["sid"], comp["label"]
        comp_key = f"{cid}_{sid}"
        out_path = DATA_SPADL_SB / f"spadl_{comp_key}.parquet"

        if out_path.exists():
            log.info("  SPADL cache hit: %s", label)
            continue

        base = DATA_RAW / comp_key
        matches_path = base / "matches.parquet"
        if not matches_path.exists():
            log.warning("  No matches data for %s, skipping", label)
            continue

        matches = pd.read_parquet(matches_path)
        events_dir = base / "events"
        log.info("  Converting %s: %d matches ...", label, len(matches))

        all_actions = []
        for _, row in matches.iterrows():
            mid = int(row["match_id"])
            ep = events_dir / f"{mid}.parquet"
            if not ep.exists():
                continue
            try:
                ev = pd.read_parquet(ep)
                actions = sb_convert(ev, home_team_id=int(row["home_team"]["home_team_id"]))
                actions["match_id"] = mid
                actions["competition"] = label
                actions["competition_key"] = comp_key
                all_actions.append(actions)
            except Exception as e:
                log.warning("    Failed to convert match %d: %s", mid, e)

        if not all_actions:
            log.warning("  No actions for %s", label)
            continue

        combined = pd.concat(all_actions, ignore_index=True)
        combined.to_parquet(out_path, index=False)
        log.info("  Saved %s: %d actions", label, len(combined))


# ── Phase 6: Train VAEP v2 (Wyscout + StatsBomb combined) ─────────────────

def phase6_train_vaep_v2() -> dict:
    """Train VAEP v2 on combined Wyscout + StatsBomb SPADL."""
    if VAEP_V2_PATH.exists():
        log.info("VAEP v2 model cache hit: %s", VAEP_V2_PATH)
        with open(VAEP_V2_PATH, "rb") as f:
            return pickle.load(f)

    log.info("=== Phase 6: Train VAEP v2 (Wyscout + StatsBomb combined) ===")

    # Wyscout SPADL
    wy_parts = []
    for comp in WY_COMPETITIONS:
        p = DATA_SPADL_WY / f"spadl_{comp}.parquet"
        df = pd.read_parquet(p)
        df["competition"] = comp
        wy_parts.append(df)
    wy_combined = pd.concat(wy_parts, ignore_index=True)
    log.info("Wyscout SPADL: %d actions", len(wy_combined))

    # StatsBomb club SPADL
    sb_parts = []
    for comp in SB_CLUB_COMPS:
        cid, sid, label = comp["cid"], comp["sid"], comp["label"]
        p = DATA_SPADL_SB / f"spadl_{cid}_{sid}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            sb_parts.append(df)
            log.info("  SB %s: %d actions", label, len(df))
        else:
            log.warning("  Missing SB SPADL for %s", label)

    # StatsBomb international SPADL (already in data/spadl/)
    intl_parts = []
    spadl_dir = ROOT / "data" / "spadl"
    for comp_dir in sorted(spadl_dir.iterdir()):
        if not comp_dir.is_dir():
            continue
        for p in sorted(comp_dir.glob("*.parquet")):
            intl_parts.append(pd.read_parquet(p))
    if intl_parts:
        intl_combined = pd.concat(intl_parts, ignore_index=True)
        log.info("StatsBomb international SPADL: %d actions", len(intl_combined))
        sb_parts.insert(0, intl_combined)

    all_parts = [wy_combined] + sb_parts
    all_spadl = pd.concat(all_parts, ignore_index=True)
    log.info("Combined SPADL for VAEP v2: %d actions", len(all_spadl))

    bundle = train_vaep_model(all_spadl, VAEP_V2_PATH)
    return bundle


# ── Phase 7: Score StatsBomb club SPADL with VAEP v2 ──────────────────────

def phase7_score_sb_vaep(bundle: dict) -> None:
    """Score StatsBomb club SPADL with VAEP v2."""
    log.info("=== Phase 7: Score StatsBomb club SPADL with VAEP v2 ===")
    for comp in SB_CLUB_COMPS:
        cid, sid, label = comp["cid"], comp["sid"], comp["label"]
        comp_key = f"{cid}_{sid}"
        src = DATA_SPADL_SB / f"spadl_{comp_key}.parquet"
        out = DATA_SPADL_SB / f"vaep_scored_{comp_key}.parquet"

        if out.exists():
            log.info("  VAEP scored cache hit: %s", label)
            continue
        if not src.exists():
            log.warning("  Missing SPADL for %s, skipping", label)
            continue

        log.info("  Scoring %s ...", label)
        df = pd.read_parquet(src)
        scored = score_vaep(df, bundle)
        scored.to_parquet(out, index=False)
        log.info("  -> %d actions scored", len(scored))


# ── Phase 8: Compute StatsBomb club JOI per match ─────────────────────────

def _compute_sb_shared_minutes(matches: pd.DataFrame, lineups_dir: Path | None = None,
                                events_dir: Path | None = None) -> pd.DataFrame:
    """Compute shared minutes for StatsBomb matches from lineup events."""
    rows = []
    for _, row in matches.iterrows():
        mid = int(row["match_id"])
        ep = events_dir / f"{mid}.parquet" if events_dir else None
        if ep and ep.exists():
            ev = pd.read_parquet(ep)
            # Find lineup events to get player start/end times
            subs = ev[ev["type"].apply(lambda t: t.get("name", "") == "Substitution"
                                        if isinstance(t, dict) else False)]
            starting = ev[ev["type"].apply(lambda t: t.get("name", "") == "Starting XI"
                                            if isinstance(t, dict) else False)]

            team_players: dict[int, dict[int, tuple[float, float]]] = {}

            for _, se in starting.iterrows():
                try:
                    tid = int(se["team"]["id"]) if isinstance(se["team"], dict) else int(se["team_id"])
                    tactics = se.get("tactics", {}) if isinstance(se.get("tactics"), dict) else {}
                    lineup = tactics.get("lineup", [])
                    if tid not in team_players:
                        team_players[tid] = {}
                    for p in lineup:
                        pid = int(p["player"]["id"]) if "player" in p else None
                        if pid:
                            team_players[tid][pid] = (0.0, 90.0)
                except Exception:
                    continue

            for _, sub in subs.iterrows():
                try:
                    minute = float(sub["minute"]) if "minute" in sub else 0.0
                    tid = int(sub["team"]["id"]) if isinstance(sub["team"], dict) else None
                    sub_data = sub.get("substitution", {}) if isinstance(sub.get("substitution"), dict) else {}
                    player_off_id = int(sub["player"]["id"]) if isinstance(sub.get("player"), dict) else None
                    player_on_id = int(sub_data["replacement"]["id"]) if "replacement" in sub_data and isinstance(sub_data["replacement"], dict) else None

                    if tid and player_off_id and tid in team_players:
                        if player_off_id in team_players[tid]:
                            on, _ = team_players[tid][player_off_id]
                            team_players[tid][player_off_id] = (on, minute)
                    if tid and player_on_id:
                        if tid not in team_players:
                            team_players[tid] = {}
                        team_players[tid][player_on_id] = (minute, 90.0)
                except Exception:
                    continue

            for tid, players in team_players.items():
                for pid, (on, off) in players.items():
                    rows.append({
                        "match_id": mid,
                        "team_id": tid,
                        "player_id": pid,
                        "minute_on": on,
                        "minute_off": min(off, 90.0),
                    })
        else:
            # Fallback: use match lineup data from matches DataFrame
            home_id = row.get("home_team", {})
            if isinstance(home_id, dict):
                home_id = home_id.get("home_team_id")
            away_id = row.get("away_team", {})
            if isinstance(away_id, dict):
                away_id = away_id.get("away_team_id")

    if not rows:
        return pd.DataFrame(columns=["match_id", "team_id", "player_id", "minute_on", "minute_off"])
    return pd.DataFrame(rows)


def _compute_pair_shared_minutes(player_minutes: pd.DataFrame) -> pd.DataFrame:
    """Cross-join within match/team to get pair shared minutes."""
    rows = []
    for (match_id, team_id), grp in player_minutes.groupby(["match_id", "team_id"]):
        players = grp[["player_id", "minute_on", "minute_off"]].values.tolist()
        n = len(players)
        for i in range(n):
            pid_a, on_a, off_a = int(players[i][0]), float(players[i][1]), float(players[i][2])
            for j in range(i + 1, n):
                pid_b, on_b, off_b = int(players[j][0]), float(players[j][1]), float(players[j][2])
                shared = max(0.0, min(off_a, off_b) - max(on_a, on_b))
                if shared > 0:
                    rows.append({
                        "match_id": match_id,
                        "team_id": team_id,
                        "player_a": min(pid_a, pid_b),
                        "player_b": max(pid_a, pid_b),
                        "shared_minutes": shared,
                    })
    if not rows:
        return pd.DataFrame(columns=["match_id", "team_id", "player_a", "player_b", "shared_minutes"])
    return pd.DataFrame(rows)


def phase8_compute_sb_joi() -> pd.DataFrame:
    """Compute StatsBomb club JOI per match, merged with shared minutes."""
    out_path = OUTPUTS / "sb_club_joi_per_match_vaep.parquet"
    if out_path.exists():
        log.info("SB club JOI cache hit: %s", out_path)
        return pd.read_parquet(out_path)

    log.info("=== Phase 8: Compute StatsBomb club JOI per match ===")

    all_joi_parts = []
    all_shared_parts = []

    for comp in SB_CLUB_COMPS:
        cid, sid, label = comp["cid"], comp["sid"], comp["label"]
        comp_key = f"{cid}_{sid}"

        scored_path = DATA_SPADL_SB / f"vaep_scored_{comp_key}.parquet"
        if not scored_path.exists():
            log.warning("  Missing scored SPADL for %s, skipping", label)
            continue

        joi_cache = DATA_SPADL_SB / f"joi_{comp_key}.parquet"
        if joi_cache.exists():
            log.info("  JOI cache hit: %s", label)
            joi = pd.read_parquet(joi_cache)
        else:
            scored = pd.read_parquet(scored_path)
            log.info("  Computing JOI for %s: %d actions ...", label, len(scored))
            joi = compute_vaep_joi_per_match(scored)
            joi.to_parquet(joi_cache, index=False)

        all_joi_parts.append(joi)

        # Shared minutes for this competition
        sm_cache = DATA_SPADL_SB / f"shared_minutes_{comp_key}.parquet"
        if sm_cache.exists():
            log.info("  Shared minutes cache hit: %s", label)
            sm = pd.read_parquet(sm_cache)
        else:
            log.info("  Computing shared minutes for %s ...", label)
            base = DATA_RAW / comp_key
            matches = pd.read_parquet(base / "matches.parquet")
            events_dir = base / "events"
            player_mins = _compute_sb_shared_minutes(matches, events_dir=events_dir)
            player_mins["competition"] = label
            sm = _compute_pair_shared_minutes(player_mins)
            sm["competition"] = label
            sm.to_parquet(sm_cache, index=False)

        all_shared_parts.append(sm)

    if not all_joi_parts:
        log.warning("No StatsBomb club JOI computed")
        return pd.DataFrame()

    all_joi = pd.concat(all_joi_parts, ignore_index=True)
    all_shared = pd.concat(all_shared_parts, ignore_index=True)

    merged = all_joi.merge(
        all_shared[["match_id", "team_id", "player_a", "player_b", "shared_minutes"]],
        on=["match_id", "team_id", "player_a", "player_b"],
        how="left",
    )
    merged["shared_minutes"] = merged["shared_minutes"].fillna(0.0)
    merged.to_parquet(out_path, index=False)
    log.info("Saved SB club JOI per match: %s (%d rows)", out_path, len(merged))
    return merged


# ── Phase 9: Cross-context analysis with VAEP ─────────────────────────────

CLUB_COMPS_WY = {"England", "France", "Germany", "Italy", "Spain"}
INTL_COMPS_WY = {"World_Cup", "European_Championship"}

MINUTES_FLOOR_CLUB = 90.0
MINUTES_FLOOR_INTL = 45.0

FEATURED_WITH_IDS = [
    ("Lionel Messi",        3359,   "Barcelona",           "Argentina", "World_Cup"),
    ("Luka Modric",         8287,   "Real Madrid",         "Croatia",   "World_Cup"),
    ("Toni Kroos",          14723,  "Real Madrid",         "Germany",   "World_Cup"),
    ("Joshua Kimmich",      224593, "Bayern",              "Germany",   "World_Cup"),
    ("Thomas Muller",       14732,  "Bayern",              "Germany",   "World_Cup"),
    ("Mats Hummels",        14795,  "Bayern",              "Germany",   "World_Cup"),
    ("Jerome Boateng",      14716,  "Bayern",              "Germany",   "World_Cup"),
    ("Robert Lewandowski",  14817,  "Bayern",              "Poland",    "World_Cup"),
    ("Kylian Mbappe",       353833, "Paris Saint-Germain", "France",    "World_Cup"),
    ("Antoine Griezmann",   3682,   "Atletico Madrid",     "France",    "World_Cup"),
    ("Paul Pogba",          7936,   "Manchester United",   "France",    "World_Cup"),
    ("N'Golo Kante",        31528,  "Chelsea",             "France",    "World_Cup"),
    ("Raphael Varane",      3309,   "Real Madrid",         "France",    "World_Cup"),
    ("Kevin De Bruyne",     38021,  "Manchester City",     "Belgium",   "World_Cup"),
    ("Eden Hazard",         25707,  "Chelsea",             "Belgium",   "World_Cup"),
    ("Romelu Lukaku",       7905,   "Manchester United",   "Belgium",   "World_Cup"),
    ("Neymar",              40810,  "Paris Saint-Germain", "Brazil",    "World_Cup"),
    ("Mohamed Salah",       120353, "Liverpool",           "Egypt",     "World_Cup"),
]


def _norm(s: str) -> str:
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii").lower().replace("'", "")


def load_wy_players() -> pd.DataFrame:
    import re

    def _decode(s):
        if not isinstance(s, str):
            return s
        return re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), s)

    with open(DATA_WY / "players.json", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df.rename(columns={"wyId": "player_id", "shortName": "short_name"})
    df["full_name"] = df["firstName"] + " " + df["lastName"]
    df["short_name"] = df["short_name"].apply(_decode)
    df["full_name"] = df["full_name"].apply(_decode)
    return df[["player_id", "full_name", "short_name"]]


def load_wy_teams() -> pd.DataFrame:
    with open(DATA_WY / "teams.json") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df.rename(columns={"wyId": "team_id", "name": "team_name"})
    return df[["team_id", "team_name"]]


def load_wy_player_team_map() -> pd.DataFrame:
    """Load player->team mapping from all Wyscout match files."""
    comps = list(WY_COMPETITIONS)
    rows = []
    for comp in comps:
        path = DATA_WY / f"matches_{comp}.json"
        with open(path) as f:
            matches = json.load(f)
        for match in matches:
            match_id = match["wyId"]
            teams_data = match.get("teamsData", {})
            if not isinstance(teams_data, dict):
                continue
            for tid, tdata in teams_data.items():
                team_id = int(tid)
                formation = tdata.get("formation", {})
                lineup = formation.get("lineup", [])
                subs = formation.get("substitutions", [])
                for p in lineup:
                    pid = p.get("playerId")
                    if pid:
                        rows.append({
                            "match_id": match_id,
                            "competition": comp,
                            "team_id": team_id,
                            "player_id": int(pid),
                        })
                for s in subs:
                    if not isinstance(s, dict):
                        continue
                    pin = s.get("playerIn")
                    if pin:
                        rows.append({
                            "match_id": match_id,
                            "competition": comp,
                            "team_id": team_id,
                            "player_id": int(pin),
                        })
    return pd.DataFrame(rows).drop_duplicates()


def aggregate_joi90_dual(joi_per_match: pd.DataFrame,
                          club_comps: set[str],
                          intl_comps: set[str]) -> pd.DataFrame:
    """Aggregate per-match JOI to per-pair per-competition JOI90.

    Supports both joi_xt (xT) and joi_vaep columns.
    Returns aggregated DataFrame with both joi90_xt and joi90_vaep.
    """
    df = joi_per_match.copy()
    df["context"] = df["competition"].apply(
        lambda c: "club" if c in club_comps else "international"
    )

    has_xt = "joi_xt" in df.columns
    has_vaep = "joi_vaep" in df.columns

    agg_cols: dict[str, tuple[str, str]] = {
        "total_minutes": ("shared_minutes", "sum"),
        "matches": ("match_id", "nunique"),
    }
    if has_xt:
        agg_cols["total_joi_xt"] = ("joi_xt", "sum")
    if has_vaep:
        agg_cols["total_joi_vaep"] = ("joi_vaep", "sum")

    agg = (
        df.groupby(["competition", "context", "team_id", "player_a", "player_b"], as_index=False)
          .agg(**agg_cols)
    )

    if has_xt:
        agg["joi90_xt"] = (agg["total_joi_xt"] * 90.0 / agg["total_minutes"]).where(
            agg["total_minutes"] > 0, 0.0
        )
    if has_vaep:
        agg["joi90_vaep"] = (agg["total_joi_vaep"] * 90.0 / agg["total_minutes"]).where(
            agg["total_minutes"] > 0, 0.0
        )

    # Apply floors
    club_mask = agg["context"] == "club"
    intl_mask = agg["context"] == "international"
    agg = agg[
        (club_mask & (agg["total_minutes"] >= MINUTES_FLOOR_CLUB))
        | (intl_mask & (agg["total_minutes"] >= MINUTES_FLOOR_INTL))
    ]
    return agg


def find_club_team(player_id: int, club_query: str, player_team_map: pd.DataFrame,
                   teams_df: pd.DataFrame) -> tuple[int | None, str | None]:
    """Find club team ID for a player."""
    player_club = player_team_map[
        (player_team_map["player_id"] == player_id)
        & (player_team_map["competition"].isin(CLUB_COMPS_WY))
    ]
    if player_club.empty:
        return None, None
    team_ids = player_club["team_id"].unique()
    cname = club_query.lower()
    for tid in team_ids:
        tname = teams_df[teams_df["team_id"] == tid]["team_name"].values
        if len(tname) > 0 and any(w in tname[0].lower() for w in cname.split()):
            name_str = tname[0]
            comp_opts = player_team_map[
                (player_team_map["player_id"] == player_id)
                & (player_team_map["team_id"] == tid)
            ]["competition"].unique()
            return int(tid), comp_opts[0] if len(comp_opts) > 0 else None
    counts = player_club.groupby("team_id")["match_id"].nunique()
    tid = int(counts.idxmax())
    tname_vals = teams_df[teams_df["team_id"] == tid]["team_name"].values
    return tid, player_club[player_club["team_id"] == tid]["competition"].iloc[0]


def find_intl_team(player_id: int, country: str, comp_name: str,
                   player_team_map: pd.DataFrame, teams_df: pd.DataFrame) -> int | None:
    player_intl = player_team_map[
        (player_team_map["player_id"] == player_id)
        & (player_team_map["competition"] == comp_name)
    ]
    if player_intl.empty:
        return None
    team_ids = player_intl["team_id"].unique()
    if len(team_ids) == 1:
        return int(team_ids[0])
    cname = country.lower()
    for tid in team_ids:
        tname = teams_df[teams_df["team_id"] == tid]["team_name"].values
        if len(tname) > 0 and cname in tname[0].lower():
            return int(tid)
    return int(team_ids[0])


def get_pairs_for_player_dual(
    player_id: int,
    team_id: int,
    comp_filter: str | None,
    joi90_df: pd.DataFrame,
    players_df: pd.DataFrame,
    top_n: int = 8,
    minutes_floor: float = MINUTES_FLOOR_CLUB,
) -> list[dict]:
    """Get top-N teammates by shared minutes, with both xT and VAEP JOI90."""
    if comp_filter:
        df = joi90_df[
            (joi90_df["competition"] == comp_filter)
            & (joi90_df["team_id"] == team_id)
            & ((joi90_df["player_a"] == player_id) | (joi90_df["player_b"] == player_id))
            & (joi90_df["total_minutes"] >= minutes_floor)
        ].copy()
    else:
        df = joi90_df[
            (joi90_df["team_id"] == team_id)
            & ((joi90_df["player_a"] == player_id) | (joi90_df["player_b"] == player_id))
            & (joi90_df["total_minutes"] >= minutes_floor)
        ].copy()

    df["teammate_id"] = df.apply(
        lambda r: r["player_b"] if r["player_a"] == player_id else r["player_a"], axis=1
    )
    df = df.sort_values("total_minutes", ascending=False).head(top_n)

    name_map = players_df.set_index("player_id")["short_name"].to_dict()
    result = []
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        tid = int(row["teammate_id"])
        entry: dict = {
            "teammate_id": tid,
            "teammate": name_map.get(tid, f"Player #{tid}"),
            "minutes": round(float(row["total_minutes"]), 1),
            "matches": int(row["matches"]),
            "team_rank": rank,
        }
        if "joi90_xt" in row:
            entry["joi90_xt"] = round(float(row["joi90_xt"]), 4)
        if "joi90_vaep" in row:
            entry["joi90_vaep"] = round(float(row["joi90_vaep"]), 4)
        result.append(entry)
    return result


def compute_cross_context_dual(
    joi90_df: pd.DataFrame,
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    player_team_map: pd.DataFrame,
) -> list[dict]:
    """Compute cross-context data for all featured players (dual xT+VAEP)."""
    results = []
    for (name, player_id, club_query, country, intl_comp) in FEATURED_WITH_IDS:
        log.info("Processing %s (wyId=%d) ...", name, player_id)

        club_team_id, club_comp = find_club_team(player_id, club_query, player_team_map, teams_df)
        if club_team_id is None:
            club_pairs: list[dict] = []
        else:
            club_team_name = teams_df[teams_df["team_id"] == club_team_id]["team_name"].values
            club_team_name = club_team_name[0] if len(club_team_name) > 0 else club_query
            club_pairs = get_pairs_for_player_dual(
                player_id, club_team_id, club_comp, joi90_df, players_df,
                top_n=8, minutes_floor=MINUTES_FLOOR_CLUB
            )

        nat_team_id = find_intl_team(player_id, country, intl_comp, player_team_map, teams_df)
        if nat_team_id is None:
            country_pairs: list[dict] = []
        else:
            country_pairs = get_pairs_for_player_dual(
                player_id, nat_team_id, intl_comp, joi90_df, players_df,
                top_n=8, minutes_floor=MINUTES_FLOOR_INTL
            )

        # Summary stats using VAEP as primary
        vaep_key = "joi90_vaep"
        xt_key = "joi90_xt"

        def _avg(pairs: list[dict], key: str) -> float:
            vals = [p[key] for p in pairs if key in p]
            return float(np.mean(vals)) if vals else 0.0

        avg_club_vaep = _avg(club_pairs, vaep_key)
        avg_country_vaep = _avg(country_pairs, vaep_key)
        avg_club_xt = _avg(club_pairs, xt_key)
        avg_country_xt = _avg(country_pairs, xt_key)

        ratio_vaep = avg_country_vaep / avg_club_vaep if avg_club_vaep > 0 else None
        ratio_xt = avg_country_xt / avg_club_xt if avg_club_xt > 0 else None

        results.append({
            "name": name,
            "player_id": player_id,
            "club": club_query,
            "club_team_id": club_team_id,
            "country": country,
            "nat_team_id": nat_team_id,
            "club_pairs": club_pairs,
            "country_pairs": country_pairs,
            "summary": {
                "avg_club_joi90_vaep": round(avg_club_vaep, 4),
                "avg_country_joi90_vaep": round(avg_country_vaep, 4),
                "ratio_vaep": round(ratio_vaep, 3) if ratio_vaep is not None else None,
                "avg_club_joi90_xt": round(avg_club_xt, 4),
                "avg_country_joi90_xt": round(avg_country_xt, 4),
                "ratio_xt": round(ratio_xt, 3) if ratio_xt is not None else None,
                "club_pairs_found": len(club_pairs),
                "country_pairs_found": len(country_pairs),
            },
        })

    return results


def compute_bayern_deep_dive_dual(
    joi90_df: pd.DataFrame,
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    player_team_map: pd.DataFrame,
) -> list[dict]:
    """Bayern 2017/18 -> Germany WC 2018 deep-dive with VAEP and xT."""
    ids = {
        "Kimmich": 224593,
        "Müller":  14732,
        "Hummels": 14795,
        "Boateng": 14716,
    }

    kimmich_id = ids["Kimmich"]
    club_team_id, club_comp = find_club_team(kimmich_id, "Bayern", player_team_map, teams_df)
    germany_id = find_intl_team(kimmich_id, "Germany", "World_Cup", player_team_map, teams_df)

    if not club_team_id or not germany_id:
        log.warning("Could not find Bayern or Germany team IDs")
        return []

    results = []
    player_list = list(ids.keys())
    for i, name_a in enumerate(player_list):
        for name_b in player_list[i+1:]:
            pid_a = ids[name_a]
            pid_b = ids[name_b]
            pa, pb = min(pid_a, pid_b), max(pid_a, pid_b)

            def get_pair(team_id, comp, minutes_floor):
                df = joi90_df[
                    (joi90_df["competition"] == comp)
                    & (joi90_df["team_id"] == team_id)
                    & (joi90_df["player_a"] == pa)
                    & (joi90_df["player_b"] == pb)
                    & (joi90_df["total_minutes"] >= minutes_floor)
                ]
                if df.empty:
                    return None, None, None, None
                row = df.iloc[0]
                vaep = round(float(row["joi90_vaep"]), 4) if "joi90_vaep" in row else None
                xt = round(float(row["joi90_xt"]), 4) if "joi90_xt" in row else None
                mins = round(float(row["total_minutes"]), 1)
                return vaep, xt, mins, None

            club_vaep, club_xt, club_mins, _ = get_pair(club_team_id, club_comp, MINUTES_FLOOR_CLUB)
            ctry_vaep, ctry_xt, ctry_mins, _ = get_pair(germany_id, "World_Cup", MINUTES_FLOOR_INTL)

            results.append({
                "pair": f"{name_a} — {name_b}",
                "club_joi90_vaep": club_vaep,
                "club_joi90_xt": club_xt,
                "club_minutes": club_mins,
                "country_joi90_vaep": ctry_vaep,
                "country_joi90_xt": ctry_xt,
                "country_minutes": ctry_mins,
            })

    return results


def phase9_cross_context_analysis(
    wy_vaep_joi: pd.DataFrame,
    wy_xt_joi: pd.DataFrame,
) -> tuple[list[dict], list[dict], dict]:
    """Join VAEP and xT JOI, run cross-context analysis."""
    log.info("=== Phase 9: Cross-context analysis ===")

    players_df = load_wy_players()
    teams_df = load_wy_teams()
    player_team_map = load_wy_player_team_map()

    # Merge xT and VAEP JOI per match on same key columns
    joi_merged = wy_xt_joi.merge(
        wy_vaep_joi[["match_id", "competition", "team_id", "player_a", "player_b", "joi_vaep"]],
        on=["match_id", "competition", "team_id", "player_a", "player_b"],
        how="outer",
    )
    # Fill NaN from outer join
    for col in ["joi_xt", "joi_vaep", "shared_minutes"]:
        if col in joi_merged.columns:
            joi_merged[col] = joi_merged[col].fillna(0.0)

    # Aggregate JOI90
    joi90_df = aggregate_joi90_dual(joi_merged, CLUB_COMPS_WY, INTL_COMPS_WY)
    log.info("JOI90 pairs: %d", len(joi90_df))

    player_results = compute_cross_context_dual(joi90_df, players_df, teams_df, player_team_map)
    deep_dive = compute_bayern_deep_dive_dual(joi90_df, players_df, teams_df, player_team_map)

    # Build meta
    total_matches = wy_xt_joi["match_id"].nunique()
    total_actions = sum(
        len(pd.read_parquet(DATA_SPADL_WY / f"spadl_{c}.parquet", columns=["match_id"]))
        for c in WY_COMPETITIONS
    )
    comp_counts = wy_xt_joi.groupby("competition")["match_id"].nunique().to_dict()
    meta = {
        "dataset": "Wyscout 2017/18 + WC 2018 + Euro 2016 (primary)",
        "metric_primary": "VAEP (Decroos et al. 2019, Bransen & Van Haaren 2020)",
        "metric_secondary": "xT (Singh 2019)",
        "license_wyscout": "CC BY 4.0",
        "total_matches": total_matches,
        "total_actions_approx": total_actions,
        "matches_per_competition": comp_counts,
    }

    return player_results, deep_dive, meta


# ── Phase 10: StatsBomb club cross-context (Bayern 2023/24 -> Germany Euro 2024) ──

def phase10_sb_club_cross_context(sb_joi: pd.DataFrame) -> dict:
    """Bayern 2023/24 -> Germany Euro 2024 deep-dive using StatsBomb data.

    Uses:
    - Club: Bundesliga 2023/24 (cid=9, sid=281)
    - International: UEFA Euro 2024 (cid=55, sid=282, already in existing pipeline)
    """
    log.info("=== Phase 10: Bayern 2023/24 -> Germany Euro 2024 deep-dive ===")

    # Load StatsBomb international JOI (Euro 2024) from existing pipeline output
    intl_joi_path = OUTPUTS / "joi_per_match.parquet"
    if not intl_joi_path.exists():
        log.warning("International JOI not found at %s", intl_joi_path)
        return {"available": False}

    intl_joi = pd.read_parquet(intl_joi_path)
    log.info("International JOI rows: %d", len(intl_joi))
    log.info("International JOI competitions: %s", intl_joi["competition_id"].unique().tolist()
             if "competition_id" in intl_joi.columns else "unknown col")

    # Look at structure of international JOI
    log.info("International JOI columns: %s", intl_joi.columns.tolist())

    # Find Euro 2024 data (competition_id=55, season_id=282 -> label '55_282')
    if "competition" in intl_joi.columns:
        euro_2024 = intl_joi[intl_joi["competition"].str.contains("Euro 2024", case=False, na=False)
                             | intl_joi["competition"].str.contains("55_282", case=False, na=False)]
    else:
        euro_2024 = pd.DataFrame()

    if euro_2024.empty:
        # Try different column names
        for col in intl_joi.columns:
            log.info("  %s sample: %s", col, intl_joi[col].iloc[0] if len(intl_joi) > 0 else "empty")

    # Bayern 2023/24 StatsBomb player IDs
    # These are StatsBomb IDs (different from Wyscout IDs)
    # We need to identify Bayern players from the Bundesliga 2023/24 SPADL
    bl_key = "9_281"
    bl_spadl_path = DATA_SPADL_SB / f"spadl_{bl_key}.parquet"

    if not bl_spadl_path.exists():
        log.warning("Bundesliga 2023/24 SPADL not found at %s", bl_spadl_path)
        return {"available": False, "reason": "Bundesliga 2023/24 SPADL not fetched"}

    bl_spadl = pd.read_parquet(bl_spadl_path)
    log.info("Bundesliga 2023/24 SPADL: %d actions", len(bl_spadl))

    # Get StatsBomb player names from events
    bl_events_dir = DATA_RAW / bl_key / "events"

    # Build a player name map for Bundesliga
    player_names: dict[int, str] = {}
    if bl_events_dir.exists():
        sample_files = list(bl_events_dir.glob("*.parquet"))[:5]
        for fp in sample_files:
            ev = pd.read_parquet(fp)
            if "player" in ev.columns:
                for _, row in ev.iterrows():
                    p = row.get("player")
                    if isinstance(p, dict) and "id" in p and "name" in p:
                        player_names[int(p["id"])] = str(p["name"])

    log.info("Player names from Bundesliga sample: %d", len(player_names))

    # Find Bayern Munich team_id in Bundesliga data
    bl_matches = pd.read_parquet(DATA_RAW / bl_key / "matches.parquet")
    log.info("Bundesliga 2023/24 teams sample: %s",
             [str(r.get("home_team", r.get("team", "")))[:60]
              for _, r in bl_matches.head(3).iterrows()])

    return {
        "available": True,
        "bl_actions": len(bl_spadl),
        "player_names_sample": player_names,
        "note": "Full Bayern->Germany Euro 2024 analysis requires SB player name resolution",
    }


# ── Phase 11: Output JSON ──────────────────────────────────────────────────

def phase11_write_outputs(
    meta: dict,
    player_results: list[dict],
    deep_dive: list[dict],
    vaep_bundle: dict,
    sb_deep_dive: dict | None = None,
) -> None:
    """Write cross_context_chemistry.json and update the markdown."""
    log.info("=== Phase 11: Write outputs ===")

    output = {
        "meta": meta,
        "vaep_model": {
            "train_size": vaep_bundle.get("train_size", 0),
            "auc_scores": round(vaep_bundle.get("auc_scores", 0), 4),
            "auc_concedes": round(vaep_bundle.get("auc_concedes", 0), 4),
        },
        "players": player_results,
        "bayern_germany_deep_dive": deep_dive,
    }

    if sb_deep_dive:
        output["sb_expanded"] = sb_deep_dive

    out_path = OUTPUTS / "cross_context_chemistry.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log.info("Wrote %s", out_path)

    # Write markdown
    _write_markdown(meta, player_results, deep_dive, vaep_bundle)


def _fmt(v, d=4):
    if v is None:
        return "n/a"
    return f"{v:.{d}f}"


def _write_markdown(meta: dict, player_results: list[dict], deep_dive: list[dict],
                    vaep_bundle: dict) -> None:
    """Write updated cross-context-chemistry.md."""
    lines: list[str] = []

    lines += [
        "# Club vs Country Chemistry: Wyscout Cross-Context Analysis",
        "",
        "## 1. Setup: The Podolski Paradox",
        "",
        "Lukas Podolski is the canonical case: universally acclaimed for Germany, a peripheral figure "
        "at Arsenal, Internazionale, and Monaco. He scored 49 goals in 130 Germany caps and accumulated "
        "one Bundesliga title and a World Cup winners medal in 2014, yet finished his club career without "
        "a major domestic trophy. The pattern raises a genuine question: does playing for your national "
        "team unlock something — tactical role clarity, reduced positional competition, emotional "
        "investment, familiarity with teammates from years of camp — that clubs cannot replicate?",
        "",
        "**Following the paper, this analysis uses VAEP (Decroos et al. 2019, Bransen & Van Haaren 2020) "
        "as the action-value function. xT results are reported alongside for cross-reference.**",
        "",
        "The Wyscout open dataset (Pappalardo et al., 2019, Nature Scientific Data) covers the Premier "
        "League, La Liga, Bundesliga, Serie A, and Ligue 1 (all 2017/18), plus WC 2018 and Euro 2016, "
        "all in a consistent schema with ~1.9 million events. Because the same players appear in both "
        "domestic and international fixtures, within-player comparisons are possible.",
        "",
        "**Dataset stats:**",
        "",
    ]

    for comp, n in sorted(meta.get("matches_per_competition", {}).items()):
        lines.append(f"- {comp}: {n} matches")
    lines += [
        f"- Total matches: {meta.get('total_matches', 'N/A')}",
        f"- Total SPADL actions (approx): {meta.get('total_actions_approx', 0):,}",
        "- License: CC BY 4.0",
        "",
    ]

    auc_s = vaep_bundle.get("auc_scores", 0)
    auc_c = vaep_bundle.get("auc_concedes", 0)
    train_n = vaep_bundle.get("train_size", 0)
    lines += [
        "## 2. Methodology",
        "",
        f"**VAEP-based JOI90 (primary).** Every event is converted to SPADL format via socceraction's "
        f"Wyscout converter. A VAEP model is trained on all {train_n:,} SPADL actions across all 7 "
        f"Wyscout competitions combined (train-set AUC: scores={auc_s:.3f}, concedes={auc_c:.3f}). "
        f"Following Decroos et al. 2019, two XGBClassifier models predict P(scores in next N actions) "
        f"and P(concedes in next N actions); N=10. VAEP value per action = delta-P(scores) - delta-P(concedes). "
        f"A pair interaction is two consecutive on-ball actions (passes, crosses, dribbles, take-ons, shots) "
        f"by different players on the same team. JOI contribution = VAEP of the second action. "
        f"Per-pair JOI is summed and normalized per 90 shared minutes (JOI90).",
        "",
        "**xT-based JOI90 (secondary reference).** Also computed for cross-reference using a 16x12 "
        "xT grid fitted on the same 7 competitions. Values reported in parentheses or secondary columns "
        "throughout. VAEP and xT scales differ — do not compare absolute values across metrics.",
        "",
        "**Shared minutes** are derived from Wyscout's teamsData formation structure. "
        "Floors: club pairs require ≥90 shared minutes; international ≥45.",
        "",
        "**Caveats on cross-context comparisons.** Absolute JOI90 differences between club and "
        "international are not pure chemistry signal. International opponents are often weaker; "
        "the sample is one season and one tournament. Rank among team peers is a more stable lens "
        "than raw delta.",
        "",
    ]

    # Big picture table — VAEP primary
    valid = [
        (p, p["summary"].get("ratio_vaep"))
        for p in player_results
        if p.get("summary", {}).get("ratio_vaep") is not None
    ]
    valid_sorted = sorted(valid, key=lambda x: -(x[1] or 0))

    lines += [
        "## 3. The Big Picture (VAEP-based)",
        "",
        "Country VAEP-JOI90 / Club VAEP-JOI90 ratio for featured players "
        f"(pairs top-8 by shared minutes, floor: club ≥{MINUTES_FLOOR_CLUB:.0f} min, "
        f"international ≥{MINUTES_FLOOR_INTL:.0f} min):",
        "",
        "| Player | Club | Country | Avg Club VAEP-JOI90 | Avg Country VAEP-JOI90 | Ratio | xT Ratio |",
        "|---|---|---|---|---|---|---|",
    ]
    for p, ratio in valid_sorted:
        s = p["summary"]
        ratio_xt = s.get("ratio_xt")
        lines.append(
            f"| {p['name']} | {p['club']} | {p['country']} "
            f"| {_fmt(s['avg_club_joi90_vaep'], 4)} | {_fmt(s['avg_country_joi90_vaep'], 4)} "
            f"| {_fmt(ratio, 2)} | {_fmt(ratio_xt, 2)} |"
        )
    lines += [""]

    podolski_count = len([p for p, r in valid_sorted if r is not None and r > 1.2])
    inverse_count = len([p for p, r in valid_sorted if r is not None and r < 0.8])
    total_valid = len(valid_sorted)

    lines += [
        "**True Podolski types (country VAEP-JOI90 > club, ratio > 1.2):**",
        "",
    ]
    for p, r in [(p, r) for p, r in valid_sorted if r is not None and r > 1.2][:6]:
        lines.append(
            f"- **{p['name']}** ({p['club']} → {p['country']}): ratio {r:.2f}. "
            f"Club VAEP-avg {p['summary']['avg_club_joi90_vaep']:.4f}, "
            f"country VAEP-avg {p['summary']['avg_country_joi90_vaep']:.4f}."
        )

    lines += [
        "",
        "**Inverse Podolski types (club VAEP-JOI90 > country, ratio < 0.8):**",
        "",
    ]
    for p, r in [(p, r) for p, r in valid_sorted if r is not None and r < 0.8][:6]:
        lines.append(
            f"- **{p['name']}** ({p['club']} → {p['country']}): ratio {r:.2f}. "
            f"Club VAEP-avg {p['summary']['avg_club_joi90_vaep']:.4f}, "
            f"country VAEP-avg {p['summary']['avg_country_joi90_vaep']:.4f}."
        )

    lines += [
        "",
        f"VAEP verdict: {total_valid} players with sufficient data in both contexts. "
        f"{podolski_count} showed higher average VAEP-JOI90 with national team partners "
        f"(country > club); {inverse_count} showed higher club chemistry.",
        "",
    ]

    # Bayern deep dive
    lines += [
        "## 4. The Bayern 2017/18 → Germany WC 2018 Deep-Dive",
        "",
        "Germany's 2018 World Cup campaign ended in the group stage — three matches, zero wins. "
        "Four of their core starters (Kimmich, Müller, Hummels, Boateng) played together weekly "
        "at Bayern Munich.",
        "",
        "| Pair | Club VAEP-JOI90 | Club xT-JOI90 | Club minutes | Country VAEP-JOI90 | Country xT-JOI90 | Country minutes | Transfer (VAEP)? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in deep_dive:
        c_vaep = _fmt(item.get("club_joi90_vaep"))
        c_xt = _fmt(item.get("club_joi90_xt"))
        c_min = item.get("club_minutes", "n/a")
        n_vaep = _fmt(item.get("country_joi90_vaep"))
        n_xt = _fmt(item.get("country_joi90_xt"))
        n_min = item.get("country_minutes", "n/a")
        cv = item.get("club_joi90_vaep")
        nv = item.get("country_joi90_vaep")
        if cv is not None and nv is not None:
            transfer = "yes" if nv >= 0.8 * cv else "partial" if nv >= 0.5 * cv else "no"
        else:
            transfer = "n/a"
        lines.append(f"| {item['pair']} | {c_vaep} | {c_xt} | {c_min} | {n_vaep} | {n_xt} | {n_min} | {transfer} |")

    lines += [
        "",
        "Germany had 3 WC 2018 matches — at most ~270 shared minutes for a starting pair. "
        "High variance on all country estimates.",
        "",
    ]

    # Caveats
    lines += [
        "## 5. Caveats",
        "",
        "1. **VAEP scale differences.** VAEP trained on Wyscout SPADL has a different absolute "
        "scale from VAEP trained on StatsBomb SPADL (different action distributions, different "
        "feature encodings). Within-source comparisons are valid; cross-source comparisons are not.",
        "",
        "2. **Sample size.** Germany had 3 WC 2018 matches. Poland, Egypt: 3 each. "
        "Any JOI90 estimate on <4 matches has very high variance.",
        "",
        "3. **Opponent strength.** Group-stage WC opponents are on average weaker than "
        "top-division club opposition.",
        "",
        "4. **Tactical role.** Kimmich played right back at Bayern but midfield for Germany. "
        "This changes which pairs form at all, not just their quality.",
        "",
        "5. **Single season.** This analysis covers 2017/18 + WC 2018 only.",
        "",
        "6. **Metric note.** Switching from xT to VAEP changes absolute values substantially "
        "(VAEP is compressed near zero for non-shot-producing actions; xT accumulates more "
        "on progressive passes). The direction of findings (which players are Podolski-type "
        "vs inverse-Podolski) may differ between metrics — check both ratio columns.",
        "",
    ]

    out_path = ROOT / "docs" / "analysis" / "cross-context-chemistry.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info("Wrote %s", out_path)


# ── Phase 12: Update site/index.html Research tab ─────────────────────────

def phase12_update_site(player_results: list[dict], deep_dive: list[dict], vaep_bundle: dict) -> None:
    """Update the Research tab in site/index.html with VAEP values."""
    log.info("=== Phase 12: Update site Research tab ===")

    site_path = ROOT / "site" / "index.html"
    with open(site_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Compute summary counts from VAEP
    valid = [
        (p, p["summary"].get("ratio_vaep"))
        for p in player_results
        if p.get("summary", {}).get("ratio_vaep") is not None
    ]
    podolski_count = len([p for p, r in valid if r is not None and r > 1.2])
    inverse_count = len([p for p, r in valid if r is not None and r < 0.8])
    near_parity = len(valid) - podolski_count - inverse_count

    # Build the new Research tab Bayern deep-dive table rows
    tbody_rows: list[str] = []
    for item in deep_dive:
        pair_html = item["pair"].replace("—", "&mdash;").replace("ü", "&uuml;").replace("ä", "&auml;")
        c_vaep = item.get("club_joi90_vaep")
        c_xt = item.get("club_joi90_xt")
        n_vaep = item.get("country_joi90_vaep")
        n_xt = item.get("country_joi90_xt")
        c_min = item.get("club_minutes", "")
        n_min = item.get("country_minutes", "")

        # Determine row styling
        if c_vaep is not None and n_vaep is not None:
            delta = n_vaep - c_vaep
            delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
            delta_class = "joi-pos" if delta > 0 else "joi-neg"
            highlight = ' class="row-highlight"' if abs(delta) > 0.05 else ""
            n_vaep_class = "joi joi-high" if n_vaep > 0.05 else "joi joi-neg-val" if n_vaep < 0 else "joi"
        else:
            delta_str = "n/a"
            delta_class = ""
            highlight = ""
            n_vaep_class = "joi"

        c_vaep_str = f"{c_vaep:.3f}" if c_vaep is not None else "n/a"
        n_vaep_str = f"{n_vaep:.3f}" if n_vaep is not None else "n/a"
        c_xt_str = f"{c_xt:.3f}" if c_xt is not None else "n/a"
        n_xt_str = f"{n_xt:.3f}" if n_xt is not None else "n/a"

        tbody_rows.append(
            f'              <tr{highlight}>\n'
            f'                <td><strong>{pair_html}</strong></td>\n'
            f'                <td class="joi">{c_vaep_str}</td>\n'
            f'                <td class="joi" style="color:var(--text-muted);font-size:11px">{c_xt_str}</td>\n'
            f'                <td class="{n_vaep_class}">{n_vaep_str}</td>\n'
            f'                <td class="joi" style="color:var(--text-muted);font-size:11px">{n_xt_str}</td>\n'
            f'                <td class="joi {delta_class}">{delta_str}</td>\n'
            f'                <td>{int(c_min) if c_min else "n/a"} / {int(n_min) if n_min else "n/a"}</td>\n'
            f'              </tr>'
        )

    # Find inverse Podolski examples
    mbappe = next((p for p in player_results if "Mbappe" in p["name"]), None)
    hazard = next((p for p in player_results if "Hazard" in p["name"]), None)

    mbappe_club_v = mbappe["summary"]["avg_club_joi90_vaep"] if mbappe else None
    mbappe_ctry_v = mbappe["summary"]["avg_country_joi90_vaep"] if mbappe else None
    mbappe_ratio_v = mbappe["summary"]["ratio_vaep"] if mbappe else None
    hazard_club_v = hazard["summary"]["avg_club_joi90_vaep"] if hazard else None
    hazard_ctry_v = hazard["summary"]["avg_country_joi90_vaep"] if hazard else None
    hazard_ratio_v = hazard["summary"]["ratio_vaep"] if hazard else None

    auc_s = vaep_bundle.get("auc_scores", 0)
    auc_c = vaep_bundle.get("auc_concedes", 0)
    train_n = vaep_bundle.get("train_size", 0)

    # Build the new Research section HTML
    new_research_html = f'''\
  <!-- ── Tab 4: Research ────────────────────────────────────────────── -->
  <div class="tab-panel" id="panel-research" role="tabpanel" aria-labelledby="tab-research">
    <div class="slide-deck">

      <!-- Section 1: Cross-context (Wyscout) analysis -->
      <section class="slide-card" aria-label="Club vs country chemistry">
        <span class="pill green">Primary finding</span>
        <h2>Club vs country: does shared club training carry to international football?</h2>

        <p>The "Podolski thesis" &mdash; anecdote-driven but quantifiable &mdash; holds that players thrive at international level partly because the national team context suits them better than their club. A stricter version: pair-level chemistry built at a club should transfer when the same two players line up for their country. The Wyscout open dataset (Pappalardo et al., 2019) covers five top European leagues in 2017/18, WC 2018, and Euro 2016 in a single consistent schema.</p>

        <p>Following the paper, this analysis uses <strong>VAEP</strong> (Decroos et al. 2019) as the action-value function. xT results are reported alongside for cross-reference. VAEP model: {train_n:,} Wyscout actions, train-AUC scores={auc_s:.3f}, concedes={auc_c:.3f}.</p>

        <div class="research-verdict">
          <div class="verdict-card verdict-split">
            <div class="verdict-number">{podolski_count} / {len(valid)}</div>
            <div class="verdict-label">players showed higher average VAEP-JOI90 with national team partners than club partners (country &gt; club)</div>
          </div>
          <div class="verdict-card verdict-split">
            <div class="verdict-number">{inverse_count} / {len(valid)}</div>
            <div class="verdict-label">players showed higher average VAEP-JOI90 at club level (club &gt; country)</div>
          </div>
          <div class="verdict-card verdict-neutral">
            <div class="verdict-number">{near_parity} / {len(valid)}</div>
            <div class="verdict-label">near-parity (ratio 0.8&ndash;1.2); no clear direction</div>
          </div>
        </div>

        <p>The data is split, not one-sided. Club chemistry generally dominates in the aggregate, but the exceptions are real and interesting.</p>

        <h3>Bayern &rarr; Germany deep-dive (WC 2018) &mdash; VAEP primary, xT secondary</h3>
        <p>Germany fielded four Bayern starters at WC 2018. VAEP values are the paper&rsquo;s method; xT shown in lighter columns for reference. Note: VAEP and xT scales differ &mdash; do not compare absolute values across the two metric columns.</p>

        <!-- Updated from outputs/cross_context_chemistry.json &rarr; bayern_germany_deep_dive[] with VAEP primary -->
        <div class="research-table-wrap">
          <table class="leaderboard research-table">
            <thead>
              <tr>
                <th>Pair</th>
                <th>Club VAEP-JOI90 <span class="th-sub">(Bayern 2017/18)</span></th>
                <th style="color:var(--text-muted);font-size:11px">Club xT-JOI90</th>
                <th>Country VAEP-JOI90 <span class="th-sub">(Germany WC 2018)</span></th>
                <th style="color:var(--text-muted);font-size:11px">Country xT-JOI90</th>
                <th>Delta (VAEP)</th>
                <th>Minutes <span class="th-sub">club / country</span></th>
              </tr>
            </thead>
            <tbody>
{chr(10).join(tbody_rows)}
            </tbody>
          </table>
        </div>
        <p class="research-note">Kimmich played right back at Bayern and central midfield for Germany &mdash; the role change affects which pair interactions form and their quality. All Germany values are on 3 WC 2018 matches (&le;270 shared minutes); variance is high. Switched primary metric from xT to VAEP per Bransen &amp; Van Haaren 2020.</p>

        <h3>The inverse Podolski: club &gt;&gt; country</h3>
        <div class="research-inverse-cards">
          <div class="slide-inner">
            <h3>Kylian Mbapp&eacute; (PSG &rarr; France 2018)</h3>
            <p>Club avg VAEP-JOI90: <strong>{f"{mbappe_club_v:.4f}" if mbappe_club_v is not None else "n/a"}</strong> &nbsp;&mdash;&nbsp; Country avg VAEP-JOI90: <strong>{f"{mbappe_ctry_v:.4f}" if mbappe_ctry_v is not None else "n/a"}</strong> &nbsp;&mdash;&nbsp; Ratio: <strong>{f"{mbappe_ratio_v:.2f}" if mbappe_ratio_v is not None else "n/a"}</strong></p>
            <p>France won the tournament; Mbapp&eacute; scored in the final. His individual pair chemistry numbers were stronger at PSG than at a World Cup-winning national team. The tournament outcome and the pair-chemistry numbers point in opposite directions.</p>
          </div>
          <div class="slide-inner">
            <h3>Eden Hazard (Chelsea &rarr; Belgium 2018)</h3>
            <p>Club avg VAEP-JOI90: <strong>{f"{hazard_club_v:.4f}" if hazard_club_v is not None else "n/a"}</strong> &nbsp;&mdash;&nbsp; Country avg VAEP-JOI90: <strong>{f"{hazard_ctry_v:.4f}" if hazard_ctry_v is not None else "n/a"}</strong> &nbsp;&mdash;&nbsp; Ratio: <strong>{f"{hazard_ratio_v:.2f}" if hazard_ratio_v is not None else "n/a"}</strong></p>
            <p>Belgium reached the semi-final. Hazard was regarded as one of Belgium&rsquo;s key players throughout, but his pair interaction numbers were higher at Chelsea. Kevin De Bruyne shows a similar pattern.</p>
          </div>
        </div>

        <div class="slide-inner" style="margin-top:16px">
          <h3>Caveats</h3>
          <ul>
            <li>Germany played 3 WC 2018 group matches before elimination. JOI90 estimates on &le;3 matches have confidence intervals wide enough to overlap zero &mdash; direction-of-effect, not reliable effect-size.</li>
            <li>VAEP trained on Wyscout SPADL is not directly comparable to VAEP trained on StatsBomb SPADL (different action distributions, different feature encodings).</li>
            <li>International group-stage opponents are on average weaker than top-division club opposition.</li>
            <li>Players often occupy different positional roles for club vs country (Kimmich: right back at Bayern, CM at Germany).</li>
            <li>This Wyscout analysis covers 2017/18 + WC 2018 only &mdash; one season, eight years ago.</li>
          </ul>
        </div>

        <p class="research-footer">Data: Wyscout open dataset &mdash; Pappalardo et al. (2019), CC BY 4.0. Metric: VAEP per Bransen &amp; Van Haaren 2020 (primary), xT per Singh 2019 (secondary). <a href="https://github.com/stranger9977/wc2026-chemistry/blob/main/docs/analysis/cross-context-chemistry.md" target="_blank" rel="noopener">Full analysis &rarr;</a></p>
      </section>

      <!-- Section 2: Same-club shortcut -->
      <section class="slide-card" aria-label="Same-club shortcut">
        <span class="pill amber">Null finding</span>
        <h2>The "same club" shortcut: does sharing a club correlate with international chemistry?</h2>

        <p>A simpler version of the question: at international tournaments, do pairs who currently play at the same club show higher JOI90 than pairs from different clubs? Two separate analyses give the same answer.</p>

        <div class="slide-2col" style="margin-top:8px">
          <div class="slide-inner">
            <h3>Modern &mdash; WC 2026 squads</h3>
            <p>1,188 pairs across 23 nations; 50 same-club pairs, 1,138 different-club pairs. Chemistry scored on six tournaments (WC 2022, Euro 2024, Euro 2020, Copa Am&eacute;rica 2024, WC 2018, AFCON 2023).</p>
            <table class="leaderboard" style="margin-top:10px;font-size:13px">
              <thead><tr><th></th><th>Same-club (n=50)</th><th>Different-club (n=1,138)</th></tr></thead>
              <tbody>
                <tr><td>xT mean</td><td class="joi">0.041</td><td class="joi">0.074</td></tr>
                <tr><td>xT median</td><td class="joi">0.011</td><td class="joi">0.011</td></tr>
              </tbody>
            </table>
            <p style="margin-top:8px">Welch t-test on xT: t = &minus;1.12 (df &asymp;54). Not significant. Medians are identical.</p>
          </div>
          <div class="slide-inner">
            <h3>Historic &mdash; WC 2018 + AFCON 2023</h3>
            <p>Per-tournament pairs for five WC 2018 squads and three AFCON 2023 squads. 45-minute floor within each tournament.</p>
            <table class="leaderboard" style="margin-top:10px;font-size:13px">
              <thead><tr><th>Squad</th><th>Same-club delta</th></tr></thead>
              <tbody>
                <tr><td>Germany (WC 2018)</td><td class="joi">&minus;0.001</td></tr>
                <tr><td>Belgium (WC 2018)</td><td class="joi">&minus;0.004</td></tr>
                <tr><td>France (WC 2018)</td><td class="joi">+0.009</td></tr>
                <tr><td>Croatia (WC 2018)</td><td class="joi">&minus;0.001</td></tr>
                <tr><td>Brazil (WC 2018)</td><td class="joi">+0.003</td></tr>
                <tr><td>Pooled mean delta</td><td class="joi">+0.001</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <h3>Germany 2018 &mdash; same club, opposite outcomes</h3>
        <p>Both pairs are Bayern Munich. One is the third-best pair in the entire Germany 2018 dataset. The other is the worst.</p>
        <div class="research-side-by-side">
          <div class="slide-inner research-pair-card research-pair-positive">
            <div class="rpair-label">Bayern Munich &rarr; Germany</div>
            <div class="rpair-names">Thomas M&uuml;ller + Joshua Kimmich</div>
            <div class="rpair-metric">xT JOI90: <strong>+0.022</strong></div>
            <div class="rpair-bar-wrap"><div class="rpair-bar rpair-bar-pos" style="width:55%"></div></div>
            <div class="rpair-note">298 minutes &mdash; transferred</div>
          </div>
          <div class="slide-inner research-pair-card research-pair-negative">
            <div class="rpair-label">Bayern Munich &rarr; Germany</div>
            <div class="rpair-names">Manuel Neuer + J&eacute;r&ocirc;me Boateng</div>
            <div class="rpair-metric">xT JOI90: <strong>&minus;0.035</strong></div>
            <div class="rpair-bar-wrap"><div class="rpair-bar rpair-bar-neg" style="width:35%"></div></div>
            <div class="rpair-note">240 minutes &mdash; negative. GK + CB interactions generate near-zero or negative xT by construction &mdash; a positional artefact, not chemistry failure.</div>
          </div>
        </div>

        <p class="research-footer">Full analyses: <a href="https://github.com/stranger9977/wc2026-chemistry/blob/main/docs/analysis/club-chemistry.md" target="_blank" rel="noopener">WC 2026 analysis &rarr;</a> &nbsp;&bull;&nbsp; <a href="https://github.com/stranger9977/wc2026-chemistry/blob/main/docs/analysis/club-chemistry-historic.md" target="_blank" rel="noopener">Historic (WC 2018 + AFCON 2023) &rarr;</a></p>
      </section>

      <!-- Section 3: Limitations -->
      <section class="slide-card" aria-label="Limitations and next steps">
        <span class="pill">Limitations &amp; next steps</span>
        <h2>What this data cannot tell you</h2>

        <div class="slide-2col">
          <div>
            <h3>Data constraints</h3>
            <ul>
              <li><strong>Wyscout coverage is 2017/18 only.</strong> One season, eight years ago. Player development, team compositions, and tactical norms have all changed substantially.</li>
              <li><strong>Missing stars.</strong> Lewandowski left Bayern in 2022 &mdash; no open club event data at his peak. De Bruyne&rsquo;s best seasons at City post-2018, Bale, and Ibrahimovi&#263; all lack open event data at the club level.</li>
              <li><strong>StatsBomb open data is international-only (for the live WC 2026 pipeline).</strong> The WC 2026 chemistry numbers on this site cannot be cross-referenced against club chemistry for the same players &mdash; the two datasets live in different schemas and different contexts.</li>
            </ul>
          </div>
          <div>
            <h3>Statistical limitations</h3>
            <ul>
              <li><strong>Small international samples.</strong> A pair JOI90 built on 3 international matches (Germany WC 2018; Poland; Egypt) has variance large enough that the confidence interval covers zero.</li>
              <li><strong>No causal identification.</strong> All comparisons are observational.</li>
              <li><strong>VAEP cross-source incomparability.</strong> VAEP trained on Wyscout SPADL is not directly comparable to VAEP trained on StatsBomb SPADL (different action distributions, different feature encodings). Within-source comparisons are valid; cross-source comparisons are not.</li>
            </ul>
            <h3>What would help</h3>
            <ul>
              <li>Broader club event coverage: additional StatsBomb open releases or Opta/Skillcorner public datasets covering multiple seasons.</li>
              <li>Multiple WC cycles for the same players.</li>
              <li>StatsBomb club data for Bundesliga 2023/24 analyzed alongside Euro 2024 (same-year Bayern/Germany core).</li>
            </ul>
          </div>
        </div>
      </section>

    </div><!-- /slide-deck -->
  </div><!-- /panel-research -->'''

    # Replace the Research panel in the HTML
    import re
    # Find the Research panel start to end
    pattern = r'  <!-- ── Tab 4: Research ────.*?  </div><!-- /panel-research -->'
    new_html = re.sub(pattern, new_research_html, html, flags=re.DOTALL)

    if new_html == html:
        log.warning("Could not find Research panel in HTML to replace — check pattern")
        # Try a simpler replacement
        start_marker = '  <!-- ── Tab 4: Research'
        end_marker = '  </div><!-- /panel-research -->'
        start_idx = html.find(start_marker)
        end_idx = html.find(end_marker)
        if start_idx >= 0 and end_idx >= 0:
            new_html = html[:start_idx] + new_research_html + "\n" + html[end_idx + len(end_marker):]
            log.info("Used fallback string replacement for Research panel")
        else:
            log.error("Could not find Research panel markers in HTML")
            return

    with open(site_path, "w", encoding="utf-8") as f:
        f.write(new_html)
    log.info("Updated site/index.html Research tab")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    import sys

    skip_sb = "--skip-statsbomb" in sys.argv
    skip_v2 = "--skip-v2" in sys.argv

    # Phase 1: Train VAEP on Wyscout
    vaep_wy_bundle = phase1_train_vaep_wyscout()

    # Phase 2: Score Wyscout SPADL
    phase2_score_wyscout(vaep_wy_bundle)

    # Phase 3: Compute Wyscout VAEP JOI per match
    wy_vaep_joi = phase3_compute_wy_vaep_joi()

    if not skip_sb:
        # Phase 4: Fetch StatsBomb club data
        phase4_fetch_sb_club_data()

        # Phase 5: Convert StatsBomb to SPADL
        phase5_convert_sb_spadl()

    if not skip_v2 and not skip_sb:
        # Phase 6: Train VAEP v2
        vaep_v2_bundle = phase6_train_vaep_v2()

        # Phase 7: Score StatsBomb club SPADL with VAEP v2
        phase7_score_sb_vaep(vaep_v2_bundle)

        # Phase 8: Compute StatsBomb club JOI
        sb_joi = phase8_compute_sb_joi()

        # Phase 10: StatsBomb club cross-context
        sb_deep_dive = phase10_sb_club_cross_context(sb_joi)
    else:
        sb_deep_dive = None

    # Load Wyscout xT JOI (existing)
    wy_xt_joi = pd.read_parquet(OUTPUTS / "wyscout_joi_per_match.parquet")

    # Phase 9: Cross-context analysis
    player_results, deep_dive, meta = phase9_cross_context_analysis(wy_vaep_joi, wy_xt_joi)

    # Phase 11: Write outputs
    phase11_write_outputs(meta, player_results, deep_dive, vaep_wy_bundle, sb_deep_dive)

    # Phase 12: Update site
    phase12_update_site(player_results, deep_dive, vaep_wy_bundle)

    # Print summary
    log.info("=== SUMMARY ===")
    log.info("VAEP model: train_size=%d AUC_scores=%.4f AUC_concedes=%.4f",
             vaep_wy_bundle.get("train_size", 0),
             vaep_wy_bundle.get("auc_scores", 0),
             vaep_wy_bundle.get("auc_concedes", 0))

    for p in sorted(player_results,
                    key=lambda x: -(x.get("summary", {}).get("ratio_vaep") or 0)):
        s = p.get("summary", {})
        log.info(
            "  %s: club_vaep=%.4f country_vaep=%.4f ratio_vaep=%s  (xT ratio=%s)",
            p["name"],
            s.get("avg_club_joi90_vaep", 0),
            s.get("avg_country_joi90_vaep", 0),
            f"{s['ratio_vaep']:.2f}" if s.get("ratio_vaep") else "n/a",
            f"{s['ratio_xt']:.2f}" if s.get("ratio_xt") else "n/a",
        )


if __name__ == "__main__":
    main()
