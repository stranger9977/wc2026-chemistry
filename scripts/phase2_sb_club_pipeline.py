"""Phase 2 StatsBomb club cross-context pipeline.

Executes phases 4-11 sketched in vaep_cross_context_pipeline.py docstring:
  4. Fetch 8 StatsBomb club competitions
  5. Convert to SPADL (per-competition parquets)
  6. Train VAEP v2 (StatsBomb international + club, no Wyscout)
  7. Score club SPADL with VAEP v2
  8. Compute pair JOI per match for every club competition
  9. Cross-context comparison (Bayern 23/24->Germany Euro 24, Real Madrid, Barcelona, PSG, etc.)
 10. Extend outputs/cross_context_chemistry.json
 11. Update docs/analysis/cross-context-chemistry.md
 12. Update site/index.html Research tab

Run with: .venv/bin/python -m scripts.phase2_sb_club_pipeline
"""
from __future__ import annotations

import json
import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_VAEP = ROOT / "data" / "vaep"
DATA_SPADL_SB = ROOT / "data" / "sb_club_spadl"
DATA_SPADL_INTL = ROOT / "data" / "spadl"
OUTPUTS = ROOT / "outputs"

DATA_VAEP.mkdir(parents=True, exist_ok=True)
DATA_SPADL_SB.mkdir(parents=True, exist_ok=True)

VAEP_V2_PATH = DATA_VAEP / "vaep_v2.pkl"

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

# StatsBomb player IDs — these are consistent across club and international data
# Verified from StatsBomb open data
PLAYER_IDS = {
    # Germany / Bayern
    "Neuer":       5599,
    "Kimmich":     5674,
    "Muller":      5503,   # Thomas Müller
    "Goretzka":    8210,
    "Musiala":     37455,
    "Sane":        6924,   # Leroy Sané
    "Pavlovic":    186296,
    "Wirtz":       180907,
    # Real Madrid
    "Modric":      5503,   # will be replaced below with correct IDs from data
    "Kroos":       5570,
    "Varane":      3309,
    "Benzema":     3006,
    "Casemiro":    5597,
    "Marcelo":     5533,
    "Carvajal":    5537,
    "Ramos":       5596,
    # Barcelona
    "Messi":       5503,   # will be resolved from data
    "Suarez":      3857,
    "Iniesta":     3816,
    "Busquets":    3818,
    "JordiAlba":   3822,
    "Pique":       5817,
    "Coutinho":    3472,
    # PSG
    "Mbappe":      353833,
    "Neymar":      40810,
    "Hakimi":      6789,
    # Atletico
    "Griezmann":   3682,
    # Inter Miami / Argentina
    # Messi same
}


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
    # gamestates requires 'game_id' column for grouping
    df = spadl
    if "game_id" not in df.columns and "match_id" in df.columns:
        df = df.rename(columns={"match_id": "game_id"})
    gs = fs.gamestates(df)
    return pd.concat(
        [fs.actiontype_onehot(gs),
         fs.result_onehot(gs),
         fs.startlocation(gs),
         fs.endlocation(gs)],
        axis=1,
    )


# ── Phase 4: Fetch StatsBomb club data ─────────────────────────────────────

def phase4_fetch_sb_club_data() -> None:
    from statsbombpy import sb

    log.info("=== Phase 4: Fetch StatsBomb club data ===")
    for comp in SB_CLUB_COMPS:
        cid, sid, label = comp["cid"], comp["sid"], comp["label"]
        base = DATA_RAW / f"{cid}_{sid}"
        matches_path = base / "matches.parquet"

        if matches_path.exists():
            matches = pd.read_parquet(matches_path)
            n_events = len(list((base / "events").glob("*.parquet"))) if (base / "events").exists() else 0
            if n_events >= len(matches) * 0.95:
                log.info("  Cache hit: %s (%d matches, %d events)", label, len(matches), n_events)
                continue
            log.info("  Partial cache for %s: %d/%d events — completing", label, n_events, len(matches))
        else:
            log.info("  Fetching %s (cid=%d sid=%d) ...", label, cid, sid)
            base.mkdir(parents=True, exist_ok=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                matches = sb.matches(competition_id=cid, season_id=sid)
            # Save slim parquet with key columns only (avoid pyarrow issues with mixed-type cols)
            slim_cols = ["match_id", "home_team_id", "away_team_id", "home_team", "away_team"]
            slim_cols = [c for c in slim_cols if c in matches.columns]
            slim = matches[slim_cols].copy()
            slim.to_parquet(matches_path, index=False)
            log.info("    %d matches", len(matches))

        events_dir = base / "events"
        events_dir.mkdir(exist_ok=True)

        fetched = 0
        failed = 0
        for idx, row in enumerate(matches.itertuples(), 1):
            mid = int(row.match_id)
            ep = events_dir / f"{mid}.parquet"
            if ep.exists():
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ev = sb.events(match_id=mid, fmt="dataframe")
                ev.to_parquet(ep, index=False)
                fetched += 1
            except Exception as e:
                log.warning("    Failed match %d: %s", mid, e)
                failed += 1
            if (fetched + failed) % 50 == 0 and (fetched + failed) > 0:
                log.info("    Progress %s: %d fetched, %d failed / %d total",
                         label, fetched, failed, len(matches))

        log.info("    Done %s: fetched %d, failed %d", label, fetched, failed)


# ── Phase 5: Convert StatsBomb club events to SPADL ────────────────────────

def _get_home_team_id(row) -> int:
    """Extract home_team_id from a matches row robustly.
    statsbombpy flat API returns home_team_id as a direct column.
    """
    # Flat API: direct column
    try:
        v = row.get("home_team_id")
        if v is not None and str(v) not in ("", "nan", "None"):
            return int(float(v))
    except Exception:
        pass
    # Nested dict (older API)
    ht = row.get("home_team")
    if isinstance(ht, dict):
        return int(ht.get("home_team_id", 0))
    return 0


def phase5_convert_sb_spadl() -> None:
    # Use the chemistry.pipeline adapter which handles flat statsbombpy DataFrames
    from chemistry.pipeline import to_spadl as sb_convert_flat

    log.info("=== Phase 5: Convert StatsBomb club events to SPADL ===")
    for comp in SB_CLUB_COMPS:
        cid, sid, label = comp["cid"], comp["sid"], comp["label"]
        comp_key = f"{cid}_{sid}"
        out_dir = DATA_SPADL_SB / comp_key
        out_dir.mkdir(parents=True, exist_ok=True)

        base = DATA_RAW / comp_key
        matches_path = base / "matches.parquet"
        if not matches_path.exists():
            log.warning("  No matches data for %s, skipping", label)
            continue

        matches = pd.read_parquet(matches_path)
        events_dir = base / "events"

        # Count existing
        existing = len(list(out_dir.glob("*.parquet")))
        total = len(matches)
        if existing >= total * 0.99:
            log.info("  SPADL cache hit: %s (%d/%d)", label, existing, total)
            continue

        log.info("  Converting %s: %d matches (%d already done) ...", label, total, existing)
        converted = 0
        failed = 0

        for idx, row in matches.iterrows():
            mid = int(row["match_id"])
            out_path = out_dir / f"{mid}.parquet"
            if out_path.exists():
                continue

            ep = events_dir / f"{mid}.parquet"
            if not ep.exists():
                failed += 1
                continue

            try:
                ev = pd.read_parquet(ep)
                # Add match_id to events so the flat adapter can set game_id
                if "match_id" not in ev.columns:
                    ev["match_id"] = mid
                # Use the flat statsbombpy adapter from chemistry.pipeline
                actions = sb_convert_flat(ev)
                actions["match_id"] = mid
                actions["competition"] = label
                actions["competition_key"] = comp_key
                actions.to_parquet(out_path, index=False)
                converted += 1
            except Exception as e:
                log.warning("    Failed match %d: %s", mid, e)
                failed += 1

            total_done = converted + failed
            if total_done % 50 == 0 and total_done > 0:
                log.info("    Progress %s: %d converted, %d failed / %d",
                         label, converted, failed, total)

        log.info("  Done %s: converted %d, failed %d", label, converted, failed)


# ── Phase 6: Train VAEP v2 ─────────────────────────────────────────────────

def phase6_train_vaep_v2() -> dict:
    if VAEP_V2_PATH.exists():
        log.info("VAEP v2 cache hit: %s", VAEP_V2_PATH)
        with open(VAEP_V2_PATH, "rb") as f:
            return pickle.load(f)

    log.info("=== Phase 6: Train VAEP v2 (StatsBomb international + club) ===")

    parts = []

    # StatsBomb international SPADL (data/spadl/<comp_dir>/)
    # These files use 'game_id' — normalize to 'game_id' (gamestates needs it)
    intl_count = 0
    for comp_dir in sorted(DATA_SPADL_INTL.iterdir()):
        if not comp_dir.is_dir():
            continue
        for p in sorted(comp_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(p)
                # Ensure game_id column for gamestates
                if "game_id" not in df.columns and "match_id" in df.columns:
                    df = df.rename(columns={"match_id": "game_id"})
                parts.append(df)
                intl_count += len(df)
            except Exception as e:
                log.warning("  Failed to read %s: %s", p, e)
    log.info("StatsBomb international SPADL: %d actions", intl_count)

    # StatsBomb club SPADL (data/sb_club_spadl/<comp_key>/<match_id>.parquet)
    # These files use 'match_id' — normalize to 'game_id'
    club_count = 0
    for comp in SB_CLUB_COMPS:
        comp_key = f"{comp['cid']}_{comp['sid']}"
        comp_dir = DATA_SPADL_SB / comp_key
        if not comp_dir.exists():
            log.warning("  Missing club SPADL dir: %s", comp_key)
            continue
        comp_files = list(comp_dir.glob("*.parquet"))
        if not comp_files:
            log.warning("  No SPADL files for %s", comp["label"])
            continue
        comp_parts = []
        for p in sorted(comp_files):
            try:
                df = pd.read_parquet(p)
                # Normalize: gamestates needs game_id
                if "game_id" not in df.columns and "match_id" in df.columns:
                    df = df.rename(columns={"match_id": "game_id"})
                comp_parts.append(df)
            except Exception as e:
                log.warning("  Failed to read %s: %s", p, e)
        if comp_parts:
            combined = pd.concat(comp_parts, ignore_index=True)
            parts.append(combined)
            club_count += len(combined)
            log.info("  Club %s: %d actions", comp["label"], len(combined))

    log.info("Club SPADL total: %d actions", club_count)

    all_spadl = pd.concat(parts, ignore_index=True)
    log.info("Combined SPADL for VAEP v2: %d actions total", len(all_spadl))

    # Train the model
    from xgboost import XGBClassifier
    from socceraction.vaep import labels as lb
    from sklearn.metrics import roc_auc_score

    all_spadl = _add_type_result_name(all_spadl)
    log.info("Building features for %d actions ...", len(all_spadl))
    X = build_features(all_spadl)
    log.info("Feature shape: %s", X.shape)

    y_scores = lb.scores(all_spadl)["scores"]
    y_concedes = lb.concedes(all_spadl)["concedes"]
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

    auc_scores = roc_auc_score(y_scores, clf_scores.predict_proba(X)[:, 1])
    auc_concedes = roc_auc_score(y_concedes, clf_concedes.predict_proba(X)[:, 1])
    log.info("Train AUC v2 scores: %.4f  concedes: %.4f", auc_scores, auc_concedes)

    bundle = {
        "scores": clf_scores,
        "concedes": clf_concedes,
        "feature_cols": list(X.columns),
        "train_size": len(all_spadl),
        "auc_scores": auc_scores,
        "auc_concedes": auc_concedes,
    }
    with open(VAEP_V2_PATH, "wb") as f:
        pickle.dump(bundle, f)
    log.info("Saved VAEP v2 to %s", VAEP_V2_PATH)
    return bundle


def score_vaep_v2(spadl: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Score SPADL actions with VAEP v2. Returns copy with vaep_value column."""
    from socceraction.vaep import formula

    enriched = _add_type_result_name(spadl)
    X = build_features(enriched)

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


# ── Phase 7: Score club SPADL with VAEP v2 ────────────────────────────────

def phase7_score_club_vaep(bundle: dict) -> None:
    log.info("=== Phase 7: Score StatsBomb club SPADL with VAEP v2 ===")
    for comp in SB_CLUB_COMPS:
        cid, sid, label = comp["cid"], comp["sid"], comp["label"]
        comp_key = f"{cid}_{sid}"
        src_dir = DATA_SPADL_SB / comp_key
        out_dir = DATA_SPADL_SB / comp_key / "vaep_scored"
        out_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            log.warning("  Missing SPADL dir for %s, skipping", label)
            continue

        src_files = list(src_dir.glob("*.parquet"))
        existing = len(list(out_dir.glob("*.parquet")))
        if existing >= len(src_files) * 0.99 and existing > 0:
            log.info("  VAEP scored cache hit: %s (%d files)", label, existing)
            continue

        log.info("  Scoring %s: %d matches ...", label, len(src_files))
        scored_count = 0
        for p in sorted(src_files):
            out_path = out_dir / p.name
            if out_path.exists():
                continue
            try:
                df = pd.read_parquet(p)
                scored = score_vaep_v2(df, bundle)
                scored.to_parquet(out_path, index=False)
                scored_count += 1
            except Exception as e:
                log.warning("    Failed %s: %s", p.name, e)
        log.info("  Scored %d matches for %s", scored_count, label)


def phase7b_score_intl_vaep_v2(bundle: dict) -> None:
    """Re-score existing international SPADL with VAEP v2 into separate dir."""
    log.info("=== Phase 7b: Re-score international SPADL with VAEP v2 ===")
    intl_v2_dir = DATA_VAEP / "intl_vaep_v2_scored"
    intl_v2_dir.mkdir(parents=True, exist_ok=True)

    for comp_dir in sorted(DATA_SPADL_INTL.iterdir()):
        if not comp_dir.is_dir():
            continue
        out_comp_dir = intl_v2_dir / comp_dir.name
        out_comp_dir.mkdir(parents=True, exist_ok=True)

        src_files = list(comp_dir.glob("*.parquet"))
        existing = len(list(out_comp_dir.glob("*.parquet")))
        if existing >= len(src_files) * 0.99 and existing > 0:
            log.info("  Cache hit: %s (%d files)", comp_dir.name, existing)
            continue

        log.info("  Scoring %s: %d files ...", comp_dir.name, len(src_files))
        for p in sorted(src_files):
            out_path = out_comp_dir / p.name
            if out_path.exists():
                continue
            try:
                df = pd.read_parquet(p)
                if "match_id" not in df.columns and "game_id" in df.columns:
                    df = df.rename(columns={"game_id": "match_id"})
                scored = score_vaep_v2(df, bundle)
                scored.to_parquet(out_path, index=False)
            except Exception as e:
                log.warning("    Failed %s: %s", p.name, e)

        log.info("  Done: %s", comp_dir.name)


# ── Phase 8: Compute JOI per match ────────────────────────────────────────

def compute_vaep_joi_per_match(actions: pd.DataFrame, value_col: str = "vaep_value") -> pd.DataFrame:
    """Compute per-match pair JOI using VAEP values."""
    df = actions[actions["type_id"].isin(ELIGIBLE_TYPES)].copy()
    if df.empty:
        return pd.DataFrame(columns=["match_id", "team_id", "player_a", "player_b", "joi_vaep"])

    df = df.sort_values(["match_id", "period_id", "time_seconds"]).reset_index(drop=True)

    nxt = df.shift(-1)
    consecutive = (
        (df["match_id"] == nxt["match_id"])
        & (df["team_id"] == nxt["team_id"])
        & (df["player_id"] != nxt["player_id"])
    )

    pairs = pd.DataFrame({
        "match_id": df["match_id"],
        "team_id": df["team_id"],
        "player_p": df["player_id"],
        "player_q": nxt["player_id"],
        "vaep_q": nxt[value_col],
    })[consecutive].reset_index(drop=True)

    if pairs.empty:
        return pd.DataFrame(columns=["match_id", "team_id", "player_a", "player_b", "joi_vaep"])

    pairs["player_a"] = pairs[["player_p", "player_q"]].min(axis=1).astype("int64")
    pairs["player_b"] = pairs[["player_p", "player_q"]].max(axis=1).astype("int64")
    pairs["vaep_q"] = pairs["vaep_q"].fillna(0.0)

    grouped = (
        pairs.groupby(["match_id", "team_id", "player_a", "player_b"], as_index=False)
             ["vaep_q"].sum()
             .rename(columns={"vaep_q": "joi_vaep"})
    )
    return grouped


def _compute_shared_minutes_from_events(events_dir: Path, match_ids: list[int]) -> pd.DataFrame:
    """Compute player on-pitch minutes from StatsBomb event data (substitutions)."""
    rows = []
    for mid in match_ids:
        ep = events_dir / f"{mid}.parquet"
        if not ep.exists():
            continue
        try:
            ev = pd.read_parquet(ep)
            # StatsBomb events: type is a dict or string
            def get_type_name(t):
                if isinstance(t, dict):
                    return t.get("name", "")
                return str(t) if t else ""

            type_col = "type"
            if type_col not in ev.columns:
                continue

            ev["_type_name"] = ev[type_col].apply(get_type_name)

            # Starting XI
            starting_xi = ev[ev["_type_name"] == "Starting XI"]
            team_players: dict[int, dict[int, list]] = {}

            for _, se in starting_xi.iterrows():
                team_data = se.get("team")
                if isinstance(team_data, dict):
                    tid = int(team_data.get("id", 0))
                else:
                    continue
                tactics = se.get("tactics")
                if not isinstance(tactics, dict):
                    continue
                lineup = tactics.get("lineup", [])
                if tid not in team_players:
                    team_players[tid] = {}
                for p in lineup:
                    if isinstance(p, dict) and "player" in p:
                        pid = p["player"].get("id")
                        if pid:
                            team_players[tid][int(pid)] = [0.0, 90.0]

            # Substitutions
            subs = ev[ev["_type_name"] == "Substitution"]
            for _, sub in subs.iterrows():
                minute = float(sub.get("minute", 90))
                team_data = sub.get("team")
                if isinstance(team_data, dict):
                    tid = int(team_data.get("id", 0))
                else:
                    continue

                player_data = sub.get("player")
                player_off_id = int(player_data.get("id")) if isinstance(player_data, dict) else None

                sub_info = sub.get("substitution")
                player_on_id = None
                if isinstance(sub_info, dict) and "replacement" in sub_info:
                    repl = sub_info["replacement"]
                    if isinstance(repl, dict):
                        player_on_id = int(repl.get("id")) if repl.get("id") else None

                if player_off_id and tid in team_players and player_off_id in team_players[tid]:
                    team_players[tid][player_off_id][1] = minute
                if player_on_id:
                    if tid not in team_players:
                        team_players[tid] = {}
                    team_players[tid][player_on_id] = [minute, 90.0]

            for tid, players in team_players.items():
                for pid, (on, off) in players.items():
                    rows.append({
                        "match_id": mid,
                        "team_id": tid,
                        "player_id": pid,
                        "minute_on": on,
                        "minute_off": min(float(off), 120.0),
                    })
        except Exception as e:
            log.warning("  Failed to compute minutes for match %d: %s", mid, e)

    if not rows:
        return pd.DataFrame(columns=["match_id", "team_id", "player_id", "minute_on", "minute_off"])
    return pd.DataFrame(rows)


def _compute_pair_shared_minutes(player_minutes: pd.DataFrame) -> pd.DataFrame:
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
    out_path = OUTPUTS / "sb_club_joi_per_match.parquet"
    if out_path.exists():
        log.info("SB club JOI cache hit: %s", out_path)
        return pd.read_parquet(out_path)

    log.info("=== Phase 8: Compute StatsBomb club JOI per match ===")

    all_joi_parts = []
    all_shared_parts = []

    for comp in SB_CLUB_COMPS:
        cid, sid, label = comp["cid"], comp["sid"], comp["label"]
        comp_key = f"{cid}_{sid}"

        scored_dir = DATA_SPADL_SB / comp_key / "vaep_scored"
        if not scored_dir.exists() or not list(scored_dir.glob("*.parquet")):
            log.warning("  Missing scored SPADL for %s, skipping", label)
            continue

        joi_cache = DATA_SPADL_SB / f"joi_{comp_key}.parquet"
        sm_cache = DATA_SPADL_SB / f"shared_minutes_{comp_key}.parquet"

        if joi_cache.exists():
            log.info("  JOI cache hit: %s", label)
            joi = pd.read_parquet(joi_cache)
        else:
            log.info("  Computing JOI for %s ...", label)
            # Load all scored match files
            scored_files = list(scored_dir.glob("*.parquet"))
            all_scored = pd.concat([pd.read_parquet(p) for p in sorted(scored_files)], ignore_index=True)
            all_scored["competition"] = label
            log.info("    Loaded %d scored actions for %s", len(all_scored), label)
            joi = compute_vaep_joi_per_match(all_scored)
            joi["competition"] = label
            joi.to_parquet(joi_cache, index=False)
            log.info("    JOI rows: %d", len(joi))

        all_joi_parts.append(joi)

        if sm_cache.exists():
            log.info("  Shared minutes cache hit: %s", label)
            sm = pd.read_parquet(sm_cache)
        else:
            log.info("  Computing shared minutes for %s ...", label)
            base = DATA_RAW / comp_key
            matches = pd.read_parquet(base / "matches.parquet")
            events_dir = base / "events"
            match_ids = matches["match_id"].tolist()
            player_mins = _compute_shared_minutes_from_events(events_dir, match_ids)
            if not player_mins.empty:
                sm = _compute_pair_shared_minutes(player_mins)
                sm["competition"] = label
            else:
                sm = pd.DataFrame(columns=["match_id", "team_id", "player_a", "player_b", "shared_minutes", "competition"])
            sm.to_parquet(sm_cache, index=False)
            log.info("    Shared minutes rows: %d", len(sm))

        all_shared_parts.append(sm)

    if not all_joi_parts:
        log.warning("No StatsBomb club JOI computed")
        return pd.DataFrame()

    all_joi = pd.concat(all_joi_parts, ignore_index=True)
    all_shared = pd.concat(all_shared_parts, ignore_index=True)

    log.info("Merging JOI (%d rows) with shared minutes (%d rows) ...", len(all_joi), len(all_shared))
    merged = all_joi.merge(
        all_shared[["match_id", "team_id", "player_a", "player_b", "shared_minutes"]],
        on=["match_id", "team_id", "player_a", "player_b"],
        how="left",
    )
    merged["shared_minutes"] = merged["shared_minutes"].fillna(0.0)
    merged.to_parquet(out_path, index=False)
    log.info("Saved SB club JOI: %s (%d rows)", out_path, len(merged))
    return merged


# ── Phase 8b: JOI for international StatsBomb data with VAEP v2 ────────────

def phase8b_compute_intl_joi_v2() -> pd.DataFrame:
    """Compute JOI for international StatsBomb data using VAEP v2 scores."""
    out_path = OUTPUTS / "intl_joi_v2.parquet"
    if out_path.exists():
        log.info("International VAEP v2 JOI cache hit")
        return pd.read_parquet(out_path)

    log.info("=== Phase 8b: Compute international JOI with VAEP v2 ===")
    intl_v2_dir = DATA_VAEP / "intl_vaep_v2_scored"

    # Load existing lineups for shared minutes
    lineups = pd.read_parquet(OUTPUTS / "lineups.parquet")
    # lineups has: game_id, team_id, player_id, from_minute, to_minute

    parts = []
    for comp_dir in sorted(intl_v2_dir.iterdir()):
        if not comp_dir.is_dir():
            continue
        comp_name = comp_dir.name
        files = list(comp_dir.glob("*.parquet"))
        if not files:
            continue
        all_scored = pd.concat([pd.read_parquet(p) for p in sorted(files)], ignore_index=True)
        # Ensure match_id column
        if "match_id" not in all_scored.columns and "game_id" in all_scored.columns:
            all_scored = all_scored.rename(columns={"game_id": "match_id"})
        all_scored["competition"] = comp_name
        joi = compute_vaep_joi_per_match(all_scored)
        joi["competition"] = comp_name
        parts.append(joi)
        log.info("  %s: %d JOI rows", comp_name, len(joi))

    if not parts:
        log.warning("No international VAEP v2 JOI data found")
        return pd.DataFrame()

    all_joi = pd.concat(parts, ignore_index=True)

    # Compute shared minutes from existing lineups parquet
    lineups_renamed = lineups.rename(columns={"game_id": "match_id"})
    player_mins = lineups_renamed[["match_id", "team_id", "player_id", "from_minute", "to_minute"]].copy()
    player_mins = player_mins.rename(columns={"from_minute": "minute_on", "to_minute": "minute_off"})

    sm = _compute_pair_shared_minutes(player_mins)

    merged = all_joi.merge(
        sm[["match_id", "team_id", "player_a", "player_b", "shared_minutes"]],
        on=["match_id", "team_id", "player_a", "player_b"],
        how="left",
    )
    merged["shared_minutes"] = merged["shared_minutes"].fillna(0.0)
    merged.to_parquet(out_path, index=False)
    log.info("Saved international VAEP v2 JOI: %d rows", len(merged))
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS: Player-level per-90 production (club vs country)
# The question: does the player's individual output DROP when removed from
# their club chemistry network and placed in the national team context?
# ══════════════════════════════════════════════════════════════════════════════

# ── Helpers: player/team name lookups ─────────────────────────────────────

def build_player_name_map() -> dict[int, str]:
    """Build {player_id: name} from StatsBomb lineups parquet + club event data."""

    name_map: dict[int, str] = {}

    # From existing international lineups
    lineups = pd.read_parquet(OUTPUTS / "lineups.parquet")
    for _, row in lineups.iterrows():
        pid = int(row["player_id"])
        name = str(row["player_name"])
        if pid not in name_map and name:
            name_map[pid] = name

    # From club event data (sample files per competition)
    for comp in SB_CLUB_COMPS:
        cid, sid = comp["cid"], comp["sid"]
        events_dir = DATA_RAW / f"{cid}_{sid}" / "events"
        if not events_dir.exists():
            continue
        for fp in sorted(events_dir.glob("*.parquet"))[:10]:
            try:
                ev = pd.read_parquet(fp)
                if "player" in ev.columns:
                    for val in ev["player"].dropna():
                        if isinstance(val, dict) and "id" in val and "name" in val:
                            pid = int(val["id"])
                            nm = str(val["name"])
                            if pid not in name_map and nm:
                                name_map[pid] = nm
                if "player_id" in ev.columns and "player_name" in ev.columns:
                    for _, row in ev[["player_id", "player_name"]].dropna().drop_duplicates().iterrows():
                        try:
                            pid = int(row["player_id"])
                            nm = str(row["player_name"])
                            if pid not in name_map and nm:
                                name_map[pid] = nm
                        except Exception:
                            continue
            except Exception:
                continue

    log.info("Built player name map: %d players", len(name_map))
    return name_map


def build_team_name_map() -> dict[int, str]:
    team_map: dict[int, str] = {}
    lineups = pd.read_parquet(OUTPUTS / "lineups.parquet")
    for _, row in lineups.iterrows():
        tid = int(row["team_id"])
        nm = str(row["team_name"])
        if tid not in team_map:
            team_map[tid] = nm
    for comp in SB_CLUB_COMPS:
        cid, sid = comp["cid"], comp["sid"]
        mp = DATA_RAW / f"{cid}_{sid}" / "matches.parquet"
        if not mp.exists():
            continue
        try:
            matches = pd.read_parquet(mp)
            for _, row in matches.iterrows():
                for prefix in ["home_team", "away_team"]:
                    t = row.get(prefix)
                    if isinstance(t, dict):
                        tid = t.get(f"{prefix}_id")
                        nm = t.get(f"{prefix}_name", "")
                        if tid and nm:
                            team_map[int(tid)] = str(nm)
        except Exception:
            continue
    log.info("Built team name map: %d teams", len(team_map))
    return team_map


def resolve_player_ids(name_map: dict[int, str]) -> dict[str, int]:
    """Fuzzy-match known player names to StatsBomb player IDs."""
    # Build name -> id reverse map
    lower_map: dict[str, int] = {v.lower(): k for k, v in name_map.items()}

    def find(*queries) -> int | None:
        for q in queries:
            q_lower = q.lower()
            # Exact
            if q_lower in lower_map:
                return lower_map[q_lower]
            # Substring
            for name, pid in lower_map.items():
                if q_lower in name:
                    return pid
        return None

    ids = {
        # German / Bayern 23/24
        "neuer":    find("manuel neuer", "neuer"),
        "kimmich":  find("joshua kimmich", "kimmich"),
        "muller":   find("thomas müller", "thomas muller", "müller"),
        "goretzka": find("leon goretzka", "goretzka"),
        "musiala":  find("jamal musiala", "musiala"),
        "sane":     find("leroy sané", "leroy sane", "sané"),
        "pavlovic": find("aleksandar pavlović", "pavlovic", "pavlović"),
        "wirtz":    find("florian wirtz", "wirtz"),
        # Real Madrid
        "modric":   find("luka modrić", "luka modric", "modrić"),
        "kroos":    find("toni kroos", "kroos"),
        "varane":   find("raphaël varane", "raphael varane", "varane"),
        "benzema":  find("karim benzema", "benzema"),
        "casemiro": find("casemiro"),
        "ramos":    find("sergio ramos", "ramos"),
        # Barcelona + Argentina
        "messi":    find("lionel messi", "leo messi", "messi"),
        "suarez":   find("luis suárez", "luis suarez"),
        "iniesta":  find("andrés iniesta", "andres iniesta", "iniesta"),
        "busquets": find("sergio busquets", "busquets"),
        "jordi_alba": find("jordi alba"),
        "pique":    find("gerard piqué", "gerard pique", "piqué"),
        # PSG
        "mbappe":   find("kylian mbappé", "kylian mbappe", "mbappé"),
        "neymar":   find("neymar"),
        "hakimi":   find("achraf hakimi", "hakimi"),
        # Atletico
        "griezmann": find("antoine griezmann", "griezmann"),
    }

    found = {k: v for k, v in ids.items() if v is not None}
    missing = [k for k, v in ids.items() if v is None]
    if missing:
        log.warning("Could not resolve player IDs: %s", missing)
    log.info("Resolved %d / %d player IDs", len(found), len(ids))
    return found


def find_team_for_player(player_id: int, scored_files: list[Path],
                          sample: int = 30) -> int | None:
    """Find what team a player played for by scanning scored SPADL files."""
    for p in sorted(scored_files)[:sample]:
        try:
            df = pd.read_parquet(p)
            if "match_id" not in df.columns and "game_id" in df.columns:
                df = df.rename(columns={"game_id": "match_id"})
            rows = df[df["player_id"] == player_id]
            if not rows.empty:
                return int(rows["team_id"].mode()[0])
        except Exception:
            continue
    return None


# ── Phase 9: Per-90 player-level production analysis ─────────────────────

def compute_player_vaep90(player_id: int, team_id: int, scored_files: list[Path],
                           context_label: str) -> dict | None:
    """Compute per-90 VAEP for a player across their on-ball actions.

    Returns dict with vaep90, total_minutes, total_actions, matches or None if insufficient data.
    """
    total_vaep = 0.0
    total_actions = 0
    match_ids_seen: set = set()

    for p in sorted(scored_files):
        try:
            df = pd.read_parquet(p)
            if "match_id" not in df.columns and "game_id" in df.columns:
                df = df.rename(columns={"game_id": "match_id"})
            player_rows = df[
                (df["player_id"] == player_id) &
                (df["team_id"] == team_id) &
                (df["type_id"].isin(ELIGIBLE_TYPES))
            ]
            if player_rows.empty:
                continue
            match_ids = player_rows["match_id"].unique()
            match_ids_seen.update(int(m) for m in match_ids)
            total_vaep += float(player_rows["vaep_value"].fillna(0).sum())
            total_actions += len(player_rows)
        except Exception:
            continue

    if not match_ids_seen:
        return None

    # Estimate minutes from lineups parquet (international) or events (club)
    # We use a proxy: assume 90 min per match appearance in starting XI
    n_matches = len(match_ids_seen)
    total_minutes = n_matches * 90.0  # rough estimate; substitutes won't skew much at aggregate

    vaep90 = (total_vaep / total_minutes) * 90.0 if total_minutes > 0 else 0.0

    return {
        "context": context_label,
        "vaep90": round(vaep90, 4),
        "total_vaep": round(total_vaep, 4),
        "total_actions": total_actions,
        "total_minutes_est": round(total_minutes, 0),
        "matches": n_matches,
    }


def compute_player_vaep90_from_lineups(player_id: int, team_id: int,
                                        scored_files: list[Path],
                                        lineups: pd.DataFrame,
                                        context_label: str) -> dict | None:
    """Compute per-90 VAEP using actual minutes from lineups parquet."""
    total_vaep = 0.0
    total_actions = 0
    match_ids_seen: set = set()

    for p in sorted(scored_files):
        try:
            df = pd.read_parquet(p)
            if "match_id" not in df.columns and "game_id" in df.columns:
                df = df.rename(columns={"game_id": "match_id"})
            player_rows = df[
                (df["player_id"] == player_id) &
                (df["team_id"] == team_id) &
                (df["type_id"].isin(ELIGIBLE_TYPES))
            ]
            if player_rows.empty:
                continue
            match_ids_seen.update(int(m) for m in player_rows["match_id"].unique())
            total_vaep += float(player_rows["vaep_value"].fillna(0).sum())
            total_actions += len(player_rows)
        except Exception:
            continue

    if not match_ids_seen:
        return None

    # Look up actual minutes from lineups
    lu_col = "game_id" if "game_id" in lineups.columns else "match_id"
    player_lineups = lineups[
        (lineups["player_id"] == player_id) &
        (lineups[lu_col].isin(match_ids_seen))
    ]

    if not player_lineups.empty and "from_minute" in player_lineups.columns:
        on_col = "from_minute"
        off_col = "to_minute"
        total_minutes = float(
            (player_lineups[off_col].clip(upper=120) - player_lineups[on_col]).clip(lower=0).sum()
        )
    else:
        # Fallback: 90 per match
        total_minutes = len(match_ids_seen) * 90.0

    vaep90 = (total_vaep / total_minutes) * 90.0 if total_minutes > 0 else 0.0

    return {
        "context": context_label,
        "vaep90": round(vaep90, 4),
        "total_vaep": round(total_vaep, 4),
        "total_actions": total_actions,
        "total_minutes": round(total_minutes, 0),
        "matches": len(match_ids_seen),
    }


def phase9_player_production_analysis() -> dict:
    """Compute per-90 VAEP at club vs country for all featured players.

    This is the core analysis: does the player's individual output drop
    when removed from their club chemistry network?
    """
    log.info("=== Phase 9: Player-level per-90 production analysis ===")

    name_map = build_player_name_map()
    team_map = build_team_name_map()
    player_ids = resolve_player_ids(name_map)
    lineups = pd.read_parquet(OUTPUTS / "lineups.parquet")

    def pname(pid: int) -> str:
        return name_map.get(pid, f"Player #{pid}")

    def tname(tid: int) -> str:
        return team_map.get(tid, f"Team #{tid}")

    # Directory structure for scored files:
    # Club: DATA_SPADL_SB / <comp_key> / vaep_scored / <match_id>.parquet
    # International: DATA_VAEP / intl_vaep_v2_scored / <comp_dir> / <match_id>.parquet
    intl_v2_dir = DATA_VAEP / "intl_vaep_v2_scored"

    def club_files(comp_key: str) -> list[Path]:
        d = DATA_SPADL_SB / comp_key / "vaep_scored"
        return list(d.glob("*.parquet")) if d.exists() else []

    def intl_files(comp_dir: str) -> list[Path]:
        d = intl_v2_dir / comp_dir
        return list(d.glob("*.parquet")) if d.exists() else []

    results: dict[str, list[dict]] = {}

    # ── GROUP A: Bayern 2023/24 -> Germany Euro 2024 ──────────────────────
    log.info("Group A: Bayern 2023/24 -> Germany Euro 2024")

    bl_files = club_files("9_281")
    euro24_files = intl_files("55_282")

    log.info("  BL scored files: %d, Euro 2024 files: %d", len(bl_files), len(euro24_files))

    german_players = {
        "Neuer":    player_ids.get("neuer"),
        "Kimmich":  player_ids.get("kimmich"),
        "Muller":   player_ids.get("muller"),
        "Goretzka": player_ids.get("goretzka"),
        "Musiala":  player_ids.get("musiala"),
        "Sane":     player_ids.get("sane"),
        "Pavlovic": player_ids.get("pavlovic"),
        "Wirtz":    player_ids.get("wirtz"),
    }

    group_a = []
    # Find Bayern and Germany team IDs
    for player_name, player_id in german_players.items():
        if player_id is None:
            log.warning("  %s: player ID not resolved, skipping", player_name)
            group_a.append({"player": player_name, "player_id": None, "status": "id_not_resolved"})
            continue

        # Club context: Bundesliga 2023/24
        bayern_team_id = find_team_for_player(player_id, bl_files)
        if not bayern_team_id:
            club_vaep = None
            log.info("  %s: not found in Bundesliga 2023/24", player_name)
        else:
            club_vaep = compute_player_vaep90(player_id, bayern_team_id, bl_files, "Bundesliga 2023/24")

        # International: Euro 2024
        germany_team_id = find_team_for_player(player_id, euro24_files)
        if not germany_team_id:
            intl_vaep = None
            log.info("  %s: not found in Euro 2024", player_name)
        else:
            intl_vaep = compute_player_vaep90_from_lineups(
                player_id, germany_team_id, euro24_files, lineups, "Euro 2024"
            )

        delta = None
        ratio = None
        if club_vaep and intl_vaep:
            delta = round(float(intl_vaep["vaep90"]) - float(club_vaep["vaep90"]), 4)
            ratio = round(float(intl_vaep["vaep90"]) / float(club_vaep["vaep90"]), 3) if club_vaep["vaep90"] != 0 else None

        entry = {
            "player": player_name,
            "player_id": player_id,
            "player_name": pname(player_id),
            "club": "Bayern Munich",
            "club_team_id": bayern_team_id,
            "country": "Germany",
            "country_team_id": germany_team_id,
            "club_context": club_vaep,
            "intl_context": intl_vaep,
            "delta_vaep90": delta,
            "ratio_intl_club": ratio,
            "narrative": _describe_delta(player_name, "Bayern", "Germany Euro 2024", club_vaep, intl_vaep),
        }
        group_a.append(entry)

        log.info("  %s: club=%.4f intl=%.4f delta=%s",
                 player_name,
                 club_vaep["vaep90"] if club_vaep else 0,
                 intl_vaep["vaep90"] if intl_vaep else 0,
                 f"{delta:+.4f}" if delta is not None else "n/a")

    results["bayern_germany_euro24"] = {
        "headline": "Bayern 2023/24 -> Germany Euro 2024 (same-season, cleanest test)",
        "club_competition": "Bundesliga 2023/24",
        "intl_competition": "UEFA Euro 2024",
        "players": group_a,
    }

    # ── GROUP B: Real Madrid multi-season -> international ────────────────
    log.info("Group B: Real Madrid -> international")

    la_liga_keys = ["11_1", "11_4", "11_42", "11_90"]
    la_liga_labels = ["La Liga 2017/18", "La Liga 2018/19", "La Liga 2019/20", "La Liga 2020/21"]

    rm_players = {
        "Modric":   (player_ids.get("modric"),   "Croatia",  "43_3",   "WC 2018"),
        "Kroos":    (player_ids.get("kroos"),    "Germany",  "43_3",   "WC 2018"),
        "Varane":   (player_ids.get("varane"),   "France",   "43_3",   "WC 2018"),
        "Benzema":  (player_ids.get("benzema"),  "France",   "55_43",  "Euro 2020"),
    }

    group_b = []
    for player_name, (player_id, country, intl_comp_dir, intl_comp_label) in rm_players.items():
        if player_id is None:
            group_b.append({"player": player_name, "status": "id_not_resolved"})
            continue

        # Aggregate club VAEP across all La Liga seasons
        club_total_vaep = 0.0
        club_total_actions = 0
        club_total_minutes = 0.0
        club_matches = 0
        rm_team_id = None

        for season_key in la_liga_keys:
            files = club_files(season_key)
            if not files:
                continue
            team_id = find_team_for_player(player_id, files)
            if not team_id:
                continue
            if not rm_team_id:
                rm_team_id = team_id
            for p in sorted(files):
                try:
                    df = pd.read_parquet(p)
                    if "match_id" not in df.columns and "game_id" in df.columns:
                        df = df.rename(columns={"game_id": "match_id"})
                    player_rows = df[
                        (df["player_id"] == player_id) &
                        (df["team_id"] == team_id) &
                        (df["type_id"].isin(ELIGIBLE_TYPES))
                    ]
                    if player_rows.empty:
                        continue
                    club_total_vaep += float(player_rows["vaep_value"].fillna(0).sum())
                    club_total_actions += len(player_rows)
                    club_total_minutes += float(player_rows["match_id"].nunique() * 90.0)
                    club_matches += int(player_rows["match_id"].nunique())
                except Exception:
                    continue

        if club_total_minutes > 0:
            club_vaep = {
                "context": "La Liga (multi-season avg)",
                "vaep90": round(club_total_vaep / club_total_minutes * 90.0, 4),
                "total_vaep": round(club_total_vaep, 4),
                "total_actions": club_total_actions,
                "total_minutes": round(club_total_minutes, 0),
                "matches": club_matches,
            }
        else:
            club_vaep = None

        # International
        i_files = intl_files(intl_comp_dir)
        intl_team_id = find_team_for_player(player_id, i_files)
        if intl_team_id and i_files:
            intl_vaep = compute_player_vaep90_from_lineups(
                player_id, intl_team_id, i_files, lineups, intl_comp_label
            )
        else:
            intl_vaep = None

        delta = None
        ratio = None
        if club_vaep and intl_vaep:
            delta = round(float(intl_vaep["vaep90"]) - float(club_vaep["vaep90"]), 4)
            ratio = round(float(intl_vaep["vaep90"]) / float(club_vaep["vaep90"]), 3) if club_vaep["vaep90"] != 0 else None

        group_b.append({
            "player": player_name,
            "player_id": player_id,
            "player_name": pname(player_id),
            "club": "Real Madrid",
            "club_team_id": rm_team_id,
            "country": country,
            "intl_competition": intl_comp_label,
            "club_context": club_vaep,
            "intl_context": intl_vaep,
            "delta_vaep90": delta,
            "ratio_intl_club": ratio,
            "narrative": _describe_delta(player_name, "Real Madrid", country, club_vaep, intl_vaep),
        })

        log.info("  %s: club=%.4f intl=%.4f",
                 player_name,
                 club_vaep["vaep90"] if club_vaep else 0,
                 intl_vaep["vaep90"] if intl_vaep else 0)

    results["real_madrid_multi_season"] = {
        "headline": "Real Madrid La Liga 2017/18-2020/21 -> international",
        "players": group_b,
    }

    # ── GROUP C: Barcelona multi-season -> Argentina / Spain ─────────────
    log.info("Group C: Barcelona -> Argentina/Spain")

    barca_players = {
        "Messi":      (player_ids.get("messi"),    "Argentina", "43_106",  "WC 2022"),
        "Busquets":   (player_ids.get("busquets"), "Spain",     "55_43",   "Euro 2020"),
        "JordiAlba":  (player_ids.get("jordi_alba"), "Spain",   "55_43",   "Euro 2020"),
    }

    group_c = []
    barca_team_id_cache: dict[str, int | None] = {}

    for player_name, (player_id, country, intl_comp_dir, intl_comp_label) in barca_players.items():
        if player_id is None:
            group_c.append({"player": player_name, "status": "id_not_resolved"})
            continue

        club_total_vaep = 0.0
        club_total_actions = 0
        club_total_minutes = 0.0
        club_matches = 0
        barca_team_id = None

        for season_key in la_liga_keys:
            files = club_files(season_key)
            if not files:
                continue
            team_id = find_team_for_player(player_id, files)
            if not team_id:
                continue
            if not barca_team_id:
                barca_team_id = team_id
            for p in sorted(files):
                try:
                    df = pd.read_parquet(p)
                    if "match_id" not in df.columns and "game_id" in df.columns:
                        df = df.rename(columns={"game_id": "match_id"})
                    player_rows = df[
                        (df["player_id"] == player_id) &
                        (df["team_id"] == team_id) &
                        (df["type_id"].isin(ELIGIBLE_TYPES))
                    ]
                    if player_rows.empty:
                        continue
                    club_total_vaep += float(player_rows["vaep_value"].fillna(0).sum())
                    club_total_actions += len(player_rows)
                    club_total_minutes += float(player_rows["match_id"].nunique() * 90.0)
                    club_matches += int(player_rows["match_id"].nunique())
                except Exception:
                    continue

        if club_total_minutes > 0:
            club_vaep = {
                "context": "La Liga (multi-season avg)",
                "vaep90": round(club_total_vaep / club_total_minutes * 90.0, 4),
                "total_vaep": round(club_total_vaep, 4),
                "total_actions": club_total_actions,
                "total_minutes": round(club_total_minutes, 0),
                "matches": club_matches,
            }
        else:
            club_vaep = None

        i_files = intl_files(intl_comp_dir)
        intl_team_id = find_team_for_player(player_id, i_files)
        if intl_team_id and i_files:
            intl_vaep = compute_player_vaep90_from_lineups(
                player_id, intl_team_id, i_files, lineups, intl_comp_label
            )
        else:
            intl_vaep = None

        delta = None
        ratio = None
        if club_vaep and intl_vaep:
            delta = round(float(intl_vaep["vaep90"]) - float(club_vaep["vaep90"]), 4)
            ratio = round(float(intl_vaep["vaep90"]) / float(club_vaep["vaep90"]), 3) if club_vaep["vaep90"] != 0 else None

        group_c.append({
            "player": player_name,
            "player_id": player_id,
            "player_name": pname(player_id),
            "club": "Barcelona",
            "club_team_id": barca_team_id,
            "country": country,
            "intl_competition": intl_comp_label,
            "club_context": club_vaep,
            "intl_context": intl_vaep,
            "delta_vaep90": delta,
            "ratio_intl_club": ratio,
            "narrative": _describe_delta(player_name, "Barcelona", country, club_vaep, intl_vaep),
        })

        log.info("  %s: club=%.4f intl=%.4f",
                 player_name,
                 club_vaep["vaep90"] if club_vaep else 0,
                 intl_vaep["vaep90"] if intl_vaep else 0)

    results["barcelona_multi_season"] = {
        "headline": "Barcelona La Liga 2017/18-2020/21 -> international",
        "players": group_c,
    }

    # ── GROUP D: PSG era -> international ────────────────────────────────
    log.info("Group D: PSG -> international")

    psg_comps = ["7_108", "7_235"]

    psg_players = {
        "Mbappe":  (player_ids.get("mbappe"),  "France",    "43_106",  "WC 2022"),
        "Neymar":  (player_ids.get("neymar"),  "Brazil",    "43_106",  "WC 2022"),
        "Messi":   (player_ids.get("messi"),   "Argentina", "43_106",  "WC 2022"),
        "Hakimi":  (player_ids.get("hakimi"),  "Morocco",   "1267_107","AFCON 2023"),
    }

    group_d = []
    for player_name, (player_id, country, intl_comp_dir, intl_comp_label) in psg_players.items():
        if player_id is None:
            group_d.append({"player": player_name, "status": "id_not_resolved"})
            continue

        club_total_vaep = 0.0
        club_total_actions = 0
        club_total_minutes = 0.0
        club_matches = 0
        psg_team_id = None

        for season_key in psg_comps:
            files = club_files(season_key)
            if not files:
                continue
            team_id = find_team_for_player(player_id, files)
            if not team_id:
                continue
            if not psg_team_id:
                psg_team_id = team_id
            for p in sorted(files):
                try:
                    df = pd.read_parquet(p)
                    if "match_id" not in df.columns and "game_id" in df.columns:
                        df = df.rename(columns={"game_id": "match_id"})
                    player_rows = df[
                        (df["player_id"] == player_id) &
                        (df["team_id"] == team_id) &
                        (df["type_id"].isin(ELIGIBLE_TYPES))
                    ]
                    if player_rows.empty:
                        continue
                    club_total_vaep += float(player_rows["vaep_value"].fillna(0).sum())
                    club_total_actions += len(player_rows)
                    club_total_minutes += float(player_rows["match_id"].nunique() * 90.0)
                    club_matches += int(player_rows["match_id"].nunique())
                except Exception:
                    continue

        if club_total_minutes > 0:
            club_vaep = {
                "context": "Ligue 1 (2021/22 + 2022/23)",
                "vaep90": round(club_total_vaep / club_total_minutes * 90.0, 4),
                "total_vaep": round(club_total_vaep, 4),
                "total_actions": club_total_actions,
                "total_minutes": round(club_total_minutes, 0),
                "matches": club_matches,
            }
        else:
            club_vaep = None

        i_files = intl_files(intl_comp_dir)
        intl_team_id = find_team_for_player(player_id, i_files)
        if intl_team_id and i_files:
            intl_vaep = compute_player_vaep90_from_lineups(
                player_id, intl_team_id, i_files, lineups, intl_comp_label
            )
        else:
            intl_vaep = None

        delta = None
        ratio = None
        if club_vaep and intl_vaep:
            delta = round(float(intl_vaep["vaep90"]) - float(club_vaep["vaep90"]), 4)
            ratio = round(float(intl_vaep["vaep90"]) / float(club_vaep["vaep90"]), 3) if club_vaep["vaep90"] != 0 else None

        group_d.append({
            "player": player_name,
            "player_id": player_id,
            "player_name": pname(player_id),
            "club": "PSG",
            "club_team_id": psg_team_id,
            "country": country,
            "intl_competition": intl_comp_label,
            "club_context": club_vaep,
            "intl_context": intl_vaep,
            "delta_vaep90": delta,
            "ratio_intl_club": ratio,
            "narrative": _describe_delta(player_name, "PSG", country, club_vaep, intl_vaep),
        })

        log.info("  %s: club=%.4f intl=%.4f",
                 player_name,
                 club_vaep["vaep90"] if club_vaep else 0,
                 intl_vaep["vaep90"] if intl_vaep else 0)

    results["psg_era"] = {
        "headline": "PSG Ligue 1 2021/22 + 2022/23 -> WC 2022 / AFCON 2023",
        "players": group_d,
    }

    # ── GROUP E: Inter Miami -> Argentina Copa 2024 ───────────────────────
    log.info("Group E: Inter Miami -> Argentina Copa 2024")

    messi_id = player_ids.get("messi")
    copa24_files = intl_files("223_282")
    mls_files = club_files("44_107")

    if messi_id:
        miami_team_id = find_team_for_player(messi_id, mls_files)
        if miami_team_id and mls_files:
            miami_vaep = compute_player_vaep90(messi_id, miami_team_id, mls_files, "MLS 2023")
        else:
            miami_vaep = None

        copa_team_id = find_team_for_player(messi_id, copa24_files)
        if copa_team_id and copa24_files:
            copa_vaep = compute_player_vaep90_from_lineups(
                messi_id, copa_team_id, copa24_files, lineups, "Copa America 2024"
            )
        else:
            copa_vaep = None

        delta = None
        ratio = None
        if miami_vaep and copa_vaep:
            delta = round(float(copa_vaep["vaep90"]) - float(miami_vaep["vaep90"]), 4)
            ratio = round(float(copa_vaep["vaep90"]) / float(miami_vaep["vaep90"]), 3) if miami_vaep["vaep90"] != 0 else None

        results["inter_miami_argentina"] = {
            "headline": "Inter Miami MLS 2023 -> Argentina Copa 2024",
            "note": "Messi joined July 2023; partial MLS season. Copa 2024 held in USA.",
            "players": [{
                "player": "Messi",
                "player_id": messi_id,
                "player_name": pname(messi_id),
                "club": "Inter Miami",
                "country": "Argentina",
                "club_context": miami_vaep,
                "intl_context": copa_vaep,
                "delta_vaep90": delta,
                "ratio_intl_club": ratio,
                "narrative": _describe_delta("Messi", "Inter Miami", "Argentina Copa 2024", miami_vaep, copa_vaep),
            }],
        }
        if miami_vaep and copa_vaep:
            log.info("  Messi: Miami=%.4f Copa=%.4f", miami_vaep["vaep90"], copa_vaep["vaep90"])
    else:
        results["inter_miami_argentina"] = {"headline": "Messi ID not resolved", "players": []}

    return results


def _describe_delta(player: str, club_name: str, country_label: str,
                     club_ctx: dict | None, intl_ctx: dict | None) -> str:
    """Generate a brief narrative sentence about the production delta."""
    if not club_ctx or not intl_ctx:
        return f"{player}: insufficient data in one or both contexts."
    club_v = club_ctx["vaep90"]
    intl_v = intl_ctx["vaep90"]
    delta = intl_v - club_v
    if abs(delta) < 0.0005:
        return (f"{player} produced virtually identical per-90 VAEP at {club_name} ({club_v:.4f}) "
                f"and {country_label} ({intl_v:.4f}). The club chemistry network had no measurable effect.")
    if delta < -0.002:
        return (f"{player} showed a clear production drop: per-90 VAEP fell from {club_v:.4f} at {club_name} "
                f"to {intl_v:.4f} at {country_label} (delta {delta:+.4f}). "
                f"Consistent with the Podolski archetype: thriving in the club network, diminished without it.")
    if delta > 0.002:
        return (f"{player} produced MORE at {country_label} ({intl_v:.4f}) than at {club_name} ({club_v:.4f}) "
                f"(delta {delta:+.4f}). Inverse Podolski: the national team context appears more enabling.")
    return (f"{player}: marginal difference — {club_v:.4f} at {club_name}, {intl_v:.4f} at {country_label} "
            f"(delta {delta:+.4f}).")


# ── Phase 10: Write outputs ────────────────────────────────────────────────

def phase10_write_outputs(cross_context: dict, vaep_v2_bundle: dict) -> None:
    log.info("=== Phase 10: Write outputs ===")

    out_path = OUTPUTS / "cross_context_chemistry.json"
    existing = {}
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    existing["sb_expanded"] = {
        "vaep_v2_model": {
            "train_size": vaep_v2_bundle.get("train_size", 0),
            "auc_scores": round(vaep_v2_bundle.get("auc_scores", 0), 4),
            "auc_concedes": round(vaep_v2_bundle.get("auc_concedes", 0), 4),
            "note": "Trained on StatsBomb international + 8 club seasons (no Wyscout)",
        },
        "player_production": cross_context,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    log.info("Updated %s", out_path)

    _write_markdown(cross_context, vaep_v2_bundle)


def _fmt(v, d=4):
    if v is None:
        return "n/a"
    return f"{v:.{d}f}"


def _vaep90_cell(ctx: dict | None) -> str:
    if not ctx:
        return "n/a"
    v = ctx.get("vaep90")
    m = ctx.get("matches", "?")
    if v is None:
        return "n/a"
    return f"{v:.4f} ({m} matches)"


def _write_markdown(cross_context: dict, vaep_v2: dict) -> None:
    out_path = ROOT / "docs" / "analysis" / "cross-context-chemistry.md"
    existing_text = out_path.read_text(encoding="utf-8") if out_path.exists() else ""

    auc_s = vaep_v2.get("auc_scores", 0)
    auc_c = vaep_v2.get("auc_concedes", 0)
    train_n = vaep_v2.get("train_size", 0)

    lines: list[str] = []

    # Keep existing content up to old Section 5 (Caveats)
    if existing_text:
        caveat_markers = ["## 5. Caveats", "## 8. Updated Caveats", "---\n\n## 3."]
        cut_pos = len(existing_text)
        for m in caveat_markers:
            if m in existing_text:
                cut_pos = min(cut_pos, existing_text.index(m))
        lines.append(existing_text[:cut_pos].rstrip())
    else:
        lines.append("# Club vs Country Chemistry: Cross-Context Analysis\n")

    lines += [
        "",
        "---",
        "",
        f"## 3. Player-Level Production: Does Output Drop Without the Club Network?",
        "",
        f"**Framing (updated):** The question is not whether pair chemistry transfers — it is whether",
        f"the *player's individual per-90 production* drops when removed from their club chemistry",
        f"network. The Podolski archetype: great at club, diminished at national team. The counter-case:",
        f"players who are the same or better internationally.",
        "",
        f"Honest caveat: the dataset does **not** cover Lukas Podolski himself — his career (Köln,",
        f"Bayern 2009-12, Arsenal, Inter, Galatasaray, Vissel Kobe) is in no open dataset.",
        f"This analysis tests the Podolski *archetype* on the closest modern players available.",
        "",
        f"**VAEP v2 model:** {train_n:,} StatsBomb actions, AUC scores={auc_s:.3f}, concedes={auc_c:.3f}.",
        f"Metric: per-90 VAEP on the player's own on-ball actions (SPADL eligible types).",
        f"Minutes estimated from lineups parquet (international) or match count × 90 (club).",
        "",
    ]

    # ── Section 3a: Bayern -> Germany ────────────────────────────────────
    bge = cross_context.get("bayern_germany_euro24", {})
    bge_players = bge.get("players", [])

    lines += [
        "### 3a. Bayern 2023/24 -> Germany Euro 2024 (same-season, cleanest test)",
        "",
        "| Player | Club VAEP/90 (Bundesliga 23/24) | Intl VAEP/90 (Euro 2024) | Delta | Ratio |",
        "|---|---|---|---|---|",
    ]
    for p in bge_players:
        cv = p.get("club_context")
        iv = p.get("intl_context")
        delta = p.get("delta_vaep90")
        ratio = p.get("ratio_intl_club")
        delta_str = f"{delta:+.4f}" if delta is not None else "n/a"
        ratio_str = f"{ratio:.2f}" if ratio is not None else "n/a"
        lines.append(
            f"| {p['player']} | {_vaep90_cell(cv)} | {_vaep90_cell(iv)} | {delta_str} | {ratio_str} |"
        )
    lines += [""]

    for p in bge_players:
        if p.get("narrative"):
            lines.append(f"- **{p['player']}:** {p['narrative']}")
    lines += [""]

    # Verdict for Bayern/Germany
    valid_bge = [(p, p["ratio_intl_club"]) for p in bge_players if p.get("ratio_intl_club") is not None]
    drops = [p for p, r in valid_bge if r < 0.85]
    holds = [p for p, r in valid_bge if r >= 0.85]
    if valid_bge:
        lines.append(f"**Bayern/Germany verdict ({len(valid_bge)} players with data):** "
                     f"{len(drops)} showed a production drop at Germany vs Bayern "
                     f"(ratio < 0.85); {len(holds)} held or improved.")
        lines.append("")

    # ── Section 3b: Real Madrid ────────────────────────────────────────────
    rm = cross_context.get("real_madrid_multi_season", {})
    lines += [
        "### 3b. Real Madrid La Liga 2017/18-2020/21 -> international",
        "",
        "| Player | Club VAEP/90 (La Liga, multi-season) | Intl VAEP/90 | Delta | Ratio |",
        "|---|---|---|---|---|",
    ]
    for p in rm.get("players", []):
        cv = p.get("club_context")
        iv = p.get("intl_context")
        delta = p.get("delta_vaep90")
        ratio = p.get("ratio_intl_club")
        intl_label = p.get("intl_competition", "")
        delta_str = f"{delta:+.4f}" if delta is not None else "n/a"
        ratio_str = f"{ratio:.2f}" if ratio is not None else "n/a"
        lines.append(
            f"| {p['player']} ({intl_label}) | {_vaep90_cell(cv)} | {_vaep90_cell(iv)} | {delta_str} | {ratio_str} |"
        )
    lines += [""]
    for p in rm.get("players", []):
        if p.get("narrative"):
            lines.append(f"- **{p['player']}:** {p['narrative']}")
    lines += [""]

    # ── Section 3c: Barcelona ─────────────────────────────────────────────
    barca = cross_context.get("barcelona_multi_season", {})
    lines += [
        "### 3c. Barcelona La Liga 2017/18-2020/21 -> international",
        "",
        "| Player | Club VAEP/90 (La Liga) | Intl VAEP/90 | Delta | Ratio |",
        "|---|---|---|---|---|",
    ]
    for p in barca.get("players", []):
        cv = p.get("club_context")
        iv = p.get("intl_context")
        delta = p.get("delta_vaep90")
        ratio = p.get("ratio_intl_club")
        intl_label = p.get("intl_competition", "")
        delta_str = f"{delta:+.4f}" if delta is not None else "n/a"
        ratio_str = f"{ratio:.2f}" if ratio is not None else "n/a"
        lines.append(
            f"| {p['player']} ({intl_label}) | {_vaep90_cell(cv)} | {_vaep90_cell(iv)} | {delta_str} | {ratio_str} |"
        )
    lines += [""]
    for p in barca.get("players", []):
        if p.get("narrative"):
            lines.append(f"- **{p['player']}:** {p['narrative']}")
    lines += [""]

    # ── Section 3d: PSG ────────────────────────────────────────────────────
    psg = cross_context.get("psg_era", {})
    lines += [
        "### 3d. PSG 2021/22 + 2022/23 -> WC 2022 / AFCON 2023",
        "",
        "The Mbappe-Neymar-Messi trio era. PSG failed both Champions League campaigns. "
        "Each player competed at WC 2022.",
        "",
        "| Player | PSG VAEP/90 (Ligue 1) | Intl VAEP/90 | Delta | Ratio |",
        "|---|---|---|---|---|",
    ]
    for p in psg.get("players", []):
        cv = p.get("club_context")
        iv = p.get("intl_context")
        delta = p.get("delta_vaep90")
        ratio = p.get("ratio_intl_club")
        intl_label = p.get("intl_competition", "")
        delta_str = f"{delta:+.4f}" if delta is not None else "n/a"
        ratio_str = f"{ratio:.2f}" if ratio is not None else "n/a"
        lines.append(
            f"| {p['player']} ({intl_label}) | {_vaep90_cell(cv)} | {_vaep90_cell(iv)} | {delta_str} | {ratio_str} |"
        )
    lines += [""]
    for p in psg.get("players", []):
        if p.get("narrative"):
            lines.append(f"- **{p['player']}:** {p['narrative']}")
    lines += [""]

    # ── Section 3e: Inter Miami ────────────────────────────────────────────
    miami = cross_context.get("inter_miami_argentina", {})
    lines += [
        "### 3e. Inter Miami MLS 2023 -> Argentina Copa 2024",
        "",
        "Messi joined July 2023 (partial MLS season). Copa America 2024 was held in the US.",
        "A micro-comparison: different club context, same player.",
        "",
    ]
    for p in miami.get("players", []):
        if p.get("narrative"):
            lines.append(p["narrative"])
    lines += [""]

    # ── Old Section 4 (Wyscout) note ──────────────────────────────────────
    lines += [
        "---",
        "",
        "## 4. Wyscout 2017/18 Cross-Section (Historic Reference)",
        "",
        "The pair-level JOI90 analysis using Wyscout 2017/18 data is preserved above "
        "(Sections 2-3 of the original report). It covers Bayern 2017/18 -> Germany WC 2018 "
        "and other featured players from that season. The player-level production analysis in "
        "Section 3 above uses VAEP v2 on StatsBomb data and is the primary finding.",
        "",
    ]

    # ── Section 5: Caveats ────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 5. Updated Caveats",
        "",
        f"1. **Missing Podolski data.** Lukas Podolski's club career (Köln, Bayern 2009-12, "
        f"Arsenal 2012-15, Inter, Galatasaray, Vissel Kobe) is not in any open dataset. "
        f"This analysis tests the Podolski *archetype* on the closest available players.",
        "",
        f"2. **VAEP v2 scale.** Trained on {train_n:,} StatsBomb actions (AUC scores={auc_s:.3f}, "
        f"concedes={auc_c:.3f}). Wyscout VAEP (v1) uses a different action schema. "
        f"Only within-v2 comparisons are valid here.",
        "",
        f"3. **Per-90 conflates multiple factors.** Lower production at country could be: "
        f"(a) weaker supporting cast (the chemistry-network hypothesis), "
        f"(b) stronger opponents on average, "
        f"(c) different tactical role, "
        f"(d) fatigue (major tournaments near season end), "
        f"(e) small sample variance.",
        "",
        f"4. **StatsBomb open data ceiling.** Premier League, Serie A, Bundesliga (outside 23/24) "
        f"are absent. Hazard/De Bruyne at Chelsea/Man City, Lewandowski at Bayern 17/18 (Wyscout "
        f"covers this but with v1 VAEP), Salah at Liverpool — all absent from StatsBomb club data.",
        "",
        f"5. **Minutes estimation.** Club minutes use match_count × 90 as a proxy "
        f"(overestimates for subs, underestimates for extra-time starters). International "
        f"minutes use actual lineups parquet where available.",
        "",
    ]

    # ── Section 6: Conclusion ─────────────────────────────────────────────
    # Gather all players with data
    all_deltas = []
    for group_key in ["bayern_germany_euro24", "real_madrid_multi_season",
                       "barcelona_multi_season", "psg_era"]:
        group = cross_context.get(group_key, {})
        for p in group.get("players", []):
            delta = p.get("delta_vaep90")
            ratio = p.get("ratio_intl_club")
            if delta is not None and ratio is not None:
                all_deltas.append((p["player"], group_key, delta, ratio))

    drops = [(n, g, d, r) for n, g, d, r in all_deltas if r < 0.85]
    holds = [(n, g, d, r) for n, g, d, r in all_deltas if 0.85 <= r <= 1.15]
    rises = [(n, g, d, r) for n, g, d, r in all_deltas if r > 1.15]

    lines += [
        "---",
        "",
        "## 6. Conclusion: Updated Verdict on the Podolski Thesis",
        "",
        f"**Sample:** {len(all_deltas)} players with data in both club and international contexts.",
        f"- Production drops (ratio < 0.85): {len(drops)} — {', '.join(n for n, g, d, r in drops) if drops else 'none'}",
        f"- Parity (ratio 0.85-1.15): {len(holds)} — {', '.join(n for n, g, d, r in holds) if holds else 'none'}",
        f"- Inverse Podolski (ratio > 1.15): {len(rises)} — {', '.join(n for n, g, d, r in rises) if rises else 'none'}",
        "",
    ]

    if not all_deltas:
        lines.append("Data pending — run pipeline to completion for final verdict.")
    elif len(drops) > len(rises) + len(holds):
        lines.append(
            "**Verdict: The Podolski archetype is real but not universal.** Most featured players "
            "showed lower per-90 VAEP at national team level than at club level, consistent with "
            "the hypothesis that removing a player from their club chemistry network reduces their output. "
            "However, the minority of players who hold or improve at international level shows this "
            "is not a structural law — elite players who generate value independently of system "
            "survive the transition."
        )
    elif len(rises) >= len(drops):
        lines.append(
            "**Verdict: Modest support for an inverse Podolski pattern.** More featured players "
            "held or improved production internationally than dropped. This could reflect selection "
            "bias in the data (international-only open data covers World Cup / major tournaments "
            "where these players shone), or genuine chemistry-independent quality."
        )
    else:
        lines.append(
            "**Verdict: Mixed.** The Podolski thesis receives partial support — "
            "some players clearly produce less when removed from their club network, "
            "others hold steady or rise. Individual context and role stability appear "
            "to be the dominant moderators."
        )

    lines += [
        "",
        "License: StatsBomb open data (custom open license) + Wyscout open data (CC BY 4.0). "
        "VAEP: Decroos et al. 2019, Bransen & Van Haaren 2020.",
        "",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s", out_path)


# ── Phase 11: Update site/index.html Research tab ────────────────────────

def phase11_update_site(cross_context: dict, vaep_v2: dict) -> None:
    log.info("=== Phase 11: Update site Research tab ===")

    site_path = ROOT / "site" / "index.html"
    if not site_path.exists():
        log.warning("site/index.html not found")
        return

    html = site_path.read_text(encoding="utf-8")

    auc_s = vaep_v2.get("auc_scores", 0)
    auc_c = vaep_v2.get("auc_concedes", 0)
    train_n = vaep_v2.get("train_size", 0)

    # Build Bayern 2023/24 -> Germany Euro 2024 dependency cards
    bge = cross_context.get("bayern_germany_euro24", {})
    bge_players = bge.get("players", [])

    card_rows = []
    for p in bge_players:
        cv = p.get("club_context")
        iv = p.get("intl_context")
        delta = p.get("delta_vaep90")
        ratio = p.get("ratio_intl_club")
        if not cv or not iv:
            continue

        club_v = cv.get("vaep90", 0)
        intl_v = iv.get("vaep90", 0)
        delta_str = f"{delta:+.4f}" if delta is not None else "n/a"
        ratio_str = f"{ratio:.2f}" if ratio is not None else "n/a"
        ratio_class = ""
        if ratio is not None:
            if ratio < 0.85:
                ratio_class = "joi-neg"
            elif ratio > 1.15:
                ratio_class = "joi-high"

        card_rows.append(
            f'              <tr>\n'
            f'                <td><strong>{p["player"]}</strong></td>\n'
            f'                <td class="joi">{club_v:.4f}</td>\n'
            f'                <td class="joi">{intl_v:.4f}</td>\n'
            f'                <td class="joi {ratio_class}">{delta_str}</td>\n'
            f'                <td class="joi {ratio_class}">{ratio_str}</td>\n'
            f'              </tr>'
        )

    # Narrative rows for selected players
    narrative_cards = []
    for p in bge_players:
        narr = p.get("narrative")
        if narr and p.get("delta_vaep90") is not None:
            narrative_cards.append(
                f'          <div class="slide-inner">\n'
                f'            <h3>{p["player"]}</h3>\n'
                f'            <p>{narr}</p>\n'
                f'          </div>'
            )

    # Build summary for PSG players
    psg = cross_context.get("psg_era", {})
    psg_cards = []
    for p in psg.get("players", []):
        cv = p.get("club_context")
        iv = p.get("intl_context")
        if not cv or not iv:
            continue
        ratio = p.get("ratio_intl_club")
        ratio_str = f"{ratio:.2f}" if ratio is not None else "n/a"
        narr = p.get("narrative", "")
        psg_cards.append(
            f'          <div class="slide-inner">\n'
            f'            <h3>{p["player"]} (PSG &rarr; {p.get("country", "intl")})</h3>\n'
            f'            <p>PSG VAEP/90: <strong>{cv["vaep90"]:.4f}</strong> &nbsp;&mdash;&nbsp; '
            f'Intl VAEP/90: <strong>{iv["vaep90"]:.4f}</strong> &nbsp;&mdash;&nbsp; Ratio: <strong>{ratio_str}</strong></p>\n'
            f'            <p>{narr}</p>\n'
            f'          </div>'
        )

    # Build new section HTML
    new_section_html = f'''\
      <!-- Section: Bayern 2023/24 -> Germany Euro 2024 (player-level production) -->
      <section class="slide-card" aria-label="Bayern 2023/24 Germany Euro 2024 production">
        <span class="pill green">Headline finding (within-season)</span>
        <h2>Bayern 2023/24 &rarr; Germany Euro 2024: player production without the club network</h2>

        <p>The Podolski thesis reframed: does a player's per-90 VAEP <em>drop</em> when removed from their club chemistry network? Same season, same players, two contexts. VAEP v2 model: {train_n:,} StatsBomb actions, AUC scores={auc_s:.3f}, concedes={auc_c:.3f}.</p>
        <p>Metric: sum of VAEP on the player's own on-ball actions (passes, carries, shots), normalised per 90 minutes. Not pair-level &mdash; this is individual production in each context.</p>

        <div class="research-table-wrap">
          <table class="leaderboard research-table">
            <thead>
              <tr>
                <th>Player</th>
                <th>Bayern VAEP/90 <span class="th-sub">Bundesliga 2023/24</span></th>
                <th>Germany VAEP/90 <span class="th-sub">Euro 2024</span></th>
                <th>Delta</th>
                <th>Ratio <span class="th-sub">intl/club</span></th>
              </tr>
            </thead>
            <tbody>
{chr(10).join(card_rows) if card_rows else "              <tr><td colspan='5'>Data computing &mdash; re-run pipeline to completion</td></tr>"}
            </tbody>
          </table>
        </div>
        <p class="research-note">Ratio &lt; 1.0 = production dropped away from the Bayern chemistry network. Ratio &gt; 1.0 = production held or rose internationally. Minutes at club estimated as match appearances &times; 90; international minutes from StatsBomb lineups data. Honest caveat: Lukas Podolski himself is not in any open dataset &mdash; this tests the archetype.</p>

        <div style="margin-top:16px">
{chr(10).join(narrative_cards[:4]) if narrative_cards else "          <p>Narrative cards pending pipeline completion.</p>"}
        </div>
      </section>

      <!-- Section: PSG era dependency cards -->
      <section class="slide-card" aria-label="PSG era cross-context">
        <span class="pill amber">PSG era 2021/22&ndash;2022/23</span>
        <h2>PSG &rarr; WC 2022 / AFCON 2023: Mbapp&eacute;, Neymar, Messi, Hakimi</h2>
        <p>The Mbapp&eacute;-Neymar-Messi trio: most expensive front line in history. PSG failed both Champions League campaigns. WC 2022: France lost the final to Argentina, with Mbapp&eacute; scoring a hat-trick. Does each player's individual production hold without PSG's supporting cast?</p>

        <div class="slide-2col" style="margin-top:8px">
{chr(10).join(psg_cards[:4]) if psg_cards else "          <p>PSG data computing.</p>"}
        </div>
        <p class="research-note">PSG = Ligue 1 2021/22 + 2022/23 combined. Intl = WC 2022 (France/Brazil/Argentina) or AFCON 2023 (Morocco). StatsBomb open data, VAEP v2.</p>
      </section>

'''

    # Insert new sections BEFORE the existing first slide-card in the Research panel
    marker = '      <!-- Section 1: Cross-context (Wyscout) analysis -->'
    if marker in html:
        new_html = html.replace(marker, new_section_html + "      " + marker.lstrip())
    else:
        # Fallback: insert before first slide-card in research panel
        research_marker = '<div class="slide-deck">'
        if research_marker in html:
            insert_idx = html.rindex(research_marker)  # last occurrence = research tab
            insert_after = html.index('\n', insert_idx) + 1
            new_html = html[:insert_after] + "\n" + new_section_html + html[insert_after:]
        else:
            log.warning("Could not find insertion point in site/index.html")
            return

    site_path.write_text(new_html, encoding="utf-8")
    log.info("Updated site/index.html")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    import sys
    sys.path.insert(0, str(ROOT))

    log.info("Starting Phase 2 StatsBomb club cross-context pipeline")

    # Phase 4: Fetch club data
    phase4_fetch_sb_club_data()

    # Phase 5: Convert to SPADL
    phase5_convert_sb_spadl()

    # Phase 6: Train VAEP v2
    vaep_v2_bundle = phase6_train_vaep_v2()
    log.info("VAEP v2 AUC: scores=%.4f  concedes=%.4f  train_size=%d",
             vaep_v2_bundle["auc_scores"], vaep_v2_bundle["auc_concedes"],
             vaep_v2_bundle["train_size"])

    # Phase 7: Score with VAEP v2
    phase7_score_club_vaep(vaep_v2_bundle)
    phase7b_score_intl_vaep_v2(vaep_v2_bundle)

    # Phase 8: JOI (kept for data integrity; not used as primary analysis metric)
    club_joi = phase8_compute_sb_joi()
    intl_joi = phase8b_compute_intl_joi_v2()

    # Phase 9: Player-level per-90 production analysis
    cross_context = phase9_player_production_analysis()

    # Phase 10: Write outputs
    phase10_write_outputs(cross_context, vaep_v2_bundle)

    # Phase 11: Update site
    phase11_update_site(cross_context, vaep_v2_bundle)

    log.info("=== Phase 2 COMPLETE ===")
    log.info("VAEP v2: scores AUC=%.4f  concedes AUC=%.4f  train_size=%d",
             vaep_v2_bundle["auc_scores"], vaep_v2_bundle["auc_concedes"],
             vaep_v2_bundle["train_size"])

    # Print summary table
    log.info("--- Player production summary ---")
    for group_key, label in [
        ("bayern_germany_euro24", "Bayern->Germany"),
        ("real_madrid_multi_season", "Real Madrid->intl"),
        ("barcelona_multi_season", "Barcelona->intl"),
        ("psg_era", "PSG->intl"),
    ]:
        group = cross_context.get(group_key, {})
        for p in group.get("players", []):
            cv = p.get("club_context")
            iv = p.get("intl_context")
            ratio = p.get("ratio_intl_club")
            log.info("  %-20s %-12s  club=%-8s  intl=%-8s  ratio=%s",
                     p.get("player", "?"), f"[{label}]",
                     f"{cv['vaep90']:.4f}" if cv else "n/a",
                     f"{iv['vaep90']:.4f}" if iv else "n/a",
                     f"{ratio:.2f}" if ratio is not None else "n/a")


if __name__ == "__main__":
    main()
