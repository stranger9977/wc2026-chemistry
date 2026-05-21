"""Build player-level chemistry dataset v3.

Anchored on Bransen & Van Haaren 2020 "Player Chemistry":
  - Section 3.1 JOI (Joint Offensive Impact): sum VAEP for consecutive
    actions in the same possession where two different teammates act,
    normalized per 90 shared minutes.
  - Plus passing-graph centrality measures and a Beal-style embeddedness
    score for "interactional alignment" with the rest of the squad.

Plus a "career familiarity" cross-context table per the American Soccer
Analysis Jan 2026 piece.

Inputs (all already present in the repo):
  - data/vaep_scored_v2/<ctx>/*.parquet     - VAEP v2 scored SPADL actions per match
  - data/spadl/<ctx>/*.parquet              - international SPADL (for pass graph)
  - data/sb_club_spadl/<ctx>/*.parquet      - club SPADL (for pass graph)
  - data/raw/<ctx>/events/*.parquet         - raw events with Starting XI / subs
  - data/raw/<ctx>/matches.parquet          - team_id <-> team_name mapping
  - outputs/sb_lineups_all.parquet          - per-match minutes_played per player

Outputs:
  - outputs/player_chemistry_v3.parquet
  - outputs/career_familiarity.parquet
  - outputs/player_chemistry_v3.NOTES.md

Run:  .venv/bin/python -m scripts.build_player_chemistry_v3

Methodological notes
--------------------
* OFFENSIVE vs DEFENSIVE split: VAEP v2 scored data here has a single
  ``vaep_value`` column (positive credit for actions that raise scoring
  probability or lower conceding probability). We split per-action by
  SPADL action type: types in DEFENSIVE_TYPES (tackle, interception,
  clearance, keeper_*) count toward defensive_value; everything else
  counts toward offensive_value. This is a standard convention but is a
  coarser split than what socceraction's ``formula.value`` computes
  natively (which we don't have in vaep_v2 cache). Documented here so
  users know.

* JOI convention: matches existing repo convention in
  scripts/vaep_cross_context_pipeline.py — sum the VAEP value of the
  *second* action in each consecutive (different teammate, same team,
  same match) pair. JOI90 = JOI * 90 / shared_minutes.

* Shared minutes: computed from raw event data Starting XI + Substitution
  events. Players who start are assumed on at 0; players subbed off get
  off-minute set to the sub minute; players subbed on get on-minute set
  to the sub minute and off=90 (or 120 if extra time). Red cards are NOT
  modeled — they're rare and we accept that minor inaccuracy.

* Minutes for player-context aggregates: we trust outputs/sb_lineups_all
  ``minutes_played`` which is StatsBomb's official figure.

* Top-5 partner requirement: pair must have >= 180 shared minutes in the
  context, per spec. JOI90 only valid above this threshold.

* Per-context minimum: skip player-contexts with < 180 minutes played.

* Centrality on pass graph: uses SPADL pass action type (type_id=0) with
  result_id=1 (success). Edge weights = number of completed passes
  between two players (we infer the receiver from the *next* action by a
  same-team teammate within the same match/period). Eigenvector
  centrality computed via power iteration on the symmetrized
  weighted-adjacency matrix.

* Embeddedness ("interactional alignment"): for each of the player's
  top-5 partners, sum the partner's JOI90 with the OTHER teammates
  (excluding the player itself) in the same context. We then sum those
  values across the top-5 partners.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_VAEP_V2 = DATA / "vaep_scored_v2"
DATA_SPADL_INTL = DATA / "spadl"
DATA_SPADL_CLUB = DATA / "sb_club_spadl"
OUTPUTS = ROOT / "outputs"
LINEUPS_PATH = OUTPUTS / "sb_lineups_all.parquet"

OUT_CHEM = OUTPUTS / "player_chemistry_v3.parquet"
OUT_FAMILIARITY = OUTPUTS / "career_familiarity.parquet"
OUT_NOTES = OUTPUTS / "player_chemistry_v3.NOTES.md"

# Contexts and human-readable labels (verified against StatsBomb open
# data competitions.json on 2026-05-21):
#   43_3      FIFA World Cup 2018
#   55_43     UEFA Euro 2020
#   43_106    FIFA World Cup 2022
#   55_282    UEFA Euro 2024
#   223_282   Copa America 2024
#   1267_107  African Cup of Nations 2023
INTL_CONTEXTS = [
    {"key": "43_3",     "label": "WC 2018",     "type": "international"},
    {"key": "55_43",    "label": "Euro 2020",   "type": "international"},
    {"key": "43_106",   "label": "WC 2022",     "type": "international"},
    {"key": "55_282",   "label": "Euro 2024",   "type": "international"},
    {"key": "223_282",  "label": "Copa 2024",   "type": "international"},
    {"key": "1267_107", "label": "AFCON 2023",  "type": "international"},
]
CLUB_CONTEXTS = [
    {"key": "11_1",   "label": "Barcelona La Liga 17/18",   "type": "club"},
    {"key": "11_4",   "label": "Barcelona La Liga 18/19",   "type": "club"},
    {"key": "11_42",  "label": "Barcelona La Liga 19/20",   "type": "club"},
    {"key": "11_90",  "label": "Barcelona La Liga 20/21",   "type": "club"},
    {"key": "7_108",  "label": "PSG Ligue 1 21/22",         "type": "club"},
    {"key": "7_235",  "label": "PSG Ligue 1 22/23",         "type": "club"},
    {"key": "9_281",  "label": "Leverkusen Bundesliga 23/24", "type": "club"},
    {"key": "44_107", "label": "Inter Miami MLS 23",        "type": "club"},
]
ALL_CONTEXTS = INTL_CONTEXTS + CLUB_CONTEXTS

# SPADL action types treated as defensive for off/def vaep split.
DEFENSIVE_TYPES: frozenset[int] = frozenset({9, 10, 14, 15, 16, 17, 18})
# Pass-ish action types used for JOI consecutive-action sequences.
# Matches ELIGIBLE_TYPES from existing pipeline (passes, crosses, set
# pieces, dribbles, take-ons, shots, throw-ins).
ELIGIBLE_JOI_TYPES: frozenset[int] = frozenset({0, 1, 3, 4, 5, 6, 7, 11, 12, 13, 21})
# Pass-only action types for the passing graph centrality computation.
PASS_TYPES: frozenset[int] = frozenset({0, 1, 3, 4, 5, 6})

MIN_CONTEXT_MINUTES = 180.0
MIN_PAIR_MINUTES = 180.0  # threshold for JOI90 validity per spec
TOP_K = 5


# ── Helpers ────────────────────────────────────────────────────────────────


def load_vaep_for_context(ctx_key: str) -> pd.DataFrame:
    """Load all VAEP v2 scored actions for a context, ensure match_id col."""
    d = DATA_VAEP_V2 / ctx_key
    parts = []
    for p in sorted(d.glob("*.parquet")):
        df = pd.read_parquet(p)
        if "match_id" not in df.columns:
            df["match_id"] = df["game_id"]
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def load_spadl_for_context(ctx_key: str, ctx_type: str) -> pd.DataFrame:
    """Load all SPADL actions for a context (used for pass graph)."""
    if ctx_type == "international":
        d = DATA_SPADL_INTL / ctx_key
    else:
        d = DATA_SPADL_CLUB / ctx_key
    parts = []
    for p in sorted(d.glob("*.parquet")):
        df = pd.read_parquet(p)
        if "match_id" not in df.columns:
            df["match_id"] = df["game_id"]
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def load_team_name_map(ctx_key: str) -> dict[int, str]:
    """Build team_id -> team_name from the context's matches.parquet."""
    p = DATA_RAW / ctx_key / "matches.parquet"
    if not p.exists():
        return {}
    m = pd.read_parquet(p)
    mapping: dict[int, str] = {}
    for _, row in m.iterrows():
        # Some files have nested dicts (club competitions), some flat (intl)
        ht = row.get("home_team")
        ht_id = row.get("home_team_id")
        at = row.get("away_team")
        at_id = row.get("away_team_id")
        if isinstance(ht, dict):
            ht_id = ht.get("home_team_id") or ht.get("id")
            ht = ht.get("home_team_name") or ht.get("name")
        if isinstance(at, dict):
            at_id = at.get("away_team_id") or at.get("id")
            at = at.get("away_team_name") or at.get("name")
        if ht_id is not None and ht:
            mapping[int(ht_id)] = str(ht)
        if at_id is not None and at:
            mapping[int(at_id)] = str(at)
    return mapping


def compute_player_match_onoff(ctx_key: str, match_ids: list[int]) -> pd.DataFrame:
    """Compute on/off minute per player per match from raw events."""
    events_dir = DATA_RAW / ctx_key / "events"
    rows = []
    for mid in match_ids:
        ep = events_dir / f"{mid}.parquet"
        if not ep.exists():
            continue
        try:
            ev = pd.read_parquet(ep)
        except Exception as e:
            log.warning("  Failed to read events for %s/%s: %s", ctx_key, mid, e)
            continue

        def get_type_name(t):
            if isinstance(t, dict):
                return t.get("name", "")
            return str(t) if t else ""

        ev["_type_name"] = ev["type"].apply(get_type_name)

        # Determine max minute observed (to handle extra-time matches)
        max_minute = float(ev["minute"].max()) if "minute" in ev.columns else 90.0
        default_off = min(max(max_minute, 90.0), 120.0)

        # team_id resolution: international events have ``team`` as a
        # string + a separate ``team_id`` column; club events have
        # ``team`` as a dict with an ``id`` field.
        def get_tid(row):
            td = row.get("team")
            if isinstance(td, dict):
                v = td.get("id") or td.get("team_id")
                if v is not None:
                    return int(v)
            tid_col = row.get("team_id")
            if tid_col is not None and not (isinstance(tid_col, float) and np.isnan(tid_col)):
                try:
                    return int(tid_col)
                except (TypeError, ValueError):
                    return None
            return None

        starting_xi = ev[ev["_type_name"] == "Starting XI"]
        team_players: dict[int, dict[int, list[float]]] = {}
        for _, se in starting_xi.iterrows():
            tid = get_tid(se)
            if tid is None:
                continue
            tactics = se.get("tactics")
            if not isinstance(tactics, dict):
                continue
            lineup = tactics.get("lineup", [])
            team_players.setdefault(tid, {})
            for p in lineup:
                if isinstance(p, dict) and "player" in p:
                    pid = p["player"].get("id")
                    if pid:
                        team_players[tid][int(pid)] = [0.0, default_off]

        subs = ev[ev["_type_name"] == "Substitution"]
        for _, sub in subs.iterrows():
            minute = float(sub.get("minute", default_off))
            tid = get_tid(sub)
            if tid is None:
                continue
            player_data = sub.get("player")
            player_off_id = (
                int(player_data.get("id"))
                if isinstance(player_data, dict) and player_data.get("id")
                else None
            )
            sub_info = sub.get("substitution")
            player_on_id = None
            if isinstance(sub_info, dict) and "replacement" in sub_info:
                repl = sub_info["replacement"]
                if isinstance(repl, dict) and repl.get("id"):
                    player_on_id = int(repl["id"])

            if player_off_id and tid in team_players and player_off_id in team_players[tid]:
                team_players[tid][player_off_id][1] = minute
            if player_on_id:
                team_players[tid].setdefault(player_on_id, [minute, default_off])
                # If player_on already appears (unusual), keep earliest on time.
                if team_players[tid][player_on_id][0] > minute:
                    team_players[tid][player_on_id][0] = minute

        for tid, players in team_players.items():
            for pid, (on, off) in players.items():
                rows.append({
                    "match_id": int(mid),
                    "team_id": int(tid),
                    "player_id": int(pid),
                    "minute_on": float(on),
                    "minute_off": float(min(off, 120.0)),
                })

    if not rows:
        return pd.DataFrame(columns=["match_id", "team_id", "player_id", "minute_on", "minute_off"])
    return pd.DataFrame(rows)


def compute_pair_shared_minutes(player_onoff: pd.DataFrame) -> pd.DataFrame:
    """Cross-join within match/team to get pair shared minutes."""
    rows = []
    for (match_id, team_id), grp in player_onoff.groupby(["match_id", "team_id"]):
        players = grp[["player_id", "minute_on", "minute_off"]].values.tolist()
        n = len(players)
        for i in range(n):
            pid_a, on_a, off_a = int(players[i][0]), float(players[i][1]), float(players[i][2])
            for j in range(i + 1, n):
                pid_b, on_b, off_b = int(players[j][0]), float(players[j][1]), float(players[j][2])
                shared = max(0.0, min(off_a, off_b) - max(on_a, on_b))
                if shared > 0:
                    rows.append({
                        "match_id": int(match_id),
                        "team_id": int(team_id),
                        "player_a": min(pid_a, pid_b),
                        "player_b": max(pid_a, pid_b),
                        "shared_minutes": shared,
                    })
    if not rows:
        return pd.DataFrame(columns=["match_id", "team_id", "player_a", "player_b", "shared_minutes"])
    return pd.DataFrame(rows)


def compute_joi_per_match(actions: pd.DataFrame) -> pd.DataFrame:
    """JOI per (match, team, player_a, player_b) — sum VAEP of the second
    action in each consecutive (same-team, same-match, two-teammate) pair.

    Matches the existing convention in
    scripts/vaep_cross_context_pipeline.py.
    """
    df = actions[actions["type_id"].isin(ELIGIBLE_JOI_TYPES)].copy()
    if df.empty:
        return pd.DataFrame(columns=["match_id", "team_id", "player_a", "player_b", "joi_vaep"])
    df = df.sort_values(["match_id", "period_id", "time_seconds"]).reset_index(drop=True)

    nxt = df.shift(-1)
    consecutive = (
        (df["match_id"] == nxt["match_id"])
        & (df["team_id"] == nxt["team_id"])
        & (df["player_id"] != nxt["player_id"])
        & (df["player_id"].notna())
        & (nxt["player_id"].notna())
    )

    pairs = pd.DataFrame({
        "match_id": df["match_id"],
        "team_id": df["team_id"],
        "player_p": df["player_id"],
        "player_q": nxt["player_id"],
        "vaep_q": nxt["vaep_value"],
    })[consecutive].reset_index(drop=True)

    if pairs.empty:
        return pd.DataFrame(columns=["match_id", "team_id", "player_a", "player_b", "joi_vaep"])

    pairs["player_a"] = pairs[["player_p", "player_q"]].min(axis=1).astype("int64")
    pairs["player_b"] = pairs[["player_p", "player_q"]].max(axis=1).astype("int64")
    pairs["match_id"] = pairs["match_id"].astype("int64")
    pairs["team_id"] = pairs["team_id"].astype("int64")
    pairs["vaep_q"] = pairs["vaep_q"].fillna(0.0)

    return (
        pairs.groupby(["match_id", "team_id", "player_a", "player_b"], as_index=False)["vaep_q"]
             .sum()
             .rename(columns={"vaep_q": "joi_vaep"})
    )


def compute_pass_edges(actions: pd.DataFrame) -> pd.DataFrame:
    """Estimate completed-pass edges (sender, receiver, team) per context.

    SPADL passes don't have an explicit recipient field. We infer the
    recipient as the *next* same-team action's actor within the same
    match and period.
    """
    df = actions.copy()
    df = df.sort_values(["match_id", "period_id", "time_seconds"]).reset_index(drop=True)
    nxt = df.shift(-1)

    is_pass = (
        df["type_id"].isin(PASS_TYPES)
        & (df["result_id"] == 1)  # successful only
        & (df["match_id"] == nxt["match_id"])
        & (df["period_id"] == nxt["period_id"])
        & (df["team_id"] == nxt["team_id"])
        & (df["player_id"].notna())
        & (nxt["player_id"].notna())
        & (df["player_id"] != nxt["player_id"])
    )

    edges = pd.DataFrame({
        "team_id": df["team_id"],
        "sender_id": df["player_id"],
        "receiver_id": nxt["player_id"],
    })[is_pass].reset_index(drop=True)
    if edges.empty:
        return pd.DataFrame(columns=["team_id", "sender_id", "receiver_id", "count"])

    edges["sender_id"] = edges["sender_id"].astype("int64")
    edges["receiver_id"] = edges["receiver_id"].astype("int64")
    edges["team_id"] = edges["team_id"].astype("int64")
    grouped = edges.groupby(["team_id", "sender_id", "receiver_id"], as_index=False).size()
    grouped = grouped.rename(columns={"size": "count"})
    return grouped


def eigenvector_centrality_numpy(adj: np.ndarray, max_iter: int = 1000, tol: float = 1e-8) -> np.ndarray:
    """Power-iteration eigenvector centrality on an undirected weighted
    adjacency matrix. Returns L2-normalized principal eigenvector
    (non-negative entries by Perron–Frobenius for non-negative adj).
    """
    n = adj.shape[0]
    if n == 0:
        return np.zeros(0)
    x = np.ones(n) / np.sqrt(n)
    for _ in range(max_iter):
        x_new = adj @ x
        norm = np.linalg.norm(x_new)
        if norm == 0:
            return np.zeros(n)
        x_new = x_new / norm
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    # Make non-negative (sign convention)
    if (x < 0).sum() > (x > 0).sum():
        x = -x
    return x


def compute_team_centrality(edges_team: pd.DataFrame) -> tuple[dict[int, float], dict[int, float], float]:
    """Returns (eigenvector_centrality_by_player, weighted_degree_by_player,
    team_total_pass_count)."""
    if edges_team.empty:
        return {}, {}, 0.0

    # Symmetrize: edge weight (i,j) = total passes in either direction.
    sym = (
        edges_team.assign(a=edges_team[["sender_id", "receiver_id"]].min(axis=1),
                          b=edges_team[["sender_id", "receiver_id"]].max(axis=1))
                  .groupby(["a", "b"], as_index=False)["count"].sum()
    )

    players = sorted(set(sym["a"].tolist()) | set(sym["b"].tolist()))
    idx = {pid: i for i, pid in enumerate(players)}
    n = len(players)
    adj = np.zeros((n, n), dtype=float)
    for _, row in sym.iterrows():
        i, j, c = idx[int(row["a"])], idx[int(row["b"])], float(row["count"])
        adj[i, j] = c
        adj[j, i] = c

    eig_vec = eigenvector_centrality_numpy(adj)
    eig_by_player = {players[i]: float(eig_vec[i]) for i in range(n)}

    # Weighted degree = fraction of team total pass volume the player is
    # involved in (sender or receiver). Each directed pass counts once
    # toward both endpoints' degree; team total = 2 * total_passes.
    total_passes = float(edges_team["count"].sum())
    if total_passes == 0:
        return eig_by_player, {p: 0.0 for p in players}, 0.0

    degree_by_player: dict[int, float] = {p: 0.0 for p in players}
    sender_sum = edges_team.groupby("sender_id")["count"].sum().to_dict()
    receiver_sum = edges_team.groupby("receiver_id")["count"].sum().to_dict()
    for p in players:
        deg = float(sender_sum.get(p, 0)) + float(receiver_sum.get(p, 0))
        # Fraction of total directed pass endpoints (= 2 * total_passes)
        degree_by_player[p] = deg / (2.0 * total_passes)

    return eig_by_player, degree_by_player, total_passes


# ── Main build ─────────────────────────────────────────────────────────────


def build_context(ctx: dict, lineups_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build player-context rows + per-pair shared minutes table for one context.

    Returns:
        (player_context_rows_df, pair_shared_minutes_df)
    """
    ctx_key = ctx["key"]
    ctx_label = ctx["label"]
    ctx_type = ctx["type"]
    log.info("=== Context %s (%s) ===", ctx_key, ctx_label)

    # ── Lineups slice
    lin = lineups_all[lineups_all["competition_key"] == ctx_key].copy()
    if lin.empty:
        log.warning("  No lineups for %s, skipping", ctx_key)
        return pd.DataFrame(), pd.DataFrame()

    # ── Team name map (team_name in lineups -> team_id via raw matches)
    team_id_map = load_team_name_map(ctx_key)
    name_to_id = {v: k for k, v in team_id_map.items()}
    lin["team_id"] = lin["team_name"].map(name_to_id).astype("Int64")
    missing_team = lin["team_id"].isna().sum()
    if missing_team:
        log.warning("  %d lineup rows have no team_id mapping (will inherit from SPADL)", missing_team)

    # ── Load VAEP-scored actions and SPADL
    actions = load_vaep_for_context(ctx_key)
    if actions.empty:
        log.warning("  No VAEP v2 actions for %s, skipping", ctx_key)
        return pd.DataFrame(), pd.DataFrame()

    # Backfill team_id from actions for any lineup rows missing it
    if missing_team:
        # For each (match_id, player_id), most-frequent team_id from actions
        a_team = (
            actions.dropna(subset=["player_id"])
                   .groupby(["match_id", "player_id"])["team_id"].agg(lambda s: int(s.mode().iloc[0]))
                   .reset_index()
                   .rename(columns={"team_id": "team_id_from_actions"})
        )
        lin = lin.merge(a_team, on=["match_id", "player_id"], how="left")
        lin["team_id"] = lin["team_id"].fillna(lin["team_id_from_actions"]).astype("Int64")
        lin = lin.drop(columns=["team_id_from_actions"])

    # ── On/off minutes per player per match (from raw events)
    match_ids = sorted(set(int(m) for m in lin["match_id"].unique()))
    onoff = compute_player_match_onoff(ctx_key, match_ids)
    if onoff.empty:
        log.warning("  No on/off minutes computed for %s — pair JOI90 will be empty", ctx_key)

    # ── Pair shared minutes (per match)
    pair_sm_per_match = compute_pair_shared_minutes(onoff) if not onoff.empty else pd.DataFrame(
        columns=["match_id", "team_id", "player_a", "player_b", "shared_minutes"]
    )

    # Aggregate to context-level pair shared minutes
    if not pair_sm_per_match.empty:
        pair_sm_ctx = (
            pair_sm_per_match.groupby(["team_id", "player_a", "player_b"], as_index=False)["shared_minutes"].sum()
        )
    else:
        pair_sm_ctx = pd.DataFrame(columns=["team_id", "player_a", "player_b", "shared_minutes"])

    # ── JOI per match -> aggregate to context-level
    joi_per_match = compute_joi_per_match(actions)
    if not joi_per_match.empty:
        joi_ctx = (
            joi_per_match.groupby(["team_id", "player_a", "player_b"], as_index=False)["joi_vaep"].sum()
        )
    else:
        joi_ctx = pd.DataFrame(columns=["team_id", "player_a", "player_b", "joi_vaep"])

    # Merge JOI with shared minutes; compute JOI90
    pair_ctx = pair_sm_ctx.merge(
        joi_ctx, on=["team_id", "player_a", "player_b"], how="left"
    )
    pair_ctx["joi_vaep"] = pair_ctx["joi_vaep"].fillna(0.0)
    pair_ctx["joi90_vaep"] = np.where(
        pair_ctx["shared_minutes"] >= MIN_PAIR_MINUTES,
        pair_ctx["joi_vaep"] * 90.0 / pair_ctx["shared_minutes"].clip(lower=1e-9),
        np.nan,
    )

    # ── Per-player aggregates from VAEP-scored actions
    actions_p = actions.dropna(subset=["player_id"]).copy()
    actions_p["player_id"] = actions_p["player_id"].astype("int64")
    actions_p["offensive_value"] = np.where(
        actions_p["type_id"].isin(DEFENSIVE_TYPES), 0.0, actions_p["vaep_value"]
    )
    actions_p["defensive_value"] = np.where(
        actions_p["type_id"].isin(DEFENSIVE_TYPES), actions_p["vaep_value"], 0.0
    )

    player_aggs = (
        actions_p.groupby("player_id", as_index=False)
                 .agg(n_actions=("action_id", "count"),
                      sum_vaep=("vaep_value", "sum"),
                      sum_off=("offensive_value", "sum"),
                      sum_def=("defensive_value", "sum"))
    )

    # ── Per-player minutes / n_matches / team
    # Sum minutes_played across matches in context
    lin["player_id"] = lin["player_id"].astype("int64")
    lin_grp = (
        lin.dropna(subset=["team_id"]).copy()
    )
    lin_grp["team_id"] = lin_grp["team_id"].astype("int64")

    player_min = (
        lin_grp.groupby(["player_id"], as_index=False)
               .agg(minutes_played=("minutes_played", "sum"),
                    n_matches=("match_id", "nunique"),
                    player_name=("player_name", "first"))
    )

    # Most-played team for each player
    player_team_min = (
        lin_grp.groupby(["player_id", "team_id"], as_index=False)["minutes_played"].sum()
    )
    player_team_min = player_team_min.sort_values(["player_id", "minutes_played"], ascending=[True, False])
    player_top_team = player_team_min.drop_duplicates(subset=["player_id"], keep="first")[
        ["player_id", "team_id"]
    ]
    player_min = player_min.merge(player_top_team, on="player_id", how="left")
    player_min["team_name"] = player_min["team_id"].map(team_id_map)

    # Apply minimum threshold
    player_min = player_min[player_min["minutes_played"] >= MIN_CONTEXT_MINUTES].copy()
    if player_min.empty:
        log.warning("  No players >= %g minutes in %s", MIN_CONTEXT_MINUTES, ctx_key)
        return pd.DataFrame(), pair_sm_ctx.assign(context_id=ctx_key)

    # ── Centrality per team from pass graph
    spadl_actions = load_spadl_for_context(ctx_key, ctx_type)
    if spadl_actions.empty:
        log.warning("  No SPADL actions for %s; centrality will be NaN", ctx_key)
        eig_by_team: dict[int, dict[int, float]] = {}
        deg_by_team: dict[int, dict[int, float]] = {}
    else:
        edges = compute_pass_edges(spadl_actions)
        eig_by_team = {}
        deg_by_team = {}
        for tid, sub in edges.groupby("team_id"):
            eig, deg, _ = compute_team_centrality(sub)
            eig_by_team[int(tid)] = eig
            deg_by_team[int(tid)] = deg

    # ── Build player-context rows
    rows = []
    name_lookup = lin.drop_duplicates(subset=["player_id"]).set_index("player_id")["player_name"].to_dict()

    # JOI90 for each player (rows in pair_ctx contain only pairs with
    # shared >= 180; joi90 might be NaN otherwise).
    pair_valid = pair_ctx[pair_ctx["joi90_vaep"].notna()].copy()

    for _, r in player_min.iterrows():
        pid = int(r["player_id"])
        team_id = int(r["team_id"]) if pd.notna(r["team_id"]) else None

        # Player aggregates
        agg_row = player_aggs[player_aggs["player_id"] == pid]
        if agg_row.empty:
            n_actions = 0
            sum_vaep = sum_off = sum_def = 0.0
        else:
            ar = agg_row.iloc[0]
            n_actions = int(ar["n_actions"])
            sum_vaep = float(ar["sum_vaep"])
            sum_off = float(ar["sum_off"])
            sum_def = float(ar["sum_def"])
        minutes = float(r["minutes_played"])
        per90 = lambda x: float(x * 90.0 / minutes) if minutes > 0 else 0.0

        # Top-5 partners by joi90_vaep
        partner_rows = pair_valid[
            ((pair_valid["player_a"] == pid) | (pair_valid["player_b"] == pid))
            & (pair_valid["team_id"] == team_id)
        ].copy()
        partner_rows["partner_id"] = np.where(
            partner_rows["player_a"] == pid, partner_rows["player_b"], partner_rows["player_a"]
        )
        partner_rows = partner_rows.sort_values("joi90_vaep", ascending=False).head(TOP_K)

        partner_ids = partner_rows["partner_id"].astype(int).tolist()
        partner_joi90s = partner_rows["joi90_vaep"].tolist()
        partner_names = [name_lookup.get(int(p), str(int(p))) for p in partner_ids]
        joi_top5_sum = float(np.nansum(partner_joi90s)) if partner_joi90s else float("nan")

        # Embeddedness: for each top-5 partner, sum that partner's
        # JOI90 with OTHER teammates (excluding pid) in the same team.
        embed_total = 0.0
        if team_id is not None and partner_ids:
            for partner_id in partner_ids:
                others = pair_valid[
                    (pair_valid["team_id"] == team_id)
                    & (
                        ((pair_valid["player_a"] == partner_id) & (pair_valid["player_b"] != pid))
                        | ((pair_valid["player_b"] == partner_id) & (pair_valid["player_a"] != pid))
                    )
                ]
                embed_total += float(others["joi90_vaep"].sum())
        embed = embed_total if partner_ids else float("nan")

        # Centrality
        cent_eig = float("nan")
        cent_deg = float("nan")
        if team_id in eig_by_team and pid in eig_by_team[team_id]:
            cent_eig = eig_by_team[team_id][pid]
        if team_id in deg_by_team and pid in deg_by_team[team_id]:
            cent_deg = deg_by_team[team_id][pid]

        row = {
            "player_id": pid,
            "player_name": r["player_name"],
            "context_type": ctx_type,
            "context_id": ctx_key,
            "context_label": ctx_label,
            "team_id": team_id,
            "team_name": r.get("team_name"),
            "minutes_played": minutes,
            "n_matches": int(r["n_matches"]),
            "n_actions": n_actions,
            "per90_vaep": per90(sum_vaep),
            "per90_offensive": per90(sum_off),
            "per90_defensive": per90(sum_def),
            "joi_top5_sum": joi_top5_sum,
            "centrality_eigen": cent_eig,
            "centrality_weighted_degree": cent_deg,
            "embeddedness_score": embed,
        }
        # Pad to 5 partners
        for i in range(TOP_K):
            if i < len(partner_ids):
                row[f"top_partner_{i+1}_id"] = int(partner_ids[i])
                row[f"top_partner_{i+1}_joi90"] = float(partner_joi90s[i])
                row[f"top_partner_{i+1}_name"] = partner_names[i]
            else:
                row[f"top_partner_{i+1}_id"] = pd.NA
                row[f"top_partner_{i+1}_joi90"] = float("nan")
                row[f"top_partner_{i+1}_name"] = pd.NA
        rows.append(row)

    chem_df = pd.DataFrame(rows)
    # Annotate pair_sm_ctx with context_id for career familiarity table
    pair_sm_ctx = pair_sm_ctx.assign(context_id=ctx_key)
    return chem_df, pair_sm_ctx


def build_career_familiarity(all_pair_sm: pd.DataFrame) -> pd.DataFrame:
    """Aggregate shared-minutes per (player_a, player_b) across all contexts."""
    if all_pair_sm.empty:
        return pd.DataFrame(
            columns=["player_a_id", "player_b_id", "total_shared_minutes", "n_contexts_shared", "contexts_shared"]
        )
    # all_pair_sm has team_id; we want to aggregate ignoring team — a
    # pair might appear in multiple teams across contexts (e.g.
    # international + club). Sum across all contexts/teams.
    df = all_pair_sm.copy()
    df["player_a_id"] = df[["player_a", "player_b"]].min(axis=1).astype("int64")
    df["player_b_id"] = df[["player_a", "player_b"]].max(axis=1).astype("int64")
    df = df[df["player_a_id"] != df["player_b_id"]]
    # First, sum per (pair, context) — same pair may appear under
    # multiple team_ids in same context for intl pairs that played for
    # different national teams (rare), but typical case is a single team_id.
    per_pair_ctx = (
        df.groupby(["player_a_id", "player_b_id", "context_id"], as_index=False)["shared_minutes"].sum()
    )
    agg = per_pair_ctx.groupby(["player_a_id", "player_b_id"], as_index=False).agg(
        total_shared_minutes=("shared_minutes", "sum"),
        n_contexts_shared=("context_id", "nunique"),
        contexts_shared=("context_id", lambda s: ",".join(sorted(set(s)))),
    )
    return agg


def main():
    log.info("Loading global lineups: %s", LINEUPS_PATH)
    lineups_all = pd.read_parquet(LINEUPS_PATH)
    log.info("  %d lineup rows across %d contexts", len(lineups_all), lineups_all["competition_key"].nunique())

    all_rows: list[pd.DataFrame] = []
    all_pair_sm: list[pd.DataFrame] = []
    usable_contexts: list[tuple[str, int]] = []
    for ctx in ALL_CONTEXTS:
        try:
            chem_df, pair_sm = build_context(ctx, lineups_all)
        except Exception as e:
            log.exception("  Build failed for %s: %s", ctx["key"], e)
            continue
        if not chem_df.empty:
            all_rows.append(chem_df)
            usable_contexts.append((ctx["key"], len(chem_df)))
        if not pair_sm.empty:
            all_pair_sm.append(pair_sm)

    if not all_rows:
        log.error("No player-context rows produced — aborting")
        return

    chemistry = pd.concat(all_rows, ignore_index=True)
    chemistry.to_parquet(OUT_CHEM, index=False)
    log.info("Wrote %s: %d rows", OUT_CHEM, len(chemistry))

    pair_all = pd.concat(all_pair_sm, ignore_index=True) if all_pair_sm else pd.DataFrame()
    familiarity = build_career_familiarity(pair_all)
    familiarity.to_parquet(OUT_FAMILIARITY, index=False)
    log.info("Wrote %s: %d rows", OUT_FAMILIARITY, len(familiarity))

    # ── Verification prints
    print("\n=== VERIFICATION ===")
    print(f"player_chemistry_v3.parquet: {len(chemistry)} rows")
    print(f"career_familiarity.parquet:  {len(familiarity)} rows")

    print("\nTop-10 player-contexts by per90_vaep (min 180 mins):")
    top10 = chemistry.sort_values("per90_vaep", ascending=False).head(10)
    print(top10[["player_name", "context_label", "team_name", "minutes_played", "per90_vaep", "joi_top5_sum"]].to_string(index=False))

    print("\nMessi rows (player_id=5503):")
    messi = chemistry[chemistry["player_id"] == 5503]
    print(messi[["context_label", "team_name", "minutes_played", "per90_vaep", "joi_top5_sum"]].to_string(index=False))

    high_min = chemistry[chemistry["minutes_played"] >= 500]
    if len(high_min):
        notnull_frac = high_min["joi_top5_sum"].notna().mean()
        print(f"\njoi_top5_sum non-null fraction (mins>=500): {notnull_frac:.3f}")

    # ── Notes file
    top5_lines = []
    for _, r in chemistry.sort_values("per90_vaep", ascending=False).head(5).iterrows():
        top5_lines.append(
            f"- {r['player_name']} — {r['context_label']} ({r['team_name']}): "
            f"per90_vaep={r['per90_vaep']:.4f}, minutes={r['minutes_played']:.0f}, "
            f"joi_top5_sum={r['joi_top5_sum']:.4f}"
        )
    ctx_lines = [f"- {k}: {n} player-context rows" for k, n in usable_contexts]

    notes = f"""# player_chemistry_v3 build notes

Generated: 2026-05-21

## Final row counts
- `player_chemistry_v3.parquet`: **{len(chemistry)}** rows
- `career_familiarity.parquet`: **{len(familiarity)}** pairs

## Contexts with usable data
{chr(10).join(ctx_lines)}

## Top-5 player-contexts by per90_vaep (sanity check)
{chr(10).join(top5_lines)}

## Methodological decisions
- VAEP v2 scored files have a single ``vaep_value`` column; we split
  per-action by SPADL action type. Defensive types (tackle, interception,
  clearance, keeper actions) contribute to ``per90_defensive``; everything
  else contributes to ``per90_offensive``. This is a coarser split than
  socceraction's native offensive/defensive value split (not in cache).

- JOI follows the existing repo convention
  (scripts/vaep_cross_context_pipeline.py): sum VAEP of the *second*
  action of every consecutive (different teammate, same team, same
  match) pair. JOI90 = JOI * 90 / shared_minutes, only valid for pairs
  with >= 180 shared minutes in the context.

- Shared minutes per pair computed from raw event Starting XI +
  Substitution events. Players who don't appear in events default to
  not being on the pitch. Red cards not modeled (rare).

- Pass graph for centrality uses SPADL pass-family types
  (pass/cross/freekick/corner) with result=success. Receiver inferred
  as next same-team-period action's actor. Eigenvector centrality
  computed via numpy power iteration on the symmetrized adjacency
  (networkx not installed in .venv and pip install blocked per spec).

- Per-context minimum: 180 minutes (matches site filter).

- ``embeddedness_score``: for each of player P's top-5 partners Q, sum
  Q's JOI90 with each other teammate R (R != P), then sum across all
  five partners. Higher = top partners are themselves well-connected.

- ``career_familiarity``: shared minutes summed across ALL contexts
  (club + intl), per player pair. Same player in different teams within
  the same context is collapsed via team-aware match aggregation, then
  pair shared minutes summed.
"""
    OUT_NOTES.write_text(notes)
    log.info("Wrote %s", OUT_NOTES)


if __name__ == "__main__":
    main()
