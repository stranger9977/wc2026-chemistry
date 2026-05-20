"""Wyscout cross-context chemistry pipeline.

Stages (each cached; safe to re-run):
  1. Convert Wyscout JSON events -> SPADL parquets (per competition)
  2. Fit one xT model across all competitions
  3. Score every action with delta-xT
  4. Compute per-match pair JOI (summed delta-xT of second action in consecutive pair)
  5. Compute shared minutes per pair per match from substitution events
  6. Output: outputs/wyscout_joi_per_match.parquet
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from socceraction.spadl import wyscout as wy_spadl
from socceraction.xthreat import ExpectedThreat

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_WY = ROOT / "data" / "wyscout"
DATA_SPADL = ROOT / "data" / "wyscout_spadl"
OUTPUTS = ROOT / "outputs"

DATA_SPADL.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

XT_MODEL_PATH = DATA_WY / "xt.pkl"
JOI_PER_MATCH_PATH = OUTPUTS / "wyscout_joi_per_match.parquet"

COMPETITIONS = [
    "England",
    "France",
    "Germany",
    "Italy",
    "Spain",
    "European_Championship",
    "World_Cup",
]

# ---------------------------------------------------------------------------
# Stage 1 – SPADL conversion
# ---------------------------------------------------------------------------

def load_matches(comp: str) -> pd.DataFrame:
    path = DATA_WY / f"matches_{comp}.json"
    with open(path) as f:
        matches = json.load(f)
    df = pd.DataFrame(matches)
    return df


PERIOD_MAP = {"1H": 1, "2H": 2, "E1": 3, "E2": 4, "P": 5}


def load_events(comp: str) -> pd.DataFrame:
    path = DATA_WY / f"events_{comp}.json"
    log.info("Loading events for %s ...", comp)
    with open(path) as f:
        events = json.load(f)
    df = pd.DataFrame(events)
    # Rename to match what socceraction's wyscout converter expects
    df = df.rename(columns={
        "id": "event_id",
        "matchId": "game_id",
        "teamId": "team_id",
        "playerId": "player_id",
        "eventId": "type_id",
        "subEventId": "subtype_id",
        "eventSec": "milliseconds",
    })
    df["milliseconds"] = df["milliseconds"] * 1000.0
    df["period_id"] = df["matchPeriod"].map(PERIOD_MAP).fillna(1).astype(int)
    return df


def convert_competition(comp: str) -> pd.DataFrame:
    """Convert one competition to SPADL. Returns combined DataFrame."""
    out_path = DATA_SPADL / f"spadl_{comp}.parquet"
    if out_path.exists():
        log.info("SPADL cache hit: %s", comp)
        return pd.read_parquet(out_path)

    log.info("Converting %s to SPADL ...", comp)
    matches = load_matches(comp)
    events_df = load_events(comp)

    # Wyscout match structure: wyId, label, dateutc, etc.
    # Need home_team_id per match
    all_actions = []
    for _, match in matches.iterrows():
        match_id = match["wyId"]
        # teamsData: dict keyed by team id, with "side" -> "home"/"away"
        teams_data = match.get("teamsData", {})
        home_team_id = None
        for tid, tdata in teams_data.items():
            if tdata.get("side") == "home":
                home_team_id = int(tid)
                break
        if home_team_id is None:
            log.warning("No home team found for match %s, skipping", match_id)
            continue

        match_events = events_df[events_df["game_id"] == match_id].copy()
        if match_events.empty:
            continue

        try:
            actions = wy_spadl.convert_to_actions(match_events, home_team_id)
            # game_id is already set by converter from game_id column
            actions["match_id"] = actions["game_id"]
            actions["competition"] = comp
            all_actions.append(actions)
        except Exception as e:
            log.warning("Failed to convert match %s: %s", match_id, e)
            continue

    if not all_actions:
        log.error("No actions converted for %s", comp)
        return pd.DataFrame()

    combined = pd.concat(all_actions, ignore_index=True)
    combined.to_parquet(out_path, index=False)
    log.info("Saved SPADL for %s: %d actions", comp, len(combined))
    return combined


def stage1_convert_all() -> pd.DataFrame:
    """Convert all competitions, return combined SPADL DataFrame."""
    all_parts = []
    for comp in COMPETITIONS:
        df = convert_competition(comp)
        if not df.empty:
            all_parts.append(df)

    combined = pd.concat(all_parts, ignore_index=True)
    log.info("Total SPADL actions across all competitions: %d", len(combined))
    return combined


# ---------------------------------------------------------------------------
# Stage 2 – xT model fitting
# ---------------------------------------------------------------------------

def stage2_fit_xt(all_actions: pd.DataFrame) -> ExpectedThreat:
    """Fit (or load) xT model on all SPADL actions."""
    if XT_MODEL_PATH.exists():
        log.info("xT model cache hit: %s", XT_MODEL_PATH)
        with open(XT_MODEL_PATH, "rb") as f:
            xt = pickle.load(f)
        return xt

    log.info("Fitting xT model on %d actions ...", len(all_actions))
    xt = ExpectedThreat(l=16, w=12)
    xt.fit(all_actions)

    XT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(XT_MODEL_PATH, "wb") as f:
        pickle.dump(xt, f)

    # Sanity check: print top-scoring cells
    top = np.unravel_index(np.argsort(xt.xT.ravel())[-5:], xt.xT.shape)
    log.info("Top xT cells (row, col): %s", list(zip(top[0].tolist(), top[1].tolist())))
    return xt


# ---------------------------------------------------------------------------
# Stage 3 – Score actions
# ---------------------------------------------------------------------------

def score_actions(actions: pd.DataFrame, xt: ExpectedThreat) -> pd.DataFrame:
    """Add xt_delta column to actions. Returns actions with xt_delta."""
    try:
        raw = xt.rate(actions)
        actions = actions.copy()
        actions["xt_delta"] = pd.Series(raw, index=actions.index).fillna(0.0)
    except Exception as e:
        log.warning("xt.rate failed: %s; filling zeros", e)
        actions = actions.copy()
        actions["xt_delta"] = 0.0
    return actions


# ---------------------------------------------------------------------------
# Stage 4 – Per-match pair JOI
# ---------------------------------------------------------------------------

# SPADL type IDs that count as on-the-ball actions for JOI.
# 0=pass, 1=cross, 2=throw_in, 3=freekick_crossed, 4=freekick_short,
# 5=corner_crossed, 6=corner_short, 7=take_on, 8=foul, 9=tackle,
# 10=interception, 11=shot, 12=shot_penalty, 13=shot_freekick,
# 14=keeper_save, 15=keeper_claim, 16=keeper_punch, 17=keeper_pick_up,
# 18=clearance, 19=bad_touch, 20=non_action, 21=dribble, 22=goalkick
ELIGIBLE_TYPES: frozenset[int] = frozenset({0, 1, 3, 4, 5, 6, 7, 11, 12, 13, 21})


def compute_joi_per_match(actions: pd.DataFrame) -> pd.DataFrame:
    """Compute per-match pair JOI (xT-based).

    Returns DataFrame with columns:
        match_id, competition, player_a, player_b, team_id, joi_xt
    """
    df = actions[actions["type_id"].isin(ELIGIBLE_TYPES)].copy()
    df = df.sort_values(["match_id", "period_id", "time_seconds"]).reset_index(drop=True)

    # Shift to get next action
    nxt = df.shift(-1)

    consecutive = (
        (df["match_id"] == nxt["match_id"])
        & (df["team_id"] == nxt["team_id"])
        & (df["player_id"] != nxt["player_id"])
    )

    pairs = pd.DataFrame({
        "match_id": df["match_id"],
        "competition": df["competition"],
        "team_id": df["team_id"],
        "player_p": df["player_id"],
        "player_q": nxt["player_id"],
        "xt_q": nxt["xt_delta"],
    })[consecutive].reset_index(drop=True)

    # Canonicalize pair order
    pairs["player_a"] = pairs[["player_p", "player_q"]].min(axis=1).astype("int64")
    pairs["player_b"] = pairs[["player_p", "player_q"]].max(axis=1).astype("int64")
    pairs["xt_q"] = pairs["xt_q"].fillna(0.0)

    grouped = (
        pairs.groupby(["match_id", "competition", "team_id", "player_a", "player_b"], as_index=False)
             ["xt_q"].sum()
             .rename(columns={"xt_q": "joi_xt"})
    )
    return grouped


# ---------------------------------------------------------------------------
# Stage 5 – Shared minutes per pair per match
# ---------------------------------------------------------------------------

def compute_shared_minutes(comp: str, matches: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """Derive per-player minutes intervals from lineups and substitutions.

    Returns DataFrame: match_id, player_id, minute_on, minute_off
    """
    rows = []
    for _, match in matches.iterrows():
        match_id = match["wyId"]
        teams_data = match.get("teamsData", {})

        # Get all substitution events for this match (events_df has been renamed)
        subs = events_df[
            (events_df["game_id"] == match_id)
            & (events_df["eventName"] == "Substitution")
        ].copy()

        for tid, tdata in teams_data.items():
            team_id = int(tid)
            lineup = tdata.get("formation", {}).get("lineup", [])
            bench = tdata.get("formation", {}).get("bench", [])
            subs_list = tdata.get("formation", {}).get("substitutions", [])

            # Players in starting lineup
            starters = set()
            for player in lineup:
                pid = player.get("playerId")
                if pid:
                    starters.add(int(pid))

            # Build substitution map
            # subs_list: [{"playerIn": id, "playerOut": id, "minute": m}, ...]
            sub_out = {}  # player_id -> minute subbed out
            sub_in = {}   # player_id -> minute subbed in

            for sub in subs_list:
                # Some matches have "null" string entries — skip
                if not isinstance(sub, dict):
                    continue
                pout = sub.get("playerOut")
                pin = sub.get("playerIn")
                minute = sub.get("minute", 90)
                if pout:
                    sub_out[int(pout)] = float(minute)
                if pin:
                    sub_in[int(pin)] = float(minute)

            # Assign intervals
            for pid in starters:
                off = sub_out.get(pid, 90.0)
                rows.append({
                    "match_id": match_id,
                    "team_id": team_id,
                    "player_id": pid,
                    "minute_on": 0.0,
                    "minute_off": min(float(off), 90.0),
                })

            for pid, minute_on in sub_in.items():
                rows.append({
                    "match_id": match_id,
                    "team_id": team_id,
                    "player_id": pid,
                    "minute_on": float(minute_on),
                    "minute_off": 90.0,
                })

    if not rows:
        return pd.DataFrame(columns=["match_id", "team_id", "player_id", "minute_on", "minute_off"])
    return pd.DataFrame(rows)


def compute_pair_shared_minutes(player_minutes: pd.DataFrame) -> pd.DataFrame:
    """Cross-join players per match/team to get shared minutes for each pair.

    Returns DataFrame: match_id, team_id, player_a, player_b, shared_minutes
    """
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


def stage5_shared_minutes() -> pd.DataFrame:
    """Compute shared minutes for all competitions. Cached."""
    cache_path = DATA_SPADL / "shared_minutes.parquet"
    if cache_path.exists():
        log.info("Shared minutes cache hit")
        return pd.read_parquet(cache_path)

    log.info("Computing shared minutes for all competitions ...")
    all_parts = []
    for comp in COMPETITIONS:
        matches = load_matches(comp)
        # Pass an empty DataFrame for events_df — compute_shared_minutes only uses
        # teamsData from matches (the events_df query for Substitution events is unused)
        empty_events = pd.DataFrame(columns=["game_id", "eventName"])
        player_mins = compute_shared_minutes(comp, matches, empty_events)
        player_mins["competition"] = comp
        pair_mins = compute_pair_shared_minutes(player_mins)
        pair_mins["competition"] = comp
        all_parts.append(pair_mins)
        log.info("  %s: %d pair-match rows", comp, len(pair_mins))

    combined = pd.concat(all_parts, ignore_index=True)
    combined.to_parquet(cache_path, index=False)
    log.info("Total pair-match shared minutes rows: %d", len(combined))
    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== Stage 1: SPADL conversion ===")
    all_actions = stage1_convert_all()

    log.info("=== Stage 2: xT model fitting ===")
    xt = stage2_fit_xt(all_actions)

    log.info("=== Stage 3: Score actions ===")
    # Score competition by competition to avoid OOM
    scored_parts = []
    for comp in COMPETITIONS:
        cache_path = DATA_SPADL / f"scored_{comp}.parquet"
        if cache_path.exists():
            log.info("Scored cache hit: %s", comp)
            scored_parts.append(pd.read_parquet(cache_path))
            continue
        comp_actions = all_actions[all_actions["competition"] == comp].copy()
        log.info("Scoring %s: %d actions ...", comp, len(comp_actions))
        scored = score_actions(comp_actions, xt)
        scored.to_parquet(cache_path, index=False)
        scored_parts.append(scored)
        log.info("  Scored %s", comp)
        del comp_actions, scored

    all_scored = pd.concat(scored_parts, ignore_index=True)
    del all_actions, scored_parts

    log.info("=== Stage 4: Per-match pair JOI ===")
    joi_parts = []
    for comp in COMPETITIONS:
        cache_path = DATA_SPADL / f"joi_{comp}.parquet"
        if cache_path.exists():
            log.info("JOI cache hit: %s", comp)
            joi_parts.append(pd.read_parquet(cache_path))
            continue
        comp_scored = all_scored[all_scored["competition"] == comp].copy()
        log.info("Computing JOI for %s: %d actions ...", comp, len(comp_scored))
        joi = compute_joi_per_match(comp_scored)
        joi.to_parquet(cache_path, index=False)
        joi_parts.append(joi)
        log.info("  JOI %s: %d pair-match rows", comp, len(joi))
        del comp_scored

    all_joi = pd.concat(joi_parts, ignore_index=True)
    del all_scored, joi_parts

    log.info("=== Stage 5: Shared minutes ===")
    shared_mins = stage5_shared_minutes()

    log.info("=== Stage 6: Merge JOI with shared minutes ===")
    merged = all_joi.merge(
        shared_mins[["match_id", "team_id", "player_a", "player_b", "shared_minutes"]],
        on=["match_id", "team_id", "player_a", "player_b"],
        how="left",
    )
    merged["shared_minutes"] = merged["shared_minutes"].fillna(0.0)

    merged.to_parquet(JOI_PER_MATCH_PATH, index=False)
    log.info("Saved per-match JOI: %s (%d rows)", JOI_PER_MATCH_PATH, len(merged))

    # Summary
    log.info("=== Summary ===")
    log.info("Competitions: %s", merged["competition"].value_counts().to_dict())
    log.info("Total pair-match interactions: %d", len(merged))
    log.info("Unique matches with interactions: %d", merged["match_id"].nunique())


if __name__ == "__main__":
    main()
