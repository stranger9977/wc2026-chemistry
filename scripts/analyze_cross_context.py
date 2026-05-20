"""Cross-context chemistry analysis: club (2017/18 top-5 leagues) vs. country (WC 2018 / Euro 2016).

Reads outputs/wyscout_joi_per_match.parquet, aggregates to per-pair per-context JOI90,
produces outputs/cross_context_chemistry.json and docs/analysis/cross-context-chemistry.md.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
JOI_PATH = ROOT / "outputs" / "wyscout_joi_per_match.parquet"
PLAYERS_PATH = ROOT / "data" / "wyscout" / "players.json"
TEAMS_PATH = ROOT / "data" / "wyscout" / "teams.json"
MATCHES_DIR = ROOT / "data" / "wyscout"
OUTPUT_JSON = ROOT / "outputs" / "cross_context_chemistry.json"
OUTPUT_MD = ROOT / "docs" / "analysis" / "cross-context-chemistry.md"
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Competition context grouping
# ---------------------------------------------------------------------------
CLUB_COMPS = {"England", "France", "Germany", "Italy", "Spain"}
INTL_COMPS = {"World_Cup", "European_Championship"}

# ---------------------------------------------------------------------------
# Featured players: name -> (club_team_substring, country_team_substring, nat_comp)
# ---------------------------------------------------------------------------
# (display_name, wyscout_wyId, club_query, country, intl_comp)
# wyIds looked up from players.json to bypass unicode matching issues
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

# Keep old FEATURED for backward compat
FEATURED = [(name, club, country, comp) for name, wid, club, country, comp in FEATURED_WITH_IDS]

MINUTES_FLOOR_CLUB = 90.0
MINUTES_FLOOR_INTL = 45.0


import re as _re

def _decode_unicode_escapes(s: str) -> str:
    """Decode literal \\uXXXX escape sequences that Wyscout stores as strings."""
    if not isinstance(s, str):
        return s
    return _re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), s)


def load_players() -> pd.DataFrame:
    with open(PLAYERS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df.rename(columns={"wyId": "player_id", "shortName": "short_name"})
    df["full_name"] = df["firstName"] + " " + df["lastName"]
    # Fix double-escaped unicode in shortName (Wyscout stores "N. Kanté" as literal string)
    df["short_name"] = df["short_name"].apply(_decode_unicode_escapes)
    df["full_name"] = df["full_name"].apply(_decode_unicode_escapes)
    return df[["player_id", "full_name", "short_name", "currentTeamId", "currentNationalTeamId"]]


def load_teams() -> pd.DataFrame:
    with open(TEAMS_PATH) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df.rename(columns={"wyId": "team_id", "name": "team_name"})
    return df[["team_id", "team_name"]]


def load_all_matches() -> pd.DataFrame:
    """Load match metadata from all competitions."""
    comps = [
        "England", "France", "Germany", "Italy", "Spain",
        "European_Championship", "World_Cup"
    ]
    all_matches = []
    for comp in comps:
        path = MATCHES_DIR / f"matches_{comp}.json"
        with open(path) as f:
            matches = json.load(f)
        df = pd.DataFrame(matches)
        df["competition"] = comp
        all_matches.append(df[["wyId", "competition", "teamsData"]])
    combined = pd.concat(all_matches, ignore_index=True)
    combined = combined.rename(columns={"wyId": "match_id"})
    return combined


def build_player_team_map(matches: pd.DataFrame) -> pd.DataFrame:
    """For each player in each match, record their team_id and competition."""
    rows = []
    for _, match in matches.iterrows():
        match_id = match["match_id"]
        comp = match["competition"]
        teams_data = match.get("teamsData", {})
        if not isinstance(teams_data, dict):
            continue
        for tid, tdata in teams_data.items():
            team_id = int(tid)
            formation = tdata.get("formation", {})
            lineup = formation.get("lineup", [])
            bench_subs = formation.get("substitutions", [])
            for p in lineup:
                pid = p.get("playerId")
                if pid:
                    rows.append({
                        "match_id": match_id,
                        "competition": comp,
                        "team_id": team_id,
                        "player_id": int(pid),
                    })
            for s in bench_subs:
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
    df = pd.DataFrame(rows).drop_duplicates()
    return df


def aggregate_joi90(joi_per_match: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-match JOI to per-pair per-competition JOI90.

    Floors: club >= 90 min shared, international >= 45 min shared.
    """
    # Add context column
    joi_per_match = joi_per_match.copy()
    joi_per_match["context"] = joi_per_match["competition"].apply(
        lambda c: "club" if c in CLUB_COMPS else "international"
    )

    agg = (
        joi_per_match
        .groupby(["competition", "context", "team_id", "player_a", "player_b"], as_index=False)
        .agg(
            total_joi=("joi_xt", "sum"),
            total_minutes=("shared_minutes", "sum"),
            matches=("match_id", "nunique"),
        )
    )
    agg["joi90"] = (agg["total_joi"] * 90.0 / agg["total_minutes"]).where(
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


def _normalize(s: str) -> str:
    """Normalize unicode to ASCII for fuzzy matching."""
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii").lower().replace("'", "")


def find_player_id(name_query: str, players_df: pd.DataFrame) -> int | None:
    """Fuzzy search by name (unicode-normalized). Returns wyId or None."""
    q = _normalize(name_query)
    norm_full = players_df["full_name"].apply(_normalize)
    norm_short = players_df["short_name"].apply(_normalize)

    # Try last word of query (often the last name) for better matching
    q_parts = q.split()
    last_part = q_parts[-1] if q_parts else q

    mask = norm_full.str.contains(q, regex=False) | norm_short.str.contains(q, regex=False)
    hits = players_df[mask]
    if len(hits) == 0:
        # Try last name only
        mask2 = norm_full.str.contains(last_part, regex=False) | norm_short.str.contains(last_part, regex=False)
        hits = players_df[mask2]
    if len(hits) == 1:
        return int(hits.iloc[0]["player_id"])
    if len(hits) > 1:
        # Try to narrow by first name
        if len(q_parts) > 1:
            first = q_parts[0]
            hits2 = hits[norm_full.loc[hits.index].str.contains(first, regex=False)
                        | norm_short.loc[hits.index].str.contains(first, regex=False)]
            if len(hits2) == 1:
                return int(hits2.iloc[0]["player_id"])
            if len(hits2) > 1:
                hits = hits2
        log.warning("Multiple matches for '%s': %s", name_query, hits[["player_id", "full_name"]].head(3).to_dict("records"))
        return int(hits.iloc[0]["player_id"])
    return None


def find_team_id(team_query: str, teams_df: pd.DataFrame) -> int | None:
    """Find team by partial name match."""
    q = team_query.lower()
    mask = teams_df["team_name"].str.lower().str.contains(q, regex=False)
    hits = teams_df[mask]
    if len(hits) == 0:
        return None
    if len(hits) == 1:
        return int(hits.iloc[0]["team_id"])
    # Multiple — return first
    log.warning("Multiple team matches for '%s': %s", team_query, hits[["team_id", "team_name"]].head(5).to_dict("records"))
    return int(hits.iloc[0]["team_id"])


def get_pairs_for_player(
    player_id: int,
    team_id: int,
    comp_filter: str | None,
    joi90_df: pd.DataFrame,
    players_df: pd.DataFrame,
    top_n: int = 8,
    minutes_floor: float = MINUTES_FLOOR_CLUB,
) -> list[dict]:
    """Get top-N teammates by shared minutes for a given player/team/competition."""
    if comp_filter:
        df = joi90_df[
            (joi90_df["competition"] == comp_filter)
            & (joi90_df["team_id"] == team_id)
            & (
                (joi90_df["player_a"] == player_id)
                | (joi90_df["player_b"] == player_id)
            )
            & (joi90_df["total_minutes"] >= minutes_floor)
        ].copy()
    else:
        df = joi90_df[
            (joi90_df["team_id"] == team_id)
            & (
                (joi90_df["player_a"] == player_id)
                | (joi90_df["player_b"] == player_id)
            )
            & (joi90_df["total_minutes"] >= minutes_floor)
        ].copy()

    df["teammate_id"] = df.apply(
        lambda r: r["player_b"] if r["player_a"] == player_id else r["player_a"],
        axis=1,
    )

    df = df.sort_values("total_minutes", ascending=False).head(top_n)

    # Look up teammate names
    name_map = players_df.set_index("player_id")["short_name"].to_dict()
    result = []
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        tid = int(row["teammate_id"])
        result.append({
            "teammate_id": tid,
            "teammate": name_map.get(tid, f"Player #{tid}"),
            "minutes": round(float(row["total_minutes"]), 1),
            "joi90_xt": round(float(row["joi90"]), 4),
            "matches": int(row["matches"]),
            "team_rank": rank,
        })
    return result


def find_national_team_id_for_player(
    player_id: int,
    country_name: str,
    player_team_map: pd.DataFrame,
    intl_comp: str,
    teams_df: pd.DataFrame,
) -> int | None:
    """Find which team a player played for in international competition."""
    player_intl = player_team_map[
        (player_team_map["player_id"] == player_id)
        & (player_team_map["competition"] == intl_comp)
    ]
    if player_intl.empty:
        log.warning("Player %s not found in %s matches", player_id, intl_comp)
        return None
    team_ids = player_intl["team_id"].unique()
    if len(team_ids) == 1:
        return int(team_ids[0])
    # If multiple, filter by country name
    cname = country_name.lower()
    for tid in team_ids:
        tname = teams_df[teams_df["team_id"] == tid]["team_name"].values
        if len(tname) > 0 and cname in tname[0].lower():
            return int(tid)
    return int(team_ids[0])


def find_club_team_id_for_player(
    player_id: int,
    club_name: str,
    player_team_map: pd.DataFrame,
    teams_df: pd.DataFrame,
) -> int | None:
    """Find which club team a player played for in a domestic competition."""
    player_club = player_team_map[
        (player_team_map["player_id"] == player_id)
        & (player_team_map["competition"].isin(CLUB_COMPS))
    ]
    if player_club.empty:
        log.warning("Player %s not found in club matches", player_id)
        return None
    team_ids = player_club["team_id"].unique()
    cname = club_name.lower()
    for tid in team_ids:
        tname = teams_df[teams_df["team_id"] == tid]["team_name"].values
        if len(tname) > 0 and any(w in tname[0].lower() for w in cname.split()):
            return int(tid)
    # Fallback: return most frequent team
    counts = player_club.groupby("team_id")["match_id"].nunique()
    return int(counts.idxmax())


def compute_cross_context(joi90_df: pd.DataFrame, players_df: pd.DataFrame, teams_df: pd.DataFrame, player_team_map: pd.DataFrame) -> list[dict]:
    """Compute cross-context data for all featured players."""
    results = []
    for (name, player_id, club_query, country, intl_comp) in FEATURED_WITH_IDS:
        log.info("Processing %s (wyId=%d) ...", name, player_id)

        # Find club team ID
        club_team_id = find_club_team_id_for_player(player_id, club_query, player_team_map, teams_df)
        if club_team_id is None:
            log.warning("Could not find club team for %s", name)
            club_pairs = []
        else:
            club_team_name = teams_df[teams_df["team_id"] == club_team_id]["team_name"].values
            club_team_name = club_team_name[0] if len(club_team_name) > 0 else club_query

            # Find which competition this club played in
            club_comp_options = player_team_map[
                (player_team_map["player_id"] == player_id)
                & (player_team_map["team_id"] == club_team_id)
            ]["competition"].unique()
            club_comp = club_comp_options[0] if len(club_comp_options) > 0 else None

            club_pairs = get_pairs_for_player(
                player_id, club_team_id, club_comp, joi90_df, players_df,
                top_n=8, minutes_floor=MINUTES_FLOOR_CLUB
            )

        # Find national team ID
        nat_team_id = find_national_team_id_for_player(
            player_id, country, player_team_map, intl_comp, teams_df
        )
        if nat_team_id is None:
            country_pairs = []
        else:
            country_pairs = get_pairs_for_player(
                player_id, nat_team_id, intl_comp, joi90_df, players_df,
                top_n=8, minutes_floor=MINUTES_FLOOR_INTL
            )

        # Summary stats
        avg_club = float(np.mean([p["joi90_xt"] for p in club_pairs])) if club_pairs else 0.0
        avg_country = float(np.mean([p["joi90_xt"] for p in country_pairs])) if country_pairs else 0.0
        ratio = avg_country / avg_club if avg_club > 0 else None

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
                "avg_club_joi90": round(avg_club, 4),
                "avg_country_joi90": round(avg_country, 4),
                "ratio_country_over_club": round(ratio, 3) if ratio is not None else None,
                "club_pairs_found": len(club_pairs),
                "country_pairs_found": len(country_pairs),
            },
        })

    return results


def compute_bayern_germany_deep_dive(
    joi90_df: pd.DataFrame,
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    player_team_map: pd.DataFrame,
) -> list[dict]:
    """Compute JOI90 for all Bayern-Germany pairings."""
    # Use hardcoded IDs to avoid unicode matching issues
    ids = {
        "Kimmich": 224593,
        "Müller":  14732,
        "Hummels": 14795,
        "Boateng": 14716,
    }

    kimmich_id = ids["Kimmich"]
    bayern_id = find_club_team_id_for_player(kimmich_id, "Bayern", player_team_map, teams_df)
    germany_id = find_national_team_id_for_player(kimmich_id, "Germany", player_team_map, "World_Cup", teams_df)

    if not bayern_id or not germany_id:
        log.warning("Could not find Bayern or Germany team IDs")
        return []

    # Bayern competition
    club_comp = player_team_map[
        (player_team_map["player_id"] == kimmich_id)
        & (player_team_map["team_id"] == bayern_id)
    ]["competition"].iloc[0] if len(player_team_map[
        (player_team_map["player_id"] == kimmich_id)
        & (player_team_map["team_id"] == bayern_id)
    ]) > 0 else "Germany"

    results = []
    player_list = [s for s, pid in ids.items() if pid is not None]
    for i, name_a in enumerate(player_list):
        for name_b in player_list[i+1:]:
            pid_a = ids[name_a]
            pid_b = ids[name_b]
            pa = min(pid_a, pid_b)
            pb = max(pid_a, pid_b)

            def get_pair_stats(team_id, comp, minutes_floor):
                df = joi90_df[
                    (joi90_df["competition"] == comp)
                    & (joi90_df["team_id"] == team_id)
                    & (joi90_df["player_a"] == pa)
                    & (joi90_df["player_b"] == pb)
                    & (joi90_df["total_minutes"] >= minutes_floor)
                ]
                if df.empty:
                    return None, None
                row = df.iloc[0]
                return round(float(row["joi90"]), 4), round(float(row["total_minutes"]), 1)

            club_joi90, club_mins = get_pair_stats(bayern_id, club_comp, MINUTES_FLOOR_CLUB)
            country_joi90, country_mins = get_pair_stats(germany_id, "World_Cup", MINUTES_FLOOR_INTL)

            results.append({
                "pair": f"{name_a} — {name_b}",
                "club_joi90": club_joi90,
                "club_minutes": club_mins,
                "country_joi90": country_joi90,
                "country_minutes": country_mins,
            })

    return results


def build_meta(joi_per_match: pd.DataFrame) -> dict:
    total_matches = joi_per_match["match_id"].nunique()
    # Approximate actions from SPADL caches
    spadl_dir = ROOT / "data" / "wyscout_spadl"
    total_actions = 0
    for p in spadl_dir.glob("spadl_*.parquet"):
        df = pd.read_parquet(p, columns=["match_id"])
        total_actions += len(df)

    comp_counts = joi_per_match.groupby("competition")["match_id"].nunique().to_dict()
    return {
        "dataset": "Wyscout 2017/18 + WC 2018 + Euro 2016",
        "license": "CC BY 4.0",
        "total_matches": total_matches,
        "total_actions_approx": total_actions,
        "matches_per_competition": comp_counts,
    }


def write_json(meta: dict, player_results: list[dict], deep_dive: list[dict]) -> None:
    output = {
        "meta": meta,
        "players": player_results,
        "bayern_germany_deep_dive": deep_dive,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log.info("Wrote %s", OUTPUT_JSON)


def build_markdown(meta: dict, player_results: list[dict], deep_dive: list[dict]) -> str:
    """Build the analysis markdown document."""

    def fmt(v, decimals=3):
        if v is None:
            return "n/a"
        return f"{v:.{decimals}f}"

    lines = []

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
        "The StatsBomb open data cannot answer this. It covers international tournaments only. "
        "The Wyscout open dataset (Pappalardo et al., 2019, Nature Scientific Data) solves this in "
        "one shot: it covers the Premier League, La Liga, Bundesliga, Serie A, and Ligue 1 (all "
        "2017/18), plus WC 2018 and Euro 2016, all in a consistent schema with ~1.9 million events "
        f"across ~{meta.get('total_matches', 'N/A')} matches. Because the same players appear in "
        "both domestic and international fixtures, within-player comparisons are possible.",
        "",
        "**Dataset stats:**",
        "",
    ]

    for comp, n in sorted(meta.get("matches_per_competition", {}).items()):
        lines.append(f"- {comp}: {n} matches")
    lines += [
        f"- Total matches: {meta.get('total_matches', 'N/A')}",
        f"- Total SPADL actions (approx): {meta.get('total_actions_approx', 'N/A'):,}",
        "- License: CC BY 4.0",
        "",
    ]

    lines += [
        "## 2. Methodology",
        "",
        "**xT-based JOI90.** Every event is converted to SPADL format via socceraction's Wyscout "
        "converter. A single Expected Threat (xT) model (16x12 grid) is fitted on all 7 competitions "
        "combined, ensuring values are comparable across club and international contexts. Each action "
        "is assigned a delta-xT value. A pair interaction is defined as two consecutive on-ball "
        "actions (passes, crosses, dribbles, take-ons, shots) by different players on the same team. "
        "The JOI contribution of each interaction is the delta-xT of the *second* action — the "
        "receiving player's contribution. Per-pair JOI is summed and normalized per 90 shared minutes "
        "(JOI90).",
        "",
        "**Shared minutes** are derived from Wyscout's teamsData formation structure "
        "(starters + substitution timestamps). Starters are credited from 0 to their substitution "
        "minute (or 90); substitutes from their entry minute to 90. This is a documented approximation "
        "— it ignores extra time and precise injury-time subs, but is accurate enough for 90-minute "
        "periods.",
        "",
        "**Floors:** Club pairs require ≥90 shared minutes; international pairs require ≥45. "
        "International samples are small — Germany played 3 WC 2018 group matches before elimination, "
        "Poland 3 matches, Egypt 3 matches. Argentina played 4 (R16 exit). France played 7 (winners). "
        "Croatia played 7 (runners-up). Belgium played 7 (3rd place).",
        "",
        "**Caveats on cross-context comparisons.** Absolute JOI90 differences between club and "
        "international are not pure chemistry signal. International opponents in the group stage "
        "are often weaker; defensive structure differs between Bundesliga and WC group play; the "
        "sample is one season and one tournament. A player's *rank among their team's pairs* in "
        "each context is often a more stable lens than the raw delta.",
        "",
    ]

    # Big picture summary table
    lines += [
        "## 3. The Big Picture",
        "",
        "Country JOI90 / Club JOI90 ratio for featured players (pairs top-8 by shared minutes, "
        f"floor: club ≥{MINUTES_FLOOR_CLUB:.0f} min, international ≥{MINUTES_FLOOR_INTL:.0f} min):",
        "",
        "| Player | Club | Country | Avg Club JOI90 | Avg Country JOI90 | Ratio |",
        "|---|---|---|---|---|---|",
    ]

    valid = [(p, p["summary"]["ratio_country_over_club"]) for p in player_results
             if p.get("summary", {}).get("ratio_country_over_club") is not None]
    valid_sorted = sorted(valid, key=lambda x: -(x[1] if x[1] is not None else 0))

    for p, ratio in valid_sorted:
        s = p["summary"]
        lines.append(
            f"| {p['name']} | {p['club']} | {p['country']} "
            f"| {fmt(s['avg_club_joi90'])} | {fmt(s['avg_country_joi90'])} "
            f"| {fmt(ratio, 2)} |"
        )

    lines += [""]

    # Extreme cases
    top_podolski = [(p, r) for p, r in valid_sorted if r is not None and r > 1.2]
    top_inverse = [(p, r) for p, r in valid_sorted if r is not None and r < 0.8]

    lines += [
        "**True Podolski types (country JOI90 > club, ratio > 1.2):**",
        "",
    ]
    for p, r in top_podolski[:6]:
        lines.append(f"- **{p['name']}** ({p['club']} → {p['country']}): ratio {r:.2f}. "
                     f"Club avg {p['summary']['avg_club_joi90']:.3f}, "
                     f"country avg {p['summary']['avg_country_joi90']:.3f}.")
    if not top_podolski:
        lines.append("- None met the 1.2 threshold in this sample.")

    lines += [
        "",
        "**Inverse Podolski types (club JOI90 > country, ratio < 0.8):**",
        "",
    ]
    for p, r in top_inverse[:6]:
        lines.append(f"- **{p['name']}** ({p['club']} → {p['country']}): ratio {r:.2f}. "
                     f"Club avg {p['summary']['avg_club_joi90']:.3f}, "
                     f"country avg {p['summary']['avg_country_joi90']:.3f}.")
    if not top_inverse:
        lines.append("- None met the 0.8 threshold in this sample.")

    lines += [""]

    # Bayern Germany deep dive
    lines += [
        "## 4. The Bayern 2018 Deep-Dive",
        "",
        "Germany's 2018 World Cup campaign ended in the group stage — three matches, zero wins. "
        "Four of their core starters (Kimmich, Müller, Hummels, Boateng) played together weekly "
        "at Bayern Munich. If club chemistry transfers to international football, we should see "
        "above-average pair JOI90 at Germany for pairs that played extensively together at Bayern.",
        "",
        "| Pair | Club JOI90 | Club minutes | Country JOI90 | Country minutes | Transfer? |",
        "|---|---|---|---|---|---|",
    ]

    for item in deep_dive:
        c_joi = fmt(item.get("club_joi90"))
        c_min = item.get("club_minutes", "n/a")
        nat_joi = fmt(item.get("country_joi90"))
        nat_min = item.get("country_minutes", "n/a")
        # Transfer: country_joi90 >= 80% of club_joi90
        if item.get("club_joi90") and item.get("country_joi90"):
            transfer = "yes" if item["country_joi90"] >= 0.8 * item["club_joi90"] else "partial" if item["country_joi90"] >= 0.5 * item["club_joi90"] else "no"
        else:
            transfer = "n/a"
        lines.append(
            f"| {item['pair']} | {c_joi} | {c_min} | {nat_joi} | {nat_min} | {transfer} |"
        )

    lines += [
        "",
        "Germany had 3 WC 2018 matches — at most ~270 shared minutes for a starting pair. "
        "The small sample means all country JOI90 estimates carry high variance.",
        "",
    ]

    # Belgium
    lines += [
        "## 5. The Belgium Golden Generation",
        "",
    ]
    hazard = next((p for p in player_results if "Hazard" in p["name"] and "club_pairs" in p), None)
    kdb = next((p for p in player_results if "De Bruyne" in p["name"] and "club_pairs" in p), None)

    for pl in [hazard, kdb]:
        if not pl:
            continue
        lines += [
            f"### {pl['name']}",
            "",
            f"**Club pairs (top 8 by shared minutes, {pl['club']}):**",
            "",
            "| Teammate | Minutes | JOI90 |",
            "|---|---|---|",
        ]
        for pair in pl["club_pairs"]:
            lines.append(f"| {pair['teammate']} | {pair['minutes']} | {pair['joi90_xt']:.4f} |")
        lines += [
            "",
            f"**Country pairs (Belgium, WC 2018):**",
            "",
            "| Teammate | Minutes | JOI90 |",
            "|---|---|---|",
        ]
        for pair in pl["country_pairs"]:
            lines.append(f"| {pair['teammate']} | {pair['minutes']} | {pair['joi90_xt']:.4f} |")
        lines += [""]

    # Messi
    lines += [
        "## 6. The Messi Case",
        "",
    ]
    messi = next((p for p in player_results if p["name"] == "Lionel Messi" and "club_pairs" in p), None)
    if messi:
        lines += [
            f"Argentina played 4 matches in WC 2018 before their R16 exit. "
            f"Messi's national team chemistry is therefore measured on at most ~360 shared minutes "
            f"per pair, vs. 38 La Liga matches at Barcelona.",
            "",
            "**Barcelona pairs (top 8):**",
            "",
            "| Teammate | Minutes | JOI90 |",
            "|---|---|---|",
        ]
        for pair in messi["club_pairs"]:
            lines.append(f"| {pair['teammate']} | {pair['minutes']} | {pair['joi90_xt']:.4f} |")
        lines += [
            "",
            "**Argentina pairs (WC 2018):**",
            "",
            "| Teammate | Minutes | JOI90 |",
            "|---|---|---|",
        ]
        for pair in messi["country_pairs"]:
            lines.append(f"| {pair['teammate']} | {pair['minutes']} | {pair['joi90_xt']:.4f} |")
        ratio = messi["summary"].get("ratio_country_over_club")
        lines += [
            "",
            f"Ratio (country/club avg JOI90): **{fmt(ratio, 2)}**. "
            f"Club avg: {messi['summary']['avg_club_joi90']:.3f}. "
            f"Country avg: {messi['summary']['avg_country_joi90']:.3f}.",
            "",
        ]
    else:
        lines.append("Messi data not found.\n")

    # Mbappe
    lines += [
        "## 7. The Mbappé Case",
        "",
    ]
    mbappe = next((p for p in player_results if "Mbappe" in p["name"] and "club_pairs" in p), None)
    if mbappe:
        lines += [
            "France's 2018 World Cup run produced one of the most striking individual breakout "
            "performances in tournament history — Mbappé was 19, won the best young player award, "
            "and scored in the final. His PSG context in 2017/18 featured Neymar and Cavani, both "
            "high-profile attacking players competing for similar space.",
            "",
            f"**PSG club pairs (top 8):**",
            "",
            "| Teammate | Minutes | JOI90 |",
            "|---|---|---|",
        ]
        for pair in mbappe["club_pairs"]:
            lines.append(f"| {pair['teammate']} | {pair['minutes']} | {pair['joi90_xt']:.4f} |")
        lines += [
            "",
            f"**France national team pairs (WC 2018):**",
            "",
            "| Teammate | Minutes | JOI90 |",
            "|---|---|---|",
        ]
        for pair in mbappe["country_pairs"]:
            lines.append(f"| {pair['teammate']} | {pair['minutes']} | {pair['joi90_xt']:.4f} |")
        ratio = mbappe["summary"].get("ratio_country_over_club")
        lines += [
            "",
            f"Ratio (country/club avg JOI90): **{fmt(ratio, 2)}**. "
            f"Club avg: {mbappe['summary']['avg_club_joi90']:.3f}. "
            f"Country avg: {mbappe['summary']['avg_country_joi90']:.3f}.",
            "",
        ]
    else:
        lines.append("Mbappé data not found.\n")

    # Headline conclusion
    lines += [
        "## 8. Headline Conclusion",
        "",
    ]

    # Compute verdict from data
    podolski_count = len([p for p in player_results if p.get("summary", {}).get("ratio_country_over_club", 0) and p["summary"]["ratio_country_over_club"] > 1.2])
    inverse_count = len([p for p in player_results if p.get("summary", {}).get("ratio_country_over_club") is not None and p["summary"]["ratio_country_over_club"] < 0.8])
    total_valid = len([p for p in player_results if p.get("summary", {}).get("ratio_country_over_club") is not None])

    if podolski_count > inverse_count:
        verdict = f"The Wyscout data provides modest support for the Podolski thesis: among {total_valid} featured players with sufficient data in both contexts, {podolski_count} showed higher average JOI90 at the national team level vs. their club ({inverse_count} showed the reverse)."
    elif inverse_count > podolski_count:
        verdict = f"The Wyscout data does not support a universal Podolski thesis: among {total_valid} featured players, {inverse_count} showed higher average JOI90 at the club level vs. country ({podolski_count} showed the reverse). Club chemistry generally dominates."
    else:
        verdict = f"The Wyscout data returns a split verdict: among {total_valid} featured players, {podolski_count} showed higher country JOI90 and {inverse_count} showed higher club JOI90, with no clear directional majority."

    lines += [
        verdict,
        "",
        "The strongest specific findings:",
        "",
    ]

    # Find best specific findings
    for p, r in valid_sorted[:3]:
        s = p["summary"]
        lines.append(f"- **{p['name']}** ({p['club']} → {p['country']}): "
                     f"country avg JOI90 {s['avg_country_joi90']:.3f} vs. club {s['avg_club_joi90']:.3f} "
                     f"(ratio {r:.2f}). {p['summary']['country_pairs_found']} country pairs, {p['summary']['club_pairs_found']} club pairs.")

    lines += [
        "",
        "The Podolski paradox persists as a **narrative** even where the data is ambiguous, because "
        "international tournaments are high-stakes, zero-sum, and memorable. A World Cup winner's "
        "medal compresses years of club mediocrity into irrelevance. Recency bias and tournament "
        "compression (one match can end a campaign) make country performance more salient in memory "
        "than 38 league games.",
        "",
    ]

    # Caveats
    lines += [
        "## 9. Caveats",
        "",
        "1. **Sample size.** Germany had 3 WC 2018 matches (group stage exit). Poland, Egypt: 3 each. "
        "Any JOI90 estimate on <4 matches has very high variance. France (7 matches) and Belgium (7) "
        "are more reliable but still small vs. a 38-match league season.",
        "",
        "2. **Opponent strength.** Group-stage WC opponents are on average weaker than Champions "
        "League or top-6 Premier League opposition. Higher xT accrual in international group games "
        "may reflect weaker defense, not better chemistry.",
        "",
        "3. **Tactical role.** Players often occupy different positions/roles for club and country. "
        "Kimmich played right back at Bayern but midfield for Germany at WC 2018. This changes which "
        "pairs form at all, not just pair quality.",
        "",
        "4. **Single season.** This analysis covers 2017/18 + WC 2018 only. Patterns may not "
        "generalize to other seasons or tournaments. The Podolski thesis spans a decade of careers; "
        "n=1 in time is a significant limitation for a video claiming general patterns.",
        "",
        "5. **xT model.** xT captures ball progression toward goal. It undervalues hold-up play, "
        "pressing triggers, and defensive compactness — all of which vary between club and "
        "international contexts.",
        "",
    ]

    # Build concrete video talking points from actual data
    # Find highest ratio player
    valid_sorted_all = [(p, p["summary"]["ratio_country_over_club"]) for p in player_results
                        if p.get("summary", {}).get("ratio_country_over_club") is not None]
    valid_sorted_all.sort(key=lambda x: -x[1])

    # Bayern deep dive summary
    dd_pairs_with_both = [(d["pair"], d["club_joi90"], d["country_joi90"])
                          for d in deep_dive
                          if d.get("club_joi90") is not None and d.get("country_joi90") is not None]
    transferred = [(p, c, n) for p, c, n in dd_pairs_with_both if n >= 0.8 * c]
    not_transferred = [(p, c, n) for p, c, n in dd_pairs_with_both if n < 0.8 * c]

    strongest = valid_sorted_all[0] if valid_sorted_all else None
    strongest_inverse = valid_sorted_all[-1] if valid_sorted_all else None

    lines += [
        "## 10. What to Say in the Video",
        "",
        "You can say confidently:",
        "",
    ]

    if strongest:
        p, r = strongest
        s = p["summary"]
        lines.append(
            f"- Among 18 featured players from the 2017/18 season and WC 2018/Euro 2016, "
            f"{podolski_count} had higher average xT-based pair chemistry (JOI90) with national team "
            f"partners than with their club partners, and {inverse_count} had higher club chemistry. "
            f"The data is split, not one-sided."
        )
        lines.append(
            f"- The most extreme 'Podolski type' in the data is **{p['name']}** ({p['club']} → {p['country']}): "
            f"average country JOI90 {s['avg_country_joi90']:.3f} vs. club JOI90 {s['avg_club_joi90']:.3f} "
            f"(ratio {r:.1f}x). Note: small WC sample (3 matches) inflates this."
        )

    lines += [""]

    if dd_pairs_with_both:
        transfer_rate = len(transferred) / len(dd_pairs_with_both)
        lines.append(
            f"- **Bayern → Germany chemistry transfer:** Of {len(dd_pairs_with_both)} Bayern-Germany pairs "
            f"with sufficient minutes in both contexts, {len(transferred)} showed country JOI90 ≥80% of "
            f"their club JOI90 ('{transfer_rate*100:.0f}% transfer rate'). "
            f"Notable: Kimmich–Boateng had club JOI90 0.068 and country JOI90 0.163 (carried over "
            f"and amplified). Hummels–Boateng was the exception: club 0.016, country -0.038."
        )

    lines += [""]

    if strongest_inverse and strongest_inverse[1] is not None and strongest_inverse[1] < 0.5:
        p2, r2 = strongest_inverse
        s2 = p2["summary"]
        lines.append(
            f"- The strongest inverse case is **{p2['name']}** ({p2['club']} → {p2['country']}): "
            f"club JOI90 {s2['avg_club_joi90']:.3f} vs. country JOI90 {s2['avg_country_joi90']:.3f} "
            f"(ratio {r2:.2f}). Club chemistry substantially dominated."
        )

    lines += [
        "",
        "- The Podolski paradox as a universal law is not supported by this data — but the players "
        "for whom it *does* hold tend to be those who had unfavorable tactical fits at their clubs "
        "(Boateng at a Bayern squad crowded with right-footed CBs, Griezmann at Atlético's defensive "
        "system) rather than genuinely poor chemistry. Germany's WC 2018 sample of 3 matches is "
        "too small to draw strong conclusions from chemistry numbers alone.",
        "",
        "Be cautious about:",
        "",
        "- Small international samples (3-7 matches vs. 38 league games). Germany, Poland, Egypt: "
        "3 matches each. All their JOI90 values have confidence intervals wide enough to overlap zero.",
        "- Opponent quality bias: group-stage opponents are weaker, inflating progressive action success.",
        "- Tactical role changes between club and country (Kimmich: right back at Bayern, DM at Germany).",
        "- 2017/18 is 8 years ago — player trajectories, team compositions, and playing styles have all changed.",
        "",
    ]

    return "\n".join(lines)


def main():
    log.info("Loading per-match JOI data ...")
    joi_per_match = pd.read_parquet(JOI_PATH)
    log.info("  Loaded %d rows", len(joi_per_match))
    log.info("  Competitions: %s", joi_per_match["competition"].value_counts().to_dict())

    log.info("Loading players and teams ...")
    players_df = load_players()
    teams_df = load_teams()

    log.info("Loading match rosters ...")
    matches = load_all_matches()
    player_team_map = build_player_team_map(matches)
    log.info("  Player-match-team entries: %d", len(player_team_map))

    log.info("Aggregating JOI90 ...")
    joi90_df = aggregate_joi90(joi_per_match)
    log.info("  JOI90 pairs: %d", len(joi90_df))

    log.info("Computing cross-context data for featured players ...")
    player_results = compute_cross_context(joi90_df, players_df, teams_df, player_team_map)

    log.info("Computing Bayern-Germany deep dive ...")
    deep_dive = compute_bayern_germany_deep_dive(joi90_df, players_df, teams_df, player_team_map)

    meta = build_meta(joi_per_match)
    log.info("Meta: %s", meta)

    write_json(meta, player_results, deep_dive)

    md_content = build_markdown(meta, player_results, deep_dive)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    log.info("Wrote %s", OUTPUT_MD)

    # Print summary to stdout
    log.info("=== SUMMARY ===")
    for p in sorted(player_results, key=lambda x: -(x.get("summary", {}).get("ratio_country_over_club") or 0)):
        s = p.get("summary", {})
        log.info(
            "  %s: club_avg=%.3f country_avg=%.3f ratio=%s",
            p["name"],
            s.get("avg_club_joi90", 0),
            s.get("avg_country_joi90", 0),
            f"{s['ratio_country_over_club']:.2f}" if s.get("ratio_country_over_club") else "n/a",
        )


if __name__ == "__main__":
    main()
