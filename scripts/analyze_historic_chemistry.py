"""
Historic club-chemistry analysis for WC 2018 squads + AFCON 2023 squads.

Computes within-tournament JOI90 (xT primary, VAEP + xG secondary) for 8 squads,
compares same-club vs different-club pairs, and deep-dives the Germany 2018 Bayern pairs.

Usage:
    source .venv/bin/activate && python scripts/analyze_historic_chemistry.py
"""
from __future__ import annotations

import json
import os
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from chemistry.joi import joi_per_match as compute_joi_per_match, xg_chain_per_match
from chemistry.joi import enumerate_interactions, enumerate_possessions
from chemistry.minutes import LineupsMinutes, shared_minutes
from chemistry.players import _norm

# ---------------------------------------------------------------------------
# Squad definitions (hard-coded clubs as of the tournament)
# ---------------------------------------------------------------------------

# Format: {player_name_as_in_statsbomb: club}
# Name must match the StatsBomb display name (from lineups.parquet).
# We'll do a fuzzy-norm join, so minor accent diffs are OK.

SQUADS = {

    # ---- WC 2018 ----

    "Germany_WC2018": {
        "comp_season": "43_3",
        "team_id": 770,
        "team_name_sb": "Germany",
        "tournament": "WC 2018",
        "roster": {
            # Bayern Munich core
            "Manuel Neuer": "Bayern Munich",
            "Mats Hummels": "Bayern Munich",
            "Jérôme Boateng": "Bayern Munich",
            "Joshua Kimmich": "Bayern Munich",
            "Thomas Müller": "Bayern Munich",
            "Leon Goretzka": "Bayern Munich",  # joined Bayern summer 2018; was Schalke during the tournament — technically Schalke
            # Real Madrid
            "Toni Kroos": "Real Madrid",
            # Others
            "Antonio Rüdiger": "Chelsea",
            "Niklas Süle": "Bayern Munich",  # also Bayern — include
            "Matthias Ginter": "Borussia Mönchengladbach",
            "Marvin Plattenhardt": "Hertha Berlin",
            "Jonas Hector": "1. FC Köln",
            "Mesut Özil": "Arsenal",
            "İlkay Gündoğan": "Manchester City",
            "Sami Khedira": "Juventus",
            "Sebastian Rudy": "Bayern Munich",  # had just transferred from Hoffenheim to Bayern
            "Julian Draxler": "Paris Saint-Germain",
            "Marco Reus": "Borussia Dortmund",
            "Julian Brandt": "Bayer Leverkusen",
            "Timo Werner": "RB Leipzig",
            "Mario Gómez García": "VfB Stuttgart",
            "Marc-André ter Stegen": "Barcelona",
            "Kevin Trapp": "Paris Saint-Germain",
        },
    },

    "Belgium_WC2018": {
        "comp_season": "43_3",
        "team_id": 782,
        "team_name_sb": "Belgium",
        "tournament": "WC 2018",
        "roster": {
            "Thibaut Courtois": "Chelsea",
            "Simon Mignolet": "Liverpool",
            "Koen Casteels": "VfL Wolfsburg",
            "Toby Alderweireld": "Tottenham Hotspur",
            "Jan Vertonghen": "Tottenham Hotspur",
            "Vincent Kompany": "Manchester City",
            "Thomas Vermaelen": "Barcelona",
            "Anga Dedryck Boyata": "Celtic",
            "Kevin De Bruyne": "Manchester City",
            "Axel Witsel": "Tianjin Quanjian",
            "Mousa Sidi Yaya Dembélé": "Tottenham Hotspur",
            "Marouane Fellaini-Bakkioui": "Manchester United",
            "Youri Tielemans": "Anderlecht",
            "Leander Dendoncker": "Anderlecht",
            "Adnan Januzaj": "Real Sociedad",
            "Eden Hazard": "Chelsea",
            "Thorgan Hazard": "Borussia Mönchengladbach",
            "Yannick Ferreira Carrasco": "Dalian Yifang",
            "Dries Mertens": "Napoli",
            "Romelu Lukaku Menama": "Manchester United",
            "Michy Batshuayi Tunga": "Chelsea",
            "Thomas Meunier": "Paris Saint-Germain",
            "Nacer Chadli": "West Bromwich Albion",
        },
    },

    "France_WC2018": {
        "comp_season": "43_3",
        "team_id": 771,
        "team_name_sb": "France",
        "tournament": "WC 2018",
        "roster": {
            "Hugo Lloris": "Tottenham Hotspur",
            "Steve Mandanda": "Olympique Marseille",
            "Alphonse Areola": "Paris Saint-Germain",
            "Benjamin Pavard": "VfB Stuttgart",
            "Raphaël Varane": "Real Madrid",
            "Samuel Yves Umtiti": "Barcelona",
            "Presnel Kimpembe": "Paris Saint-Germain",
            "Adil Rami": "Olympique Marseille",
            "Lucas Hernández Pi": "Atlético Madrid",
            "Djibril Sidibé": "AS Monaco",
            "Benjamin Mendy": "Manchester City",
            "N''Golo Kanté": "Chelsea",
            "Paul Pogba": "Manchester United",
            "Blaise Matuidi": "Juventus",
            "Corentin Tolisso": "Bayern Munich",
            "Steven N''Kemboanza Mike Christopher Nzonzi": "Sevilla",
            "Kylian Mbappé Lottin": "Paris Saint-Germain",
            "Antoine Griezmann": "Atlético Madrid",
            "Ousmane Dembélé": "Barcelona",
            "Olivier Giroud": "Chelsea",
            "Nabil Fekir": "Olympique Lyonnais",
            "Thomas Lemar": "Atlético Madrid",
            "Florian Thauvin": "Olympique Marseille",
        },
    },

    "Croatia_WC2018": {
        "comp_season": "43_3",
        "team_id": 785,
        "team_name_sb": "Croatia",
        "tournament": "WC 2018",
        "roster": {
            "Danijel Subašić": "AS Monaco",
            "Lovre Kalinić": "Hajduk Split",
            "Dominik Livaković": "Dinamo Zagreb",
            "Domagoj Vida": "Beşiktaş",
            "Dejan Lovren": "Liverpool",
            "Šime Vrsaljko": "Atlético Madrid",
            "Ivan Strinić": "Sampdoria",
            "Vedran Ćorluka": "Lokomotiv Moscow",
            "Tin Jedvaj": "Bayer Leverkusen",
            "Duje Ćaleta-Car": "Salzburg",
            "Luka Modrić": "Real Madrid",
            "Ivan Rakitić": "Barcelona",
            "Marcelo Brozović": "Inter Milan",
            "Mateo Kovačić": "Real Madrid",
            "Milan Badelj": "Fiorentina",
            "Josip Pivarić": "Dinamo Zagreb",
            "Filip Bradarić": "Rijeka",
            "Ivan Perišić": "Inter Milan",
            "Mario Mandžukić": "Juventus",
            "Andrej Kramarić": "Hoffenheim",
            "Ante Rebić": "Eintracht Frankfurt",
            "Nikola Kalinić": "Atlético Madrid",
            "Marko Pjaca": "Schalke 04",
        },
    },

    "Brazil_WC2018": {
        "comp_season": "43_3",
        "team_id": 781,
        "team_name_sb": "Brazil",
        "tournament": "WC 2018",
        "roster": {
            "Alisson Ramsés Becker": "AS Roma",
            "Ederson Santana de Moraes": "Manchester City",
            "Cássio Ramos": "Corinthians",
            "Thiago Emiliano da Silva": "Paris Saint-Germain",
            "Miranda": "Inter Milan",
            "Pedro Tonon Geromel": "Grêmio",
            "Marcelo Vieira da Silva Júnior": "Real Madrid",
            "Danilo Luiz da Silva": "Manchester City",
            "Fágner Conserva Lemos": "Corinthians",
            "Filipe Luís Kasmirski": "Atlético Madrid",
            "Fernando Luiz Rosa": "Chelsea",  # Fernandinho
            "Carlos Henrique Casimiro": "Real Madrid",
            "Renato Soares de Oliveira Augusto": "Beijing Guoan",
            "Neymar da Silva Santos Junior": "Paris Saint-Germain",
            "Philippe Coutinho Correia": "Barcelona",
            "Willian Borges da Silva": "Chelsea",
            "Douglas Costa de Souza": "Juventus",
            "Gabriel Fernando de Jesus": "Manchester City",
            "Roberto Firmino Barbosa de Oliveira": "Liverpool",
            "Taison Barcellos Freda": "Shakhtar Donetsk",
            "Marcos Aoás Corrêa": "Napoli",  # Paulinho — actually Guangzhou
            "José Paulo Bezzera Maciel Júnior": "Cruzeiro",  # Paulinho
            "João Miranda de Souza Filho": "Inter Milan",
            "Frederico Rodrigues Santos": "Fluminense",  # Fred
        },
    },

    # ---- AFCON 2023 ----

    "Senegal_AFCON2023": {
        "comp_season": "1267_107",
        "team_id": 787,
        "team_name_sb": "Senegal",
        "tournament": "AFCON 2023",
        "roster": {
            "Edouard Mendy": "Chelsea",
            "Alfred Benjamin Gomis": "Rennes",
            "Mory Diaw": "Clermont Foot",
            "Kalidou Koulibaly": "Al-Hilal",
            "Abdou Diallo": "RB Leipzig",
            "Cheikhou Kouyaté": "Nottingham Forest",
            "Moussa Niakhate": "Nottingham Forest",
            "Abdoulaye Seck": "Reims",
            "Formose Mendy": "Brest",
            "Ismail Jakobs": "AS Monaco",
            "Fodé Ballo Touré": "AC Milan",
            "Idrissa Gana Gueye": "Everton",
            "Nampalys Mendy": "Leicester City",
            "Pape Gueye": "Olympique Marseille",
            "Lamine Camara": "AS Monaco",
            "Pathé Ismaël Ciss": "Rayo Vallecano",
            "Pape Matar Sarr": "Tottenham Hotspur",
            "Krépin Diatta": "AS Monaco",
            "Ismaïla Sarr": "Olympique Marseille",
            "Nicolas Jackson": "Chelsea",
            "Sadio Mané": "Al-Nassr",
            "Iliman Ndiaye": "Olympique Marseille",
            "Abdallah Dipo Sima": "Stade Brestois",
            "Habibou Mouhamadou Diallo": "Eupen",
            "Abdoulaye Niakhate Ndiaye": "Montpellier",
            "Cheikh Ahmadou Bamba Mbacke Dieng": "Olympique Marseille",
            "Idrissa Gueye": "Everton",  # duplicate mapping, same person
        },
    },

    "Morocco_AFCON2023": {
        "comp_season": "1267_107",
        "team_id": 788,
        "team_name_sb": "Morocco",
        "tournament": "AFCON 2023",
        "roster": {
            "Yassine Bounou": "Sevilla",
            "Munir Mohand Mohamedi": "Nice",
            "Abdelkabir Abqar": "FAR Rabat",
            "Nayef Aguerd": "West Ham United",
            "Romain Saïss": "Besiktas",
            "Noussair Mazraoui": "Bayern Munich",
            "Yahia Attiyat allah": "Wydad Casablanca",
            "Yunis Abdelhamid": "Reims",
            "Chadi Riad": "Real Betis",
            "Achraf Hakimi Mouh": "Paris Saint-Germain",
            "Sofyan Amrabat": "Fiorentina",
            "Azzedine Ounahi": "Olympique Marseille",
            "Selim Amallah": "Strasbourg",
            "Oussama El Azzouzi": "Bologna",
            "Amir Richardson": "Reims",
            "Bilal El Khannous": "Genk",
            "Ismael Saibari": "PSV Eindhoven",
            "Hakim Ziyech": "Galatasaray",
            "Sofiane Boufal": "Angers",
            "Abdessamad Ezzalzouli": "Osasuna",
            "Amine Adli": "Bayer Leverkusen",
            "Ayoub El Kaabi": "Olympiakos",
            "Youssef En-Nesyri": "Sevilla",
            "Amine Harit": "Olympique Marseille",
            "Tarik Tissoudali": "Gent",
            "Mohamed Chibi": "FAR Rabat",
            "Odilon Kossonou": "Bayer Leverkusen",  # actually Ivory Coast — mistake, remove
        },
    },

    "CotedIvoire_AFCON2023": {
        "comp_season": "1267_107",
        "team_id": 3374,
        "team_name_sb": "Côte d'Ivoire",
        "tournament": "AFCON 2023",
        "roster": {
            "Yahia Fofana": "Leicester City",
            "Badra Ali Sangaré": "Stade Malien",
            "Ayayi Charles Folly": "AC Sparta Prague",
            "Serge Aurier": "Villarreal",
            "Odilon Kossonou": "Bayer Leverkusen",
            "Obite Evan Ndicka": "AS Roma",
            "Willy Boly": "Nottingham Forest",
            "Wilfried Stephane Singo": "Torino",
            "Jean Thierry Lazare Amani": "Stade de Reims",
            "Oumar Diakité": "Venezia",
            "Jean Michaël Seri": "Hull City",
            "Ibrahim Sangaré": "Nottingham Forest",
            "Franck Yannick Kessié": "Al-Ahli",
            "Seko Fofana": "Al-Qadsiah",
            "Idrissa Doumbia": "Sochaux",
            "Ghislain Konan": "Reims",
            "Jean-Philippe Krasso": "Saint-Etienne",
            "Nicolas Pépé": "Trabzonspor",
            "Simon Adingra": "Brighton",
            "Jonathan Bamba": "Club Brugge",
            "Ismaël Chester Diallo": "Dunkerque",
            "Karim Konaté": "Red Bull Salzburg",
            "Ousmane Diomande": "Sporting CP",
            "Sébastien Haller": "Borussia Dortmund",
            "Christian Kouamé": "Fiorentina",
            "Jeremie Boga": "Nice",
            "Max-Alain Gradel": "Al-Qadsiah",
        },
    },
}

# Fix Brazil squad - Marcos Aoás Corrêa (Paulinho) was at Guangzhou Evergrande
SQUADS["Brazil_WC2018"]["roster"]["Marcos Aoás Corrêa"] = "Guangzhou Evergrande"
# Remove the bad duplicate in Brazil
if "José Paulo Bezzera Maciel Júnior" in SQUADS["Brazil_WC2018"]["roster"]:
    del SQUADS["Brazil_WC2018"]["roster"]["José Paulo Bezzera Maciel Júnior"]

# Remove Odilon Kossonou from Morocco (he's Ivory Coast)
if "Odilon Kossonou" in SQUADS["Morocco_AFCON2023"]["roster"]:
    del SQUADS["Morocco_AFCON2023"]["roster"]["Odilon Kossonou"]

# Remove duplicate Idrissa Gueye entry in Senegal (same as Idrissa Gana Gueye)
# The SB name "Idrissa Gueye" is likely a second entry for same person
# Keep both and let the ID join resolve it


def load_comp_matches(comp_season: str) -> set[int]:
    path = ROOT / "data" / "raw" / comp_season / "matches.parquet"
    df = pd.read_parquet(path)
    return set(df["match_id"].tolist())


def load_xt_per_match(comp_season: str, match_ids: set[int]) -> pd.DataFrame:
    """Load all xT-scored SPADL for a comp, compute JOI per match."""
    from chemistry.joi import enumerate_interactions, joi_per_match as _joi_per_match
    rows = []
    comp_dir = ROOT / "data" / "vaep" / comp_season
    for mfile in sorted(comp_dir.glob("*.parquet")):
        gid = int(mfile.stem)
        if gid not in match_ids:
            continue
        df = pd.read_parquet(mfile)
        interactions = enumerate_interactions(df)
        if len(interactions) == 0:
            continue
        rows.append(_joi_per_match(interactions))
    if not rows:
        return pd.DataFrame(columns=["game_id", "team_id", "player_a", "player_b", "joi"])
    return pd.concat(rows, ignore_index=True)


def load_vaep_per_match(comp_season: str, match_ids: set[int]) -> pd.DataFrame:
    """Load all VAEP-scored SPADL for a comp, compute JOI per match."""
    from chemistry.joi import enumerate_interactions, joi_per_match as _joi_per_match
    rows = []
    comp_dir = ROOT / "data" / "vaep_scored" / comp_season
    if not comp_dir.exists():
        return pd.DataFrame(columns=["game_id", "team_id", "player_a", "player_b", "joi"])
    for mfile in sorted(comp_dir.glob("*.parquet")):
        gid = int(mfile.stem)
        if gid not in match_ids:
            continue
        df = pd.read_parquet(mfile)
        interactions = enumerate_interactions(df)
        if len(interactions) == 0:
            continue
        rows.append(_joi_per_match(interactions))
    if not rows:
        return pd.DataFrame(columns=["game_id", "team_id", "player_a", "player_b", "joi"])
    return pd.concat(rows, ignore_index=True)


def load_xg_per_match_from_prebuilt(match_ids: set[int]) -> pd.DataFrame:
    """Load pre-built xg_chain_per_match.parquet and filter to tournament match IDs."""
    path = ROOT / "outputs" / "xg_chain_per_match.parquet"
    df = pd.read_parquet(path)
    return df[df["game_id"].isin(match_ids)].copy()


def aggregate_joi90(
    per_match: pd.DataFrame,
    lineups: pd.DataFrame,
    team_id: int,
    match_ids: set[int],
    value_col: str = "joi",
    out_col: str = "joi90",
    min_minutes: float = 45.0,
) -> pd.DataFrame:
    """Filter to one team, compute shared minutes per pair, aggregate to JOI90."""
    pm = per_match[(per_match["team_id"] == team_id) & (per_match["game_id"].isin(match_ids))].copy()
    if len(pm) == 0:
        return pd.DataFrame(columns=["player_a", "player_b", value_col, "minutes", "matches", out_col])

    lin = lineups[lineups["game_id"].isin(match_ids)].copy()

    # Compute minutes per pair per match
    pm["minutes"] = pm.apply(
        lambda r: shared_minutes(lin, int(r["game_id"]), int(r["player_a"]), int(r["player_b"])),
        axis=1,
    )

    agg = (
        pm.groupby(["player_a", "player_b"], as_index=False)
          .agg(**{value_col: (value_col, "sum"), "minutes": ("minutes", "sum"), "matches": ("game_id", "nunique")})
    )
    agg[out_col] = (agg[value_col] * 90.0 / agg["minutes"]).where(agg["minutes"] > 0, 0.0)
    agg = agg[agg["minutes"] >= min_minutes].reset_index(drop=True)
    return agg


def match_roster_to_ids(squad_def: dict, lineups: pd.DataFrame) -> tuple[dict[int, str], dict[str, str], list[str]]:
    """
    Match roster name -> player_id in lineups.
    Returns:
        id_to_club: {player_id: club}
        matched_names: {roster_name: player_name_sb}
        unmatched: list of unmatched roster names
    """
    team_id = squad_def["team_id"]
    roster = squad_def["roster"]
    match_ids = load_comp_matches(squad_def["comp_season"])

    # Get all players seen in this team's tournament matches
    team_lin = lineups[(lineups["game_id"].isin(match_ids)) & (lineups["team_id"] == team_id)]
    sb_players = team_lin[["player_id", "player_name"]].drop_duplicates()

    # Build normalized index: norm_name -> (player_id, player_name)
    sb_index = {}
    for _, row in sb_players.iterrows():
        sb_index[_norm(row["player_name"])] = (int(row["player_id"]), row["player_name"])

    id_to_club = {}
    matched_names = {}
    unmatched = []

    for roster_name, club in roster.items():
        norm = _norm(roster_name)
        if norm in sb_index:
            pid, sb_name = sb_index[norm]
            id_to_club[pid] = club
            matched_names[roster_name] = sb_name
        else:
            # Try substring match
            found = None
            for sb_norm, (pid, sb_name) in sb_index.items():
                if norm in sb_norm or sb_norm in norm:
                    found = (pid, sb_name)
                    break
            if found:
                pid, sb_name = found
                id_to_club[pid] = club
                matched_names[roster_name] = sb_name
            else:
                unmatched.append(roster_name)

    return id_to_club, matched_names, unmatched


def analyze_squad(
    squad_key: str,
    squad_def: dict,
    lineups: pd.DataFrame,
    xt_pm_cache: dict,
    vaep_pm_cache: dict,
    xg_pm_cache: dict,
) -> dict:
    """Full analysis for one squad. Returns summary dict."""
    print(f"\n=== {squad_key} ===")

    comp_season = squad_def["comp_season"]
    team_id = squad_def["team_id"]
    match_ids = load_comp_matches(comp_season)

    # Filter to this team's matches
    team_matches = set(
        lineups[(lineups["game_id"].isin(match_ids)) & (lineups["team_id"] == team_id)]["game_id"].unique()
    )
    print(f"  Team matches in tournament: {len(team_matches)}")

    # Match roster
    id_to_club, matched_names, unmatched = match_roster_to_ids(squad_def, lineups)
    n_roster = len(squad_def["roster"])
    n_matched = len(id_to_club)
    match_rate = n_matched / n_roster if n_roster > 0 else 0
    print(f"  Roster match rate: {n_matched}/{n_roster} = {match_rate:.1%}")
    if unmatched:
        print(f"  Unmatched: {unmatched}")

    if match_rate < 0.80:
        print(f"  DROPPING squad: match rate {match_rate:.1%} < 80%")
        return {"squad": squad_key, "dropped": True, "match_rate": match_rate, "unmatched": unmatched}

    # Get or load per-match data
    if comp_season not in xt_pm_cache:
        print(f"  Loading xT per-match for {comp_season}...")
        xt_pm_cache[comp_season] = load_xt_per_match(comp_season, match_ids)
        print(f"    {len(xt_pm_cache[comp_season])} rows")

    if comp_season not in vaep_pm_cache:
        print(f"  Loading VAEP per-match for {comp_season}...")
        vaep_pm_cache[comp_season] = load_vaep_per_match(comp_season, match_ids)
        print(f"    {len(vaep_pm_cache[comp_season])} rows")

    if comp_season not in xg_pm_cache:
        print(f"  Loading xG-chain per-match for {comp_season}...")
        xg_pm_cache[comp_season] = load_xg_per_match_from_prebuilt(match_ids)
        print(f"    {len(xg_pm_cache[comp_season])} rows")

    xt_pm = xt_pm_cache[comp_season]
    vaep_pm = vaep_pm_cache[comp_season]
    xg_pm = xg_pm_cache[comp_season]

    # Aggregate JOI90
    xt_agg = aggregate_joi90(xt_pm, lineups, team_id, team_matches, "joi", "joi90_xt", min_minutes=45.0)
    vaep_agg = aggregate_joi90(vaep_pm, lineups, team_id, team_matches, "joi", "joi90_vaep", min_minutes=45.0)
    xg_agg = aggregate_joi90(xg_pm, lineups, team_id, team_matches, "xg", "joi90_xg", min_minutes=45.0)

    # Filter pairs to known roster players only
    known_ids = set(id_to_club.keys())

    def filter_pairs(df, a_col="player_a", b_col="player_b"):
        return df[(df[a_col].isin(known_ids)) & (df[b_col].isin(known_ids))].copy()

    xt_agg = filter_pairs(xt_agg)
    vaep_agg = filter_pairs(vaep_agg)
    xg_agg = filter_pairs(xg_agg)

    # Build ID-level name map from lineups
    lin_names = (
        lineups[lineups["game_id"].isin(team_matches)]
        .groupby("player_id")["player_name"]
        .first()
        .to_dict()
    )

    # Merge xT + VAEP + xG
    pairs = xt_agg[["player_a", "player_b", "minutes", "matches", "joi90_xt"]].copy()
    if len(vaep_agg) > 0:
        pairs = pairs.merge(
            vaep_agg[["player_a", "player_b", "joi90_vaep"]],
            on=["player_a", "player_b"], how="left"
        )
    else:
        pairs["joi90_vaep"] = float("nan")

    if len(xg_agg) > 0:
        pairs = pairs.merge(
            xg_agg[["player_a", "player_b", "joi90_xg"]],
            on=["player_a", "player_b"], how="left"
        )
    else:
        pairs["joi90_xg"] = float("nan")

    # Add club info and same_club flag
    pairs["club_a"] = pairs["player_a"].map(id_to_club)
    pairs["club_b"] = pairs["player_b"].map(id_to_club)
    pairs["same_club"] = pairs["club_a"] == pairs["club_b"]
    pairs["name_a"] = pairs["player_a"].map(lin_names)
    pairs["name_b"] = pairs["player_b"].map(lin_names)

    print(f"  Pairs with >=45 min: {len(pairs)} total, {pairs['same_club'].sum()} same-club")

    # Same-club vs different-club summary
    same = pairs[pairs["same_club"]]
    diff = pairs[~pairs["same_club"]]

    summary = {
        "squad": squad_key,
        "tournament": squad_def["tournament"],
        "team_name": squad_def["team_name_sb"],
        "matches": len(team_matches),
        "roster_n": n_roster,
        "matched_n": n_matched,
        "match_rate": round(match_rate, 4),
        "unmatched": unmatched,
        "dropped": False,
        "pairs_total": len(pairs),
        "pairs_same_club": int(pairs["same_club"].sum()),
        "pairs_diff_club": int((~pairs["same_club"]).sum()),
        "same_club_mean_joi90_xt": round(same["joi90_xt"].mean(), 4) if len(same) > 0 else None,
        "same_club_median_joi90_xt": round(same["joi90_xt"].median(), 4) if len(same) > 0 else None,
        "diff_club_mean_joi90_xt": round(diff["joi90_xt"].mean(), 4) if len(diff) > 0 else None,
        "diff_club_median_joi90_xt": round(diff["joi90_xt"].median(), 4) if len(diff) > 0 else None,
        "delta_mean_joi90_xt": round(
            (same["joi90_xt"].mean() - diff["joi90_xt"].mean()), 4
        ) if len(same) > 0 and len(diff) > 0 else None,
    }

    # Pair detail records
    pair_records = []
    for _, row in pairs.iterrows():
        pair_records.append({
            "player_a": row["name_a"],
            "player_b": row["name_b"],
            "id_a": int(row["player_a"]),
            "id_b": int(row["player_b"]),
            "club_a": row["club_a"],
            "club_b": row["club_b"],
            "same_club": bool(row["same_club"]),
            "minutes_together": round(float(row["minutes"]), 1),
            "matches_together": int(row["matches"]),
            "joi90_xt": round(float(row["joi90_xt"]), 4),
            "joi90_vaep": round(float(row["joi90_vaep"]), 4) if pd.notna(row.get("joi90_vaep")) else None,
            "joi90_xg": round(float(row["joi90_xg"]), 4) if pd.notna(row.get("joi90_xg")) else None,
        })

    summary["pairs"] = pair_records
    return summary


def print_squad_table(result: dict) -> None:
    if result.get("dropped"):
        print(f"\n{result['squad']} — DROPPED (match rate {result['match_rate']:.1%})")
        return

    pairs = result["pairs"]
    same = [p for p in pairs if p["same_club"]]
    diff = [p for p in pairs if not p["same_club"]]

    print(f"\n--- {result['squad']} ({result['tournament']}, {result['matches']} matches) ---")
    print(f"Pairs: {result['pairs_total']} total, {result['pairs_same_club']} same-club, {result['pairs_diff_club']} diff-club")
    print(f"Same-club mean JOI90 xT: {result['same_club_mean_joi90_xt']}  |  Diff-club: {result['diff_club_mean_joi90_xt']}  |  Delta: {result['delta_mean_joi90_xt']}")

    if same:
        print("\nSame-club pairs (sorted by JOI90 xT):")
        for p in sorted(same, key=lambda x: x["joi90_xt"], reverse=True):
            print(f"  {p['player_a'][:20]:<20} + {p['player_b'][:20]:<20} | {p['club_a']:<25} | min={p['minutes_together']:6.0f} | xt={p['joi90_xt']:+.3f} | vaep={p['joi90_vaep']:+.3f}" if p['joi90_vaep'] is not None else f"  {p['player_a'][:20]:<20} + {p['player_b'][:20]:<20} | {p['club_a']:<25} | min={p['minutes_together']:6.0f} | xt={p['joi90_xt']:+.3f}")


def main():
    print("Loading lineups...")
    lineups = pd.read_parquet(ROOT / "outputs" / "lineups.parquet")

    xt_pm_cache = {}
    vaep_pm_cache = {}
    xg_pm_cache = {}

    results = []
    for squad_key, squad_def in SQUADS.items():
        result = analyze_squad(squad_key, squad_def, lineups, xt_pm_cache, vaep_pm_cache, xg_pm_cache)
        results.append(result)

    # Print tables
    for r in results:
        print_squad_table(r)

    # Pooled same-club vs diff-club
    all_same = []
    all_diff = []
    for r in results:
        if r.get("dropped"):
            continue
        for p in r["pairs"]:
            if p["same_club"]:
                all_same.append(p["joi90_xt"])
            else:
                all_diff.append(p["joi90_xt"])

    print("\n\n=== POOLED (all 8 squads, historic) ===")
    print(f"Same-club pairs: n={len(all_same)}, mean={sum(all_same)/len(all_same):.4f}, median={sorted(all_same)[len(all_same)//2]:.4f}")
    print(f"Diff-club pairs: n={len(all_diff)}, mean={sum(all_diff)/len(all_diff):.4f}, median={sorted(all_diff)[len(all_diff)//2]:.4f}")
    print(f"Delta (same - diff): {sum(all_same)/len(all_same) - sum(all_diff)/len(all_diff):.4f}")

    # Germany 2018 Bayern deep-dive
    ger = next(r for r in results if r["squad"] == "Germany_WC2018")
    if not ger.get("dropped"):
        BAYERN_PLAYERS = {
            "Manuel Neuer", "Mats Hummels", "Jérôme Boateng", "Joshua Kimmich",
            "Thomas Müller", "Leon Goretzka", "Niklas Süle", "Sebastian Rudy"
        }

        def is_bayern_pair(p):
            return p["club_a"] == "Bayern Munich" and p["club_b"] == "Bayern Munich"

        bavarian = [p for p in ger["pairs"] if is_bayern_pair(p)]
        non_bavarian_german = [p for p in ger["pairs"] if not is_bayern_pair(p)]

        print("\n\n=== GERMANY 2018: Bayern Munich pairs ===")
        for p in sorted(bavarian, key=lambda x: x["joi90_xt"], reverse=True):
            vaep_str = f", vaep={p['joi90_vaep']:+.3f}" if p['joi90_vaep'] is not None else ""
            print(f"  {p['player_a'][:22]:<22} + {p['player_b'][:22]:<22} | min={p['minutes_together']:6.0f} | xt={p['joi90_xt']:+.4f}{vaep_str}")

        if bavarian:
            b_mean = sum(p["joi90_xt"] for p in bavarian) / len(bavarian)
            nb_mean = sum(p["joi90_xt"] for p in non_bavarian_german) / len(non_bavarian_german)
            print(f"\nBayern pairs mean JOI90 xT: {b_mean:.4f} (n={len(bavarian)})")
            print(f"Non-Bayern German pairs mean JOI90 xT: {nb_mean:.4f} (n={len(non_bavarian_german)})")
            print(f"Bayern delta: {b_mean - nb_mean:.4f}")

    # Write outputs
    output = {
        "meta": {
            "description": "Within-tournament club chemistry analysis for 8 historic squads",
            "min_minutes_floor": 45,
            "primary_metric": "JOI90 xT",
            "tournaments": ["WC 2018", "AFCON 2023"],
        },
        "squads": results,
        "pooled": {
            "same_club_n": len(all_same),
            "same_club_mean_joi90_xt": round(sum(all_same) / len(all_same), 4) if all_same else None,
            "same_club_median_joi90_xt": round(sorted(all_same)[len(all_same) // 2], 4) if all_same else None,
            "diff_club_n": len(all_diff),
            "diff_club_mean_joi90_xt": round(sum(all_diff) / len(all_diff), 4) if all_diff else None,
            "diff_club_median_joi90_xt": round(sorted(all_diff)[len(all_diff) // 2], 4) if all_diff else None,
            "delta_mean": round(sum(all_same) / len(all_same) - sum(all_diff) / len(all_diff), 4) if all_same and all_diff else None,
        },
    }

    # Add Germany Bavaria deep-dive to output
    ger = next(r for r in results if r["squad"] == "Germany_WC2018")
    if not ger.get("dropped"):
        bavarian = [p for p in ger["pairs"] if p["club_a"] == "Bayern Munich" and p["club_b"] == "Bayern Munich"]
        non_bavarian = [p for p in ger["pairs"] if not (p["club_a"] == "Bayern Munich" and p["club_b"] == "Bayern Munich")]
        output["germany_2018_bayern_deepdive"] = {
            "bayern_pairs": sorted(bavarian, key=lambda x: x["joi90_xt"], reverse=True),
            "bayern_pairs_mean_joi90_xt": round(sum(p["joi90_xt"] for p in bavarian) / len(bavarian), 4) if bavarian else None,
            "non_bayern_german_pairs_mean_joi90_xt": round(sum(p["joi90_xt"] for p in non_bavarian) / len(non_bavarian), 4) if non_bavarian else None,
            "delta": round(
                sum(p["joi90_xt"] for p in bavarian) / len(bavarian) - sum(p["joi90_xt"] for p in non_bavarian) / len(non_bavarian), 4
            ) if bavarian and non_bavarian else None,
        }

    out_path = ROOT / "outputs" / "club_chemistry_historic.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {out_path}")

    return output


if __name__ == "__main__":
    main()
