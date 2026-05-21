"""Predict 2026 WC squad chemistry from our observational corpus.

For each of 48 WC 2026 nation YAMLs, match named players to
``player_chemistry_v3.parquet`` and compute four squad-level signals:

  a) mean most-recent-club centrality (eigen + weighted degree)
  b) intra-squad shared-minutes density (from career_familiarity)
  c) shared-club density (clubs / players, lower = more concentrated)
  d) mean observed JOI90 across pairs that actually shared a context

Composite rank = average z-score across the four signals.

Writes ``outputs/wc2026_squad_chemistry.json``.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent
SQUAD_DIR = REPO / "squads" / "wc2026"
PCV3_PATH = REPO / "outputs" / "player_chemistry_v3.parquet"
FAM_PATH = REPO / "outputs" / "career_familiarity.parquet"
OUT_PATH = REPO / "outputs" / "wc2026_squad_chemistry.json"
GENERATED = "2026-05-21"


# ---------------------------------------------------------------------------
# name normalisation
# ---------------------------------------------------------------------------

_SPECIAL_MAP = str.maketrans(
    {
        "ø": "o",
        "Ø": "o",
        "ß": "ss",
        "æ": "ae",
        "Æ": "ae",
        "œ": "oe",
        "Œ": "oe",
        "đ": "d",
        "Đ": "d",
        "ł": "l",
        "Ł": "l",
        "ð": "d",
        "Ð": "d",
        "þ": "th",
        "Þ": "th",
        "ı": "i",
        "İ": "i",
        "ʼ": "'",
        "’": "'",
        "‘": "'",
    }
)


def normalize_name(s: str) -> str:
    """Lowercase, strip accents/diacritics, normalise whitespace."""
    if s is None:
        return ""
    # handle characters that NFKD doesn't decompose
    s = s.translate(_SPECIAL_MAP)
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ASCII", "ignore").decode("ASCII")
    s = s.lower()
    # treat hyphens as word separators (Zaïre-Emery, Alexander-Arnold)
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokens(s: str) -> list[str]:
    return [t for t in normalize_name(s).split(" ") if t]


# ---------------------------------------------------------------------------
# manual nickname aliases for famous players where conventional name has no
# overlap with the dataset's full legal name (e.g. Pedri = Pedro González López)
# ---------------------------------------------------------------------------
NICKNAME_ALIASES: dict[str, str] = {
    # Spain
    "pedri": "Pedro González López",
    "gavi": "Pablo Martín Páez Gavira",
    "rodri": "Rodrigo Hernández Cascante",
    # Portugal
    "vitinha": "Vitor Machado Ferreira",
    "diogo jota": "Diogo José Teixeira da Silva",
    "bernardo silva": "Bernardo Mota Veiga de Carvalho e Silva",
    "bruno fernandes": "Bruno Miguel Borges Fernandes",
    "ruben dias": "Rúben Santos Gato Alves Dias",
    "joao felix": "João Félix Sequeira",
    # Brazil
    "casemiro": "Carlos Henrique Casimiro",
    "marquinhos": "Marcos Aoás Corrêa",
    "raphinha": "Raphael Dias Belloli",
    "vinicius jr": "Vinícius José Paixão de Oliveira Júnior",
    "vinicius junior": "Vinícius José Paixão de Oliveira Júnior",
    "vini jr": "Vinícius José Paixão de Oliveira Júnior",
    "rodrygo": "Rodrygo Silva de Goes",
    # Colombia
    "lerma jr": "Jefferson Andrés Lerma Solís",
    "lerma": "Jefferson Andrés Lerma Solís",
}


def last_initial_key(s: str) -> str | None:
    toks = tokens(s)
    if len(toks) < 2:
        return None
    return f"{toks[-1]}|{toks[0][0]}"


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------


def build_dataset_index(pcv3: pd.DataFrame) -> dict:
    """Return helpers for matching squad names to player_chemistry_v3."""
    players = pcv3[["player_id", "player_name"]].drop_duplicates("player_id")
    rows = []
    for pid, name in zip(players["player_id"], players["player_name"]):
        rows.append((int(pid), name, set(tokens(name)), last_initial_key(name)))

    # token bag for fast lookup of subset matches
    token_to_pids: dict[str, set[int]] = defaultdict(set)
    for pid, _, toks, _ in rows:
        for t in toks:
            if len(t) >= 3:  # ignore very short tokens
                token_to_pids[t].add(pid)

    # last-initial fallback
    last_key_to_pids: dict[str, set[int]] = defaultdict(set)
    for pid, _, _, key in rows:
        if key:
            last_key_to_pids[key].add(pid)

    pid_to_info = {pid: (name, toks) for pid, name, toks, _ in rows}
    return {
        "token_to_pids": token_to_pids,
        "last_key_to_pids": last_key_to_pids,
        "pid_to_info": pid_to_info,
    }


def match_player(query: str, idx: dict) -> int | None:
    """Return the player_id whose name best matches ``query``.

    Strategy:
      0) nickname-alias rewrite
      1) token-subset match (query tokens subset of dataset tokens or vice versa)
      2) last-name + first-initial fallback (with first-letter check)
      3) None
    """
    # rewrite by nickname alias if known
    q_norm = normalize_name(query)
    if q_norm in NICKNAME_ALIASES:
        query = NICKNAME_ALIASES[q_norm]

    q_tok_list = tokens(query)
    q_toks = set(q_tok_list)
    if not q_toks:
        return None
    q_surname = q_tok_list[-1]
    q_first = q_tok_list[0]

    # candidate pool from any rare-ish token
    cand: set[int] = set()
    for t in q_toks:
        if len(t) >= 3 and t in idx["token_to_pids"]:
            cand |= idx["token_to_pids"][t]

    best: tuple[int, int, int] | None = None  # (score, -len_diff, pid)
    for pid in cand:
        _, d_toks = idx["pid_to_info"][pid]
        if not d_toks:
            continue
        inter = q_toks & d_toks
        if not inter:
            continue
        subset_q = q_toks.issubset(d_toks)
        subset_d = d_toks.issubset(q_toks)

        # The surname (last token) must appear in the dataset name. This
        # is the gating constraint that prevents Cole Palmer -> Kasey Palmer
        # type matches from being acceptable on their own.
        surname_in_d = q_surname in d_toks
        # also accept if any q-token is the dataset's last token (handles
        # cases where the dataset has additional given/family names appended)
        d_surname = list(d_toks)
        # the actual final-token of dataset name, in original order:
        d_last = idx["pid_to_info"][pid][0]
        d_last_tokens = tokens(d_last)
        d_final = d_last_tokens[-1] if d_last_tokens else ""
        surname_match = surname_in_d or (d_final in q_toks)
        if not surname_match:
            continue

        # additionally require some first-name signal: either the YAML's
        # first token appears in the dataset tokens, or the dataset first
        # token starts with the same letter as the YAML first token.
        first_match = (
            q_first in d_toks
            or (d_last_tokens and d_last_tokens[0][:1] == q_first[:1])
        )
        if not (subset_q or subset_d) and not first_match:
            continue

        score = len(inter) * 10
        if subset_q:
            score += 5
        if subset_d:
            score += 5
        if first_match:
            score += 2
        # prefer closer length match
        len_diff = abs(len(q_toks) - len(d_toks))
        key = (score, -len_diff, pid)
        if best is None or key > best:
            best = key

    if best is not None and best[0] >= 10:
        return best[2]

    # fallback: last-name + first-initial
    key = last_initial_key(query)
    if key and key in idx["last_key_to_pids"]:
        pids = idx["last_key_to_pids"][key]
        if len(pids) == 1:
            return next(iter(pids))
        # pick the candidate whose token set has highest overlap with query
        best_pid, best_overlap = None, -1
        for pid in pids:
            _, d_toks = idx["pid_to_info"][pid]
            overlap = len(q_toks & d_toks)
            if overlap > best_overlap:
                best_pid, best_overlap = pid, overlap
        return best_pid

    return None


# ---------------------------------------------------------------------------
# context ordering (for "most recent")
# ---------------------------------------------------------------------------


def context_year(label: str) -> int:
    """Pull a 4-digit year from a context_label, or use the 2-digit season end."""
    m = re.search(r"(20\d{2})", label or "")
    if m:
        return int(m.group(1))
    # e.g. "Barcelona La Liga 17/18" -> 2018
    m = re.search(r"(\d{2})/(\d{2})", label or "")
    if m:
        return 2000 + int(m.group(2))
    m = re.search(r"\b(\d{2})\b", label or "")
    if m:
        return 2000 + int(m.group(1))
    return 0


def most_recent_club_row(pcv3_player: pd.DataFrame) -> pd.Series | None:
    club_rows = pcv3_player[pcv3_player["context_type"] == "club"]
    if club_rows.empty:
        return None
    club_rows = club_rows.assign(_yr=club_rows["context_label"].map(context_year))
    club_rows = club_rows.sort_values("_yr", ascending=False)
    return club_rows.iloc[0]


# ---------------------------------------------------------------------------
# club name normalisation
# ---------------------------------------------------------------------------

_CLUB_ALIASES = {
    "psg": "paris saint germain",
    "paris saint-germain": "paris saint germain",
    "man city": "manchester city",
    "man utd": "manchester united",
    "man united": "manchester united",
    "bayern": "bayern munich",
    "atletico madrid": "atletico madrid",
    "real betis": "real betis",
    "inter": "internazionale",
    "inter milan": "internazionale",
}


def norm_club(s: str) -> str:
    n = normalize_name(s)
    n = n.replace("-", " ")
    n = re.sub(r"\bfc\b", "", n)
    n = re.sub(r"\bcf\b", "", n)
    n = re.sub(r"\bac\b", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return _CLUB_ALIASES.get(n, n)


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------


def load_squads() -> list[dict]:
    squads = []
    for path in sorted(SQUAD_DIR.glob("*.yaml")):
        with open(path) as f:
            data = yaml.safe_load(f)
        squads.append(data)
    return squads


# ---------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    print("Loading data...", file=sys.stderr)
    pcv3 = pd.read_parquet(PCV3_PATH)
    fam = pd.read_parquet(FAM_PATH)
    idx = build_dataset_index(pcv3)

    # familiarity lookup
    fam_lookup: dict[tuple[int, int], float] = {
        (int(a), int(b)): float(m)
        for a, b, m in zip(
            fam["player_a_id"], fam["player_b_id"], fam["total_shared_minutes"]
        )
    }
    fam_contexts: dict[tuple[int, int], str] = {
        (int(a), int(b)): c
        for a, b, c in zip(
            fam["player_a_id"], fam["player_b_id"], fam["contexts_shared"]
        )
    }

    # pcv3 indexed by player for repeated lookups
    pcv3_by_pid: dict[int, pd.DataFrame] = {
        int(pid): grp for pid, grp in pcv3.groupby("player_id")
    }

    # ---- pair-JOI map: for every (player, partner) pair, the most recent JOI90
    pair_joi: dict[tuple[int, int], tuple[float, int]] = {}  # (a,b) -> (joi90, yr)
    for _, row in pcv3.iterrows():
        a = int(row["player_id"])
        yr = context_year(row["context_label"])
        for i in range(1, 6):
            pid_b = row.get(f"top_partner_{i}_id")
            joi = row.get(f"top_partner_{i}_joi90")
            if pid_b is None or pd.isna(pid_b) or pd.isna(joi):
                continue
            b = int(pid_b)
            key = (min(a, b), max(a, b))
            if key in pair_joi:
                if yr > pair_joi[key][1]:
                    pair_joi[key] = (float(joi), yr)
            else:
                pair_joi[key] = (float(joi), yr)

    squads = load_squads()
    print(f"Loaded {len(squads)} squads", file=sys.stderr)

    # ---- per-squad processing
    squad_records: list[dict] = []
    total_players = 0
    total_matched = 0
    unmatched_by_nation: dict[str, list[str]] = {}

    for sq in squads:
        nation = sq["nation"]
        code = sq["nation_code"]
        players = sq["players"]
        total_players += len(players)

        matched: list[tuple[int, str, str, str]] = []  # (pid, raw_name, raw_club, pos)
        unmatched: list[str] = []
        for p in players:
            pid = match_player(p["name"], idx)
            if pid is None:
                unmatched.append(p["name"])
            else:
                matched.append(
                    (pid, p["name"], p.get("club", ""), p.get("position", ""))
                )
        total_matched += len(matched)
        if unmatched:
            unmatched_by_nation[code] = unmatched

        n = len(players)
        n_matched = len(matched)
        # flag if the YAML is empty (squad TBA) OR if <50% of named players
        # were matched into our observational corpus
        low_match = (n == 0) or (n_matched < 0.5 * n)

        # ---- signal (a) most-recent-club centrality
        eigens: list[float] = []
        wdegs: list[float] = []
        for pid, _, _, _ in matched:
            g = pcv3_by_pid.get(pid)
            if g is None:
                continue
            row = most_recent_club_row(g)
            if row is None:
                continue
            ce = row["centrality_eigen"]
            cwd = row["centrality_weighted_degree"]
            if not pd.isna(ce):
                eigens.append(float(ce))
            if not pd.isna(cwd):
                wdegs.append(float(cwd))
        mean_eigen = float(sum(eigens) / len(eigens)) if eigens else None
        mean_wdeg = float(sum(wdegs) / len(wdegs)) if wdegs else None

        # ---- signal (b) intra-squad familiarity
        # dedupe pids so we don't double-count a player if two YAML entries
        # matched the same dataset id (extremely unlikely, defensive)
        pids = sorted({pid for pid, _, _, _ in matched})
        n_pairs = max(1, len(pids) * (len(pids) - 1) // 2)
        shared_sum = 0.0
        pair_shared: list[tuple[int, int, float]] = []
        for a, b in combinations(sorted(pids), 2):
            key = (min(a, b), max(a, b))
            sm = fam_lookup.get(key, 0.0)
            shared_sum += sm
            if sm > 0:
                pair_shared.append((a, b, sm))
        familiarity_score = shared_sum / (n_pairs * 1000.0)

        # ---- signal (c) shared-club density
        clubs_norm = [norm_club(p.get("club", "")) for p in players if p.get("club")]
        if clubs_norm:
            shared_club_density = len(set(clubs_norm)) / len(clubs_norm)
        else:
            shared_club_density = 1.0

        # ---- signal (d) mean observed pair JOI90
        joi_vals: list[float] = []
        for a, b in combinations(sorted(pids), 2):
            key = (min(a, b), max(a, b))
            if key in pair_joi:
                joi_vals.append(pair_joi[key][0])
        mean_pair_joi = float(sum(joi_vals) / len(joi_vals)) if joi_vals else None
        n_qualifying_pairs = len(joi_vals)

        # ---- spine: top-3 pairs by intra-squad shared minutes
        pair_shared.sort(key=lambda x: x[2], reverse=True)
        spine: list[dict] = []
        name_of = {pid: nm for pid, nm, _, _ in matched}
        for a, b, sm in pair_shared[:3]:
            ctx_str = fam_contexts.get((min(a, b), max(a, b)), "")
            ctx_ids = [c.strip() for c in (ctx_str.split(",") if ctx_str else []) if c.strip()]
            # resolve shared clubs/teams by finding contexts where both
            # players appear on the same team_name
            shared_clubs: list[str] = []
            shared_contexts: list[str] = []
            seen_club: set[str] = set()
            seen_ctx: set[str] = set()
            if ctx_ids:
                sub = pcv3[pcv3["context_id"].isin(ctx_ids)]
                for cid in ctx_ids:
                    rows = sub[sub["context_id"] == cid]
                    teams_a = set(rows[rows["player_id"] == a]["team_name"].tolist())
                    teams_b = set(rows[rows["player_id"] == b]["team_name"].tolist())
                    common = teams_a & teams_b
                    for c in common:
                        if c not in seen_club:
                            shared_clubs.append(c)
                            seen_club.add(c)
                    for lbl in rows["context_label"].unique():
                        if lbl not in seen_ctx:
                            shared_contexts.append(lbl)
                            seen_ctx.add(lbl)
            spine.append(
                {
                    "player_a": name_of.get(a, str(a)),
                    "player_b": name_of.get(b, str(b)),
                    "shared_minutes": int(sm),
                    "shared_clubs": shared_clubs,
                    "shared_contexts": shared_contexts,
                }
            )

        squad_records.append(
            {
                "nation_code": code,
                "nation": nation,
                "n_players": n,
                "n_matched": n_matched,
                "low_match_warning": bool(low_match),
                "signals": {
                    "mean_club_centrality_eigen": mean_eigen,
                    "mean_club_centrality_weighted_degree": mean_wdeg,
                    "intra_squad_familiarity_score": familiarity_score,
                    "shared_club_density": shared_club_density,
                    "mean_observed_pair_joi90": mean_pair_joi,
                    "n_qualifying_pairs": n_qualifying_pairs,
                },
                "spine_pairs": spine,
            }
        )

    # ---- z-scores across the four signals (over the 48 squads)
    def zscores(values: list[float | None], invert: bool = False) -> list[float | None]:
        nums = [v for v in values if v is not None]
        if not nums:
            return [None] * len(values)
        mu = sum(nums) / len(nums)
        var = sum((v - mu) ** 2 for v in nums) / len(nums)
        sd = var ** 0.5 if var > 0 else 1.0
        out: list[float | None] = []
        for v in values:
            if v is None:
                out.append(None)
            else:
                z = (v - mu) / sd
                out.append(-z if invert else z)
        return out

    # We use the *average* of the two centrality measures as a single signal,
    # so the composite is over four signals total:
    #   (a) mean of (z_eigen, z_wdeg)
    #   (b) familiarity
    #   (c) shared-club density (inverted: lower = better)
    #   (d) mean observed pair joi90
    eigen_vals = [r["signals"]["mean_club_centrality_eigen"] for r in squad_records]
    wdeg_vals = [r["signals"]["mean_club_centrality_weighted_degree"] for r in squad_records]
    fam_vals = [r["signals"]["intra_squad_familiarity_score"] for r in squad_records]
    scd_vals = [r["signals"]["shared_club_density"] for r in squad_records]
    joi_vals_all = [r["signals"]["mean_observed_pair_joi90"] for r in squad_records]

    z_eigen = zscores(eigen_vals)
    z_wdeg = zscores(wdeg_vals)
    z_fam = zscores(fam_vals)
    z_scd = zscores(scd_vals, invert=True)
    z_joi = zscores(joi_vals_all)

    for i, r in enumerate(squad_records):
        parts: list[float] = []
        cent_parts = [z for z in (z_eigen[i], z_wdeg[i]) if z is not None]
        if cent_parts:
            parts.append(sum(cent_parts) / len(cent_parts))
        if z_fam[i] is not None:
            parts.append(z_fam[i])
        if z_scd[i] is not None:
            parts.append(z_scd[i])
        if z_joi[i] is not None:
            parts.append(z_joi[i])
        r["composite_z"] = float(sum(parts) / len(parts)) if parts else 0.0
        r["component_z"] = {
            "club_centrality_eigen_z": z_eigen[i],
            "club_centrality_weighted_degree_z": z_wdeg[i],
            "intra_squad_familiarity_z": z_fam[i],
            "shared_club_density_z_inverted": z_scd[i],
            "mean_observed_pair_joi90_z": z_joi[i],
        }

    # rank
    squad_records.sort(key=lambda r: r["composite_z"], reverse=True)
    for rank, r in enumerate(squad_records, start=1):
        r["rank"] = rank

    output = {
        "meta": {
            "n_squads": len(squad_records),
            "generated": GENERATED,
            "matching_summary": {
                "total_players": total_players,
                "matched_to_dataset": total_matched,
                "unmatched_by_nation": unmatched_by_nation,
            },
            "scoring_notes": (
                "composite_z is the unweighted mean of four standardised signals: "
                "(1) club-centrality (avg of eigen+weighted-degree z), "
                "(2) intra-squad familiarity (shared minutes per pair / 1000), "
                "(3) shared-club density (inverted -- lower fraction is better), "
                "(4) mean observed pair JOI90 across pairs that co-played in our corpus. "
                "Nations with <50% of players matched carry low_match_warning=true."
            ),
        },
        "ranking": [
            {
                "rank": r["rank"],
                "nation_code": r["nation_code"],
                "nation": r["nation"],
                "composite_z": round(r["composite_z"], 4),
                "n_players": r["n_players"],
                "n_matched": r["n_matched"],
                "low_match_warning": r["low_match_warning"],
                "signals": {
                    k: (round(v, 6) if isinstance(v, float) else v)
                    for k, v in r["signals"].items()
                },
                "component_z": {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in r["component_z"].items()
                },
                "spine_pairs": r["spine_pairs"],
            }
            for r in squad_records
        ],
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # ---- console report
    print("\n=== Unmatched players by nation ===", file=sys.stderr)
    for code in sorted(unmatched_by_nation):
        names = unmatched_by_nation[code]
        print(f"  {code} ({len(names)}): {', '.join(names)}", file=sys.stderr)

    print(
        f"\nMatched {total_matched}/{total_players} players "
        f"({100*total_matched/max(total_players,1):.1f}%)",
        file=sys.stderr,
    )

    print("\nTop 5 by composite_z:", file=sys.stderr)
    for r in squad_records[:5]:
        flag = " [LOW MATCH]" if r["low_match_warning"] else ""
        print(
            f"  {r['rank']:>2}. {r['nation_code']} {r['nation']:<25}"
            f"  z={r['composite_z']:+.3f}  matched={r['n_matched']}/{r['n_players']}{flag}",
            file=sys.stderr,
        )

    print("\nBottom 5 by composite_z:", file=sys.stderr)
    for r in squad_records[-5:]:
        flag = " [LOW MATCH]" if r["low_match_warning"] else ""
        print(
            f"  {r['rank']:>2}. {r['nation_code']} {r['nation']:<25}"
            f"  z={r['composite_z']:+.3f}  matched={r['n_matched']}/{r['n_players']}{flag}",
            file=sys.stderr,
        )

    print(f"\nWrote {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
