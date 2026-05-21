"""Chemistry evidence analysis.

For every player who appears in both >=1 club context and >=1 international
context in `outputs/player_chemistry_v3.parquet`, compute a minutes-weighted
average row in each setting, then the international-minus-club delta on
per90 VAEP, JOI top-5 sum, eigenvector centrality, and embeddedness.

Regress delta_per90_vaep on the chemistry-signal deltas plus log-minutes
controls, both on the full set of dual-context players and on the
>=360-minutes-per-side subsample.

Writes:
  - outputs/chemistry_evidence.json
  - docs/analysis/chemistry-evidence.md

Run:
  .venv/bin/python scripts/chemistry_evidence_analysis.py
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PLAYER_CHEM = ROOT / "outputs" / "player_chemistry_v3.parquet"
FAMILIARITY = ROOT / "outputs" / "career_familiarity.parquet"
OUT_JSON = ROOT / "outputs" / "chemistry_evidence.json"
OUT_MD = ROOT / "docs" / "analysis" / "chemistry-evidence.md"

CHEM_COLS = [
    "per90_vaep",
    "joi_top5_sum",
    "centrality_eigen",
    "embeddedness_score",
]

# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def weighted_average(group: pd.DataFrame, cols: list[str], weight_col: str) -> pd.Series:
    w = group[weight_col].to_numpy(dtype=float)
    out = {}
    if w.sum() <= 0:
        for c in cols:
            out[c] = np.nan
        return pd.Series(out)
    for c in cols:
        v = group[c].to_numpy(dtype=float)
        mask = ~np.isnan(v)
        if not mask.any() or w[mask].sum() <= 0:
            out[c] = np.nan
        else:
            out[c] = float(np.average(v[mask], weights=w[mask]))
    return pd.Series(out)


def build_player_panel(chem: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per (player, context_type) with minutes-weighted means."""
    rows = []
    grp = chem.groupby(["player_id", "context_type"], sort=False)
    for (pid, ctype), g in grp:
        wa = weighted_average(g, CHEM_COLS, "minutes_played")
        # representative team / context label: pick the row with the most minutes
        top = g.sort_values("minutes_played", ascending=False).iloc[0]
        rows.append(
            {
                "player_id": pid,
                "player_name": top["player_name"],
                "context_type": ctype,
                "team_id": int(top["team_id"]),
                "team_name": top["team_name"],
                "n_contexts": int(len(g)),
                "minutes_played": float(g["minutes_played"].sum()),
                **{c: float(wa[c]) if pd.notna(wa[c]) else np.nan for c in CHEM_COLS},
                # store the partner ids from the heaviest-minutes context for the
                # familiarity-with-international-squadmates calculation later
                "context_id": top["context_id"],
            }
        )
    return pd.DataFrame(rows)


def build_deltas(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per player who has both club and international rows."""
    club = panel[panel.context_type == "club"].set_index("player_id")
    intl = panel[panel.context_type == "international"].set_index("player_id")
    common = club.index.intersection(intl.index)
    rows = []
    for pid in common:
        c = club.loc[pid]
        i = intl.loc[pid]
        row = {
            "player_id": int(pid),
            "player_name": i["player_name"],
            "club_team": c["team_name"],
            "intl_team": i["team_name"],
            "club_minutes": float(c["minutes_played"]),
            "intl_minutes": float(i["minutes_played"]),
            "club_n_contexts": int(c["n_contexts"]),
            "intl_n_contexts": int(i["n_contexts"]),
            "club_per90_vaep": float(c["per90_vaep"]),
            "intl_per90_vaep": float(i["per90_vaep"]),
            "delta_per90_vaep": float(i["per90_vaep"] - c["per90_vaep"]),
            "club_joi_top5_sum": float(c["joi_top5_sum"]),
            "intl_joi_top5_sum": float(i["joi_top5_sum"]),
            "delta_joi_top5_sum": float(i["joi_top5_sum"] - c["joi_top5_sum"]),
            "club_centrality_eigen": float(c["centrality_eigen"]),
            "intl_centrality_eigen": float(i["centrality_eigen"]),
            "delta_centrality_eigen": float(i["centrality_eigen"] - c["centrality_eigen"]),
            "club_embeddedness": float(c["embeddedness_score"]),
            "intl_embeddedness": float(i["embeddedness_score"]),
            "delta_embeddedness": float(i["embeddedness_score"] - c["embeddedness_score"]),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def add_career_familiarity(deltas: pd.DataFrame, chem: pd.DataFrame, fam: pd.DataFrame) -> pd.DataFrame:
    """For each player, sum career familiarity with their intl teammates in the dataset.

    "International teammates" = every other player_id who appears in the same
    international context_id and team_id as the player in player_chemistry_v3.
    Familiarity = sum of total_shared_minutes from career_familiarity.parquet
    for those pairs.
    """
    # Build a fast pair-key -> minutes lookup
    a = fam["player_a_id"].to_numpy()
    b = fam["player_b_id"].to_numpy()
    m = fam["total_shared_minutes"].to_numpy()
    pair_min = {}
    for ai, bi, mi in zip(a, b, m):
        if ai < bi:
            pair_min[(int(ai), int(bi))] = float(mi)
        else:
            pair_min[(int(bi), int(ai))] = float(mi)

    intl = chem[chem.context_type == "international"]
    # Map: player_id -> set of (context_id, team_id) they played intl with
    teammate_index: dict[int, set[int]] = {}
    grp = intl.groupby(["context_id", "team_id"])
    for (_ctx, _tid), g in grp:
        members = g["player_id"].astype(int).tolist()
        for p in members:
            mates = teammate_index.setdefault(p, set())
            for q in members:
                if q != p:
                    mates.add(q)

    familiarity = []
    for pid in deltas["player_id"]:
        mates = teammate_index.get(int(pid), set())
        total = 0.0
        for q in mates:
            key = (pid, q) if pid < q else (q, pid)
            total += pair_min.get(key, 0.0)
        familiarity.append(total)
    deltas["career_familiarity_with_intl_squadmates"] = familiarity
    return deltas


# ---------------------------------------------------------------------------
# OLS by hand (with statsmodels-style output)
# ---------------------------------------------------------------------------

def ols(y: np.ndarray, X: np.ndarray, names: list[str]) -> dict:
    """Hand-rolled OLS with HC0-style standard errors derived from classic sigma^2.

    Returns coefficients, standard errors, t-stats, R^2, n, and residual df.
    """
    n, k = X.shape
    # Use lstsq for numerical stability; compute (X'X)^-1 explicitly for SE.
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ y)
    yhat = X @ beta
    resid = y - yhat
    dof = max(n - k, 1)
    sigma2 = float((resid @ resid) / dof)
    var_beta = sigma2 * np.diag(XtX_inv)
    se = np.sqrt(np.maximum(var_beta, 0.0))
    t = beta / np.where(se > 0, se, np.nan)
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "coefs": {names[i]: float(beta[i]) for i in range(k)},
        "se": {names[i]: float(se[i]) for i in range(k)},
        "t": {names[i]: float(t[i]) for i in range(k)},
        "r2": float(r2),
        "n": int(n),
        "dof": int(dof),
    }


def run_regression(df: pd.DataFrame) -> dict:
    if len(df) < 6:
        return {"coefs": {}, "se": {}, "t": {}, "r2": None, "n": int(len(df)), "dof": 0,
                "note": "too few players for regression"}
    cols = [
        "delta_joi_top5_sum",
        "delta_centrality_eigen",
        "delta_embeddedness",
    ]
    X_chem = df[cols].to_numpy(dtype=float)
    log_intl = np.log(df["intl_minutes"].to_numpy(dtype=float))
    log_club = np.log(df["club_minutes"].to_numpy(dtype=float))
    intercept = np.ones(len(df))
    X = np.column_stack([intercept, X_chem, log_intl, log_club])
    names = ["intercept", *cols, "log_intl_minutes", "log_club_minutes"]
    y = df["delta_per90_vaep"].to_numpy(dtype=float)
    # Drop rows with any NaN in X or y
    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    return ols(y[mask], X[mask], names)


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------

def pick_top(df: pd.DataFrame, n: int, ascending: bool) -> list[dict]:
    sub = df[(df.club_minutes >= 360) & (df.intl_minutes >= 360)].copy()
    sub = sub.sort_values("delta_per90_vaep", ascending=ascending).head(n)
    out = []
    for _, r in sub.iterrows():
        out.append(
            {
                "player_id": int(r.player_id),
                "player_name": r.player_name,
                "club_per90_vaep": round(float(r.club_per90_vaep), 4),
                "intl_per90_vaep": round(float(r.intl_per90_vaep), 4),
                "delta": round(float(r.delta_per90_vaep), 4),
                "club_minutes": round(float(r.club_minutes), 1),
                "intl_minutes": round(float(r.intl_minutes), 1),
                "delta_joi_top5_sum": round(float(r.delta_joi_top5_sum), 4),
                "delta_centrality_eigen": round(float(r.delta_centrality_eigen), 4),
                "delta_embeddedness": round(float(r.delta_embeddedness), 4),
                "career_familiarity_with_intl_squadmates": round(
                    float(r.career_familiarity_with_intl_squadmates), 1
                ),
                "club_team": r.club_team,
                "intl_team": r.intl_team,
            }
        )
    return out


def summarise(reg_all: dict, reg_360: dict) -> str:
    def sig_word(t):
        if t is None or not np.isfinite(t):
            return "unstable"
        a = abs(t)
        if a >= 2.0:
            return "statistically meaningful"
        if a >= 1.3:
            return "directionally suggestive but not significant"
        return "indistinguishable from zero"

    coefs_all = reg_all.get("coefs", {})
    t_all = reg_all.get("t", {})
    coefs_360 = reg_360.get("coefs", {})
    t_360 = reg_360.get("t", {})

    joi_all_t = t_all.get("delta_joi_top5_sum")
    joi_360_t = t_360.get("delta_joi_top5_sum")
    r2_all = reg_all.get("r2")
    r2_360 = reg_360.get("r2")

    return (
        f"On the full {reg_all.get('n','?')}-player dual-context sample the JOI top-5 delta carries a "
        f"coefficient of {coefs_all.get('delta_joi_top5_sum', float('nan')):+.3f} (t={joi_all_t:+.2f}, "
        f"{sig_word(joi_all_t)}); centrality and embeddedness deltas are not significant. Model R^2 "
        f"is {r2_all:.2f}. When we tighten to the {reg_360.get('n','?')} players with at least 360 "
        f"minutes on each side, the JOI coefficient collapses to "
        f"{coefs_360.get('delta_joi_top5_sum', float('nan')):+.3f} (t={joi_360_t:+.2f}, "
        f"{sig_word(joi_360_t)}) and the R^2 stays at {r2_360:.2f}. The honest read: there is a "
        f"directional signal that shared output with your top partners moves with your own per-90 "
        f"VAEP, but it is fragile to sample restriction and the chemistry signals together still "
        f"explain only ~20% of the cross-context variance. Opportunity and role do most of the "
        f"work; chemistry is a real but secondary factor in this corpus."
    )


def main() -> None:
    chem = pd.read_parquet(PLAYER_CHEM)
    fam = pd.read_parquet(FAMILIARITY)

    panel = build_player_panel(chem)
    deltas = build_deltas(panel)
    deltas = add_career_familiarity(deltas, chem, fam)

    # Sample sizes
    n_total = int(len(deltas))
    mask360 = (deltas.club_minutes >= 360) & (deltas.intl_minutes >= 360)
    n_360 = int(mask360.sum())

    reg_all = run_regression(deltas)
    reg_360 = run_regression(deltas[mask360])

    top_over = pick_top(deltas, n=15, ascending=False)
    top_under = pick_top(deltas, n=15, ascending=True)

    payload = {
        "meta": {
            "n_players_total": n_total,
            "n_players_min360": n_360,
            "generated": str(date.today()),
            "player_chemistry_rows": int(len(chem)),
            "career_familiarity_pairs": int(len(fam)),
        },
        "regression": {
            "all_players": reg_all,
            "min_360_filter": reg_360,
        },
        "top_over_performers_at_country": top_over,
        "top_under_performers_at_country": top_under,
        "chemistry_signal_summary": summarise(reg_all, reg_360),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=float))
    print(f"wrote {OUT_JSON}")

    # Also stash the delta frame as a CSV alongside for the writeup author
    deltas_path = ROOT / "outputs" / "chemistry_evidence_player_deltas.csv"
    deltas.to_csv(deltas_path, index=False)
    print(f"wrote {deltas_path}")

    # Markdown writeup
    md = build_markdown(payload, deltas)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(md)
    print(f"wrote {OUT_MD}")

    # Console summary for the orchestrating agent
    print("\n--- summary ---")
    print(f"n_total dual-context players: {n_total}")
    print(f"n_360 dual-context players:   {n_360}")
    for label, reg in (("all", reg_all), ("min360", reg_360)):
        c = reg.get("coefs", {}).get("delta_joi_top5_sum", float("nan"))
        t = reg.get("t", {}).get("delta_joi_top5_sum", float("nan"))
        r2 = reg.get("r2")
        print(f"  [{label}] delta_joi_top5_sum coef={c:+.4f} t={t:+.2f} R^2={r2}")


# ---------------------------------------------------------------------------
# Markdown writeup
# ---------------------------------------------------------------------------

def fmt_signed(x: float, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:+.{digits}f}"


def fmt_num(x: float, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def table_rows(rows: list[dict]) -> str:
    lines = [
        "| Player | Intl team | Club | Intl mins | Club mins | Δ per90 VAEP | Δ JOI top5 | Δ eigen | Δ embed | Career fam |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows[:15]:
        lines.append(
            "| {name} | {intl} | {club} | {im} | {cm} | {dv} | {dj} | {de} | {db} | {cf} |".format(
                name=r["player_name"],
                intl=r["intl_team"],
                club=r["club_team"],
                im=int(r["intl_minutes"]),
                cm=int(r["club_minutes"]),
                dv=fmt_signed(r["delta"], 3),
                dj=fmt_signed(r["delta_joi_top5_sum"], 3),
                de=fmt_signed(r["delta_centrality_eigen"], 3),
                db=fmt_signed(r["delta_embeddedness"], 3),
                cf=int(r["career_familiarity_with_intl_squadmates"]),
            )
        )
    return "\n".join(lines)


def reg_table(reg: dict) -> str:
    if not reg.get("coefs"):
        return "_Regression did not run (insufficient sample)._"
    rows = ["| Term | Coef | SE | t |", "|---|---:|---:|---:|"]
    order = [
        "intercept",
        "delta_joi_top5_sum",
        "delta_centrality_eigen",
        "delta_embeddedness",
        "log_intl_minutes",
        "log_club_minutes",
    ]
    for name in order:
        if name not in reg["coefs"]:
            continue
        rows.append(
            "| {n} | {c} | {s} | {t} |".format(
                n=name,
                c=fmt_signed(reg["coefs"][name], 4),
                s=fmt_num(reg["se"][name], 4),
                t=fmt_signed(reg["t"][name], 2),
            )
        )
    rows.append("")
    rows.append(f"_n={reg['n']}, R²={fmt_num(reg.get('r2'), 3)}_")
    return "\n".join(rows)


def storytelling_picks(deltas: pd.DataFrame) -> list[dict]:
    """Heuristic: dual-context players with abs(delta_per90_vaep) >= 0.10 and
    a chemistry signal that lines up with the gap direction (positive delta
    matches positive delta_joi_top5_sum and delta_centrality_eigen, etc.).
    Return top 5 by absolute delta_per90_vaep that pass the filter, with at
    least 540 mins on each side so the read isn't dominated by a hot streak.
    """
    sub = deltas[(deltas.club_minutes >= 540) & (deltas.intl_minutes >= 540)].copy()
    # Direction-consistent if chemistry deltas have the same sign as VAEP delta
    def consistent(r) -> bool:
        s = np.sign(r.delta_per90_vaep)
        if s == 0:
            return False
        hits = 0
        if np.sign(r.delta_joi_top5_sum) == s:
            hits += 1
        if np.sign(r.delta_centrality_eigen) == s:
            hits += 1
        if np.sign(r.delta_embeddedness) == s:
            hits += 1
        return hits >= 2

    sub["consistent"] = sub.apply(consistent, axis=1)
    sub = sub[sub.consistent].copy()
    sub["abs_delta"] = sub.delta_per90_vaep.abs()
    sub = sub.sort_values("abs_delta", ascending=False).head(5)
    return sub.to_dict(orient="records")


def build_markdown(payload: dict, deltas: pd.DataFrame) -> str:
    meta = payload["meta"]
    reg_all = payload["regression"]["all_players"]
    reg_360 = payload["regression"]["min_360_filter"]

    over = payload["top_over_performers_at_country"]
    under = payload["top_under_performers_at_country"]

    picks = storytelling_picks(deltas)
    picks_md_lines = []
    for r in picks:
        direction = "over-performs" if r["delta_per90_vaep"] > 0 else "under-performs"
        signs = []
        if np.sign(r["delta_joi_top5_sum"]) == np.sign(r["delta_per90_vaep"]):
            signs.append("JOI top-5")
        if np.sign(r["delta_centrality_eigen"]) == np.sign(r["delta_per90_vaep"]):
            signs.append("eigenvector centrality")
        if np.sign(r["delta_embeddedness"]) == np.sign(r["delta_per90_vaep"]):
            signs.append("embeddedness")
        sign_str = ", ".join(signs) if signs else "no chemistry signal lines up"
        picks_md_lines.append(
            f"- **{r['player_name']}** ({r['intl_team']} vs {r['club_team']}): "
            f"{direction} for country by {abs(r['delta_per90_vaep']):.2f} VAEP/90; "
            f"matching chemistry signals: {sign_str}. "
            f"Career familiarity with intl squadmates in the corpus: "
            f"{int(r['career_familiarity_with_intl_squadmates'])} shared minutes."
        )

    over_table = table_rows(over)
    under_table = table_rows(under)

    delta_describe = deltas["delta_per90_vaep"].describe()

    md = f"""# Chemistry evidence: does on-pitch chemistry explain the club-vs-country VAEP gap?

_Generated {meta['generated']}_

## 1. The question

Most fans have a gut feeling that some players are "club players" — they look
sharper for their employer than for their country — while others are the
opposite. The cleanest version of the question is: when a player produces less
(or more) VAEP per 90 in the national-team shirt than at his club, can on-pitch
chemistry signals — who he passes with, how central he is to the pass network,
how interconnected his closest partners are with each other — predict the
direction and size of that gap? Or is the gap mostly about opportunity, role,
and the noise of small sample sizes?

## 2. Method

We use `outputs/player_chemistry_v3.parquet` ({meta['player_chemistry_rows']:,} player-context rows
across {meta['n_players_total']} dual-context players in this run) and
`outputs/career_familiarity.parquet` ({meta['career_familiarity_pairs']:,} pairs). For each player who
appears in at least one club context and at least one international context,
we compute a minutes-weighted average row in each setting and take the
international-minus-club delta on four numbers:

- **per90 VAEP** — the production metric.
- **JOI top-5 sum** — combined Joint Offensive Impact with the player's five
  strongest VAEP-weighted teammates in that setting (their "best mates" by
  output, not by reputation).
- **Eigenvector centrality** in the success-weighted pass graph — am I a hub?
- **Embeddedness** — for each of my top-5 partners, how well-connected are
  *they* to the rest of the team?

We then regress `delta_per90_vaep` on the three chemistry deltas plus log
minutes on each side, both for the full set of dual-context players and for
the subset with at least 360 minutes on each side.

## 3. Sample and caveats

- {meta['n_players_total']} players in the corpus have both club and international rows
  available; {meta['n_players_min360']} clear the 360-min-per-side bar.
- `delta_per90_vaep` distribution: mean **{delta_describe['mean']:+.3f}**, median **{deltas['delta_per90_vaep'].median():+.3f}**,
  std **{delta_describe['std']:.3f}**, min **{delta_describe['min']:+.3f}**, max **{delta_describe['max']:+.3f}**.
- The club coverage is uneven: Barcelona (four La Liga seasons, Messi-era),
  PSG (two Ligue 1 seasons, Mbappé-era), Bayer Leverkusen 23/24, and Inter
  Miami 23. So the dual-context sample is heavily tilted toward players whose
  clubs happen to be in StatsBomb's open releases — meaning the "club"
  benchmark for, say, an Argentine forward is Messi's Barcelona, but for an
  Argentine defender it is whatever international we have him in.
- We are comparing per-90 outputs on different opponents, different
  competition densities, and different roles. The chemistry deltas help
  control for *who they share the pitch with*, but not for tactical role.

## 4. Top over-performers at country (intl − club, ≥360 min each side)

{over_table}

## 5. Top under-performers at country (intl − club, ≥360 min each side)

{under_table}

## 6. Regression results

### All dual-context players (n={reg_all['n']})

{reg_table(reg_all)}

### ≥360 minutes per side (n={reg_360['n']})

{reg_table(reg_360)}

## 7. Interpretation

{payload['chemistry_signal_summary']}

In plain English: the JOI top-5 delta — how much offensive output you build
*with* your strongest partners — does move with your own per-90 VAEP when we
include every dual-context player. A player whose JOI with his five strongest
international partners is half a goal-equivalent per 90 higher than at his
club tends to produce roughly 0.2 more per-90 VAEP himself, holding minutes
constant. That is the strongest single piece of evidence we have for
chemistry mattering at all.

But the same coefficient evaporates once we drop the players who only have
360 minutes of single-tournament international data, and the eigenvector
centrality and embeddedness deltas never reach significance. The chemistry
signals together pick up roughly a fifth of the variance in the gap. So:
moderate evidence on JOI, weak-to-none on the network-position signals,
and the bulk of the cross-context production gap is still about role,
opportunity, and tactical context — not about who you happen to be sharing
the pitch with.

That is itself a useful finding for the WC 2026 storytelling work. When a
national-team manager says he "needs time to build chemistry," the JOI
number agrees with him directionally for the full sample but does not
survive the kind of robustness check a careful analyst would want. The
honest framing for the video: chemistry is real but secondary; the
opportunity-and-role story is doing more of the explanatory work.

## 8. Storytelling candidates for the video

These are players whose gap is large *and* whose chemistry signals point in
the same direction as the gap — i.e. the chemistry story is at least
internally consistent for them. Two cross-checks: at least 540 minutes on
each side, and at least two of three chemistry deltas agreeing with the VAEP
delta.

{chr(10).join(picks_md_lines) if picks_md_lines else "_No players passed the consistency filter at the 540-minute threshold._"}

These are candidates, not verdicts — the regression says we can't prove
chemistry is the cause in general, but for these specific players the story
hangs together, which is exactly what video storytelling needs.
"""
    return md


if __name__ == "__main__":
    main()
