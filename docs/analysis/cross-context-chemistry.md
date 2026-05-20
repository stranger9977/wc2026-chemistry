# Club vs Country Chemistry: Wyscout Cross-Context Analysis

## 1. Setup: The Podolski Paradox

Lukas Podolski is the canonical case: universally acclaimed for Germany, a peripheral figure at Arsenal, Internazionale, and Monaco. He scored 49 goals in 130 Germany caps and accumulated one Bundesliga title and a World Cup winners medal in 2014, yet finished his club career without a major domestic trophy. The pattern raises a genuine question: does playing for your national team unlock something — tactical role clarity, reduced positional competition, emotional investment, familiarity with teammates from years of camp — that clubs cannot replicate?

**Following the paper, this analysis uses VAEP (Decroos et al. 2019, Bransen & Van Haaren 2020) as the action-value function. xT results are reported alongside for cross-reference.**

The Wyscout open dataset (Pappalardo et al., 2019, Nature Scientific Data) covers the Premier League, La Liga, Bundesliga, Serie A, and Ligue 1 (all 2017/18), plus WC 2018 and Euro 2016, all in a consistent schema with ~1.9 million events. Because the same players appear in both domestic and international fixtures, within-player comparisons are possible.

**Dataset stats:**

- England: 380 matches
- European_Championship: 51 matches
- France: 380 matches
- Germany: 306 matches
- Italy: 380 matches
- Spain: 380 matches
- World_Cup: 64 matches
- Total matches: 1941
- Total SPADL actions (approx): 2,464,092
- License: CC BY 4.0

## 2. Methodology

**VAEP-based JOI90 (primary).** Every event is converted to SPADL format via socceraction's Wyscout converter. A VAEP model is trained on all 2,464,092 SPADL actions across all 7 Wyscout competitions combined (train-set AUC: scores=0.768, concedes=0.792). Following Decroos et al. 2019, two XGBClassifier models predict P(scores in next N actions) and P(concedes in next N actions); N=10. VAEP value per action = delta-P(scores) - delta-P(concedes). A pair interaction is two consecutive on-ball actions (passes, crosses, dribbles, take-ons, shots) by different players on the same team. JOI contribution = VAEP of the second action. Per-pair JOI is summed and normalized per 90 shared minutes (JOI90).

**xT-based JOI90 (secondary reference).** Also computed for cross-reference using a 16x12 xT grid fitted on the same 7 competitions. Values reported in parentheses or secondary columns throughout. VAEP and xT scales differ — do not compare absolute values across metrics.

**Shared minutes** are derived from Wyscout's teamsData formation structure. Floors: club pairs require ≥90 shared minutes; international ≥45.

**Caveats on cross-context comparisons.** Absolute JOI90 differences between club and international are not pure chemistry signal. International opponents are often weaker; the sample is one season and one tournament. Rank among team peers is a more stable lens than raw delta.

## 3. The Big Picture (VAEP-based)

Country VAEP-JOI90 / Club VAEP-JOI90 ratio for featured players (pairs top-8 by shared minutes, floor: club ≥90 min, international ≥45 min):

| Player | Club | Country | Avg Club VAEP-JOI90 | Avg Country VAEP-JOI90 | Ratio | xT Ratio |
|---|---|---|---|---|---|---|
| Jerome Boateng | Bayern | Germany | 0.0268 | 0.0566 | 2.11 | 14.84 |
| Romelu Lukaku | Manchester United | Belgium | 0.0753 | 0.1276 | 1.69 | -0.47 |
| Kylian Mbappe | Paris Saint-Germain | France | 0.0842 | 0.0898 | 1.07 | 0.31 |
| Eden Hazard | Chelsea | Belgium | 0.0443 | 0.0449 | 1.01 | 0.33 |
| Kevin De Bruyne | Manchester City | Belgium | 0.0900 | 0.0858 | 0.95 | 0.65 |
| Antoine Griezmann | Atletico Madrid | France | 0.0719 | 0.0566 | 0.79 | 1.52 |
| Joshua Kimmich | Bayern | Germany | 0.0705 | 0.0546 | 0.78 | 2.10 |
| Mohamed Salah | Liverpool | Egypt | 0.1112 | 0.0808 | 0.73 | 0.72 |
| Toni Kroos | Real Madrid | Germany | 0.0365 | 0.0265 | 0.73 | 1.54 |
| Lionel Messi | Barcelona | Argentina | 0.1615 | 0.1083 | 0.67 | 1.01 |
| Luka Modric | Real Madrid | Croatia | 0.0425 | 0.0198 | 0.47 | 0.97 |
| Paul Pogba | Manchester United | France | 0.0556 | 0.0203 | 0.36 | 1.34 |
| N'Golo Kante | Chelsea | France | 0.0301 | 0.0106 | 0.35 | 0.76 |
| Neymar | Paris Saint-Germain | Brazil | 0.1838 | 0.0464 | 0.25 | 0.82 |
| Robert Lewandowski | Bayern | Poland | 0.1052 | 0.0042 | 0.04 | 0.18 |
| Thomas Muller | Bayern | Germany | 0.0699 | -0.0440 | -0.63 | 1.08 |

**True Podolski types (country VAEP-JOI90 > club, ratio > 1.2):**

- **Jerome Boateng** (Bayern → Germany): ratio 2.11. Club VAEP-avg 0.0268, country VAEP-avg 0.0566.
- **Romelu Lukaku** (Manchester United → Belgium): ratio 1.69. Club VAEP-avg 0.0753, country VAEP-avg 0.1276.

**Inverse Podolski types (club VAEP-JOI90 > country, ratio < 0.8):**

- **Antoine Griezmann** (Atletico Madrid → France): ratio 0.79. Club VAEP-avg 0.0719, country VAEP-avg 0.0566.
- **Joshua Kimmich** (Bayern → Germany): ratio 0.78. Club VAEP-avg 0.0705, country VAEP-avg 0.0546.
- **Mohamed Salah** (Liverpool → Egypt): ratio 0.73. Club VAEP-avg 0.1112, country VAEP-avg 0.0808.
- **Toni Kroos** (Real Madrid → Germany): ratio 0.73. Club VAEP-avg 0.0365, country VAEP-avg 0.0265.
- **Lionel Messi** (Barcelona → Argentina): ratio 0.67. Club VAEP-avg 0.1615, country VAEP-avg 0.1083.
- **Luka Modric** (Real Madrid → Croatia): ratio 0.47. Club VAEP-avg 0.0425, country VAEP-avg 0.0198.

VAEP verdict: 16 players with sufficient data in both contexts. 2 showed higher average VAEP-JOI90 with national team partners (country > club); 11 showed higher club chemistry.

## 4. The Bayern 2017/18 → Germany WC 2018 Deep-Dive

Germany's 2018 World Cup campaign ended in the group stage — three matches, zero wins. Four of their core starters (Kimmich, Müller, Hummels, Boateng) played together weekly at Bayern Munich.

| Pair | Club VAEP-JOI90 | Club xT-JOI90 | Club minutes | Country VAEP-JOI90 | Country xT-JOI90 | Country minutes | Transfer (VAEP)? |
|---|---|---|---|---|---|---|---|
| Kimmich — Müller | 0.1130 | 0.0383 | 1472.0 | 0.0064 | 0.0323 | 207.0 | no |
| Kimmich — Hummels | 0.0027 | 0.0088 | 1823.0 | 0.0088 | 0.0331 | 180.0 | yes |
| Kimmich — Boateng | 0.0390 | 0.0684 | 1241.0 | 0.1782 | 0.1627 | 180.0 | yes |
| Müller — Hummels | -0.0137 | 0.0038 | 1170.0 | -0.0717 | -0.0015 | 117.0 | no |
| Müller — Boateng | 0.0652 | 0.0193 | 1273.0 | 0.0273 | 0.0175 | 180.0 | no |
| Hummels — Boateng | 0.0984 | 0.0163 | 992.0 | -0.0206 | -0.0379 | 90.0 | no |

Germany had 3 WC 2018 matches — at most ~270 shared minutes for a starting pair. High variance on all country estimates.

## 5. Caveats

1. **VAEP scale differences.** VAEP trained on Wyscout SPADL has a different absolute scale from VAEP trained on StatsBomb SPADL (different action distributions, different feature encodings). Within-source comparisons are valid; cross-source comparisons are not.

2. **Sample size.** Germany had 3 WC 2018 matches. Poland, Egypt: 3 each. Any JOI90 estimate on <4 matches has very high variance.

3. **Opponent strength.** Group-stage WC opponents are on average weaker than top-division club opposition.

4. **Tactical role.** Kimmich played right back at Bayern but midfield for Germany. This changes which pairs form at all, not just their quality.

5. **Single season.** This analysis covers 2017/18 + WC 2018 only.

6. **Metric note.** Switching from xT to VAEP changes absolute values substantially (VAEP is compressed near zero for non-shot-producing actions; xT accumulates more on progressive passes). The direction of findings (which players are Podolski-type vs inverse-Podolski) may differ between metrics — check both ratio columns.
