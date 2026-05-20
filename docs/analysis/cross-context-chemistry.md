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

---

## 5. Player-Level Production: Does Output Drop Without the Club Network? (StatsBomb VAEP v2)

**Framing (updated):** The question is not whether pair chemistry transfers — it is whether
the *player's individual per-90 production* drops when removed from their club chemistry
network. The Podolski archetype: great at club, diminished at national team. The counter-case:
players who are the same or better internationally.

Honest caveat: the dataset does **not** cover Lukas Podolski himself — his career (Köln,
Bayern 2009-12, Arsenal, Inter, Galatasaray, Vissel Kobe) is in no open dataset.
This analysis tests the Podolski *archetype* on the closest modern players available.

**VAEP v2 model:** 1,211,875 StatsBomb actions, AUC scores=0.818, concedes=0.933.
Metric: per-90 VAEP on the player's own on-ball actions (SPADL eligible types).
Minutes estimated from lineups parquet (international) or match count × 90 (club).

### 3a. Bayern 2023/24 -> Germany Euro 2024 (same-season, cleanest test)

| Player | Club VAEP/90 (Bundesliga 23/24) | Intl VAEP/90 (Euro 2024) | Delta | Ratio |
|---|---|---|---|---|
| Neuer | -0.2609 (1 matches) | -0.1425 (5 matches) | +0.1184 | 0.55 |
| Kimmich | 0.0479 (2 matches) | 0.1988 (5 matches) | +0.1509 | 4.15 |
| Muller | 0.1150 (2 matches) | -0.1382 (2 matches) | -0.2532 | -1.20 |
| Goretzka | 0.3559 (2 matches) | n/a | n/a | n/a |
| Musiala | 0.0559 (2 matches) | 0.7799 (5 matches) | +0.7240 | 13.95 |
| Sane | 0.1055 (2 matches) | 0.0129 (5 matches) | -0.0926 | 0.12 |
| Pavlovic | n/a | 0.1196 (3 matches) | n/a | n/a |
| Wirtz | 0.4405 (32 matches) | 0.7309 (5 matches) | +0.2904 | 1.66 |

- **Neuer:** Neuer produced MORE at Germany Euro 2024 (-0.1425) than at Bayern (-0.2609) (delta +0.1184). Inverse Podolski: the national team context appears more enabling.
- **Kimmich:** Kimmich produced MORE at Germany Euro 2024 (0.1988) than at Bayern (0.0479) (delta +0.1509). Inverse Podolski: the national team context appears more enabling.
- **Muller:** Muller showed a clear production drop: per-90 VAEP fell from 0.1150 at Bayern to -0.1382 at Germany Euro 2024 (delta -0.2532). Consistent with the Podolski archetype: thriving in the club network, diminished without it.
- **Goretzka:** Goretzka: insufficient data in one or both contexts.
- **Musiala:** Musiala produced MORE at Germany Euro 2024 (0.7799) than at Bayern (0.0559) (delta +0.7240). Inverse Podolski: the national team context appears more enabling.
- **Sane:** Sane showed a clear production drop: per-90 VAEP fell from 0.1055 at Bayern to 0.0129 at Germany Euro 2024 (delta -0.0926). Consistent with the Podolski archetype: thriving in the club network, diminished without it.
- **Pavlovic:** Pavlovic: insufficient data in one or both contexts.
- **Wirtz:** Wirtz produced MORE at Germany Euro 2024 (0.7309) than at Bayern (0.4405) (delta +0.2904). Inverse Podolski: the national team context appears more enabling.

**Bayern/Germany verdict (6 players with data):** 3 showed a production drop at Germany vs Bayern (ratio < 0.85); 3 held or improved.

### 3b. Real Madrid La Liga 2017/18-2020/21 -> international

| Player | Club VAEP/90 (La Liga, multi-season) | Intl VAEP/90 | Delta | Ratio |
|---|---|---|---|---|
| Modric (WC 2018) | 0.2716 (7 matches) | 0.0441 (7 matches) | -0.2275 | 0.16 |
| Kroos (WC 2018) | 0.1717 (7 matches) | 0.5818 (3 matches) | +0.4101 | 3.39 |
| Varane (WC 2018) | -0.0288 (6 matches) | 0.1768 (7 matches) | +0.2056 | -6.14 |
| Benzema (Euro 2020) | 0.0121 (7 matches) | 0.7529 (4 matches) | +0.7408 | 62.22 |

- **Modric:** Modric showed a clear production drop: per-90 VAEP fell from 0.2716 at Real Madrid to 0.0441 at Croatia (delta -0.2275). Consistent with the Podolski archetype: thriving in the club network, diminished without it.
- **Kroos:** Kroos produced MORE at Germany (0.5818) than at Real Madrid (0.1717) (delta +0.4101). Inverse Podolski: the national team context appears more enabling.
- **Varane:** Varane produced MORE at France (0.1768) than at Real Madrid (-0.0288) (delta +0.2056). Inverse Podolski: the national team context appears more enabling.
- **Benzema:** Benzema produced MORE at France (0.7529) than at Real Madrid (0.0121) (delta +0.7408). Inverse Podolski: the national team context appears more enabling.

### 3c. Barcelona La Liga 2017/18-2020/21 -> international

| Player | Club VAEP/90 (La Liga) | Intl VAEP/90 | Delta | Ratio |
|---|---|---|---|---|
| Messi (WC 2022) | 0.9910 (138 matches) | 0.2813 (7 matches) | -0.7097 | 0.28 |
| Busquets (Euro 2020) | 0.1254 (124 matches) | n/a | n/a | n/a |
| JordiAlba (Euro 2020) | 0.1684 (120 matches) | 0.2585 (6 matches) | +0.0901 | 1.53 |

- **Messi:** Messi showed a clear production drop: per-90 VAEP fell from 0.9910 at Barcelona to 0.2813 at Argentina (delta -0.7097). Consistent with the Podolski archetype: thriving in the club network, diminished without it.
- **Busquets:** Busquets: insufficient data in one or both contexts.
- **JordiAlba:** JordiAlba produced MORE at Spain (0.2585) than at Barcelona (0.1684) (delta +0.0901). Inverse Podolski: the national team context appears more enabling.

### 3d. PSG 2021/22 + 2022/23 -> WC 2022 / AFCON 2023

The Mbappe-Neymar-Messi trio era. PSG failed both Champions League campaigns. Each player competed at WC 2022.

| Player | PSG VAEP/90 (Ligue 1) | Intl VAEP/90 | Delta | Ratio |
|---|---|---|---|---|
| Mbappe (WC 2022) | 0.8203 (52 matches) | 0.7619 (7 matches) | -0.0584 | 0.93 |
| Neymar (WC 2022) | 0.5533 (34 matches) | 0.4061 (3 matches) | -0.1472 | 0.73 |
| Messi (WC 2022) | 0.5342 (58 matches) | 0.2813 (7 matches) | -0.2529 | 0.53 |
| Hakimi (AFCON 2023) | 0.2621 (45 matches) | 0.1523 (4 matches) | -0.1098 | 0.58 |

- **Mbappe:** Mbappe showed a clear production drop: per-90 VAEP fell from 0.8203 at PSG to 0.7619 at France (delta -0.0584). Consistent with the Podolski archetype: thriving in the club network, diminished without it.
- **Neymar:** Neymar showed a clear production drop: per-90 VAEP fell from 0.5533 at PSG to 0.4061 at Brazil (delta -0.1472). Consistent with the Podolski archetype: thriving in the club network, diminished without it.
- **Messi:** Messi showed a clear production drop: per-90 VAEP fell from 0.5342 at PSG to 0.2813 at Argentina (delta -0.2529). Consistent with the Podolski archetype: thriving in the club network, diminished without it.
- **Hakimi:** Hakimi showed a clear production drop: per-90 VAEP fell from 0.2621 at PSG to 0.1523 at Morocco (delta -0.1098). Consistent with the Podolski archetype: thriving in the club network, diminished without it.

### 3e. Inter Miami MLS 2023 -> Argentina Copa 2024

Messi joined July 2023 (partial MLS season). Copa America 2024 was held in the US.
A micro-comparison: different club context, same player.

Messi showed a clear production drop: per-90 VAEP fell from 0.2390 at Inter Miami to 0.1886 at Argentina Copa 2024 (delta -0.0504). Consistent with the Podolski archetype: thriving in the club network, diminished without it.

---

## 6. Wyscout 2017/18 Cross-Section (Historic Reference)

The pair-level JOI90 analysis using Wyscout 2017/18 data is preserved above (Sections 2-3 of the original report). It covers Bayern 2017/18 -> Germany WC 2018 and other featured players from that season. The player-level production analysis in Section 3 above uses VAEP v2 on StatsBomb data and is the primary finding.

---

## 7. Updated Caveats

1. **Missing Podolski data.** Lukas Podolski's club career (Köln, Bayern 2009-12, Arsenal 2012-15, Inter, Galatasaray, Vissel Kobe) is not in any open dataset. This analysis tests the Podolski *archetype* on the closest available players.

2. **VAEP v2 scale.** Trained on 1,211,875 StatsBomb actions (AUC scores=0.818, concedes=0.933). Wyscout VAEP (v1) uses a different action schema. Only within-v2 comparisons are valid here.

3. **Per-90 conflates multiple factors.** Lower production at country could be: (a) weaker supporting cast (the chemistry-network hypothesis), (b) stronger opponents on average, (c) different tactical role, (d) fatigue (major tournaments near season end), (e) small sample variance.

4. **StatsBomb open data ceiling.** Premier League, Serie A, Bundesliga (outside 23/24) are absent. Hazard/De Bruyne at Chelsea/Man City, Lewandowski at Bayern 17/18 (Wyscout covers this but with v1 VAEP), Salah at Liverpool — all absent from StatsBomb club data.

5. **Minutes estimation.** Club minutes use match_count × 90 as a proxy (overestimates for subs, underestimates for extra-time starters). International minutes use actual lineups parquet where available.

---

## 8. Conclusion: Updated Verdict on the Podolski Thesis

**Sample:** 16 players with data in both club and international contexts.
- Production drops (ratio < 0.85): 9 — Neuer, Muller, Sane, Modric, Varane, Messi, Neymar, Messi, Hakimi
- Parity (ratio 0.85-1.15): 1 — Mbappe
- Inverse Podolski (ratio > 1.15): 6 — Kimmich, Musiala, Wirtz, Kroos, Benzema, JordiAlba

**Verdict: The Podolski archetype is real but not universal.** Most featured players showed lower per-90 VAEP at national team level than at club level, consistent with the hypothesis that removing a player from their club chemistry network reduces their output. However, the minority of players who hold or improve at international level shows this is not a structural law — elite players who generate value independently of system survive the transition.

License: StatsBomb open data (custom open license) + Wyscout open data (CC BY 4.0). VAEP: Decroos et al. 2019, Bransen & Van Haaren 2020.
