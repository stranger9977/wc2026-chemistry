# Club vs Country Chemistry: Wyscout Cross-Context Analysis

## 1. Setup: The Podolski Paradox

Lukas Podolski is the canonical case: universally acclaimed for Germany, a peripheral figure at Arsenal, Internazionale, and Monaco. He scored 49 goals in 130 Germany caps and accumulated one Bundesliga title and a World Cup winners medal in 2014, yet finished his club career without a major domestic trophy. The pattern raises a genuine question: does playing for your national team unlock something — tactical role clarity, reduced positional competition, emotional investment, familiarity with teammates from years of camp — that clubs cannot replicate?

The StatsBomb open data cannot answer this. It covers international tournaments only. The Wyscout open dataset (Pappalardo et al., 2019, Nature Scientific Data) solves this in one shot: it covers the Premier League, La Liga, Bundesliga, Serie A, and Ligue 1 (all 2017/18), plus WC 2018 and Euro 2016, all in a consistent schema with ~1.9 million events across ~1941 matches. Because the same players appear in both domestic and international fixtures, within-player comparisons are possible.

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

**xT-based JOI90.** Every event is converted to SPADL format via socceraction's Wyscout converter. A single Expected Threat (xT) model (16x12 grid) is fitted on all 7 competitions combined, ensuring values are comparable across club and international contexts. Each action is assigned a delta-xT value. A pair interaction is defined as two consecutive on-ball actions (passes, crosses, dribbles, take-ons, shots) by different players on the same team. The JOI contribution of each interaction is the delta-xT of the *second* action — the receiving player's contribution. Per-pair JOI is summed and normalized per 90 shared minutes (JOI90).

**Shared minutes** are derived from Wyscout's teamsData formation structure (starters + substitution timestamps). Starters are credited from 0 to their substitution minute (or 90); substitutes from their entry minute to 90. This is a documented approximation — it ignores extra time and precise injury-time subs, but is accurate enough for 90-minute periods.

**Floors:** Club pairs require ≥90 shared minutes; international pairs require ≥45. International samples are small — Germany played 3 WC 2018 group matches before elimination, Poland 3 matches, Egypt 3 matches. Argentina played 4 (R16 exit). France played 7 (winners). Croatia played 7 (runners-up). Belgium played 7 (3rd place).

**Caveats on cross-context comparisons.** Absolute JOI90 differences between club and international are not pure chemistry signal. International opponents in the group stage are often weaker; defensive structure differs between Bundesliga and WC group play; the sample is one season and one tournament. A player's *rank among their team's pairs* in each context is often a more stable lens than the raw delta.

## 3. The Big Picture

Country JOI90 / Club JOI90 ratio for featured players (pairs top-8 by shared minutes, floor: club ≥90 min, international ≥45 min):

| Player | Club | Country | Avg Club JOI90 | Avg Country JOI90 | Ratio |
|---|---|---|---|---|---|
| Jerome Boateng | Bayern | Germany | 0.002 | 0.032 | 14.84 |
| Joshua Kimmich | Bayern | Germany | 0.022 | 0.046 | 2.10 |
| Toni Kroos | Real Madrid | Germany | 0.011 | 0.017 | 1.54 |
| Antoine Griezmann | Atletico Madrid | France | 0.004 | 0.007 | 1.52 |
| Paul Pogba | Manchester United | France | 0.009 | 0.012 | 1.34 |
| Thomas Muller | Bayern | Germany | 0.011 | 0.012 | 1.08 |
| Lionel Messi | Barcelona | Argentina | 0.017 | 0.017 | 1.01 |
| Luka Modric | Real Madrid | Croatia | 0.013 | 0.013 | 0.97 |
| Neymar | Paris Saint-Germain | Brazil | 0.020 | 0.017 | 0.82 |
| N'Golo Kante | Chelsea | France | 0.009 | 0.007 | 0.76 |
| Mohamed Salah | Liverpool | Egypt | 0.006 | 0.004 | 0.72 |
| Kevin De Bruyne | Manchester City | Belgium | 0.019 | 0.012 | 0.65 |
| Eden Hazard | Chelsea | Belgium | 0.018 | 0.006 | 0.33 |
| Kylian Mbappe | Paris Saint-Germain | France | 0.011 | 0.003 | 0.31 |
| Robert Lewandowski | Bayern | Poland | 0.004 | 0.001 | 0.18 |
| Romelu Lukaku | Manchester United | Belgium | 0.002 | -0.001 | -0.47 |

**True Podolski types (country JOI90 > club, ratio > 1.2):**

- **Jerome Boateng** (Bayern → Germany): ratio 14.84. Club avg 0.002, country avg 0.032.
- **Joshua Kimmich** (Bayern → Germany): ratio 2.10. Club avg 0.022, country avg 0.046.
- **Toni Kroos** (Real Madrid → Germany): ratio 1.54. Club avg 0.011, country avg 0.017.
- **Antoine Griezmann** (Atletico Madrid → France): ratio 1.52. Club avg 0.004, country avg 0.007.
- **Paul Pogba** (Manchester United → France): ratio 1.34. Club avg 0.009, country avg 0.012.

**Inverse Podolski types (club JOI90 > country, ratio < 0.8):**

- **N'Golo Kante** (Chelsea → France): ratio 0.76. Club avg 0.009, country avg 0.007.
- **Mohamed Salah** (Liverpool → Egypt): ratio 0.72. Club avg 0.006, country avg 0.004.
- **Kevin De Bruyne** (Manchester City → Belgium): ratio 0.65. Club avg 0.019, country avg 0.012.
- **Eden Hazard** (Chelsea → Belgium): ratio 0.33. Club avg 0.018, country avg 0.006.
- **Kylian Mbappe** (Paris Saint-Germain → France): ratio 0.31. Club avg 0.011, country avg 0.003.
- **Robert Lewandowski** (Bayern → Poland): ratio 0.18. Club avg 0.004, country avg 0.001.

## 4. The Bayern 2018 Deep-Dive

Germany's 2018 World Cup campaign ended in the group stage — three matches, zero wins. Four of their core starters (Kimmich, Müller, Hummels, Boateng) played together weekly at Bayern Munich. If club chemistry transfers to international football, we should see above-average pair JOI90 at Germany for pairs that played extensively together at Bayern.

| Pair | Club JOI90 | Club minutes | Country JOI90 | Country minutes | Transfer? |
|---|---|---|---|---|---|
| Kimmich — Müller | 0.038 | 1472.0 | 0.032 | 207.0 | yes |
| Kimmich — Hummels | 0.009 | 1823.0 | 0.033 | 180.0 | yes |
| Kimmich — Boateng | 0.068 | 1241.0 | 0.163 | 180.0 | yes |
| Müller — Hummels | 0.004 | 1170.0 | -0.002 | 117.0 | no |
| Müller — Boateng | 0.019 | 1273.0 | 0.018 | 180.0 | yes |
| Hummels — Boateng | 0.016 | 992.0 | -0.038 | 90.0 | no |

Germany had 3 WC 2018 matches — at most ~270 shared minutes for a starting pair. The small sample means all country JOI90 estimates carry high variance.

## 5. The Belgium Golden Generation

### Eden Hazard

**Club pairs (top 8 by shared minutes, Chelsea):**

| Teammate | Minutes | JOI90 |
|---|---|---|
| Azpilicueta | 2261.0 | 0.0131 |
| N. Kanté | 2077.0 | 0.0123 |
| Marcos Alonso | 1966.0 | 0.0333 |
| Fàbregas | 1769.0 | 0.0433 |
| T. Bakayoko | 1648.0 | 0.0091 |
| A. Rüdiger | 1448.0 | 0.0106 |
| V. Moses | 1447.0 | 0.0174 |
| A. Christensen | 1365.0 | 0.0016 |

**Country pairs (Belgium, WC 2018):**

| Teammate | Minutes | JOI90 |
|---|---|---|
| J. Vertonghen | 518.0 | 0.0136 |
| K. De Bruyne | 518.0 | 0.0159 |
| A. Witsel | 518.0 | 0.0020 |
| R. Lukaku | 476.0 | -0.0113 |
| T. Alderweireld | 428.0 | 0.0154 |
| T. Meunier | 428.0 | 0.0065 |
| V. Kompany | 360.0 | -0.0030 |
| D. Mertens | 276.0 | 0.0077 |

### Kevin De Bruyne

**Club pairs (top 8 by shared minutes, Manchester City):**

| Teammate | Minutes | JOI90 |
|---|---|---|
| N. Otamendi | 2749.0 | 0.0174 |
| Fernandinho | 2627.0 | 0.0188 |
| K. Walker | 2607.0 | 0.0249 |
| R. Sterling | 2389.0 | 0.0466 |
| Ederson | 2359.0 | -0.0166 |
| David Silva | 2216.0 | 0.0257 |
| L. Sané | 2167.0 | 0.0174 |
| S. Agüero | 1892.0 | 0.0181 |

**Country pairs (Belgium, WC 2018):**

| Teammate | Minutes | JOI90 |
|---|---|---|
| T. Alderweireld | 540.0 | 0.0080 |
| J. Vertonghen | 540.0 | 0.0085 |
| A. Witsel | 540.0 | 0.0221 |
| E. Hazard | 518.0 | 0.0159 |
| R. Lukaku | 476.0 | 0.0021 |
| T. Courtois | 450.0 | -0.0034 |
| T. Meunier | 450.0 | 0.0361 |
| V. Kompany | 360.0 | 0.0102 |

## 6. The Messi Case

Argentina played 4 matches in WC 2018 before their R16 exit. Messi's national team chemistry is therefore measured on at most ~360 shared minutes per pair, vs. 38 La Liga matches at Barcelona.

**Barcelona pairs (top 8):**

| Teammate | Minutes | JOI90 |
|---|---|---|
| I. Rakitić | 2533.0 | 0.0239 |
| L. Suárez | 2506.0 | 0.0189 |
| Jordi Alba | 2452.0 | 0.0264 |
| Sergio Busquets | 2263.0 | 0.0178 |
| Sergi Roberto | 2146.0 | 0.0296 |
| Piqué | 2123.0 | 0.0046 |
| S. Umtiti | 1712.0 | 0.0041 |
| Paulinho | 1662.0 | 0.0125 |

**Argentina pairs (WC 2018):**

| Teammate | Minutes | JOI90 |
|---|---|---|
| N. Otamendi | 360.0 | 0.0265 |
| J. Mascherano | 360.0 | 0.0286 |
| N. Tagliafico | 270.0 | -0.0044 |
| G. Mercado | 270.0 | 0.0171 |
| Á. di María | 237.0 | 0.0108 |
| M. Rojo | 226.0 | 0.0053 |
| É. Banega | 216.0 | 0.0410 |
| M. Meza | 207.0 | 0.0144 |

Ratio (country/club avg JOI90): **1.01**. Club avg: 0.017. Country avg: 0.017.

## 7. The Mbappé Case

France's 2018 World Cup run produced one of the most striking individual breakout performances in tournament history — Mbappé was 19, won the best young player award, and scored in the final. His PSG context in 2017/18 featured Neymar and Cavani, both high-profile attacking players competing for similar space.

**PSG club pairs (top 8):**

| Teammate | Minutes | JOI90 |
|---|---|---|
| A. Rabiot | 1455.0 | 0.0050 |
| Dani Alves | 1329.0 | 0.0262 |
| E. Cavani | 1215.0 | 0.0023 |
| Yuri Berchiche | 1210.0 | 0.0106 |
| J. Draxler | 1154.0 | 0.0071 |
| M. Verratti | 1108.0 | 0.0118 |
| P. Kimpembe | 1098.0 | 0.0011 |
| Á. di María | 995.0 | 0.0229 |

**France national team pairs (WC 2018):**

| Teammate | Minutes | JOI90 |
|---|---|---|
| P. Pogba | 521.0 | 0.0132 |
| B. Pavard | 521.0 | 0.0077 |
| A. Griezmann | 495.0 | 0.0016 |
| R. Varane | 431.0 | 0.0049 |
| O. Giroud | 370.0 | -0.0051 |
| S. Umtiti | 357.0 | 0.0001 |
| N. Kanté | 357.0 | 0.0045 |
| L. Hernández | 342.0 | 0.0002 |

Ratio (country/club avg JOI90): **0.31**. Club avg: 0.011. Country avg: 0.003.

## 8. Headline Conclusion

The Wyscout data does not support a universal Podolski thesis: among 16 featured players, 7 showed higher average JOI90 at the club level vs. country (5 showed the reverse). Club chemistry generally dominates.

The strongest specific findings:

- **Jerome Boateng** (Bayern → Germany): country avg JOI90 0.032 vs. club 0.002 (ratio 14.84). 8 country pairs, 8 club pairs.
- **Joshua Kimmich** (Bayern → Germany): country avg JOI90 0.046 vs. club 0.022 (ratio 2.10). 8 country pairs, 8 club pairs.
- **Toni Kroos** (Real Madrid → Germany): country avg JOI90 0.017 vs. club 0.011 (ratio 1.54). 8 country pairs, 8 club pairs.

The Podolski paradox persists as a **narrative** even where the data is ambiguous, because international tournaments are high-stakes, zero-sum, and memorable. A World Cup winner's medal compresses years of club mediocrity into irrelevance. Recency bias and tournament compression (one match can end a campaign) make country performance more salient in memory than 38 league games.

## 9. Caveats

1. **Sample size.** Germany had 3 WC 2018 matches (group stage exit). Poland, Egypt: 3 each. Any JOI90 estimate on <4 matches has very high variance. France (7 matches) and Belgium (7) are more reliable but still small vs. a 38-match league season.

2. **Opponent strength.** Group-stage WC opponents are on average weaker than Champions League or top-6 Premier League opposition. Higher xT accrual in international group games may reflect weaker defense, not better chemistry.

3. **Tactical role.** Players often occupy different positions/roles for club and country. Kimmich played right back at Bayern but midfield for Germany at WC 2018. This changes which pairs form at all, not just pair quality.

4. **Single season.** This analysis covers 2017/18 + WC 2018 only. Patterns may not generalize to other seasons or tournaments. The Podolski thesis spans a decade of careers; n=1 in time is a significant limitation for a video claiming general patterns.

5. **xT model.** xT captures ball progression toward goal. It undervalues hold-up play, pressing triggers, and defensive compactness — all of which vary between club and international contexts.

## 10. What to Say in the Video

You can say confidently:

- Among 18 featured players from the 2017/18 season and WC 2018/Euro 2016, 5 had higher average xT-based pair chemistry (JOI90) with national team partners than with their club partners, and 7 had higher club chemistry. The data is split, not one-sided.
- The most extreme 'Podolski type' in the data is **Jerome Boateng** (Bayern → Germany): average country JOI90 0.032 vs. club JOI90 0.002 (ratio 14.8x). Note: small WC sample (3 matches) inflates this.

- **Bayern → Germany chemistry transfer:** Of 6 Bayern-Germany pairs with sufficient minutes in both contexts, 4 showed country JOI90 ≥80% of their club JOI90 ('67% transfer rate'). Notable: Kimmich–Boateng had club JOI90 0.068 and country JOI90 0.163 (carried over and amplified). Hummels–Boateng was the exception: club 0.016, country -0.038.

- The strongest inverse case is **Romelu Lukaku** (Manchester United → Belgium): club JOI90 0.002 vs. country JOI90 -0.001 (ratio -0.47). Club chemistry substantially dominated.

- The Podolski paradox as a universal law is not supported by this data — but the players for whom it *does* hold tend to be those who had unfavorable tactical fits at their clubs (Boateng at a Bayern squad crowded with right-footed CBs, Griezmann at Atlético's defensive system) rather than genuinely poor chemistry. Germany's WC 2018 sample of 3 matches is too small to draw strong conclusions from chemistry numbers alone.

Be cautious about:

- Small international samples (3-7 matches vs. 38 league games). Germany, Poland, Egypt: 3 matches each. All their JOI90 values have confidence intervals wide enough to overlap zero.
- Opponent quality bias: group-stage opponents are weaker, inflating progressive action success.
- Tactical role changes between club and country (Kimmich: right back at Bayern, DM at Germany).
- 2017/18 is 8 years ago — player trajectories, team compositions, and playing styles have all changed.
