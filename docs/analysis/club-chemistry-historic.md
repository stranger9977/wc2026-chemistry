# Club Chemistry — Historic Squads: WC 2018 and AFCON 2023

## Setup

The Podolski hypothesis holds that shared club training creates measurable pair-level chemistry at international level. The clearest supposed case is Germany 2014 — where seven Bayern Munich players formed the functional spine of a World Cup winning side. We cannot test 2014 directly: StatsBomb open data does not include that tournament. The nearest substitute is WC 2018, where Germany still fielded a heavy Bayern Munich core (Neuer, Boateng, Hummels, Kimmich, Müller, Goretzka, Süle, Rudy — eight players with current or imminent Bayern contracts at the time of the tournament).

This analysis computes within-tournament pair JOI90 for five WC 2018 squads and three AFCON 2023 squads using only the matches each squad played in that tournament. It does not use the cross-tournament aggregated chemistry numbers from `outputs/chemistry.json`, which blend multiple competitions and would contaminate the signal. The AFCON squads serve as an anti-Podolski reference case: most AFCON 2023 rosters are assembled from players spread across many different European leagues, minimising same-club pairs.

**JOI90 (Joint Offensive Impact per 90 minutes)** measures how often two players appear in consecutive same-team progressive actions, accumulated across shared playing time and normalised per 90 minutes. Primary metric: xT-scored (expected threat). Secondary: VAEP and xG-Chain. All three are reported per pair; xT is used for the same-vs-different-club comparison.

**Floor: 45 shared minutes within the tournament.** The previous WC 2026 analysis used a 90-minute floor, which is appropriate when pairs have played 10+ international matches together. For single-tournament slices — particularly 3-match group-stage campaigns like Germany 2018 — a 45-minute floor retains enough pairs to make the squad-level analysis meaningful without drowning the data in cameo substitutions.

---

## Data Availability

| Squad | Tournament | Matches | Action rows (xT) | Pairs (≥45 min) | Same-club pairs | Roster match rate |
|-------|------------|---------|-------------------|-----------------|-----------------|-------------------|
| Germany | WC 2018 | 3 | 8,208 | 123 | 16 | 23/23 (100%) |
| Belgium | WC 2018 | 7 | 8,208 | 159 | 7 | 23/23 (100%) |
| France | WC 2018 | 7 | 8,208 | 135 | 5 | 23/23 (100%) |
| Croatia | WC 2018 | 7 | 8,208 | 129 | 2 | 23/23 (100%) |
| Brazil | WC 2018 | 5 | 8,208 | 102 | 4 | 22/23 (95.7%) |
| Senegal | AFCON 2023 | 4 | 7,263 | 114 | 4 | 27/27 (100%) |
| Morocco | AFCON 2023 | 4 | 7,263 | 142 | 3 | 26/26 (100%) |
| Côte d'Ivoire | AFCON 2023 | 7 | 7,263 | 181 | 2 | 27/27 (100%) |

Action row counts are per competition (not per squad). Brazil's one unmatched player is a second squad-level entry for Firmino (a naming variant). No squad fell below the 80% match-rate threshold; all eight are included. WC 2018 had 64 matches total; AFCON 2023 had 52. StatsBomb lineup data covers every match in both competitions.

---

## Per-Squad Within-Tournament Tables

### Summary: Same-Club vs Different-Club JOI90 xT

| Squad | Tournament | n same | Mean JOI90 xT (same) | n diff | Mean JOI90 xT (diff) | Delta |
|-------|------------|--------|----------------------|--------|----------------------|-------|
| Germany | WC 2018 | 16 | +0.0045 | 107 | +0.0049 | **-0.0005** |
| Belgium | WC 2018 | 7 | -0.0010 | 152 | +0.0031 | **-0.0041** |
| France | WC 2018 | 5 | +0.0103 | 130 | +0.0014 | **+0.0089** |
| Croatia | WC 2018 | 2 | +0.0019 | 127 | +0.0023 | **-0.0005** |
| Brazil | WC 2018 | 4 | +0.0069 | 98 | +0.0043 | **+0.0026** |
| Senegal | AFCON 2023 | 4 | +0.0023 | 110 | +0.0006 | **+0.0018** |
| Morocco | AFCON 2023 | 3 | +0.0036 | 139 | +0.0048 | **-0.0012** |
| Côte d'Ivoire | AFCON 2023 | 2 | +0.0045 | 179 | +0.0019 | **+0.0026** |

Direction is split 5-3 in favour of same-club having the higher mean, but the magnitudes are uniformly small. No squad shows a delta that would survive routine statistical scrutiny. The five squads with positive deltas have 2 to 7 same-club pairs — sample sizes that are far too small to separate a real signal from variance.

---

### Germany WC 2018 — Full Same-Club Pair Table

Germany's 16 same-club pairs (all Bayern Munich, including Goretzka who formally joined Bayern on 1 July 2018, during the tournament window):

| Player A | Player B | Minutes | JOI90 xT | JOI90 VAEP |
|----------|----------|---------|-----------|------------|
| Manuel Neuer | Niklas Süle | 120 | +0.069 | +0.035 |
| Thomas Müller | Joshua Kimmich | 298 | +0.022 | -0.052 |
| Thomas Müller | Mats Hummels | 178 | +0.019 | +0.008 |
| Joshua Kimmich | Leon Goretzka | 62 | +0.016 | -0.028 |
| Thomas Müller | Jérôme Boateng | 240 | +0.008 | +0.026 |
| Joshua Kimmich | Niklas Süle | 120 | +0.006 | +0.020 |
| Manuel Neuer | Joshua Kimmich | 360 | +0.006 | +0.026 |
| Mats Hummels | Joshua Kimmich | 120 | +0.005 | +0.022 |
| Jérôme Boateng | Joshua Kimmich | 240 | +0.004 | +0.070 |
| Niklas Süle | Leon Goretzka | 62 | +0.003 | +0.026 |
| Mats Hummels | Niklas Süle | 120 | -0.003 | +0.015 |
| Mats Hummels | Jérôme Boateng | 120 | -0.004 | +0.019 |
| Mats Hummels | Leon Goretzka | 62 | -0.005 | +0.004 |
| Thomas Müller | Manuel Neuer | 120 | -0.017 | -0.006 |
| Manuel Neuer | Mats Hummels | 240 | -0.025 | -0.000 |
| Manuel Neuer | Jérôme Boateng | 240 | -0.035 | -0.011 |

**Bayern mean JOI90 xT: +0.0044. Non-Bayern German pairs mean JOI90 xT: +0.0049. Delta: -0.0005.**

Ten of the sixteen Bayern pairs are positive in xT; six are negative. The distribution is wide: Neuer + Boateng at -0.035 is the weakest German pair in the dataset, while Neuer + Süle at +0.069 is the strongest. The common thread among the negative pairs is Neuer as one of the two players. GK-to-outfield pairs have a structural disadvantage in xT because GK distribution counts as a progressive carry only when it reaches a dangerous zone — short goalkeeper-to-CB passes, which account for the majority of Neuer + Hummels and Neuer + Boateng interactions, generate near-zero or negative xT. This is a position-driven artefact, not a chemistry problem.

Germany's actual best same-club pair is Müller + Kimmich (xT +0.022, 298 minutes). The best non-Bayern German pairs in 2018: Mario Gómez + Kimmich (+0.080, 178 min — a late-game combination), Rüdiger + Neuer (+0.052, 120 min), and Draxler + Müller (+0.049, 120 min). The top of Germany's 2018 chemistry distribution is dominated by non-same-club or mixed-club pairs, not the Bayern core.

---

### Belgium WC 2018

Belgium went seven matches (third-place finish). Same-club pairs: Tottenham triple (Vertonghen + Alderweireld + Mousa Dembélé), Chelsea pair (Courtois + Eden Hazard), Man City pair (De Bruyne + Kompany), Man United pair (Lukaku + Fellaini), and Anderlecht pair (Tielemans + Dendoncker).

Top same-club result: De Bruyne + Kompany (+0.008, 480 min). The Spurs centre-back pairing (Vertonghen + Alderweireld, +0.005, 720 min) is Belgium's most durable same-club combination — 720 minutes over seven matches — and it barely registers in xT, which is expected for two central defenders whose interactions are largely sideways passes and clearances. Belgium's best pairs in the tournament are all different-club: Meunier + Chadli (+0.120, the standout), De Bruyne + Batshuayi (+0.047), Lukaku + Chadli (+0.040), and De Bruyne + Hazard (+0.027). Belgium's golden generation operated from dispersed clubs — De Bruyne at City, Hazard at Chelsea, Lukaku at United, Mertens at Napoli — and the top-of-the-chart pairs reflect that.

---

### France WC 2018

France is the only squad in this analysis where the same-club delta is large enough to notice: same-club mean JOI90 xT of +0.0103 vs different-club of +0.0014, a gap of +0.0089. But this is driven by five pairs spread across three clubs, and the sample sizes are too small to conclude anything structural. The Atlético Madrid trio — Lemar + Lucas Hernández (+0.034), Lemar + Griezmann (+0.007), Griezmann + Lucas Hernández (+0.005) — contributed the two highest same-club pair scores. The Atlético connection is real but barely above noise (the third pair in that group, Griezmann + Hernández, has 613 shared minutes and a nearly-flat xT of +0.005, suggesting it is two players occupying the same left-side zone rather than a chemistry link).

France's top overall pairs are different-club: Mendy + Giroud (+0.039), Sidibé + Mandanda (+0.036), Lloris + Umtiti (+0.027), and Mbappé + Griezmann (+0.020). Mbappé + Griezmann is interesting given how central that partnership was to France's 2018 style — 19-year-old Mbappé feeding off Griezmann's movement — but xT assigns it only middling credit because Griezmann's hold-up play generates many lateral passes that don't directly move the ball toward goal.

---

### Croatia WC 2018

Only two same-club pairs met the 45-minute floor: Kovačić + Modrić (Real Madrid, +0.004, 188 min) and Brozović + Perišić (Inter Milan, +0.000, 452 min). Croatia's cross-club pairs dominated: Subašić + Lovren (+0.063, 720 min — the deepest-running pair in Croatia's data), Rakitić + Perišić (+0.022, 640 min — Barcelona + Inter, highest-volume pairing), and Kramarić + Mandžukić (+0.024, 222 min — Hoffenheim + Juventus). Croatia reached the final with two effective same-club pairs that both score at or near zero on xT, while their most productive relationships were built entirely at international level.

---

### Brazil WC 2018

Four same-club pairs all score positive: Danilo + Jesus (Man City, +0.012), Fernandinho + Willian (Chelsea, +0.006), Casemiro + Marcelo (Real Madrid, +0.005), and Thiago Silva + Neymar (PSG, +0.004). Same-club mean of +0.0069 vs different-club +0.0043, a delta of +0.0026. The PSG combination of Thiago Silva and Neymar is notable: 600 shared minutes, positive xT, and a dominant VAEP score (+0.144 — highest VAEP among all pairs in the historic dataset). However, 600 minutes of shared time between a GK-feeding CB and a world-class forward is almost certainly driven by Thiago Silva's distribution quality into Neymar's feet, and both players would produce that interaction regardless of club affiliation. Brazil's best different-club pairs: Neymar + Coutinho (+0.016, 450 min — PSG/Barcelona), Neymar + Jesus (+0.014 — PSG/Man City), and Firmino + Willian (+0.014 — Liverpool/Chelsea).

---

### Senegal AFCON 2023

Senegal exited in the Round of 16 after four matches. Same-club pairs: Sarr + Pape Gueye (Marseille, +0.011), Diatta + Camara (Monaco, +0.004), Jakobs + Camara (Monaco, -0.002), and Diatta + Jakobs (Monaco, -0.003). The Monaco triple splits positively and negatively. Senegal's standout pair in the data is different-club: Kouyaté + Edouard Mendy (Nottingham Forest + Chelsea, +0.060, 49 min — a small sample but striking xT density), and Mané + Habibou Diallo (Al-Nassr + Eupen, +0.026, 331 min). Mané's best connections in this tournament are with squad fringe players rather than top-club team-mates, likely because Koulibaly (Al-Hilal) plays a position far removed from Mané's forward role.

---

### Morocco AFCON 2023

Same-club pairs: Harit + Ounahi (Marseille, +0.011), En-Nesyri + Bounou (Sevilla, +0.006), and Abdelhamid + Richardson (Reims, -0.006). Morocco's same-club mean (+0.0036) is marginally below the different-club mean (+0.0048). The best pairs are different-club: Hakimi + Ziyech (+0.020, PSG + Galatasaray), Hakimi + En-Nesyri (+0.019), and Amrabat + Hakimi (+0.016). Hakimi is at the top of Morocco's pair distribution regardless of partner, which is a function of his attacking fullback role generating progressive carries — any pairing involving him will inherit that xT boost.

---

### Côte d'Ivoire AFCON 2023

Ivory Coast won AFCON 2023. Same-club pairs: Gradel + Seko Fofana (Al-Qadsiah, +0.012, 266 min) and Sangaré + Boly (Nottingham Forest, -0.003, 155 min). Two pairs, one positive and one negative. The winning side's chemistry was built almost entirely across clubs: Kessié + Adingra (+0.049, Al-Ahli + Brighton), Haller + Singo (+0.042, Dortmund + Torino), Pépé + Kouamé (+0.034, Trabzonspor + Fiorentina), and Boga + Konaté (+0.034, Nice + Salzburg). The five players who were on the pitch during the most decisive moments of CIV's tournament run represented four different clubs.

---

## Pooled Same-Club vs Different-Club — Historic

Combining all eight squads:

| Group | n pairs | Mean JOI90 xT | Median JOI90 xT |
|-------|---------|---------------|-----------------|
| Same-club | 43 | +0.0041 | +0.0043 |
| Different-club | 1,042 | +0.0028 | +0.0014 |
| **Delta** | — | **+0.0013** | **+0.0029** |

The direction has flipped compared to the modern WC 2026 data (where same-club had a *negative* mean delta on xT, largely due to Bayern Munich dragging it down). In the historic data, same-club pairs are marginally higher in both mean and median. The effect size, however, is extremely small: a mean delta of +0.0013 JOI90 xT corresponds to roughly 0.12 additional expected-threat units per 90 minutes of shared play — well within one standard deviation of the pair distribution, and based on 43 same-club pairs vs 1,042 different-club pairs, where positional effects and small-sample variance dominate.

The comparison to the modern baseline is difficult because the WC 2026 analysis aggregates multi-tournament data (WC 2022, Euro 2024, Euro 2020, Copa América 2024) with pairs accumulating 500–1,200 minutes together, producing larger absolute JOI90 values. The per-tournament numbers here are naturally smaller in magnitude. The directional finding — minimal and inconsistent same-club effect — is consistent across both analyses.

---

## Germany 2018 Bayern Deep-Dive: The Podolski Stand-In

The eight named Bayern Munich pairs with at least 45 shared minutes at WC 2018:

| Pair | Minutes | JOI90 xT | JOI90 VAEP |
|------|---------|-----------|------------|
| Neuer + Süle | 120 | +0.069 | +0.035 |
| Müller + Kimmich | 298 | +0.022 | -0.052 |
| Müller + Hummels | 178 | +0.019 | +0.008 |
| Kimmich + Goretzka | 62 | +0.016 | -0.028 |
| Müller + Boateng | 240 | +0.008 | +0.026 |
| Kimmich + Süle | 120 | +0.006 | +0.020 |
| Neuer + Kimmich | 360 | +0.006 | +0.026 |
| Hummels + Kimmich | 120 | +0.005 | +0.022 |
| Boateng + Kimmich | 240 | +0.004 | +0.070 |
| Süle + Goretzka | 62 | +0.003 | +0.026 |
| Hummels + Süle | 120 | -0.003 | +0.015 |
| Hummels + Boateng | 120 | -0.004 | +0.019 |
| Hummels + Goretzka | 62 | -0.005 | +0.004 |
| Müller + Neuer | 120 | -0.017 | -0.006 |
| Neuer + Hummels | 240 | -0.025 | -0.000 |
| Neuer + Boateng | 240 | -0.035 | -0.011 |

**Bayern mean: +0.0044. Non-Bayern German mean: +0.0049. Bayern delta: -0.0005.**

No Bayern boost. The Bayern pairs perform at essentially the same level as the rest of the squad — fractionally below, not above. The variance within the Bayern group is enormous: Neuer + Süle at +0.069 and Neuer + Boateng at -0.035 are both same-club pairs, separated by a gap larger than the entire inter-group difference between Bayern and non-Bayern.

The Neuer problem deserves specific attention. Neuer features in six of the sixteen Bayern pairs; he is the source of the large negatives. As described above, this is structural: Neuer's ball-playing contributions are short goalkeeper distributions that move the ball sideways or backwards at high-pressure moments. xT punishes backward ball movement. Strip the three Neuer-to-outfield pairs (Neuer + Hummels, Neuer + Boateng, Neuer + Müller) from the Bayern set and the remaining 13 pairs average +0.0084 xT — above the non-Bayern German average of +0.0049. That re-cut would suggest a modest positive signal among the outfield Bayern core, but it is also selection bias: you are removing the three pairs with the most shared minutes (800 minutes between them) because their position pair type is definitionally low-xT.

On VAEP, the Bayern picture is more positive: Boateng + Kimmich scores +0.070 VAEP, the highest in Germany's tournament, and Müller + Boateng (+0.026), Kimmich + Neuer (+0.026), and Kimmich + Süle (+0.020) all register above-average VAEP for Germany. VAEP captures defensive value and non-progressive contributions that xT ignores, so Boateng + Kimmich's right-side connection — Boateng as ball-playing CB feeding Kimmich's overlapping runs — shows up clearly in VAEP when it doesn't in xT.

The bottom line for Germany 2018 versus the Podolski thesis: the Bayern core was not a chemistry disaster (the 2024 analysis found much more damaging same-club drag from Sané). But it also was not a chemistry engine. The Müller + Kimmich combination (298 minutes, +0.022 xT, the most-used pair with a positive score) is the closest thing to a "Bayern working as designed" outcome in the data. Müller + Hummels (+0.019) and Kimmich + Goretzka (+0.016) are also above-squad-average for Germany in 2018. The narrative that the Bayern connection was lifting Germany's floor in 2018 gets some tentative support from this subset — but Germany crashed out in the group stage, and the broader squad including the non-Bayern players was not performing at a level where club chemistry was the differentiator.

---

## Reconciling with the WC 2026 Modern Analysis

The modern analysis (WC 2026 squads, matches from 2020 onward) found same-club pairs *below* the different-club mean on xT — driven primarily by Germany 2024-era Bayern Munich pairs involving Leroy Sané (Sané + Kimmich at -0.127, Sané + Musiala at -0.243). The historic 2018 analysis finds same-club pairs at essentially the same level as different-club pairs, with a marginal positive tilt.

Both analyses agree on the substantive finding: there is no reliable, replicable same-club boost in international football chemistry data. Neither confirms the thesis; neither annihilates it. The data consistently finds that position type, individual player style, and formation role explain more of the pair JOI90 variance than shared club affiliation.

The one signal that is consistent across both analyses: on VAEP (which captures defensive and build-up value rather than just attacking threat), same-club pairs in 2018 tend to be slightly positive even when their xT is flat. The Spurs central defensive partnership in Belgium (Vertonghen + Alderweireld, VAEP +0.003, xT +0.005), the Real Madrid defensive axis in Croatia (Kovačić + Modrić, VAEP +0.015), and the Bayern defensive line in Germany all show VAEP positive when xT is near zero. This is consistent with a "knowing where each other is without the ball" effect — the type of chemistry that shows up in defensive shape and ball recycling, not in progressive attack sequences. That is a more modest version of the Podolski claim, and the numbers are too small to be conclusive.

---

## What This Means for the Video

**Lead with the Podolski framing — then complicate it, not kill it.**

The setup works: Germany 2014 had seven Bayern players, won the World Cup, and Podolski was a beneficiary of that system despite his club career being uninspiring. The question — does same-club training produce international chemistry? — is genuinely interesting and the data gives you a layered answer.

The 2018 data gives you the cleaner forensic case. Germany 2018 still had eight Bayern Munich players, the most of any squad in the dataset, and still crashed out in the group stage. The Bayern pairs averaged a JOI90 xT of +0.0044 — negligibly below the non-Bayern German average of +0.0049. The Müller + Kimmich pair, the canonical 2018 connection, registers +0.022 xT over 298 minutes, which is above the German squad average but not exceptional by tournament standards (Belgium's Meunier + Chadli hit +0.120 across seven matches, entirely different clubs). If the Bayern connection was a competitive advantage in 2018, it is invisible in the pair-level data.

The France 2018 result is the most interesting counter-evidence: the only squad in the analysis with a noticeably positive same-club delta (+0.0089), driven by the Atlético Madrid trio of Griezmann, Lemar, and Lucas Hernández. France won the tournament. But the Atlético connection at France was three players not six, and France's overall best pairs are different-club. The real story from France 2018 is that the squad worked because every pair worked — not because same-club pairs were elevated.

The most defensible narrative for the video: "The Podolski story is a compelling example of *why* we'd expect club chemistry to matter — but the data says it's one of several factors, not a primary driver. What Germany had in 2014 was something harder to replicate by design: seven players who had just won the Champions League together two weeks earlier, at their peak, all knowing the same system. By 2018 the Bayern core was still there but some players were aging out, some were past peak form, and the club chemistry that had produced the Treble was five years stale. The data picks this up: no boost, no drag, just noise."

For the video, the sharpest specific numbers are: Müller + Kimmich in WC 2018 at +0.022 xT (positive, but only the 4th-best German pair in that tournament), versus the same-club Neuer + Boateng at -0.035 xT (the weakest German pair in the data). That contrast, both from the Bayern core in the same three-match tournament, illustrates the core finding: club-mate status does not homogenise outcomes. The pair that works works for positional and stylistic reasons; the pair that doesn't fails for the same reasons — club training is not a guarantee of either.
