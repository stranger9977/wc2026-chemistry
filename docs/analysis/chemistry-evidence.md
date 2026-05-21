# Chemistry evidence: does on-pitch chemistry explain the club-vs-country VAEP gap?

_Generated 2026-05-21_

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

We use `outputs/player_chemistry_v3.parquet` (2,164 player-context rows
across 109 dual-context players in this run) and
`outputs/career_familiarity.parquet` (24,213 pairs). For each player who
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

- 109 players in the corpus have both club and international rows
  available; 40 clear the 360-min-per-side bar.
- `delta_per90_vaep` distribution: mean **-0.003**, median **-0.042**,
  std **0.265**, min **-0.585**, max **+1.102**.
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

| Player | Intl team | Club | Intl mins | Club mins | Δ per90 VAEP | Δ JOI top5 | Δ eigen | Δ embed | Career fam |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Georginio Wijnaldum | Netherlands | Paris Saint-Germain | 360 | 919 | +0.613 | +0.320 | +0.101 | +0.798 | 3411 |
| Patrik Schick | Czech Republic | Bayer Leverkusen | 405 | 1064 | +0.603 | -0.697 | +0.154 | -3.688 | 5090 |
| Fabián Ruiz Peña | Spain | Paris Saint-Germain | 549 | 1568 | +0.313 | -0.035 | +0.138 | +0.297 | 5835 |
| Philippe Coutinho Correia | Brazil | Barcelona | 438 | 3604 | +0.283 | -0.239 | +0.268 | -2.203 | 5142 |
| Raphaël Varane | France | Real Madrid | 1508 | 360 | +0.168 | +0.058 | +0.154 | +0.261 | 16083 |
| Samuel Yves Umtiti | France | Barcelona | 540 | 4565 | +0.106 | +0.083 | +0.082 | -1.676 | 6805 |
| Yassine Bounou | Morocco | Girona | 900 | 360 | +0.102 | +0.020 | -0.062 | +0.493 | 9228 |
| Frenkie de Jong | Netherlands | Barcelona | 794 | 4614 | +0.092 | +0.102 | +0.110 | -0.240 | 8386 |
| Carlos Henrique Casimiro | Brazil | Real Madrid | 692 | 360 | +0.067 | +0.035 | -0.092 | +0.387 | 8247 |
| Jordi Alba Ramos | Spain | Barcelona | 1086 | 10400 | +0.059 | -0.069 | +0.065 | -2.949 | 33094 |
| Pedro González López | Spain | Barcelona | 1084 | 2245 | +0.045 | -0.313 | +0.047 | -1.458 | 18356 |
| Edmond Fayçal Tapsoba | Burkina Faso | Bayer Leverkusen | 360 | 2086 | +0.013 | -0.009 | +0.121 | -2.731 | 2964 |
| Piero Martín Hincapié Reyna | Ecuador | Bayer Leverkusen | 630 | 1484 | +0.002 | -0.094 | +0.197 | -2.860 | 6372 |
| Ivan Rakitić | Croatia | Barcelona | 576 | 6643 | -0.005 | -0.195 | +0.044 | -2.619 | 6430 |
| Marco Verratti | Italy | Paris Saint-Germain | 399 | 3352 | -0.012 | -0.039 | +0.020 | -1.056 | 7735 |

## 5. Top under-performers at country (intl − club, ≥360 min each side)

| Player | Intl team | Club | Intl mins | Club mins | Δ per90 VAEP | Δ JOI top5 | Δ eigen | Δ embed | Career fam |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Lionel Andrés Messi Cuccittini | Argentina | Barcelona | 1464 | 16979 | -0.585 | -0.474 | -0.072 | -1.301 | 16901 |
| Gerard Piqué Bernabéu | Spain | Barcelona | 360 | 9577 | -0.428 | -0.030 | -0.027 | -1.727 | 21734 |
| Ousmane Dembélé | France | Barcelona | 767 | 4162 | -0.364 | -0.327 | +0.053 | -2.045 | 11886 |
| Kylian Mbappé Lottin | France | Paris Saint-Germain | 1924 | 4351 | -0.363 | -0.655 | +0.006 | -0.694 | 22681 |
| Sergio Busquets i Burgos | Spain | Barcelona | 1060 | 9927 | -0.193 | +0.013 | -0.120 | -2.601 | 32880 |
| Javier Alejandro Mascherano | Argentina | Barcelona | 360 | 540 | -0.192 | +0.172 | +0.402 | -1.175 | 3580 |
| Presnel Kimpembe | France | Paris Saint-Germain | 360 | 2407 | -0.188 | -0.037 | +0.186 | -0.474 | 6002 |
| Achraf Hakimi Mouh | Morocco | Paris Saint-Germain | 1261 | 3629 | -0.168 | -0.139 | +0.191 | -1.938 | 12665 |
| Jan Oblak | Slovenia | Atlético Madrid | 360 | 720 | -0.160 | +0.025 | +0.101 | +0.696 | 3920 |
| Luis Alberto Suárez Díaz | Uruguay | Barcelona | 450 | 7215 | -0.159 | -0.476 | +0.052 | -1.847 | 5043 |
| Jonathan David | Canada | Lille | 644 | 360 | -0.158 | +0.169 | -0.035 | +0.159 | 6644 |
| Neymar da Silva Santos Junior | Brazil | Paris Saint-Germain | 699 | 2786 | -0.134 | -0.271 | +0.101 | -1.758 | 10132 |
| Tomáš Vaclík | Czech Republic | Sevilla | 450 | 360 | -0.119 | +0.007 | +0.089 | +0.499 | 4520 |
| Sergio Ramos García | Spain | Paris Saint-Germain | 360 | 3316 | -0.101 | -0.026 | +0.163 | -1.480 | 4077 |
| Thibaut Courtois | Belgium | Real Madrid | 1350 | 360 | -0.091 | +0.016 | +0.060 | +0.381 | 12627 |

## 6. Regression results

### All dual-context players (n=109)

| Term | Coef | SE | t |
|---|---:|---:|---:|
| intercept | +0.2825 | 0.2624 | +1.08 |
| delta_joi_top5_sum | +0.3793 | 0.1186 | +3.20 |
| delta_centrality_eigen | +0.2305 | 0.1979 | +1.16 |
| delta_embeddedness | -0.0315 | 0.0253 | -1.25 |
| log_intl_minutes | -0.0027 | 0.0408 | -0.07 |
| log_club_minutes | -0.0451 | 0.0262 | -1.72 |

_n=109, R²=0.178_

### ≥360 minutes per side (n=40)

| Term | Coef | SE | t |
|---|---:|---:|---:|
| intercept | +0.9029 | 0.5567 | +1.62 |
| delta_joi_top5_sum | +0.0584 | 0.1949 | +0.30 |
| delta_centrality_eigen | +0.2837 | 0.3487 | +0.81 |
| delta_embeddedness | -0.0398 | 0.0405 | -0.98 |
| log_intl_minutes | -0.0503 | 0.0753 | -0.67 |
| log_club_minutes | -0.0884 | 0.0425 | -2.08 |

_n=40, R²=0.204_

## 7. Interpretation

On the full 109-player dual-context sample the JOI top-5 delta carries a coefficient of +0.379 (t=+3.20, statistically meaningful); centrality and embeddedness deltas are not significant. Model R^2 is 0.18. When we tighten to the 40 players with at least 360 minutes on each side, the JOI coefficient collapses to +0.058 (t=+0.30, indistinguishable from zero) and the R^2 stays at 0.20. The honest read: there is a directional signal that shared output with your top partners moves with your own per-90 VAEP, but it is fragile to sample restriction and the chemistry signals together still explain only ~20% of the cross-context variance. Opportunity and role do most of the work; chemistry is a real but secondary factor in this corpus.

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

- **Lionel Andrés Messi Cuccittini** (Argentina vs Barcelona): under-performs for country by 0.59 VAEP/90; matching chemistry signals: JOI top-5, eigenvector centrality, embeddedness. Career familiarity with intl squadmates in the corpus: 16901 shared minutes.
- **Ousmane Dembélé** (France vs Barcelona): under-performs for country by 0.36 VAEP/90; matching chemistry signals: JOI top-5, embeddedness. Career familiarity with intl squadmates in the corpus: 11886 shared minutes.
- **Kylian Mbappé Lottin** (France vs Paris Saint-Germain): under-performs for country by 0.36 VAEP/90; matching chemistry signals: JOI top-5, embeddedness. Career familiarity with intl squadmates in the corpus: 22681 shared minutes.
- **Fabián Ruiz Peña** (Spain vs Paris Saint-Germain): over-performs for country by 0.31 VAEP/90; matching chemistry signals: eigenvector centrality, embeddedness. Career familiarity with intl squadmates in the corpus: 5835 shared minutes.
- **Sergio Busquets i Burgos** (Spain vs Barcelona): under-performs for country by 0.19 VAEP/90; matching chemistry signals: eigenvector centrality, embeddedness. Career familiarity with intl squadmates in the corpus: 32880 shared minutes.

These are candidates, not verdicts — the regression says we can't prove
chemistry is the cause in general, but for these specific players the story
hangs together, which is exactly what video storytelling needs.
