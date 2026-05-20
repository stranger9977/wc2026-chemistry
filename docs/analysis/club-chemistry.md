# Club Chemistry: Do Same-Club Pairs Actually Connect Better for Their Countries?

The Podolski hypothesis — that shared club training creates measurable international chemistry — is intuitively compelling. Germany's 2014 core was functionally a Bayern Munich satellite team. But we cannot test that directly: our event-data window runs 2020 onward (WC 2022, Euro 2024, Euro 2020, Copa América 2024), and Podolski retired in 2017. What we can do is run the modern version on WC 2026 squads: do pairs who currently share a club show systematically higher JOI90 when representing their country?

**Data scope.** 23 nations had squad YAMLs with club-level detail (Argentina, Australia, Belgium, Brazil, Canada, Colombia, Croatia, Ecuador, England, France, Germany, Japan, South Korea, Morocco, Mexico, Netherlands, Portugal, Saudi Arabia, Senegal, Switzerland, Uruguay, USA). Player names in `chemistry.json` use full legal or display names; squad YAMLs use short names. Matching used display names from `players_by_id` with a three-candidate fallback (display name, pair display field, raw name). The 90-minute floor (one full match together) filtered out noise from cameos. After joining, 1,188 pairs across 23 nations were usable: 50 same-club pairs and 1,138 different-club pairs. Unmatched pairs are almost entirely players who featured in earlier tournaments but are absent from the WC 2026 predicted squad — not a systematic bias. Saudi Arabia had zero matchable pairs after normalization and is excluded.

---

## Top-Line Finding: No Consistent Same-Club Boost

| Metric | Same-club (n=50) | Different-club (n=1,138) | Delta (same minus diff) |
|--------|-----------------|--------------------------|-------------------------|
| JOI90 xT — mean | 0.041 | 0.074 | **-0.033** |
| JOI90 xT — median | 0.011 | 0.011 | 0.000 |
| JOI90 VAEP — mean | 0.033 | 0.017 | +0.016 |
| JOI90 xG — mean | 0.288 | 0.229 | **+0.059** |

The headline answer: **mixed, and the effect is small.** On the primary metric (xT), same-club pairs are actually slightly lower in mean, though the medians are nearly identical. The Welch t-test on xT yields t = -1.12 on approximately 54 degrees of freedom — nowhere near statistical significance. On xG, the same-club advantage is +0.059 with t = 2.83 (df ≈ 52), which is the one metric where there is a detectable signal. VAEP trends positive for same-club pairs but is also non-significant.

The xT mean gap inverts because of Germany. Bayern Munich contributes 21 of the 50 same-club pairs, and many of them — particularly anything involving Leroy Sané alongside other attackers, and Manuel Neuer alongside field players — score in negative xT territory. That drags the same-club mean down substantially. Once you understand that dynamic (see the Germany section below), the picture clarifies: shared club status is neither a reliable predictor nor a reliable anti-predictor of JOI90. Position and individual player tendencies overwhelm any club-cohort signal at the pair level.

The xG finding is the most interesting: same-club pairs are more likely to end up in scenarios producing expected goals. That is consistent with a "playing into the same spaces" story from shared training — but the sample size (50 pairs) is too small to conclude much, and the effect could be confounded by the Argentina Inter Miami pair (Messi + De Paul, 12 matches, 1,267 minutes together) which has enormous leverage.

---

## Per-Nation Breakdown

Eight nations had at least two same-club pairs with sufficient minutes. Sorted by number of same-club pairs:

| Nation | Top club | Same-club pairs (n) | Mean JOI90 xT (same) | Mean JOI90 xT (rest) | Delta |
|--------|----------|--------------------|-----------------------|----------------------|-------|
| Germany | Bayern Munich | 21 | -0.018 | +0.054 | -0.072 |
| Portugal | Bayern Munich* | 6 | -0.019 | +0.075 | -0.094 |
| England | Arsenal | 4 | +0.011 | +0.107 | -0.097 |
| Brazil | Real Madrid | 3 | +0.260 | +0.180 | +0.081 |
| Spain | Barcelona | 3 | +0.136 | +0.008 | +0.128 |
| Argentina | Inter Miami | 2 | +0.414 | +0.134 | +0.280 |
| Netherlands | Liverpool | 2 | +0.035 | +0.109 | -0.074 |
| USA | AC Milan | 2 | +0.072 | -0.039 | +0.111 |

*Portugal's same-club pairs are Palhinha + Guerreiro (both Bayern Munich) plus PSG's trio of Vitinha, João Neves, and Gonçalo Ramos.

**Germany.** The Bayern Munich contingent in the 2024-era Germany squad is large — Neuer, Kimmich, Gnabry, Müller, Sané, Musiala, Tah, Wirtz — but the pair-level results are erratic. Gnabry + Musiala (xT 0.371) and Müller + Musiala (0.310) are genuine highlights, but Leroy Sané is a consistent drag: Sané + Kimmich (-0.127), Sané + Musiala (-0.243), Sané + Gnabry (-0.324) are among the worst-performing pairs in the entire dataset for Germany. This is not a squad chemistry problem; it reflects Sané's high-entropy, ball-carrying style generating negative xT whenever the chain breaks down, regardless of whether his partner trains with him every day. Meanwhile, Germany's best-performing pairs in 2020+ international football — Rüdiger + Schlotterbeck (Real Madrid / Dortmund, xT 0.707), Havertz + Raum (Arsenal / Leipzig, xT 0.449), Gündogan + Schlotterbeck (Barca / Dortmund, xT 0.424) — are all different-club combinations.

**Spain.** Barcelona shows up positively: Pedri + Gavi (xT 0.437, 4 matches) is the best same-club pair in the Spanish data, and Pedri + Lamine Yamal (0.052) shows early promise. But Ferrán Torres + Pedri (-0.081) drags the Barca trio mean to 0.136 — directionally positive but driven by two of the three pairs rather than all three.

**Brazil.** Real Madrid's Vinícius Júnior + Rodrygo (0.453) and Militão + Vinícius Júnior (0.408) are strong, and the delta favors same-club pairs (+0.081). But note that the Seleção's off-the-charts performers are different-club: Neymar + Vinícius Júnior (xT 2.204 across 3 matches — top of the entire dataset), Neymar + Raphinha (0.994), and Neymar + Casemiro (0.951). Those three are all from the three matches Brazil played with Neymar back and Scaleta/Ancelotti fielding a full-strength lineup. Brazil is a nation where the superstar chemistry seems to override club affiliation entirely.

**Argentina.** The Inter Miami pair (Messi + De Paul, xT 0.731 over 1,267 minutes and 12 matches) is the single most-used same-club pairing in the dataset and has the second-highest mean among nations. But it is also arguable that Messi + De Paul would have produced this regardless of whether De Paul joined Inter Miami — De Paul has been Messi's on-field workhorse since Copa América 2021, and the club move followed the established international partnership rather than created it.

---

## "Modern Podolski" Candidates: High JOI90 Despite No Shared Club

These pairs played at least three matches and 270 shared minutes together at international level, and produced strong xT despite playing in different leagues. Filtered to pairs where no current-club connection exists.

**Ivan Perišić + Borna Sosa (Croatia) — xT 1.230, 5 matches, 449 min.** Hajduk Split vs Stuttgart. The best cross-club pair in the dataset. The left-flank combination between Perišić and Sosa — Sosa bombing forward, Perišić cutting in — has operated at a higher throughput than any same-club pairing in the data. Perišić is at the end of his career at Hajduk; Sosa has never played at the same club as him. Pure international chemistry.

**Memphis Depay + Daley Blind (Netherlands) — xT 0.873, 9 matches, 618 min.** Corinthians vs Girona. These two played at Ajax together in the early 2010s and you can see vestiges of that pattern even now, years after both left Amsterdam. A case where historical club chemistry — built at a club neither currently plays for — appears to carry into the present.

**Kylian Mbappé + Theo Hernandez (France) — xT 0.840, 11 matches, 1,228 min.** Real Madrid vs Al-Hilal. The most durable and high-volume combination in the French data. They have never been teammates, yet this left-flank pairing has functioned as France's connective tissue through Euro 2024 and the subsequent Nations League cycle. Hernandez playing as an attacking LB feeding Mbappé into the half-space is a shape that has emerged entirely at international level.

**Neymar + Vinícius Júnior (Brazil) — xT 2.204, 3 matches, 279 min.** Al-Hilal vs Real Madrid. The highest xT in the entire dataset by a wide margin. The caveat is only three matches (Neymar's injury-interrupted return), but the number is extreme enough that it warrants mention. These two have never shared a club. Their combination is presumably built on Brazil youth-system familiarity and their friendship dynamic — the type of thing Podolski/Klose had going, translated to modern Brazilian football.

**Teun Koopmeiners + Denzel Dumfries (Netherlands) — xT 0.821, 3 matches.** Juventus vs Inter Milan. Two players in direct domestic opposition who connect brilliantly for the national team. Rival club chemistry.

**Jules Koundé + Ousmane Dembélé (France) — xT 0.800, 10 matches, 709 min.** Barcelona vs PSG. These two were teammates at Barcelona until Dembélé left for PSG in 2023. Some of this connection may be residual club chemistry — they built their understanding at Barça — but the partnership has continued and strengthened even after the club separation. A hybrid of the historical-club and international chemistry effects.

**Pervis Estupiñán + Piero Hincapié (Ecuador) — xT 1.016, 3 matches, 360 min.** Brighton vs Bayer Leverkusen. Ecuador's left-side combination operating at an exceptional level despite playing in different leagues and under heavy event-data uncertainty (small sample for Ecuador overall).

**Raphaël Guerreiro + João Félix (Portugal) — xT 0.762, 5 matches, 316 min.** Bayern Munich vs Barcelona. Two attackers with radically different club trajectories who combine freely in the Portugal system.

---

## Club-Mate Exemplars: Shared Club Translating to the International Stage

These pairs share a club and are among the best-performing in the dataset (minimum 3 matches, 270 minutes).

**Lionel Messi + Rodrigo De Paul (Argentina / Inter Miami) — xT 0.731, 12 matches, 1,267 min.** The most reliable same-club pairing by volume. De Paul has been Messi's enforcer and link player since 2021; the Inter Miami stint formalized a partnership that already existed. With 12 matches logged together, this is the closest thing in the data to a proven, enduring club-into-country chemistry story.

**Vinícius Júnior + Rodrygo (Brazil / Real Madrid) — xT 0.453, VAEP 0.338, 4 matches.** The VAEP score here (0.338) is the highest of any pair in the dataset by a significant margin — meaning in terms of value-added actions, the Real Madrid left-and-right wing partnership for Brazil is extraordinary. The xT is slightly lower because Rodrygo tracks right and the combination sometimes involves deep sequences rather than direct progressive carries.

**Éder Militão + Vinícius Júnior (Brazil / Real Madrid) — xT 0.408, 6 matches, 522 min.** A defender-to-forward same-club link: Militão's progressive carrying from CB sets up Vinícius' runs. Six matches is a solid sample.

**Pedri + Gavi (Spain / Barcelona) — xT 0.437, 4 matches, 252 min.** The prototype Barcelona double-pivot combination at international level. Minutes are limited due to injury absences — notably Pedri's long-term knee issues — but when both are fit, Spain's midfield throughput visibly improves.

**Serge Gnabry + Jamal Musiala (Germany / Bayern Munich) — xT 0.371, 3 matches, 282 min.** The one same-club combination from Germany's Bayern core that shows a genuine lift. A generational handover pair — Gnabry as the experienced wide player, Musiala drifting inside.

**Mateo Kovačić + Joško Gvardiol (Croatia / Manchester City) — xT 0.175, 14 matches, 1,274 min.** Fourteen matches is the deepest dataset entry among same-club pairs. The xT is modest — defensive and box-to-box profiles generate less progressive xT — but consistency over 1,274 minutes is its own signal. The Man City defensive-to-midfield connection is real; it just operates further from goal.

**Christian Pulisic + Yunus Musah (USA / AC Milan) — xT 0.122, 6 matches, 506 min.** The American axis that has quietly become one of the more stable combinations in the USMNT setup. Both playing Serie A football together has clearly helped: their off-ball coordination and pressing patterns mirror what they do at club level.

**Declan Rice + Bukayo Saka (England / Arsenal) — xT 0.122, 14 matches, 1,140 min.** Same volume as Kovačić + Gvardiol. Rice as the deep recycler and Saka as the wide creator is a legitimate club-into-country chemistry case — they execute this combination 30+ times a week at the Emirates, and the England setup benefits from it.

---

## Caveats

**Sample size.** Fifty same-club pairs is a small n for drawing strong conclusions. Germany's 21 pairs alone constitute 42% of the same-club sample. A few outlier players (Sané, Neymar) exert disproportionate leverage on means.

**Club labels are May 2026, not historical.** The WC 2022 data in the JOI90 model covers matches from 2020 onward. Several players changed clubs significantly during that window — Tchouaméni moved from Monaco to Real Madrid, Koundé from Sevilla to Barcelona, De Paul from Udinese/Lazio to Inter Miami. The "same club" flag reflects where players are *now*, not where they were when these matches were played. This misclassification goes in both directions and likely attenuates any same-club signal.

**No proof of causation.** Even where same-club pairs show elevated JOI90, we cannot separate three effects: (a) shared club training creating chemistry, (b) managers pairing high-quality same-club teammates who would be high-chemistry regardless, and (c) selection feedback where pairs that work at international level influence club transfer decisions (the Messi/De Paul → Inter Miami example fits this story).

**JOI90 is a proximity and touch-sequence metric.** It measures how often two players are involved in the same advancing sequences, not whether they personally understand each other. A GK + CB pair will naturally accrue different JOI90 than two attacking midfielders. Position confounding is real and uncontrolled for in this analysis.

**Event-data window.** The 2020+ window misses the peak of the "Bayern for Germany" effect the hypothesis is about. We are measuring a different generation under partially different circumstances.

---

## What This Means for the Video

The Podolski framing still works — and in some ways the modern data makes it *more* interesting, not less. The story you can tell:

The 2014 Germany model assumed that cramming seven Bayern Munich players into a squad would produce built-in chemistry. Our 2020-2026 data on the current Germany squad tests that assumption with fresh eyes, and the answer is: it doesn't automatically work. Bayern teammates Leroy Sané and Joshua Kimmich generate a JOI90 xT of -0.127 across ten Germany matches. Sané and Musiala are at -0.243. Those are negative-chemistry pairs despite training together five days a week. Meanwhile, Rüdiger (Real Madrid) + Schlotterbeck (Dortmund) hit 0.707 — the best German pair in the data.

The modern Podolski story runs the other way too: Mbappé and Theo Hernandez have never shared a club, but their left-flank partnership is the most productive relationship in France's data — 11 matches, 1,228 minutes, xT 0.840. That connection was built entirely in international camps, not club training. The data points to something the video can say: club-based chemistry is a useful heuristic, but it is not a prerequisite, and it can actively mislead. What the 2014 Germany team really had was seven players who had just won the Champions League together and knew each other at peak form. The club connection was a proxy for something deeper — shared competitive context. That is much harder to replicate by design.

The one place same-club chemistry shows up cleanly: Argentina. Messi and De Paul have 12 international matches logged together, more than any other same-club pair in the dataset, and they are statistically the top same-club combination. But notably that partnership preceded the Inter Miami move — De Paul was Messi's anchor from Copa América 2021 onward when De Paul was still at Atlético. The club co-location followed the chemistry; it did not create it.
