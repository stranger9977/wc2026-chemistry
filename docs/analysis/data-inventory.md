# Open Soccer Event Data: Inventory for the Podolski-Thesis Question

_Survey conducted May 2026. Research only — no pipeline changes._

---

## 1. TL;DR

Two freely available event-level datasets exist that could directly improve coverage for the club-vs-country chemistry question: the Wyscout open dataset (2017/18, already in use) and the StatsBomb open data repo (internationals plus multiple domestic seasons, already in use). No extension of the Wyscout open data beyond 2017/18 exists as a free download; the only new open event dataset of note since 2022 is the Bassek et al. Bundesliga set (7 matches, CC-BY 4.0), which is too small to add chemistry pairs but is the first public dataset that integrates tracking with official DFL events. The data ceiling for this question is substantial: multi-season, multi-player club coverage with consistent event schemas for the same cohort of elite players simply does not exist openly beyond what Wyscout and StatsBomb have published; any attempt to extend the analysis to 2018/19 or later seasons requires either a commercial Wyscout/Opta subscription or pivoting to aggregate stats from FBref/Understat, which cannot support pair-level JOI computations.

---

## 2. Sources We Already Use

**StatsBomb open data** (`github.com/statsbomb/open-data`): Full event-level JSON, ~3,400 events/match, consistent schema across all competitions. International coverage is comprehensive: WC 2018 and 2022 (with 360 data for 2022), Euro 2020 and 2024 (with 360 for 2024), Copa América 2024, AFCON 2023, women's tournaments. Club coverage is narrower: La Liga 2004–2021 is the deepest run; Bundesliga covers only 2015/16 and 2023/24; CL is mostly finals and late knockout stages. The most recent open release (April 2026) added UEFA Women's Euros 2025 (31 matches, 360 data). No WC 2026 data has been released yet (tournament ongoing at time of writing). License: StatsBomb Non-Commercial.

**Wyscout open dataset** (Pappalardo et al., 2019; `figshare.com/collections/Soccer_match_event_dataset/4415000`): Top 5 European leagues 2017/18 plus WC 2018 plus Euro 2016. ~1.94 million events across 1,941 matches. CC BY 4.0. This is the only open dataset that places the same players in both a full domestic-league season and an international tournament within a consistent schema, which is what makes cross-context JOI computation tractable.

---

## 3. Sources to Add (High Value)

Listed in priority order for the Podolski thesis.

### 3a. Transfermarkt Datasets (`github.com/dcaribou/transfermarkt-datasets`)

**URL:** `https://github.com/dcaribou/transfermarkt-datasets`  
**License:** CC0-1.0 (public domain).  
**Coverage:** 79,000+ matches, 37,000+ players, 1,800,000+ club and international appearances, 12 relational tables. Includes national team records (caps, goals, current national team ID) per player, refreshed weekly.  
**Schema:** Relational (competitions, clubs, players, games, appearances, transfers). Not event-level. Per-appearance rows with minutes played, goals, assists, cards.  
**Integration effort: Low.** Plain CSV download; standard pandas join. No scraping required.  
**What it adds for Podolski:** For every player in our JOI analysis, we can pull a career-arc table: which club, which season, how many minutes at each club vs. national team, career timeline. This directly operationalizes "was the player in a good club context or a peripheral one?" — the missing covariate in our current analysis. Podolski's club career (Arsenal periphery, Inter bench) vs. Germany centrality becomes a structured variable rather than narrative. Also enables identifying players across eras whose club/country usage differential is extreme, expanding the analysis beyond the 18 featured players.

---

### 3b. FBref via `soccerdata` / `worldfootballR` (derived stats only)

**URL:** `https://fbref.com` (scraped via `github.com/probberechts/soccerdata` or R `worldfootballR`)  
**License:** No official open license; scraping permitted under FBref ToS for non-commercial research. Rate-limit with delays; cache locally.  
**Coverage:** Big 5 leagues + Champions League + World Cup + Euros + Copa América + AFCON. xG/xA per season from ~2017/18 (earlier for some competitions). World Cup data from 1930; European Championship from 2000; international friendlies from 2014. Individual player season stats for all featured players (Messi, Modrić, Kimmich, Mbappé, De Bruyne, Hazard, Lewandowski, Neymar, Salah, etc.) available continuously from at least 2014/15.  
**Schema:** Aggregate per-90 stats (goals, assists, xG, xA, npxG, progressive passes, progressive carries, pressures, etc.) per player per competition per season. Not event-level. Cannot compute JOI90 from FBref.  
**Integration effort: Low-Medium.** `soccerdata` provides a Python-native scraper with session caching; `worldfootballR` provides the same in R. Requires patience (rate limits) but no schema translation.  
**What it adds for Podolski:** For the 18 featured players, we can pull parallel per-90 stats for club and national-team seasons and run a different version of the hypothesis: does xG/90 or xA/90 systematically differ between club and country contexts? This is weaker evidence than pair JOI (no co-occurrence structure), but it adds a corroborating or contradicting aggregate signal across more seasons (2014/15–2024/25) for the same players. Also provides the "good club context" operationalization: if a player's club xG/90 is high and country xG/90 is higher, that is a direct restatement of the Podolski thesis at the individual level without needing a partner.

---

### 3c. Bassek et al. Integrated Bundesliga Dataset (Sportec/DFL, 2025)

**URL:** `https://github.com/spoho-datascience/idsse-data`; data on Figshare DOI 10.6084/m9.figshare.28196177  
**License:** CC-BY 4.0. Attribution to DFL required.  
**Coverage:** 7 Bundesliga matches (2 first division, 5 second division; 2022/23 season). 207 players, 10 teams, 11,137 events, 1,002,644 tracking frames at 25 Hz.  
**Schema:** Three synchronized XML files per match: match metadata, event data (timestamped discrete actions with xG), position data (x/y coordinates per player and ball at 25 Hz). Full tracking integration via the `floodlight` Python package.  
**Integration effort: Medium.** XML format requires a new loader; `floodlight` handles it. No SPADL converter exists for this format yet; would need mapping from DFL event taxonomy to SPADL actions before JOI computation is possible.  
**What it adds for Podolski:** In isolation, 7 matches is too small for reliable JOI pair estimates. The value is methodological: this is the only open data with synchronized tracking + events at full resolution, enabling off-ball pair co-occurrence analysis (do two players move into coordinated positions even when not directly interacting on-ball?). As a proof-of-concept for extending JOI with spatial proximity scores, it could sharpen the methodology before applying it if larger Bundesliga data ever opens up.

---

## 4. Sources Investigated but Dropping

**Wyscout v2 / Pappalardo extension:** No public extension beyond the original 2017/18 + WC 2018 + Euro 2016 dataset has been released. The Figshare collection is at version 5, but no additional seasons have been added. Follow-on academic papers by Pappalardo et al. reference derived metrics, not new event data. Drop: no coverage beyond what we already have.

**SoccerNet (soccer-net.org):** Covers 500 broadcast videos from six European leagues (2014–2017). Action spotting only — three event classes in v1 (Goal, Yellow/Red Card, Substitution); 17 classes in v2 (adds Foul, Corner, Free Kick, Offside, etc.). Average one annotatable event every 6.9 minutes. Not dense enough for pair-level JOI computation, which requires every on-ball touch. SoccerNet-GSR adds tracking bounding-box positions (2.36 million annotations), but these are 30-second video clips, not full-match continuous data. Drop: wrong event granularity; no club/country overlap.

**SkillCorner open data (`github.com/SkillCorner/opendata`):** 10 Australian A-League matches (2024/25), broadcast tracking at 10 fps. No event data; no European top-flight coverage; no international matches. Drop: wrong league, no events, too small.

**Metrica Sports sample data (`github.com/metrica-sports/sample-data`):** 3 anonymised matches (teams and players unnamed), tracking + event data. Useful for methods development; useless for named-player chemistry analysis. Drop: anonymised, no player identity.

**Bransen & Van Haaren player-chemistry repo (`github.com/soccer-analytics-research/player-chemistry`):** The MITSSAC 2020 paper used a commercial Wyscout subscription covering 106 competitions, 361 seasons, 106,496 matches — far beyond the open dataset. The companion GitHub repo contains only precomputed top-5 pair rankings per season, in markdown, with no underlying data. No downloadable event dataset was released. Drop: precomputed summaries only, not raw events.

**SoccerNet tracking (2023/24 challenges):** Player bounding-box tracking from broadcast video, 200 30-second clips, for detection/re-identification challenges. Not full-match sequential data; no club identity metadata usable for JOI. Drop: wrong format for chemistry analysis.

**Fjelstul World Cup Database (`github.com/jfjelstul/worldcup`):** 27 relational datasets covering all men's World Cups 1930–2022 and all women's 1991–2019. Includes goals, penalties, bookings, substitutions. CC-BY-SA 4.0. Only four event types. Cannot compute JOI from goal/card data alone. Drop: event taxonomy too sparse; no club data.

**openfootball / football.db:** Match results and schedules only. No on-ball events. Drop: aggregate, no events.

**Hugging Face soccer datasets:** 93 datasets tagged "football" on HuggingFace as of May 2026. The largest relevant one is `aloobun/fbref_understat_combined` (110k rows, combined FBref + Understat aggregate stats). None contain raw event streams for JOI computation. Several are computer-vision oriented (player detection, segmentation). Drop: no event-level chemistry-relevant data.

**PFF FC public data (via kloppy):** Kloppy documentation lists PFF FC as having public tracking data accessible via Google Drive. Coverage, schema, and license not publicly documented in stable form as of May 2026; the kloppy entry is marked partial/in-progress. Drop until stable.

**Impect open data (via kloppy):** Kloppy lists Impect as having public event data. Impect covers 160+ competitions with ~2,500 events/match and Packing metrics. However, no stable public download URL or open license documentation was found for a free sample tier. Drop: license unclear, coverage unknown for public tier.

**Stats Perform / Opta:** Fully commercial. No open academic license. The Opta Forum research competition accepts submissions using Opta data, but data access is granted by individual arrangement, multi-day approval turnaround, and not reusable beyond the competition. Flag as requiring formal request but not immediately actionable.

**Hawk-Eye, Second Spectrum, Sportlogiq, InStat, Synergy:** All fully commercial. None have confirmed open data releases. Hawk-Eye's public presence is officiating technology; Second Spectrum produces analytics products for NBA/EPL teams with no academic-open tier. InStat and Synergy are subscription-only. Drop: no open data.

**Understat (standalone):** Shot-level xG data for Big 5 leagues from 2014/15. No structured player pair co-occurrence data; shot-level granularity cannot reconstruct SPADL sequences. Value is limited to per-player per-season xG/xA comparison, handled above under FBref.

**Transfermarkt (direct scraping):** The `dcaribou/transfermarkt-datasets` project (listed in section 3a above) provides a pre-built clean dataset under CC0, making direct scraping redundant. Drop as a standalone scraping task; use the pre-built dataset instead.

---

## 5. The Data Ceiling

The fundamental gap is **multi-season club event data for the same cohort of elite players**. The Wyscout open dataset gives us one season (2017/18) for five leagues. The StatsBomb open data gives us La Liga continuously from 2004–2021 but only isolated seasons for other leagues. Neither dataset makes it possible to trace, say, Kevin De Bruyne's evolution at Manchester City from 2015/16 through 2022/23 alongside his Belgium appearances over the same period.

Concretely:

- **Wyscout commercial API:** covers 106+ competitions and 360+ seasons with full event streams. Without it, we cannot replicate the Bransen/Van Haaren dataset. Pricing starts at ~$325/year for a basic plan; elite research access is by negotiation.
- **Opta / Stats Perform:** covers all major European leagues, all international tournaments, Champions League in full (not just finals). The most complete event data in the world; no open tier exists.
- **StatsBomb API:** pays-per-match subscription beyond the open repo. La Liga is our best-covered league in the open repo; for Bundesliga, Ligue 1, Serie A continuity past 2016, or Premier League beyond 2004 and 2016, the commercial API is required.

The specific questions this ceiling blocks:

1. **Multi-season trajectory:** Did Hazard's club/country JOI ratio shift over his Chelsea career (2012–2019) or stay consistently low? Cannot answer without 2012–2016 Ligue 1 / Premier League events.
2. **The actual 2014 Bayern-Germany thesis:** WC 2014 is not in any open dataset. The canonical case study is permanently dark.
3. **Any player outside the Big 5 + WC/Euro:** Lewandowski at Borussia Dortmund (2011–2014) vs. Poland is not in the open data.
4. **Post-2021 club seasons at scale:** Mbappé's PSG trajectory post-2017/18, De Bruyne's City peak years (2019/20–2022/23), Modrić's Real Madrid run beyond the 2018/19 CL — none open.

---

## 6. Recommendation

**Add to the pipeline next, in this order:**

1. **Transfermarkt Datasets (CC0-1.0):** One pip install and a join. Immediately unlocks career-arc context for every player in the analysis: which club, which season, minutes played, goals, national team caps. Zero integration risk.

2. **FBref via `soccerdata`:** Pull per-90 xG/xA for club and national-team seasons for the 18 featured players, 2015/16–2024/25. Run the Podolski hypothesis at the per-player-season level as a corroboration or contradiction of the event-based JOI findings. Medium effort (rate-limited scraping, caching), high narrative payoff.

3. **Stop investigating after that.** The next tier (Wyscout commercial, Opta, StatsBomb API) requires either money or multi-week approval processes. The marginal improvement in statistical power does not justify the overhead for a single-video project. The existing two open sources (Wyscout 2017/18 + StatsBomb 2020+) already cover three distinct methodological windows (historic Wyscout cross-context, modern StatsBomb internationals, and the WC 2026 squad structure). Adding FBref aggregate stats adds a fourth corroborating angle.

---

## 7. Coverage Matrix

Rows: key players from the Wyscout featured list. Columns: data sources. Cells: year ranges with event-level coverage (E) or aggregate stats only (A).

| Player | Club | Wyscout open (E) | StatsBomb open (E) | FBref/worldfootballR (A) | Transfermarkt-datasets (A) |
|---|---|---|---|---|---|
| Lionel Messi | Inter Miami (was Barça) | 2017/18 La Liga + WC 2018 + Euro 2016 | La Liga 2004–2021; Copa América 2024; WC 2022 | 2004/05–2024/25 | career |
| Luka Modrić | Real Madrid | 2017/18 La Liga + WC 2018 + Euro 2016 | La Liga 2004–2021; Euro 2020, 2024; WC 2022 | 2012/13–2024/25 | career |
| Joshua Kimmich | Bayern Munich | 2017/18 Bundesliga + WC 2018 | Bundesliga 2015/16 + 2023/24; Euro 2020, 2024; WC 2022 | 2015/16–2024/25 | career |
| Kylian Mbappé | Real Madrid | 2017/18 Ligue 1 + WC 2018 | Ligue 1 2015/16 + 2021/22 + 2022/23; WC 2018, 2022; Euro 2020, 2024 | 2016/17–2024/25 | career |
| Kevin De Bruyne | Manchester City | 2017/18 Premier League + WC 2018 + Euro 2016 | Premier League 2003/04 + 2015/16; Euro 2020, 2024; WC 2022 | 2013/14–2024/25 | career |
| Eden Hazard | Real Madrid (was Chelsea) | 2017/18 Premier League + WC 2018 + Euro 2016 | Premier League 2003/04 + 2015/16; Euro 2020 | 2009/10–2022/23 | career |
| Robert Lewandowski | Barcelona | 2017/18 Bundesliga + WC 2018 | Bundesliga 2015/16 + 2023/24; WC 2022 | 2010/11–2024/25 | career |
| Neymar | Al-Hilal (was PSG/Barça) | 2017/18 Ligue 1 + WC 2018 | La Liga 2004–2021; Copa América 2024; WC 2022 | 2009/10–2024/25 | career |
| Mohamed Salah | Liverpool | 2017/18 Premier League + WC 2018 | Premier League 2003/04 + 2015/16; AFCON 2023; WC 2022 | 2012/13–2024/25 | career |
| Paul Pogba | Manchester United (retired) | 2017/18 Premier League + WC 2018 + Euro 2016 | Premier League 2003/04 + 2015/16; WC 2018, 2022; Euro 2020 | 2011/12–2023/24 | career |
| Antoine Griezmann | Atletico Madrid | 2017/18 La Liga + WC 2018 + Euro 2016 | La Liga 2004–2021; WC 2018, 2022; Euro 2020, 2024 | 2009/10–2024/25 | career |
| N'Golo Kanté | Al-Ittihad (was Chelsea) | 2017/18 Premier League + WC 2018 + Euro 2016 | Premier League 2003/04 + 2015/16; WC 2018, 2022; Euro 2020 | 2012/13–2024/25 | career |
| Toni Kroos | Real Madrid (retired 2024) | 2017/18 La Liga + WC 2018 + Euro 2016 | La Liga 2004–2021; WC 2022; Euro 2024 | 2007/08–2023/24 | career |
| Romelu Lukaku | Roma (was Man Utd) | 2017/18 Premier League + WC 2018 + Euro 2016 | Premier League 2003/04 + 2015/16; WC 2022; Euro 2020, 2024 | 2009/10–2024/25 | career |
| Thomas Müller | Bayern Munich | 2017/18 Bundesliga + WC 2018 | Bundesliga 2015/16 + 2023/24; WC 2018, 2022; Euro 2020 | 2009/10–2024/25 | career |
| Jerome Boateng | PSV (was Bayern) | 2017/18 Bundesliga + WC 2018 | Bundesliga 2015/16 + 2023/24; WC 2022 | 2009/10–2024/25 | career |

**Key:** E = full event-level stream, suitable for JOI pair computation. A = aggregate season stats (goals, xG, xA, minutes) only, suitable for individual performance comparison but not pair chemistry.

**Notable gaps:**
- Premier League 2016/17–2022/23 for De Bruyne, Kanté, Salah, Hazard: dark. The commercial StatsBomb API would illuminate these.
- Bundesliga 2016/17–2022/23 for Kimmich, Lewandowski, Müller, Boateng: dark except for 2023/24 (StatsBomb open).
- Ligue 1 2016/17–2020/21 for Mbappé, Neymar, Pogba: dark.
- International appearances pre-2018 beyond Euro 2016 and WC 2018: dark for event-level.
- WC 2014 (the canonical Bayern-Germany case): permanently dark in open data.
