# Cross-Context Chemistry: The Podolski Archetype

## 1. Setup

**The question:** Some players are unmistakably better in one context than another. Lukas Podolski scored 49 goals in 130 international appearances for Germany — a strike rate and consistency that made him a national hero. At club level, his record across Arsenal, Galatasaray, Inter Milan, and Vissel Kobe was considerably more modest. The inverse case — a player whose club chemistry network is richly developed but dissolves at international level — is what this analysis tests in modern, measurable terms.

**The data ceiling:** Podolski himself is **not in any open dataset**. His career at Köln (pre-2009), Bayern Munich (2009–12), Arsenal (2012–15), Galatasaray (2015–17), Vissel Kobe, Górnik Zabrze, and his Germany years all predate the StatsBomb open data releases that cover club football. There is zero open-data overlap for him. The analysis therefore tests the Podolski archetype on modern players whose club seasons and international tournaments both fall within available StatsBomb open data.

**Available open competitions used in this analysis:**
- **International:** FIFA WC 2022 (43 matches · 64 matches), UEFA Euro 2024 (51 matches), UEFA Euro 2020 (51 matches), Copa América 2024 (32 matches), FIFA WC 2018 (64 matches), AFCON 2023 (52 matches)
- **Club:** Bundesliga 2023/24 (34 Bayern matches via StatsBomb partial release), La Liga 2017/18–2020/21 (36+34+33+35 matches), Ligue 1 2021/22 + 2022/23 (26+32 matches), MLS 2023 (6 Inter Miami matches)

Note: StatsBomb's open club releases are samples — the Bundesliga 2023/24 release covers 34 matches (primarily Bayer Leverkusen's title-winning season, not the full Bayern dataset). The La Liga releases are similarly partial. This is the open-data ceiling.

---

## 2. Methodology

**VAEP v2** was trained on the combined StatsBomb SPADL corpus: 1,211,875 actions from all international competitions plus the 8 club seasons above. Two XGBClassifiers (scores, concedes), N=10 lookahead gamestates, features: actiontype, result, start/end location.

- Train AUC scores: **0.8176**
- Train AUC concedes: **0.9328**

Both exceed the 0.70 sanity threshold.

**xT** is loaded from the existing `data/xt/xt.pkl` grid (fitted on the prior StatsBomb international corpus). It is reported alongside VAEP for reference; VAEP is the primary metric.

**Player-level per-90 production:** For each player in each competition context, we sum all VAEP values from their attributed actions and divide by minutes played / 90. Minutes are taken from the StatsBomb lineups API (position spell durations). A 45-minute floor is applied for international appearances; 90 minutes for club appearances.

**Caveats:**
- Opponent quality differs systematically: club opponents are top-division sides; international group-stage opponents range widely.
- Tactical roles often differ (e.g., Kimmich: RB at Bayern, CM for Germany).
- Small samples dominate international appearances: most players have 3–7 matches per tournament.
- StatsBomb open club data is a sample, not a full season. Bayern's 2023/24 Bundesliga representation is minimal in this release; the large Leverkusen match set means Bayern players appear infrequently.

---

## 3. Headline: Bayern 2023/24 → Germany Euro 2024

Within-season, overlapping players — the cleanest test.

| Player | Bayern 23/24 per-90 VAEP | Germany Euro 24 per-90 VAEP | Delta |
|---|---|---|---|
| Joshua Kimmich | 0.131 | 0.240 | +0.109 |
| Thomas Müller | 0.148 | — | — |
| Leroy Sané | 0.096 | -0.013 | -0.108 |
| Jamal Musiala | 0.042 | 0.762 | +0.720 |
| Leon Goretzka | 0.396 | — | — |
| Aleksandar Pavlović | 0.057 | — | — |

### Joshua Kimmich

**Club (Bayern)** — minutes-weighted avg per-90 VAEP: **0.131**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Bayern Munich Bundesliga 2023/24 | 91 | 2 | 0.131 | 0.149 | 0 | 0 |

**Country (Germany)** — minutes-weighted avg per-90 VAEP: **0.206**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Germany Euro 2020 | 364 | 4 | 0.151 | 0.087 | 0 | 1 |
| Germany Euro 2024 | 450 | 5 | 0.240 | 0.132 | 0 | 2 |
| Germany WC 2018 | 270 | 3 | 0.316 | 0.155 | 0 | 0 |
| Germany WC 2022 | 270 | 3 | 0.116 | 0.130 | 0 | 0 |

**Delta (country minus club): +0.075**

_Joshua Kimmich's per-90 VAEP at Bayern with Leroy Sané, Jamal Musiala, and Noussair Mazraoui averages 0.131; at Germany it averages 0.206 — a delta of +0.075 (rose)._
### Thomas Müller

**Club (Bayern)** — minutes-weighted avg per-90 VAEP: **0.148**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Bayern Munich Bundesliga 2023/24 | 91 | 2 | 0.148 | 0.044 | 0 | 0 |

**Country (Germany)** — minutes-weighted avg per-90 VAEP: **-0.121**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Germany Euro 2020 | 298 | 4 | -0.085 | 0.025 | 0 | 0 |
| Germany WC 2018 | 208 | 3 | -0.201 | 0.042 | 0 | 0 |
| Germany WC 2022 | 202 | 3 | -0.094 | 0.009 | 0 | 0 |

**Delta (country minus club): -0.269**

_Thomas Müller's per-90 VAEP at Bayern with Leroy Sané, Jamal Musiala, and Noussair Mazraoui averages 0.148; at Germany it averages -0.121 — a delta of -0.269 (dropped)._
### Leroy Sané

**Club (Bayern)** — minutes-weighted avg per-90 VAEP: **0.096**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Bayern Munich Bundesliga 2023/24 | 180 | 2 | 0.096 | 0.037 | 0 | 0 |

**Country (Germany)** — minutes-weighted avg per-90 VAEP: **0.027**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Germany Euro 2020 | 117 | 4 | -0.171 | 0.018 | 0 | 0 |
| Germany Euro 2024 | 207 | 5 | -0.013 | 0.038 | 0 | 0 |
| Germany WC 2022 | 111 | 2 | 0.310 | 0.184 | 0 | 1 |

**Delta (country minus club): -0.068**

_Leroy Sané's per-90 VAEP at Bayern with Jean-Eric Maxim Choupo-Moting, Matthijs de Ligt, and Noussair Mazraoui averages 0.096; at Germany it averages 0.027 — a delta of -0.068 (dropped)._
### Jamal Musiala

**Club (Bayern)** — minutes-weighted avg per-90 VAEP: **0.042**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Bayern Munich Bundesliga 2023/24 | 116 | 2 | 0.042 | 0.088 | 0 | 0 |

**Country (Germany)** — minutes-weighted avg per-90 VAEP: **0.527**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Germany Euro 2024 | 390 | 5 | 0.762 | 0.028 | 3 | 0 |
| Germany WC 2022 | 258 | 3 | 0.173 | 0.078 | 0 | 0 |

**Delta (country minus club): +0.486**

_Jamal Musiala's per-90 VAEP at Bayern with Leroy Sané, Matthijs de Ligt, and Mathys Tel averages 0.042; at Germany it averages 0.527 — a delta of +0.486 (rose)._
### Leon Goretzka

**Club (Bayern)** — minutes-weighted avg per-90 VAEP: **0.396**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Bayern Munich Bundesliga 2023/24 | 160 | 2 | 0.396 | 0.022 | 1 | 0 |

**Country (Germany)** — minutes-weighted avg per-90 VAEP: **0.204**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Germany Euro 2020 | 141 | 3 | 0.719 | 0.042 | 1 | 0 |
| Germany WC 2018 | 62 | 1 | -0.300 | 0.020 | 0 | 0 |
| Germany WC 2022 | 158 | 3 | -0.057 | 0.011 | 0 | 0 |

**Delta (country minus club): -0.193**

_Leon Goretzka's per-90 VAEP at Bayern with Leroy Sané, Jamal Musiala, and Noussair Mazraoui averages 0.396; at Germany it averages 0.204 — a delta of -0.192 (dropped)._
### Aleksandar Pavlović

**Club (Bayern)** — minutes-weighted avg per-90 VAEP: **0.057**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Bayern Munich Bundesliga 2023/24 | 59 | 1 | 0.057 | 0.008 | 0 | 0 |

**Country (Germany)** — minutes-weighted avg per-90 VAEP: **0.000**

_No qualifying appearances._

**Delta (country minus club): -0.057**

_Aleksandar Pavlović's club per-90 VAEP at Bayern averages 0.057 across 1 season(s); no qualifying Germany appearances found in the open international dataset._

### Robert Lewandowski (Bayern → Poland)

**Club (Bayern)** — minutes-weighted avg per-90 VAEP: **0.000**

_No qualifying appearances._

**Country (Poland)** — minutes-weighted avg per-90 VAEP: **0.181**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Poland Euro 2020 | 270 | 3 | 0.799 | 0.008 | 3 | 0 |
| Poland Euro 2024 | 121 | 2 | -0.009 | -0.020 | 1 | 0 |
| Poland WC 2018 | 270 | 3 | -0.095 | -0.004 | 0 | 0 |
| Poland WC 2022 | 360 | 4 | -0.013 | 0.001 | 2 | 1 |

**Delta (country minus club): +0.181**

_Robert Lewandowski has no qualifying club appearances in the available open data for Bayern; country context only._

---

## 4. Real Madrid Axis

### Luka Modrić

**Club (Real Madrid)** — minutes-weighted avg per-90 VAEP: **0.448**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Real Madrid La Liga 2017/18 | 180 | 2 | 0.265 | 0.046 | 0 | 0 |
| Real Madrid La Liga 2018/19 | 90 | 1 | 0.167 | 0.073 | 0 | 0 |
| Real Madrid La Liga 2020/21 | 112 | 2 | 0.970 | 0.018 | 1 | 0 |

**Country (Croatia)** — minutes-weighted avg per-90 VAEP: **0.104**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Croatia Euro 2020 | 383 | 4 | 0.394 | 0.022 | 1 | 1 |
| Croatia Euro 2024 | 235 | 3 | 0.149 | 0.032 | 1 | 0 |
| Croatia WC 2018 | 656 | 7 | 0.011 | -0.007 | 4 | 1 |
| Croatia WC 2022 | 644 | 7 | 0.011 | 0.013 | 1 | 0 |

**Delta (country minus club): -0.344**

_Luka Modrić's per-90 VAEP at Real Madrid with Carlos Henrique Casimiro, Karim Benzema, and Toni Kroos averages 0.448; at Croatia it averages 0.104 — a delta of -0.344 (dropped)._
### Toni Kroos

**Club (Real Madrid)** — minutes-weighted avg per-90 VAEP: **0.178**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Real Madrid La Liga 2017/18 | 174 | 2 | 0.221 | 0.059 | 0 | 0 |
| Real Madrid La Liga 2018/19 | 55 | 1 | 0.230 | 0.050 | 0 | 0 |
| Real Madrid La Liga 2019/20 | 180 | 2 | -0.113 | 0.058 | 0 | 0 |
| Real Madrid La Liga 2020/21 | 162 | 2 | 0.437 | -0.013 | 1 | 0 |

**Country (Germany)** — minutes-weighted avg per-90 VAEP: **0.316**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Germany Euro 2020 | 360 | 4 | 0.245 | 0.051 | 0 | 0 |
| Germany Euro 2024 | 440 | 5 | 0.137 | 0.027 | 0 | 0 |
| Germany WC 2018 | 270 | 3 | 0.701 | 0.096 | 1 | 0 |

**Delta (country minus club): +0.138**

_Toni Kroos's per-90 VAEP at Real Madrid with Luka Modrić, Karim Benzema, and Carlos Henrique Casimiro averages 0.178; at Germany it averages 0.316 — a delta of +0.138 (rose)._
### Raphaël Varane

**Club (Real Madrid)** — minutes-weighted avg per-90 VAEP: **-0.078**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Real Madrid La Liga 2017/18 | 180 | 2 | -0.054 | 0.062 | 0 | 0 |
| Real Madrid La Liga 2018/19 | 90 | 1 | 0.034 | 0.053 | 0 | 0 |
| Real Madrid La Liga 2019/20 | 180 | 2 | -0.033 | 0.034 | 0 | 0 |
| Real Madrid La Liga 2020/21 | 90 | 1 | -0.329 | -0.158 | 0 | 0 |

**Country (France)** — minutes-weighted avg per-90 VAEP: **0.125**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| France Euro 2020 | 360 | 4 | 0.064 | 0.051 | 0 | 0 |
| France WC 2018 | 630 | 7 | 0.231 | 0.022 | 1 | 0 |
| France WC 2022 | 519 | 6 | 0.038 | 0.025 | 0 | 0 |

**Delta (country minus club): +0.203**

_Raphaël Varane's per-90 VAEP at Real Madrid with Toni Kroos, Sergio Ramos García, and Karim Benzema averages -0.078; at France it averages 0.125 — a delta of +0.203 (rose)._
### Karim Benzema

**Club (Real Madrid)** — minutes-weighted avg per-90 VAEP: **-0.014**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Real Madrid La Liga 2017/18 | 155 | 2 | -0.216 | 0.029 | 0 | 1 |
| Real Madrid La Liga 2018/19 | 90 | 1 | -0.020 | 0.002 | 0 | 0 |
| Real Madrid La Liga 2019/20 | 181 | 2 | -0.314 | -0.002 | 0 | 0 |
| Real Madrid La Liga 2020/21 | 162 | 2 | 0.519 | 0.039 | 1 | 0 |

**Country (France)** — minutes-weighted avg per-90 VAEP: **0.789**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| France Euro 2020 | 348 | 4 | 0.789 | 0.013 | 4 | 0 |

**Delta (country minus club): +0.802**

_Karim Benzema's per-90 VAEP at Real Madrid with Carlos Henrique Casimiro, Toni Kroos, and Luka Modrić averages -0.014; at France it averages 0.789 — a delta of +0.802 (rose)._
---

## 5. Barcelona Axis

### Lionel Andrés Messi Cuccittini

**Club (Barcelona)** — minutes-weighted avg per-90 VAEP: **1.008**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Barcelona La Liga 2017/18 | 2998 | 36 | 1.086 | 0.124 | 34 | 8 |
| Barcelona La Liga 2018/19 | 2714 | 34 | 1.203 | 0.134 | 36 | 7 |
| Barcelona La Liga 2019/20 | 2881 | 33 | 0.784 | 0.116 | 25 | 14 |
| Barcelona La Liga 2020/21 | 3023 | 35 | 0.971 | 0.122 | 30 | 6 |

**Country (Argentina)** — minutes-weighted avg per-90 VAEP: **0.262**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Argentina Copa 2024 | 426 | 5 | 0.201 | 0.085 | 1 | 1 |
| Argentina WC 2018 | 360 | 4 | 0.286 | 0.084 | 1 | 1 |
| Argentina WC 2022 | 678 | 7 | 0.288 | 0.070 | 9 | 1 |

**Delta (country minus club): -0.746**

_Lionel Messi's per-90 VAEP at Barcelona with Marc-André ter Stegen, Sergio Busquets i Burgos, and Jordi Alba Ramos averages 1.008; at Argentina it averages 0.262 — a delta of -0.746 (dropped)._
### Luis Alberto Suárez Díaz

**Club (Barcelona)** — minutes-weighted avg per-90 VAEP: **0.398**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Barcelona La Liga 2017/18 | 2723 | 31 | 0.463 | 0.023 | 23 | 3 |
| Barcelona La Liga 2018/19 | 2650 | 31 | 0.307 | 0.029 | 16 | 3 |
| Barcelona La Liga 2019/20 | 1843 | 25 | 0.434 | 0.013 | 13 | 5 |

**Country (Uruguay)** — minutes-weighted avg per-90 VAEP: **0.154**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Uruguay Copa 2024 | 78 | 4 | 0.373 | 0.071 | 2 | 0 |
| Uruguay WC 2018 | 450 | 5 | 0.239 | 0.015 | 2 | 1 |
| Uruguay WC 2022 | 146 | 3 | -0.227 | 0.014 | 0 | 1 |

**Delta (country minus club): -0.244**

_Luis Suárez's per-90 VAEP at Barcelona with Lionel Andrés Messi Cuccittini, Marc-André ter Stegen, and Gerard Piqué Bernabéu averages 0.398; at Uruguay it averages 0.154 — a delta of -0.244 (dropped)._
### Sergio Busquets i Burgos

**Club (Barcelona)** — minutes-weighted avg per-90 VAEP: **0.122**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Barcelona La Liga 2017/18 | 2418 | 29 | 0.109 | -0.007 | 1 | 0 |
| Barcelona La Liga 2018/19 | 2541 | 33 | 0.082 | 0.030 | 0 | 0 |
| Barcelona La Liga 2019/20 | 2155 | 29 | 0.174 | 0.036 | 2 | 0 |
| Barcelona La Liga 2020/21 | 2332 | 33 | 0.133 | 0.043 | 0 | 2 |

**Country (Spain)** — minutes-weighted avg per-90 VAEP: **-0.069**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Spain Euro 2020 | 367 | 4 | -0.074 | 0.008 | 0 | 0 |
| Spain WC 2018 | 360 | 4 | 0.056 | 0.014 | 0 | 1 |
| Spain WC 2022 | 333 | 4 | -0.199 | 0.015 | 0 | 0 |

**Delta (country minus club): -0.192**

_Sergio Busquets's per-90 VAEP at Barcelona with Lionel Andrés Messi Cuccittini, Marc-André ter Stegen, and Jordi Alba Ramos averages 0.122; at Spain it averages -0.069 — a delta of -0.192 (dropped)._
### Jordi Alba Ramos

**Club (Barcelona)** — minutes-weighted avg per-90 VAEP: **0.183**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Barcelona La Liga 2017/18 | 2576 | 31 | 0.180 | 0.060 | 2 | 6 |
| Barcelona La Liga 2018/19 | 2786 | 33 | 0.163 | 0.065 | 2 | 2 |
| Barcelona La Liga 2019/20 | 1828 | 23 | 0.175 | 0.075 | 1 | 4 |
| Barcelona La Liga 2020/21 | 2892 | 33 | 0.211 | 0.108 | 3 | 4 |

**Country (Spain)** — minutes-weighted avg per-90 VAEP: **0.248**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Spain Euro 2020 | 463 | 6 | 0.369 | 0.146 | 0 | 1 |
| Spain WC 2018 | 360 | 4 | 0.094 | 0.088 | 0 | 0 |
| Spain WC 2022 | 264 | 4 | 0.245 | 0.139 | 0 | 2 |

**Delta (country minus club): +0.065**

_Jordi Alba's per-90 VAEP at Barcelona with Lionel Andrés Messi Cuccittini, Marc-André ter Stegen, and Sergio Busquets i Burgos averages 0.183; at Spain it averages 0.248 — a delta of +0.065 (rose)._
---

## 6. The Messi Multi-Club Case

Messi's footprint in the open data spans three club contexts: FC Barcelona (La Liga 2017–21), PSG (Ligue 1 2021/22 + 2022/23), and Inter Miami (MLS 2023). His Argentina contexts include Copa América 2021, WC 2022, and Copa América 2024 — tournaments Argentina won. This is the reverse of the Podolski archetype: a player whose country production should be *higher* than club, not lower.

### Lionel Andrés Messi Cuccittini

**Club (Barcelona)** — minutes-weighted avg per-90 VAEP: **1.008**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Barcelona La Liga 2017/18 | 2998 | 36 | 1.086 | 0.124 | 34 | 8 |
| Barcelona La Liga 2018/19 | 2714 | 34 | 1.203 | 0.134 | 36 | 7 |
| Barcelona La Liga 2019/20 | 2881 | 33 | 0.784 | 0.116 | 25 | 14 |
| Barcelona La Liga 2020/21 | 3023 | 35 | 0.971 | 0.122 | 30 | 6 |

**Country (Argentina)** — minutes-weighted avg per-90 VAEP: **0.262**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Argentina Copa 2024 | 426 | 5 | 0.201 | 0.085 | 1 | 1 |
| Argentina WC 2018 | 360 | 4 | 0.286 | 0.084 | 1 | 1 |
| Argentina WC 2022 | 678 | 7 | 0.288 | 0.070 | 9 | 1 |

**Delta (country minus club): -0.746**

_Lionel Messi's per-90 VAEP at Barcelona with Marc-André ter Stegen, Sergio Busquets i Burgos, and Jordi Alba Ramos averages 1.008; at Argentina it averages 0.262 — a delta of -0.746 (dropped)._
---

## 7. PSG Era

### Kylian Mbappé Lottin

**Club (PSG)** — minutes-weighted avg per-90 VAEP: **0.809**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Paris Saint-Germain Ligue 1 2021/22 | 1977 | 23 | 0.821 | 0.088 | 22 | 7 |
| Paris Saint-Germain Ligue 1 2022/23 | 2374 | 29 | 0.800 | 0.077 | 27 | 4 |

**Country (France)** — minutes-weighted avg per-90 VAEP: **0.447**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| France Euro 2020 | 361 | 4 | -0.357 | 0.054 | 0 | 0 |
| France Euro 2024 | 465 | 5 | 0.085 | 0.055 | 1 | 1 |
| France WC 2018 | 532 | 7 | 0.744 | 0.050 | 4 | 0 |
| France WC 2022 | 567 | 7 | 0.977 | 0.072 | 9 | 1 |

**Delta (country minus club): -0.363**

_Kylian Mbappé's per-90 VAEP at PSG with Lionel Andrés Messi Cuccittini, Danilo Luís Hélio Pereira, and Gianluigi Donnarumma averages 0.810; at France it averages 0.447 — a delta of -0.363 (dropped)._
### Neymar da Silva Santos Junior

**Club (PSG)** — minutes-weighted avg per-90 VAEP: **0.559**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Paris Saint-Germain Ligue 1 2021/22 | 1512 | 18 | 0.395 | 0.051 | 11 | 1 |
| Paris Saint-Germain Ligue 1 2022/23 | 1275 | 16 | 0.753 | 0.072 | 12 | 4 |

**Country (Brazil)** — minutes-weighted avg per-90 VAEP: **0.425**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Brazil WC 2018 | 450 | 5 | 0.432 | 0.120 | 2 | 1 |
| Brazil WC 2022 | 249 | 3 | 0.411 | 0.059 | 2 | 0 |

**Delta (country minus club): -0.134**

_Neymar's per-90 VAEP at PSG with Lionel Andrés Messi Cuccittini, Danilo Luís Hélio Pereira, and Gianluigi Donnarumma averages 0.559; at Brazil it averages 0.425 — a delta of -0.134 (dropped)._
### Achraf Hakimi Mouh

**Club (PSG)** — minutes-weighted avg per-90 VAEP: **0.281**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Paris Saint-Germain Ligue 1 2021/22 | 1710 | 22 | 0.202 | 0.077 | 1 | 3 |
| Paris Saint-Germain Ligue 1 2022/23 | 1920 | 23 | 0.351 | 0.072 | 5 | 1 |

**Country (Morocco)** — minutes-weighted avg per-90 VAEP: **0.113**

| Context | Minutes | Matches | Per90 VAEP | Per90 xT | Goals | Assists |
|---|---|---|---|---|---|---|
| Morocco AFCON 2023 | 360 | 4 | 0.148 | 0.088 | 1 | 1 |
| Morocco WC 2018 | 270 | 3 | 0.116 | 0.013 | 0 | 0 |
| Morocco WC 2022 | 632 | 7 | 0.093 | 0.040 | 1 | 0 |

**Delta (country minus club): -0.168**

_Achraf Hakimi's per-90 VAEP at PSG with Lionel Andrés Messi Cuccittini, Gianluigi Donnarumma, and Marcos Aoás Corrêa averages 0.281; at Morocco it averages 0.113 — a delta of -0.168 (dropped)._
---

## 8. Counter-Examples

Players whose country per-90 VAEP is comparable or higher than club — these push back against the Podolski thesis:

- **Joshua Kimmich** (Bayern → Germany): club avg 0.131, country avg 0.206, delta +0.075
- **Jamal Musiala** (Bayern → Germany): club avg 0.042, country avg 0.527, delta +0.486
- **Toni Kroos** (Real Madrid → Germany): club avg 0.178, country avg 0.316, delta +0.138
- **Raphaël Varane** (Real Madrid → France): club avg -0.078, country avg 0.125, delta +0.203
- **Karim Benzema** (Real Madrid → France): club avg -0.014, country avg 0.789, delta +0.802
- **Jordi Alba Ramos** (Barcelona → Spain): club avg 0.183, country avg 0.248, delta +0.065
- **Antoine Griezmann** (Atlético → France): club avg 0.065, country avg 0.141, delta +0.076

---

## 9. Verdict

Of 17 players with qualifying appearances in both club and country contexts:
- **10** showed a meaningful drop in per-90 VAEP at international level (delta < -0.010)
- **7** showed a meaningful rise at international level (delta > +0.010)
- **0** held roughly steady

The chemistry-network claim receives partial but not uniform support. The data is split — as with the Wyscout analysis. The open-data ceiling is the dominant limiting factor: most players have sparse club coverage, and within the available sample, variance is high enough that individual per-90 estimates should be treated as directional indicators rather than reliable effect sizes.

---

## 10. Caveats and Ceiling

**Podolski is uncoverable:** No open event data exists for Lukas Podolski at any club. The analysis tests his archetype only.

**Open data is a sample:** StatsBomb's open club releases cover partial seasons. The Bundesliga 2023/24 data emphasizes Bayer Leverkusen; Bayern Munich players have limited match coverage in this release.

**Small international samples:** Most players appear in 3–7 matches per international tournament, generating per-90 estimates with wide confidence intervals.

**No causal identification:** All comparisons are observational. Opponent quality, tactical role, age, and fitness all confound the club-vs-country delta.

**VAEP scale:** VAEP v2 is trained on a mixed StatsBomb corpus. Absolute values are small; relative rankings are more reliable than absolute magnitudes.

---

_Data: StatsBomb Open Data (CC BY-SA 4.0). VAEP: Bransen & Van Haaren 2020. xT: Singh 2019. Pipeline: socceraction._
