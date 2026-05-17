"""Bootstrap squad YAML files for all 48 WC 2026 nations."""
from __future__ import annotations

from pathlib import Path
import yaml

OUT_DIR = Path("squads/wc2026")

# Full 48-team WC 2026 qualifier list (confirmed as of mid-May 2026)
# Hosts: USA, CAN, MEX (auto-qualified)
# AFC: 8 direct + 1 IC playoff (Iraq won IC playoff vs Bolivia Mar 31 2026)
# CAF: 9 direct + 1 via playoff route (DR Congo)
# CONCACAF: 3 hosts + 3 through qualifying (Curaçao, Haiti, Panama)
# CONMEBOL: 6 direct (ARG, BRA, COL, ECU, PAR, URU)
# OFC: New Zealand (direct)
# UEFA: 16 (AUT, BEL, BIH, CRO, CZE, ENG, FRA, GER, NED, NOR, POR, SCO, ESP, SWE, SUI, TUR)
# Italy did NOT qualify (lost UEFA playoff final to Bosnia & Herzegovina)
# Tunisia DID qualify (CAF direct)
# Notable new qualifiers: Curaçao, Jordan, Uzbekistan, Iraq (IC playoff), DR Congo
QUALIFIERS: list[tuple[str, str, str, str]] = [
    # Hosts
    ("USA", "United States",            "us",    "#bf0d3e"),
    ("CAN", "Canada",                   "ca",    "#d52b1e"),
    ("MEX", "Mexico",                   "mx",    "#006847"),
    # CONMEBOL (6)
    ("ARG", "Argentina",                "ar",    "#75aadb"),
    ("BRA", "Brazil",                   "br",    "#fedf00"),
    ("URU", "Uruguay",                  "uy",    "#5db8de"),
    ("ECU", "Ecuador",                  "ec",    "#ffce00"),
    ("COL", "Colombia",                 "co",    "#fcd116"),
    ("PAR", "Paraguay",                 "py",    "#d52b1e"),
    # UEFA (16)
    ("FRA", "France",                   "fr",    "#0055a4"),
    ("ESP", "Spain",                    "es",    "#aa151b"),
    ("GER", "Germany",                  "de",    "#000000"),
    ("ENG", "England",                  "gb-eng","#fff"),
    ("POR", "Portugal",                 "pt",    "#006600"),
    ("NED", "Netherlands",              "nl",    "#ff6600"),
    ("BEL", "Belgium",                  "be",    "#ed2939"),
    ("CRO", "Croatia",                  "hr",    "#171796"),
    ("SUI", "Switzerland",              "ch",    "#d52b1e"),
    ("AUT", "Austria",                  "at",    "#ed2939"),
    ("NOR", "Norway",                   "no",    "#ef2b2d"),
    ("SCO", "Scotland",                 "gb-sct","#003da5"),
    ("SWE", "Sweden",                   "se",    "#006aa7"),
    ("TUR", "Turkey",                   "tr",    "#e30a17"),
    ("CZE", "Czech Republic",           "cz",    "#d7141a"),
    ("BIH", "Bosnia and Herzegovina",   "ba",    "#002395"),
    # CAF (9 direct + DR Congo via CAF playoff = 10 total)
    ("MAR", "Morocco",                  "ma",    "#c1272d"),
    ("SEN", "Senegal",                  "sn",    "#00853f"),
    ("EGY", "Egypt",                    "eg",    "#ce1126"),
    ("TUN", "Tunisia",                  "tn",    "#e70013"),
    ("NGA", "Nigeria",                  "ng",    "#008751"),
    ("CIV", "Côte d'Ivoire",            "ci",    "#ff8200"),
    ("ALG", "Algeria",                  "dz",    "#006233"),
    ("GHA", "Ghana",                    "gh",    "#006b3f"),
    ("CPV", "Cape Verde",               "cv",    "#003893"),
    ("RSA", "South Africa",             "za",    "#007a4d"),
    ("COD", "DR Congo",                 "cd",    "#007fff"),
    # Note: Nigeria DID NOT qualify — lost CAF playoff to DR Congo
    # AFC (8 direct + Iraq via IC playoff = 9 total)
    ("JPN", "Japan",                    "jp",    "#bd0029"),
    ("KOR", "South Korea",              "kr",    "#cd2e3a"),
    ("IRN", "Iran",                     "ir",    "#239f40"),
    ("AUS", "Australia",                "au",    "#ffcd00"),
    ("KSA", "Saudi Arabia",             "sa",    "#006c35"),
    ("QAT", "Qatar",                    "qa",    "#8d1b3d"),
    ("JOR", "Jordan",                   "jo",    "#007a3d"),
    ("UZB", "Uzbekistan",               "uz",    "#1eb53a"),
    ("IRQ", "Iraq",                     "iq",    "#ce1126"),
    # CONCACAF (3 non-host: Curaçao, Haiti, Panama)
    ("CUR", "Curaçao",                  "cw",    "#003da5"),
    ("HAI", "Haiti",                    "ht",    "#00209f"),
    ("PAN", "Panama",                   "pa",    "#da121a"),
    # OFC (1)
    ("NZL", "New Zealand",              "nz",    "#000000"),
]


def write_stub(code: str, name: str, flag_iso: str, color: str) -> Path:
    target = OUT_DIR / f"{code}.yaml"
    if target.exists():
        return target
    data = {
        "nation": name,
        "nation_code": code,
        "flag_iso": flag_iso,
        "manager": "TBD",
        "formation": "4-3-3",
        "team_color": color,
        "players": [],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return target


def main() -> None:
    written = []
    for code, name, flag_iso, color in QUALIFIERS:
        p = write_stub(code, name, flag_iso, color)
        written.append(p)
    print(f"Stubs in {OUT_DIR}: {len(written)}")


if __name__ == "__main__":
    main()
