"""Bootstrap squad YAML files for all 32 WC 2026 nations."""
from __future__ import annotations

from pathlib import Path
import yaml

OUT_DIR = Path("squads/wc2026")

QUALIFIERS: list[tuple[str, str, str, str]] = [
    ("USA", "United States",     "us", "#bf0d3e"),
    ("CAN", "Canada",             "ca", "#d52b1e"),
    ("MEX", "Mexico",             "mx", "#006847"),
    ("ARG", "Argentina",          "ar", "#75aadb"),
    ("BRA", "Brazil",             "br", "#fedf00"),
    ("URU", "Uruguay",            "uy", "#5db8de"),
    ("ECU", "Ecuador",            "ec", "#ffce00"),
    ("COL", "Colombia",           "co", "#fcd116"),
    ("PAR", "Paraguay",           "py", "#d52b1e"),
    ("FRA", "France",             "fr", "#0055a4"),
    ("ESP", "Spain",              "es", "#aa151b"),
    ("GER", "Germany",            "de", "#000000"),
    ("ENG", "England",            "gb-eng", "#fff"),
    ("POR", "Portugal",           "pt", "#006600"),
    ("NED", "Netherlands",        "nl", "#ff6600"),
    ("BEL", "Belgium",            "be", "#ed2939"),
    ("CRO", "Croatia",            "hr", "#171796"),
    ("ITA", "Italy",              "it", "#1c5ed6"),
    ("SUI", "Switzerland",        "ch", "#d52b1e"),
    ("DEN", "Denmark",            "dk", "#c8102e"),
    ("AUT", "Austria",            "at", "#ed2939"),
    ("MAR", "Morocco",            "ma", "#c1272d"),
    ("SEN", "Senegal",            "sn", "#00853f"),
    ("EGY", "Egypt",              "eg", "#ce1126"),
    ("TUN", "Tunisia",            "tn", "#e70013"),
    ("NGA", "Nigeria",            "ng", "#008751"),
    ("CIV", "Côte d'Ivoire",      "ci", "#ff8200"),
    ("JPN", "Japan",              "jp", "#bd0029"),
    ("KOR", "South Korea",        "kr", "#cd2e3a"),
    ("IRN", "Iran",               "ir", "#239f40"),
    ("AUS", "Australia",          "au", "#ffcd00"),
    ("KSA", "Saudi Arabia",       "sa", "#006c35"),
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
