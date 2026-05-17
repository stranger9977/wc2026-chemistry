"""Best-effort headshot fetch from Wikipedia Commons.

For each player in any squads/wc2026/*.yaml, query Wikipedia for their page and
save the page image to assets/headshots/<slug>.jpg. Players without a hit are
skipped (left to the initials fallback). Respects rate limits.
"""
from __future__ import annotations

import argparse
import logging
import re
import time
import unicodedata
from pathlib import Path

import requests

from chemistry.squads import load_all_squads

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("headshots")

OUT = Path("assets/headshots")
WP = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "wc2026-chemistry/0.1 (https://github.com/stranger9977/wc2026-chemistry)"}


def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()


def find_image_url(player_name: str) -> str | None:
    params = {
        "action": "query", "format": "json", "prop": "pageimages",
        "piprop": "original", "titles": player_name, "redirects": 1,
    }
    r = requests.get(WP, params=params, headers=HEADERS, timeout=10)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    for page in pages.values():
        img = page.get("original", {}).get("source")
        if img:
            return img
    return None


def fetch_one(player_name: str) -> Path | None:
    target = OUT / f"{slugify(player_name)}.jpg"
    if target.exists():
        return target
    url = find_image_url(player_name)
    if not url:
        log.info("No image found: %s", player_name)
        return None
    img = requests.get(url, headers=HEADERS, timeout=15)
    if not img.ok:
        log.warning("Image fetch failed for %s: %s", player_name, img.status_code)
        return None
    OUT.mkdir(parents=True, exist_ok=True)
    target.write_bytes(img.content)
    log.info("Saved %s", target)
    time.sleep(0.5)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--squads", default="squads/wc2026", type=Path)
    args = parser.parse_args()
    squads = load_all_squads(args.squads)
    seen: set[str] = set()
    for sq in squads.values():
        for p in sq.players:
            if p.name in seen:
                continue
            seen.add(p.name)
            try:
                fetch_one(p.name)
            except Exception as exc:
                log.warning("Error fetching %s: %s", p.name, exc)


if __name__ == "__main__":
    main()
