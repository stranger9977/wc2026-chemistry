"""Shorten StatsBomb registered player names to readable display names.

Order of preference for the display name:
  1. StatsBomb's `player_nickname` field (e.g., "Neymar", "Vinícius Júnior", "Casemiro")
  2. Heuristic from the registered name.

The heuristic:
  - Strip particles (da, de, do, dos, das, del, von, van, der, di, le, la)
  - Detect and preserve trailing "Junior"/"Júnior"/"Jr" as " Jr."
  - For 1 remaining token: use it
  - For 2 tokens: keep both
  - For 3 tokens: first + last
  - For 4+ tokens: first + (second-to-last) — captures Spanish paternal family name
"""
from __future__ import annotations

PARTICLES = {"da", "de", "do", "dos", "das", "del", "von", "van", "der",
             "di", "le", "la", "du", "de_la", "y"}
JR_TOKENS = {"junior", "júnior", "jr", "jr."}


def shorten_name(full_name: str, nickname: str | None = None) -> str:
    if nickname:
        return nickname.strip()
    if not full_name:
        return full_name

    raw_tokens = full_name.split()
    # Detect trailing Jr / Junior / Júnior
    is_jr = False
    if raw_tokens and raw_tokens[-1].lower().rstrip(".") in {"jr", "junior", "júnior"}:
        is_jr = True
        raw_tokens = raw_tokens[:-1]

    # Strip particles
    tokens = [t for t in raw_tokens if t.lower() not in PARTICLES]

    if not tokens:
        result = full_name
    elif len(tokens) == 1:
        result = tokens[0]
    elif len(tokens) == 2:
        result = " ".join(tokens)
    elif len(tokens) == 3:
        result = f"{tokens[0]} {tokens[-1]}"
    else:
        # 4+ tokens: Spanish paternal family name is typically at index -2
        result = f"{tokens[0]} {tokens[-2]}"

    if is_jr:
        result = f"{result} Jr."
    return result
