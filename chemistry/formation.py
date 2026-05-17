"""Map StatsBomb position labels to (x, y) coordinates on a 105×68 SPADL pitch."""
from __future__ import annotations

# Mapping from StatsBomb position name to short abbreviation for display
SB_TO_ABBREV: dict[str, str] = {
    "Goalkeeper": "GK",
    "Right Back": "RB",
    "Right Wing Back": "RWB",
    "Right Center Back": "RCB",
    "Center Back": "CB",
    "Left Center Back": "LCB",
    "Left Back": "LB",
    "Left Wing Back": "LWB",
    "Right Defensive Midfield": "RDM",
    "Center Defensive Midfield": "CDM",
    "Left Defensive Midfield": "LDM",
    "Right Midfield": "RM",
    "Right Center Midfield": "RCM",
    "Center Midfield": "CM",
    "Left Midfield": "LM",
    "Left Center Midfield": "LCM",
    "Right Attacking Midfield": "RAM",
    "Center Attacking Midfield": "CAM",
    "Left Attacking Midfield": "LAM",
    "Right Wing": "RW",
    "Left Wing": "LW",
    "Right Center Forward": "RCF",
    "Center Forward": "CF",
    "Left Center Forward": "LCF",
    "Striker": "ST",
    "Secondary Striker": "SS",
}


def short_position(sb_position: str | None) -> str | None:
    """Return the 2-3 letter abbreviation for a StatsBomb position name."""
    if not sb_position:
        return None
    return SB_TO_ABBREV.get(sb_position)


# Position buckets for coherent 4-3-3 XI selection
POSITION_BUCKETS: dict[str, list[str]] = {
    "GK":  ["Goalkeeper"],
    "RB":  ["Right Back", "Right Wing Back"],
    "RCB": ["Right Center Back"],
    "CB":  ["Center Back"],
    "LCB": ["Left Center Back"],
    "LB":  ["Left Back", "Left Wing Back"],
    "RDM": ["Right Defensive Midfield", "Center Defensive Midfield"],
    "CM":  ["Center Midfield", "Center Defensive Midfield", "Center Attacking Midfield"],
    "LDM": ["Left Defensive Midfield", "Center Defensive Midfield"],
    "RW":  ["Right Wing", "Right Center Forward", "Right Attacking Midfield"],
    "ST":  ["Striker", "Center Forward", "Secondary Striker"],
    "LW":  ["Left Wing", "Left Center Forward", "Left Attacking Midfield"],
}

# Preferred 4-3-3 slot order
PREFERRED_XI = ["GK", "RB", "RCB", "CB", "LCB", "LB", "RDM", "CM", "LDM", "RW", "ST", "LW"]

# Broad fallback buckets by role
_BROAD_BUCKETS: dict[str, list[str]] = {
    "GK":  ["Goalkeeper"],
    "RB":  ["Right Back", "Right Wing Back", "Right Center Back", "Center Back"],
    "RCB": ["Right Center Back", "Center Back", "Left Center Back"],
    "CB":  ["Center Back", "Right Center Back", "Left Center Back"],
    "LCB": ["Left Center Back", "Center Back", "Right Center Back"],
    "LB":  ["Left Back", "Left Wing Back", "Left Center Back", "Center Back"],
    "RDM": ["Right Defensive Midfield", "Center Defensive Midfield",
            "Left Defensive Midfield", "Right Midfield", "Center Midfield"],
    "CM":  ["Center Midfield", "Right Center Midfield", "Left Center Midfield",
            "Center Defensive Midfield", "Center Attacking Midfield"],
    "LDM": ["Left Defensive Midfield", "Center Defensive Midfield",
            "Right Defensive Midfield", "Left Midfield", "Center Midfield"],
    "RW":  ["Right Wing", "Right Center Forward", "Right Attacking Midfield",
            "Right Midfield", "Center Forward"],
    "ST":  ["Striker", "Center Forward", "Secondary Striker", "Right Center Forward",
            "Left Center Forward"],
    "LW":  ["Left Wing", "Left Center Forward", "Left Attacking Midfield",
            "Left Midfield", "Center Forward"],
}


def pick_pitch_xi(
    minutes_by_player: "pd.DataFrame",
    pos_mode: "pd.Series",
    effective_ids: "set[int]",
) -> list[int]:
    """Return up to 11 player_ids forming a coherent 4-3-3 starting XI.

    Guarantees exactly 1 GK (if one exists) and fills outfield slots in
    PREFERRED_XI order, falling back to broader role buckets when needed.

    Parameters
    ----------
    minutes_by_player : pd.DataFrame
        Index = player_id (int), sorted descending by total_mins.
    pos_mode : pd.Series
        Index = player_id, values = dominant StatsBomb position string.
    effective_ids : set[int]
        Pool of eligible players.
    """
    chosen: list[int] = []
    chosen_set: set[int] = set()

    def candidates_for_positions(positions: list[str]) -> list[int]:
        return [
            int(pid)
            for pid in minutes_by_player.index
            if int(pid) in effective_ids
            and int(pid) not in chosen_set
            and pos_mode.get(int(pid)) in positions
        ]

    for slot in PREFERRED_XI:
        # Try primary bucket first
        cands = candidates_for_positions(POSITION_BUCKETS[slot])
        if not cands:
            # Try broad fallback
            cands = candidates_for_positions(_BROAD_BUCKETS.get(slot, POSITION_BUCKETS[slot]))
        if cands:
            chosen.append(cands[0])
            chosen_set.add(cands[0])
        if len(chosen) >= 11:
            break

    # Top-up to 11 with most-played remaining (any position)
    for pid in minutes_by_player.index:
        if len(chosen) >= 11:
            break
        ipid = int(pid)
        if ipid not in chosen_set and ipid in effective_ids:
            chosen.append(ipid)
            chosen_set.add(ipid)

    return chosen[:11]

# StatsBomb position labels and their preferred pitch locations.
# Coordinates: x is along the long axis (own goal = 0, opp goal = 105).
#              y is across the pitch (left sideline = 0, right sideline = 68).
SB_POSITION_XY: dict[str, tuple[float, float]] = {
    "Goalkeeper": (6, 34),

    # Back line — keep separated horizontally
    "Right Back":               (22, 58),
    "Right Center Back":        (18, 44),
    "Center Back":              (18, 34),
    "Left Center Back":         (18, 24),
    "Left Back":                (22, 10),
    "Right Wing Back":          (35, 60),
    "Left Wing Back":           (35, 8),

    # Defensive midfield
    "Right Defensive Midfield":  (40, 44),
    "Center Defensive Midfield": (40, 34),
    "Left Defensive Midfield":   (40, 24),

    # Central midfield
    "Right Midfield":     (55, 50),
    "Center Midfield":    (55, 34),
    "Left Midfield":      (55, 18),
    "Right Center Midfield": (55, 42),
    "Left Center Midfield":  (55, 26),

    # Attacking midfield
    "Right Attacking Midfield":  (70, 44),
    "Center Attacking Midfield": (70, 34),
    "Left Attacking Midfield":   (70, 24),

    # Wide forwards
    "Right Wing":         (85, 58),
    "Left Wing":          (85, 10),

    # Striker line
    "Right Center Forward": (92, 42),
    "Center Forward":       (92, 34),
    "Left Center Forward":  (92, 26),
    "Striker":              (95, 34),
    "Secondary Striker":    (82, 34),
}


def position_xy(sb_position: str | None) -> tuple[float, float] | None:
    if not sb_position:
        return None
    return SB_POSITION_XY.get(sb_position)


def disambiguate_overlaps(positions: dict[int, tuple[float, float]]) -> dict[int, tuple[float, float]]:
    """If two players share the exact same (x, y), nudge them apart vertically.

    Players are processed in stable order; the first keeps the original spot,
    the second is offset by +6 then -6 then +12 etc.
    """
    by_coord: dict[tuple[float, float], list[int]] = {}
    for pid, xy in positions.items():
        by_coord.setdefault(xy, []).append(pid)
    out = dict(positions)
    for (x, y), pids in by_coord.items():
        if len(pids) <= 1:
            continue
        offsets = [0, 6, -6, 12, -12, 18, -18]
        for pid, dy in zip(sorted(pids), offsets):
            new_y = max(2, min(66, y + dy))
            out[pid] = (x, new_y)
    return out
