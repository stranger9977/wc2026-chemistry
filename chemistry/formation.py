"""Map StatsBomb position labels to (x, y) coordinates on a 105×68 SPADL pitch."""
from __future__ import annotations

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
