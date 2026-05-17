"""1920x320 lower-third PNG card for a single chemistry pair."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


@dataclass
class CardOptions:
    width_px: int = 1920
    height_px: int = 320
    bg: str = "#0e1117"
    accent: str = "#bf0d3e"
    text: str = "#e6edf3"
    muted: str = "#8b949e"


def render_pair_card(
    pair: dict,
    *,
    nation: str,
    flag_iso: str,
    opts: Optional[CardOptions] = None,
    out_path: Optional[Path] = None,
) -> Path:
    opts = opts or CardOptions()
    fig = plt.figure(figsize=(opts.width_px / 100, opts.height_px / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, opts.width_px); ax.set_ylim(0, opts.height_px); ax.set_axis_off()
    fig.patch.set_alpha(0)

    ax.add_patch(Rectangle((0, 0), 12, opts.height_px, color=opts.accent, lw=0))
    ax.add_patch(Rectangle((12, 0), opts.width_px - 12, opts.height_px,
                           color=opts.bg, alpha=0.92, lw=0))

    ax.text(60, opts.height_px - 40, nation.upper(),
            color=opts.muted, fontsize=18, weight="bold", family="monospace",
            ha="left", va="top")

    a_label = pair.get("player_a_display") or pair["player_a_name"]
    b_label = pair.get("player_b_display") or pair["player_b_name"]
    label = f"{a_label}  +  {b_label}"
    ax.text(60, opts.height_px - 90, label,
            color=opts.text, fontsize=48, weight="bold",
            ha="left", va="top")

    ax.text(60, 80,
            f"JOI90 {pair['joi90']:.3f}    {int(pair['minutes'])} mins shared    {pair['matches']} matches",
            color=opts.muted, fontsize=22, ha="left", va="top", family="monospace")

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, transparent=True, dpi=100, bbox_inches=None, pad_inches=0)
        plt.close(fig)
    return out_path
