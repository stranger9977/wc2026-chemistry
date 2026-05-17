"""Render Figure-1-style pitch chemistry graphs from chemistry.json."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mplsoccer import Pitch

import cairosvg


DEFAULT_POSITIONS: dict[str, tuple[float, float]] = {
    "GK":  (8,  34),
    "RB":  (25, 58), "RWB": (40, 60), "RCB": (22, 44),
    "CB":  (22, 34), "LCB": (22, 24), "LB":  (25, 10), "LWB": (40, 8),
    "RDM": (35, 42), "DM":  (35, 34), "LDM": (35, 26),
    "RCM": (55, 42), "CM":  (55, 34), "LCM": (55, 26),
    "RM":  (60, 56), "LM":  (60, 12),
    "RAM": (70, 42), "AM":  (70, 34), "LAM": (70, 26),
    "RW":  (85, 56), "LW":  (85, 12),
    "ST":  (90, 34), "CF": (90, 34), "SS":  (82, 34),
}


@dataclass
class PitchOptions:
    width: int = 1920
    height: int = 1080
    background: str = "transparent"
    accent: str = "#4ade80"
    cold: str = "#ef4444"
    edge_min_width: float = 1.5
    edge_max_width: float = 6.0
    show_labels: bool = True


def _color_for_joi(joi90: float, vmin: float, vmax: float, opts: PitchOptions) -> str:
    if vmax == vmin:
        return opts.accent
    t = (joi90 - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    cold = tuple(int(opts.cold[i:i+2], 16) for i in (1, 3, 5))
    hot  = tuple(int(opts.accent[i:i+2], 16) for i in (1, 3, 5))
    rgb = tuple(int(c + (h - c) * t) for c, h in zip(cold, hot))
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def render_pitch_chemistry(
    nation_entry: dict,
    opts: Optional[PitchOptions] = None,
) -> plt.Figure:
    opts = opts or PitchOptions()
    pitch = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68,
                  line_color="#ffffff", pitch_color="none", linewidth=1.2)
    fig, ax = pitch.draw(figsize=(opts.width / 100, opts.height / 100), constrained_layout=False)
    if opts.background != "transparent":
        fig.patch.set_facecolor(opts.background)
    else:
        fig.patch.set_alpha(0)

    squad = nation_entry["squad"]
    pairs = nation_entry["pairs"]

    name_to_pos: dict[str, tuple[float, float]] = {}
    for p in squad["players"]:
        pos = DEFAULT_POSITIONS.get(p["position"], (50, 34))
        name_to_pos[p["name"]] = pos

    if pairs:
        vmin = min(p["joi90"] for p in pairs)
        vmax = max(p["joi90"] for p in pairs)
    else:
        vmin = vmax = 0.0

    for p in pairs:
        a, b = p["player_a_name"], p["player_b_name"]
        pa = name_to_pos.get(a)
        pb = name_to_pos.get(b)
        if not pa or not pb:
            continue
        col = _color_for_joi(p["joi90"], vmin, vmax, opts)
        m = min(p["minutes"], 600) / 600.0
        lw = opts.edge_min_width + (opts.edge_max_width - opts.edge_min_width) * m
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=col,
                linewidth=lw, solid_capstyle="round", alpha=0.85, zorder=1)

    for name, (x, y) in name_to_pos.items():
        ax.add_patch(Circle((x, y), 1.6, color="#ffffff", zorder=3))
        ax.add_patch(Circle((x, y), 1.3, color=squad["team_color"], zorder=4))
        if opts.show_labels:
            ax.text(x, y + 3.0, name, color="#ffffff", ha="center", va="bottom",
                    fontsize=10, weight="bold", zorder=5)

    ax.set_xlim(0, 105); ax.set_ylim(0, 68); ax.set_axis_off()
    return fig


def fig_to_png(fig: plt.Figure, path: Path, *, transparent: bool = True, dpi: int = 200) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, transparent=transparent, dpi=dpi, bbox_inches="tight", pad_inches=0)
    return path


def fig_to_svg(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0, transparent=True)
    return path


def svg_to_4k_png(svg_path: Path, png_path: Path, width: int = 3840) -> Path:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=width)
    return png_path
