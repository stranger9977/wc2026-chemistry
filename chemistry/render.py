"""Render Figure-1-style pitch chemistry graphs from chemistry.json."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.patheffects as plt_path_effects
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
    players_by_id = nation_entry.get("players_by_id", {})
    pitch_ids = set(nation_entry.get("pitch_player_ids") or [int(k) for k in players_by_id.keys()])

    id_to_xy: dict[int, tuple[float, float, str, str | None]] = {}
    for pid_str, p in players_by_id.items():
        pid = int(pid_str)
        if pid in pitch_ids:
            label = p.get("display_name") or p["name"]
            id_to_xy[pid] = (p["x"], p["y"], label, p.get("sb_position"))

    # If players_by_id is empty, fall back to the old name-based approach so
    # legacy nation entries still render something.
    if not id_to_xy:
        name_to_pos: dict[str, tuple[float, float]] = {}
        for p in squad["players"]:
            pos = DEFAULT_POSITIONS.get(p["position"], (50, 34))
            name_to_pos[p["name"]] = pos
        for name, (x, y) in name_to_pos.items():
            # Use a fake id based on name hash to reuse same draw path
            id_to_xy[hash(name) & 0xFFFFFFFF] = (x, y, name, None)

    eligible_pairs = [p for p in nation_entry["pairs"]
                      if p["player_a_id"] in pitch_ids and p["player_b_id"] in pitch_ids]
    eligible_pairs.sort(key=lambda p: -p["joi90"])
    TOP_N = 15
    top_pairs = eligible_pairs[:TOP_N]
    dim_pairs = eligible_pairs[TOP_N:]

    if top_pairs:
        vmin = min(p["joi90"] for p in top_pairs)
        vmax = max(p["joi90"] for p in top_pairs)
    else:
        vmin = vmax = 0.0

    # Dim edges
    for p in dim_pairs:
        if p["player_a_id"] not in id_to_xy or p["player_b_id"] not in id_to_xy:
            continue
        xa, ya, _, _pa = id_to_xy[p["player_a_id"]]
        xb, yb, _, _pb = id_to_xy[p["player_b_id"]]
        ax.plot([xa, xb], [ya, yb], color="#6b7280", linewidth=0.8, alpha=0.18,
                solid_capstyle="round", zorder=0)

    # Top edges
    for p in top_pairs:
        if p["player_a_id"] not in id_to_xy or p["player_b_id"] not in id_to_xy:
            continue
        xa, ya, _, _pa = id_to_xy[p["player_a_id"]]
        xb, yb, _, _pb = id_to_xy[p["player_b_id"]]
        col = _color_for_joi(p["joi90"], vmin, vmax, opts)
        m = min(p["minutes"], 600) / 600.0
        lw = opts.edge_min_width + (opts.edge_max_width - opts.edge_min_width) * m
        ax.plot([xa, xb], [ya, yb], color=col, linewidth=lw,
                solid_capstyle="round", alpha=0.95, zorder=1)

    # Markers + labels with smart placement to avoid overlap
    from chemistry.formation import short_position
    placed_points: list[tuple[float, float]] = []
    for pid, (x, y, name, sb_pos) in id_to_xy.items():
        ax.add_patch(Circle((x, y), 1.8, color="#ffffff", zorder=3))
        ax.add_patch(Circle((x, y), 1.5, color=squad["team_color"], zorder=4))
        offset = -3.0
        for (xp, yp) in placed_points:
            if abs(xp - x) < 8 and abs(yp - y) < 4:
                offset = 3.8
                break
        if opts.show_labels:
            va = "top" if offset > 0 else "bottom"
            ax.text(x, y + offset, name, color="#ffffff", ha="center", va=va,
                    fontsize=9, weight="bold", zorder=5,
                    path_effects=[plt_path_effects.withStroke(linewidth=2.5, foreground="black")])
            pos_abbrev = short_position(sb_pos)
            if pos_abbrev:
                pos_offset = offset + (1.7 if offset > 0 else -1.7)
                ax.text(x, y + pos_offset, pos_abbrev, color="#9ca3af", ha="center", va=va,
                        fontsize=7, weight="bold", zorder=5,
                        path_effects=[plt_path_effects.withStroke(linewidth=2, foreground="black")])
        placed_points.append((x, y))

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


def render_leaderboard(doc: dict, out_path: Path,
                       title: str = "World Cup 2026 — Top Chemistry Pairs") -> Path:
    items = doc["leaderboard"][:20]
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.set_axis_off()

    ax.text(0.0, 1.0, title, color="#e6edf3", fontsize=36, weight="bold",
            transform=ax.transAxes, va="top")

    row_height = 0.04
    for i, p in enumerate(items):
        y = 0.92 - i * row_height
        ax.text(0.00, y, f"{i+1:>2}.", color="#8b949e", fontsize=16,
                transform=ax.transAxes, family="monospace")
        a_label = p.get('player_a_display') or p['player_a_name']
        b_label = p.get('player_b_display') or p['player_b_name']
        ax.text(0.05, y, f"{a_label}  +  {b_label}",
                color="#e6edf3", fontsize=18, transform=ax.transAxes)
        ax.text(0.78, y, f"JOI90  {p['joi90']:.3f}",
                color="#4ade80", fontsize=18, weight="bold",
                transform=ax.transAxes, family="monospace")
        ax.text(0.92, y, f"{int(p['minutes'])} mins",
                color="#8b949e", fontsize=14, transform=ax.transAxes, family="monospace")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(),
                bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return out_path


def render_landing_grid(doc: dict, out_path: Path) -> Path:
    nations = list(doc["nations"].items())
    fig = plt.figure(figsize=(38.4, 21.6), dpi=100)
    fig.patch.set_facecolor("#0e1117")
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.94])
    ax.set_facecolor("#0e1117"); ax.set_axis_off()
    ax.set_xlim(0, 4); ax.set_ylim(0, 8)
    ax.invert_yaxis()
    ax.text(0, -0.4, "World Cup 2026 — Chemistry",
            color="#e6edf3", fontsize=64, weight="bold", va="top")

    for idx, (code, entry) in enumerate(nations):
        col = idx % 4
        row = idx // 4
        x, y = col + 0.05, row + 0.1
        squad = entry["squad"]
        ax.text(x, y + 0.08, squad["nation"], color="#e6edf3", fontsize=28, weight="bold")
        ax.text(x, y + 0.22, squad["manager"], color="#8b949e", fontsize=16)
        top = entry["pairs"][:1]
        if top:
            p = top[0]
            a_label = p.get('player_a_display') or p['player_a_name']
            b_label = p.get('player_b_display') or p['player_b_name']
            ax.text(x, y + 0.42, f"{a_label} + {b_label}",
                    color="#e6edf3", fontsize=18)
            ax.text(x, y + 0.55, f"JOI90 {p['joi90']:.3f}",
                    color="#4ade80", fontsize=18, weight="bold", family="monospace")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(),
                bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return out_path
