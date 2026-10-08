"""Shared Matplotlib style for the ICRA paper figures (copied from one-phase-paper/figures_drafts/paper_style.py).

Ported from the author's earlier papers (depthbasedRL/plot_figures/_style.py, itself
derived from sapg/plot_figures): Times serif text with STIX math, hidden top/right
spines, soft #333 spines and ticks, no gridlines, frameless figure-level legends
placed below the panels, bold "(Ours)", hero blue vs orange foil, dpi-600 saves.

Font sizes are the two-column (3.5 in) set; pass ``scale`` to ``configure`` for
wider figures (the old 6.5 in single-column figures used ~1.25x).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# IEEE conference column widths (ieeeconf, letter): 3.45 in single, 7.0 in double.
COLUMN_WIDTH = 3.45
TEXT_WIDTH = 7.0

COLORS: dict[str, str] = {
    "hero": "#2C7BB6",        # blue: ours
    "foil": "#E08214",        # orange: main baseline / cost
    "third": "#9970AB",       # purple: secondary baseline
    "reference": "#4D4D4D",   # dark grey: human / privileged reference
    "hero_light": "#92C5DE",  # light blue: sim (paired with hero for real)
    "neutral": "#888888",     # separators, reference lines
    "spine": "#333333",
    "ink": "#222222",
    "muted_ink": "#555555",
}
BLUE_RAMP = ["#08519C", "#3182BD", "#6BAED6", "#41B6C4"]      # ordinal, dark -> light
GREEN_RAMP = ["#BAE4B3", "#74C476", "#31A354", "#006D2C"]     # ordinal, light -> dark
LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D"]

BAR_ALPHA = 0.9
BAND_ALPHA = 0.18


def configure(scale: float = 1.0) -> None:
    """Apply the global rcParams. Call once before creating a figure."""
    s = scale
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = [
        "Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif",
    ]
    matplotlib.rcParams.update({
        "mathtext.fontset": "stix",
        "font.size": 8 * s,
        "axes.labelsize": 9 * s,
        "axes.titlesize": 10 * s,
        "axes.titleweight": "normal",
        "xtick.labelsize": 8 * s,
        "ytick.labelsize": 8 * s,
        "legend.fontsize": 7.5 * s,
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": COLORS["spine"],
        "axes.linewidth": 0.8,
        "axes.labelcolor": COLORS["ink"],
        "text.color": COLORS["ink"],
        "xtick.color": COLORS["spine"],
        "ytick.color": COLORS["spine"],
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color(COLORS["spine"])
    ax.tick_params(axis="both", which="major", length=3, width=0.8, colors=COLORS["spine"])
    ax.grid(False)


def panel_label(fig: plt.Figure, x: float, y: float, text: str, size: float = 10) -> None:
    fig.text(x, y, text, ha="left", va="center", fontsize=size, fontweight="bold",
             color=COLORS["ink"])


def bottom_legend(fig: plt.Figure, handles, labels, *, y: float = 0.0, ncol: int | None = None,
                  bold: tuple[str, ...] = ("(Ours)", "(ours)"), **kwargs):
    """Frameless figure-level legend centred below the panels; bold 'ours' entries."""
    kwargs = {
        "loc": "lower center", "bbox_to_anchor": (0.5, y), "ncol": ncol or len(labels),
        "frameon": False, "handlelength": 1.2, "handleheight": 1.0, "handletextpad": 0.4,
        "columnspacing": 1.0, "borderpad": 0.0, "borderaxespad": 0.0, **kwargs,
    }
    leg = fig.legend(handles, labels, **kwargs)
    for text in leg.get_texts():
        if any(tag in text.get_text() for tag in bold):
            text.set_fontweight("bold")
    return leg


def save_figure(fig: plt.Figure, stem: str | Path, formats=("pdf", "png"),
                pad_inches: float = 0.02) -> list[Path]:
    """Write <stem>.pdf (paper asset) and <stem>.png (600-dpi preview)."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=pad_inches,
                    facecolor="white", edgecolor="none")
        paths.append(path)
        print(path)
    return paths
