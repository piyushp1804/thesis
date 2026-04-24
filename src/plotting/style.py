"""Publication-grade matplotlib style shared by all figure scripts.

Call :func:`setup` once before creating any figure in a script; it sets
rcParams so every figure is produced at 300 dpi with a consistent serif
font, fixed palette, and aspect-ratio/padding defaults appropriate for a
two-column thesis page.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


PALETTE = {
    "ga": "#1f77b4",
    "pso": "#d62728",
    "nsga2": "#2ca02c",
    "surrogate": "#ff7f0e",
    "fem": "#7f7f7f",
    "llm": "#9467bd",
    "random": "#8c564b",
    "literature": "#000000",
    "feasible": "#2ca02c",
    "infeasible": "#d62728",
}

FIGSIZE_SINGLE = (4.5, 3.2)   # single column
FIGSIZE_WIDE = (6.8, 3.4)     # full page width, short
FIGSIZE_SQUARE = (4.5, 4.5)   # square plot (parity, heatmap)
FIGSIZE_TALL = (4.5, 5.5)     # tall plot (box plots)


def setup() -> None:
    """Apply the thesis-wide matplotlib defaults."""
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.4,
            "patch.linewidth": 0.8,
        }
    )


def save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    """Save ``fig`` into ``out_dir/name.png``; also write an SVG sibling."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    svg = out_dir / f"{name}.svg"
    fig.savefig(png)
    fig.savefig(svg)
    plt.close(fig)
    return png
