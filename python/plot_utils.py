"""
Shared plotting utilities for high-quality and consistent figures.
Provides styling, tick formatting, legend, and helper functions for common plot patterns.
Disclaimer: written with Claude

Usage:
    from plot_utils import init_style, line_plot

    cfg = PlotConfig(
    xlabel="Stride",
    ylabel="Bandwidth (GB/s)",
    title="Example Plot",
    logx=True,
    xticks=[1,2,4,8,16]
    )

    x = [1,2,4,8,16]
    ys = [[10, 9, 7, 5, 3]]
    labels = ["Example series"]

    line_plot(x, ys, labels, outfile="example", cfg=cfg)
"""

import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path


def init_style():
    """Initialize consistent style for all plots.
    - Research paper aesthetic
    - Inside ticks on all sides
    - Major ticks only
    - Grid disabled
    - Smaller markers/lines
    - Legend without frame
    """

    plt.rcParams.update({
        # Font / text
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,

        # Lines & markers
        "lines.linewidth": 1.5,
        "lines.markersize": 4,

        # Ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,

        # Grid
        "grid.alpha": 0.25,

        # Legend
        "legend.frameon": False,
    })


@dataclass
class PlotConfig:
    xlabel: str
    ylabel: str
    title: Optional[str] = None
    logx: bool = False
    logy: bool = False
    xticks: Optional[List] = None
    yticks: Optional[List] = None
    figsize: tuple = (8, 5)


def line_plot(x, ys, labels, *, outfile, cfg: PlotConfig):
    """Create a standardized line plot.

    Args:
        x: shared x values
        ys: list of y series
        labels: legend labels
        outfile: path to output .png
        cfg: PlotConfig instance
    """

    init_style()
    fig, ax = plt.subplots(figsize=cfg.figsize)

    # Plot lines
    for y, label in zip(ys, labels):
        ax.plot(x, y, marker="o", label=label)

    ax.set_xlabel(cfg.xlabel)
    ax.set_ylabel(cfg.ylabel)

    # Optional title
    if cfg.title:
        ax.set_title(cfg.title)

    # Tick options
    if cfg.logx:
        ax.set_xscale("log")
    if cfg.logy:
        ax.set_yscale("log")

    if cfg.xticks:
        ax.set_xticks(cfg.xticks)
        ax.set_xticklabels([str(x) for x in cfg.xticks])
    if cfg.yticks:
        ax.set_yticks(cfg.yticks)

    # Only show major ticks
    ax.minorticks_off()

    # Legend
    ax.legend()

    fig.tight_layout()

    plots_dir = Path("../plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    outfile = plots_dir / (str(outfile).replace(".png", "") + ".png")
    fig.savefig(outfile, dpi=300)
    print(f"Saved plot: {outfile}")
