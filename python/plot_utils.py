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
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union


def init_style():
    """Initialize consistent style for all plots.
    - Research paper aesthetic
    - Inside ticks on all sides
    - Major ticks only
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
    subtitle: Optional[str] = None
    logx: bool = False
    logy: bool = False
    xticks: Optional[List] = None
    yticks: Optional[List] = None
    figsize: tuple = (8, 5)
    xlim: Optional[tuple] = None
    ylim: Optional[tuple] = None
    grid: bool = True


def _prepare_outfile(outfile) -> Path:
    plots_dir = Path(__file__).resolve().parent.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir / (str(outfile).replace(".png", "") + ".png")


def _resolve_style(
    style: Optional[Union[Dict, Sequence]], idx: int, label: str
) -> Optional[str]:
    if style is None:
        return None
    if isinstance(style, dict):
        return style.get(label)
    if isinstance(style, (list, tuple)):
        if idx < len(style):
            return style[idx]
        return style[-1]
    return style


def _annotate_peak(ax, x_vals, y_vals, label, color):
    if not y_vals:
        return
    peak_idx = max(range(len(y_vals)), key=lambda i: y_vals[i])
    x_peak = x_vals[peak_idx]
    y_peak = y_vals[peak_idx]
    ax.annotate(
        f"peak {label}\n{y_peak:.1f} GB/s",
        xy=(x_peak, y_peak),
        xytext=(x_peak, y_peak * 1.05),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color=color or "black", lw=1.1),
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.35",
            fc="white",
            ec=color or "black",
            lw=0.8,
            alpha=0.85,
        ),
        color=color or "black",
    )


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

    if cfg.subtitle:
        ax.text(
            x=0.5,
            y=0.95,
            s=cfg.subtitle,
            ha="center",
            transform=ax.transAxes,
            fontsize=10,
            color="gray",
        )

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
    if cfg.xlim:
        ax.set_xlim(cfg.xlim)
    if cfg.ylim:
        ax.set_ylim(cfg.ylim)

    # Only show major ticks
    ax.minorticks_off()

    if cfg.grid:
        ax.grid(True, alpha=0.4, linestyle="--")
        ax.set_axisbelow(True)

    # Legend
    ax.legend()

    fig.tight_layout()

    outfile = _prepare_outfile(outfile)
    fig.savefig(outfile, dpi=300)
    print(f"Saved plot: {outfile}")


def plot_with_peak(
    x,
    ys,
    labels,
    *,
    outfile,
    cfg: PlotConfig,
    palette: Optional[Union[Dict, Sequence]] = None,
    fill_alpha: float = 0.0,
    linewidth: float = 2.0,
    legend_title: Optional[str] = None,
    legend_loc: str = "best",
    annotate: bool = True,
):
    """Enhanced line plot with fill + peak annotation. Designed to generalize
    the custom styling previously used in plot_bandwidths."""

    init_style()
    fig, ax = plt.subplots(figsize=cfg.figsize)

    for idx, (y, label) in enumerate(zip(ys, labels)):
        color = _resolve_style(palette, idx, label)
        marker = "."
        line, = ax.plot(
            x,
            y,
            marker=marker,
            markersize=6.5,
            linewidth=linewidth,
            color=color,
            label=label,
        )
        shade_color = color or line.get_color()
        if fill_alpha > 0:
            ax.fill_between(x, y, color=shade_color, alpha=fill_alpha)
        if annotate:
            _annotate_peak(ax, x, y, label, shade_color)

    ax.set_xlabel(cfg.xlabel)
    ax.set_ylabel(cfg.ylabel)

    if cfg.title:
        ax.set_title(cfg.title)
    if cfg.logx:
        ax.set_xscale("log")
    if cfg.logy:
        ax.set_yscale("log")
    if cfg.xticks:
        ax.set_xticks(cfg.xticks)
        ax.set_xticklabels([str(xval) for xval in cfg.xticks])
    if cfg.yticks:
        ax.set_yticks(cfg.yticks)
    if cfg.xlim:
        ax.set_xlim(cfg.xlim)
    if cfg.ylim:
        ax.set_ylim(cfg.ylim)

    ax.set_facecolor("#EAEAF2")
    ax.patch.set_alpha(0.4)
    ax.grid(True, alpha=0.4, linestyle="--")
    ax.set_axisbelow(True)

    ax.minorticks_off()
    ax.legend(title=legend_title, loc=legend_loc)

    fig.tight_layout()
    outfile = _prepare_outfile(outfile)
    fig.savefig(outfile, dpi=300)
    print(f"Saved plot: {outfile}")


def plot_with_error_bars(
    x,
    y_triplets,
    labels,
    *,
    outfile,
    cfg: PlotConfig,
    palette: Optional[Union[Dict, Sequence]] = None,
    markers: Optional[Union[Dict, Sequence]] = None,
    linewidth: float = 2.0,
    capsize: float = 4.0,
    fill_alpha: float = 0.12,
    legend_title: Optional[str] = None,
    legend_loc: str = "best",
):
    """Plot series that carry (low, center, high) measurements per x-point,
    rendering both error bars and a translucent band between the bounds.

    Args:
        x: shared x values.
        y_triplets: iterable of series, where each series entry is (low, mid, high).
        labels: legend labels for each series.
        outfile: output name.
        cfg: PlotConfig.
    """

    init_style()
    fig, ax = plt.subplots(figsize=cfg.figsize)

    for idx, (series, label) in enumerate(zip(y_triplets, labels)):
        if any(len(vals) != 3 for vals in series):
            raise ValueError("Each measurement must contain exactly (low, mid, high) values.")
        lows = [vals[0] for vals in series]
        mids = [vals[1] for vals in series]
        highs = [vals[2] for vals in series]
        lower_err = [mid - low for mid, low in zip(mids, lows)]
        upper_err = [high - mid for high, mid in zip(highs, mids)]
        color = _resolve_style(palette, idx, label)
        marker = _resolve_style(markers, idx, label) or "o"
        err_container = ax.errorbar(
            x,
            mids,
            yerr=[lower_err, upper_err],
            marker=marker,
            linestyle="-",
            linewidth=linewidth,
            color=color,
            ecolor=color,
            capsize=capsize,
            label=label,
        )
        line = err_container[0]
        shade_color = color or line.get_color()
        if fill_alpha > 0:
            ax.fill_between(x, lows, highs, color=shade_color, alpha=fill_alpha)

    ax.set_xlabel(cfg.xlabel)
    ax.set_ylabel(cfg.ylabel)

    if cfg.title:
        ax.set_title(cfg.title)
    if cfg.logx:
        ax.set_xscale("log")
    if cfg.logy:
        ax.set_yscale("log")
    if cfg.xticks:
        ax.set_xticks(cfg.xticks)
        ax.set_xticklabels([str(xval) for xval in cfg.xticks])
    if cfg.yticks:
        ax.set_yticks(cfg.yticks)
    if cfg.xlim:
        ax.set_xlim(cfg.xlim)
    if cfg.ylim:
        ax.set_ylim(cfg.ylim)

    ax.minorticks_off()
    if cfg.grid:
        ax.grid(True, alpha=0.4, linestyle="--")
        ax.set_axisbelow(True)
    ax.legend(title=legend_title, loc=legend_loc)

    fig.tight_layout()
    outfile = _prepare_outfile(outfile)
    fig.savefig(outfile, dpi=300)
    print(f"Saved plot: {outfile}")
