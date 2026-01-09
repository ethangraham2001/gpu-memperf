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
import numpy as np


def init_style():
    """Initialize consistent style for all plots.
    - Research paper aesthetic
    - Inside ticks on all sides
    - Major ticks only
    - Smaller markers/lines
    - Legend without frame
    """

    plt.rcParams.update(
        {
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
        }
    )


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


def _apply_axes_config(ax, cfg: PlotConfig):
    """Apply common axis configuration based solely on PlotConfig."""
    ax.set_xlabel(cfg.xlabel)
    ax.set_ylabel(cfg.ylabel)

    # Title and optional subtitle
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

    # Tick / scale options
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

    # Grid handling
    if cfg.grid:
        ax.grid(True, alpha=0.4, linestyle="--")
        ax.set_axisbelow(True)


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

    _apply_axes_config(ax, cfg)

    # Only show major ticks
    ax.minorticks_off()

    # Legend
    ax.legend()

    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    print(f"Saved plot: {outfile}")
    plt.close(fig)


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
        (line,) = ax.plot(
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

    _apply_axes_config(ax, cfg)

    ax.set_facecolor("#EAEAF2")
    ax.patch.set_alpha(0.4)

    ax.minorticks_off()
    ax.legend(title=legend_title, loc=legend_loc)

    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    print(f"Saved plot: {outfile}")
    plt.close(fig)


def plot_with_error_bars(
    x,
    y_triplets,
    labels,
    *,
    outfile,
    cfg: PlotConfig,
    palette: Optional[Union[Dict, Sequence]] = None,
    markers: Optional[Union[Dict, Sequence]] = None,
    linewidth: float = 1.0,
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
            raise ValueError(
                "Each measurement must contain exactly (low, mid, high) values."
            )
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
            linestyle="--",
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

    _apply_axes_config(ax, cfg)

    ax.minorticks_off()
    ax.legend(title=legend_title, loc=legend_loc)

    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    print(f"Saved plot: {outfile}")
    plt.close(fig)


def plot_with_error_bars_raw(
    x,
    y_raw,
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
    """Plot series from raw measurements (expecting ~3-5 per point).
    Calculates min, mean, max automatically.

    Args:
        x: shared x values.
        y_raw: iterable of series. Each series is a list of lists (one list of measurements per x point).
               e.g. [ [run1, run2, run3], [run1...], ... ] for each x.
        labels: legend labels.
        outfile: output path.
        cfg: PlotConfig.
    """
    triplets_all = []
    for series in y_raw:
        # series corresponds to one line
        # it contains a list of measurements for each x
        series_triplets = []
        for measurements in series:
            if not measurements:
                series_triplets.append((0, 0, 0))
                continue
            low = min(measurements)
            high = max(measurements)
            mid = sum(measurements) / len(measurements)
            series_triplets.append((low, mid, high))
        triplets_all.append(series_triplets)

    plot_with_error_bars(
        x,
        triplets_all,
        labels,
        outfile=outfile,
        cfg=cfg,
        palette=palette,
        markers=markers,
        linewidth=linewidth,
        capsize=capsize,
        fill_alpha=fill_alpha,
        legend_title=legend_title,
        legend_loc=legend_loc,
    )


def plot_with_box_plots(
    x,
    y_raw,
    labels,
    *,
    outfile,
    cfg: PlotConfig,
    palette: Optional[Union[Dict, Sequence]] = None,
    legend_title: Optional[str] = None,
    legend_loc: str = "best",
    box_width: float = 0.6,
    jitter_frac: float = 0.1,
):
    """Create a grouped box plot with scatter overlay.

    Args:
        x: shared x values (ticks).
        y_raw: iterable of series. Each series is a list of lists (measurements per x point).
        labels: legend labels for each series.
        outfile: output path.
        cfg: PlotConfig.
    """
    init_style()
    fig, ax = plt.subplots(figsize=cfg.figsize)

    n_series = len(y_raw)
    n_ticks = len(x)
    
    indices = range(n_ticks)
    total_group_width = box_width
    single_box_width = total_group_width / n_series if n_series > 0 else total_group_width

    for i, (series, label) in enumerate(zip(y_raw, labels)):
        positions = []
        valid_data = []
        valid_pos = []
        jitter_scales = []
        for idx, d in enumerate(series):
            if not d:
                continue
            base_x = idx
            available_series = [
                s_idx
                for s_idx, s in enumerate(y_raw)
                if idx < len(s) and s[idx]
            ]
            if len(available_series) <= 1:
                pos = base_x
            else:
                ranges = {}
                for s_idx in available_series:
                    data = y_raw[s_idx][idx]
                    min_val = min(data)
                    max_val = max(data)
                    span = max(max_val - min_val, 1e-12)
                    pad = span * 0.15
                    ranges[s_idx] = (min_val - pad, max_val + pad)
                overlap_group = []
                for s_idx in available_series:
                    i_min, i_max = ranges[i]
                    s_min, s_max = ranges[s_idx]
                    if not (i_max < s_min or i_min > s_max):
                        overlap_group.append(s_idx)
                if len(overlap_group) <= 1:
                    pos = base_x
                    jitter_scale = 0.0
                else:
                    rank = overlap_group.index(i)
                    offset = (rank - (len(overlap_group) - 1) / 2) * single_box_width
                    pos = base_x + offset
                    jitter_scale = jitter_frac
                jitter_scales.append(jitter_scale)
            
            positions.append(pos)
            valid_data.append(d)
            valid_pos.append(pos)

        if not valid_data:
            continue

        color = _resolve_style(palette, i, label)
        if color is None:
            # Robustly get next color from cycle
            color = ax._get_lines.get_next_color()

        # Plot boxplot
        bp = ax.boxplot(
            valid_data,
            positions=valid_pos,
            widths=single_box_width * 0.8,
            patch_artist=True,
            showfliers=False, # We will plot points manually
            medianprops=dict(color="black", linewidth=1.2),
            capprops=dict(color="black", linewidth=1),
            whiskerprops=dict(color="black", linewidth=1),
        )
        
        # Style boxes and legend
        for j, box in enumerate(bp['boxes']):
            box.set_facecolor(color)
            box.set_alpha(0.75)
            box.set_edgecolor("black")
            box.set_linewidth(1)
            box.set_zorder(3)
            if j == 0:
                box.set_label(label)

        # Scatter overlay
        for data_points, pos, jitter_scale in zip(valid_data, valid_pos, jitter_scales):
            # Jitter
            jitter = np.random.uniform(
                -single_box_width * jitter_scale,
                single_box_width * jitter_scale,
                size=len(data_points),
            )
            ax.scatter(
                [pos] * len(data_points) + jitter, 
                data_points, 
                color=color, 
                edgecolor='black', 
                linewidth=0.5,
                s=12, 
                alpha=0.6,
                zorder=2
            )

    # Temporarily clear xticks from cfg to prevent _apply_axes_config from setting them wrong
    original_xticks = cfg.xticks
    cfg.xticks = None
    
    _apply_axes_config(ax, cfg)
    
    cfg.xticks = original_xticks
    
    # Must be done BEFORE setting ticks, otherwise labels might be reset
    ax.set_xscale("linear") 
    
    # Override x ticks
    ax.set_xticks(indices)
    ax.set_xticklabels([str(val) for val in x])

    ax.minorticks_off()
    ax.legend(title=legend_title, loc=legend_loc)

    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    print(f"Saved plot: {outfile}")
    plt.close(fig)
