"""
plot_strided_access.py - plot strided access bandwidth vs stride with working set comparison

Plots bandwidth (TB/s or GB/s) against stride, with different lines for each 
working set size. Uses error bars to show variance across multiple runs.
"""

import argparse
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import pandas as pd
from plot_utils import PlotConfig, _apply_axes_config, plot_with_error_bars, init_style


def parse_working_set_size(ws_str: str) -> tuple:
    """
    Parse working set string like '8.388608MB' into (numeric_value, unit_str, formatted_label).
    Returns (value_in_bytes, original_string, formatted_label)
    """
    import re
    match = re.match(r'([0-9.]+)(MB|GB|KB)', ws_str)
    if not match:
        return (float('inf'), ws_str, ws_str)
    
    value_str, unit = match.groups()
    value = float(value_str)
    
    # Convert from decimal units (1000-based) to bytes
    decimal_multiplier = {'KB': 1000, 'MB': 1000**2, 'GB': 1000**3}[unit]
    bytes_value = value * decimal_multiplier
    
    # Map units to binary unit names
    unit_map = {'KB': 'KiB', 'MB': 'MiB', 'GB': 'GiB'}
    binary_unit = unit_map[unit]
    
    # Convert bytes to binary units (1024-based)
    binary_divisor = {'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}[unit]
    binary_value = bytes_value / binary_divisor
    
    # Format label: round to integer for clean power-of-2 display
    int_value = int(round(binary_value))
    formatted = f"{int_value}{binary_unit}"
    
    return (bytes_value, ws_str, formatted)


def parse_strided_access_csv(csv_file: Path) -> Dict[str, Dict]:
    """
    Parse strided access CSV and organize data by working set and stride.
    
    Expected CSV format:
      blocks,threads_per_block,working_set,iters,stride,rep,bandwidth
    
    Returns:
      {
        "working_set_name": {
          stride: [bandwidth_gbps, ...]
        }
      }
    """
    df = pd.read_csv(csv_file)
    
    # Convert bandwidth from bytes/s to GB/s
    df['bandwidth_gbps'] = df['bandwidth'].astype(float) / 1e9
    
    # Group by working_set, then by stride
    result = {}
    for ws_name, ws_group in df.groupby('working_set'):
        ws_data = {}
        
        for stride, stride_group in ws_group.groupby('stride'):
            ws_data[stride] = stride_group['bandwidth_gbps'].tolist()
        
        result[ws_name] = ws_data
    
    return result


def plot_strided_access_bandwidth(csv_file: Path, output_file: Path, mode: str = "L1"):
    """
    Plot strided access bandwidth vs stride with error bars for each working set.
    """
    
    data = parse_strided_access_csv(csv_file)
    
    # Sort working sets numerically by size
    ws_with_sizes = [(parse_working_set_size(ws)[0], ws) for ws in data.keys()]
    ws_with_sizes.sort()
    sorted_ws = [ws for _, ws in ws_with_sizes]
    
    # Create formatted labels for legend
    ws_labels = {ws: parse_working_set_size(ws)[2] for ws in sorted_ws}
    
    # Collect all unique strides (should be same across all working sets)
    all_strides = set()
    for ws_data in data.values():
        all_strides.update(ws_data.keys())
    all_strides = sorted(all_strides)
    
    # Prepare data for plot_with_error_bars and plot_with_box_plots
    y_triplets = []
    y_raw = []
    labels = []
    
    for ws_name in sorted_ws:
        ws_data = data[ws_name]
        series_triplets = []
        series_raw = []
        for stride in all_strides:
            if stride in ws_data:
                raw_vals = ws_data[stride]
                min_bw = min(raw_vals)
                mean_bw = sum(raw_vals) / len(raw_vals)
                max_bw = max(raw_vals)
                series_triplets.append((min_bw, mean_bw, max_bw))
                series_raw.append(raw_vals)
            else:
                series_triplets.append((0, 0, 0))  # Fallback if missing
                series_raw.append([])
        y_triplets.append(series_triplets)
        y_raw.append(series_raw)
        labels.append(f"{ws_labels[ws_name]} working set")
    
    cfg = PlotConfig(
        xlabel="Stride",
        ylabel="Bandwidth (GB/s)",
        title=f"Bandwidth vs stride ({mode})",
        xticks=all_strides,
        logx=True,
        figsize=(10, 6),
        grid=True,
    )
    
    # Use plot_with_error_bars to render with error bars
    plot_with_error_bars(
        all_strides,
        y_triplets,
        labels,
        outfile=output_file,
        cfg=cfg,
        legend_title="Working set size",
        legend_loc="lower left",
    )

    box_output = output_file.with_name(output_file.stem + "_box" + output_file.suffix)
    cfg_box = PlotConfig(
        xlabel="Stride",
        ylabel="Bandwidth (GB/s)",
        title=f"Bandwidth vs stride ({mode})",
        xticks=all_strides,
        logx=False,
        figsize=(10, 6),
        grid=True,
    )

    plot_overlapping_box_plots(
        all_strides,
        y_raw,
        labels,
        outfile=box_output,
        cfg=cfg_box,
        legend_title="Working set size",
        legend_loc="lower left",
        box_width=0.2,
    )

def plot_overlapping_box_plots(
    x,
    y_raw,
    labels,
    *,
    outfile,
    cfg: PlotConfig,
    legend_title: str,
    legend_loc: str,
    box_width: float,
):
    init_style()
    fig, ax = plt.subplots(figsize=cfg.figsize)

    n_ticks = len(x)
    indices = range(n_ticks)

    for i, (series, label) in enumerate(zip(y_raw, labels)):
        positions = list(indices)
        valid_data = []
        valid_pos = []
        for d, p in zip(series, positions):
            if d:
                valid_data.append(d)
                valid_pos.append(p)

        if not valid_data:
            continue

        color = ax._get_lines.get_next_color()
        bp = ax.boxplot(
            valid_data,
            positions=valid_pos,
            widths=box_width * 0.9,
            patch_artist=True,
            showfliers=False,
            whis=(0, 100),
            medianprops=dict(color="black", linewidth=1.4),
            boxprops=dict(color="black", linewidth=1),
            capprops=dict(color="black", linewidth=0),
            whiskerprops=dict(color="black", linewidth=1),
        )

        for j, box in enumerate(bp["boxes"]):
            box.set_facecolor(color)
            box.set_alpha(0.25)
            box.set_edgecolor("black")
            box.set_linewidth(1)
            if j == 0:
                box.set_label(label)

        for data_points, pos in zip(valid_data, valid_pos):
            if not data_points:
                continue
            y_min = min(data_points)
            y_max = max(data_points)
            x_vals = [pos, pos]
            ax.scatter(
                x_vals,
                [y_min, y_max],
                color=color,
                edgecolor="black",
                linewidth=0.5,
                s=20,
                alpha=0.9,
                zorder=3,
            )

    original_xticks = cfg.xticks
    cfg.xticks = None
    _apply_axes_config(ax, cfg)
    cfg.xticks = original_xticks

    ax.set_xscale("linear")
    ax.set_xticks(indices)
    ax.set_xticklabels([str(val) for val in x])

    ax.minorticks_off()
    ax.legend(title=legend_title, loc=legend_loc)

    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    print(f"Saved plot: {outfile}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot strided access bandwidth vs stride from CSV"
    )
    parser.add_argument("csv_file", help="Path to result.csv")
    parser.add_argument(
        "--output", 
        default="strided_access_bandwidth.png",
        help="Output plot filename"
    )
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        exit(1)
    
    output_path = Path(args.output)
    plot_strided_access_bandwidth(csv_path, output_path)
