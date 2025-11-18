import argparse
import pandas as pd
from pathlib import Path

from plot_utils import PlotConfig, line_plot


def plot_register_shared(csv_file: Path, output_file: Path) -> None:
    """
    Plot register to shared memory bandwidth
    """

    # Load CSV
    df = pd.read_csv(csv_file)

    # Extract x and per-thread y series
    xticks = sorted(df["stride"].unique())
    ys = []
    labels = []

    for threads, group in df.groupby("threads"):
        group_sorted = group.sort_values("stride")
        ys.append(group_sorted["bandwidthGBps"].tolist())
        labels.append(f"{threads} threads")

    # Configure plot
    cfg = PlotConfig(
        xlabel="Stride",
        ylabel="Bandwidth (GB/s)",
        title="Shared Memory to Register Bandwidth (Single SM, 8 KiB read)",
        logx=True,
        xticks=xticks,
    )

    line_plot(xticks, ys, labels, outfile=output_file, cfg=cfg)


def main():
    parser = argparse.ArgumentParser(
        usage="python plot_threads.py <csv_file> [--output output_file]"
    )
    parser.add_argument("csv_file", type=Path, help="Path to the input CSV file")
    parser.add_argument("--output", type=Path, default=Path("plot_shared_to_register"))

    args = parser.parse_args()
    plot_register_shared(args.csv_file, args.output)


if __name__ == "__main__":
    main()
