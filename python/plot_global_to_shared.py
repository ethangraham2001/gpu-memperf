import argparse
from pathlib import Path
import pandas as pd

from plot_utils import PlotConfig, plot_with_peak, _prepare_outfile


def plot_global_to_shared(csv_file: Path, output_file: Path) -> None:
    df = pd.read_csv(csv_file)
    xticks = sorted(df["fops"].unique())
    ys = []
    labels = []
    for mode, group in df.groupby("mode"):
        group_sorted = group.sort_values("fops")
        bw = (group_sorted["buf_size"] / 1e9) / (group_sorted["ms"] / 1000.0)
        ys.append(bw.tolist())
        labels.append(str(mode))

    cfg = PlotConfig(
        xlabel="#operations per 32-bit value",
        ylabel="Bandwidth (GB/s)",
        title="Global to Shared Memory Bandwidth (4KiB tile)",
        subtitle="108 Blocks, 1024 Threads Per Block",
        logx=True,
        xticks=xticks,
    )

    plot_with_peak(xticks, ys, labels, outfile=output_file, cfg=cfg)


def main():
    parser = argparse.ArgumentParser(
        usage="python plot_global_to_shared.py <csv_file> [--output output_file]"
    )
    parser.add_argument("csv_file", type=Path, help="Path to the input CSV file")
    parser.add_argument("--output", type=Path, default=Path("plot_shared_to_register"))
    args = parser.parse_args()
    args = parser.parse_args()
    plot_global_to_shared(args.csv_file, _prepare_outfile(args.output))


if __name__ == "__main__":
    main()
