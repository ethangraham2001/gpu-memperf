import argparse
import pandas as pd
from pathlib import Path

from plot_utils import PlotConfig, line_plot, plot_with_error_bars, _prepare_outfile


def plot_shared_memory_multiple_threads(csv_file: Path, output_file: Path) -> None:
    """
    Plot shared memory to register bandwidth for multiple thread.
    """

    df = pd.read_csv(csv_file)

    # If multiple runs measured, take the average across runs for more robustness
    if "run" in df.columns:
        df = df.groupby(["threads", "stride"]).agg({"bandwidthGBps": "mean"}).reset_index()

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


def plot_shared_memory_error_bars(csv_file: Path, output_file: Path) -> None:
    """
    Plot shared memory to register bandwidth with error bars across runs.

    Expects a CSV with multiple runs of the same benchmark with at least: run, stride, bandwidthGBps.
    We aggregate over runs per stride and show (min, mean, max).
    """

    df = pd.read_csv(csv_file)

    # Check that required columns exist
    required_cols = {"run", "stride", "bandwidthGBps"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV file is missing required columns: {missing}")

    # If multiple experiments were measured, take the benchmark with the maximum threads and bytes
    if "threads" in df.columns:
        max_threads = df["threads"].max()
        df = df[df["threads"] == max_threads]
    if "bytes" in df.columns:
        max_bytes = df["bytes"].max()
        df = df[df["bytes"] == max_bytes]

    df_sorted = df.sort_values("stride")
    xticks = sorted(df_sorted["stride"].unique())

    # Aggregate over runs per stride for error bars
    triplets = []
    for stride in xticks:
        group = df_sorted[df_sorted["stride"] == stride]["bandwidthGBps"]
        low = group.min()
        mid = group.mean()
        high = group.max()
        triplets.append((low, mid, high))

    # Configure plot
    cfg = PlotConfig(
        xlabel="Stride",
        ylabel="Bandwidth (GB/s)",
        title="Shared Memory to Register Bandwidth (Single SM, 8 KiB read)",
        logx=True,
        xticks=xticks,
    )

    y_triplets = [triplets]
    labels = [f"{threads} threads" for threads in df_sorted["threads"].unique()]

    plot_with_error_bars(
        xticks,
        y_triplets,
        labels,
        outfile=output_file,
        cfg=cfg,
        legend_title=None,
    )


def main():
    parser = argparse.ArgumentParser(
        usage="python plot_shared_to_register.py <csv_file> [--output output_file] [--mode mode]"
    )
    parser.add_argument("csv_file", type=Path, help="Path to the input CSV file")
    parser.add_argument("--output", type=Path, default=Path("plot_shared_to_register"))
    parser.add_argument(
        "--mode",
        type=str,
        default="error_bars",
        choices=["multiple_threads", "error_bars"],
    )

    args = parser.parse_args()
    if args.mode == "multiple_threads":
        plot_shared_memory_multiple_threads(
            args.csv_file, _prepare_outfile(args.output)
        )
    elif args.mode == "error_bars":
        plot_shared_memory_error_bars(args.csv_file, _prepare_outfile(args.output))


if __name__ == "__main__":
    main()
