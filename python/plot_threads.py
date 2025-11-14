"""
Plot benchmark results with different thread counts.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_register_shared(csv_file: Path, output_file: Path) -> None:
    """
    Plot register to shared memory bandwidth from CSV data.
    Args:
        csv_file: Path to CSV file.
        output_file: Output filename for plot
    """

    # Read CSV
    df = pd.read_csv(csv_file)

    # Create the plot
    plt.figure(figsize=(8, 5))
    for threads, group in df.groupby("threads"):
        plt.plot(
            group["stride"],
            group["bandwidthGBps"],
            marker="o",
            label=f"{threads} threads",
        )

    plt.xlabel("Stride")
    plt.ylabel("Bandwidth (GB/s)")
    plt.title("Register to Shared Memory Bandwidth for Single SM (Read 8 KiB)")
    plt.legend(title="Threads per block", loc="upper right")

    # Log scale for stride
    xticks = [1, 2, 4, 8, 16, 32]
    plt.xscale("log")
    plt.xticks(xticks, [str(x) for x in xticks])
    plt.grid(True, which="major", linestyle="--", linewidth=0.5)
    plt.minorticks_off()
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to: {output_file}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        usage="python plot_threads.py <csv_file> [--output output_file]"
    )
    parser.add_argument("csv_file", type=Path, help="Path to the input CSV file")
    parser.add_argument("--output", type=Path, default=Path("../plots/plot_threads.png"))

    args = parser.parse_args()

    plot_register_shared(args.csv_file, args.output)


if __name__ == "__main__":
    main()
