"""
Simple utility for making a scatter plot.

Disclaimer: written by Claude. We should make something better later on.
"""

import matplotlib.pyplot as plt
import pandas as pd
import sys


def plot_csv(
    csv_file, x_col="bytes", y_col="avg_access_latency", output_file="plot.png"
):
    """
    Simple CSV plotter

    Args:
        csv_file: Path to CSV file
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        output_file: Output filename for plot
    """
    # Read CSV
    df = pd.read_csv(csv_file)

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_col], df[y_col], marker="o", markersize=3, linewidth=1)

    # Labels and formatting
    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel(y_col.replace("_", " ").title())
    plt.xscale("log")
    plt.title(f'{y_col.replace("_", " ").title()} vs {x_col.replace("_", " ").title()}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save and show
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_file}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot.py <csv_file> [x_col] [y_col] [output_file]")
        sys.exit(1)

    csv_file = sys.argv[1]
    x_col = sys.argv[2] if len(sys.argv) > 2 else "bytes"
    y_col = sys.argv[3] if len(sys.argv) > 3 else "avg_access_latency"
    output_file = sys.argv[4] if len(sys.argv) > 4 else "plot.png"

    plot_csv(csv_file, x_col, y_col, output_file)
