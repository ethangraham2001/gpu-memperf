import os
from pathlib import Path

# Keep matplotlib cache writable inside the repo to avoid user-level permission issues
BASE_DIR = Path(__file__).resolve().parent.parent
MPL_DIR = BASE_DIR / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

from plot_utils import PlotConfig, plot_with_peak


def plot_l1_l2_one_block() -> None:
    warps = [1, 2, 4, 8, 16, 32]
    ys = [
        [9.063623, 18.088662, 34.468148, 38.537461, 38.311778, 40.231897],  # L1
        [1.687016, 3.032096, 6.100351, 10.047408, 10.133646, 10.131737],  # L2
    ]
    labels = ["L1 (1 block)", "L2 (1 block)"]
    palette = {
        "L1 (1 block)": "#1f77b4",
        "L2 (1 block)": "#e4572e",
    }
    cfg = PlotConfig(
        xlabel="Warps per block",
        ylabel="Bandwidth (GB/s)",
        title="Random access bandwidth vs warps (1 block)",
        xticks=warps,
        figsize=(7.2, 4.4),
        ylim=(0, max(max(series) for series in ys) * 1.2),
    )

    plot_with_peak(
        warps,
        ys,
        labels,
        outfile="l1_l2_one_block",
        cfg=cfg,
        palette=palette,
        fill_alpha=0.08,
        linewidth=2.2,
        legend_title="Cache mode",
        legend_loc="lower right",
    )


def plot_l2_multi_block() -> None:
    warps = [1, 2, 4, 8, 16, 32]
    data = {
        32: [53.068453, 95.419440, 149.069475, 143.936693, 143.962337, 143.787706],
        64: [95.689186, 149.182194, 151.596526, 153.599865, 153.698549, 153.754673],
        128: [142.369220, 144.578109, 140.216469, 139.839878, 140.469976, 140.262480],
    }
    block_counts = [32, 64, 128]
    ys = [data[count] for count in block_counts]
    labels = [f"{count} blocks (L2)" for count in block_counts]
    palette = {
        "32 blocks (L2)": "#5b8ff9",
        "64 blocks (L2)": "#f6bd60",
        "128 blocks (L2)": "#9c89b8",
    }
    cfg = PlotConfig(
        xlabel="Warps per block",
        ylabel="Bandwidth (GB/s)",
        title="Random access L2 bandwidth vs warps",
        xticks=warps,
        figsize=(7.2, 4.4),
        ylim=(0, max(max(series) for series in ys) * 1.1),
    )

    plot_with_peak(
        warps,
        ys,
        labels,
        outfile="l2_blocks",
        cfg=cfg,
        palette=palette,
        linewidth=2.2,
        legend_title="Block count",
        legend_loc="lower right",
    )


def main() -> None:
    plot_l1_l2_one_block()
    plot_l2_multi_block()
    print(f"Plots written to {BASE_DIR / 'plots'}")


if __name__ == "__main__":
    main()
