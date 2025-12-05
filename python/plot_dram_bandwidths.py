from __future__ import annotations

from plot_utils import PlotConfig, plot_with_error_bars

WARP_COUNTS = [1, 2, 4, 8, 16, 32]

# Raw measurements: three runs per working-set/warp combination.
# Values marked "MB" are converted to GB/s (decimal) to keep units consistent.
RAW_MEASUREMENTS = {
    "16 MB": {
        1: [(1.631994, "GB"), (1.638555, "GB"), (1.636050, "GB")],
        2: [(3.182466, "GB"), (3.171050, "GB"), (3.170679, "GB")],
        4: [(6.067206, "GB"), (6.062547, "GB"), (6.059130, "GB")],
        8: [(9.898586, "GB"), (9.907257, "GB"), (9.910324, "GB")],
        16: [(10.137519, "GB"), (10.137531, "GB"), (10.137955, "GB")],
        32: [(10.149310, "GB"), (10.149434, "GB"), (10.149207, "GB")],
    },
    "32 MB": {
        1: [(748.061779, "MB"), (748.724708, "MB"), (761.997835, "MB")],
        2: [(1.487825, "GB"), (1.489116, "GB"), (1.517635, "GB")],
        4: [(2.924483, "GB"), (2.923058, "GB"), (2.984497, "GB")],
        8: [(5.654606, "GB"), (5.654369, "GB"), (5.769170, "GB")],
        16: [(9.185391, "GB"), (9.186361, "GB"), (9.460186, "GB")],
        32: [(9.078312, "GB"), (9.078075, "GB"), (9.330706, "GB")],
    },
    "64 MB": {
        1: [(725.519103, "MB"), (725.064632, "MB"), (728.053554, "MB")],
        2: [(1.430390, "GB"), (1.430789, "GB"), (1.430755, "GB")],
        4: [(2.806560, "GB"), (2.800007, "GB"), (2.817103, "GB")],
        8: [(5.409953, "GB"), (5.404371, "GB"), (5.415174, "GB")],
        16: [(8.739165, "GB"), (8.750847, "GB"), (8.727647, "GB")],
        32: [(8.722638, "GB"), (8.733912, "GB"), (8.711226, "GB")],
    },
    "128 MB": {
        1: [(720.850433, "MB"), (718.563843, "MB"), (717.755077, "MB")],
        2: [(1.415237, "GB"), (1.414842, "GB"), (1.416769, "GB")],
        4: [(2.786411, "GB"), (2.774211, "GB"), (2.772725, "GB")],
        8: [(5.331839, "GB"), (5.322631, "GB"), (5.325588, "GB")],
        16: [(8.510022, "GB"), (8.506373, "GB"), (8.555143, "GB")],
        32: [(8.514101, "GB"), (8.522351, "GB"), (8.560297, "GB")],
    },
}

BLOCK_COUNT_MEASUREMENTS = {
    1: {
        1: [(720.850433, "MB"), (718.563843, "MB"), (717.755077, "MB")],
        2: [(1.415237, "GB"), (1.414842, "GB"), (1.416769, "GB")],
        4: [(2.786411, "GB"), (2.774211, "GB"), (2.772725, "GB")],
        8: [(5.331839, "GB"), (5.322631, "GB"), (5.325588, "GB")],
        16: [(8.510022, "GB"), (8.506373, "GB"), (8.555143, "GB")],
        32: [(8.514101, "GB"), (8.522351, "GB"), (8.560297, "GB")],
    },
    2: {
        1: [(1.424502, "GB"), (1.426312, "GB"), (1.425449, "GB")],
        2: [(2.777625, "GB"), (2.784437, "GB"), (2.782283, "GB")],
        4: [(5.363553, "GB"), (5.372602, "GB"), (5.368101, "GB")],
        8: [(9.340167, "GB"), (9.332520, "GB"), (9.333614, "GB")],
        16: [(11.023281, "GB"), (11.021354, "GB"), (11.021779, "GB")],
        32: [(10.977723, "GB"), (10.976764, "GB"), (10.977552, "GB")],
    },
    16: {
        1: [(9.974170, "GB")],
        2: [(15.178795, "GB")],
        4: [(17.102425, "GB")],
        8: [(17.251450, "GB")],
        16: [(17.270078, "GB")],
        32: [(17.266122, "GB")],
    },
    18: {
        1: [(12.193882, "GB"), (12.189464, "GB"), (12.185615, "GB")],
        2: [(19.597815, "GB"), (20.156860, "GB"), (19.885064, "GB")],
        4: [(38.769562, "GB"), (38.925766, "GB"), (38.854332, "GB")],
        8: [(63.408081, "GB"), (63.457337, "GB"), (63.576497, "GB")],
        16: [(56.725062, "GB"), (57.094788, "GB"), (55.157249, "GB")],
        32: [(59.460635, "GB"), (58.970845, "GB"), (60.849373, "GB")],
    },
    36: {
        1: [(22.267553, "GB"), (22.226364, "GB"), (22.242684, "GB")],
        2: [(37.776543, "GB"), (37.756280, "GB"), (37.820565, "GB")],
        4: [(68.809994, "GB"), (68.672085, "GB"), (68.702693, "GB")],
        8: [(63.027584, "GB"), (66.280872, "GB"), (66.062672, "GB")],
        16: [(37.145135, "GB"), (30.835722, "GB"), (35.792433, "GB")],
        32: [(30.335357, "GB"), (31.297159, "GB"), (30.674856, "GB")],
    },
    72: {
        1: [(36.132352, "GB"), (35.937962, "GB"), (35.976241, "GB")],
        2: [(57.756814, "GB"), (57.836297, "GB"), (56.744124, "GB")],
        4: [(75.708836, "GB"), (74.762756, "GB"), (73.751321, "GB")],
        8: [(94.474286, "GB"), (95.076218, "GB"), (96.799033, "GB")],
        16: [(34.673750, "GB"), (33.886839, "GB"), (34.225501, "GB")],
        32: [(37.956514, "GB"), (37.115902, "GB"), (37.901977, "GB")],
    },
    108: {
        1: [(46.156345, "GB"), (46.705681, "GB"), (46.221821, "GB")],
        2: [(91.814604, "GB"), (94.413343, "GB"), (93.754700, "GB")],
        4: [(93.152796, "GB"), (94.136201, "GB"), (93.765226, "GB")],
        8: [(33.150338, "GB"), (33.186336, "GB"), (32.942672, "GB")],
        16: [(20.753717, "GB"), (20.447885, "GB"), (20.404255, "GB")],
        32: [(37.418542, "GB"), (38.657222, "GB"), (36.012664, "GB")],
    },
}


def mb_to_gb(value: float, unit: str) -> float:
    if unit.lower().startswith("mb"):
        return value / 1000.0
    return value


def prepare_triplets(measurements, label_fn):
    triplets = []
    labels = []
    for key, runs in measurements.items():
        labels.append(label_fn(key))
        series = []
        for warp in WARP_COUNTS:
            raw_values = runs.get(warp, [])
            if not raw_values:
                raise ValueError(f"Missing measurements for warp {warp} and key {key}.")
            values = [mb_to_gb(val, unit) for val, unit in raw_values]
            low = min(values)
            high = max(values)
            mid = sum(values) / len(values)
            series.append((low, mid, high))
        triplets.append(series)
    return triplets, labels


def plot_dram_bandwidths():
    y_triplets, labels = prepare_triplets(RAW_MEASUREMENTS, lambda ws: f"{ws} working set")
    cfg = PlotConfig(
        xlabel="Warps per block",
        ylabel="Bandwidth (GB/s)",
        title="DRAM random access bandwidth vs warps",
        xticks=WARP_COUNTS,
        figsize=(7.5, 4.6),
        ylim=(0, 11),
    )
    palette = [
        "#1f77b4",
        "#f28e2c",
        "#8cd17d",
        "#d62728",
    ]
    markers = ["o", "s", "^", "D"]
    plot_with_error_bars(
        WARP_COUNTS,
        y_triplets,
        labels,
        outfile="dram_best_working_set",
        cfg=cfg,
        palette=palette,
        markers=markers,
        linewidth=2.0,
        capsize=4.5,
        fill_alpha=0.08,
        legend_title="Working set size",
        legend_loc="lower right",
    )


def plot_block_count_sweep():
    y_triplets, labels = prepare_triplets(BLOCK_COUNT_MEASUREMENTS, lambda n: f"{n} blocks")
    cfg = PlotConfig(
        xlabel="Warps per block",
        ylabel="Bandwidth (GB/s)",
        title="DRAM bandwidth vs warps (128 MB, varying blocks)",
        xticks=WARP_COUNTS,
        figsize=(7.8, 4.8),
        ylim=(0, 110),
    )
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    ]
    markers = ["o", "s", "^", "D", "v", "P", "*"]
    plot_with_error_bars(
        WARP_COUNTS,
        y_triplets,
        labels,
        outfile="dram_block_counts",
        cfg=cfg,
        palette=palette,
        markers=markers,
        linewidth=2.2,
        capsize=4.5,
        fill_alpha=0.08,
        legend_title="Blocks",
        legend_loc="upper right",
    )


def main():
    plot_dram_bandwidths()
    plot_block_count_sweep()


if __name__ == "__main__":
    main()
