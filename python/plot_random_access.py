import pandas as pd
from pathlib import Path
from plot_utils import PlotConfig, plot_with_error_bars_raw, plot_with_box_plots

def load_results(results_dir: Path):
    results = {}
    for sub in results_dir.iterdir():
        if sub.is_dir() and "random_access" in sub.name:
            csv_path = sub / "result.csv"
            if csv_path.exists():
                results[sub.name] = pd.read_csv(csv_path)
    return results

def prepare_data_for_plotting(df):
    if "num_blocks" not in df.columns:
        df["num_blocks"] = 1
        
    grouped = df.groupby(["num_blocks", "num_warps"])["bandwidth"]
    
    data = {}
    all_warps = set()
    
    for (blocks, warps), group in grouped:
        if blocks not in data:
            data[blocks] = {}
        # Convert to GB/s
        data[blocks][warps] = (group / 1e9).tolist() 
        all_warps.add(warps)
        
    sorted_warps = sorted(list(all_warps))
    
    return sorted_warps, data

def prepare_series_list(x_vals, block_data):
    series = []
    for x in x_vals:
        series.append(block_data.get(x, []))
    return series

def plot_single_with_error_bars(name, df, plot_dir):
    x_vals, data_by_blocks = prepare_data_for_plotting(df)
    block_counts = sorted(data_by_blocks.keys())
    y_raw_list = []
    labels = []
    
    for blocks in block_counts:
        y_raw = prepare_series_list(x_vals, data_by_blocks[blocks])
        y_raw_list.append(y_raw)
        labels.append(f"{blocks} blocks")
    
    label = name.replace("random_access_", "").upper()
    title = f"Random Access Bandwidth - {label}"
    outfile = plot_dir / f"{name}_bandwidth"
    
    cfg = PlotConfig(
        xlabel="Warps per block",
        ylabel="Bandwidth (GB/s)",
        title=title,
        xticks=x_vals,
        figsize=(7.5, 4.6),
    )
    
    plot_with_error_bars_raw(
        x_vals,
        y_raw_list,
        labels,
        outfile=outfile,
        cfg=cfg,
        legend_title="Block Count",
        legend_loc="best",
        fill_alpha=0.15
    )

def plot_single_box_plot(name, df, plot_dir):
    x_vals, data_by_blocks = prepare_data_for_plotting(df)
    block_counts = sorted(data_by_blocks.keys())
    y_raw_list = []
    labels = []
    
    for blocks in block_counts:
        y_raw = prepare_series_list(x_vals, data_by_blocks[blocks])
        y_raw_list.append(y_raw)
        labels.append(f"{blocks} blocks")
    
    label = name.replace("random_access_", "").upper()
    title = f"Random Access Bandwidth (Box Plot) - {label}"
    outfile = plot_dir / f"{name}_bandwidth_boxplot"
    
    cfg = PlotConfig(
        xlabel="Warps per block",
        ylabel="Bandwidth (GB/s)",
        title=title,
        xticks=x_vals,
        figsize=(7.5, 4.6),
    )
    
    plot_with_box_plots(
        x_vals,
        y_raw_list,
        labels,
        outfile=outfile,
        cfg=cfg,
        legend_title="Block Count",
        legend_loc="best"
    )

def plot_combined(results, plot_dir):
    mapping = {
        "random_access_l1": 1,
        "random_access_l2": 108,
        "random_access_dram": 72
    }
    
    valid_keys = [k for k in mapping.keys() if k in results]
    if not valid_keys:
        return

    y_raw_list = []
    labels = []
    x_vals_final = None 

    palette = []
    colors = {"random_access_l1": "#1f77b4", "random_access_l2": "#ff7f0e", "random_access_dram": "#2ca02c"}

    for k in valid_keys:
        target_blocks = mapping[k]
        df = results[k]
        
        if "num_blocks" in df.columns:
            df_filtered = df[df["num_blocks"] == target_blocks]
        else:
            df_filtered = df
            
        if df_filtered.empty:
            print(f"Warning: No data for {k} with {target_blocks} blocks. Skipping.")
            continue
            
        x_vals, data_by_blocks = prepare_data_for_plotting(df_filtered)
        
        if target_blocks in data_by_blocks:
             series = prepare_series_list(x_vals, data_by_blocks[target_blocks])
        elif len(data_by_blocks) == 1:
             key = list(data_by_blocks.keys())[0]
             series = prepare_series_list(x_vals, data_by_blocks[key])
        else:
             continue

        if x_vals_final is None:
            x_vals_final = x_vals
        
        y_raw_list.append(series)
        labels.append(k.replace("random_access_", "").upper())
        palette.append(colors.get(k, "black"))

    if not y_raw_list:
        return

    cfg = PlotConfig(
        xlabel="Warps per block",
        ylabel="Bandwidth (GB/s)",
        title="Random Access Bandwidth - Combined",
        xticks=x_vals_final,
        figsize=(7.5, 4.6), 
        logy=True,
    )
    outfile = plot_dir / "random_combined_bandwidth"

    plot_with_error_bars_raw(
        x_vals_final,
        y_raw_list,
        labels,
        outfile=outfile,
        cfg=cfg,
        palette=palette,
        legend_title="Memory Level",
        legend_loc="best",
        fill_alpha=0.15
    )

def plot_combined_box_plot(results, plot_dir):
    mapping = {
        "random_access_l1": 1,
        "random_access_l2": 108,
        "random_access_dram": 72
    }
    
    valid_keys = [k for k in mapping.keys() if k in results]
    if not valid_keys:
        return

    y_raw_list = []
    labels = []
    x_vals_final = None 

    palette = []
    colors = {"random_access_l1": "#1f77b4", "random_access_l2": "#ff7f0e", "random_access_dram": "#2ca02c"}

    for k in valid_keys:
        target_blocks = mapping[k]
        df = results[k]
        
        if "num_blocks" in df.columns:
            df_filtered = df[df["num_blocks"] == target_blocks]
        else:
            df_filtered = df
            
        if df_filtered.empty:
            continue
            
        x_vals, data_by_blocks = prepare_data_for_plotting(df_filtered)
        
        if target_blocks in data_by_blocks:
             series = prepare_series_list(x_vals, data_by_blocks[target_blocks])
        elif len(data_by_blocks) == 1:
             key = list(data_by_blocks.keys())[0]
             series = prepare_series_list(x_vals, data_by_blocks[key])
        else:
             continue

        if x_vals_final is None:
            x_vals_final = x_vals
        
        y_raw_list.append(series)
        labels.append(k.replace("random_access_", "").upper())
        palette.append(colors.get(k, "black"))

    if not y_raw_list:
        return

    cfg = PlotConfig(
        xlabel="Warps per block",
        ylabel="Bandwidth (GB/s)",
        title="Random Access Bandwidth - Combined (Box Plot)",
        xticks=x_vals_final,
        figsize=(7.5, 4.6), 
        logy=True,
    )
    outfile = plot_dir / "random_combined_bandwidth_boxplot"

    plot_with_box_plots(
        x_vals_final,
        y_raw_list,
        labels,
        outfile=outfile,
        cfg=cfg,
        palette=palette,
        legend_title="Memory Level",
        legend_loc="best"
    )

def plot_all(results_dir: Path, plot_dir: Path):
    if not results_dir.exists():
        print(f"Results directory {results_dir} does not exist.")
        return

    results = load_results(results_dir)
    
    if not results:
        print("No random access results found.")
        return

    # Create subfolder for random access plots
    ra_plot_dir = plot_dir / "random_access"
    ra_plot_dir.mkdir(parents=True, exist_ok=True)

    # Plot individual
    for name, df in results.items():
        plot_single_with_error_bars(name, df, ra_plot_dir)
        plot_single_box_plot(name, df, ra_plot_dir)
            
    # Plot combined
    plot_combined(results, ra_plot_dir)
    plot_combined_box_plot(results, ra_plot_dir)

if __name__ == "__main__":
    # Standalone execution
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="orchestrator_out")
    parser.add_argument("--out", type=str, default="plots")
    args = parser.parse_args()
    
    plot_all(Path(args.results), Path(args.out))
