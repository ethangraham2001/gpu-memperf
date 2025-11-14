import matplotlib.pyplot as plt
import io
import csv

# --- Your Data ---
# Store the raw data in a multiline string
data_str = """mode,tile_size,buf_size,fops,ms
4096,1073741824,sync,1,4.02432
4096,1073741824,sync,2,4.18269
4096,1073741824,sync,4,4.48291
4096,1073741824,sync,8,5.0889
4096,1073741824,sync,16,6.30874
4096,1073741824,sync,32,8.82214
4096,1073741824,sync,64,13.8136
4096,1073741824,sync,128,24.0459
4096,1073741824,sync,256,44.1522
4096,1073741824,sync,512,84.6523
4096,1073741824,async_2x,1,4.40838
4096,1073741824,async_2x,2,4.43014
4096,1073741824,async_2x,4,4.41203
4096,1073741824,async_2x,8,4.40266
4096,1073741824,async_2x,16,5.11882
4096,1073741824,async_2x,32,7.46371
4096,1073741824,async_2x,64,12.2995
4096,1073741824,async_2x,128,21.982
4096,1073741824,async_2x,256,41.5838
4096,1073741824,async_2x,512,80.7122
"""

# --- Data Parsing ---
# Use a dictionary to store the data for each line
plot_data = {
    "sync": {"fops": [], "gb_per_s": []},
    "async_2x": {"fops": [], "gb_per_s": []},
}

# Use io.StringIO to treat the string as a file
# Use csv.reader to parse the CSV data
f = io.StringIO(data_str)
reader = csv.reader(f)

# Read and skip the header row
header = next(reader)

# Variables to store the constant values for the title
title_mode = ""
title_tile_size = ""
first_row = True

# Loop over the data rows
for row in reader:
    # On the first data row, grab the constant values for the title
    if first_row:
        title_mode = row[0]
        title_tile_size = row[1]
        first_row = False

    # Get the values we care about
    buf_size = row[2]
    fops_val = int(row[3])
    ms_val = float(row[4])

    # Calculate GB/s (GiB/s)
    # tile_size is 1073741824 bytes (1 GiB)
    # time_in_s = ms_val / 1000.0
    # gib_per_s = 1 GiB / time_in_s = 1 / (ms_val / 1000.0) = 1000.0 / ms_val
    gb_per_s_val = 1000.0 / ms_val

    # Append the data to the correct list based on the 'buf_size' column
    if buf_size in plot_data:
        plot_data[buf_size]["fops"].append(fops_val)
        plot_data[buf_size]["gb_per_s"].append(gb_per_s_val)

# --- Plotting ---
# Create a figure and an axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the 'sync' data
ax.plot(
    plot_data["sync"]["fops"],
    plot_data["sync"]["gb_per_s"],
    label="sync",
    marker="o",
    linestyle="-",
)

# Plot the 'async_2x' data
ax.plot(
    plot_data["async_2x"]["fops"],
    plot_data["async_2x"]["gb_per_s"],
    label="async_2x",
    marker="x",
    linestyle="--",
)

# --- Customize the Plot ---
# Set labels
ax.set_xlabel("Number of F-Operations (fops)")
ax.set_ylabel("Bandwidth (GiB/s)")

# Set title using the values we saved
ax.set_title(f"Performance (Mode: {title_mode}, Tile Size: {title_tile_size})")

# The 'fops' data is powers of 2 (1, 2, 4...), so a log scale is best
ax.set_xscale("log")
# The bandwidth data will decay exponentially, so log scale is good here too
ax.set_yscale("log")

# To make the log x-axis clearer, set the ticks to the actual fops values
fops_ticks = plot_data["sync"]["fops"]
ax.set_xticks(fops_ticks)
ax.set_xticklabels(fops_ticks)  # Show "64", "128" instead of 10^1.8, 10^2.1

# Add a legend
ax.legend()

# Add a grid for readability
ax.grid(True, which="both", ls="--", alpha=0.6)

# --- Save the Output ---
output_filename = "fops_vs_ms_plot.png"
try:
    plt.savefig(output_filename)
    print(f"Successfully saved plot to {output_filename}")
except Exception as e:
    print(f"Error saving plot: {e}")
