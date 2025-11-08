import matplotlib.pyplot as plt
import io
import csv

# --- Your Data ---
# Store the raw data in a multiline string
# (This is the "new" format you provided)
data_str = """mode,tile_size,buf_size,fops,ms
sync,4096,1073741824,1,4.10499
sync,4096,1073741824,2,4.2456
sync,4096,1073741824,4,4.51251
sync,4096,1073741824,8,5.06886
sync,4096,1073741824,16,6.2727
sync,4096,1073741824,32,8.60522
sync,4096,1073741824,64,13.7645
sync,4096,1073741824,128,22.9389
sync,4096,1073741824,256,41.8392
sync,4096,1073741824,512,80.1684
async_2x,4096,1073741824,1,4.43971
async_2x,4096,1073741824,2,4.43059
async_2x,4096,1073741824,4,4.43318
async_2x,4096,1073741824,8,4.46605
async_2x,4096,1073741824,16,5.49779
async_2x,4096,1073741824,32,8.39427
async_2x,4096,1073741824,64,14.3294
async_2x,4096,1073741824,128,26.1569
async_2x,4096,1073741824,256,49.8782
async_2x,4096,1073741824,512,97.4375
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
title_tile_size = ""
title_buf_size = ""
first_row = True

# Loop over the data rows
for row in reader:
    # On the first data row, grab the constant values for the title
    if first_row:
        title_tile_size = row[1]
        title_buf_size = row[2]
        first_row = False

    # Get the values we care about
    mode_key = row[0]
    fops_val = int(row[3])
    ms_val = float(row[4])

    # Calculate GB/s (GiB/s)
    # tile_size is 1073741824 bytes (1 GiB)
    # time_in_s = ms_val / 1000.0
    # gib_per_s = 1 GiB / time_in_s = 1 / (ms_val / 1000.0) = 1000.0 / ms_val
    gb_per_s_val = 1000.0 / ms_val

    # Append the data to the correct list based on the 'mode' column
    if mode_key in plot_data:
        plot_data[mode_key]["fops"].append(fops_val)
        plot_data[mode_key]["gb_per_s"].append(gb_per_s_val)

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
ax.set_title(f"Performance (Tile Size: {title_tile_size}, Buf Size: {title_buf_size})")

# The 'fops' data is powers of 2 (1, 2, 4...), so a log scale is best
ax.set_xscale("log")
# The bandwidth data will decay exponentially, so log scale is good here too
ax.set_yscale("log")

# To make the log x-axis clearer, set the ticks to the actual fops values
if plot_data["sync"]["fops"]:
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
    # plt.show() # Uncomment this line if you want to display the plot
except Exception as e:
    print(f"Error saving plot: {e}")
