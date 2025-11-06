import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

dirname = os.path.dirname(__file__)

def plot_csv(csv_file, output_file):


    df = pd.read_csv("../logs/" + csv_file)

    
    # Convert bandwidth to GB/s
    df['bandwidth_tbps'] = df['bandwidth'].astype(float) / 1e12

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['stride'], df['bandwidth_tbps'], 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Stride', fontsize=12)
    plt.ylabel('Bandwidth (TB/s)', fontsize=12)
    plt.title('Memory Bandwidth vs Stride Pattern', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('linear')  # Log scale for better visualization
    plt.xscale('log')  # Log scale for stride as well
    plt.xticks(df['stride'], df['stride'])

    plt.ylim(bottom=df['bandwidth_tbps'].min() * 0.5, top=df['bandwidth_tbps'].max() * 1.2)

    # Add annotations for key points
    for i, row in df.iterrows():
        plt.annotate(f'{row["bandwidth_tbps"]:.2f}', 
                    (row['stride'], row['bandwidth_tbps']),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center', fontsize=9)

    plt.savefig(dirname + "/../plots/" + output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage='%(prog)s [options]')
    parser.add_argument('csv_file')
    parser.add_argument('--output_file', default= "plot.png")

    args = parser.parse_args()

    plot_csv(args.csv_file, args.output_file)