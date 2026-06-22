import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_temporal_metrics(csv_path_1, csv_path_2, model_name_1="Model 1", model_name_2="Model 2", output_dir="."):
    """
    Plots MATD_fused, MATD_vis, and MATD_ir against frame index for two models.
    """
    print(f"Loading CSV 1: {csv_path_1}")
    try:
        df1 = pd.read_csv(csv_path_1)
    except FileNotFoundError:
        print(f"Error: File not found {csv_path_1}")
        return

    print(f"Loading CSV 2: {csv_path_2}")
    try:
        df2 = pd.read_csv(csv_path_2)
    except FileNotFoundError:
        print(f"Error: File not found {csv_path_2}")
        return

    # Ensure required columns exist
    required_cols = ['MATD_fused', 'MATD_vis', 'MATD_ir']
    for col in required_cols:
        if col not in df1.columns:
            print(f"Error: Column '{col}' missing in {csv_path_1}. Run the updated pipeline first.")
            return
        if col not in df2.columns:
            print(f"Error: Column '{col}' missing in {csv_path_2}. Run the updated pipeline first.")
            return

    # Create the plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot for Model 1
    axes[0].plot(df1.index, df1['MATD_vis'], label='MATD Visible', color='blue', alpha=0.6, linewidth=1)
    axes[0].plot(df1.index, df1['MATD_ir'], label='MATD Infrared', color='red', alpha=0.6, linewidth=1)
    axes[0].plot(df1.index, df1['MATD_fused'], label='MATD Fused', color='green', linewidth=1.5)
    axes[0].set_title(f"{model_name_1}: Temporal Difference Over Time")
    axes[0].set_ylabel("Mean Absolute Temporal Difference")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot for Model 2
    axes[1].plot(df2.index, df2['MATD_vis'], label='MATD Visible', color='blue', alpha=0.6, linewidth=1)
    axes[1].plot(df2.index, df2['MATD_ir'], label='MATD Infrared', color='red', alpha=0.6, linewidth=1)
    axes[1].plot(df2.index, df2['MATD_fused'], label='MATD Fused', color='green', linewidth=1.5)
    axes[1].set_title(f"{model_name_2}: Temporal Difference Over Time")
    axes[1].set_xlabel("Frame Index")
    axes[1].set_ylabel("Mean Absolute Temporal Difference")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    
    output_filename = os.path.join(output_dir, f"temporal_comparison_{model_name_1}_vs_{model_name_2}.png")
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved to: {output_filename}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot temporal metrics comparison from two CSV files.")
    parser.add_argument("csv1", help="Path to the first CSV file")
    parser.add_argument("csv2", help="Path to the second CSV file")
    parser.add_argument("--name1", default="Model 1", help="Name of the first model")
    parser.add_argument("--name2", default="Model 2", help="Name of the second model")
    parser.add_argument("--output", default=".", help="Output directory for the plot")
    
    args = parser.parse_args()
    
    plot_temporal_metrics(args.csv1, args.csv2, args.name1, args.name2, args.output)