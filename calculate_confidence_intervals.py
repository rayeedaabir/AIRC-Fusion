import argparse
import pandas as pd
import numpy as np
from scipy import stats

def analyze_metrics(csv_file_path):
    """
    Loads per-frame metrics from a CSV and calculates the mean and 95% confidence interval.
    """
    print(f"\n--- Statistical Analysis for: {csv_file_path} ---")
    
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"ERROR: File not found at {csv_file_path}")
        return

    # Identify metric columns (exclude paths and identifiers)
    metric_columns = [col for col in df.columns if 'path' not in col.lower() and 'index' not in col.lower()]

    print(f"{'Metric':<25} | {'Mean':<12} | {'95% Confidence Interval':<25}")
    print("-" * 70)

    for column in metric_columns:
        # Ensure data is numeric and drop any missing values
        data = pd.to_numeric(df[column], errors='coerce').dropna()
        
        if len(data) < 2:
            print(f"{column:<25} | Not enough data to analyze.")
            continue

        # Calculate statistics
        mean = np.mean(data)
        n = len(data)
        # stats.sem calculates the standard error of the mean (std_dev / sqrt(n))
        std_error = stats.sem(data)

        # Calculate the 95% confidence interval for the mean
        # Using t-distribution is robust for any sample size
        ci = stats.t.interval(0.95, df=n-1, loc=mean, scale=std_error)

        print(f"{column:<25} | {mean:<12.5f} | [{ci[0]:.5f}, {ci[1]:.5f}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate 95% Confidence Intervals from Per-Frame Metrics CSV.")
    parser.add_argument("csv_file", help="Path to the per-frame metrics CSV file.")
    args = parser.parse_args()
    
    analyze_metrics(args.csv_file)