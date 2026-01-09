"""
Replicating the calculations done in this paper: https://www.pnas.org/doi/epdf/10.1073/pnas.2003773117 to have a metric that defines the hydrophobic clustering in a protein sequence 

Input:
- CSV with a column named "sequence"

Usage:
    python compute_HpC.py 
        --input_csv input_sequences.csv 
        --output_csv output_with_HpC.csv 
        --window_size 3
"""

import numpy as np
import random
import seaborn as sns
import pandas as pd
import argparse

# Define the MJHW scale 
#The Bowman et al paper uses a combination of the Hoop-Woods hydrophobicity scale and the Miyazawa–Jernigan hydrophobicity scale
MJHW_scale = {
    'A': -0.0645, 'R': -1.21, 'N': -0.551, 'D': -1.34, 'C': 0.502,
    'Q': -0.518, 'E': -1.35, 'G': -0.382, 'H': -0.0737, 'I': 1.14,
    'L': 1.33, 'K': -1.47, 'M': 0.767, 'F': 1.48, 'P': -0.424,
    'S': -0.540, 'T': -0.216, 'W': 1.29, 'Y': 0.812, 'V': 0.811
}

# Function to calculate the MJHW hydropathy value for a sequence
def calculate_hydropathy(sequence):
    return np.array([MJHW_scale[aa] for aa in sequence])

# Function to calculate HpC for a given sequence
def calculate_HpC(sequence, window_size=3):
    hydropathy_values = calculate_hydropathy(sequence)
    sliding_windows = [
        hydropathy_values[i:i+window_size]
        for i in range(len(hydropathy_values) - window_size + 1)
    ]

    window_means = np.array([window.mean() for window in sliding_windows])
    # Sum of positive area under the curve
    positive_auc = np.sum(window_means[window_means > 0])
    # Normalization by the number of windows
    HpC = positive_auc / len(sliding_windows)
    return HpC

def main():
    parser = argparse.ArgumentParser(description="Compute HpC for protein sequences")
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Input CSV file containing a 'sequence' column"
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Output CSV file with HpC values added"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=3,
        help="Sliding window size (default: 3)"
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    if "sequence" not in df.columns:
        raise ValueError("Input CSV must contain a 'sequence' column")

    HpC_vals = [
        calculate_HpC(seq, window_size=args.window_size)
        for seq in df["sequence"]
    ]

    df["HpC"] = HpC_vals
    df.to_csv(args.output_csv, index=False)

    print(f"HpC results saved to {args.output_csv}.")

