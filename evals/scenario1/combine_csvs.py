#!/usr/bin/env python3
"""
Simple script to combine all evaluation CSV files from scenario1 into one consolidated file.
Usage: python combine_csvs.py
"""

from pathlib import Path

import pandas as pd

SCENARIO_DIR = Path(__file__).parent


def main():
    """Combine all evaluation CSV files into one."""
    # Define paths
    output_file = SCENARIO_DIR / "combined_evaluations.csv"

    # Find all evaluation CSV files
    eval_files = list(SCENARIO_DIR.glob("*_evaluation.csv"))
    eval_files.sort()  # Sort to ensure consistent order

    if not eval_files:
        print("No evaluation CSV files found!")
        return

    print(f"Found {len(eval_files)} evaluation files:")
    for file in eval_files:
        print(f"  - {file.name}")

    # Combine files
    combined_data = []

    for file_path in eval_files:
        # Read CSV
        df = pd.read_csv(file_path)

        # Add source file column
        df.insert(0, "source_file", file_path.name)

        combined_data.append(df)
        print(f"Added {len(df)} rows from {file_path.name}")

    # Combine and save
    combined_df = pd.concat(combined_data, ignore_index=True)
    combined_df.to_csv(output_file, index=False)

    print(f"\n✅ Combined {len(combined_df)} rows into: {output_file}")
    print(f"Columns: {list(combined_df.columns)}")


if __name__ == "__main__":
    main()
