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
    eval_files = list(SCENARIO_DIR.glob("step*_evaluation.csv"))
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
        # Extract step info from filename
        step_num = file_path.stem.split("_")[0]  # e.g., "step1" from "step1_evaluation"

        # Read CSV
        df = pd.read_csv(file_path)

        # Add metadata columns
        df.insert(0, "step", step_num)
        df.insert(1, "step_description", get_step_description(step_num))

        combined_data.append(df)
        print(f"Added {len(df)} rows from {file_path.name}")

    # Combine and save
    combined_df = pd.concat(combined_data, ignore_index=True)
    combined_df.to_csv(output_file, index=False)

    print(f"\n✅ Combined {len(combined_df)} rows into: {output_file}")
    print(f"Columns: {list(combined_df.columns)}")

    # Show summary
    print("\nSummary by step:")
    for step in combined_df["step"].unique():
        count = len(combined_df[combined_df["step"] == step])
        desc = combined_df[combined_df["step"] == step]["step_description"].iloc[0]
        print(f"  {step}: {desc} ({count} queries)")


def get_step_description(step_num: str) -> str:
    """Get description for each step."""
    descriptions = {
        "step1": "Data Filtering & Customer Cohort Identification",
        "step2": "ARPU Calculation",
        "step3": "Churn Rate Calculation",
        "step4": "LTV Calculation",
        "step5": "CAC to LTV Analysis",
    }
    return descriptions.get(step_num, "Unknown Step")


if __name__ == "__main__":
    main()
