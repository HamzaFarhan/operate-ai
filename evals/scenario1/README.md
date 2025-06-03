# Scenario 1 Evaluation Files

This directory contains evaluation datasets for a 5-step SaaS customer analytics scenario.

## Files

### Individual Step Evaluations
- `step1_evaluation.csv` - Data Filtering & Customer Cohort Identification (5 queries)
- `step2_evaluation.csv` - ARPU Calculation (5 queries)  
- `step3_evaluation.csv` - Churn Rate Calculation (5 queries)
- `step4_evaluation.csv` - LTV Calculation (5 queries)
- `step5_evaluation.csv` - CAC to LTV Analysis (5 queries)

### Combined Evaluation
- `combined_evaluations.csv` - All 25 queries combined with step metadata

### Combination Script
- `combine_csvs.py` - Script to combine all evaluation CSV files

## CSV Structure

Each evaluation CSV contains:
- `query` - The question/task to evaluate
- `expected_output` - The ground truth answer
- `strategy` - Explanation of how to solve the query

The combined CSV adds:
- `step` - Step identifier (step1, step2, etc.)
- `step_description` - Human-readable step description

## Combining CSVs

Run the combination script from the scenario1 directory:

```bash
uv run evals/scenario1/combine_csvs.py
```

The script will:
1. Automatically find all `step*_evaluation.csv` files
2. Add step metadata columns
3. Combine into `combined_evaluations.csv`
4. Show summary statistics

## Scenario Overview

This evaluation tests a complete SaaS customer analytics workflow:

1. **Step 1**: Identify customers active in January 2023
2. **Step 2**: Calculate Average Revenue Per User (ARPU) for 2023
3. **Step 3**: Calculate churn rates during 2023
4. **Step 4**: Calculate Customer Lifetime Value (LTV) using ARPU and churn
5. **Step 5**: Analyze Customer Acquisition Cost (CAC) to LTV ratios

Each step builds on the previous ones, testing both individual calculations and end-to-end analytical reasoning. 