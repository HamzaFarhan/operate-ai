"""
Step 4: Customer Lifetime Value (LTV) Calculation
Generates ground truth for LTV calculations using the formula: LTV = (ARPU / Churn Rate) × Profit Margin.
Combines results from Step 2 (ARPU) and Step 3 (Churn Rate).
"""

import json
from pathlib import Path

import pandas as pd


def get_jan_2023_cohort_ltv():
    """
    Calculates LTV for customers who were active in Jan 2023.
    Uses formula: LTV = (ARPU / Churn Rate) × Profit Margin

    Returns:
        dict: LTV calculations by various segments
    """
    # Import functions from previous steps
    import sys

    sys.path.append("evals/scenario1")
    from step2_arpu_calculation import get_jan_2023_cohort_arpu
    from step3_churn_rate_calculation import get_jan_2023_cohort_churn_rates

    # Get ARPU and churn rate data
    arpu_data = get_jan_2023_cohort_arpu()
    churn_data = get_jan_2023_cohort_churn_rates()

    # Profit margin assumption (typically 70-80% for SaaS companies)
    PROFIT_MARGIN = 0.75  # 75% profit margin

    def calculate_ltv(arpu, churn_rate):
        """Calculate LTV using the formula"""
        if churn_rate == 0:
            # If no churn, use 5-year assumption (20% annual churn)
            churn_rate = 0.2
        return round((arpu / churn_rate) * PROFIT_MARGIN, 2)

    # Calculate LTV by Subscription Type
    ltv_by_sub_type = {}
    arpu_sub = arpu_data["arpu_by_subscription_type"]
    churn_sub = churn_data["churn_by_subscription_type"]

    for sub_type in arpu_sub.keys():
        if sub_type in churn_sub:
            ltv_by_sub_type[sub_type] = calculate_ltv(arpu_sub[sub_type], churn_sub[sub_type])

    # Calculate LTV by Plan Type
    ltv_by_plan_type = {}
    arpu_plan = arpu_data["arpu_by_plan_type"]
    churn_plan = churn_data["churn_by_plan_type"]

    for plan_type in arpu_plan.keys():
        if plan_type in churn_plan:
            ltv_by_plan_type[plan_type] = calculate_ltv(arpu_plan[plan_type], churn_plan[plan_type])

    # Calculate LTV by Industry
    ltv_by_industry = {}
    arpu_industry = arpu_data["arpu_by_industry"]
    churn_industry = churn_data["churn_by_industry"]

    for industry in arpu_industry.keys():
        if industry in churn_industry:
            ltv_by_industry[industry] = calculate_ltv(arpu_industry[industry], churn_industry[industry])

    # Calculate LTV by Acquisition Channel
    ltv_by_channel = {}
    arpu_channel = arpu_data["arpu_by_channel"]
    churn_channel = churn_data["churn_by_channel"]

    for channel in arpu_channel.keys():
        if channel in churn_channel:
            ltv_by_channel[channel] = calculate_ltv(arpu_channel[channel], churn_channel[channel])

    return {
        "ltv_by_subscription_type": ltv_by_sub_type,
        "ltv_by_plan_type": ltv_by_plan_type,
        "ltv_by_industry": ltv_by_industry,
        "ltv_by_channel": ltv_by_channel,
        "profit_margin_used": PROFIT_MARGIN,
        "ltv_formula": "LTV = (ARPU / Churn Rate) × Profit Margin",
        "underlying_arpu_data": arpu_data,
        "underlying_churn_data": churn_data,
    }


def generate_step4_evaluation():
    """Generate evaluation dataset for Step 4"""

    # Get ground truth
    ground_truth = get_jan_2023_cohort_ltv()

    # Create evaluation cases
    eval_cases = [
        {
            "query": "Calculate the Customer Lifetime Value (LTV) for the Jan 2023 customer cohort by subscription type (Monthly, Annual, and Total combined)",
            "expected_output": json.dumps(ground_truth["ltv_by_subscription_type"], indent=2),
            "strategy": "Use the formula LTV = (ARPU / Churn Rate) × Profit Margin. Take ARPU by subscription type from Step 2, churn rates from Step 3, apply 75% profit margin. Higher ARPU and lower churn = higher LTV. Annual customers should have higher LTV than monthly.",
        },
        {
            "query": "Show LTV breakdown by initial plan type (Basic, Pro, Enterprise) for the Jan 2023 cohort",
            "expected_output": json.dumps(ground_truth["ltv_by_plan_type"], indent=2),
            "strategy": "Calculate LTV using ARPU and churn rates by plan type. Enterprise customers should have highest LTV (high ARPU, low churn), Basic should have lowest LTV (low ARPU, high churn). This shows customer segment value.",
        },
        {
            "query": "Calculate LTV by industry segment for customers who were active in January 2023",
            "expected_output": json.dumps(ground_truth["ltv_by_industry"], indent=2),
            "strategy": "Apply LTV formula using industry-specific ARPU and churn rates. Shows which industries provide most valuable long-term customers. Industries with high ARPU and low churn will have highest LTV.",
        },
        {
            "query": "What is the LTV by acquisition channel for the January 2023 customer cohort?",
            "expected_output": json.dumps(ground_truth["ltv_by_channel"], indent=2),
            "strategy": "Calculate LTV using channel-specific ARPU and churn rates. Reveals which marketing channels bring the most valuable customers over their lifetime. Channels with low churn and high ARPU generate highest LTV.",
        },
        {
            "query": "What profit margin was used in the LTV calculation and what is the formula?",
            "expected_output": f"Profit Margin: {ground_truth['profit_margin_used']}, Formula: {ground_truth['ltv_formula']}",
            "strategy": "The LTV calculation uses a 75% profit margin assumption (typical for SaaS) and the standard formula: LTV = (Average Revenue Per User / Churn Rate) × Profit Margin. This accounts for the fact that not all revenue becomes profit.",
        },
    ]

    # Save to CSV
    eval_df = pd.DataFrame(eval_cases)
    output_path = Path("evals/scenario1/step4_evaluation.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(output_path, index=False)

    print(f"Step 4 evaluation dataset saved to {output_path}")
    print("Function used: get_jan_2023_cohort_ltv()")
    print(f"Generated {len(eval_cases)} evaluation cases")
    print(f"Profit margin used: {ground_truth['profit_margin_used']}")
    print(f"Formula: {ground_truth['ltv_formula']}")

    # Print sample LTV values for verification
    print("\nSample LTV Results:")
    print(f"Total LTV: ${ground_truth['ltv_by_subscription_type']['Total']}")
    print(f"Enterprise LTV: ${ground_truth['ltv_by_plan_type']['Enterprise']}")
    print(f"Annual LTV: ${ground_truth['ltv_by_subscription_type']['Annual']}")

    return eval_df


if __name__ == "__main__":
    generate_step4_evaluation()
