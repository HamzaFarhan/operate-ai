"""
Step 1: Data Filtering & Customer Cohort Identification
Generates ground truth for identifying customers active in Jan 2023 and their initial subscription details.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def get_active_customers_jan_2023():
    """
    Identifies customers who were active at the start of Jan 2023 and returns their cohort details.

    Returns:
        dict: Contains counts and breakdowns by various segments
    """
    # Load the generated data
    data_dir = Path("operateai_scenario1_data")

    customers_df = pd.read_csv(data_dir / "customers.csv")
    subscriptions_df = pd.read_csv(data_dir / "subscriptions.csv")

    # Convert dates to datetime
    subscriptions_df["StartDate"] = pd.to_datetime(subscriptions_df["StartDate"])
    subscriptions_df["EndDate"] = pd.to_datetime(subscriptions_df["EndDate"], errors="coerce")

    # Define the period
    jan_2023_start = datetime(2023, 1, 1)
    jan_2023_end = datetime(2023, 1, 31)

    # Find subscriptions active in Jan 2023
    # Active means: started before or during Jan 2023 AND (ended after Jan 1 2023 OR still active)
    active_subs_jan_2023 = subscriptions_df[
        (subscriptions_df["StartDate"] <= jan_2023_end)
        & ((subscriptions_df["EndDate"] >= jan_2023_start) | pd.isna(subscriptions_df["EndDate"]))
    ]

    # Get unique active customers
    active_customer_ids = active_subs_jan_2023["CustomerID"].unique()

    # For each active customer, get their INITIAL subscription details (first subscription)
    initial_subs = []
    for customer_id in active_customer_ids:
        customer_subs = subscriptions_df[subscriptions_df["CustomerID"] == customer_id].sort_values("StartDate")
        if len(customer_subs) > 0:
            initial_sub = customer_subs.iloc[0]
            initial_subs.append(
                {
                    "CustomerID": customer_id,
                    "InitialPlanName": initial_sub["PlanName"],
                    "InitialSubscriptionType": initial_sub["SubscriptionType"],
                    "InitialStartDate": initial_sub["StartDate"],
                }
            )

    initial_subs_df = pd.DataFrame(initial_subs)

    # Merge with customer details
    cohort_df = initial_subs_df.merge(customers_df, on="CustomerID", how="left")

    # Generate summary statistics
    result = {
        "total_active_customers": len(active_customer_ids),
        "by_subscription_type": cohort_df["InitialSubscriptionType"].value_counts().to_dict(),
        "by_plan_type": cohort_df["InitialPlanName"].value_counts().to_dict(),
        "by_industry": cohort_df["IndustrySegment"].value_counts().to_dict(),
        "by_acquisition_channel": cohort_df["AcquisitionChannel"].value_counts().to_dict(),
        "by_geography": cohort_df["Geography"].value_counts().to_dict(),
        "by_customer_type": cohort_df["CustomerType"].value_counts().to_dict(),
    }

    return result


def generate_step1_evaluation():
    """Generate evaluation dataset for Step 1"""

    # Get ground truth
    ground_truth = get_active_customers_jan_2023()

    # Create evaluation cases
    eval_cases = [
        {
            "query": "How many customers were active in January 2023?",
            "expected_output": str(ground_truth["total_active_customers"]),
            "strategy": "Count unique customers who had active subscriptions during January 2023. A subscription is active if it started before or during Jan 2023 AND either ended after Jan 1 2023 or is still ongoing.",
        },
        {
            "query": "Show me the breakdown of active Jan 2023 customers by their initial subscription type (Monthly vs Annual)",
            "expected_output": json.dumps(ground_truth["by_subscription_type"], indent=2),
            "strategy": "For each customer active in Jan 2023, look at their very first subscription to determine if they initially chose Monthly or Annual billing. Use initial choice, not current status.",
        },
        {
            "query": "Break down active Jan 2023 customers by their initial plan type (Basic, Pro, Enterprise)",
            "expected_output": json.dumps(ground_truth["by_plan_type"], indent=2),
            "strategy": "For each customer active in Jan 2023, identify which plan they started with originally (Basic, Pro, or Enterprise). This shows their initial commitment level, not what they might have upgraded/downgraded to later.",
        },
        {
            "query": "Show the industry distribution of customers who were active in January 2023",
            "expected_output": json.dumps(ground_truth["by_industry"], indent=2),
            "strategy": "Group the Jan 2023 active customers by their industry segment (Tech, Healthcare, Retail, Education, Other). This helps understand which industries our active customer base comes from.",
        },
        {
            "query": "What are the acquisition channels for customers active in January 2023?",
            "expected_output": json.dumps(ground_truth["by_acquisition_channel"], indent=2),
            "strategy": "Show how our Jan 2023 active customers originally found us - through Paid Search, Social Media, Email, Affiliate, or Content marketing. This tells us which channels brought our most engaged customers.",
        },
    ]

    # Save to CSV
    eval_df = pd.DataFrame(eval_cases)
    output_path = Path("evals/scenario1/step1_evaluation.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(output_path, index=False)

    print(f"Step 1 evaluation dataset saved to {output_path}")
    print("Function used: get_active_customers_jan_2023()")
    print(f"Generated {len(eval_cases)} evaluation cases")

    return eval_df


if __name__ == "__main__":
    generate_step1_evaluation()
