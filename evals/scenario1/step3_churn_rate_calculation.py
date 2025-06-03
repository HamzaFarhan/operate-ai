"""
Step 3: Churn Rate Calculation
Generates ground truth for churn rate calculations by subscription type, plan type, industry, and channel.
Handles the 0% churn assumption (5-year customer lifetime).
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def get_jan_2023_cohort_churn_rates():
    """
    Calculates churn rates for customers who were active in Jan 2023,
    based on who churned during 2023.

    Returns:
        dict: Churn rate calculations by various segments
    """
    # Load the generated data
    data_dir = Path("operateai_scenario1_data")

    customers_df = pd.read_csv(data_dir / "customers.csv")
    subscriptions_df = pd.read_csv(data_dir / "subscriptions.csv")

    # Convert dates to datetime
    subscriptions_df["StartDate"] = pd.to_datetime(subscriptions_df["StartDate"])
    subscriptions_df["EndDate"] = pd.to_datetime(subscriptions_df["EndDate"], errors="coerce")

    # Step 1: Get Jan 2023 active customers (same as Step 1)
    jan_2023_start = datetime(2023, 1, 1)
    jan_2023_end = datetime(2023, 1, 31)

    active_subs_jan_2023 = subscriptions_df[
        (subscriptions_df["StartDate"] <= jan_2023_end)
        & ((subscriptions_df["EndDate"] >= jan_2023_start) | pd.isna(subscriptions_df["EndDate"]))
    ]

    active_customer_ids = active_subs_jan_2023["CustomerID"].unique()

    # Get initial subscription details for each customer
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
                }
            )

    initial_subs_df = pd.DataFrame(initial_subs)
    cohort_df = initial_subs_df.merge(customers_df, on="CustomerID", how="left")

    # Step 2: Identify customers who churned during 2023
    year_2023_start = datetime(2023, 1, 1)
    year_2023_end = datetime(2023, 12, 31)

    # Find customers who churned during 2023 (Status = 'Churned' and EndDate in 2023)
    churned_customers = set()
    for customer_id in active_customer_ids:
        customer_subs = subscriptions_df[subscriptions_df["CustomerID"] == customer_id]
        churned_subs = customer_subs[
            (customer_subs["Status"] == "Churned")
            & (customer_subs["EndDate"] >= year_2023_start)
            & (customer_subs["EndDate"] <= year_2023_end)
        ]
        if len(churned_subs) > 0:
            churned_customers.add(customer_id)

    # Step 3: Calculate churn rates by different segments
    FIVE_YEAR_ANNUAL_CHURN_RATE = 0.2  # 1/5 = 20% annual churn rate for 5-year assumption

    def calculate_churn_rate(customers_in_segment):
        total_customers = len(customers_in_segment)
        churned_count = sum(1 for cid in customers_in_segment if cid in churned_customers)

        if total_customers == 0:
            return 0.0

        raw_churn_rate = churned_count / total_customers

        # If 0% churn, apply 5-year assumption
        if raw_churn_rate == 0:
            return FIVE_YEAR_ANNUAL_CHURN_RATE

        return round(raw_churn_rate, 4)

    # By Subscription Type (Monthly/Annual)
    churn_by_sub_type = {}
    for sub_type in cohort_df["InitialSubscriptionType"].unique():
        customers_in_segment = cohort_df[cohort_df["InitialSubscriptionType"] == sub_type]["CustomerID"].tolist()
        churn_by_sub_type[sub_type] = calculate_churn_rate(customers_in_segment)

    # Total churn rate (all customers combined)
    churn_by_sub_type["Total"] = calculate_churn_rate(active_customer_ids.tolist())

    # By Plan Type (Basic/Pro/Enterprise)
    churn_by_plan_type = {}
    for plan_type in cohort_df["InitialPlanName"].unique():
        customers_in_segment = cohort_df[cohort_df["InitialPlanName"] == plan_type]["CustomerID"].tolist()
        churn_by_plan_type[plan_type] = calculate_churn_rate(customers_in_segment)

    # By Industry
    churn_by_industry = {}
    for industry in cohort_df["IndustrySegment"].unique():
        customers_in_segment = cohort_df[cohort_df["IndustrySegment"] == industry]["CustomerID"].tolist()
        churn_by_industry[industry] = calculate_churn_rate(customers_in_segment)

    # By Acquisition Channel
    churn_by_channel = {}
    for channel in cohort_df["AcquisitionChannel"].unique():
        customers_in_segment = cohort_df[cohort_df["AcquisitionChannel"] == channel]["CustomerID"].tolist()
        churn_by_channel[channel] = calculate_churn_rate(customers_in_segment)

    return {
        "churn_by_subscription_type": churn_by_sub_type,
        "churn_by_plan_type": churn_by_plan_type,
        "churn_by_industry": churn_by_industry,
        "churn_by_channel": churn_by_channel,
        "total_customers_in_cohort": len(active_customer_ids),
        "total_churned_customers": len(churned_customers),
        "five_year_assumption_rate": FIVE_YEAR_ANNUAL_CHURN_RATE,
        "churned_customer_ids": list(churned_customers),
    }


def generate_step3_evaluation():
    """Generate evaluation dataset for Step 3"""

    # Get ground truth
    ground_truth = get_jan_2023_cohort_churn_rates()

    # Create evaluation cases
    eval_cases = [
        {
            "query": "Calculate the churn rate for the Jan 2023 customer cohort by subscription type (Monthly, Annual, and Total combined)",
            "expected_output": json.dumps(ground_truth["churn_by_subscription_type"], indent=2),
            "strategy": "Take customers active in Jan 2023, count how many churned during 2023, divide by total customers in each subscription type segment. Churn Rate = Churned Customers / Total Customers. If 0% churn, assume 5-year customer lifetime (20% annual churn rate).",
        },
        {
            "query": "Show churn rate breakdown by initial plan type (Basic, Pro, Enterprise) for the Jan 2023 cohort",
            "expected_output": json.dumps(ground_truth["churn_by_plan_type"], indent=2),
            "strategy": "Group Jan 2023 customers by their original plan choice, count churns in each group during 2023, calculate churn rate per plan. Higher-value customers (Enterprise) typically have lower churn rates.",
        },
        {
            "query": "Calculate churn rate by industry segment for customers who were active in January 2023",
            "expected_output": json.dumps(ground_truth["churn_by_industry"], indent=2),
            "strategy": "Segment the Jan 2023 cohort by industry, track which customers churned during 2023 by industry, calculate churn rates. Shows which industries have more stable vs volatile customer relationships.",
        },
        {
            "query": "What is the churn rate by acquisition channel for the January 2023 customer cohort?",
            "expected_output": json.dumps(ground_truth["churn_by_channel"], indent=2),
            "strategy": "Group customers by how they originally found us, calculate what percentage churned during 2023 by channel. Reveals which marketing channels bring customers who stick around vs those who leave quickly.",
        },
        {
            "query": "How many customers from the Jan 2023 cohort actually churned during 2023?",
            "expected_output": str(ground_truth["total_churned_customers"]),
            "strategy": 'Count customers from the Jan 2023 active cohort who had a subscription with Status="Churned" and EndDate between Jan 1, 2023 and Dec 31, 2023. This is the raw number of customers lost from our starting cohort.',
        },
    ]

    # Save to CSV
    eval_df = pd.DataFrame(eval_cases)
    output_path = Path("evals/scenario1/step3_evaluation.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(output_path, index=False)

    print(f"Step 3 evaluation dataset saved to {output_path}")
    print("Function used: get_jan_2023_cohort_churn_rates()")
    print(f"Generated {len(eval_cases)} evaluation cases")
    print(f"Total customers in cohort: {ground_truth['total_customers_in_cohort']}")
    print(f"Total churned customers: {ground_truth['total_churned_customers']}")
    print(f"Five-year assumption rate: {ground_truth['five_year_assumption_rate']}")

    return eval_df


if __name__ == "__main__":
    generate_step3_evaluation()
