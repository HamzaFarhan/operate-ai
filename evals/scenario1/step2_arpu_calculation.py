"""
Step 2: Average Revenue Per User (ARPU) Calculation
Generates ground truth for ARPU calculations by subscription type, plan type, industry, and channel.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def get_jan_2023_cohort_arpu():
    """
    Calculates ARPU for customers who were active in Jan 2023, based on their
    profit generated during Jan-Dec 2023.

    Returns:
        dict: ARPU calculations by various segments
    """
    # Load the generated data
    data_dir = Path("operateai_scenario1_data")

    customers_df = pd.read_csv(data_dir / "customers.csv")
    subscriptions_df = pd.read_csv(data_dir / "subscriptions.csv")
    orders_df = pd.read_csv(data_dir / "orders.csv")

    # Convert dates to datetime
    subscriptions_df["StartDate"] = pd.to_datetime(subscriptions_df["StartDate"])
    subscriptions_df["EndDate"] = pd.to_datetime(subscriptions_df["EndDate"], errors="coerce")
    orders_df["OrderDate"] = pd.to_datetime(orders_df["OrderDate"])

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

    # Step 2: Get orders from Jan-Dec 2023 for these customers
    year_2023_start = datetime(2023, 1, 1)
    year_2023_end = datetime(2023, 12, 31)

    cohort_orders_2023 = orders_df[
        (orders_df["CustomerID"].isin(active_customer_ids))
        & (orders_df["OrderDate"] >= year_2023_start)
        & (orders_df["OrderDate"] <= year_2023_end)
    ]

    # Step 3: Calculate ARPU by different segments

    # By Subscription Type (Monthly/Annual)
    arpu_by_sub_type = {}
    for sub_type in cohort_df["InitialSubscriptionType"].unique():
        customers_in_segment = cohort_df[cohort_df["InitialSubscriptionType"] == sub_type]["CustomerID"]
        segment_orders = cohort_orders_2023[cohort_orders_2023["CustomerID"].isin(customers_in_segment)]
        total_profit = segment_orders["Profit"].sum()
        customer_count = len(customers_in_segment)
        arpu_by_sub_type[sub_type] = round(total_profit / customer_count, 2) if customer_count > 0 else 0

    # Total ARPU (Monthly + Annual combined)
    total_profit = cohort_orders_2023["Profit"].sum()
    total_customers = len(active_customer_ids)
    arpu_by_sub_type["Total"] = round(total_profit / total_customers, 2) if total_customers > 0 else 0

    # By Plan Type (Basic/Pro/Enterprise)
    arpu_by_plan_type = {}
    for plan_type in cohort_df["InitialPlanName"].unique():
        customers_in_segment = cohort_df[cohort_df["InitialPlanName"] == plan_type]["CustomerID"]
        segment_orders = cohort_orders_2023[cohort_orders_2023["CustomerID"].isin(customers_in_segment)]
        total_profit = segment_orders["Profit"].sum()
        customer_count = len(customers_in_segment)
        arpu_by_plan_type[plan_type] = round(total_profit / customer_count, 2) if customer_count > 0 else 0

    # By Industry
    arpu_by_industry = {}
    for industry in cohort_df["IndustrySegment"].unique():
        customers_in_segment = cohort_df[cohort_df["IndustrySegment"] == industry]["CustomerID"]
        segment_orders = cohort_orders_2023[cohort_orders_2023["CustomerID"].isin(customers_in_segment)]
        total_profit = segment_orders["Profit"].sum()
        customer_count = len(customers_in_segment)
        arpu_by_industry[industry] = round(total_profit / customer_count, 2) if customer_count > 0 else 0

    # By Acquisition Channel
    arpu_by_channel = {}
    for channel in cohort_df["AcquisitionChannel"].unique():
        customers_in_segment = cohort_df[cohort_df["AcquisitionChannel"] == channel]["CustomerID"]
        segment_orders = cohort_orders_2023[cohort_orders_2023["CustomerID"].isin(customers_in_segment)]
        total_profit = segment_orders["Profit"].sum()
        customer_count = len(customers_in_segment)
        arpu_by_channel[channel] = round(total_profit / customer_count, 2) if customer_count > 0 else 0

    return {
        "arpu_by_subscription_type": arpu_by_sub_type,
        "arpu_by_plan_type": arpu_by_plan_type,
        "arpu_by_industry": arpu_by_industry,
        "arpu_by_channel": arpu_by_channel,
        "total_profit_2023": float(total_profit),
        "total_customers": int(total_customers),
        "cohort_order_count": len(cohort_orders_2023),
    }


def generate_step2_evaluation():
    """Generate evaluation dataset for Step 2"""

    # Get ground truth
    ground_truth = get_jan_2023_cohort_arpu()

    # Create evaluation cases
    eval_cases = [
        {
            "query": "Calculate the Average Revenue Per User (ARPU) for the Jan 2023 customer cohort by subscription type (Monthly, Annual, and Total combined)",
            "expected_output": json.dumps(ground_truth["arpu_by_subscription_type"], indent=2),
            "strategy": "Take customers active in Jan 2023, get all their orders from Jan-Dec 2023, sum up profit by their initial subscription type (Monthly/Annual), then divide total profit by number of customers in each segment. ARPU = Total Profit / Number of Customers.",
        },
        {
            "query": "Show ARPU breakdown by initial plan type (Basic, Pro, Enterprise) for the Jan 2023 cohort",
            "expected_output": json.dumps(ground_truth["arpu_by_plan_type"], indent=2),
            "strategy": "Group Jan 2023 customers by their original plan choice (Basic, Pro, Enterprise), calculate total profit each group generated in 2023, then divide by customer count in each group. Higher plans should show higher ARPU.",
        },
        {
            "query": "Calculate ARPU by industry segment for customers who were active in January 2023",
            "expected_output": json.dumps(ground_truth["arpu_by_industry"], indent=2),
            "strategy": "Segment the Jan 2023 cohort by industry (Tech, Healthcare, Retail, Education, Other), sum their 2023 profits by industry, divide by customer count per industry. Shows which industries generate most revenue per customer.",
        },
        {
            "query": "What is the ARPU by acquisition channel for the January 2023 customer cohort?",
            "expected_output": json.dumps(ground_truth["arpu_by_channel"], indent=2),
            "strategy": "Group customers by how they originally found us (Paid Search, Social Media, Email, etc.), calculate average profit per customer by channel during 2023. Reveals which marketing channels bring the most valuable customers.",
        },
        {
            "query": "What was the total profit generated in 2023 by customers who were active in January 2023?",
            "expected_output": str(ground_truth["total_profit_2023"]),
            "strategy": "Sum up all profit from orders placed between Jan 1, 2023 and Dec 31, 2023 by customers who were active subscribers in January 2023. This is the total revenue contribution from this specific customer cohort.",
        },
    ]

    # Save to CSV
    eval_df = pd.DataFrame(eval_cases)
    output_path = Path("evals/scenario1/step2_evaluation.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(output_path, index=False)

    print(f"Step 2 evaluation dataset saved to {output_path}")
    print("Function used: get_jan_2023_cohort_arpu()")
    print(f"Generated {len(eval_cases)} evaluation cases")
    print(f"Total customers in cohort: {ground_truth['total_customers']}")
    print(f"Total profit generated in 2023: ${ground_truth['total_profit_2023']:,.2f}")

    return eval_df


if __name__ == "__main__":
    generate_step2_evaluation()
