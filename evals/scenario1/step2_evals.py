"""
Step 2: Average Revenue Per User (ARPU) Calculation
Combines ground truth generation with evaluation dataset creation.
"""

import json
from datetime import datetime
from pathlib import Path

import logfire
import pandas as pd
from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset

from operate_ai.evals import EqEvaluator, Query, eval_task

logfire.configure()


class ARPUBySubscriptionType(BaseModel):
    monthly: float = Field(description="ARPU for customers with monthly subscriptions")
    annual: float = Field(description="ARPU for customers with annual subscriptions")
    total: float = Field(description="Combined ARPU for all customers")


class ARPUByPlanType(BaseModel):
    basic: float = Field(description="ARPU for customers with basic plans")
    pro: float = Field(description="ARPU for customers with pro plans")
    enterprise: float = Field(description="ARPU for customers with enterprise plans")


class ARPUByIndustry(BaseModel):
    education: float = Field(description="ARPU for customers in education industry")
    tech: float = Field(description="ARPU for customers in tech industry")
    healthcare: float = Field(description="ARPU for customers in healthcare industry")
    retail: float = Field(description="ARPU for customers in retail industry")
    other: float = Field(description="ARPU for customers in other industries")


class ARPUByChannel(BaseModel):
    affiliate: float = Field(description="ARPU for customers acquired through affiliate")
    email: float = Field(description="ARPU for customers acquired through email")
    social_media: float = Field(description="ARPU for customers acquired through social media")
    content: float = Field(description="ARPU for customers acquired through content")
    paid_search: float = Field(description="ARPU for customers acquired through paid search")


type ResultT = float | ARPUBySubscriptionType | ARPUByPlanType | ARPUByIndustry | ARPUByChannel


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


# Define queries once for reuse
QUERIES = {
    "by_subscription_type": "Calculate the Average Revenue Per User (ARPU) for customers who were active in January 2023, broken down by their initial subscription type (Monthly, Annual, and Total combined). Active customers are those who had subscriptions that started before or during Jan 2023 AND either ended after Jan 1 2023 or are still ongoing. Use their 2023 profit data (Jan-Dec 2023) to calculate ARPU.",
    "by_plan_type": "Show ARPU breakdown by initial plan type (Basic, Pro, Enterprise) for customers who were active in January 2023. Use their 2023 profit to calculate average revenue per user by original plan choice.",
    "by_industry": "Calculate ARPU by industry segment for customers who were active in January 2023. Use 2023 profit data to determine average revenue per user by industry.",
    "by_channel": "What is the ARPU by acquisition channel for customers who were active in January 2023? Calculate using their 2023 profit data.",
    "total_profit": "What was the total profit generated in 2023 by customers who were active in January 2023?",
}


def create_step2_dataset():
    """Create the evaluation dataset using ground truth data"""
    # Get ground truth
    ground_truth = get_jan_2023_cohort_arpu()

    # Convert ground truth data to expected format
    subscription_type_data = ground_truth["arpu_by_subscription_type"]
    plan_type_data = ground_truth["arpu_by_plan_type"]
    industry_data = ground_truth["arpu_by_industry"]
    channel_data = ground_truth["arpu_by_channel"]

    dataset = Dataset[Query[ResultT], ResultT](
        cases=[
            Case(
                name="step2_1",
                inputs=Query(
                    query=QUERIES["by_subscription_type"],
                    output_type=ARPUBySubscriptionType,
                ),
                expected_output=ARPUBySubscriptionType(
                    monthly=subscription_type_data.get("Monthly", 0.0),
                    annual=subscription_type_data.get("Annual", 0.0),
                    total=subscription_type_data.get("Total", 0.0),
                ),
            ),
            Case(
                name="step2_2",
                inputs=Query(
                    query=QUERIES["by_plan_type"],
                    output_type=ARPUByPlanType,
                ),
                expected_output=ARPUByPlanType(
                    basic=plan_type_data.get("Basic", 0.0),
                    pro=plan_type_data.get("Pro", 0.0),
                    enterprise=plan_type_data.get("Enterprise", 0.0),
                ),
            ),
            Case(
                name="step2_3",
                inputs=Query(
                    query=QUERIES["by_industry"],
                    output_type=ARPUByIndustry,
                ),
                expected_output=ARPUByIndustry(
                    education=industry_data.get("Education", 0.0),
                    tech=industry_data.get("Tech", 0.0),
                    healthcare=industry_data.get("Healthcare", 0.0),
                    retail=industry_data.get("Retail", 0.0),
                    other=industry_data.get("Other", 0.0),
                ),
            ),
            Case(
                name="step2_4",
                inputs=Query(
                    query=QUERIES["by_channel"],
                    output_type=ARPUByChannel,
                ),
                expected_output=ARPUByChannel(
                    affiliate=channel_data.get("Affiliate", 0.0),
                    email=channel_data.get("Email", 0.0),
                    social_media=channel_data.get("Social Media", 0.0),
                    content=channel_data.get("Content", 0.0),
                    paid_search=channel_data.get("Paid Search", 0.0),
                ),
            ),
            Case(
                name="step2_5",
                inputs=Query(
                    query=QUERIES["total_profit"],
                    output_type=float,
                ),
                expected_output=ground_truth["total_profit_2023"],
            ),
        ],
        evaluators=[EqEvaluator[ResultT]()],
    )

    return dataset


def generate_csv():
    """Generate CSV file with evaluation cases"""
    ground_truth = get_jan_2023_cohort_arpu()

    eval_cases = [
        {
            "query": QUERIES["by_subscription_type"],
            "expected_output": json.dumps(ground_truth["arpu_by_subscription_type"], indent=2),
        },
        {
            "query": QUERIES["by_plan_type"],
            "expected_output": json.dumps(ground_truth["arpu_by_plan_type"], indent=2),
        },
        {
            "query": QUERIES["by_industry"],
            "expected_output": json.dumps(ground_truth["arpu_by_industry"], indent=2),
        },
        {
            "query": QUERIES["by_channel"],
            "expected_output": json.dumps(ground_truth["arpu_by_channel"], indent=2),
        },
        {
            "query": QUERIES["total_profit"],
            "expected_output": str(ground_truth["total_profit_2023"]),
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


def evaluate():
    dataset = create_step2_dataset()
    report = dataset.evaluate_sync(task=eval_task, name="step2_evals")
    report.print(include_output=True, include_expected_output=True, include_input=True, include_averages=True)


if __name__ == "__main__":
    # Generate CSV
    generate_csv()

    # Run evaluation (can be commented out)
    # evaluate()
