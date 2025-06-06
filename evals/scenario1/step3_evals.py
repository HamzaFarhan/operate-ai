"""
Step 3: Churn Rate Calculation
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


class ChurnRateBySubscriptionType(BaseModel):
    Monthly: float = Field(description="Churn rate for customers with Monthly subscriptions")
    Annual: float = Field(description="Churn rate for customers with Annual subscriptions")
    Total: float = Field(description="Overall churn rate for all customers")


class ChurnRateByPlanType(BaseModel):
    Basic: float = Field(description="Churn rate for customers with Basic plans")
    Pro: float = Field(description="Churn rate for customers with Pro plans")
    Enterprise: float = Field(description="Churn rate for customers with Enterprise plans")


class ChurnRateByIndustry(BaseModel):
    Retail: float = Field(description="Churn rate for Retail industry customers")
    Tech: float = Field(description="Churn rate for Tech industry customers")
    Healthcare: float = Field(description="Churn rate for Healthcare industry customers")
    Education: float = Field(description="Churn rate for Education industry customers")
    Other: float = Field(description="Churn rate for Other industry customers")


class ChurnRateByChannel(BaseModel):
    Paid_Search: float = Field(description="Churn rate for customers acquired via Paid Search")
    Social_Media: float = Field(description="Churn rate for customers acquired via Social Media")
    Email: float = Field(description="Churn rate for customers acquired via Email")
    Affiliate: float = Field(description="Churn rate for customers acquired via Affiliate")
    Content: float = Field(description="Churn rate for customers acquired via Content")


type ResultT = int | ChurnRateBySubscriptionType | ChurnRateByPlanType | ChurnRateByIndustry | ChurnRateByChannel


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


# Define queries once for reuse
QUERIES = {
    "by_subscription_type": "Calculate the churn rate for customers who were active in January 2023, broken down by their initial subscription type (Monthly, Annual, and Total combined). Active customers are those who had subscriptions that started before or during Jan 2023 AND either ended after Jan 1 2023 or are still ongoing. Count customers who churned during 2023 (Status='Churned' with EndDate in 2023). If any segment shows 0% churn, assume a 5-year customer lifetime (20% annual churn rate).",
    "by_plan_type": "Show churn rate breakdown by initial plan type (Basic, Pro, Enterprise) for customers who were active in January 2023. Apply 5-year assumption (20% churn) if no actual churns occurred in any segment.",
    "by_industry": "Calculate churn rate by industry segment for customers who were active in January 2023. Use Status='Churned' and EndDate in 2023 to identify churned customers.",
    "by_channel": "What is the churn rate by acquisition channel for customers who were active in January 2023? Count churns during 2023 and apply 5-year assumption if needed.",
    "total_churned": "How many customers from the January 2023 active cohort actually churned during 2023? Count customers with Status='Churned' and EndDate between Jan 1-Dec 31, 2023.",
}


def create_step3_dataset():
    """Create the evaluation dataset using ground truth data"""
    # Get ground truth
    ground_truth = get_jan_2023_cohort_churn_rates()

    # Convert ground truth data to expected format
    subscription_data = ground_truth["churn_by_subscription_type"]
    plan_data = ground_truth["churn_by_plan_type"]
    industry_data = ground_truth["churn_by_industry"]
    channel_data = ground_truth["churn_by_channel"]

    dataset = Dataset[Query[ResultT], ResultT](
        cases=[
            Case(
                name="step3_1",
                inputs=Query(
                    query=QUERIES["by_subscription_type"],
                    output_type=ChurnRateBySubscriptionType,
                ),
                expected_output=ChurnRateBySubscriptionType(
                    Monthly=subscription_data.get("Monthly", 0.0),
                    Annual=subscription_data.get("Annual", 0.0),
                    Total=subscription_data.get("Total", 0.0),
                ),
            ),
            Case(
                name="step3_2",
                inputs=Query(
                    query=QUERIES["by_plan_type"],
                    output_type=ChurnRateByPlanType,
                ),
                expected_output=ChurnRateByPlanType(
                    Basic=plan_data.get("Basic", 0.0),
                    Pro=plan_data.get("Pro", 0.0),
                    Enterprise=plan_data.get("Enterprise", 0.0),
                ),
            ),
            Case(
                name="step3_3",
                inputs=Query(
                    query=QUERIES["by_industry"],
                    output_type=ChurnRateByIndustry,
                ),
                expected_output=ChurnRateByIndustry(
                    Retail=industry_data.get("Retail", 0.0),
                    Tech=industry_data.get("Tech", 0.0),
                    Healthcare=industry_data.get("Healthcare", 0.0),
                    Education=industry_data.get("Education", 0.0),
                    Other=industry_data.get("Other", 0.0),
                ),
            ),
            Case(
                name="step3_4",
                inputs=Query(
                    query=QUERIES["by_channel"],
                    output_type=ChurnRateByChannel,
                ),
                expected_output=ChurnRateByChannel(
                    Paid_Search=channel_data.get("Paid Search", 0.0),
                    Social_Media=channel_data.get("Social Media", 0.0),
                    Email=channel_data.get("Email", 0.0),
                    Affiliate=channel_data.get("Affiliate", 0.0),
                    Content=channel_data.get("Content", 0.0),
                ),
            ),
            Case(
                name="step3_5",
                inputs=Query(
                    query=QUERIES["total_churned"],
                    output_type=int,
                ),
                expected_output=ground_truth["total_churned_customers"],
            ),
        ],
        evaluators=[EqEvaluator[ResultT]()],
    )

    return dataset


def generate_csv():
    """Generate CSV file without strategy column for step3 evaluation"""
    # Get ground truth
    ground_truth = get_jan_2023_cohort_churn_rates()

    eval_cases = [
        {
            "query": QUERIES["by_subscription_type"],
            "expected_output": json.dumps(ground_truth["churn_by_subscription_type"], indent=2),
        },
        {
            "query": QUERIES["by_plan_type"],
            "expected_output": json.dumps(ground_truth["churn_by_plan_type"], indent=2),
        },
        {
            "query": QUERIES["by_industry"],
            "expected_output": json.dumps(ground_truth["churn_by_industry"], indent=2),
        },
        {
            "query": QUERIES["by_channel"],
            "expected_output": json.dumps(ground_truth["churn_by_channel"], indent=2),
        },
        {
            "query": QUERIES["total_churned"],
            "expected_output": str(ground_truth["total_churned_customers"]),
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


def evaluate():
    """Run the evaluation (can be commented out)"""
    dataset = create_step3_dataset()
    report = dataset.evaluate_sync(task=eval_task, name="step3_evals")
    report.print(include_output=True, include_expected_output=True, include_input=True, include_averages=True)


if __name__ == "__main__":
    # Generate CSV
    generate_csv()

    # Run evaluation (can be commented out)
    # evaluate()
