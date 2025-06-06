"""
Step 1: Data Filtering & Customer Cohort Identification
Combines ground truth generation with evaluation dataset creation.
"""

import json
from datetime import datetime
from pathlib import Path

import logfire
import pandas as pd
from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset

from operate_ai.evals import EqEvaluator, PrevQuery, Query, eval_task

logfire.configure()


class CustomersPerSubscriptionType(BaseModel):
    monthly: int = Field(description="The number of customers with a monthly subscription.")
    annual: int = Field(description="The number of customers with an annual subscription.")


class CustomersPerPlanType(BaseModel):
    basic: int = Field(description="The number of customers with a basic plan.")
    pro: int = Field(description="The number of customers with a pro plan.")
    enterprise: int = Field(description="The number of customers with an enterprise plan.")


class CustomersPerIndustry(BaseModel):
    retail: int = Field(description="The number of customers in the retail industry.")
    tech: int = Field(description="The number of customers in the tech industry.")
    healthcare: int = Field(description="The number of customers in the healthcare industry.")
    education: int = Field(description="The number of customers in the education industry.")
    other: int = Field(description="The number of customers in the other industry.")


class CustomersPerAcquisitionChannel(BaseModel):
    paid_search: int = Field(description="The number of customers acquired through paid search.")
    social_media: int = Field(description="The number of customers acquired through social media.")
    email: int = Field(description="The number of customers acquired through email.")
    affiliate: int = Field(description="The number of customers acquired through affiliate marketing.")
    content: int = Field(description="The number of customers acquired through content marketing.")


type ResultT = (
    int
    | CustomersPerSubscriptionType
    | CustomersPerPlanType
    | CustomersPerIndustry
    | CustomersPerAcquisitionChannel
)


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


# Define queries once for reuse
QUERIES = {
    "total_count": "How many customers were active in January 2023? Active customers are those who had subscriptions that started before or during Jan 2023 AND either ended after Jan 1 2023 or are still ongoing.",
    "by_subscription_type": "Show me the breakdown of customers who were active in January 2023 by their initial subscription type (Monthly vs Annual). Use each customer's very first subscription to determine their original billing preference.",
    "by_plan_type": "Break down customers who were active in January 2023 by their initial plan type (Basic, Pro, Enterprise). Show the distribution based on their original plan choice.",
    "by_industry": "Show the industry distribution of customers who were active in January 2023.",
    "by_acquisition_channel": "What are the acquisition channels for customers who were active in January 2023?",
}


def create_step1_dataset():
    """Create the evaluation dataset using ground truth data"""
    # Get ground truth
    ground_truth = get_active_customers_jan_2023()

    # Convert ground truth data to expected format
    subscription_type_data = ground_truth["by_subscription_type"]
    plan_type_data = ground_truth["by_plan_type"]
    industry_data = ground_truth["by_industry"]
    acquisition_data = ground_truth["by_acquisition_channel"]

    prev_query = PrevQuery(
        query="How many customers were active in January 2023?",
        result=ground_truth["total_active_customers"],
    )

    dataset = Dataset[Query[ResultT], ResultT](
        cases=[
            Case(
                name="step1_1",
                inputs=Query(
                    query=QUERIES["total_count"],
                    output_type=int,
                ),
                expected_output=ground_truth["total_active_customers"],
            ),
            Case(
                name="step1_2",
                inputs=Query(
                    query=QUERIES["by_subscription_type"],
                    output_type=CustomersPerSubscriptionType,
                ),
                expected_output=CustomersPerSubscriptionType(
                    monthly=subscription_type_data.get("Monthly", 0),
                    annual=subscription_type_data.get("Annual", 0),
                ),
            ),
            Case(
                name="step1_3",
                inputs=Query(
                    query=QUERIES["by_plan_type"],
                    output_type=CustomersPerPlanType,
                ),
                expected_output=CustomersPerPlanType(
                    basic=plan_type_data.get("Basic", 0),
                    pro=plan_type_data.get("Pro", 0),
                    enterprise=plan_type_data.get("Enterprise", 0),
                ),
            ),
            Case(
                name="step1_4",
                inputs=Query(
                    query=QUERIES["by_industry"],
                    output_type=CustomersPerIndustry,
                ),
                expected_output=CustomersPerIndustry(
                    retail=industry_data.get("Retail", 0),
                    tech=industry_data.get("Tech", 0),
                    healthcare=industry_data.get("Healthcare", 0),
                    education=industry_data.get("Education", 0),
                    other=industry_data.get("Other", 0),
                ),
            ),
            Case(
                name="step1_5",
                inputs=Query(
                    query=QUERIES["by_acquisition_channel"],
                    output_type=CustomersPerAcquisitionChannel,
                ),
                expected_output=CustomersPerAcquisitionChannel(
                    paid_search=acquisition_data.get("Paid Search", 0),
                    social_media=acquisition_data.get("Social Media", 0),
                    email=acquisition_data.get("Email", 0),
                    affiliate=acquisition_data.get("Affiliate", 0),
                    content=acquisition_data.get("Content", 0),
                ),
            ),
        ],
        evaluators=[EqEvaluator[ResultT]()],
    )

    return dataset


def generate_csv():
    """Generate CSV file with evaluation cases"""
    ground_truth = get_active_customers_jan_2023()

    eval_cases = [
        {
            "query": QUERIES["total_count"],
            "expected_output": str(ground_truth["total_active_customers"]),
        },
        {
            "query": QUERIES["by_subscription_type"],
            "expected_output": json.dumps(ground_truth["by_subscription_type"], indent=2),
        },
        {
            "query": QUERIES["by_plan_type"],
            "expected_output": json.dumps(ground_truth["by_plan_type"], indent=2),
        },
        {
            "query": QUERIES["by_industry"],
            "expected_output": json.dumps(ground_truth["by_industry"], indent=2),
        },
        {
            "query": QUERIES["by_acquisition_channel"],
            "expected_output": json.dumps(ground_truth["by_acquisition_channel"], indent=2),
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


def evaluate():
    dataset = create_step1_dataset()
    report = dataset.evaluate_sync(task=eval_task, name="step1_evals")
    report.print(include_output=True, include_expected_output=True, include_input=True, include_averages=True)


if __name__ == "__main__":
    # Generate CSV
    generate_csv()

    # Run evaluation
    # evaluate()
