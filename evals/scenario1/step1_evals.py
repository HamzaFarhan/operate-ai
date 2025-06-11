"""
Step 1: Data Filtering & Customer Cohort Identification
Combines ground truth generation with evaluation dataset creation.
"""

import json
from datetime import datetime
from functools import partial
from pathlib import Path

import logfire
import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai.models import KnownModelName
from pydantic_ai.models.fallback import FallbackModel
from pydantic_evals import Case, Dataset

from operate_ai.cfo_graph import setup_workspace
from operate_ai.evals import EqEvaluator, Query, eval_task

logfire.configure()

MAIN_DIR = Path(__file__).parent.parent.parent
DATA_DIR = MAIN_DIR / "operateai_scenario1_data"


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
    "total_count": """Count the total number of unique customers who had at least one subscription that was active at any point during January 2023 (January 1-31, 2023). 

A subscription is considered active during January 2023 if:
- The subscription StartDate is on or before January 31, 2023 AND
- The subscription EndDate is on or after January 1, 2023 OR the subscription has no EndDate (null/NaT, meaning it's still ongoing)

Return only the count as an integer.""",
    "by_subscription_type": """For customers who were active in January 2023, provide a breakdown by their INITIAL subscription billing type (Monthly vs Annual).

Step-by-step logic:
1. Identify all customers who had at least one subscription active during January 2023 (using the same active definition as above)
2. For each of these customers, find their very FIRST subscription ever (earliest StartDate) 
3. Use that first subscription's SubscriptionType field to categorize them as either "Monthly" or "Annual"
4. Count how many customers fall into each category based on their initial billing preference

Return the counts as: {"monthly": X, "annual": Y} where X and Y are integers representing customer counts.""",
    "by_plan_type": """For customers who were active in January 2023, provide a breakdown by their INITIAL plan tier (Basic, Pro, Enterprise).

Step-by-step logic:
1. Identify all customers who had at least one subscription active during January 2023 
2. For each of these customers, find their very FIRST subscription ever (earliest StartDate)
3. Use that first subscription's PlanName field to categorize them as "Basic", "Pro", or "Enterprise"  
4. Count how many customers fall into each plan tier based on their original plan choice

Return the counts as: {"basic": X, "pro": Y, "enterprise": Z} where X, Y, Z are integers representing customer counts.""",
    "by_industry": """For customers who were active in January 2023, provide a breakdown by their industry segment.

Step-by-step logic:
1. Identify all customers who had at least one subscription active during January 2023
2. Use each customer's IndustrySegment field from the customers table to categorize them
3. Count how many customers belong to each industry: "Retail", "Tech", "Healthcare", "Education", "Other"

Return the counts as: {"retail": A, "tech": B, "healthcare": C, "education": D, "other": E} where A,B,C,D,E are integers representing customer counts.""",
    "by_acquisition_channel": """For customers who were active in January 2023, provide a breakdown by their original acquisition channel.

Step-by-step logic:
1. Identify all customers who had at least one subscription active during January 2023
2. Use each customer's AcquisitionChannel field from the customers table to categorize them  
3. Count how many customers were acquired through each channel: "Paid Search", "Social Media", "Email", "Affiliate", "Content"

Return the counts as: {"paid_search": A, "social_media": B, "email": C, "affiliate": D, "content": E} where A,B,C,D,E are integers representing customer counts.""",
}


def create_step1_dataset(start_index: int | None = None, end_index: int | None = None):
    """Create the evaluation dataset using ground truth data"""
    # Get ground truth
    ground_truth = get_active_customers_jan_2023()

    # Convert ground truth data to expected format
    subscription_type_data = ground_truth["by_subscription_type"]
    plan_type_data = ground_truth["by_plan_type"]
    industry_data = ground_truth["by_industry"]
    acquisition_data = ground_truth["by_acquisition_channel"]

    cases = [
        Case(
            name="step1_1",
            inputs=Query(
                query=QUERIES["total_count"],
                output_type=int,
            ),
            expected_output=int(ground_truth["total_active_customers"]),
        ),
        Case(
            name="step1_2",
            inputs=Query(
                query=QUERIES["by_subscription_type"],
                output_type=CustomersPerSubscriptionType,
            ),
            expected_output=CustomersPerSubscriptionType(
                monthly=int(subscription_type_data.get("Monthly", 0)),
                annual=int(subscription_type_data.get("Annual", 0)),
            ),
        ),
        Case(
            name="step1_3",
            inputs=Query(
                query=QUERIES["by_plan_type"],
                output_type=CustomersPerPlanType,
            ),
            expected_output=CustomersPerPlanType(
                basic=int(plan_type_data.get("Basic", 0)),
                pro=int(plan_type_data.get("Pro", 0)),
                enterprise=int(plan_type_data.get("Enterprise", 0)),
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
    ]
    start_index = max(0, start_index or 0)
    end_index = min(len(cases), max(end_index or len(cases), start_index + 1))
    dataset = Dataset[Query[ResultT], ResultT](
        cases=cases[start_index:end_index],
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


def evaluate(workspace_name: str = "1", start_index: int | None = None, end_index: int | None = None):
    thinking = False
    name = f"step1_evals_thinking_{thinking}"
    model: KnownModelName | FallbackModel = FallbackModel(
        "anthropic:claude-4-sonnet-20250514",
        "openai:gpt-4.1",
        "google-gla:gemini-2.5-flash-preview-05-20",
        "openai:gpt-4.1-mini",
    )
    dataset = create_step1_dataset(start_index=start_index, end_index=end_index)
    workspace_dir = MAIN_DIR / f"workspaces/{workspace_name}"
    setup_workspace(data_dir=DATA_DIR, workspace_dir=workspace_dir, delete_existing=True)

    report = dataset.evaluate_sync(
        task=partial(eval_task, model=model, use_thinking=thinking, workspace_dir=workspace_dir), name=name
    )
    report.print(include_output=True, include_expected_output=True, include_input=True, include_averages=True)


if __name__ == "__main__":
    # Generate CSV
    # generate_csv()

    # Run evaluation
    evaluate(workspace_name="1", start_index=4, end_index=5)
