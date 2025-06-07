"""
Arman evaluation queries: Complex financial and churn metrics
Combines ground truth generation with evaluation dataset creation.
"""

import shutil
from functools import partial
from pathlib import Path

import logfire
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai.models import KnownModelName
from pydantic_ai.models.fallback import FallbackModel
from pydantic_evals import Case, Dataset

from operate_ai.evals import DELTA, EqEvaluator, Query, eval_task

logfire.configure()

MAIN_DIR = Path(__file__).parent.parent.parent


class CACComparison(BaseModel):
    highest_cac_channel: str = Field(description="Channel with highest CAC.")
    highest_cac_value: float = Field(description="Highest CAC value.")
    lowest_cac_channel: str = Field(description="Channel with lowest CAC.")
    lowest_cac_value: float = Field(description="Lowest CAC value.")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CACComparison):
            return NotImplemented
        return (
            self.highest_cac_channel == other.highest_cac_channel
            and abs(round(self.highest_cac_value, 2) - round(other.highest_cac_value, 2)) < DELTA
            and self.lowest_cac_channel == other.lowest_cac_channel
            and abs(round(self.lowest_cac_value, 2) - round(other.lowest_cac_value, 2)) < DELTA
        )


type ResultT = float | int | CACComparison


def load_data():
    """Load all necessary datasets"""
    data_dir = Path("operateai_scenario1_data")

    customers_df = pd.read_csv(data_dir / "customers.csv")
    subscriptions_df = pd.read_csv(data_dir / "subscriptions.csv")
    orders_df = pd.read_csv(data_dir / "orders.csv")
    marketing_spend_df = pd.read_csv(data_dir / "marketing_spend.csv")

    # Convert date columns
    customers_df["AcquisitionDate"] = pd.to_datetime(customers_df["AcquisitionDate"])
    subscriptions_df["StartDate"] = pd.to_datetime(subscriptions_df["StartDate"])
    subscriptions_df["EndDate"] = pd.to_datetime(subscriptions_df["EndDate"], errors="coerce")
    orders_df["OrderDate"] = pd.to_datetime(orders_df["OrderDate"])
    marketing_spend_df["Date"] = pd.to_datetime(marketing_spend_df["Date"])

    return customers_df, subscriptions_df, orders_df, marketing_spend_df


def calculate_arman_ground_truth():
    """Calculate all arman evaluation ground truth values based on current data"""
    customers_df, subscriptions_df, orders_df, marketing_spend_df = load_data()

    # Content CAC from July 2024 to Dec 2024
    content_spend = marketing_spend_df[
        (marketing_spend_df["Channel"] == "Content")
        & (marketing_spend_df["Date"] >= "2024-07-01")
        & (marketing_spend_df["Date"] <= "2024-12-31")
    ]
    total_spend = content_spend["Spend"].sum()

    content_acquisitions = customers_df[
        (customers_df["AcquisitionChannel"] == "Content")
        & (customers_df["AcquisitionDate"] >= "2024-07-01")
        & (customers_df["AcquisitionDate"] <= "2024-12-31")
    ]
    content_cac = round(total_spend / len(content_acquisitions), 0) if len(content_acquisitions) > 0 else 0.0

    # CAC comparison Feb-May 2024
    spend_feb_may = marketing_spend_df[
        (marketing_spend_df["Date"] >= "2024-02-01") & (marketing_spend_df["Date"] <= "2024-05-31")
    ]
    channel_spend = spend_feb_may.groupby("Channel")["Spend"].sum()

    acquisitions_feb_may = customers_df[
        (customers_df["AcquisitionDate"] >= "2024-02-01") & (customers_df["AcquisitionDate"] <= "2024-05-31")
    ]
    channel_acquisitions = acquisitions_feb_may["AcquisitionChannel"].value_counts()

    cac_by_channel = {}
    for channel in channel_spend.index:
        if channel in channel_acquisitions.index and channel_acquisitions[channel] > 0:
            cac = channel_spend[channel] / channel_acquisitions[channel]
            cac_by_channel[channel] = round(cac, 0)

    highest_channel = max(cac_by_channel.keys(), key=lambda x: cac_by_channel[x])
    lowest_channel = min(cac_by_channel.keys(), key=lambda x: cac_by_channel[x])

    cac_comparison = CACComparison(
        highest_cac_channel=highest_channel,
        highest_cac_value=cac_by_channel[highest_channel],
        lowest_cac_channel=lowest_channel,
        lowest_cac_value=cac_by_channel[lowest_channel],
    )

    # ARPU for Tech customers in Oct 2024
    tech_customers = customers_df[customers_df["IndustrySegment"] == "Tech"]["CustomerID"].unique()
    oct_2024_orders = orders_df[
        (orders_df["CustomerID"].isin(tech_customers))
        & (orders_df["OrderDate"].dt.year == 2024)
        & (orders_df["OrderDate"].dt.month == 10)
    ]
    total_revenue = oct_2024_orders["Amount"].sum()
    unique_customers = oct_2024_orders["CustomerID"].nunique()
    arpu_tech = round(total_revenue / unique_customers, 6) if unique_customers > 0 else 0.0

    # MRR as of Dec 2024
    dec_2024 = pd.Timestamp("2024-12-31")
    active_subs = subscriptions_df[
        (subscriptions_df["StartDate"] <= dec_2024)
        & ((subscriptions_df["EndDate"].isna()) | (subscriptions_df["EndDate"] > dec_2024))
    ]
    active_customers = active_subs["CustomerID"].unique()
    customer_count = len(active_customers)

    dec_orders = orders_df[
        (orders_df["OrderDate"].dt.year == 2024)
        & (orders_df["OrderDate"].dt.month == 12)
        & (orders_df["CustomerID"].isin(active_customers))
    ]
    total_revenue_dec = dec_orders["Amount"].sum()
    arpu = total_revenue_dec / customer_count if customer_count > 0 else 0
    mrr = round(customer_count * arpu, 0)

    # Monthly basic customers Dec 2023 to Feb 2024
    monthly_basic_orders = orders_df[
        (orders_df["PlanName"] == "Basic")
        & (orders_df["SubscriptionType"] == "Monthly")
        & (orders_df["OrderDate"] >= "2023-12-01")
        & (orders_df["OrderDate"] <= "2024-02-29")
    ]
    monthly_basic_customers = monthly_basic_orders["CustomerID"].nunique()

    # Contribution margin 2024
    orders_2024 = orders_df[orders_df["OrderDate"].dt.year == 2024]
    total_profit = orders_2024["Profit"].sum()
    marketing_2024 = marketing_spend_df[marketing_spend_df["Date"].dt.year == 2024]
    total_marketing = marketing_2024["Spend"].sum()
    contribution_margin = round(total_profit - total_marketing, 2)

    # Monthly pro churn 2023 (simplified calculation)
    jan_2023_start = pd.Timestamp("2023-01-01")
    initial_monthly_pro = subscriptions_df[
        (subscriptions_df["PlanName"] == "Pro")
        & (subscriptions_df["SubscriptionType"] == "Monthly")
        & (subscriptions_df["StartDate"] <= jan_2023_start)
        & ((subscriptions_df["EndDate"] >= jan_2023_start) | subscriptions_df["EndDate"].isna())
    ]
    initial_count = len(initial_monthly_pro["CustomerID"].unique())

    churned_customers = set()
    for customer_id in initial_monthly_pro["CustomerID"].unique():
        customer_subs = subscriptions_df[subscriptions_df["CustomerID"] == customer_id]
        latest_sub = customer_subs.sort_values("StartDate").iloc[-1]
        if pd.notna(latest_sub["EndDate"]) and latest_sub["EndDate"] < pd.Timestamp("2024-01-01"):
            churned_customers.add(customer_id)

    monthly_pro_churn = round(len(churned_customers) / initial_count, 2) if initial_count > 0 else 0.0

    # Revenue retention calculations
    def calculate_revenue_retention(start_month: str, months_forward: int) -> float:
        """Calculate revenue retention for annual subscribers starting in a specific month"""
        # Find annual subscriptions that started in the target month
        target_month_start = pd.Timestamp(start_month)
        target_month_end = target_month_start + pd.offsets.MonthEnd(0)

        annual_subs_start_month = subscriptions_df[
            (subscriptions_df["SubscriptionType"] == "Annual")
            & (subscriptions_df["StartDate"] >= target_month_start)
            & (subscriptions_df["StartDate"] <= target_month_end)
        ]

        if len(annual_subs_start_month) == 0:
            return 0.0

        # Get customer IDs who started annual subscriptions in that month
        customer_ids = annual_subs_start_month["CustomerID"].unique()

        # Calculate initial revenue (first month after start)
        initial_revenue_start = target_month_start
        initial_revenue_end = target_month_start + pd.offsets.MonthEnd(0)

        initial_orders = orders_df[
            (orders_df["CustomerID"].isin(customer_ids))
            & (orders_df["SubscriptionType"] == "Annual")
            & (orders_df["OrderDate"] >= initial_revenue_start)
            & (orders_df["OrderDate"] <= initial_revenue_end)
        ]
        initial_revenue = initial_orders["Amount"].sum()

        # Calculate revenue after specified months
        future_revenue_start = target_month_start + pd.DateOffset(months=months_forward)
        future_revenue_end = future_revenue_start + pd.offsets.MonthEnd(0)

        future_orders = orders_df[
            (orders_df["CustomerID"].isin(customer_ids))
            & (orders_df["SubscriptionType"] == "Annual")
            & (orders_df["OrderDate"] >= future_revenue_start)
            & (orders_df["OrderDate"] <= future_revenue_end)
        ]
        future_revenue = future_orders["Amount"].sum()

        # Calculate retention ratio
        if initial_revenue > 0:
            retention = future_revenue / initial_revenue
            return round(retention, 2)
        else:
            return 0.0

    revenue_retention_aug_2023_12m = calculate_revenue_retention("2023-08-01", 12)  # Aug 2023 -> Aug 2024
    revenue_retention_sep_2023_6m = calculate_revenue_retention("2023-09-01", 6)  # Sep 2023 -> Mar 2024
    revenue_retention_feb_2023_12m = calculate_revenue_retention("2023-02-01", 12)  # Feb 2023 -> Feb 2024

    return {
        "revenue_retention_aug_2023_12m": revenue_retention_aug_2023_12m,
        "revenue_retention_sep_2023_6m": revenue_retention_sep_2023_6m,
        "revenue_retention_feb_2023_12m": revenue_retention_feb_2023_12m,
        "content_cac_jul_dec_2024": content_cac,
        "monthly_pro_churn_2023": monthly_pro_churn,
        "monthly_basic_customers_dec_feb": monthly_basic_customers,
        "contribution_margin_2024": contribution_margin,
        "cac_comparison_feb_may_2024": cac_comparison,
        "arpu_tech_oct_2024": arpu_tech,
        "mrr_dec_2024": mrr,
    }


# Define queries once for reuse
QUERIES = {
    "revenue_retention_aug_2023_12m": "What is the August 2023 annual subscription type by initial subscription 12 month revenue retention - 12 month revenue divided by initial revenue",
    "revenue_retention_sep_2023_6m": "What is the September 2023 annual subscription type by initial subscription 6 month revenue retention - 6 month revenue divided by initial revenue",
    "revenue_retention_feb_2023_12m": "What is the February 2023 annual subscription type 12 month revenue retention - 12 month revenue divided by initial revenue",
    "content_cac_jul_dec_2024": "What is Content CAC from July 2024 to Dec 2024 - total spend divided by acquisitions",
    "monthly_pro_churn_2023": "What is monthly plan & pro subscription type churn rate from Jan 2023 to Dec 2023 - churned subscribers divided by active subscribers on Jan 2023",
    "monthly_basic_customers_dec_feb": "How many unique customers ordered a monthly basic plan from Dec 2023 to Feb 2024 - Count unique customers that ordered during period",
    "contribution_margin_2024": "What is 2024 Contribution Margin - sum of 2024 profit minus sum of 2024 marketing expenses",
    "cac_comparison_feb_may_2024": "Which marketing acquisition channel has the highest CAC and which has the lowest during the period Feb 2024 to May 2024 - Compare spend divided by acquisitions across channels",
    "arpu_tech_oct_2024": "What is ARPU for Tech customers in Oct 2024 - total revenue divided by number of customers",
    "mrr_dec_2024": "What is MRR as of Dec 2024 - customer count multiplied by ARPU",
}


def create_arman_dataset():
    """Create the evaluation dataset using ground truth data"""
    # Get ground truth
    ground_truth = calculate_arman_ground_truth()

    dataset = Dataset[Query[ResultT], ResultT](
        cases=[
            Case(
                name="arman_1",
                inputs=Query(
                    query=QUERIES["revenue_retention_aug_2023_12m"],
                    output_type=float,
                ),
                expected_output=ground_truth["revenue_retention_aug_2023_12m"],
            ),
            Case(
                name="arman_2",
                inputs=Query(
                    query=QUERIES["revenue_retention_sep_2023_6m"],
                    output_type=float,
                ),
                expected_output=ground_truth["revenue_retention_sep_2023_6m"],
            ),
            Case(
                name="arman_3",
                inputs=Query(
                    query=QUERIES["revenue_retention_feb_2023_12m"],
                    output_type=float,
                ),
                expected_output=ground_truth["revenue_retention_feb_2023_12m"],
            ),
            Case(
                name="arman_4",
                inputs=Query(
                    query=QUERIES["content_cac_jul_dec_2024"],
                    output_type=float,
                ),
                expected_output=ground_truth["content_cac_jul_dec_2024"],
            ),
            Case(
                name="arman_5",
                inputs=Query(
                    query=QUERIES["monthly_pro_churn_2023"],
                    output_type=float,
                ),
                expected_output=ground_truth["monthly_pro_churn_2023"],
            ),
            Case(
                name="arman_6",
                inputs=Query(
                    query=QUERIES["monthly_basic_customers_dec_feb"],
                    output_type=int,
                ),
                expected_output=ground_truth["monthly_basic_customers_dec_feb"],
            ),
            Case(
                name="arman_7",
                inputs=Query(
                    query=QUERIES["contribution_margin_2024"],
                    output_type=float,
                ),
                expected_output=ground_truth["contribution_margin_2024"],
            ),
            Case(
                name="arman_8",
                inputs=Query(
                    query=QUERIES["cac_comparison_feb_may_2024"],
                    output_type=CACComparison,
                ),
                expected_output=ground_truth["cac_comparison_feb_may_2024"],
            ),
            Case(
                name="arman_9",
                inputs=Query(
                    query=QUERIES["arpu_tech_oct_2024"],
                    output_type=float,
                ),
                expected_output=ground_truth["arpu_tech_oct_2024"],
            ),
            Case(
                name="arman_10",
                inputs=Query(
                    query=QUERIES["mrr_dec_2024"],
                    output_type=float,
                ),
                expected_output=ground_truth["mrr_dec_2024"],
            ),
        ],
        evaluators=[EqEvaluator[ResultT]()],
    )

    return dataset


def generate_csv():
    """Generate CSV file with evaluation cases."""
    ground_truth = calculate_arman_ground_truth()

    eval_cases = [
        {
            "query": QUERIES["revenue_retention_aug_2023_12m"],
            "expected_output": str(ground_truth["revenue_retention_aug_2023_12m"]),
        },
        {
            "query": QUERIES["revenue_retention_sep_2023_6m"],
            "expected_output": str(ground_truth["revenue_retention_sep_2023_6m"]),
        },
        {
            "query": QUERIES["revenue_retention_feb_2023_12m"],
            "expected_output": str(ground_truth["revenue_retention_feb_2023_12m"]),
        },
        {
            "query": QUERIES["content_cac_jul_dec_2024"],
            "expected_output": str(ground_truth["content_cac_jul_dec_2024"]),
        },
        {
            "query": QUERIES["monthly_pro_churn_2023"],
            "expected_output": str(ground_truth["monthly_pro_churn_2023"]),
        },
        {
            "query": QUERIES["monthly_basic_customers_dec_feb"],
            "expected_output": str(ground_truth["monthly_basic_customers_dec_feb"]),
        },
        {
            "query": QUERIES["contribution_margin_2024"],
            "expected_output": str(ground_truth["contribution_margin_2024"]),
        },
        {
            "query": QUERIES["cac_comparison_feb_may_2024"],
            "expected_output": str(ground_truth["cac_comparison_feb_may_2024"]),
        },
        {
            "query": QUERIES["arpu_tech_oct_2024"],
            "expected_output": str(ground_truth["arpu_tech_oct_2024"]),
        },
        {
            "query": QUERIES["mrr_dec_2024"],
            "expected_output": str(ground_truth["mrr_dec_2024"]),
        },
    ]

    df = pd.DataFrame(eval_cases)
    output_path = Path("evals/scenario1/arman_evaluation.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Arman evaluation dataset saved to {output_path}")
    print("Function used: calculate_arman_ground_truth()")
    print(f"Generated {len(eval_cases)} evaluation cases")

    return df


def evaluate(workspace_name: str = "1"):
    thinking = False
    name = f"arman_evals_thinking_{thinking}"
    model: KnownModelName | FallbackModel = FallbackModel(
        "anthropic:claude-4-sonnet-20250514",
        "google-gla:gemini-2.5-flash-preview-05-20",
        "openai:gpt-4.1",
        "openai:gpt-4.1-mini",
    )
    dataset = create_arman_dataset()
    workspace_dir = MAIN_DIR / f"workspaces/{workspace_name}"
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
        logger.info(f"Removed {workspace_dir}")
    workspace_dir.mkdir(parents=True, exist_ok=True)
    workspace_data_dir = workspace_dir / "data"
    source_data_dir = MAIN_DIR / "operateai_scenario1_data"
    logger.info(f"Source data directory: {source_data_dir}")
    logger.info(f"Workspace data directory: {workspace_data_dir}")
    if source_data_dir.exists():
        workspace_data_dir.mkdir(parents=True, exist_ok=True)
        # Copy contents from source to workspace data dir
        for item in source_data_dir.iterdir():
            dest = workspace_data_dir / item.name
            if item.is_file():
                shutil.copy2(item, dest)
            elif item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
        logger.info(f"Copied data from {source_data_dir} to {workspace_data_dir}")
    else:
        logger.warning(f"Source data directory {source_data_dir} does not exist")

    report = dataset.evaluate_sync(
        task=partial(eval_task, model=model, use_thinking=thinking, workspace_dir=workspace_dir), name=name
    )
    report.print(include_output=True, include_expected_output=True, include_input=True, include_averages=True)


if __name__ == "__main__":
    # Generate CSV
    # generate_csv()

    # Run evaluation
    evaluate(workspace_name="1")
