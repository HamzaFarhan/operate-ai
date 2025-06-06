"""
Step 4: Customer Lifetime Value (LTV) Calculation
Combines ground truth generation with evaluation dataset creation.
"""

import json
from pathlib import Path

import logfire
import pandas as pd
from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset

from operate_ai.evals import EqEvaluator, Query, eval_task

logfire.configure()


class LTVBySubscriptionType(BaseModel):
    Monthly: float = Field(description="LTV for monthly subscription customers")
    Annual: float = Field(description="LTV for annual subscription customers")
    Total: float = Field(description="Combined LTV across all subscription types")


class LTVByPlanType(BaseModel):
    Basic: float = Field(description="LTV for Basic plan customers")
    Pro: float = Field(description="LTV for Pro plan customers")
    Enterprise: float = Field(description="LTV for Enterprise plan customers")


class LTVByIndustry(BaseModel):
    Retail: float = Field(description="LTV for Retail industry customers")
    Tech: float = Field(description="LTV for Tech industry customers")
    Healthcare: float = Field(description="LTV for Healthcare industry customers")
    Education: float = Field(description="LTV for Education industry customers")
    Other: float = Field(description="LTV for Other industry customers")


class LTVByChannel(BaseModel):
    PaidSearch: float = Field(description="LTV for Paid Search acquired customers")
    SocialMedia: float = Field(description="LTV for Social Media acquired customers")
    Email: float = Field(description="LTV for Email acquired customers")
    Affiliate: float = Field(description="LTV for Affiliate acquired customers")
    Content: float = Field(description="LTV for Content acquired customers")


class LTVFormulaInfo(BaseModel):
    profit_margin: float = Field(description="Profit margin used in LTV calculation")
    formula: str = Field(description="LTV calculation formula")


type ResultT = float | LTVBySubscriptionType | LTVByPlanType | LTVByIndustry | LTVByChannel | LTVFormulaInfo


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
    try:
        from step2_evals import get_jan_2023_cohort_arpu  # type: ignore
        from step3_evals import get_jan_2023_cohort_churn_rates  # type: ignore
    except ImportError:
        # Fallback to original file names if consolidated files don't exist
        try:
            from step2_arpu_calculation import get_jan_2023_cohort_arpu  # type: ignore
            from step3_churn_rate_calculation import get_jan_2023_cohort_churn_rates  # type: ignore
        except ImportError:
            raise ImportError("Required step2 and step3 files not found")

    # Get ARPU and churn rate data
    arpu_data = get_jan_2023_cohort_arpu()  # type: ignore
    churn_data = get_jan_2023_cohort_churn_rates()  # type: ignore

    # Profit margin assumption (typically 70-80% for SaaS companies)
    PROFIT_MARGIN = 0.75  # 75% profit margin

    def calculate_ltv(arpu: float, churn_rate: float) -> float:
        """Calculate LTV using the formula"""
        if churn_rate == 0:
            # If no churn, use 5-year assumption (20% annual churn)
            churn_rate = 0.2
        return round((arpu / churn_rate) * PROFIT_MARGIN, 2)

    # Calculate LTV by Subscription Type
    ltv_by_sub_type: dict[str, float] = {}
    arpu_sub = arpu_data.get("arpu_by_subscription_type", {})  # type: ignore
    churn_sub = churn_data.get("churn_by_subscription_type", {})  # type: ignore

    if isinstance(arpu_sub, dict) and isinstance(churn_sub, dict):
        for sub_type in arpu_sub.keys():
            if (
                sub_type in churn_sub
                and isinstance(arpu_sub[sub_type], (int, float))
                and isinstance(churn_sub[sub_type], (int, float))
            ):
                ltv_by_sub_type[sub_type] = calculate_ltv(float(arpu_sub[sub_type]), float(churn_sub[sub_type]))

    # Calculate LTV by Plan Type
    ltv_by_plan_type: dict[str, float] = {}
    arpu_plan = arpu_data.get("arpu_by_plan_type", {})  # type: ignore
    churn_plan = churn_data.get("churn_by_plan_type", {})  # type: ignore

    if isinstance(arpu_plan, dict) and isinstance(churn_plan, dict):
        for plan_type in arpu_plan.keys():
            if (
                plan_type in churn_plan
                and isinstance(arpu_plan[plan_type], (int, float))
                and isinstance(churn_plan[plan_type], (int, float))
            ):
                ltv_by_plan_type[plan_type] = calculate_ltv(
                    float(arpu_plan[plan_type]), float(churn_plan[plan_type])
                )

    # Calculate LTV by Industry
    ltv_by_industry: dict[str, float] = {}
    arpu_industry = arpu_data.get("arpu_by_industry", {})  # type: ignore
    churn_industry = churn_data.get("churn_by_industry", {})  # type: ignore

    if isinstance(arpu_industry, dict) and isinstance(churn_industry, dict):
        for industry in arpu_industry.keys():
            if (
                industry in churn_industry
                and isinstance(arpu_industry[industry], (int, float))
                and isinstance(churn_industry[industry], (int, float))
            ):
                ltv_by_industry[industry] = calculate_ltv(
                    float(arpu_industry[industry]), float(churn_industry[industry])
                )

    # Calculate LTV by Acquisition Channel
    ltv_by_channel: dict[str, float] = {}
    arpu_channel = arpu_data.get("arpu_by_channel", {})  # type: ignore
    churn_channel = churn_data.get("churn_by_channel", {})  # type: ignore

    if isinstance(arpu_channel, dict) and isinstance(churn_channel, dict):
        for channel in arpu_channel.keys():
            if (
                channel in churn_channel
                and isinstance(arpu_channel[channel], (int, float))
                and isinstance(churn_channel[channel], (int, float))
            ):
                ltv_by_channel[channel] = calculate_ltv(
                    float(arpu_channel[channel]), float(churn_channel[channel])
                )

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


# Define queries once for reuse
QUERIES = {
    "by_subscription_type": "Calculate the Customer Lifetime Value (LTV) for customers who were active in January 2023, broken down by their initial subscription type (Monthly, Annual, and Total combined). Active customers are those who had subscriptions that started before or during Jan 2023 AND either ended after Jan 1 2023 or are still ongoing. Use the formula: LTV = (ARPU / Churn Rate) × 75% profit margin. ARPU should be calculated from 2023 profit data, churn rate from 2023 churns (assume 20% if 0% churn).",
    "by_plan_type": "Show LTV breakdown by initial plan type (Basic, Pro, Enterprise) for customers who were active in January 2023. Use LTV = (ARPU / Churn Rate) × 75% profit margin formula.",
    "by_industry": "Calculate LTV by industry segment for customers who were active in January 2023. Apply the formula LTV = (ARPU / Churn Rate) × 75% profit margin using industry-specific ARPU and churn rates.",
    "by_channel": "What is the LTV by acquisition channel for customers who were active in January 2023? Calculate using the standard SaaS formula: LTV = (ARPU / Churn Rate) × 75% profit margin.",
    "formula_info": "What profit margin was used in the LTV calculation and what is the formula?",
}


def create_step4_dataset():
    """Create the evaluation dataset using ground truth data"""
    # Get ground truth
    ground_truth = get_jan_2023_cohort_ltv()

    # Convert ground truth data to expected format
    subscription_ltv = ground_truth["ltv_by_subscription_type"]
    plan_ltv = ground_truth["ltv_by_plan_type"]
    industry_ltv = ground_truth["ltv_by_industry"]
    channel_ltv = ground_truth["ltv_by_channel"]

    dataset = Dataset[Query[ResultT], ResultT](
        cases=[
            Case(
                name="step4_1",
                inputs=Query(
                    query=QUERIES["by_subscription_type"],
                    output_type=LTVBySubscriptionType,
                ),
                expected_output=LTVBySubscriptionType(
                    Monthly=subscription_ltv.get("Monthly", 0.0),
                    Annual=subscription_ltv.get("Annual", 0.0),
                    Total=subscription_ltv.get("Total", 0.0),
                ),
            ),
            Case(
                name="step4_2",
                inputs=Query(
                    query=QUERIES["by_plan_type"],
                    output_type=LTVByPlanType,
                ),
                expected_output=LTVByPlanType(
                    Basic=plan_ltv.get("Basic", 0.0),
                    Pro=plan_ltv.get("Pro", 0.0),
                    Enterprise=plan_ltv.get("Enterprise", 0.0),
                ),
            ),
            Case(
                name="step4_3",
                inputs=Query(
                    query=QUERIES["by_industry"],
                    output_type=LTVByIndustry,
                ),
                expected_output=LTVByIndustry(
                    Retail=industry_ltv.get("Retail", 0.0),
                    Tech=industry_ltv.get("Tech", 0.0),
                    Healthcare=industry_ltv.get("Healthcare", 0.0),
                    Education=industry_ltv.get("Education", 0.0),
                    Other=industry_ltv.get("Other", 0.0),
                ),
            ),
            Case(
                name="step4_4",
                inputs=Query(
                    query=QUERIES["by_channel"],
                    output_type=LTVByChannel,
                ),
                expected_output=LTVByChannel(
                    PaidSearch=channel_ltv.get("Paid Search", 0.0),
                    SocialMedia=channel_ltv.get("Social Media", 0.0),
                    Email=channel_ltv.get("Email", 0.0),
                    Affiliate=channel_ltv.get("Affiliate", 0.0),
                    Content=channel_ltv.get("Content", 0.0),
                ),
            ),
            Case(
                name="step4_5",
                inputs=Query(
                    query=QUERIES["formula_info"],
                    output_type=LTVFormulaInfo,
                ),
                expected_output=LTVFormulaInfo(
                    profit_margin=ground_truth["profit_margin_used"],
                    formula=ground_truth["ltv_formula"],
                ),
            ),
        ],
        evaluators=[EqEvaluator[ResultT]()],
    )

    return dataset


def generate_csv():
    """Generate CSV file for step4 evaluation without strategy column"""
    # Get ground truth
    ground_truth = get_jan_2023_cohort_ltv()

    eval_cases = [
        {
            "query": QUERIES["by_subscription_type"],
            "expected_output": json.dumps(ground_truth["ltv_by_subscription_type"], indent=2),
        },
        {
            "query": QUERIES["by_plan_type"],
            "expected_output": json.dumps(ground_truth["ltv_by_plan_type"], indent=2),
        },
        {
            "query": QUERIES["by_industry"],
            "expected_output": json.dumps(ground_truth["ltv_by_industry"], indent=2),
        },
        {
            "query": QUERIES["by_channel"],
            "expected_output": json.dumps(ground_truth["ltv_by_channel"], indent=2),
        },
        {
            "query": QUERIES["formula_info"],
            "expected_output": json.dumps(
                {"profit_margin": ground_truth["profit_margin_used"], "formula": ground_truth["ltv_formula"]},
                indent=2,
            ),
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
    if "Total" in ground_truth["ltv_by_subscription_type"]:
        print(f"Total LTV: ${ground_truth['ltv_by_subscription_type']['Total']}")
    if "Enterprise" in ground_truth["ltv_by_plan_type"]:
        print(f"Enterprise LTV: ${ground_truth['ltv_by_plan_type']['Enterprise']}")
    if "Annual" in ground_truth["ltv_by_subscription_type"]:
        print(f"Annual LTV: ${ground_truth['ltv_by_subscription_type']['Annual']}")

    return eval_df


def evaluate():
    """Run the evaluation"""
    dataset = create_step4_dataset()
    report = dataset.evaluate_sync(task=eval_task, name="step4_evals")
    report.print(include_output=True, include_expected_output=True, include_input=True, include_averages=True)


if __name__ == "__main__":
    # Generate CSV
    generate_csv()

    # Run evaluation (can be commented out)
    # evaluate()
