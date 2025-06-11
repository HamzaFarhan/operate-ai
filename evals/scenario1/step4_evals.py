"""
Step 4: Customer Lifetime Value (LTV) Calculation
Combines ground truth generation with evaluation dataset creation.
"""

import json
from pathlib import Path
from typing import Any

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


class LTVData(BaseModel):
    """Structured return type for LTV calculation function"""

    ltv_by_subscription_type: dict[str, float]
    ltv_by_plan_type: dict[str, float]
    ltv_by_industry: dict[str, float]
    ltv_by_channel: dict[str, float]
    profit_margin_used: float
    ltv_formula: str
    underlying_arpu_data: Any
    underlying_churn_data: Any


def get_jan_2023_cohort_ltv() -> LTVData:
    """
    Calculates LTV for customers who were active in Jan 2023.
    Uses formula: LTV = (ARPU / Churn Rate) × Profit Margin

    Returns:
        LTVData: LTV calculations by various segments
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
    arpu_sub = arpu_data.get("arpu_by_subscription_type", {}) if isinstance(arpu_data, dict) else {}
    churn_sub = churn_data.get("churn_by_subscription_type", {}) if isinstance(churn_data, dict) else {}

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
    arpu_plan = arpu_data.get("arpu_by_plan_type", {}) if isinstance(arpu_data, dict) else {}
    churn_plan = churn_data.get("churn_by_plan_type", {}) if isinstance(churn_data, dict) else {}

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
    arpu_industry = arpu_data.get("arpu_by_industry", {}) if isinstance(arpu_data, dict) else {}
    churn_industry = churn_data.get("churn_by_industry", {}) if isinstance(churn_data, dict) else {}

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
    arpu_channel = arpu_data.get("arpu_by_channel", {}) if isinstance(arpu_data, dict) else {}
    churn_channel = churn_data.get("churn_by_channel", {}) if isinstance(churn_data, dict) else {}

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

    return LTVData(
        ltv_by_subscription_type=ltv_by_sub_type,
        ltv_by_plan_type=ltv_by_plan_type,
        ltv_by_industry=ltv_by_industry,
        ltv_by_channel=ltv_by_channel,
        profit_margin_used=PROFIT_MARGIN,
        ltv_formula="LTV = (ARPU / Churn Rate) × Profit Margin",
        underlying_arpu_data=arpu_data,
        underlying_churn_data=churn_data,
    )


# Define queries once for reuse - improved for clarity according to evaluation rules
QUERIES = {
    "by_subscription_type": """Calculate the Customer Lifetime Value (LTV) for customers who were active in January 2023, broken down by their initial subscription type.

Step-by-step logic:
1. Identify customers who were active in January 2023 (subscription StartDate ≤ Jan 31, 2023 AND (EndDate ≥ Jan 1, 2023 OR EndDate is null))
2. Group these customers by their initial subscription type (Monthly, Annual)
3. Calculate ARPU for each subscription type using 2023 profit data
4. Calculate churn rate for each subscription type using 2023 churn data
5. Apply formula: LTV = (ARPU / Churn Rate) × 75% profit margin
6. If churn rate is 0%, assume 20% annual churn rate
7. Calculate Total as combined LTV across all subscription types

Return as: {"Monthly": X, "Annual": Y, "Total": Z} where X, Y, Z are floats rounded to 2 decimal places.""",
    "by_plan_type": """Calculate LTV breakdown by initial plan type for customers who were active in January 2023.

Step-by-step logic:
1. Use the same active customer definition as above (active during January 2023)
2. Group customers by their initial plan type (Basic, Pro, Enterprise)
3. For each plan type, calculate LTV using: LTV = (ARPU / Churn Rate) × 75% profit margin
4. Use plan-specific ARPU and churn rates from 2023 data
5. Apply 20% churn rate if actual churn rate is 0%

Return as: {"Basic": X, "Pro": Y, "Enterprise": Z} where X, Y, Z are floats rounded to 2 decimal places.""",
    "by_industry": """Calculate LTV by industry segment for customers who were active in January 2023.

Step-by-step logic:
1. Use the same active customer definition as above (active during January 2023)
2. Group customers by their industry segment (Retail, Tech, Healthcare, Education, Other)
3. For each industry, calculate LTV using: LTV = (ARPU / Churn Rate) × 75% profit margin
4. Use industry-specific ARPU and churn rates from 2023 data
5. Apply 20% churn rate if actual churn rate is 0%

Return as: {"Retail": W, "Tech": X, "Healthcare": Y, "Education": Z, "Other": A} where all values are floats rounded to 2 decimal places.""",
    "by_channel": """Calculate LTV by acquisition channel for customers who were active in January 2023.

Step-by-step logic:
1. Use the same active customer definition as above (active during January 2023)
2. Group customers by their acquisition channel (Paid Search, Social Media, Email, Affiliate, Content)
3. For each channel, calculate LTV using: LTV = (ARPU / Churn Rate) × 75% profit margin
4. Use channel-specific ARPU and churn rates from 2023 data
5. Apply 20% churn rate if actual churn rate is 0%

Return as: {"PaidSearch": V, "SocialMedia": W, "Email": X, "Affiliate": Y, "Content": Z} where all values are floats rounded to 2 decimal places.
Note: Map "Paid Search" to "PaidSearch" and "Social Media" to "SocialMedia" in the response.""",
    "formula_info": """What profit margin percentage was used in the LTV calculation and what is the exact formula?

Return as: {"profit_margin": X, "formula": "Y"} where X is the profit margin as a decimal (e.g., 0.75 for 75%) and Y is the exact formula string used.""",
}


def create_step4_dataset() -> Dataset[Query[Any], Any]:
    """Create the evaluation dataset using ground truth data"""
    # Get ground truth
    ground_truth = get_jan_2023_cohort_ltv()

    # Convert ground truth data to expected format
    subscription_ltv = ground_truth.ltv_by_subscription_type
    plan_ltv = ground_truth.ltv_by_plan_type
    industry_ltv = ground_truth.ltv_by_industry
    channel_ltv = ground_truth.ltv_by_channel

    cases: list[Case[Query[Any], Any]] = [
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
                profit_margin=ground_truth.profit_margin_used,
                formula=ground_truth.ltv_formula,
            ),
        ),
    ]

    dataset = Dataset[Query[Any], Any](
        cases=cases,
        evaluators=[EqEvaluator[Any]()],
    )

    return dataset


def generate_csv() -> pd.DataFrame:
    """Generate CSV file for step4 evaluation without strategy column"""
    # Get ground truth
    ground_truth = get_jan_2023_cohort_ltv()

    eval_cases = [
        {
            "query": QUERIES["by_subscription_type"],
            "expected_output": json.dumps(ground_truth.ltv_by_subscription_type, indent=2),
        },
        {
            "query": QUERIES["by_plan_type"],
            "expected_output": json.dumps(ground_truth.ltv_by_plan_type, indent=2),
        },
        {
            "query": QUERIES["by_industry"],
            "expected_output": json.dumps(ground_truth.ltv_by_industry, indent=2),
        },
        {
            "query": QUERIES["by_channel"],
            "expected_output": json.dumps(ground_truth.ltv_by_channel, indent=2),
        },
        {
            "query": QUERIES["formula_info"],
            "expected_output": json.dumps(
                {"profit_margin": ground_truth.profit_margin_used, "formula": ground_truth.ltv_formula},
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
    print(f"Profit margin used: {ground_truth.profit_margin_used}")
    print(f"Formula: {ground_truth.ltv_formula}")

    # Print sample LTV values for verification
    print("\nSample LTV Results:")
    if "Total" in ground_truth.ltv_by_subscription_type:
        print(f"Total LTV: ${ground_truth.ltv_by_subscription_type['Total']}")
    if "Enterprise" in ground_truth.ltv_by_plan_type:
        print(f"Enterprise LTV: ${ground_truth.ltv_by_plan_type['Enterprise']}")
    if "Annual" in ground_truth.ltv_by_subscription_type:
        print(f"Annual LTV: ${ground_truth.ltv_by_subscription_type['Annual']}")

    return eval_df


def evaluate() -> None:
    """Run the evaluation"""
    dataset = create_step4_dataset()

    # Fix the eval_task call by providing required parameters
    def task_wrapper(query: Query[Any]) -> Any:
        import asyncio

        return asyncio.run(eval_task(query=query, workspace_dir=Path.cwd(), name="step4_evals"))

    report = dataset.evaluate_sync(task=task_wrapper)
    report.print(include_output=True, include_expected_output=True, include_input=True, include_averages=True)


if __name__ == "__main__":
    # Generate CSV
    generate_csv()

    # Run evaluation (can be commented out)
    # evaluate()
