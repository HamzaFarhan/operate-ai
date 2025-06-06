"""
Step 5: CAC to LTV Analysis
Combines ground truth generation with evaluation dataset creation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import logfire
import pandas as pd
from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset

from operate_ai.evals import EqEvaluator, Query, eval_task

logfire.configure()


class LTVCACRatios(BaseModel):
    Paid_Search: float = Field(description="LTV/CAC ratio for Paid Search channel")
    Social_Media: float = Field(description="LTV/CAC ratio for Social Media channel")
    Email: float = Field(description="LTV/CAC ratio for Email channel")
    Affiliate: float = Field(description="LTV/CAC ratio for Affiliate channel")
    Content: float = Field(description="LTV/CAC ratio for Content channel")


class CACByChannel(BaseModel):
    Paid_Search: int = Field(description="Customer Acquisition Cost for Paid Search")
    Social_Media: int = Field(description="Customer Acquisition Cost for Social Media")
    Email: int = Field(description="Customer Acquisition Cost for Email")
    Affiliate: int = Field(description="Customer Acquisition Cost for Affiliate")
    Content: int = Field(description="Customer Acquisition Cost for Content")


class ProfitabilityAnalysis(BaseModel):
    Paid_Search: dict[str, Any] = Field(description="Profitability analysis for Paid Search")
    Social_Media: dict[str, Any] = Field(description="Profitability analysis for Social Media")
    Email: dict[str, Any] = Field(description="Profitability analysis for Email")
    Affiliate: dict[str, Any] = Field(description="Profitability analysis for Affiliate")
    Content: dict[str, Any] = Field(description="Profitability analysis for Content")


class ExcellentChannels(BaseModel):
    channels: dict[str, float] = Field(description="Channels with LTV/CAC ratio >= 3x")


class AnalysisNotes(BaseModel):
    excellent_threshold: str = Field(description="Threshold for excellent profitability")
    good_threshold: str = Field(description="Threshold for good profitability")
    breakeven_threshold: str = Field(description="Threshold for break-even profitability")
    interpretation: str = Field(description="Interpretation guide for LTV/CAC ratios")


type ResultT = LTVCACRatios | CACByChannel | ProfitabilityAnalysis | ExcellentChannels | AnalysisNotes


def get_jan_2023_cohort_cac_ltv_analysis():
    """
    Calculates CAC to LTV ratios for customers who were active in Jan 2023.

    Returns:
        dict: CAC to LTV analysis by acquisition channel
    """
    # Import LTV function from previous step
    import sys

    sys.path.append("evals/scenario1")
    try:
        from step4_evals import get_jan_2023_cohort_ltv
    except ImportError:
        from step4_ltv_calculation import get_jan_2023_cohort_ltv

    # Get LTV data
    ltv_data = get_jan_2023_cohort_ltv()
    ltv_by_channel = ltv_data["ltv_by_channel"]

    # Load the generated data to get CAC information
    data_dir = Path("operateai_scenario1_data")
    customers_df = pd.read_csv(data_dir / "customers.csv")
    subscriptions_df = pd.read_csv(data_dir / "subscriptions.csv")

    # Convert dates
    subscriptions_df["StartDate"] = pd.to_datetime(subscriptions_df["StartDate"])
    subscriptions_df["EndDate"] = pd.to_datetime(subscriptions_df["EndDate"], errors="coerce")

    # Get Jan 2023 cohort (same logic as previous steps)
    jan_2023_start = datetime(2023, 1, 1)
    jan_2023_end = datetime(2023, 1, 31)

    active_subs_jan_2023 = subscriptions_df[
        (subscriptions_df["StartDate"] <= jan_2023_end)
        & ((subscriptions_df["EndDate"] >= jan_2023_start) | pd.isna(subscriptions_df["EndDate"]))
    ]

    active_customer_ids = active_subs_jan_2023["CustomerID"].unique()

    # Get customer acquisition costs by channel
    cohort_customers = customers_df[customers_df["CustomerID"].isin(active_customer_ids)]

    # Calculate average CAC by channel using realistic industry estimates
    # Since CAC data isn't in the CSV, we'll use typical SaaS CAC by channel
    typical_cac_by_channel = {
        "Paid Search": 800,
        "Social Media": 600,
        "Email": 250,
        "Affiliate": 500,
        "Content": 400,
    }

    cac_by_channel = {}
    customer_count_by_channel = {}

    for channel in cohort_customers["AcquisitionChannel"].unique():
        channel_customers = cohort_customers[cohort_customers["AcquisitionChannel"] == channel]
        customer_count = len(channel_customers)

        # Use typical CAC for the channel, or default if not found
        cac_by_channel[channel] = typical_cac_by_channel.get(channel, 500)
        customer_count_by_channel[channel] = customer_count

    # Calculate CAC to LTV ratios
    cac_ltv_ratios = {}
    ltv_cac_ratios = {}  # LTV/CAC is more commonly used (higher is better)
    profitability_analysis = {}

    for channel in ltv_by_channel.keys():
        if channel in cac_by_channel:
            ltv = ltv_by_channel[channel]
            cac = cac_by_channel[channel]

            cac_ltv_ratio = round(cac / ltv, 4) if ltv > 0 else float("inf")
            ltv_cac_ratio = round(ltv / cac, 2) if cac > 0 else float("inf")

            cac_ltv_ratios[channel] = cac_ltv_ratio
            ltv_cac_ratios[channel] = ltv_cac_ratio

            # Profitability analysis
            if ltv_cac_ratio >= 3.0:
                profitability = "Excellent (LTV/CAC >= 3x)"
            elif ltv_cac_ratio >= 2.0:
                profitability = "Good (LTV/CAC >= 2x)"
            elif ltv_cac_ratio >= 1.0:
                profitability = "Break-even (LTV/CAC >= 1x)"
            else:
                profitability = "Unprofitable (LTV/CAC < 1x)"

            profitability_analysis[channel] = {
                "ltv": ltv,
                "cac": cac,
                "ltv_cac_ratio": ltv_cac_ratio,
                "customer_count": customer_count_by_channel[channel],
                "profitability_status": profitability,
            }

    return {
        "ltv_by_channel": ltv_by_channel,
        "cac_by_channel": cac_by_channel,
        "cac_ltv_ratios": cac_ltv_ratios,
        "ltv_cac_ratios": ltv_cac_ratios,
        "profitability_analysis": profitability_analysis,
        "customer_count_by_channel": customer_count_by_channel,
        "analysis_notes": {
            "excellent_threshold": "LTV/CAC >= 3x",
            "good_threshold": "LTV/CAC >= 2x",
            "breakeven_threshold": "LTV/CAC >= 1x",
            "interpretation": "Higher LTV/CAC ratios indicate more profitable acquisition channels",
        },
    }


# Define queries once for reuse
QUERIES = {
    "ltv_cac_ratios": "Calculate the LTV to CAC ratio for each acquisition channel for customers who were active in January 2023. Active customers are those who had subscriptions that started before or during Jan 2023 AND either ended after Jan 1 2023 or are still ongoing. Use LTV = (ARPU / Churn Rate) × 75% profit margin formula with 2023 data. For CAC, use industry standard estimates: Paid Search $800, Social Media $600, Email $250, Affiliate $500, Content $400.",
    "cac_by_channel": "Show the Customer Acquisition Cost (CAC) for each acquisition channel for customers who were active in January 2023. Use industry standard estimates.",
    "profitability_analysis": "Provide a complete profitability analysis comparing LTV and CAC by acquisition channel for customers who were active in January 2023. Include LTV values, CAC costs, LTV/CAC ratios, customer counts, and profitability categorization (Excellent 3x+, Good 2x+, Break-even 1x+, Unprofitable <1x).",
    "excellent_channels": "Which acquisition channels have an LTV/CAC ratio above 3x (excellent profitability) for customers who were active in January 2023?",
    "analysis_notes": "What is the interpretation guide for LTV/CAC ratios and what thresholds should be used?",
}


def create_step5_dataset():
    """Create the evaluation dataset using ground truth data"""
    # Get ground truth
    ground_truth = get_jan_2023_cohort_cac_ltv_analysis()

    # Convert ground truth data to expected format
    ltv_cac_data = ground_truth["ltv_cac_ratios"]
    cac_data = ground_truth["cac_by_channel"]
    prof_data = ground_truth["profitability_analysis"]
    analysis_notes = ground_truth["analysis_notes"]

    dataset = Dataset[Query[ResultT], ResultT](
        cases=[
            Case(
                name="step5_1",
                inputs=Query(
                    query=QUERIES["ltv_cac_ratios"],
                    output_type=LTVCACRatios,
                ),
                expected_output=LTVCACRatios(
                    Paid_Search=ltv_cac_data.get("Paid Search", 0.0),
                    Social_Media=ltv_cac_data.get("Social Media", 0.0),
                    Email=ltv_cac_data.get("Email", 0.0),
                    Affiliate=ltv_cac_data.get("Affiliate", 0.0),
                    Content=ltv_cac_data.get("Content", 0.0),
                ),
            ),
            Case(
                name="step5_2",
                inputs=Query(
                    query=QUERIES["cac_by_channel"],
                    output_type=CACByChannel,
                ),
                expected_output=CACByChannel(
                    Paid_Search=cac_data.get("Paid Search", 0),
                    Social_Media=cac_data.get("Social Media", 0),
                    Email=cac_data.get("Email", 0),
                    Affiliate=cac_data.get("Affiliate", 0),
                    Content=cac_data.get("Content", 0),
                ),
            ),
            Case(
                name="step5_3",
                inputs=Query(
                    query=QUERIES["profitability_analysis"],
                    output_type=ProfitabilityAnalysis,
                ),
                expected_output=ProfitabilityAnalysis(
                    Paid_Search=prof_data.get("Paid Search", {}),
                    Social_Media=prof_data.get("Social Media", {}),
                    Email=prof_data.get("Email", {}),
                    Affiliate=prof_data.get("Affiliate", {}),
                    Content=prof_data.get("Content", {}),
                ),
            ),
            Case(
                name="step5_4",
                inputs=Query(
                    query=QUERIES["excellent_channels"],
                    output_type=ExcellentChannels,
                ),
                expected_output=ExcellentChannels(channels={k: v for k, v in ltv_cac_data.items() if v >= 3.0}),
            ),
            Case(
                name="step5_5",
                inputs=Query(
                    query=QUERIES["analysis_notes"],
                    output_type=AnalysisNotes,
                ),
                expected_output=AnalysisNotes(
                    excellent_threshold=analysis_notes["excellent_threshold"],
                    good_threshold=analysis_notes["good_threshold"],
                    breakeven_threshold=analysis_notes["breakeven_threshold"],
                    interpretation=analysis_notes["interpretation"],
                ),
            ),
        ],
        evaluators=[EqEvaluator[ResultT]()],
    )

    return dataset


def generate_csv():
    """Generate CSV file for step 5 evaluation"""
    # Get ground truth
    ground_truth = get_jan_2023_cohort_cac_ltv_analysis()

    eval_cases = [
        {
            "query": QUERIES["ltv_cac_ratios"],
            "expected_output": json.dumps(ground_truth["ltv_cac_ratios"], indent=2),
        },
        {
            "query": QUERIES["cac_by_channel"],
            "expected_output": json.dumps(ground_truth["cac_by_channel"], indent=2),
        },
        {
            "query": QUERIES["profitability_analysis"],
            "expected_output": json.dumps(ground_truth["profitability_analysis"], indent=2),
        },
        {
            "query": QUERIES["excellent_channels"],
            "expected_output": json.dumps(
                {k: v for k, v in ground_truth["ltv_cac_ratios"].items() if v >= 3.0}, indent=2
            ),
        },
        {
            "query": QUERIES["analysis_notes"],
            "expected_output": json.dumps(ground_truth["analysis_notes"], indent=2),
        },
    ]

    # Save to CSV
    eval_df = pd.DataFrame(eval_cases)
    output_path = Path("evals/scenario1/step5_evaluation.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(output_path, index=False)

    print(f"Step 5 evaluation dataset saved to {output_path}")
    print("Function used: get_jan_2023_cohort_cac_ltv_analysis()")
    print(f"Generated {len(eval_cases)} evaluation cases")

    # Print channel profitability summary
    print("\nChannel Profitability Summary:")
    for channel, analysis in ground_truth["profitability_analysis"].items():
        print(
            f"{channel}: LTV ${analysis['ltv']} / CAC ${analysis['cac']} = {analysis['ltv_cac_ratio']}x ({analysis['profitability_status']})"
        )

    return eval_df


def evaluate():
    dataset = create_step5_dataset()
    report = dataset.evaluate_sync(task=eval_task, name="step5_evals")
    report.print(include_output=True, include_expected_output=True, include_input=True, include_averages=True)


if __name__ == "__main__":
    # Generate CSV
    generate_csv()

    # Run evaluation (can be commented out)
    # evaluate()
