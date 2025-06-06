from functools import partial

import logfire
import pandas as pd
from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset

from operate_ai.evals import DELTA, EqEvaluator, Query, eval_task

logfire.configure()


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

QUERIES = {
    "revenue_retention_jan_2023_12m": "What is the January 2023 annual subscription type by initial subscription 12 month revenue retention? Calculate as 12 month revenue divided by initial revenue.",
    "revenue_retention_dec_2023_6m": "What is the December 2023 annual subscription type by initial subscription 6 month revenue retention? Calculate as 6 month revenue divided by initial revenue.",
    "revenue_retention_jan_2023": "What is the January 2023 annual subscription type 12 month revenue retention? Calculate as 12 month revenue divided by initial revenue.",
    "content_cac_jul_dec_2024": "What is Content CAC from July 2024 to Dec 2024? Calculate as total spend divided by acquisitions.",
    "monthly_pro_churn_2023": "What is monthly plan & pro subscription type churn rate from Jan 2023 to Dec 2023? Calculate as churned subscribers divided by active subscribers on Jan 2023.",
    "monthly_basic_customers_dec_feb": "How many unique customers ordered a monthly basic plan from Dec 2023 to Feb 2024?",
    "contribution_margin_2024": "What is 2024 Contribution Margin? Calculate as sum of 2024 profit minus sum of 2024 marketing expenses.",
    "cac_comparison_feb_may_2024": "Which marketing acquisition channel has the highest CAC and which has the lowest during the period Feb 2024 to May 2024?",
    "arpu_tech_oct_2024": "What is ARPU for Tech customers in Oct 2024? Calculate as total revenue divided by number of customers.",
    "mrr_dec_2024": "What is MRR as of Dec 2024? Calculate based on customer count and ARPU.",
}

arman_dataset = Dataset[Query[ResultT], ResultT](
    cases=[
        Case(
            name="arman_1",
            inputs=Query(
                query=QUERIES["revenue_retention_jan_2023_12m"],
                output_type=float,
            ),
            expected_output=1.41,
        ),
        Case(
            name="arman_2",
            inputs=Query(
                query=QUERIES["revenue_retention_dec_2023_6m"],
                output_type=float,
            ),
            expected_output=0.0,
        ),
        Case(
            name="arman_3",
            inputs=Query(
                query=QUERIES["revenue_retention_jan_2023"],
                output_type=float,
            ),
            expected_output=1.55,
        ),
        Case(
            name="arman_4",
            inputs=Query(
                query=QUERIES["content_cac_jul_dec_2024"],
                output_type=float,
            ),
            expected_output=2921.0,
        ),
        Case(
            name="arman_5",
            inputs=Query(
                query=QUERIES["monthly_pro_churn_2023"],
                output_type=float,
            ),
            expected_output=0.36,
        ),
        Case(
            name="arman_6",
            inputs=Query(
                query=QUERIES["monthly_basic_customers_dec_feb"],
                output_type=int,
            ),
            expected_output=35,
        ),
        Case(
            name="arman_7",
            inputs=Query(
                query=QUERIES["contribution_margin_2024"],
                output_type=float,
            ),
            expected_output=-1441.29,
        ),
        Case(
            name="arman_8",
            inputs=Query(
                query=QUERIES["cac_comparison_feb_may_2024"],
                output_type=CACComparison,
            ),
            expected_output=CACComparison(
                highest_cac_channel="Affiliate",
                highest_cac_value=4838.0,
                lowest_cac_channel="Email",
                lowest_cac_value=571.0,
            ),
        ),
        Case(
            name="arman_9",
            inputs=Query(
                query=QUERIES["arpu_tech_oct_2024"],
                output_type=float,
            ),
            expected_output=252.857143,
        ),
        Case(
            name="arman_10",
            inputs=Query(
                query=QUERIES["mrr_dec_2024"],
                output_type=float,
            ),
            expected_output=13770.0,
        ),
    ],
    evaluators=[EqEvaluator[ResultT]()],
)


def generate_csv():
    """Generate CSV file with evaluation cases."""
    eval_cases = [
        {
            "query": QUERIES["revenue_retention_jan_2023_12m"],
            "expected_output": "1.41",
        },
        {
            "query": QUERIES["revenue_retention_dec_2023_6m"],
            "expected_output": "0.0",
        },
        {
            "query": QUERIES["revenue_retention_jan_2023"],
            "expected_output": "1.55",
        },
        {
            "query": QUERIES["content_cac_jul_dec_2024"],
            "expected_output": "2921.0",
        },
        {
            "query": QUERIES["monthly_pro_churn_2023"],
            "expected_output": "0.36",
        },
        {
            "query": QUERIES["monthly_basic_customers_dec_feb"],
            "expected_output": "35",
        },
        {
            "query": QUERIES["contribution_margin_2024"],
            "expected_output": "-1441.29",
        },
        {
            "query": QUERIES["cac_comparison_feb_may_2024"],
            "expected_output": str(
                CACComparison(
                    highest_cac_channel="Affiliate",
                    highest_cac_value=4838.0,
                    lowest_cac_channel="Email",
                    lowest_cac_value=571.0,
                )
            ),
        },
        {
            "query": QUERIES["arpu_tech_oct_2024"],
            "expected_output": "252.857143",
        },
        {
            "query": QUERIES["mrr_dec_2024"],
            "expected_output": "13770.0",
        },
    ]

    df = pd.DataFrame(eval_cases)
    df.to_csv("evals/scenario1/arman_evaluation.csv", index=False)
    print(f"Generated CSV with {len(eval_cases)} evaluation cases")


def evaluate():
    thinking = False
    name = f"arman_evals_thinking_{thinking}"
    report = arman_dataset.evaluate_sync(task=partial(eval_task, use_thinking=thinking), name=name)
    report.print(include_output=True, include_expected_output=True, include_input=True, include_averages=True)


if __name__ == "__main__":
    evaluate()
