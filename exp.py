import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

import duckdb
import pandas as pd
from loguru import logger
from pydantic_ai import Agent, RunContext, Tool
from pydantic_graph import End

from operate_ai.thinking import think

DATA_DIR = "operateai_scenario1_data"


@dataclass
class Deps:
    data_dir: Path | str
    stop: bool = False


def preview_df(df_path: str, num_rows: int = 3) -> list[dict[str, str]]:
    """
    Returns the first num_rows rows of the specified DataFrame.
    Should be used for previewing the data.
    """
    return pd.read_csv(df_path).head(num_rows).to_dict(orient="records")


def list_csv_files(ctx: RunContext[Deps]) -> str:
    """
    Lists all CSV files in the specified directory.
    """
    csv_files = [str(file) for file in Path(ctx.deps.data_dir).glob("*.csv")]
    res = "Available CSV files:\n"
    for file in csv_files:
        res += file + "\n"
        res += "First 2 rows:\n"
        res += json.dumps(preview_df(df_path=file, num_rows=2)) + "\n\n"
    return res


def run_sql(query: str) -> list[dict[str, str]]:
    """
    Runs a SQL query on csv file(s) using duckdb and returns the results as a list of dictionaries.
    In the query, include the path to the csv file in single quotes.
    It will return the result by doing a to_df() and then to_dict(orient="records")

    Example Query:
    SELECT * FROM 'full_path_to_csv_file.csv'
    """
    return duckdb.sql(query).to_df().to_dict(orient="records")


def ask_user(ctx: RunContext[Deps], query: str) -> str:
    """
    Ask the user for help.
    """
    res = input(f"{query} ('q' to quit)> ")
    if res == "q":
        ctx.deps.stop = True
    return res


def show_progress(ctx: RunContext[Deps], progress: str) -> str:
    """
    Show the user the progress of the analysis.
    """
    res = input(f"{progress} ('q' to quit)> ")
    if res == "q":
        ctx.deps.stop = True
    return res


agent = Agent(
    "google-gla:gemini-2.0-flash",
    instructions=(
        (
            "Use the 'run_sql' tool to solve user queries.\n"
            "Use the 'think' tool to break down the problem into steps and outline the reasoning process.\n"
            "After each meaningful step, use the 'show_progress' tool to show the user the progress of the analysis.\n"
            "For every task, try you best, but if you're stuck use 'ask_user' as a last resort to ask for help."
        ),
        list_csv_files,
    ),
    deps_type=Deps,
    tools=[preview_df, Tool(run_sql, max_retries=5), ask_user, think, show_progress],
)

query = """
Calculate customer LTV for customer from Jan 2023 to Dec 2023 by subscription types (monthly, annual and total combined),
subscription & plan types (basic, pro and enterprise),
subscription & customer industry and
subscription and acquisition channel.
Create four separate tabs for each type. Total Customer LTV in one, Customer LTV by plan in another tab, customer LTV by Industry in another tab and Customer LTV by Channel in a separate tab.
Use initial subscription and plan type to calculate LTV, not current subscription or plan type, so keep them in the subscription and plan they started in regardless of if they changed to another subscription or plan type. 
To calculate LTV use the formula (LTV = (Average Revenue per User / Churn Rate) × Profit Margin). 
To calculate churn rate, use the formula (# of customers at the beginning of the period / # of churned customers during time period), if a customer has 0% churn assume that they will be a customer for 5 years until they churn in order to calculate LTV and keep that as a  assumption in the model that can be toggled as a driver.Create a separate tab with a CAC to LTV analysis by each acquisition channel. If there is no CAC by subscription type & channel then compare LTV by total CACs by channel.
Profit per User should only include profit generated from Jan 2023 to Dec 2023 from users who ordered during those dates.
Customers at the start of the period should only be customers who have active subscriptions in Jan 2023. 
Churned customers should be customers who were active subscribers between Jan 2023 and Dec 2023 and churned during that period and should not include customers who joined after Jan 2023 since we want the churn rate for customers who were active on Jan 2023 and not new customers who joined after Jan 2023.
"""


async def main():
    deps = Deps(data_dir=DATA_DIR)
    async with agent.iter(user_prompt=query, deps=deps) as agent_run:
        node = agent_run.next_node
        all_nodes = [node]
        while not isinstance(node, End) and not deps.stop:
            pprint(node, width=200)
            print("\n\n")
            node = await agent_run.next(node=node)
            all_nodes.append(node)
        if isinstance(node, End):
            logger.success(node.data.output)
        else:
            logger.error("Failed to get final result")


if __name__ == "__main__":
    asyncio.run(main())
content = [
    {
        "AcquisitionChannel": "Email",
        "AvgProfitMargin": 0.8291666666666648,
        "AvgRevenuePerUser": 24075.0,
        "CAC": 7040.9,
        "CACToLTVRatio": 68.04421309775724,
        "ChurnRate": 0.5,
        "CustomerCount": 12,
        "LTV": 39924.37499999991,
    },
    {
        "AcquisitionChannel": "Email",
        "AvgProfitMargin": 0.8465384615384621,
        "AvgRevenuePerUser": 27247.5,
        "CAC": 7040.9,
        "CACToLTVRatio": 78.62423291602805,
        "ChurnRate": 0.5,
        "CustomerCount": 12,
        "LTV": 46132.113461538494,
    },
    {
        "AcquisitionChannel": "Content",
        "AvgProfitMargin": 0.8,
        "AvgRevenuePerUser": 4500.0,
        "CAC": 21468.16,
        "CACToLTVRatio": 0.6707607917958502,
        "ChurnRate": 1.0,
        "CustomerCount": 4,
        "LTV": 3600.0,
    },
    {
        "AcquisitionChannel": "Content",
        "AvgProfitMargin": 0.8,
        "AvgRevenuePerUser": 600.0,
        "CAC": 21468.16,
        "CACToLTVRatio": None,
        "ChurnRate": None,
        "CustomerCount": 4,
        "LTV": None,
    },
    {
        "AcquisitionChannel": "Content",
        "AvgProfitMargin": 0.8,
        "AvgRevenuePerUser": 2250.0,
        "CAC": 21468.16,
        "CACToLTVRatio": 0.3353803958979251,
        "ChurnRate": 1.0,
        "CustomerCount": 4,
        "LTV": 1800.0,
    },
    {
        "AcquisitionChannel": "Social Media",
        "AvgProfitMargin": 0.82,
        "AvgRevenuePerUser": 2880.0,
        "CAC": 41269.94,
        "CACToLTVRatio": None,
        "ChurnRate": None,
        "CustomerCount": 11,
        "LTV": None,
    },
    {
        "AcquisitionChannel": "Affiliate",
        "AvgProfitMargin": 0.8025000000000001,
        "AvgRevenuePerUser": 12060.0,
        "CAC": 13834.369999999999,
        "CACToLTVRatio": None,
        "ChurnRate": None,
        "CustomerCount": 14,
        "LTV": None,
    },
    {
        "AcquisitionChannel": "Email",
        "AvgProfitMargin": 0.85,
        "AvgRevenuePerUser": 7200.0,
        "CAC": 7040.9,
        "CACToLTVRatio": 52.152422918702676,
        "ChurnRate": 0.20000000298023224,
        "CustomerCount": 12,
        "LTV": 30599.99954402447,
    },
    {
        "AcquisitionChannel": "Paid Search",
        "AvgProfitMargin": 0.85,
        "AvgRevenuePerUser": 21600.0,
        "CAC": 72146.89,
        "CACToLTVRatio": 17.81365739880164,
        "ChurnRate": 0.20000000298023224,
        "CustomerCount": 14,
        "LTV": 91799.99863207343,
    },
    {
        "AcquisitionChannel": "Content",
        "AvgProfitMargin": 0.8128571428571425,
        "AvgRevenuePerUser": 8400.0,
        "CAC": 21468.16,
        "CACToLTVRatio": None,
        "ChurnRate": None,
        "CustomerCount": 4,
        "LTV": None,
    },
    {
        "AcquisitionChannel": "Paid Search",
        "AvgProfitMargin": 0.8200000000000001,
        "AvgRevenuePerUser": 810.0,
        "CAC": 72146.89,
        "CACToLTVRatio": None,
        "ChurnRate": None,
        "CustomerCount": 14,
        "LTV": None,
    },
    {
        "AcquisitionChannel": "Social Media",
        "AvgProfitMargin": 0.8199999999999996,
        "AvgRevenuePerUser": 4117.5,
        "CAC": 41269.94,
        "CACToLTVRatio": None,
        "ChurnRate": None,
        "CustomerCount": 11,
        "LTV": None,
    },
    {
        "AcquisitionChannel": "Affiliate",
        "AvgProfitMargin": 0.8,
        "AvgRevenuePerUser": 3600.0,
        "CAC": 13834.369999999999,
        "CACToLTVRatio": 14.572401706469176,
        "ChurnRate": 0.20000000298023224,
        "CustomerCount": 14,
        "LTV": 14399.999785423282,
    },
    {
        "AcquisitionChannel": "Email",
        "AvgProfitMargin": 0.8,
        "AvgRevenuePerUser": 600.0,
        "CAC": 7040.9,
        "CACToLTVRatio": None,
        "ChurnRate": None,
        "CustomerCount": 12,
        "LTV": None,
    },
    {
        "AcquisitionChannel": "Email",
        "AvgProfitMargin": 0.8349083785412899,
        "AvgRevenuePerUser": 13560.0,
        "CAC": 7040.9,
        "CACToLTVRatio": 19.295301929616766,
        "ChurnRate": 1.0,
        "CustomerCount": 12,
        "LTV": 11321.35761301989,
    },
    {
        "AcquisitionChannel": "Content",
        "AvgProfitMargin": 0.8,
        "AvgRevenuePerUser": 750.0,
        "CAC": 21468.16,
        "CACToLTVRatio": 0.11179346529930838,
        "ChurnRate": 1.0,
        "CustomerCount": 4,
        "LTV": 600.0,
    },
    {
        "AcquisitionChannel": "Content",
        "AvgProfitMargin": 0.8128571428571431,
        "AvgRevenuePerUser": 2520.0,
        "CAC": 21468.16,
        "CACToLTVRatio": None,
        "ChurnRate": None,
        "CustomerCount": 4,
        "LTV": None,
    },
    {
        "AcquisitionChannel": "Affiliate",
        "AvgProfitMargin": 0.85,
        "AvgRevenuePerUser": 7200.0,
        "CAC": 13834.369999999999,
        "CACToLTVRatio": None,
        "ChurnRate": None,
        "CustomerCount": 14,
        "LTV": None,
    },
    {
        "AcquisitionChannel": "Paid Search",
        "AvgProfitMargin": 0.8,
        "AvgRevenuePerUser": 600.0,
        "CAC": 72146.89,
        "CACToLTVRatio": None,
        "ChurnRate": None,
        "CustomerCount": 14,
        "LTV": None,
    },
    {
        "AcquisitionChannel": "Email",
        "AvgProfitMargin": 0.83375,
        "AvgRevenuePerUser": 12960.0,
        "CAC": 7040.9,
        "CACToLTVRatio": 36.831882287775706,
        "ChurnRate": 0.5,
        "CustomerCount": 12,
        "LTV": 21610.8,
    },
    {
        "AcquisitionChannel": "Social Media",
        "AvgProfitMargin": 0.81,
        "AvgRevenuePerUser": 3480.0,
        "CAC": 41269.94,
        "CACToLTVRatio": 3.756584034040597,
        "ChurnRate": 0.20000000298023224,
        "CustomerCount": 11,
        "LTV": 14093.999789983038,
    },
    {
        "AcquisitionChannel": "Paid Search",
        "AvgProfitMargin": 0.8069144518272424,
        "AvgRevenuePerUser": 6112.5,
        "CAC": 72146.89,
        "CACToLTVRatio": 1.4356482065383938,
        "ChurnRate": 0.6666666865348816,
        "CustomerCount": 14,
        "LTV": 7398.396659701626,
    },
    {
        "AcquisitionChannel": "Social Media",
        "AvgProfitMargin": 0.8153527352869054,
        "AvgRevenuePerUser": 12360.0,
        "CAC": 41269.94,
        "CACToLTVRatio": 5.372208338059501,
        "ChurnRate": 0.5,
        "CustomerCount": 11,
        "LTV": 20155.519616292302,
    },
    {
        "AcquisitionChannel": "Paid Search",
        "AvgProfitMargin": 0.8428804347826094,
        "AvgRevenuePerUser": 13800.0,
        "CAC": 72146.89,
        "CACToLTVRatio": None,
        "ChurnRate": None,
        "CustomerCount": 14,
        "LTV": None,
    },
    {
        "AcquisitionChannel": "Affiliate",
        "AvgProfitMargin": 0.8199999999999994,
        "AvgRevenuePerUser": 6480.0,
        "CAC": 13834.369999999999,
        "CACToLTVRatio": None,
        "ChurnRate": None,
        "CustomerCount": 14,
        "LTV": None,
    },
    {
        "AcquisitionChannel": "Social Media",
        "AvgProfitMargin": 0.8219047619047621,
        "AvgRevenuePerUser": 17610.0,
        "CAC": 41269.94,
        "CACToLTVRatio": 19.28899933658011,
        "ChurnRate": 0.20000000298023224,
        "CustomerCount": 11,
        "LTV": 72368.71320733645,
    },
    {
        "AcquisitionChannel": "Email",
        "AvgProfitMargin": 0.85,
        "AvgRevenuePerUser": 14400.0,
        "CAC": 7040.9,
        "CACToLTVRatio": 104.30484583740535,
        "ChurnRate": 0.20000000298023224,
        "CustomerCount": 12,
        "LTV": 61199.99908804894,
    },
    {
        "AcquisitionChannel": "Email",
        "AvgProfitMargin": 0.8232692307692295,
        "AvgRevenuePerUser": 15881.25,
        "CAC": 7040.9,
        "CACToLTVRatio": 44.56661326076095,
        "ChurnRate": 0.5,
        "CustomerCount": 12,
        "LTV": 26149.08894230765,
    },
    {
        "AcquisitionChannel": "Social Media",
        "AvgProfitMargin": 0.82,
        "AvgRevenuePerUser": 5760.0,
        "CAC": 41269.94,
        "CACToLTVRatio": 6.294557155378369,
        "ChurnRate": 0.20000000298023224,
        "CustomerCount": 11,
        "LTV": 23615.99964809418,
    },
    {
        "AcquisitionChannel": "Social Media",
        "AvgProfitMargin": 0.811782945736435,
        "AvgRevenuePerUser": 12190.0,
        "CAC": 41269.94,
        "CACToLTVRatio": 5.275121562754807,
        "ChurnRate": 0.5,
        "CustomerCount": 11,
        "LTV": 19791.268217054283,
    },
    {
        "AcquisitionChannel": "Email",
        "AvgProfitMargin": 0.8000000000000002,
        "AvgRevenuePerUser": 9650.0,
        "CAC": 7040.9,
        "CACToLTVRatio": 65.78704328960535,
        "ChurnRate": 0.20000000298023224,
        "CustomerCount": 12,
        "LTV": 38599.99942481519,
    },
    {
        "AcquisitionChannel": "Paid Search",
        "AvgProfitMargin": 0.8,
        "AvgRevenuePerUser": 1200.0,
        "CAC": 72146.89,
        "CACToLTVRatio": None,
        "ChurnRate": None,
        "CustomerCount": 14,
        "LTV": None,
    },
    {
        "AcquisitionChannel": "Paid Search",
        "AvgProfitMargin": 0.8,
        "AvgRevenuePerUser": 1275.0,
        "CAC": 72146.89,
        "CACToLTVRatio": 0.9896476332667579,
        "ChurnRate": 0.20000000298023224,
        "CustomerCount": 14,
        "LTV": 5099.99992400408,
    },
    {
        "AcquisitionChannel": "Email",
        "AvgProfitMargin": 0.8244579345889972,
        "AvgRevenuePerUser": 13135.0,
        "CAC": 7040.9,
        "CACToLTVRatio": 36.913195656781866,
        "ChurnRate": 0.5,
        "CustomerCount": 12,
        "LTV": 21658.509941652956,
    },
    {
        "AcquisitionChannel": "Content",
        "AvgProfitMargin": 0.8,
        "AvgRevenuePerUser": 1200.0,
        "CAC": 21468.16,
        "CACToLTVRatio": None,
        "ChurnRate": None,
        "CustomerCount": 4,
        "LTV": None,
    },
    {
        "AcquisitionChannel": "Email",
        "AvgProfitMargin": 0.8,
        "AvgRevenuePerUser": 6000.0,
        "CAC": 7040.9,
        "CACToLTVRatio": 40.903861112707986,
        "ChurnRate": 0.20000000298023224,
        "CustomerCount": 12,
        "LTV": 23999.99964237214,
    },
    {
        "AcquisitionChannel": "Paid Search",
        "AvgProfitMargin": 0.8219491525423726,
        "AvgRevenuePerUser": 5842.5,
        "CAC": 72146.89,
        "CACToLTVRatio": 0.9318673463569027,
        "ChurnRate": 1.0,
        "CustomerCount": 14,
        "LTV": 4802.237923728811,
    },
    {
        "AcquisitionChannel": "Content",
        "AvgProfitMargin": 0.8200000000000002,
        "AvgRevenuePerUser": 1620.0,
        "CAC": 21468.16,
        "CACToLTVRatio": 0.2475107321726688,
        "ChurnRate": 1.0,
        "CustomerCount": 4,
        "LTV": 1328.4000000000003,
    },
    {
        "AcquisitionChannel": "Paid Search",
        "AvgProfitMargin": 0.8000000000000002,
        "AvgRevenuePerUser": 8000.0,
        "CAC": 72146.89,
        "CACToLTVRatio": 2.4838215479558445,
        "ChurnRate": 0.5,
        "CustomerCount": 14,
        "LTV": 12800.000000000002,
    },
    {
        "AcquisitionChannel": "Affiliate",
        "AvgProfitMargin": 0.8061225637181422,
        "AvgRevenuePerUser": 18982.5,
        "CAC": 13834.369999999999,
        "CACToLTVRatio": 77.42712488122797,
        "ChurnRate": 0.20000000298023224,
        "CustomerCount": 14,
        "LTV": 76511.10668879384,
    },
    {
        "AcquisitionChannel": "Affiliate",
        "AvgProfitMargin": 0.8098245614035089,
        "AvgRevenuePerUser": 3175.0,
        "CAC": 13834.369999999999,
        "CACToLTVRatio": 2.6019762196895106,
        "ChurnRate": 1.0,
        "CustomerCount": 14,
        "LTV": 2571.1929824561407,
    },
    {
        "AcquisitionChannel": "Affiliate",
        "AvgProfitMargin": 0.8096644295302029,
        "AvgRevenuePerUser": 17745.0,
        "CAC": 13834.369999999999,
        "CACToLTVRatio": None,
        "ChurnRate": None,
        "CustomerCount": 14,
        "LTV": None,
    },
    {
        "AcquisitionChannel": "Affiliate",
        "AvgProfitMargin": 0.8,
        "AvgRevenuePerUser": 525.0,
        "CAC": 13834.369999999999,
        "CACToLTVRatio": None,
        "ChurnRate": None,
        "CustomerCount": 14,
        "LTV": None,
    },
    {
        "AcquisitionChannel": "Social Media",
        "AvgProfitMargin": 0.8049521798534351,
        "AvgRevenuePerUser": 17655.0,
        "CAC": 41269.94,
        "CACToLTVRatio": 7.575767645333934,
        "ChurnRate": 0.5,
        "CustomerCount": 11,
        "LTV": 28422.861470624794,
    },
    {
        "AcquisitionChannel": "Affiliate",
        "AvgProfitMargin": 0.8186225637181419,
        "AvgRevenuePerUser": 12120.0,
        "CAC": 13834.369999999999,
        "CACToLTVRatio": 30.121474083468133,
        "ChurnRate": 0.3333333432674408,
        "CustomerCount": 14,
        "LTV": 29765.11552972207,
    },
    {
        "AcquisitionChannel": "Affiliate",
        "AvgProfitMargin": 0.8023684210526316,
        "AvgRevenuePerUser": 6296.25,
        "CAC": 13834.369999999999,
        "CACToLTVRatio": 5.112395461068112,
        "ChurnRate": 1.0,
        "CustomerCount": 14,
        "LTV": 5051.9121710526315,
    },
    {
        "AcquisitionChannel": "Affiliate",
        "AvgProfitMargin": 0.8075000000000001,
        "AvgRevenuePerUser": 4320.0,
        "CAC": 13834.369999999999,
        "CACToLTVRatio": None,
        "ChurnRate": None,
        "CustomerCount": 14,
        "LTV": None,
    },
    {
        "AcquisitionChannel": "Email",
        "AvgProfitMargin": 0.8232692307692311,
        "AvgRevenuePerUser": 8793.75,
        "CAC": 7040.9,
        "CACToLTVRatio": 24.677380896454462,
        "ChurnRate": 0.5,
        "CustomerCount": 12,
        "LTV": 14479.247596153851,
    },
    {
        "AcquisitionChannel": "Social Media",
        "AvgProfitMargin": 0.8076744186046518,
        "AvgRevenuePerUser": 17955.0,
        "CAC": 41269.94,
        "CACToLTVRatio": 19.326383036840188,
        "ChurnRate": 0.20000000298023224,
        "CustomerCount": 11,
        "LTV": 72508.96984976476,
    },
    {
        "AcquisitionChannel": "Social Media",
        "AvgProfitMargin": 0.8,
        "AvgRevenuePerUser": 2400.0,
        "CAC": 41269.94,
        "CACToLTVRatio": 2.55876307129202,
        "ChurnRate": 0.20000000298023224,
        "CustomerCount": 11,
        "LTV": 9599.999856948854,
    },
    {
        "AcquisitionChannel": "Paid Search",
        "AvgProfitMargin": 0.82,
        "AvgRevenuePerUser": 2880.0,
        "CAC": 72146.89,
        "CACToLTVRatio": 2.2913253438458576,
        "ChurnRate": 0.20000000298023224,
        "CustomerCount": 14,
        "LTV": 11807.99982404709,
    },
    {
        "AcquisitionChannel": "Paid Search",
        "AvgProfitMargin": 0.8,
        "AvgRevenuePerUser": 2400.0,
        "CAC": 72146.89,
        "CACToLTVRatio": 1.8628661332080145,
        "ChurnRate": 0.20000000298023224,
        "CustomerCount": 14,
        "LTV": 9599.999856948854,
    },
    {
        "AcquisitionChannel": "Affiliate",
        "AvgProfitMargin": 0.825,
        "AvgRevenuePerUser": 3120.0,
        "CAC": 13834.369999999999,
        "CACToLTVRatio": 2.60481684384616,
        "ChurnRate": 1.0,
        "CustomerCount": 14,
        "LTV": 2574.0,
    },
    {
        "AcquisitionChannel": "Paid Search",
        "AvgProfitMargin": 0.8119144518272432,
        "AvgRevenuePerUser": 7466.25,
        "CAC": 72146.89,
        "CACToLTVRatio": 2.352627753278684,
        "ChurnRate": 0.5,
        "CustomerCount": 14,
        "LTV": 12123.912551910309,
    },
    {
        "AcquisitionChannel": "Affiliate",
        "AvgProfitMargin": 0.85,
        "AvgRevenuePerUser": 21600.0,
        "CAC": 13834.369999999999,
        "CACToLTVRatio": 92.89906087874101,
        "ChurnRate": 0.20000000298023224,
        "CustomerCount": 14,
        "LTV": 91799.99863207343,
    },
    {
        "AcquisitionChannel": "Social Media",
        "AvgProfitMargin": 0.8145911527884948,
        "AvgRevenuePerUser": 17397.0,
        "CAC": 41269.94,
        "CACToLTVRatio": 18.886126659354908,
        "ChurnRate": 0.20000000298023224,
        "CustomerCount": 11,
        "LTV": 70857.2103694525,
    },
    {
        "AcquisitionChannel": "Paid Search",
        "AvgProfitMargin": 0.8099999999999998,
        "AvgRevenuePerUser": 9435.0,
        "CAC": 72146.89,
        "CACToLTVRatio": 7.41493489225118,
        "ChurnRate": 0.20000000298023224,
        "CustomerCount": 14,
        "LTV": 38211.749430600554,
    },
    {
        "AcquisitionChannel": "Affiliate",
        "AvgProfitMargin": 0.82,
        "AvgRevenuePerUser": 1440.0,
        "CAC": 13834.369999999999,
        "CACToLTVRatio": None,
        "ChurnRate": None,
        "CustomerCount": 14,
        "LTV": None,
    },
    {
        "AcquisitionChannel": "Paid Search",
        "AvgProfitMargin": 0.8140672161336688,
        "AvgRevenuePerUser": 9318.75,
        "CAC": 72146.89,
        "CACToLTVRatio": 7.360347937538079,
        "ChurnRate": 0.20000000298023224,
        "CustomerCount": 14,
        "LTV": 37930.443786520475,
    },
    {
        "AcquisitionChannel": "Email",
        "AvgProfitMargin": 0.8300000000000013,
        "AvgRevenuePerUser": 14175.0,
        "CAC": 7040.9,
        "CACToLTVRatio": 100.25919832422673,
        "ChurnRate": 0.20000000298023224,
        "CustomerCount": 12,
        "LTV": 58826.24912342067,
    },
    {
        "AcquisitionChannel": "Paid Search",
        "AvgProfitMargin": 0.8,
        "AvgRevenuePerUser": 20400.0,
        "CAC": 72146.89,
        "CACToLTVRatio": 15.834362132268126,
        "ChurnRate": 0.20000000298023224,
        "CustomerCount": 14,
        "LTV": 81599.99878406528,
    },
    {
        "AcquisitionChannel": "Paid Search",
        "AvgProfitMargin": 0.8,
        "AvgRevenuePerUser": 8925.0,
        "CAC": 72146.89,
        "CACToLTVRatio": 6.927533432867304,
        "ChurnRate": 0.20000000298023224,
        "CustomerCount": 14,
        "LTV": 35699.99946802855,
    },
    {
        "AcquisitionChannel": "Email",
        "AvgProfitMargin": 0.8,
        "AvgRevenuePerUser": 2400.0,
        "CAC": 7040.9,
        "CACToLTVRatio": 16.361544445083194,
        "ChurnRate": 0.20000000298023224,
        "CustomerCount": 12,
        "LTV": 9599.999856948854,
    },
    {
        "AcquisitionChannel": "Content",
        "AvgProfitMargin": 0.8,
        "AvgRevenuePerUser": 1800.0,
        "CAC": 21468.16,
        "CACToLTVRatio": 0.5366086334366802,
        "ChurnRate": 0.5,
        "CustomerCount": 4,
        "LTV": 2880.0,
    },
    {
        "AcquisitionChannel": "Paid Search",
        "AvgProfitMargin": 0.8219491525423724,
        "AvgRevenuePerUser": 4762.5,
        "CAC": 72146.89,
        "CACToLTVRatio": 0.759609454347411,
        "ChurnRate": 1.0,
        "CustomerCount": 14,
        "LTV": 3914.532838983049,
    },
    {
        "AcquisitionChannel": "Paid Search",
        "AvgProfitMargin": 0.8238983050847454,
        "AvgRevenuePerUser": 7080.0,
        "CAC": 72146.89,
        "CACToLTVRatio": 1.1319240510575017,
        "ChurnRate": 1.0,
        "CustomerCount": 14,
        "LTV": 5833.199999999997,
    },
    {
        "AcquisitionChannel": "Social Media",
        "AvgProfitMargin": 0.8145911527884936,
        "AvgRevenuePerUser": 13176.0,
        "CAC": 41269.94,
        "CACToLTVRatio": 14.303822777700749,
        "ChurnRate": 0.20000000298023224,
        "CustomerCount": 11,
        "LTV": 53665.2643460312,
    },
    {
        "AcquisitionChannel": "Affiliate",
        "AvgProfitMargin": 0.80375,
        "AvgRevenuePerUser": 7500.0,
        "CAC": 13834.369999999999,
        "CACToLTVRatio": None,
        "ChurnRate": None,
        "CustomerCount": 14,
        "LTV": None,
    },
    {
        "AcquisitionChannel": "Social Media",
        "AvgProfitMargin": 0.8,
        "AvgRevenuePerUser": 8400.0,
        "CAC": 41269.94,
        "CACToLTVRatio": None,
        "ChurnRate": None,
        "CustomerCount": 11,
        "LTV": None,
    },
    {
        "AcquisitionChannel": "Paid Search",
        "AvgProfitMargin": 0.8187562881782258,
        "AvgRevenuePerUser": 24565.0,
        "CAC": 72146.89,
        "CACToLTVRatio": 19.514248699530047,
        "ChurnRate": 0.20000000298023224,
        "CustomerCount": 14,
        "LTV": 100563.73959697409,
    },
    {
        "AcquisitionChannel": "Social Media",
        "AvgProfitMargin": 0.8200000000000002,
        "AvgRevenuePerUser": 1215.0,
        "CAC": 41269.94,
        "CACToLTVRatio": None,
        "ChurnRate": None,
        "CustomerCount": 11,
        "LTV": None,
    },
    {
        "AcquisitionChannel": "Email",
        "AvgProfitMargin": 0.8200413322632425,
        "AvgRevenuePerUser": 6858.75,
        "CAC": 7040.9,
        "CACToLTVRatio": 19.17183935347077,
        "ChurnRate": 0.5,
        "CustomerCount": 12,
        "LTV": 11248.916975321028,
    },
    {
        "AcquisitionChannel": "Email",
        "AvgProfitMargin": 0.8,
        "AvgRevenuePerUser": 3600.0,
        "CAC": 7040.9,
        "CACToLTVRatio": 24.54231666762479,
        "ChurnRate": 0.20000000298023224,
        "CustomerCount": 12,
        "LTV": 14399.999785423282,
    },
    {
        "AcquisitionChannel": "Affiliate",
        "AvgProfitMargin": 0.8064429530201344,
        "AvgRevenuePerUser": 1820.0,
        "CAC": 13834.369999999999,
        "CACToLTVRatio": None,
        "ChurnRate": None,
        "CustomerCount": 14,
        "LTV": None,
    },
    {
        "AcquisitionChannel": "Social Media",
        "AvgProfitMargin": 0.8208604282315621,
        "AvgRevenuePerUser": 5992.5,
        "CAC": 41269.94,
        "CACToLTVRatio": 6.5555058320494375,
        "ChurnRate": 0.20000000298023224,
        "CustomerCount": 11,
        "LTV": 24595.03021439367,
    },
    {
        "AcquisitionChannel": "Content",
        "AvgProfitMargin": 0.8200000000000003,
        "AvgRevenuePerUser": 2160.0,
        "CAC": 21468.16,
        "CACToLTVRatio": 0.33001430956355843,
        "ChurnRate": 1.0,
        "CustomerCount": 4,
        "LTV": 1771.2000000000007,
    },
]
