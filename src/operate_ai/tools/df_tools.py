from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd
from pydantic_ai import RunContext

DIR = Path("../../../operateai_scenario1_data")


@dataclass
class Deps:
    data_dir: Path | str


def list_csv_files(ctx: RunContext[Deps]) -> str:
    """
    Lists all CSV files in the specified directory.
    """
    return f"Available CSV files: {[str(file) for file in Path(ctx.deps.data_dir).glob('*.csv')]}"


def preview_df(df_path: str, num_rows: int = 3) -> list[dict[str, str]]:
    """
    Returns the first num_rows rows of the specified DataFrame as a list of dictionaries.
    The dictionaries have the column names as keys and the values in the respective row as values.
    Should be used for previewing the data in the DataFrame.
    """
    return pd.read_csv(df_path).head(num_rows).to_dict(orient="records")


def run_sql(query: str) -> list[dict[str, str]]:
    """
    Runs a SQL query and returns the results as a list of dictionaries.
    In the query, include the path to the csv file in single quotes.
    """
    return duckdb.sql(query).to_df().to_dict(orient="records")
