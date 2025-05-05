import json
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from loguru import logger
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext


@dataclass
class GraphDeps:
    data_dir: str
    analysis_dir: str
    results_dir: str


def add_current_time() -> str:
    return f"<current_time>\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n</current_time>"


def preview_csv(df: Path | str | pd.DataFrame, num_rows: int = 3) -> list[dict[str, str]]:
    """
    Returns the first `num_rows` rows of the specified CSV file.
    Should be used for previewing the data.
    """
    if isinstance(df, pd.DataFrame):
        return df.head(num_rows).to_dict(orient="records")
    return pd.read_csv(df).head(num_rows).to_dict(orient="records")


def list_csv_files(ctx: RunContext[GraphDeps]) -> str:
    """
    Lists all CSV files in the specified directory.
    """
    csv_files = [str(file) for file in Path(ctx.deps.data_dir).glob("*.csv")]
    res = "\nAvailable CSV files:\n"
    for file in csv_files:
        res += file + "\n"
        res += "First row for preview:\n"
        res += json.dumps(preview_csv(df=file, num_rows=1)) + "\n\n"
    return res.strip()


def temp_file_path(file_dir: Path | str | None = None) -> str:
    file_dir = file_dir or tempfile.gettempdir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path(file_dir) / f"ai_cfo_result_{Path(file_dir).name}_{ts}.csv")


class RunSQLResult(BaseModel):
    """
    A lightweight reference to a (possibly large) query result on disk.
    """

    file_path: str
    preview: list[dict[str, Any]]
    row_count: int


def run_sql(
    ctx: RunContext[GraphDeps], query: str, preview_rows: int = 3, file_path: str | None = None
) -> RunSQLResult:
    """
    1. Runs an SQL query on csv file(s) using duckdb
    2. Writes the full result to disk
    3. Returns a ResultHandle with a small in-memory preview.

    In the query, include the full path to the csv file in single quotes. Include the extension.

    Example Query:
    SELECT * FROM 'full_path_to_csv_file.csv'

    Parameters
    ----------
    query : str
        The SQL query to execute.
    preview_rows : int, default 3
        Number of rows to preview.
    file_path : str | None, default None
        Path to save the result to in the `analysis_dir`.
        If None, a path will be created in the `analysis_dir` using the current timestamp.

    Returns
    -------
    RunSQLResult
        A lightweight reference to the result on disk.
    """
    try:
        query = query.replace('"', "'").strip()
        logger.info(f"Running SQL query: {query}")

        df = duckdb.sql(query).df()

        file_path = file_path or temp_file_path(file_dir=ctx.deps.analysis_dir)
        path = Path(file_path).expanduser().resolve().with_suffix(".csv")
        path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(path, index=False)

        return RunSQLResult(
            file_path=str(path), preview=preview_csv(df=df, num_rows=preview_rows), row_count=len(df)
        )
    except Exception as e:
        raise ModelRetry(str(e))


def write_sheet_from_file(
    ctx: RunContext[GraphDeps], file_path: str, sheet_name: str, workbook_path: str | None = None
) -> str:
    """
    1. Reads from the `file_path` of the previously received RunSQLResult
    2. Appends to / creates `workbook_path`.

    Parameters
    ----------
    file_path : str
        Path to the file to read from. Almost always a `file_path` of a previously received RunSQLResult.
    sheet_name : str
        The name of the sheet to write to.
    workbook_path : str | None, default None
        Path to the workbook to append to / create in `results_dir`.
        If None, a path will be created in the `results_dir` using the current timestamp.

    Returns
    -------
    str
        The path to the workbook.
    """

    workbook_path = workbook_path or temp_file_path(file_dir=ctx.deps.results_dir)
    wb_path = Path(workbook_path).expanduser().resolve().with_suffix(".xlsx")

    df = pd.read_csv(file_path)

    mode = "w"
    if_sheet_exists = None
    if wb_path.exists():
        mode = "a"
        if_sheet_exists = "replace"
    with pd.ExcelWriter(wb_path, mode=mode, engine="openpyxl", if_sheet_exists=if_sheet_exists) as w:
        df.to_excel(w, sheet_name=sheet_name[:31] or "Sheet1", index=False)

    return str(wb_path)


agent = Agent(
    model="google-gla:gemini-2.0-flash",
    instructions=(add_current_time, list_csv_files),
    deps_type=GraphDeps,
    retries=5,
    tools=[run_sql, write_sheet_from_file],
)

deps = GraphDeps(
    data_dir="./operateai_scenario1_data",
    analysis_dir="./operateai_scenario1_analysis",
    results_dir="./operateai_scenario1_results",
)

query = "how many customers acquired in 2023? all channels merged."
async with agent.iter(user_prompt=query, deps=deps) as run:
    async for node in run:
        print("\n\n", node, "\n\n")
