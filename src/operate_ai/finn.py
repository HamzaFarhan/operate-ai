from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb
import logfire
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_core import to_json

load_dotenv()

logfire.configure()

MAX_ANALYSIS_FILE_TOKENS = 20_000
SQL_TIMEOUT_SECONDS = 5
MAX_RETRIES = 10
MEMORY_FILE_NAME = os.getenv("MEMORY_FILE_NAME", "memory.json")
MODULE_DIR = Path(__file__).parent


def csv_summary(
    file_path: Path, top_n_categorical: int = 50, high_cardinality_threshold: int = 100
) -> dict[str, Any]:
    """
    Generates comprehensive metadata for a CSV file using DuckDB for performance.

    Args:
        file_path: Path to CSV file to analyze
        top_n_categorical: Number of top values to show for categorical columns
        high_cardinality_threshold: Threshold above which to skip detailed categorical analysis
    Returns:
        dict: Comprehensive metadata including statistics, data quality, and analysis
    """

    file_path = file_path.expanduser().resolve()
    file_name = file_path.name
    try:
        # Get basic file info efficiently with DuckDB
        row_count = duckdb.sql(f"SELECT COUNT(*) FROM read_csv('{file_path}')").fetchone()[0]  # type: ignore
        if row_count == 0:
            return {
                "file_level_summary": {
                    "file_name": file_name,
                    "shape": (0, 0),
                    "total_rows": 0,
                    "total_columns": 0,
                    "status": "empty_file",
                },
                "column_summaries": [],
            }

        # Get column info
        columns_info = duckdb.sql(f"DESCRIBE (SELECT * FROM read_csv('{file_path}') LIMIT 1)").fetchdf()  # type: ignore
        column_names = columns_info["column_name"].tolist()  # type: ignore
        column_types = dict(zip(columns_info["column_name"], columns_info["column_type"]))  # type: ignore

    except Exception as e:
        logger.warning(f"DuckDB analysis failed for {file_path}, falling back to pandas: {e}")
        return _pandas_csv_summary_fallback(pd.read_csv(file_path), file_name)  # type: ignore

    # File-level summary
    summary: dict[str, Any] = {
        "file_level_summary": {
            "file_name": file_name,
            "shape": (row_count, len(column_names)),
            "total_rows": row_count,
            "total_columns": len(column_names),
            "analysis_timestamp": datetime.now().isoformat(),
            "status": "analyzed",
        }
    }

    # Analyze each column
    column_summaries: list[dict[str, Any]] = []
    numeric_columns: list[str] = []

    for col_name in column_names:
        col_summary: dict[str, Any] = {
            "column_name": col_name,
            "original_dtype": column_types.get(col_name, "unknown"),
        }

        try:
            # Get comprehensive column statistics in one efficient query
            stats_query = f"""
            SELECT 
                COUNT(*) as total_count,
                COUNT("{col_name}") as non_null_count,
                COUNT(DISTINCT "{col_name}") as unique_count,
                -- Numeric analysis
                MIN(TRY_CAST("{col_name}" AS DOUBLE)) as min_numeric,
                MAX(TRY_CAST("{col_name}" AS DOUBLE)) as max_numeric,
                AVG(TRY_CAST("{col_name}" AS DOUBLE)) as avg_numeric,
                STDDEV(TRY_CAST("{col_name}" AS DOUBLE)) as stddev_numeric,
                VARIANCE(TRY_CAST("{col_name}" AS DOUBLE)) as variance_numeric,
                percentile_cont(0.05) WITHIN GROUP (ORDER BY TRY_CAST("{col_name}" AS DOUBLE)) as p5,
                percentile_cont(0.25) WITHIN GROUP (ORDER BY TRY_CAST("{col_name}" AS DOUBLE)) as q1,
                percentile_cont(0.5) WITHIN GROUP (ORDER BY TRY_CAST("{col_name}" AS DOUBLE)) as median,
                percentile_cont(0.75) WITHIN GROUP (ORDER BY TRY_CAST("{col_name}" AS DOUBLE)) as q3,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY TRY_CAST("{col_name}" AS DOUBLE)) as p95,
                -- Date analysis
                MIN(TRY_CAST("{col_name}" AS DATE)) as min_date,
                MAX(TRY_CAST("{col_name}" AS DATE)) as max_date,
                -- String analysis
                MIN(LENGTH(CAST("{col_name}" AS STRING))) as min_length,
                MAX(LENGTH(CAST("{col_name}" AS STRING))) as max_length,
                AVG(LENGTH(CAST("{col_name}" AS STRING))) as avg_length
            FROM read_csv('{file_path}')
            """

            stats_result = duckdb.sql(stats_query).fetchone()  # type: ignore
            if stats_result is None:
                col_summary["error"] = "Failed to get column statistics"
                continue

            # Extract basic stats
            total_count = int(stats_result[0])  # type: ignore
            non_null_count = int(stats_result[1])  # type: ignore
            unique_count = int(stats_result[2])  # type: ignore
            missing_count = total_count - non_null_count

            # Data quality metrics
            col_summary["data_quality"] = {
                "total_count": total_count,
                "non_null_count": non_null_count,
                "missing_count": missing_count,
                "missing_percentage": round((missing_count / total_count) * 100, 2) if total_count > 0 else 0,
                "completeness_percentage": round((non_null_count / total_count) * 100, 2)
                if total_count > 0
                else 0,
                "unique_count": unique_count,
                "uniqueness_percentage": round((unique_count / non_null_count) * 100, 2)
                if non_null_count > 0
                else 0,
                "cardinality_ratio": round(unique_count / non_null_count, 4) if non_null_count > 0 else 0,
            }

            # Determine data type and add appropriate statistics
            min_numeric, max_numeric = stats_result[3], stats_result[4]  # type: ignore
            avg_numeric, stddev_numeric, variance_numeric = stats_result[5], stats_result[6], stats_result[7]  # type: ignore
            p5, q1, median, q3, p95 = stats_result[8:13]  # type: ignore
            min_date, max_date = stats_result[13], stats_result[14]  # type: ignore
            min_length, max_length, avg_length = stats_result[15], stats_result[16], stats_result[17]  # type: ignore

            # Numeric column analysis
            if min_numeric is not None and max_numeric is not None and non_null_count > 0:
                col_summary["inferred_best_dtype"] = "numeric"
                col_summary["numeric_stats"] = {
                    "min": round(float(min_numeric), 3),  # type: ignore
                    "max": round(float(max_numeric), 3),  # type: ignore
                    "mean": round(float(avg_numeric), 3) if avg_numeric else None,  # type: ignore
                    "median": round(float(median), 3) if median else None,  # type: ignore
                    "std_dev": round(float(stddev_numeric), 3) if stddev_numeric else None,  # type: ignore
                    "variance": round(float(variance_numeric), 3) if variance_numeric else None,  # type: ignore
                    "range": round(float(max_numeric) - float(min_numeric), 3),  # type: ignore
                    "percentiles": {
                        "p5": round(float(p5), 3) if p5 else None,  # type: ignore
                        "q1": round(float(q1), 3) if q1 else None,  # type: ignore
                        "q3": round(float(q3), 3) if q3 else None,  # type: ignore
                        "p95": round(float(p95), 3) if p95 else None,  # type: ignore
                        "iqr": round(float(q3) - float(q1), 3) if q1 and q3 else None,  # type: ignore
                    },
                    "coefficient_of_variation": round(float(stddev_numeric) / float(avg_numeric), 3)  # type: ignore
                    if stddev_numeric and avg_numeric and avg_numeric != 0
                    else None,
                }
                numeric_columns.append(col_name)

                # Add distribution analysis for larger datasets
                if non_null_count > 100:
                    try:
                        hist_query = f"""
                        SELECT 
                            WIDTH_BUCKET(TRY_CAST("{col_name}" AS DOUBLE), {min_numeric}, {max_numeric}, 10) as bucket,
                            COUNT(*) as frequency
                        FROM read_csv('{file_path}')
                        WHERE TRY_CAST("{col_name}" AS DOUBLE) IS NOT NULL
                        GROUP BY bucket
                        ORDER BY bucket
                        """
                        hist_result = duckdb.sql(hist_query).fetchdf()  # type: ignore
                        if not hist_result.empty:  # type: ignore
                            col_summary["numeric_stats"]["distribution"] = hist_result.to_dict("records")  # type: ignore
                    except Exception:
                        pass

            # Date column analysis
            elif min_date is not None and max_date is not None:
                col_summary["inferred_best_dtype"] = "datetime"
                col_summary["datetime_stats"] = {
                    "min_date": str(min_date),  # type: ignore
                    "max_date": str(max_date),  # type: ignore
                    "date_range_days": (max_date - min_date).days  # type: ignore
                    if hasattr(max_date - min_date, "days")  # type: ignore
                    else None,
                }

                # Time series analysis
                if non_null_count > 10:
                    try:
                        date_dist_query = f"""
                        SELECT 
                            EXTRACT(YEAR FROM TRY_CAST("{col_name}" AS DATE)) as year,
                            COUNT(*) as count
                        FROM read_csv('{file_path}')
                        WHERE TRY_CAST("{col_name}" AS DATE) IS NOT NULL
                        GROUP BY year
                        ORDER BY year
                        """
                        date_dist = duckdb.sql(date_dist_query).fetchdf()  # type: ignore
                        if not date_dist.empty:  # type: ignore
                            col_summary["datetime_stats"]["yearly_distribution"] = date_dist.to_dict("records")  # type: ignore
                    except Exception:
                        pass

            # Categorical/string column analysis
            else:
                col_summary["inferred_best_dtype"] = "categorical"

                # String length statistics
                if min_length is not None:
                    col_summary["string_stats"] = {
                        "min_length": int(min_length),  # type: ignore
                        "max_length": int(max_length),  # type: ignore
                        "avg_length": round(float(avg_length), 1) if avg_length else None,  # type: ignore
                    }

                # Value counts analysis (only for reasonable cardinality)
                if unique_count <= high_cardinality_threshold and unique_count > 0:
                    try:
                        top_values_query = f"""
                        SELECT 
                            "{col_name}" as value,
                            COUNT(*) as count,
                            ROUND((COUNT(*) * 100.0 / {non_null_count}), 2) as percentage
                        FROM read_csv('{file_path}')
                        WHERE "{col_name}" IS NOT NULL
                        GROUP BY "{col_name}"
                        ORDER BY count DESC
                        LIMIT {top_n_categorical}
                        """
                        top_values = duckdb.sql(top_values_query).fetchdf()  # type: ignore

                        col_summary["categorical_stats"] = {
                            "unique_values": unique_count,
                            "is_high_cardinality": unique_count > 50,
                            "top_values": top_values.to_dict("records"),  # type: ignore
                            "top_values_json": json.dumps(
                                dict(
                                    zip(  # type: ignore
                                        top_values["value"].astype(str),  # type: ignore
                                        top_values["count"],  # type: ignore
                                    )
                                )
                            ),
                        }
                    except Exception:
                        col_summary["categorical_stats"] = {
                            "unique_values": unique_count,
                            "is_high_cardinality": True,
                            "note": "Could not analyze categorical values",
                        }
                else:
                    col_summary["categorical_stats"] = {
                        "unique_values": unique_count,
                        "is_high_cardinality": True,
                        "note": f"High cardinality ({unique_count} unique values) - analysis skipped",
                    }

            # Data quality flags
            quality_flags: list[str] = []
            if missing_count / total_count > 0.5:
                quality_flags.append("high_missing_rate")
            if unique_count == 1 and missing_count == 0:
                quality_flags.append("constant_value")
            if unique_count == total_count and missing_count == 0:
                quality_flags.append("unique_identifier")
            if unique_count / non_null_count > 0.95 and non_null_count > 10:
                quality_flags.append("near_unique")

            if quality_flags:
                col_summary["data_quality"]["flags"] = quality_flags  # type: ignore

        except Exception as e:
            logger.warning(f"Error analyzing column {col_name}: {e}")
            col_summary["error"] = f"Analysis failed: {str(e)}"

        column_summaries.append(col_summary)

    summary["column_summaries"] = column_summaries

    # Cross-column correlation analysis for numeric columns
    if len(numeric_columns) > 1:
        try:
            correlations: list[dict[str, Any]] = []
            for i, col1 in enumerate(numeric_columns):
                for col2 in numeric_columns[i + 1 :]:
                    corr_query = f"""
                    SELECT corr(
                        TRY_CAST("{col1}" AS DOUBLE), 
                        TRY_CAST("{col2}" AS DOUBLE)
                    ) as correlation
                    FROM read_csv('{file_path}')
                    WHERE TRY_CAST("{col1}" AS DOUBLE) IS NOT NULL 
                      AND TRY_CAST("{col2}" AS DOUBLE) IS NOT NULL
                    """
                    corr_result = duckdb.sql(corr_query).fetchone()  # type: ignore
                    if corr_result and corr_result[0] is not None:  # type: ignore
                        correlation = float(corr_result[0])  # type: ignore
                        if abs(correlation) > 0.1:  # Only include meaningful correlations
                            correlations.append(
                                {
                                    "column1": col1,
                                    "column2": col2,
                                    "correlation": round(correlation, 3),
                                    "strength": "strong"
                                    if abs(correlation) > 0.7
                                    else "moderate"
                                    if abs(correlation) > 0.3
                                    else "weak",
                                }
                            )

            if correlations:
                summary["cross_column_analysis"] = {
                    "correlations": correlations,
                    "correlation_count": len(correlations),
                }

        except Exception as e:
            logger.warning(f"Error in correlation analysis: {e}")

    # Overall data quality assessment
    total_cells = row_count * len(column_names)  # type: ignore
    missing_cells = sum(col.get("data_quality", {}).get("missing_count", 0) for col in column_summaries)  # type: ignore

    summary["overall_quality"] = {
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "overall_completeness": round(((total_cells - missing_cells) / total_cells) * 100, 2)  # type: ignore
        if total_cells > 0
        else 0,
        "numeric_columns": len(numeric_columns),
        "categorical_columns": len([c for c in column_summaries if c.get("inferred_best_dtype") == "categorical"]),
        "datetime_columns": len([c for c in column_summaries if c.get("inferred_best_dtype") == "datetime"]),
    }

    return summary


def _pandas_csv_summary_fallback(df: pd.DataFrame, file_name: str) -> dict[str, Any]:
    """Simplified pandas fallback when DuckDB fails."""
    if df.empty:
        return {
            "file_level_summary": {
                "file_name": file_name,
                "shape": (0, 0),
                "total_rows": 0,
                "total_columns": 0,
                "status": "empty_dataframe",
            },
            "column_summaries": [],
        }

    total_rows, total_columns = df.shape
    column_summaries: list[dict[str, Any]] = []

    for col_name in df.columns:
        col_data = df[col_name]  # type: ignore
        col_summary: dict[str, Any] = {
            "column_name": col_name,
            "original_dtype": str(col_data.dtype),  # type: ignore
            "data_quality": {
                "total_count": total_rows,
                "non_null_count": int(col_data.count()),  # type: ignore
                "missing_count": int(col_data.isnull().sum()),  # type: ignore
                "unique_count": int(col_data.nunique()),  # type: ignore
            },
        }

        # Basic type inference
        if pd.api.types.is_numeric_dtype(col_data):  # type: ignore
            col_summary["inferred_best_dtype"] = "numeric"
            try:
                desc = col_data.describe()  # type: ignore
                col_summary["numeric_stats"] = {
                    "min": round(float(desc["min"]), 3),  # type: ignore
                    "max": round(float(desc["max"]), 3),  # type: ignore
                    "mean": round(float(desc["mean"]), 3),  # type: ignore
                    "std_dev": round(float(desc["std"]), 3),  # type: ignore
                }
            except Exception:
                pass
        elif pd.api.types.is_datetime64_any_dtype(col_data):  # type: ignore
            col_summary["inferred_best_dtype"] = "datetime"
        else:
            col_summary["inferred_best_dtype"] = "categorical"

        column_summaries.append(col_summary)

    return {
        "file_level_summary": {
            "file_name": file_name,
            "shape": (total_rows, total_columns),
            "total_rows": total_rows,
            "total_columns": total_columns,
            "status": "pandas_fallback",
        },
        "column_summaries": column_summaries,
    }


class AgentDeps:
    def __init__(self, thread_dir: Path, workspace_dir: Path | None = None):
        self.thread_dir = thread_dir.expanduser().resolve()
        self.workspace_dir = workspace_dir or self.thread_dir.parent.parent
        self.analysis_dir = self.thread_dir / "analysis"
        self.results_dir = self.thread_dir / "results"
        self.memory_file_path = self.workspace_dir / MEMORY_FILE_NAME
        self.message_history_path = self.thread_dir / "message_history.json"
        self.data_dir = self.workspace_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


def add_dirs(ctx: RunContext[AgentDeps]) -> str:
    return (
        f"<data_dir>\n{str(Path(ctx.deps.data_dir).expanduser().resolve())}\n</data_dir>\n"
        f"<analysis_dir>\n{str(Path(ctx.deps.analysis_dir).expanduser().resolve())}\n</analysis_dir>\n"
        f"<results_dir>\n{str(Path(ctx.deps.results_dir).expanduser().resolve())}\n</results_dir>\n\n"
        "`analysis_result_file_name` for `run_sql` will be saved in the `analysis_dir`.\n"
        "`workbook_name` for `write_sheet_from_file` will be saved in the `results_dir`."
    )


def add_current_time() -> str:
    return f"<current_time>\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n</current_time>"


def _list_csv_files(dir: Path) -> str:
    """
    Lists all available csv files in the `dir` and their summaries.
    """
    dir_name = dir.name
    csv_files = [file.expanduser().resolve() for file in Path(dir).glob("*.csv")]
    res = f"\n<available_{dir_name}_files>\n"
    for file in csv_files:
        res += json.dumps(csv_summary(file_path=file)) + "\n\n"
    return res.strip() + f"\n</available_{dir_name}_files>"


def list_data_files(ctx: RunContext[AgentDeps]) -> str:
    """
    Lists all available csv files in the `data_dir` and their summaries.
    """
    return _list_csv_files(dir=Path(ctx.deps.data_dir))


def list_analysis_files(ctx: RunContext[AgentDeps]) -> str:
    """
    Lists all the analysis csv files created so far in the `analysis_dir` and their summaries.
    """
    return _list_csv_files(dir=Path(ctx.deps.analysis_dir))


class RunSQLResult(BaseModel):
    purpose: str
    sql_path: str
    analysis_result_file_path: str
    summary: dict[str, Any]


async def run_sql(
    ctx: RunContext[AgentDeps], purpose: str, query: str, analysis_result_file_name: str
) -> RunSQLResult:
    """Runs an SQL query on csv file(s) using DuckDB and writes the result to analysis_dir.

    Args:
        purpose: Describe what you want to achieve with the query and why. Relate it to the main task.
            This will be a user-facing message, so no need to be too technical or mention the internal errors faced if any.
        query: The SQL query to execute.
            When reading a csv, use the FULL path with the data_dir included.
            Example: 'select names from read_csv('workspaces/1/data/orders.csv')'
        analysis_result_file_name: Descriptive name of the file based on the query to save the result to in the `analysis_dir`.
            Example: 'customers_joined_in_2023.csv'
            '.csv' is optional. It will be added automatically if not provided.

    Returns:
        RunSQLResult: Contains the purpose, file paths, summary.
    """

    async def _run_with_timeout():
        query_replaced = query.replace('"', "'").strip()
        logger.info(f"Running SQL query: {query_replaced}")

        try:
            df: pd.DataFrame = duckdb.sql(query_replaced).df()  # type: ignore
        except Exception:
            # Handle transaction rollback for DuckDB transaction errors
            try:
                duckdb.sql("ROLLBACK")
                logger.info("Rolled back DuckDB transaction after error")
            except Exception:
                pass  # Rollback might fail if no transaction is active
            raise

        logger.success(f"Successfully ran SQL query: {query_replaced}")
        file_path = Path(ctx.deps.analysis_dir) / Path(analysis_result_file_name).name
        file_path = file_path.expanduser().resolve().with_suffix(".csv")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.exists():
            file_path = file_path.with_stem(f"{file_path.stem}_{uuid4()}")

        df.to_csv(file_path, index=False)
        file_path.with_suffix(".sql").write_text(query_replaced)

        return RunSQLResult(
            purpose=purpose,
            sql_path=str(file_path.with_suffix(".sql")),
            analysis_result_file_path=str(file_path),
            summary=csv_summary(file_path=file_path),
        )

    try:
        return await asyncio.wait_for(_run_with_timeout(), timeout=SQL_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        raise ModelRetry(f"SQL query execution timed out after {SQL_TIMEOUT_SECONDS} seconds")
    except FileNotFoundError as e:
        raise ModelRetry(str(e))
    except Exception as e:
        raise ModelRetry(str(e))


def load_analysis_file(ctx: RunContext[AgentDeps], file_name: str) -> list[dict[str, Any]]:
    """
    Loads an analysis csv file from the `analysis_dir` and returns all of the records.
    """
    file_path = Path(ctx.deps.analysis_dir) / Path(file_name).name
    records = pd.read_csv(file_path).to_dict(orient="records")  # type: ignore
    tokens = len(json.dumps(records)) / 4
    if tokens > MAX_ANALYSIS_FILE_TOKENS:
        raise ModelRetry(
            (
                f"The file {file_name} is too large. It has approx {tokens} tokens.\n "
                f"We can only load files with less than {MAX_ANALYSIS_FILE_TOKENS} tokens.\n"
                f"File Summary:\n{csv_summary(file_path=file_path)}\n"
                "Please try using your SQL expertise to get a smaller subset of the data. "
                "If you have already tried that, let the user know with a helpful message.\n"
                "The user won't know about tokens, but they will understand rows and columns."
            )
        )
    return records  # type: ignore


def calculate_sum(values: list[float]) -> float:
    """Calculate the sum of a list of values."""
    if len(values) < 2:
        raise ModelRetry("Need at least 2 values to calculate sum")
    return sum(values)


def calculate_difference(num1: float, num2: float) -> float:
    """Calculate the difference between two numbers."""
    return abs(num1 - num2)


def calculate_mean(values: list[float]) -> float:
    """Calculate the mean of a list of values."""
    if len(values) < 2:
        raise ModelRetry("Need at least 2 values to calculate mean")
    return sum(values) / len(values)


class WriteDataToExcelResult(BaseModel):
    excel_file_path: str


class UserInteraction(BaseModel):
    """
    Interacts with the user. Could be:
    - A question
    - A progress update
    - An assumption made that needs to be validated
    - A request for clarification
    - Anything else needed from the user to proceed
    """

    message: str = Field(description="The message to display to the user.")


class TaskResult(BaseModel):
    """
    A task result.
    """

    message: str = Field(description="The final response to the user.")


def create_agent(
    agent_deps: AgentDeps,
    model: Model | KnownModelName,
    use_excel_tools: bool = True,
    use_thinking: bool = True,
    use_memory: bool = True,
    temperature: float = 0.0,
) -> Agent[AgentDeps, RunSQLResult | WriteDataToExcelResult | UserInteraction | TaskResult]:
    thinking_server = MCPServerStdio(
        command="npx", args=["-y", "@modelcontextprotocol/server-sequential-thinking"]
    )
    memory_server = MCPServerStdio(
        command="uv",
        args=["run", str(MODULE_DIR / "memory_mcp.py")],
        env={"MEMORY_FILE_PATH": str(agent_deps.memory_file_path)},
    )
    excel_server = MCPServerStdio(command="uvx", args=[str(MODULE_DIR / "../../../excel-mcp-server"), "stdio"])
    mcp_servers: list[MCPServerStdio] = []
    prompts = [Path(MODULE_DIR / "prompts/cfo.md").read_text()]
    if use_thinking:
        mcp_servers.append(thinking_server)
    if use_memory:
        mcp_servers.append(memory_server)
        prompts.append(Path(MODULE_DIR / "prompts/memory.md").read_text())
    if use_excel_tools:
        mcp_servers.append(excel_server)
    output_types: list[type | Callable[..., Any]] = [run_sql, UserInteraction, TaskResult]
    if use_excel_tools:
        output_types.append(WriteDataToExcelResult)
    return Agent(
        model=model,
        instructions=[*prompts, add_current_time, add_dirs],
        deps_type=AgentDeps,
        retries=MAX_RETRIES,
        tools=[
            list_data_files,
            list_analysis_files,
            load_analysis_file,
            calculate_sum,
            calculate_difference,
            calculate_mean,
        ],
        mcp_servers=mcp_servers,
        output_type=output_types,
        instrument=True,
        model_settings={"temperature": temperature},
    )


async def thread(
    thread_dir: Path,
    user_prompt: str,
    model: KnownModelName | FallbackModel | None = None,
    use_excel_tools: bool = True,
    use_thinking: bool = True,
    use_memory: bool = True,
    temperature: float = 0.0,
) -> RunSQLResult | WriteDataToExcelResult | UserInteraction | TaskResult:
    agent_deps = AgentDeps(thread_dir=thread_dir)
    model = model or FallbackModel(
        "anthropic:claude-4-sonnet-20250514",
        "openai:gpt-4.1",
        "google-gla:gemini-2.5-flash-preview-05-20",
        "openai:gpt-4.1-mini",
    )
    agent = create_agent(
        model=model,
        agent_deps=agent_deps,
        use_excel_tools=use_excel_tools,
        use_thinking=use_thinking,
        use_memory=use_memory,
        temperature=temperature,
    )

    res = await agent.run(
        user_prompt=user_prompt,
        deps=agent_deps,
        message_history=ModelMessagesTypeAdapter.validate_json(agent_deps.message_history_path.read_bytes())
        if agent_deps.message_history_path.exists()
        else None,
    )
    agent_deps.message_history_path.write_bytes(to_json(res.all_messages()))

    return res.output


def setup_workspace(data_dir: Path | str, workspace_dir: Path | str, delete_existing: bool = False):
    data_dir = Path(data_dir)
    workspace_dir = Path(workspace_dir)
    if workspace_dir.exists() and delete_existing:
        shutil.rmtree(workspace_dir)
        logger.info(f"Removed {workspace_dir}")
    workspace_dir.mkdir(parents=True, exist_ok=True)
    workspace_data_dir = workspace_dir / "data"
    if data_dir.exists():
        try:
            shutil.copytree(data_dir, workspace_data_dir, dirs_exist_ok=True)
            logger.success(f"Copied data from {data_dir} to {workspace_data_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to copy data from {data_dir} to {workspace_data_dir}: {e}") from e
