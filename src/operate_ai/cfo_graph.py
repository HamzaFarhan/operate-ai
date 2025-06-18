from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Self, TypedDict
from uuid import uuid4

import duckdb
import logfire
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_graph import BaseNode, End, Graph, GraphRunContext
from pydantic_graph.persistence.file import FileStatePersistence
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

logfire.configure()

SQL_TIMEOUT_SECONDS = 10
MAX_ANALYSIS_FILE_TOKENS = 20_000
MAX_RETRIES = 10
MEMORY_FILE_NAME = os.getenv("MEMORY_FILE_NAME", "memory.json")
STEP_LIMIT = 100
MODULE_DIR = Path(__file__).parent


def user_message(content: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=content)])


class Query[OutputT]:
    def __init__(self, query: Path | str, output_type: type[OutputT] | None = None):
        self.output_type = output_type
        if isinstance(query, Path) and query.exists():
            self.query = Path(query).read_text()
        else:
            self.query = str(query)

    def __str__(self) -> str:
        return self.query

    def __repr__(self) -> str:
        return self.query


class Success(BaseModel):
    message: Any


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: RunSQLResult | WriteDataToExcelResult | str
    state_path: str


class AgentDeps(BaseModel):
    data_dir: str
    analysis_dir: str
    results_dir: str
    state_path: str

    @model_validator(mode="after")
    def create_dirs(self: Self) -> Self:
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.analysis_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class GraphState:
    chat_messages: list[ChatMessage] = field(default_factory=list)
    message_history: list[ModelMessage] = field(default_factory=list)
    run_sql_attempts: int = 0
    write_sheet_attempts: int = 0


@dataclass
class GraphDeps:
    agent: Agent[AgentDeps, TaskResult | WriteDataToExcelResult | RunSQL | UserInteraction]
    agent_deps: AgentDeps


def extract_csv_paths(sql_query: str) -> list[str]:
    """Extract CSV file paths from SQL query."""

    # Find read_csv calls with single or multiple CSVs
    read_csv_pattern = r"read_csv\(\s*'([^']*\.csv)'\s*\)|read_csv\(\s*\[([^\]]*)\]\s*\)"
    read_csv_matches = re.findall(read_csv_pattern, sql_query, re.IGNORECASE)

    # Extract paths from both single and multiple CSV cases
    paths: list[str] = []
    for single_path, multiple_paths in read_csv_matches:
        if single_path:
            paths.append(single_path)
        elif multiple_paths:
            # Extract individual paths from the array
            paths.extend(re.findall(r"'([^']*\.csv)'", multiple_paths, re.IGNORECASE))

    return paths


def add_dirs(ctx: RunContext[AgentDeps]) -> str:
    return (
        f"<data_dir>\n{str(Path(ctx.deps.data_dir).expanduser().resolve())}\n</data_dir>\n"
        f"<analysis_dir>\n{str(Path(ctx.deps.analysis_dir).expanduser().resolve())}\n</analysis_dir>\n"
        f"<results_dir>\n{str(Path(ctx.deps.results_dir).expanduser().resolve())}\n</results_dir>\n\n"
        "Write all excel workbooks in the `results_dir`. Use the full absolute path."
    )


def add_current_time() -> str:
    return f"<current_time>\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n</current_time>"


# def preview_csv(df: Path | pd.DataFrame, num_rows: int = PREVIEW_ROWS) -> list[dict[Hashable, Any]]:
#     """
#     Returns the first `num_rows` rows of the specified CSV file.
#     Should be used for previewing the data.
#     """
#     if isinstance(df, pd.DataFrame):
#         return df.head(num_rows).to_dict(orient="records")  # type: ignore
#     return pd.read_csv(df).head(num_rows).to_dict(orient="records")  # type: ignore


def csv_summary(
    df: Path | pd.DataFrame, top_n_categorical: int = 50, high_cardinality_threshold: int = 100
) -> dict[str, Any]:
    """
    Generates comprehensive metadata for a CSV file using DuckDB for performance.

    Args:
        df: Path to CSV file or DataFrame to analyze
        top_n_categorical: Number of top values to show for categorical columns
        high_cardinality_threshold: Threshold above which to skip detailed categorical analysis
    Returns:
        dict: Comprehensive metadata including statistics, data quality, and analysis
    """
    import json

    file_name = ""
    if isinstance(df, Path):
        file_name = df.name
        file_path = str(df.expanduser().resolve())

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
            return _pandas_csv_summary_fallback(pd.read_csv(df), file_name)  # type: ignore

    else:
        # DataFrame input - save to temp file for DuckDB analysis
        temp_path = temp_file_path()
        file_path = str(temp_path)
        df.to_csv(file_path, index=False)
        row_count = len(df)
        column_names = df.columns.tolist()
        column_types = {col: str(df[col].dtype) for col in df.columns}

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

                # Enhanced date format detection
                try:
                    date_format_info = asyncio.run(detect_date_format_and_parse(file_path, col_name))
                except Exception as e:
                    date_format_info = {"error": f"Date format detection failed: {str(e)}"}

                if "error" not in date_format_info:
                    col_summary["datetime_stats"]["format_detection"] = date_format_info
                    col_summary["datetime_stats"]["recommended_sql_cast"] = (
                        date_format_info["date_format"]
                        if date_format_info.get("duckdb_format")
                        else f"strptime(\"{col_name}\", '{date_format_info['date_format']}')"
                    )
                else:
                    col_summary["datetime_stats"]["format_detection"] = date_format_info

                # Time series analysis
                if non_null_count > 10:
                    try:
                        # Use detected format if available
                        if "error" not in date_format_info and date_format_info.get("duckdb_format"):
                            date_cast = date_format_info["date_format"]
                        else:
                            date_cast = f'TRY_CAST("{col_name}" AS DATE)'

                        date_dist_query = f"""
                        SELECT 
                            EXTRACT(YEAR FROM {date_cast}) as year,
                            COUNT(*) as count
                        FROM read_csv('{file_path}')
                        WHERE {date_cast} IS NOT NULL
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

    # Clean up temp file if created
    if isinstance(df, pd.DataFrame) and Path(file_path).exists():
        Path(file_path).unlink()

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


def list_csv_files(ctx: RunContext[AgentDeps]) -> str:
    """
    Lists all available csv files.
    """
    csv_files = [file.expanduser().resolve() for file in Path(ctx.deps.data_dir).glob("*.csv")]
    res = "\n<available_csv_files>\n"
    for file in csv_files:
        res += json.dumps(csv_summary(df=file)) + "\n\n"
    return res.strip() + "\n</available_csv_files>"


def list_analysis_files(ctx: RunContext[AgentDeps]) -> str:
    """
    Lists all the analysis files created so far.
    """
    csv_files = [file.expanduser().resolve() for file in Path(ctx.deps.analysis_dir).glob("*.csv")]
    res = "\n<available_analysis_files>\n"
    for file in csv_files:
        res += json.dumps(csv_summary(df=file)) + "\n\n"
    return res.strip() + "\n</available_analysis_files>"


def temp_file_path(file_dir: Path | str | None = None) -> Path:
    file_dir = file_dir or tempfile.gettempdir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(file_dir) / f"ai_cfo_result_{Path(file_dir).name}_{ts}.csv"


class RunSQL(BaseModel):
    """
    1. Runs an SQL query on csv file(s) using duckdb
    2. Writes the full result to disk
    """

    purpose: str = Field(
        description=(
            "Describe what you want to achieve with the query and why. Relate it to the main task.\n"
            "This will be a user-facing message, so no need to be too technical or mention the internal errors faced if any."
        )
    )
    query: str = Field(
        description=(
            "The SQL query to execute."
            "When reading a csv, use the FULL path with the data_dir included.\n"
            "Example: 'select names from read_csv('workspaces/1/data/orders.csv')'"
        )
    )
    file_name: str | None = Field(
        description=(
            "Descriptive name of the file based on the query to save the result to in the `analysis_dir`.\n"
            "If None, a file path will be created in the `analysis_dir` based on the current timestamp.\n"
            "Example: 'customers_joined_in_2023.csv'\n"
            "'.csv' is optional. It will be added automatically if not provided."
        )
    )
    is_task_result: bool = Field(
        default=False,
        description="If True, the result will be used as a task result and we can stop. Otherwise, it will be used as a tool result.",
    )


class RunSQLResult(BaseModel):
    """
    A lightweight reference to a (possibly large) query result on disk.
    """

    purpose: str | None = None
    sql_path: str
    csv_path: str
    summary: dict[str, Any]
    is_task_result: bool = False


async def run_sql(
    analysis_dir: str, query: str, file_name: str | None = None, is_task_result: bool = False
) -> RunSQLResult:
    def check_csv_files_exist(paths: list[str]) -> None:
        """Check if all CSV files in the paths exist."""
        for path in paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"CSV file not found: {path}")

    async def _run_with_timeout():
        query_replaced = query.replace('"', "'").strip()
        logger.info(f"Running SQL query: {query_replaced}")

        # Extract and check CSV files before executing query
        csv_paths = extract_csv_paths(query_replaced)
        if not csv_paths:
            logger.warning("No CSV files found in SQL query")
        else:
            logger.info(f"Found CSV files in query: {csv_paths}")
            check_csv_files_exist(csv_paths)

        try:
            df: pd.DataFrame = duckdb.sql(query_replaced).df()  # type: ignore
        except Exception as e:
            # Handle transaction rollback for DuckDB transaction errors
            try:
                duckdb.sql("ROLLBACK")
                logger.info("Rolled back DuckDB transaction after error")
            except Exception:
                pass  # Rollback might fail if no transaction is active
            raise e

        file_path = (
            Path(analysis_dir) / Path(file_name).name if file_name else temp_file_path(file_dir=analysis_dir)
        )
        file_path = file_path.expanduser().resolve().with_suffix(".csv")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.exists():
            file_path = file_path.with_stem(f"{file_path.stem}_{uuid4()}")

        df.to_csv(file_path, index=False)
        file_path.with_suffix(".sql").write_text(query_replaced)

        return RunSQLResult(
            sql_path=str(file_path.with_suffix(".sql")),
            csv_path=str(file_path),
            summary=csv_summary(df=df),
            is_task_result=is_task_result,
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
    Loads an analysis file from the `analysis_dir`.
    """
    file_path = Path(ctx.deps.analysis_dir) / Path(file_name).name
    records = pd.read_csv(file_path).to_dict(orient="records")  # type: ignore
    tokens = len(json.dumps(records)) / 4
    if tokens > MAX_ANALYSIS_FILE_TOKENS:
        raise ModelRetry(
            (
                f"The file {file_name} is too large. It has approx {tokens} tokens.\n "
                f"We can only load files with less than {MAX_ANALYSIS_FILE_TOKENS} tokens.\n"
                f"File Summary:\n{csv_summary(df=file_path)}\n"
                "Please try using your SQL expertise to get a smaller subset of the data. "
                "If you have already tried that, let the user know with a helpful message."
            )
        )
    return records  # type: ignore


@dataclass
class RunSQLNode(BaseNode[GraphState, GraphDeps, RunSQLResult]):
    """Run 'run_sql' tool."""

    docstring_notes = True
    purpose: str
    query: str
    file_name: str | None = None
    is_task_result: bool = False

    async def run(self, ctx: GraphRunContext[GraphState, GraphDeps]) -> RunAgentNode | End[RunSQLResult]:
        try:
            sql_result = await run_sql(
                analysis_dir=ctx.deps.agent_deps.analysis_dir,
                query=self.query,
                file_name=self.file_name,
                is_task_result=self.is_task_result,
            )
            sql_result.purpose = self.purpose
            ctx.state.chat_messages.append(
                {"role": "assistant", "content": sql_result, "state_path": ctx.deps.agent_deps.state_path}
            )
            ctx.state.message_history.append(user_message(sql_result.model_dump_json(exclude={"purpose"})))
            return End(data=sql_result)
        except Exception as e:
            logger.exception("Error running SQL query")
            ctx.state.message_history.append(user_message(content=f"Error in RunSQL: {str(e)}"))
            ctx.state.run_sql_attempts += 1
            return RunAgentNode(user_prompt="")


def calculate_sum(values: list[float]) -> float:
    """Calculate the sum of a list of values."""
    if len(values) < 2:
        raise ModelRetry("Need at least 2 values to calculate sum")
    return sum(values)


def calculate_difference(num1: float, num2: float) -> float:
    """Calculate the difference between two numbers by subtracting `num1` from `num2`"""
    if num1 == num2:
        raise ModelRetry("The two numbers are the same. Please try again.")
    return num2 - num1


def calculate_mean(values: list[float]) -> float:
    """Calculate the mean of a list of values."""
    if len(values) < 2:
        raise ModelRetry("Need at least 2 values to calculate mean")
    return sum(values) / len(values)


def get_date_cast_expression(ctx: RunContext[AgentDeps], csv_file: str, column_name: str) -> str:
    """
    Get the proper SQL expression to cast a date column based on detected format.

    Args:
        csv_file: Name of the CSV file (will be resolved from data_dir)
        column_name: Name of the date column

    Returns:
        SQL expression to properly cast the date column
    """
    # Resolve full path
    csv_path = str(Path(ctx.deps.data_dir) / csv_file)

    # Detect date format with timeout protection
    try:
        format_info = asyncio.run(detect_date_format_and_parse(csv_path, column_name))
    except Exception as e:
        return f'TRY_CAST("{column_name}" AS DATE)  -- Date detection failed: {str(e)}'

    if "error" in format_info:
        return f'TRY_CAST("{column_name}" AS DATE)  -- Format detection failed: {format_info["error"]}'

    if format_info.get("duckdb_format"):
        return format_info["date_format"].format(col=column_name)
    else:
        return f"strptime(\"{column_name}\", '{format_info['date_format']}')"


def parse_date_reference(date_ref: str) -> dict[str, str]:
    """
    Parse date reference consistently according to CFO standards.

    Args:
        date_ref: Date reference like "Jan 2023", "Q1 2023", "2023"

    Returns:
        Dict with start_date, end_date, and interpretation_type
    """
    import re
    from datetime import datetime

    date_ref = date_ref.strip()

    # Month Year patterns (Jan 2023, January 2023)
    month_year_pattern = r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})$"
    match = re.match(month_year_pattern, date_ref, re.IGNORECASE)
    if match:
        month_str, year_str = match.groups()

        # Map month names to numbers
        month_map = {
            "jan": 1,
            "january": 1,
            "feb": 2,
            "february": 2,
            "mar": 3,
            "march": 3,
            "apr": 4,
            "april": 4,
            "may": 5,
            "jun": 6,
            "june": 6,
            "jul": 7,
            "july": 7,
            "aug": 8,
            "august": 8,
            "sep": 9,
            "september": 9,
            "oct": 10,
            "october": 10,
            "nov": 11,
            "november": 11,
            "dec": 12,
            "december": 12,
        }

        month_num = month_map[month_str.lower()]
        year = int(year_str)

        # Calculate last day of month
        if month_num == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month_num + 1, 1)

        from datetime import timedelta

        last_day = (next_month - timedelta(days=1)).day

        return {
            "start_date": f"{year:04d}-{month_num:02d}-01",
            "end_date": f"{year:04d}-{month_num:02d}-{last_day:02d}",
            "interpretation_type": "full_month_period",
        }

    # Quarter patterns (Q1 2023, Q4 2023)
    quarter_pattern = r"^Q([1-4])\s+(\d{4})$"
    match = re.match(quarter_pattern, date_ref, re.IGNORECASE)
    if match:
        quarter, year_str = match.groups()
        year = int(year_str)
        quarter = int(quarter)

        quarter_starts = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}
        quarter_ends = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}

        return {
            "start_date": f"{year:04d}-{quarter_starts[quarter]}",
            "end_date": f"{year:04d}-{quarter_ends[quarter]}",
            "interpretation_type": "full_quarter_period",
        }

    # Year patterns (2023)
    year_pattern = r"^(\d{4})$"
    match = re.match(year_pattern, date_ref)
    if match:
        year = int(match.group(1))
        return {
            "start_date": f"{year:04d}-01-01",
            "end_date": f"{year:04d}-12-31",
            "interpretation_type": "full_year_period",
        }

    # Point-in-time patterns
    if date_ref.lower().startswith("as of ") or date_ref.lower().startswith("end of "):
        base_ref = date_ref.lower().replace("as of ", "").replace("end of ", "").strip()
        parsed = parse_date_reference(base_ref)
        return {
            "start_date": parsed["end_date"],
            "end_date": parsed["end_date"],
            "interpretation_type": "point_in_time_end",
        }

    if date_ref.lower().startswith("beginning of "):
        base_ref = date_ref.lower().replace("beginning of ", "").strip()
        parsed = parse_date_reference(base_ref)
        return {
            "start_date": parsed["start_date"],
            "end_date": parsed["start_date"],
            "interpretation_type": "point_in_time_start",
        }

    # If we can't parse it, return as-is with a warning
    return {
        "start_date": date_ref,
        "end_date": date_ref,
        "interpretation_type": "unparsed",
        "warning": f"Could not parse date reference: {date_ref}. Please use explicit dates or standard formats like 'Jan 2023', 'Q1 2023', '2023'",
    }


class WriteSheetFromFile(BaseModel):
    """
    Creates an excel workbook and writes a sheet to it.
    1. Reads from the `file_path` of the previously received RunSQLResult
    2. Appends to / creates `workbook_path`.
    """

    file_path: str = Field(
        description="Path to the file to read from. Almost always a `file_path` of a previously received RunSQLResult."
    )
    sheet_name: str = Field(description="The name of the sheet to write to.")
    workbook_name: str | None = Field(description="Name of the workbook to append to / create in `results_dir`.")


class WriteDataToExcelResult(BaseModel):
    file_path: str


def write_sheet_from_file(
    results_dir: str, file_path: str, sheet_name: str, workbook_name: str | None = None
) -> WriteDataToExcelResult:
    """
    Creates an excel workbook and writes a sheet to it.
    1. Reads from the `file_path` of the previously received RunSQLResult
    2. Appends to / creates `workbook_path`.

    Parameters
    ----------
    file_path : str
        Path to the file to read from. Almost always a `file_path` of a previously received RunSQLResult.
    sheet_name : str
        The name of the sheet to write to.
    workbook_name : str | None, default None
        Name of the workbook to append to / create in `results_dir`.
        If None, a path will be created in the `results_dir` using the current timestamp.

    Returns
    -------
    WriteSheetResult
    """

    workbook_path = (
        Path(results_dir) / Path(workbook_name).name if workbook_name else temp_file_path(file_dir=results_dir)
    )
    wb_path = workbook_path.expanduser().resolve().with_suffix(".xlsx")
    wb_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing to workbook: {wb_path}")

    df: pd.DataFrame = pd.read_csv(Path(file_path).expanduser().resolve())  # type: ignore

    try:
        mode = "w"
        if_sheet_exists = None
        if wb_path.exists():
            mode = "a"
            if_sheet_exists = "replace"
        with pd.ExcelWriter(wb_path, mode=mode, engine="openpyxl", if_sheet_exists=if_sheet_exists) as w:
            df.to_excel(w, sheet_name=sheet_name[:31] or "Sheet1", index=False)  # type: ignore

        _ = pd.read_excel(wb_path)  # type: ignore
        logger.info(f"Successfully wrote to workbook: {wb_path}")
    except Exception as e:
        logger.warning(f"Failed to write to workbook: {wb_path}")
        raise ModelRetry(
            (
                f"Got this error when trying to read/write the workbook.\n"
                "I have reverted the step. Try again:\n"
                f"{str(e)}"
            )
        )
    return WriteDataToExcelResult(file_path=str(wb_path))


@dataclass
class WriteSheetNode(BaseNode[GraphState, GraphDeps, WriteDataToExcelResult]):
    """Run 'write_sheet_from_file' tool."""

    docstring_notes = True
    file_path: str
    sheet_name: str
    workbook_name: str | None = None

    async def run(self, ctx: GraphRunContext[GraphState, GraphDeps]) -> RunAgentNode | End[WriteDataToExcelResult]:
        try:
            write_sheet_result = write_sheet_from_file(
                results_dir=ctx.deps.agent_deps.results_dir,
                file_path=self.file_path,
                sheet_name=self.sheet_name,
                workbook_name=self.workbook_name,
            )
            ctx.state.chat_messages.append(
                {"role": "assistant", "content": write_sheet_result, "state_path": ctx.deps.agent_deps.state_path}
            )
            ctx.state.message_history.append(user_message(write_sheet_result.model_dump_json()))
            return End(data=write_sheet_result)
        except Exception as e:
            logger.exception("Error writing sheet")
            ctx.state.message_history.append(user_message(content=f"Error in WriteSheet: {str(e)}"))
            ctx.state.write_sheet_attempts += 1
            return RunAgentNode(user_prompt="")


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


def user_interaction(message: str) -> str:
    """
    Interacts with the user. Could be:
    - A question
    - A progress update
    - An assumption made that needs to be validated
    - A request for clarification
    - Anything else needed from the user to proceed
    """
    res = input(f"{message}> ")
    return res


@dataclass
class UserInteractionNode(BaseNode[GraphState, GraphDeps, str]):
    """Pass to End."""

    docstring_notes = True
    message: str

    async def run(self, ctx: GraphRunContext[GraphState, GraphDeps]) -> End[str]:  # noqa: ARG002
        ctx.state.chat_messages.append(
            {"role": "assistant", "content": self.message, "state_path": ctx.deps.agent_deps.state_path}
        )
        return End(data=self.message)


class TaskResult(BaseModel):
    """
    A task result.
    """

    message: str = Field(description="The final response to the user.")


@dataclass
class TaskResultNode(BaseNode[GraphState, GraphDeps, TaskResult]):
    """Pass to End."""

    docstring_notes = True
    task_result: TaskResult

    async def run(self, ctx: GraphRunContext[GraphState, GraphDeps]) -> End[TaskResult]:  # noqa: ARG002
        ctx.state.chat_messages.append(
            {
                "role": "assistant",
                "content": self.task_result.message,
                "state_path": ctx.deps.agent_deps.state_path,
            }
        )
        return End(data=self.task_result)


def create_agent(
    model: Model | KnownModelName,
    workspace_dir: Path | str,
    use_excel_tools: bool = True,
    use_thinking: bool = True,
    use_memory: bool = True,
    temperature: float = 0.0,
) -> Agent[AgentDeps, TaskResult | WriteDataToExcelResult | RunSQL | UserInteraction]:
    thinking_server = MCPServerStdio(
        command="npx", args=["-y", "@modelcontextprotocol/server-sequential-thinking"]
    )
    memory_server = MCPServerStdio(
        command="uv",
        args=["run", str(MODULE_DIR / "memory_mcp.py")],
        env={"MEMORY_FILE_PATH": str(Path(workspace_dir) / Path(MEMORY_FILE_NAME))},
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
    output_types: list[type] = [UserInteraction, TaskResult, RunSQL]
    if use_excel_tools:
        output_types.append(WriteDataToExcelResult)
    return Agent(
        model=model,
        instructions=[*prompts, add_current_time, add_dirs],
        deps_type=AgentDeps,
        retries=MAX_RETRIES,
        tools=[
            list_csv_files,
            list_analysis_files,
            load_analysis_file,
            calculate_sum,
            calculate_difference,
            calculate_mean,
            get_date_cast_expression,
            parse_date_reference,
        ],
        mcp_servers=mcp_servers,
        output_type=output_types,
        instrument=True,
        model_settings={"temperature": temperature},
    )


@dataclass
class RunAgentNode(BaseNode[GraphState, GraphDeps, RunSQLResult | WriteDataToExcelResult | TaskResult | str]):
    """Run the agent."""

    docstring_notes = True
    user_prompt: str

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def _run_agent_with_retry(self, ctx: GraphRunContext[GraphState, GraphDeps]):
        async with ctx.deps.agent.run_mcp_servers():
            return await ctx.deps.agent.run(
                user_prompt=self.user_prompt,
                deps=ctx.deps.agent_deps,
                message_history=ctx.state.message_history,
                model_settings={"temperature": 0.0},
            )

    async def run(
        self, ctx: GraphRunContext[GraphState, GraphDeps]
    ) -> (
        RunSQLNode
        # | WriteSheetNode
        | UserInteractionNode
        | TaskResultNode
        | End[RunSQLResult | WriteDataToExcelResult | TaskResult | str]
    ):
        if self.user_prompt:
            ctx.state.chat_messages.append(
                {"role": "user", "content": self.user_prompt, "state_path": ctx.deps.agent_deps.state_path}
            )
        error_result = End(data="Ran into an error. Please try again.")

        try:
            res = await self._run_agent_with_retry(ctx)
        except Exception:
            logger.exception("Error running agent after retries")
            return error_result

        try:
            ctx.state.message_history += res.new_messages()
            if isinstance(res.output, RunSQL):
                if ctx.state.run_sql_attempts < MAX_RETRIES:
                    return RunSQLNode(
                        purpose=res.output.purpose,
                        query=res.output.query,
                        file_name=res.output.file_name,
                        is_task_result=res.output.is_task_result,
                    )
                return error_result
            elif isinstance(res.output, UserInteraction):
                return UserInteractionNode(message=res.output.message)
            elif isinstance(res.output, WriteDataToExcelResult):
                ctx.state.chat_messages.append(
                    {"role": "assistant", "content": res.output, "state_path": ctx.deps.agent_deps.state_path}
                )
                return End(data=res.output)
            else:
                return TaskResultNode(task_result=res.output)
        except Exception:
            logger.exception("Error processing agent result")
            return error_result


graph = Graph(nodes=(RunAgentNode, RunSQLNode, UserInteractionNode, TaskResultNode), name="CFO Graph")
# try:
#     graph.mermaid_save(Path("cfo_graph.jpg"), direction="LR", highlighted_nodes=RunAgentNode)
# except Exception:
#     logger.exception("Error saving graph")


def setup_thread_dirs(thread_dir: Path | str):
    thread_dir = Path(thread_dir).expanduser().resolve()
    thread_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir = thread_dir.parent.parent
    data_dir = workspace_dir / "data"
    analysis_dir = thread_dir / "analysis"
    results_dir = thread_dir / "results"
    states_dir = thread_dir / "states"
    data_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    states_dir.mkdir(parents=True, exist_ok=True)
    return


async def load_state(
    persistence: FileStatePersistence[GraphState, RunSQLResult | WriteDataToExcelResult | str],
) -> GraphState:
    if snapshot := await persistence.load_next():
        return snapshot.state
    snapshots = await persistence.load_all()
    return (
        GraphState(
            chat_messages=snapshots[-1].state.chat_messages, message_history=snapshots[-1].state.message_history
        )
        if snapshots
        else GraphState()
    )


def get_prev_state_path(thread_dir: Path | str) -> Path | None:
    thread_dir = Path(thread_dir).expanduser().resolve()
    states_dir = thread_dir / "states"
    prev_states = sorted(
        states_dir.glob("*.json"), key=lambda p: datetime.strptime(p.stem, "%d-%m-%Y_%H-%M-%S"), reverse=True
    )
    return prev_states[0] if prev_states else None


async def load_prev_state(
    thread_dir: Path | str, prev_state_path: Path | str | None = None, get_if_none: bool = True
) -> GraphState:
    prev_state_path = (
        Path(prev_state_path) if prev_state_path else get_prev_state_path(thread_dir) if get_if_none else None
    )
    logger.info(f"Previous state path: {prev_state_path}")
    if prev_state_path and prev_state_path.exists():
        prev_persistence = FileStatePersistence(json_file=prev_state_path.expanduser().resolve())
        prev_persistence.set_graph_types(graph=graph)
        return await load_state(persistence=prev_persistence)
    return GraphState()


async def thread(
    thread_dir: Path | str,
    user_prompt: str,
    model: KnownModelName | FallbackModel | None = None,
    use_excel_tools: bool = True,
    use_thinking: bool = True,
    use_memory: bool = True,
    prev_state_path: Path | str | None = None,
    get_prev_state_if_none: bool = True,
    temperature: float = 0.0,
) -> RunSQLResult | WriteDataToExcelResult | TaskResult | str:
    thread_dir = Path(thread_dir).expanduser().resolve()
    setup_thread_dirs(thread_dir)
    data_dir = thread_dir.parent.parent / "data"
    analysis_dir = thread_dir / "analysis"
    results_dir = thread_dir / "results"
    states_dir = thread_dir / "states"

    state = await load_prev_state(
        thread_dir=thread_dir, prev_state_path=prev_state_path, get_if_none=get_prev_state_if_none
    )
    persistence_path = states_dir / f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.json"
    persistence = FileStatePersistence(json_file=persistence_path.expanduser().resolve())
    persistence.set_graph_types(graph=graph)
    model = model or FallbackModel(
        "anthropic:claude-4-sonnet-20250514",
        "openai:gpt-4.1",
        "google-gla:gemini-2.5-flash-preview-05-20",
        "openai:gpt-4.1-mini",
    )
    agent_deps = AgentDeps(
        data_dir=str(data_dir),
        analysis_dir=str(analysis_dir),
        results_dir=str(results_dir),
        state_path=str(persistence_path),
    )
    agent = create_agent(
        model=model,
        workspace_dir=thread_dir.parent.parent,
        use_excel_tools=use_excel_tools,
        use_thinking=use_thinking,
        use_memory=use_memory,
        temperature=temperature,
    )
    res = await graph.run(
        start_node=RunAgentNode(user_prompt=user_prompt),
        state=state,
        deps=GraphDeps(agent=agent, agent_deps=agent_deps),
        persistence=persistence,
    )
    return res.output


async def run_task[OutputT](
    query: Query[OutputT],
    workspace_dir: Path | str,
    name: str | None = None,
    model: KnownModelName | FallbackModel | None = None,
    do_user_interaction: bool = True,
    use_excel_tools: bool = False,
    use_thinking: bool = False,
    use_memory: bool = False,
    step_limit: int = STEP_LIMIT,
) -> OutputT | TaskResult | RunSQLResult | WriteDataToExcelResult | str:
    workspace_dir = Path(workspace_dir)
    thread_dir = workspace_dir / "threads" / (name or str(uuid4()))
    output = None
    user_prompt = f"Task: {query.query}"
    step = 0
    while not isinstance(output, TaskResult) and step < step_limit:
        output = await thread(
            thread_dir=thread_dir,
            user_prompt=user_prompt,
            model=model,
            use_excel_tools=use_excel_tools,
            use_thinking=use_thinking,
            use_memory=use_memory,
        )
        logger.info(f"Output: {output}")
        if do_user_interaction:
            user_prompt = input(f"{output} > ")
        else:
            user_prompt = (
                "The user is not available to provide input for this task. "
                "Please proceed independently by making reasonable assumptions where needed.\n"
                "Compile all your assumptions, any issues encountered, and your solutions in your final comprehensive response.\n"
                "Push through to completion even if you think you need user clarification - just document what you assumed and why."
            )
        step += 1
    if output is None:
        raise ValueError("Output is None")
    if query.output_type is None:
        return output
    typer_agent = Agent(
        name="typer",
        model="openai:gpt-4.1-nano",
        output_type=query.output_type,
        instructions=f"Convert the result into this format: {query.output_type}",
        instrument=True,
    )
    user_prompt = (
        f"Task: {query.query}\n\nResult: {output.model_dump_json() if not isinstance(output, str) else output}"
    )
    res = await typer_agent.run(user_prompt=user_prompt)
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


async def detect_date_format_and_parse(csv_path: str, column_name: str) -> dict[str, Any]:
    """
    Detect date format and parse dates robustly with timeout protection.

    Returns:
        dict with date_format, min_date, max_date, successful_parse_count, total_count
    """

    async def _detect_with_timeout():
        return _detect_date_format_sync(csv_path, column_name)

    try:
        return await asyncio.wait_for(_detect_with_timeout(), timeout=SQL_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        return {"error": f"Date format detection timed out after {SQL_TIMEOUT_SECONDS} seconds"}
    except Exception as e:
        return {"error": f"Date format detection failed: {str(e)}"}


def _detect_date_format_sync(csv_path: str, column_name: str) -> dict[str, Any]:
    """
    Synchronous date format detection logic.
    """
    from datetime import datetime

    # Common date formats to try
    date_formats = [
        "%Y-%m-%d",  # 2023-01-15
        "%m/%d/%Y",  # 01/15/2023
        "%d/%m/%Y",  # 15/01/2023
        "%Y/%m/%d",  # 2023/01/15
        "%m-%d-%Y",  # 01-15-2023
        "%d-%m-%Y",  # 15-01-2023
        "%Y.%m.%d",  # 2023.01.15
        "%m.%d.%Y",  # 01.15.2023
        "%d.%m.%Y",  # 15.01.2023
        "%Y%m%d",  # 20230115
        "%m/%d/%y",  # 01/15/23
        "%d/%m/%y",  # 15/01/23
        "%y-%m-%d",  # 23-01-15
        "%B %d, %Y",  # January 15, 2023
        "%d %B %Y",  # 15 January 2023
        "%b %d, %Y",  # Jan 15, 2023
        "%d %b %Y",  # 15 Jan 2023
        "%Y-%m-%d %H:%M:%S",  # 2023-01-15 14:30:00
        "%m/%d/%Y %H:%M:%S",  # 01/15/2023 14:30:00
        "%Y-%m-%dT%H:%M:%S",  # 2023-01-15T14:30:00
        "%Y-%m-%dT%H:%M:%SZ",  # 2023-01-15T14:30:00Z
    ]

    try:
        # Get sample of non-null values
        sample_query = f"""
        SELECT DISTINCT "{column_name}" as date_val
        FROM read_csv('{csv_path}')
        WHERE "{column_name}" IS NOT NULL 
          AND TRIM(CAST("{column_name}" AS VARCHAR)) != ''
        LIMIT 100
        """

        sample_result = duckdb.sql(sample_query).fetchdf()  # type:ignore
        if sample_result.empty:  # type:ignore
            return {"error": "No non-null values found"}

        sample_values = sample_result["date_val"].astype(str).tolist()  # type:ignore

        best_format: str | None = None
        best_success_rate = 0.0
        best_parsed_dates: list[datetime] = []

        for date_format in date_formats:
            parsed_dates: list[datetime] = []
            success_count = 0

            for date_str in sample_values[:20]:  # Test on first 20 samples
                try:
                    if date_str and str(date_str).strip():
                        parsed_date = datetime.strptime(str(date_str).strip(), date_format)
                        parsed_dates.append(parsed_date)
                        success_count += 1
                except (ValueError, TypeError):
                    continue

            success_rate = success_count / min(len(sample_values), 20)

            if success_rate > best_success_rate and success_rate >= 0.8:  # At least 80% success
                best_format = date_format
                best_success_rate = success_rate
                best_parsed_dates = parsed_dates

        if not best_format:
            # Try DuckDB's built-in date parsing with different formats
            duckdb_formats = [
                "strptime(\"{col}\", '%Y-%m-%d')",
                "strptime(\"{col}\", '%m/%d/%Y')",
                "strptime(\"{col}\", '%d/%m/%Y')",
                "strptime(\"{col}\", '%Y/%m/%d')",
                "strptime(\"{col}\", '%m-%d-%Y')",
                "strptime(\"{col}\", '%d-%m-%Y')",
                "strptime(\"{col}\", '%Y.%m.%d')",
                "strptime(\"{col}\", '%m.%d.%Y')",
                "strptime(\"{col}\", '%d.%m.%Y')",
                "strptime(\"{col}\", '%Y%m%d')",
                "strptime(\"{col}\", '%B %d, %Y')",
                "strptime(\"{col}\", '%d %B %Y')",
                "strptime(\"{col}\", '%b %d, %Y')",
                "strptime(\"{col}\", '%d %b %Y')",
            ]

            for fmt in duckdb_formats:
                try:
                    test_query = f"""
                    SELECT 
                        MIN({fmt.format(col=column_name)}) as min_date,
                        MAX({fmt.format(col=column_name)}) as max_date,
                        COUNT({fmt.format(col=column_name)}) as success_count,
                        COUNT(*) as total_count
                    FROM read_csv('{csv_path}')
                    WHERE "{column_name}" IS NOT NULL
                    """

                    result = duckdb.sql(test_query).fetchone()  # type:ignore
                    if result and result[0] is not None and result[1] is not None:  # type:ignore
                        return {
                            "date_format": fmt,
                            "duckdb_format": True,
                            "min_date": result[0],  # type:ignore
                            "max_date": result[1],  # type:ignore
                            "successful_parse_count": result[2],  # type:ignore
                            "total_count": result[3],  # type:ignore
                            "success_rate": result[2] / result[3] if result[3] > 0 else 0,  # type:ignore
                        }
                except Exception:
                    continue

            return {"error": "No suitable date format found"}

        # Parse all dates with the best format
        full_parse_query = f"""
        SELECT COUNT(*) as total_count
        FROM read_csv('{csv_path}')
        WHERE "{column_name}" IS NOT NULL
        """

        total_result = duckdb.sql(full_parse_query).fetchone()  # type:ignore
        total_count = total_result[0] if total_result else 0  # type:ignore

        return {
            "date_format": best_format,
            "duckdb_format": False,
            "min_date": min(best_parsed_dates) if best_parsed_dates else None,
            "max_date": max(best_parsed_dates) if best_parsed_dates else None,
            "successful_parse_count": len(best_parsed_dates),
            "total_count": total_count,
            "success_rate": best_success_rate,
            "sample_parsed_dates": [d.isoformat() for d in best_parsed_dates[:5]],
        }

    except Exception as e:
        return {"error": f"Date format detection failed: {str(e)}"}


if __name__ == "__main__":
    name = None
    main_dir = MODULE_DIR.parent.parent
    workspace_dir = main_dir / "workspaces/2"
    setup_workspace(main_dir / "operateai_scenario1_data", workspace_dir, delete_existing=True)
    use_thinking = True
    query = Query[None](query=Path("/Users/hamza/dev/operate-ai/scenario_queries/1_md_format.txt"))
    # query = Query[None](query=Path("final detailed markdown. you already have everything you need."))
    model: KnownModelName | FallbackModel = FallbackModel(
        "anthropic:claude-4-sonnet-20250514",
        "openai:gpt-4.1",
        "google-gla:gemini-2.5-flash-preview-05-20",
        "openai:gpt-4.1-mini",
    )
    asyncio.run(
        run_task(
            query,
            workspace_dir=workspace_dir,
            name=name,
            model=model,
            do_user_interaction=False,
            use_thinking=use_thinking,
        )
    )
