from pathlib import Path
from typing import Any

import pandas as pd
from pydantic_ai import RunContext

from operate_ai.tools import AgentDeps


def sumif_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, sum_column: str, criteria_column: str, criteria_value: Any
) -> dict[str, Any]:
    """
    Computes the sum of values in a specified column for rows that meet a given condition (like Excel SUMIF).

    Parameters:
        df_name (str): Name of the CSV file to load.
        sum_column (str): Column whose values will be summed.
        criteria_column (str): Column to apply the condition on.
        criteria_value (Any): Value to match in the criteria_column.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "sumif",
            "results": { "computed_sum": float, "message": str },
            "formula": "=SUMIF(criteria_column, criteria_value, sum_column)"
        }

    Errors:
        Returns an error message in results if the file can't be read or a column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)

        if sum_column not in df.columns:
            raise ValueError(f"Sum column '{sum_column}' not found.")
        if criteria_column not in df.columns:
            raise ValueError(f"Criteria column '{criteria_column}' not found.")

        filtered = df[df[criteria_column] == criteria_value]
        computed_sum: float = float(filtered[sum_column].sum())
        return {
            "operation": "sumif",
            "results": {
                "computed_sum": computed_sum,
                "message": f"Sum of '{sum_column}' where '{criteria_column}' == {criteria_value} is {computed_sum}.",
            },
            "formula": f"=SUMIF({criteria_column}, {criteria_value}, {sum_column})",
        }
    except Exception as e:
        return {
            "operation": "sumif",
            "results": {"error": str(e)},
            "formula": f"=SUMIF({criteria_column}, {criteria_value}, {sum_column})",
        }


def countif_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, criteria_column: str, criteria_value: Any
) -> dict[str, Any]:
    """
    Counts the number of rows where a specified column matches a given value (like Excel COUNTIF).

    Parameters:
        df_name (str): Name of the CSV file to load.
        criteria_column (str): Column to apply the condition on.
        criteria_value (Any): Value to match in the criteria_column.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "countif",
            "results": { "computed_count": int, "message": str },
            "formula": "=COUNTIF(criteria_column, criteria_value)"
        }

    Errors:
        Returns an error message in results if the file can't be read or a column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)

        if criteria_column not in df.columns:
            raise ValueError(f"Criteria column '{criteria_column}' not found.")

        computed_count: int = int((df[criteria_column] == criteria_value).sum())
        return {
            "operation": "countif",
            "results": {
                "computed_count": computed_count,
                "message": f"Count of rows where '{criteria_column}' == {criteria_value} is {computed_count}.",
            },
            "formula": f"=COUNTIF({criteria_column}, {criteria_value})",
        }
    except Exception as e:
        return {
            "operation": "countif",
            "results": {"error": str(e)},
            "formula": f"=COUNTIF({criteria_column}, {criteria_value})",
        }


def averageif_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, average_column: str, criteria_column: str, criteria_value: Any
) -> dict[str, Any]:
    """
    Computes the average of values in a specified column for rows that meet a given condition (like Excel AVERAGEIF).

    Parameters:
        df_name (str): Name of the CSV file to load.
        average_column (str): Column whose values will be averaged.
        criteria_column (str): Column to apply the condition on.
        criteria_value (Any): Value to match in the criteria_column.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "averageif",
            "results": { "computed_average": float, "message": str },
            "formula": "=AVERAGEIF(criteria_column, criteria_value, average_column)"
        }

    Errors:
        Returns an error message in results if the file can't be read or a column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)

        if average_column not in df.columns:
            raise ValueError(f"Average column '{average_column}' not found.")
        if criteria_column not in df.columns:
            raise ValueError(f"Criteria column '{criteria_column}' not found.")

        filtered = df[df[criteria_column] == criteria_value]
        computed_average: float = float(filtered[average_column].mean())
        return {
            "operation": "averageif",
            "results": {
                "computed_average": computed_average,
                "message": f"Average of '{average_column}' where '{criteria_column}' == {criteria_value} is {computed_average}.",
            },
            "formula": f"=AVERAGEIF({criteria_column}, {criteria_value}, {average_column})",
        }
    except Exception as e:
        return {
            "operation": "averageif",
            "results": {"error": str(e)},
            "formula": f"=AVERAGEIF({criteria_column}, {criteria_value}, {average_column})",
        }
