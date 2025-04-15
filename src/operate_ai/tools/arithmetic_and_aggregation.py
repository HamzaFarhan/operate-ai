from pathlib import Path
from typing import Any

import pandas as pd
from pydantic_ai import RunContext

from operate_ai.tools import AgentDeps


def sum_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, column: str, filter_conditions: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Computes the sum of all values in a specified column from a CSV file.

    Parameters:
        df_name (str): Name of the CSV file to load.
        column (str): Column to sum.
        filter_conditions (dict[str, Any] | None): Optional column-value filters.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "sum",
            "results": { "computed_sum": float, "message": str },
            "formula": "=SUM(column)"
        }

    Errors:
        Returns an error message in results if the file can't be read or column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found.")

        if filter_conditions:
            for key, value in filter_conditions.items():
                df = df[df[key] == value]

        computed_sum: float = float(df[column].sum())
        return {
            "operation": "sum",
            "results": {"computed_sum": computed_sum, "message": f"Sum of column '{column}' is {computed_sum}."},
            "formula": f"=SUM({column})",
        }
    except Exception as e:
        return {"operation": "sum", "results": {"error": str(e)}, "formula": f"=SUM({column})"}


def average_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, column: str, filter_conditions: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Computes the average (mean) of all values in a specified column from a CSV file.

    Parameters:
        df_name (str): Name of the CSV file to load.
        column (str): Column to average.
        filter_conditions (dict[str, Any] | None): Optional column-value filters.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "average",
            "results": { "computed_average": float, "message": str },
            "formula": "=AVERAGE(column)"
        }

    Errors:
        Returns an error message in results if the file can't be read or column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found.")

        if filter_conditions:
            for key, value in filter_conditions.items():
                df = df[df[key] == value]

        computed_average: float = float(df[column].mean())
        return {
            "operation": "average",
            "results": {
                "computed_average": computed_average,
                "message": f"Average of column '{column}' is {computed_average}.",
            },
            "formula": f"=AVERAGE({column})",
        }
    except Exception as e:
        return {"operation": "average", "results": {"error": str(e)}, "formula": f"=AVERAGE({column})"}


def min_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, column: str, filter_conditions: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Computes the minimum value in a specified column from a CSV file.

    Parameters:
        df_name (str): Name of the CSV file to load.
        column (str): Column to find the minimum of.
        filter_conditions (dict[str, Any] | None): Optional column-value filters.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "min",
            "results": { "computed_min": float, "message": str },
            "formula": "=MIN(column)"
        }

    Errors:
        Returns an error message in results if the file can't be read or column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found.")

        if filter_conditions:
            for key, value in filter_conditions.items():
                df = df[df[key] == value]

        computed_min: float = float(df[column].min())
        return {
            "operation": "min",
            "results": {
                "computed_min": computed_min,
                "message": f"Minimum of column '{column}' is {computed_min}.",
            },
            "formula": f"=MIN({column})",
        }
    except Exception as e:
        return {"operation": "min", "results": {"error": str(e)}, "formula": f"=MIN({column})"}


def max_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, column: str, filter_conditions: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Computes the maximum value in a specified column from a CSV file.

    Parameters:
        df_name (str): Name of the CSV file to load.
        column (str): Column to find the maximum of.
        filter_conditions (dict[str, Any] | None): Optional column-value filters.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "max",
            "results": { "computed_max": float, "message": str },
            "formula": "=MAX(column)"
        }

    Errors:
        Returns an error message in results if the file can't be read or column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found.")

        if filter_conditions:
            for key, value in filter_conditions.items():
                df = df[df[key] == value]

        computed_max: float = float(df[column].max())
        return {
            "operation": "max",
            "results": {
                "computed_max": computed_max,
                "message": f"Maximum of column '{column}' is {computed_max}.",
            },
            "formula": f"=MAX({column})",
        }
    except Exception as e:
        return {"operation": "max", "results": {"error": str(e)}, "formula": f"=MAX({column})"}


def product_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, column: str, filter_conditions: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Computes the product (multiplication) of all values in a specified column from a CSV file.

    Parameters:
        df_name (str): Name of the CSV file to load.
        column (str): Column to multiply.
        filter_conditions (dict[str, Any] | None): Optional column-value filters.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "product",
            "results": { "computed_product": float, "message": str },
            "formula": "=PRODUCT(column)"
        }

    Errors:
        Returns an error message in results if the file can't be read or column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found.")

        if filter_conditions:
            for key, value in filter_conditions.items():
                df = df[df[key] == value]

        computed_product: float = float(df[column].prod())
        return {
            "operation": "product",
            "results": {
                "computed_product": computed_product,
                "message": f"Product of column '{column}' is {computed_product}.",
            },
            "formula": f"=PRODUCT({column})",
        }
    except Exception as e:
        return {"operation": "product", "results": {"error": str(e)}, "formula": f"=PRODUCT({column})"}
