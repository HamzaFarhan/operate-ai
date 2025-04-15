from pathlib import Path
from typing import Any

import pandas as pd
from pydantic_ai import RunContext

from operate_ai.tools import AgentDeps


def stdev_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, column: str, population: bool = True
) -> dict[str, Any]:
    """
    Calculate the standard deviation for a full population or a sample (Excel STDEV.P/STDEV.S).
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found.")
        values = df[column].dropna()
        if not pd.api.types.is_numeric_dtype(values):
            raise ValueError(f"Column '{column}' must be numeric.")
        # pandas std: ddof=0 for population, ddof=1 for sample
        ddof = 0 if population else 1
        stdev = float(values.std(ddof=ddof))
        formula = f"=STDEV.{'P' if population else 'S'}({column})"
        return {
            "operation": "stdev",
            "results": {
                "stdev": stdev,
                "message": f"Standard deviation of column '{column}' is {stdev}."
            },
            "formula": formula
        }
    except Exception as e:
        formula = f"=STDEV.{'P' if population else 'S'}({column})"
        return {
            "operation": "stdev",
            "results": {"error": str(e)},
            "formula": formula
        }


def var_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, column: str, population: bool = True
) -> dict[str, Any]:
    """
    Calculate the variance for a population or sample (Excel VAR.P/VAR.S).
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found.")
        values = df[column].dropna()
        if not pd.api.types.is_numeric_dtype(values):
            raise ValueError(f"Column '{column}' must be numeric.")
        # pandas var: ddof=0 for population, ddof=1 for sample
        ddof = 0 if population else 1
        variance = float(values.var(ddof=ddof))
        formula = f"=VAR.{'P' if population else 'S'}({column})"
        return {
            "operation": "var",
            "results": {
                "variance": variance,
                "message": f"Variance of column '{column}' is {variance}."
            },
            "formula": formula
        }
    except Exception as e:
        formula = f"=VAR.{'P' if population else 'S'}({column})"
        return {
            "operation": "var",
            "results": {"error": str(e)},
            "formula": formula
        }


def median_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, column: str
) -> dict[str, Any]:
    """
    Determine the median (middle value) in a dataset (Excel MEDIAN).

    Parameters:
        df_name (str): Name of the CSV file to load.
        column (str): Column to calculate the median for.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "median",
            "results": { "median": float, "message": str },
            "formula": "=MEDIAN(column)"
        }

    Errors:
        Returns an error message in results if the file can't be read or a column is missing.
    """
    pass


def mode_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, column: str
) -> dict[str, Any]:
    """
    Determine the mode (most frequent value) in a dataset (Excel MODE).

    Parameters:
        df_name (str): Name of the CSV file to load.
        column (str): Column to calculate the mode for.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "mode",
            "results": { "mode": Any, "message": str },
            "formula": "=MODE(column)"
        }

    Errors:
        Returns an error message in results if the file can't be read or a column is missing.
    """
    pass


def correl_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, column_x: str, column_y: str
) -> dict[str, Any]:
    """
    Measure the correlation between two datasets (Excel CORREL).

    Parameters:
        df_name (str): Name of the CSV file to load.
        column_x (str): First column.
        column_y (str): Second column.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "correl",
            "results": { "correlation": float, "message": str },
            "formula": "=CORREL(column_x, column_y)"
        }

    Errors:
        Returns an error message in results if the file can't be read or columns are missing.
    """
    pass


def covariance_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, column_x: str, column_y: str, population: bool = True
) -> dict[str, Any]:
    """
    Measure the covariance between two datasets (Excel COVARIANCE.P/COVARIANCE.S).

    Parameters:
        df_name (str): Name of the CSV file to load.
        column_x (str): First column.
        column_y (str): Second column.
        population (bool): If True, use population formula; else, use sample formula.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "covariance",
            "results": { "covariance": float, "message": str },
            "formula": "=COVARIANCE.P(column_x, column_y) or =COVARIANCE.S(column_x, column_y)"
        }

    Errors:
        Returns an error message in results if the file can't be read or columns are missing.
    """
    pass


def trend_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, y_column: str, x_column: str = None, new_x: list[Any] = None
) -> dict[str, Any]:
    """
    Predict future values based on historical data trends (Excel TREND).

    Parameters:
        df_name (str): Name of the CSV file to load.
        y_column (str): Dependent variable column.
        x_column (str, optional): Independent variable column.
        new_x (list, optional): New x values to predict y for.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "trend",
            "results": { "trend": list, "message": str },
            "formula": "=TREND(y_column, [x_column], [new_x])"
        }

    Errors:
        Returns an error message in results if the file can't be read or columns are missing.
    """
    pass


def forecast_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, y_column: str, x_column: str, new_x: Any
) -> dict[str, Any]:
    """
    Predict a future value based on historical data (Excel FORECAST/FORECAST.LINEAR).

    Parameters:
        df_name (str): Name of the CSV file to load.
        y_column (str): Dependent variable column.
        x_column (str): Independent variable column.
        new_x (Any): The new x value to predict y for.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "forecast",
            "results": { "forecast": float, "message": str },
            "formula": "=FORECAST(new_x, y_column, x_column)"
        }

    Errors:
        Returns an error message in results if the file can't be read or columns are missing.
    """
    pass


def growth_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, y_column: str, x_column: str = None, new_x: list[Any] = None
) -> dict[str, Any]:
    """
    Forecast exponential growth trends (Excel GROWTH).

    Parameters:
        df_name (str): Name of the CSV file to load.
        y_column (str): Dependent variable column.
        x_column (str, optional): Independent variable column.
        new_x (list, optional): New x values to predict y for.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "growth",
            "results": { "growth": list, "message": str },
            "formula": "=GROWTH(y_column, [x_column], [new_x])"
        }

    Errors:
        Returns an error message in results if the file can't be read or columns are missing.
    """
    pass

