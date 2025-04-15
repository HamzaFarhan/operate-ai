from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic_ai import RunContext

from operate_ai.tools import AgentDeps


def stdev_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, column: str, population: bool = True
) -> dict[str, Any]:
    """
    Calculate the standard deviation for a full population or a sample (Excel STDEV.P/STDEV.S).

    Parameters:
        df_name (str): Name of the CSV file to load.
        column (str): Column to calculate the standard deviation for.
        population (bool): If True, use population formula; else, use sample formula.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "stdev",
            "results": { "stdev": float, "message": str },
            "formula": "=STDEV.P(column) or =STDEV.S(column)"
        }

    Errors:
        Returns an error message in results if the file can't be read or a column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found.")
        data = df[column].dropna()
        if population:
            stdev = float(data.std(ddof=0))
            formula = f"=STDEV.P({column})"
        else:
            stdev = float(data.std(ddof=1))
            formula = f"=STDEV.S({column})"
        return {
            "operation": "stdev",
            "results": {"stdev": stdev, "message": f"Standard deviation of '{column}' is {stdev}."},
            "formula": formula,
        }
    except Exception as e:
        return {
            "operation": "stdev",
            "results": {"error": str(e)},
            "formula": f"=STDEV.P({column}) or =STDEV.S({column})",
        }


def var_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, column: str, population: bool = True
) -> dict[str, Any]:
    """
    Calculate the variance for a population or sample (Excel VAR.P/VAR.S).

    Parameters:
        df_name (str): Name of the CSV file to load.
        column (str): Column to calculate the variance for.
        population (bool): If True, use population formula; else, use sample formula.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "var",
            "results": { "variance": float, "message": str },
            "formula": "=VAR.P(column) or =VAR.S(column)"
        }

    Errors:
        Returns an error message in results if the file can't be read or a column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found.")
        data = df[column].dropna()
        if population:
            variance = float(data.var(ddof=0))
            formula = f"=VAR.P({column})"
        else:
            variance = float(data.var(ddof=1))
            formula = f"=VAR.S({column})"
        return {
            "operation": "var",
            "results": {"variance": variance, "message": f"Variance of '{column}' is {variance}."},
            "formula": formula,
        }
    except Exception as e:
        return {
            "operation": "var",
            "results": {"error": str(e)},
            "formula": f"=VAR.P({column}) or =VAR.S({column})",
        }


def median_formula_tool(ctx: RunContext[AgentDeps], df_name: str, column: str) -> dict[str, Any]:
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
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found.")
        data = df[column].dropna()
        median = float(data.median())
        return {
            "operation": "median",
            "results": {"median": median, "message": f"Median of '{column}' is {median}."},
            "formula": f"=MEDIAN({column})",
        }
    except Exception as e:
        return {"operation": "median", "results": {"error": str(e)}, "formula": f"=MEDIAN({column})"}


def mode_formula_tool(ctx: RunContext[AgentDeps], df_name: str, column: str) -> dict[str, Any]:
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
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found.")
        data = df[column].dropna()
        mode = data.mode().values[0]
        return {
            "operation": "mode",
            "results": {"mode": mode, "message": f"Mode of '{column}' is {mode}."},
            "formula": f"=MODE({column})",
        }
    except Exception as e:
        return {"operation": "mode", "results": {"error": str(e)}, "formula": f"=MODE({column})"}


def correl_formula_tool(ctx: RunContext[AgentDeps], df_name: str, column_x: str, column_y: str) -> dict[str, Any]:
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
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)
        for col in [column_x, column_y]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found.")
        data_x = pd.Series(df[column_x].dropna().astype(float))
        data_y = pd.Series(df[column_y].dropna().astype(float))
        correlation = float(data_x.corr(data_y))
        return {
            "operation": "correl",
            "results": {
                "correlation": correlation,
                "message": f"Correlation between '{column_x}' and '{column_y}' is {correlation}.",
            },
            "formula": f"=CORREL({column_x}, {column_y})",
        }
    except Exception as e:
        return {"operation": "correl", "results": {"error": str(e)}, "formula": f"=CORREL({column_x}, {column_y})"}


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
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)
        for col in [column_x, column_y]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found.")
        data_x = pd.Series(df[column_x].dropna().astype(float))
        data_y = pd.Series(df[column_y].dropna().astype(float))
        if population:
            covariance = float(data_x.cov(data_y, ddof=0))
            formula = f"=COVARIANCE.P({column_x}, {column_y})"
        else:
            covariance = float(data_x.cov(data_y, ddof=1))
            formula = f"=COVARIANCE.S({column_x}, {column_y})"
        return {
            "operation": "covariance",
            "results": {
                "covariance": covariance,
                "message": f"Covariance between '{column_x}' and '{column_y}' is {covariance}.",
            },
            "formula": formula,
        }
    except Exception as e:
        return {
            "operation": "covariance",
            "results": {"error": str(e)},
            "formula": f"=COVARIANCE.P({column_x}, {column_y}) or =COVARIANCE.S({column_x}, {column_y})",
        }


def trend_formula_tool(
    ctx: RunContext[AgentDeps],
    df_name: str,
    y_column: str,
    x_column: str | None = None,
    new_x: list[Any] | None = None,
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
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)

        if y_column not in df.columns:
            raise ValueError(f"Column '{y_column}' not found.")

        y = df[y_column].dropna().astype(float)

        if x_column is None:
            x = np.arange(len(y))
        else:
            if x_column not in df.columns:
                raise ValueError(f"Column '{x_column}' not found.")
            x = df[x_column].dropna().astype(float)

        if new_x is None:
            new_x = list(x)

        # Calculate linear regression
        slope, intercept = np.polyfit(x, y, 1)
        trend_values = slope * np.array(new_x) + intercept

        return {
            "operation": "trend",
            "results": {
                "trend": trend_values.tolist(),
                "message": f"Trend calculated for '{y_column}'",
            },
            "formula": f"=TREND({y_column}{', ' + x_column if x_column else ''}{', ' + str(new_x) if new_x != x else ''})",
        }
    except Exception as e:
        return {
            "operation": "trend",
            "results": {"error": str(e)},
            "formula": f"=TREND({y_column}{', ' + x_column if x_column else ''}{', ' + str(new_x) if new_x else ''})",
        }


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
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)

        for col in [y_column, x_column]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found.")

        y = df[y_column].dropna().astype(float)
        x = df[x_column].dropna().astype(float)

        # Calculate linear regression and predict
        slope, intercept = np.polyfit(x, y, 1)
        forecast = float(slope * float(new_x) + intercept)

        return {
            "operation": "forecast",
            "results": {
                "forecast": forecast,
                "message": f"Forecast for x={new_x} is {forecast}",
            },
            "formula": f"=FORECAST({new_x}, {y_column}, {x_column})",
        }
    except Exception as e:
        return {
            "operation": "forecast",
            "results": {"error": str(e)},
            "formula": f"=FORECAST({new_x}, {y_column}, {x_column})",
        }


def growth_formula_tool(
    ctx: RunContext[AgentDeps],
    df_name: str,
    y_column: str,
    x_column: str | None = None,
    new_x: list[Any] | None = None,
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
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)

        if y_column not in df.columns:
            raise ValueError(f"Column '{y_column}' not found.")

        y = df[y_column].dropna().astype(float)

        if x_column is None:
            x = np.arange(len(y))
        else:
            if x_column not in df.columns:
                raise ValueError(f"Column '{x_column}' not found.")
            x = df[x_column].dropna().astype(float)

        if new_x is None:
            new_x = list(x)

        # Calculate exponential growth model (y = a*e^(b*x))
        log_y = np.log(y)
        slope, intercept = np.polyfit(x, log_y, 1)
        growth_values = np.exp(intercept + slope * np.array(new_x))

        return {
            "operation": "growth",
            "results": {
                "growth": growth_values.tolist(),
                "message": f"Growth forecast for '{y_column}'",
            },
            "formula": f"=GROWTH({y_column}{', ' + x_column if x_column else ''}{', ' + str(new_x) if new_x != x else ''})",
        }
    except Exception as e:
        return {
            "operation": "growth",
            "results": {"error": str(e)},
            "formula": f"=GROWTH({y_column}{', ' + x_column if x_column else ''}{', ' + str(new_x) if new_x else ''})",
        }
