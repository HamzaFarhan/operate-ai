from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic_ai import RunContext

from operate_ai.tools import AgentDeps


def unique_formula_tool(ctx: RunContext[AgentDeps], df_name: str, column_name: str) -> dict[str, Any]:
    """
    Extracts a list of unique values from a specified column in a CSV file (like Excel UNIQUE).

    Parameters:
        df_name (str): Name of the CSV file to load.
        column_name (str): Column from which to extract unique values.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "unique",
            "results": { "unique_values": List[Any], "count": int, "message": str },
            "formula": "=UNIQUE(column_name)"
        }

    Errors:
        Returns an error message in results if the file can't be read or the column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)

        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found.")

        unique_values = df[column_name].unique().tolist()
        count = len(unique_values)

        return {
            "operation": "unique",
            "results": {
                "unique_values": unique_values,
                "count": count,
                "message": f"Found {count} unique values in '{column_name}'.",
            },
            "formula": f"=UNIQUE({column_name})",
        }
    except Exception as e:
        return {
            "operation": "unique",
            "results": {"error": str(e)},
            "formula": f"=UNIQUE({column_name})",
        }


def sort_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, column_name: str, ascending: bool = True
) -> dict[str, Any]:
    """
    Sorts data in a specified column of a CSV file (like Excel SORT).

    Parameters:
        df_name (str): Name of the CSV file to load.
        column_name (str): Column to sort by.
        ascending (bool, optional): Whether to sort in ascending order. Defaults to True.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "sort",
            "results": { "sorted_data": List[dict], "message": str },
            "formula": "=SORT(range, [sort_index], [sort_order])"
        }

    Errors:
        Returns an error message in results if the file can't be read or the column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)

        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found.")

        sorted_df = df.sort_values(by=column_name, ascending=ascending)
        sorted_data = sorted_df.to_dict("records")

        order_text = "ascending" if ascending else "descending"
        return {
            "operation": "sort",
            "results": {
                "sorted_data": sorted_data,
                "message": f"Data sorted by '{column_name}' in {order_text} order.",
            },
            "formula": f"=SORT({df_name}, {column_name}, {1 if ascending else -1})",
        }
    except Exception as e:
        return {
            "operation": "sort",
            "results": {"error": str(e)},
            "formula": f"=SORT({df_name}, {column_name}, {1 if ascending else -1})",
        }


def sortby_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, sort_by_column: str, ascending: bool = True
) -> dict[str, Any]:
    """
    Sorts data in a CSV file by a specified column (like Excel SORTBY).

    Parameters:
        df_name (str): Name of the CSV file to load.
        sort_by_column (str): Column to sort by.
        ascending (bool, optional): Whether to sort in ascending order. Defaults to True.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "sortby",
            "results": { "sorted_data": List[dict], "message": str },
            "formula": "=SORTBY(array, by_array, [sort_order])"
        }

    Errors:
        Returns an error message in results if the file can't be read or the column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)

        if sort_by_column not in df.columns:
            raise ValueError(f"Column '{sort_by_column}' not found.")

        sorted_df = df.sort_values(by=sort_by_column, ascending=ascending)
        sorted_data = sorted_df.to_dict("records")

        order_text = "ascending" if ascending else "descending"
        return {
            "operation": "sortby",
            "results": {
                "sorted_data": sorted_data,
                "message": f"Data sorted by '{sort_by_column}' in {order_text} order.",
            },
            "formula": f"=SORTBY({df_name}, {sort_by_column}, {1 if ascending else -1})",
        }
    except Exception as e:
        return {
            "operation": "sortby",
            "results": {"error": str(e)},
            "formula": f"=SORTBY({df_name}, {sort_by_column}, {1 if ascending else -1})",
        }


def filter_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, filter_column: str, filter_value: Any
) -> dict[str, Any]:
    """
    Filters data in a CSV file based on a condition (like Excel FILTER).

    Parameters:
        df_name (str): Name of the CSV file to load.
        filter_column (str): Column to apply the filter on.
        filter_value (Any): Value to filter by (rows where filter_column equals this value will be returned).
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "filter",
            "results": { "filtered_data": List[dict], "count": int, "message": str },
            "formula": "=FILTER(range, condition)"
        }

    Errors:
        Returns an error message in results if the file can't be read or the column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)

        if filter_column not in df.columns:
            raise ValueError(f"Column '{filter_column}' not found.")

        filtered_df = df[df[filter_column] == filter_value]
        filtered_data = filtered_df.to_dict("records")  # type: ignore
        count = len(filtered_data)

        return {
            "operation": "filter",
            "results": {
                "filtered_data": filtered_data,
                "count": count,
                "message": f"Found {count} rows where '{filter_column}' equals {filter_value}.",
            },
            "formula": f"=FILTER({df_name}, {filter_column}={filter_value})",
        }
    except Exception as e:
        return {
            "operation": "filter",
            "results": {"error": str(e)},
            "formula": f"=FILTER({df_name}, {filter_column}={filter_value})",
        }


def sequence_formula_tool(rows: int, columns: int = 1, start: int = 1, step: int = 1) -> dict[str, Any]:
    """
    Generates a sequence of numbers in an array format (like Excel SEQUENCE).

    Parameters:
        rows (int): Number of rows in the sequence.
        columns (int, optional): Number of columns in the sequence. Defaults to 1.
        start (int, optional): Starting value of the sequence. Defaults to 1.
        step (int, optional): Step value for the sequence. Defaults to 1.
        ctx (RunContext): Runtime context.

    Returns:
        dict[str, Any]: {
            "operation": "sequence",
            "results": { "sequence": List[List[int]], "message": str },
            "formula": "=SEQUENCE(rows, [columns], [start], [step])"
        }

    Errors:
        Returns an error message in results if invalid parameters are provided.
    """
    try:
        if rows <= 0 or columns <= 0:
            raise ValueError("Rows and columns must be positive integers.")

        sequence = []
        current_value = start

        for i in range(rows):
            row = []
            for j in range(columns):
                row.append(current_value)
                current_value += step
            sequence.append(row)

        return {
            "operation": "sequence",
            "results": {
                "sequence": sequence,
                "message": f"Generated sequence with {rows} rows and {columns} columns, starting at {start} with step {step}.",
            },
            "formula": f"=SEQUENCE({rows}, {columns}, {start}, {step})",
        }
    except Exception as e:
        return {
            "operation": "sequence",
            "results": {"error": str(e)},
            "formula": f"=SEQUENCE({rows}, {columns}, {start}, {step})",
        }


def rand_formula_tool(count: int = 1) -> dict[str, Any]:
    """
    Generates random numbers between 0 and 1 (like Excel RAND).

    Parameters:
        count (int, optional): Number of random values to generate. Defaults to 1.
        ctx (RunContext): Runtime context.

    Returns:
        dict[str, Any]: {
            "operation": "rand",
            "results": { "random_values": List[float], "message": str },
            "formula": "=RAND()"
        }

    Errors:
        Returns an error message in results if invalid parameters are provided.
    """
    try:
        if count <= 0:
            raise ValueError("Count must be a positive integer.")

        random_values = np.random.random(count).tolist()

        return {
            "operation": "rand",
            "results": {
                "random_values": random_values,
                "message": f"Generated {count} random values between 0 and 1.",
            },
            "formula": "=RAND()",
        }
    except Exception as e:
        return {
            "operation": "rand",
            "results": {"error": str(e)},
            "formula": "=RAND()",
        }


def randbetween_formula_tool(lower: int, upper: int, count: int = 1) -> dict[str, Any]:
    """
    Generates random integers between specified lower and upper bounds (like Excel RANDBETWEEN).

    Parameters:
        lower (int): Lower bound (inclusive).
        upper (int): Upper bound (inclusive).
        count (int, optional): Number of random values to generate. Defaults to 1.
        ctx (RunContext): Runtime context.

    Returns:
        dict[str, Any]: {
            "operation": "randbetween",
            "results": { "random_values": List[int], "message": str },
            "formula": "=RANDBETWEEN(lower, upper)"
        }

    Errors:
        Returns an error message in results if invalid parameters are provided.
    """
    try:
        if lower > upper:
            raise ValueError("Lower bound must be less than or equal to upper bound.")
        if count <= 0:
            raise ValueError("Count must be a positive integer.")

        random_values = np.random.randint(lower, upper + 1, count).tolist()

        return {
            "operation": "randbetween",
            "results": {
                "random_values": random_values,
                "message": f"Generated {count} random integers between {lower} and {upper}.",
            },
            "formula": f"=RANDBETWEEN({lower}, {upper})",
        }
    except Exception as e:
        return {
            "operation": "randbetween",
            "results": {"error": str(e)},
            "formula": f"=RANDBETWEEN({lower}, {upper})",
        }
