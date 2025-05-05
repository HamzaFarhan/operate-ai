from pathlib import Path
from typing import Any

import pandas as pd
from pydantic_ai import RunContext

from operate_ai.excel_tools import AgentDeps


def if_formula_tool(
    ctx: RunContext[AgentDeps],
    df_name: str,
    condition_column: str,
    condition_operator: str,
    condition_value: Any,
    true_value: Any,
    false_value: Any,
) -> dict[str, Any]:
    """
    Returns different values depending on whether a condition is met (like Excel IF).

    Parameters:
        df_name (str): Name of the CSV file to load.
        condition_column (str): Column to test the condition on.
        condition_operator (str): Operator for the condition (e.g., '>', '<', '==', '!=', '>=', '<=').
        condition_value (Any): Value to compare against.
        true_value (Any): Value to return if the condition is True.
        false_value (Any): Value to return if the condition is False.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "if",
            "results": { "values": list, "message": str },
            "formula": "=IF(condition, true_value, false_value)"
        }

    Errors:
        Returns an error message in results if the file can't be read, column is missing, or condition is invalid.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)
        if condition_column not in df.columns:
            raise ValueError(f"Column '{condition_column}' not found.")
        # Evaluate condition
        ops = {
            ">": lambda x: x > condition_value,
            "<": lambda x: x < condition_value,
            "==": lambda x: x == condition_value,
            "!=": lambda x: x != condition_value,
            ">=": lambda x: x >= condition_value,
            "<=": lambda x: x <= condition_value,
        }
        if condition_operator not in ops:
            raise ValueError(f"Unsupported operator '{condition_operator}'.")
        mask = ops[condition_operator](df[condition_column])
        values = [true_value if cond else false_value for cond in mask]
        return {
            "operation": "if",
            "results": {
                "values": values,
                "message": f"IF applied on column '{condition_column}' with operator '{condition_operator}' and value '{condition_value}'.",
            },
            "formula": f"=IF({condition_column} {condition_operator} {condition_value}, {true_value}, {false_value})",
        }
    except Exception as e:
        return {
            "operation": "if",
            "results": {"error": str(e)},
            "formula": f"=IF({condition_column} {condition_operator} {condition_value}, {true_value}, {false_value})",
        }


def iferror_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, formula_column: str, alternative_value: Any
) -> dict[str, Any]:
    """
    Returns a specified value if a formula results in an error (like Excel IFERROR).

    Parameters:
        df_name (str): Name of the CSV file to load.
        formula_column (str): Column containing the formula or values to check for errors.
        alternative_value (Any): Value to return if an error is detected.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "iferror",
            "results": { "values": list, "message": str },
            "formula": "=IFERROR(formula, alternative_value)"
        }

    Errors:
        Returns an error message in results if the file can't be read or column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)
        if formula_column not in df.columns:
            raise ValueError(f"Column '{formula_column}' not found.")
        values = []
        for v in df[formula_column]:
            try:
                if pd.isna(v):
                    raise ValueError("NaN")
                values.append(v)
            except Exception:
                values.append(alternative_value)
        return {
            "operation": "iferror",
            "results": {"values": values, "message": f"IFERROR applied on column '{formula_column}'."},
            "formula": f"=IFERROR({formula_column}, {alternative_value})",
        }
    except Exception as e:
        return {
            "operation": "iferror",
            "results": {"error": str(e)},
            "formula": f"=IFERROR({formula_column}, {alternative_value})",
        }


def ifna_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, formula_column: str, alternative_value: Any
) -> dict[str, Any]:
    """
    Returns a specified value if a formula results in #N/A (like Excel IFNA).

    Parameters:
        df_name (str): Name of the CSV file to load.
        formula_column (str): Column containing the formula or values to check for #N/A.
        alternative_value (Any): Value to return if #N/A is detected.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "ifna",
            "results": { "values": list, "message": str },
            "formula": "=IFNA(formula, alternative_value)"
        }

    Errors:
        Returns an error message in results if the file can't be read or column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)
        if formula_column not in df.columns:
            raise ValueError(f"Column '{formula_column}' not found.")
        values = []
        for v in df[formula_column]:
            if pd.isna(v):
                values.append(alternative_value)
            else:
                values.append(v)
        return {
            "operation": "ifna",
            "results": {"values": values, "message": f"IFNA applied on column '{formula_column}'."},
            "formula": f"=IFNA({formula_column}, {alternative_value})",
        }
    except Exception as e:
        return {
            "operation": "ifna",
            "results": {"error": str(e)},
            "formula": f"=IFNA({formula_column}, {alternative_value})",
        }


def ifs_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, conditions: list[tuple[str, str, Any, Any]], default_value: Any
) -> dict[str, Any]:
    """
    Tests multiple conditions without nesting IF statements (like Excel IFS).

    Parameters:
        df_name (str): Name of the CSV file to load.
        conditions (list[tuple[str, str, Any, Any]]): Each tuple is (column, operator, value, result_value).
        default_value (Any): Value to return if no conditions are met.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "ifs",
            "results": { "values": list, "message": str },
            "formula": "=IFS(condition1, value1, condition2, value2, ..., TRUE, default_value)"
        }

    Errors:
        Returns an error message in results if the file can't be read, column is missing, or conditions are invalid.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)
        # Validate columns
        for col in [c[0] for c in conditions]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found.")
        results = []
        for _, row in df.iterrows():
            matched = False
            for col, op, val, result_val in conditions:
                if op not in [">", "<", "==", "!=", ">=", "<="]:
                    raise ValueError(f"Unsupported operator '{op}'.")
                if eval(f"row[col] {op} val"):
                    results.append(result_val)
                    matched = True
                    break
            if not matched:
                results.append(default_value)
        return {
            "operation": "ifs",
            "results": {
                "values": results,
                "message": f"IFS applied with {len(conditions)} conditions and default value.",
            },
            "formula": "=IFS(...)",
        }
    except Exception as e:
        return {"operation": "ifs", "results": {"error": str(e)}, "formula": "=IFS(...)"}


def and_formula_tool(
    ctx: RunContext[AgentDeps],
    df_name: str,
    condition_columns: list[str],
    condition_operators: list[str],
    condition_values: list[Any],
) -> dict[str, Any]:
    """
    Combines multiple logical conditions (like Excel AND).

    Parameters:
        df_name (str): Name of the CSV file to load.
        condition_columns (list[str]): Columns to test conditions on.
        condition_operators (list[str]): Operators for each condition.
        condition_values (list[Any]): Values to compare against for each condition.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "and",
            "results": { "values": list, "message": str },
            "formula": "=AND(condition1, condition2, ...)"
        }

    Errors:
        Returns an error message in results if the file can't be read, columns are missing, or conditions are invalid.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)
        # Validate columns
        for col in condition_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found.")
        if not (len(condition_columns) == len(condition_operators) == len(condition_values)):
            raise ValueError(
                "condition_columns, condition_operators, and condition_values must have the same length."
            )
        results = []
        for _, row in df.iterrows():
            res = True
            for col, op, val in zip(condition_columns, condition_operators, condition_values):
                if op not in [">", "<", "==", "!=", ">=", "<="]:
                    raise ValueError(f"Unsupported operator '{op}'.")
                if not eval(f"row[col] {op} val"):
                    res = False
                    break
            results.append(res)
        return {
            "operation": "and",
            "results": {"values": results, "message": f"AND applied across {len(condition_columns)} conditions."},
            "formula": "=AND(...)",
        }
    except Exception as e:
        return {"operation": "and", "results": {"error": str(e)}, "formula": "=AND(...)"}


def or_formula_tool(
    ctx: RunContext[AgentDeps],
    df_name: str,
    condition_columns: list[str],
    condition_operators: list[str],
    condition_values: list[Any],
) -> dict[str, Any]:
    """
    Combines multiple logical conditions, returning True if any are met (like Excel OR).

    Parameters:
        df_name (str): Name of the CSV file to load.
        condition_columns (list[str]): Columns to test conditions on.
        condition_operators (list[str]): Operators for each condition.
        condition_values (list[Any]): Values to compare against for each condition.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "or",
            "results": { "values": list, "message": str },
            "formula": "=OR(condition1, condition2, ...)"
        }

    Errors:
        Returns an error message in results if the file can't be read, columns are missing, or conditions are invalid.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)
        # Validate columns
        for col in condition_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found.")
        if not (len(condition_columns) == len(condition_operators) == len(condition_values)):
            raise ValueError(
                "condition_columns, condition_operators, and condition_values must have the same length."
            )
        results = []
        for _, row in df.iterrows():
            res = False
            for col, op, val in zip(condition_columns, condition_operators, condition_values):
                if op not in [">", "<", "==", "!=", ">=", "<="]:
                    raise ValueError(f"Unsupported operator '{op}'.")
                if eval(f"row[col] {op} val"):
                    res = True
                    break
            results.append(res)
        return {
            "operation": "or",
            "results": {"values": results, "message": f"OR applied across {len(condition_columns)} conditions."},
            "formula": "=OR(...)",
        }
    except Exception as e:
        return {"operation": "or", "results": {"error": str(e)}, "formula": "=OR(...)"}


def not_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, condition_column: str, condition_operator: str, condition_value: Any
) -> dict[str, Any]:
    """
    Modifies a logical condition, returning the opposite boolean result (like Excel NOT).

    Parameters:
        df_name (str): Name of the CSV file to load.
        condition_column (str): Column to test the condition on.
        condition_operator (str): Operator for the condition.
        condition_value (Any): Value to compare against.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "not",
            "results": { "values": list, "message": str },
            "formula": "=NOT(condition)"
        }

    Errors:
        Returns an error message in results if the file can't be read, column is missing, or condition is invalid.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)
        if condition_column not in df.columns:
            raise ValueError(f"Column '{condition_column}' not found.")
        ops = {
            ">": lambda x: x > condition_value,
            "<": lambda x: x < condition_value,
            "==": lambda x: x == condition_value,
            "!=": lambda x: x != condition_value,
            ">=": lambda x: x >= condition_value,
            "<=": lambda x: x <= condition_value,
        }
        if condition_operator not in ops:
            raise ValueError(f"Unsupported operator '{condition_operator}'.")
        mask = ops[condition_operator](df[condition_column])
        values = [not cond for cond in mask]
        return {
            "operation": "not",
            "results": {
                "values": values,
                "message": f"NOT applied on column '{condition_column}' with operator '{condition_operator}' and value '{condition_value}'.",
            },
            "formula": f"=NOT({condition_column} {condition_operator} {condition_value})",
        }
    except Exception as e:
        return {
            "operation": "not",
            "results": {"error": str(e)},
            "formula": f"=NOT({condition_column} {condition_operator} {condition_value})",
        }
