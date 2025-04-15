from pathlib import Path
from typing import Any

import pandas as pd
from pydantic_ai import RunContext

from operate_ai.tools import AgentDeps


def vlookup_formula_tool(
    ctx: RunContext[AgentDeps],
    df_name: str,
    lookup_value: Any,
    lookup_column: str,
    return_column: str,
    range_lookup: bool = False,
) -> dict[str, Any]:
    """
    Searches for a value in the lookup_column and returns the corresponding value from the return_column (like Excel VLOOKUP).

    Parameters:
        df_name (str): Name of the CSV file to load.
        lookup_value (Any): Value to search for in the lookup_column.
        lookup_column (str): Column to search for the lookup_value.
        return_column (str): Column from which to return the value.
        range_lookup (bool): Whether to allow approximate match (default False, exact match).
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "vlookup",
            "results": { "matched_value": Any, "message": str },
            "formula": "=VLOOKUP(lookup_value, table_array, col_index, [range_lookup])"
        }

    Errors:
        Returns an error message in results if the file can't be read or a column is missing or no match is found.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)
        if lookup_column not in df.columns:
            raise ValueError(f"Lookup column '{lookup_column}' not found.")
        if return_column not in df.columns:
            raise ValueError(f"Return column '{return_column}' not found.")
        if range_lookup:
            # Approximate match: find the last row where lookup_column <= lookup_value
            sorted_df = df.sort_values(by=lookup_column)
            candidates = sorted_df[sorted_df[lookup_column] <= lookup_value]
            if candidates.empty:
                raise ValueError(
                    f"No approximate match found for value '{lookup_value}' in column '{lookup_column}'."
                )
            matched_row = candidates.iloc[-1]
        else:
            matches = df[df[lookup_column] == lookup_value]
            if matches.empty:
                raise ValueError(f"No exact match found for value '{lookup_value}' in column '{lookup_column}'.")
            matched_row = matches.iloc[0]
        matched_value = matched_row[return_column]
        return {
            "operation": "vlookup",
            "results": {
                "matched_value": matched_value,
                "message": f"Found value '{matched_value}' in column '{return_column}' for lookup '{lookup_value}'.",
            },
            "formula": f"=VLOOKUP({lookup_value}, [table], [col_index], {range_lookup})",
        }
    except Exception as e:
        return {
            "operation": "vlookup",
            "results": {"error": str(e)},
            "formula": f"=VLOOKUP({lookup_value}, [table], [col_index], {range_lookup})",
        }


def hlookup_formula_tool(
    ctx: RunContext[AgentDeps],
    df_name: str,
    lookup_value: Any,
    lookup_row: int,
    return_row: int,
    range_lookup: bool = False,
) -> dict[str, Any]:
    """
    Searches for a value in the specified lookup_row and returns the corresponding value from the return_row (like Excel HLOOKUP).

    Parameters:
        df_name (str): Name of the CSV file to load.
        lookup_value (Any): Value to search for in the lookup_row.
        lookup_row (int): Index of the row to search for the lookup_value (0-based).
        return_row (int): Index of the row from which to return the value (0-based).
        range_lookup (bool): Whether to allow approximate match (default False, exact match).
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "hlookup",
            "results": { "matched_value": Any, "message": str },
            "formula": "=HLOOKUP(lookup_value, table_array, row_index, [range_lookup])"
        }

    Errors:
        Returns an error message in results if the file can't be read or no match is found.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path, header=None)
        if lookup_row >= len(df.index):
            raise ValueError(f"Lookup row index {lookup_row} out of bounds.")
        if return_row >= len(df.index):
            raise ValueError(f"Return row index {return_row} out of bounds.")
        lookup_row_values = df.iloc[lookup_row]
        if range_lookup:
            # Approximate match: find the last column where value <= lookup_value
            sorted_cols = lookup_row_values.sort_values()
            candidates = sorted_cols[sorted_cols <= lookup_value]
            if candidates.empty:
                raise ValueError(f"No approximate match found for value '{lookup_value}' in row {lookup_row}.")
            matched_col = candidates.index[-1]
        else:
            matches = lookup_row_values[lookup_row_values == lookup_value]
            if matches.empty:
                raise ValueError(f"No exact match found for value '{lookup_value}' in row {lookup_row}.")
            matched_col = matches.index[0]
        matched_value = df.iloc[return_row, matched_col]
        return {
            "operation": "hlookup",
            "results": {
                "matched_value": matched_value,
                "message": f"Found value '{matched_value}' at row {return_row}, column {matched_col} for lookup '{lookup_value}'.",
            },
            "formula": f"=HLOOKUP({lookup_value}, [table], {return_row}, {range_lookup})",
        }
    except Exception as e:
        return {
            "operation": "hlookup",
            "results": {"error": str(e)},
            "formula": f"=HLOOKUP({lookup_value}, [table], {return_row}, {range_lookup})",
        }


def index_formula_tool(ctx: RunContext[AgentDeps], df_name: str, row: int, column: int) -> dict[str, Any]:
    """
    Returns the value at a given row and column index (like Excel INDEX).

    Parameters:
        df_name (str): Name of the CSV file to load.
        row (int): Row index (0-based).
        column (int): Column index (0-based).
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "index",
            "results": { "value": Any, "message": str },
            "formula": "=INDEX(return_range, row, column)"
        }

    Errors:
        Returns an error message in results if the file can't be read or indices are out of bounds.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)
        if row < 0 or row >= len(df.index):
            raise ValueError(f"Row index {row} out of bounds.")
        if column < 0 or column >= len(df.columns):
            raise ValueError(f"Column index {column} out of bounds.")
        value = df.iloc[row, column]
        return {
            "operation": "index",
            "results": {"value": value, "message": f"Value at row {row}, column {column} is '{value}'."},
            "formula": f"=INDEX([range], {row}, {column})",
        }
    except Exception as e:
        return {"operation": "index", "results": {"error": str(e)}, "formula": f"=INDEX([range], {row}, {column})"}


def match_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, lookup_value: Any, lookup_column: str
) -> dict[str, Any]:
    """
    Finds the relative position (index) of a value in a column (like Excel MATCH).

    Parameters:
        df_name (str): Name of the CSV file to load.
        lookup_value (Any): Value to search for in the lookup_column.
        lookup_column (str): Column to search within.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "match",
            "results": { "index": int, "message": str },
            "formula": "=MATCH(lookup_value, lookup_range, 0)"
        }

    Errors:
        Returns an error message in results if the file can't be read or value not found.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)
        if lookup_column not in df.columns:
            raise ValueError(f"Lookup column '{lookup_column}' not found.")
        col_values = df[lookup_column].tolist()
        try:
            position = col_values.index(lookup_value)
        except ValueError:
            raise ValueError(f"Value '{lookup_value}' not found in column '{lookup_column}'.")
        return {
            "operation": "match",
            "results": {
                "index": position,
                "message": f"Value '{lookup_value}' found at position {position} (0-based) in column '{lookup_column}'.",
            },
            "formula": f"=MATCH({lookup_value}, [range], 0)",
        }
    except Exception as e:
        return {
            "operation": "match",
            "results": {"error": str(e)},
            "formula": f"=MATCH({lookup_value}, [range], 0)",
        }


def xlookup_formula_tool(
    ctx: RunContext[AgentDeps],
    df_name: str,
    lookup_value: Any,
    lookup_column: str,
    return_column: str,
    if_not_found: Any = None,
) -> dict[str, Any]:
    """
    Searches for a value in the lookup_column and returns the corresponding value from the return_column (like Excel XLOOKUP).

    Parameters:
        df_name (str): Name of the CSV file to load.
        lookup_value (Any): Value to search for in the lookup_column.
        lookup_column (str): Column to search for the lookup_value.
        return_column (str): Column from which to return the value.
        if_not_found (Any): Value to return if no match is found (default None).
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "xlookup",
            "results": { "matched_value": Any, "message": str },
            "formula": "=XLOOKUP(lookup_value, lookup_array, return_array, [if_not_found])"
        }

    Errors:
        Returns an error message in results if the file can't be read or a column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)
        if lookup_column not in df.columns:
            raise ValueError(f"Lookup column '{lookup_column}' not found.")
        if return_column not in df.columns:
            raise ValueError(f"Return column '{return_column}' not found.")
        matches = df[df[lookup_column] == lookup_value]
        if matches.empty:
            if if_not_found is not None:
                return {
                    "operation": "xlookup",
                    "results": {
                        "matched_value": if_not_found,
                        "message": f"No match found for '{lookup_value}'. Returning 'if_not_found' value.",
                    },
                    "formula": f"=XLOOKUP({lookup_value}, [lookup_array], [return_array], {if_not_found})",
                }
            raise ValueError(f"No match found for value '{lookup_value}' in column '{lookup_column}'.")
        matched_value = matches.iloc[0][return_column]
        return {
            "operation": "xlookup",
            "results": {
                "matched_value": matched_value,
                "message": f"Found value '{matched_value}' in column '{return_column}' for lookup '{lookup_value}'.",
            },
            "formula": f"=XLOOKUP({lookup_value}, [lookup_array], [return_array], {if_not_found})",
        }
    except Exception as e:
        return {
            "operation": "xlookup",
            "results": {"error": str(e)},
            "formula": f"=XLOOKUP({lookup_value}, [lookup_array], [return_array], {if_not_found})",
        }


def offset_formula_tool(
    ctx: RunContext[AgentDeps],
    df_name: str,
    reference_row: int,
    reference_col: int,
    rows: int,
    cols: int,
    height: int = 1,
    width: int = 1,
) -> dict[str, Any]:
    """
    Returns a value or range offset from a starting cell (like Excel OFFSET).

    Parameters:
        df_name (str): Name of the CSV file to load.
        reference_row (int): Starting row index (0-based).
        reference_col (int): Starting column index (0-based).
        rows (int): Number of rows to offset.
        cols (int): Number of columns to offset.
        height (int): Height of the returned range (default 1).
        width (int): Width of the returned range (default 1).
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "offset",
            "results": { "value": Any, "message": str },
            "formula": "=OFFSET(reference, rows, cols, [height], [width])"
        }

    Errors:
        Returns an error message in results if the file can't be read or indices are out of bounds.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)
        start_row = reference_row + rows
        start_col = reference_col + cols
        if start_row < 0 or start_row + height > len(df.index):
            raise ValueError("Offset row range out of bounds.")
        if start_col < 0 or start_col + width > len(df.columns):
            raise ValueError("Offset column range out of bounds.")
        result = df.iloc[start_row : start_row + height, start_col : start_col + width]
        if height == 1 and width == 1:
            value = result.iloc[0, 0]
        else:
            value = result.values.tolist()
        return {
            "operation": "offset",
            "results": {
                "value": value,
                "message": f"Offset value(s) at ({start_row}, {start_col}) with size ({height}, {width}) returned.",
            },
            "formula": f"=OFFSET([reference], {rows}, {cols}, {height}, {width})",
        }
    except Exception as e:
        return {
            "operation": "offset",
            "results": {"error": str(e)},
            "formula": f"=OFFSET([reference], {rows}, {cols}, {height}, {width})",
        }


def indirect_formula_tool(ctx: RunContext[AgentDeps], df_name: str, ref_text: str) -> dict[str, Any]:
    """
    Returns the value referenced by a text string (like Excel INDIRECT).

    Parameters:
        df_name (str): Name of the CSV file to load.
        ref_text (str): Reference as a string (e.g., 'A1', 'B2').
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "indirect",
            "results": { "value": Any, "message": str },
            "formula": "=INDIRECT(ref_text)"
        }

    Errors:
        Returns an error message in results if the file can't be read or the reference is invalid.
    """
    try:
        import re

        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)
        # Simple A1-style reference (e.g., 'B2')
        match = re.match(r"([A-Za-z]+)([0-9]+)", ref_text)
        if not match:
            raise ValueError(f"Invalid reference text '{ref_text}'.")
        col_letters, row_str = match.groups()
        col = 0
        for letter in col_letters.upper():
            col = col * 26 + (ord(letter) - ord("A") + 1)
        col -= 1  # zero-based
        row = int(row_str) - 1  # Excel is 1-based
        if row < 0 or row >= len(df.index):
            raise ValueError(f"Row index {row} out of bounds.")
        if col < 0 or col >= len(df.columns):
            raise ValueError(f"Column index {col} out of bounds.")
        value = df.iloc[row, col]
        return {
            "operation": "indirect",
            "results": {"value": value, "message": f"Value at reference '{ref_text}' is '{value}'."},
            "formula": f"=INDIRECT({ref_text})",
        }
    except Exception as e:
        return {"operation": "indirect", "results": {"error": str(e)}, "formula": f"=INDIRECT({ref_text})"}
