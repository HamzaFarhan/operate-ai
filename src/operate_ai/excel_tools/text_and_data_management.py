from pathlib import Path
from typing import Any

import pandas as pd
from pydantic_ai import RunContext

from operate_ai.excel_tools import AgentDeps


def concat_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, text_columns: list[str], separator: str = ""
) -> dict[str, Any]:
    """
    Combines text strings from specified columns with an optional separator (like Excel CONCAT/CONCATENATE).

    Parameters:
        df_name (str): Name of the CSV file to load.
        text_columns (List[str]): List of columns containing text to concatenate.
        separator (str, optional): Text to insert between concatenated values. Default is empty string.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        Dict[str, Any]: {
            "operation": "concat",
            "results": {
                "concatenated_values": list[str],
                "message": str
            },
            "formula": "=CONCAT(text1, text2, ...)" or "=CONCATENATE(text1, text2, ...)"
        }

    Errors:
        Returns an error message in results if the file can't be read or a column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)

        # Validate columns exist
        for column in text_columns:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in the dataset.")

        # Perform concatenation
        concatenated_values = df[text_columns].astype(str).agg(lambda x: separator.join(x), axis=1).tolist()

        # Construct formula representation
        formula_parts = ", ".join(text_columns)
        formula = f"=CONCAT({formula_parts})" if separator == "" else f"=CONCATENATE({formula_parts})"

        return {
            "operation": "concat",
            "results": {
                "concatenated_values": concatenated_values,
                "message": f"Successfully concatenated {len(text_columns)} columns with separator '{separator}'.",
            },
            "formula": formula,
        }
    except Exception as e:
        return {
            "operation": "concat",
            "results": {"error": str(e)},
            "formula": f"=CONCAT({', '.join(text_columns)})",
        }


def text_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, value_column: str, format_string: str
) -> dict[str, Any]:
    """
    Formats numbers or dates as text with a specified format (like Excel TEXT function).

    Parameters:
        df_name (str): Name of the CSV file to load.
        value_column (str): Column containing values to format.
        format_string (str): Format pattern to apply (e.g., "0.00%", "yyyy-mm-dd").
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        Dict[str, Any]: {
            "operation": "text",
            "results": {
                "formatted_values": list[str],
                "message": str
            },
            "formula": "=TEXT(value, format_string)"
        }

    Errors:
        Returns an error message in results if the file can't be read or the column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)

        if value_column not in df.columns:
            raise ValueError(f"Column '{value_column}' not found in the dataset.")

        # Format values according to format_string
        # Note: This is a simplified implementation; actual implementation would need
        # to handle various format strings similar to Excel's TEXT function
        formatted_values = []
        for value in df[value_column]:
            # Basic implementation - would need to be expanded for production use
            if format_string.endswith("%"):
                # Percentage format
                decimal_places = format_string.count("0")
                formatted = f"{float(value):.{decimal_places}%}"
            elif "yyyy" in format_string.lower() or "mm" in format_string.lower():
                # Date format - simplified
                formatted = pd.to_datetime(value).strftime("%Y-%m-%d")
            else:
                # Default number format
                formatted = f"{value}"
            formatted_values.append(formatted)

        return {
            "operation": "text",
            "results": {
                "formatted_values": formatted_values,
                "message": f"Successfully formatted values in '{value_column}' using format '{format_string}'.",
            },
            "formula": f'=TEXT({value_column}, "{format_string}")',
        }
    except Exception as e:
        return {
            "operation": "text",
            "results": {"error": str(e)},
            "formula": f'=TEXT({value_column}, "{format_string}")',
        }


def left_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, text_column: str, num_chars: int
) -> dict[str, Any]:
    """
    Extracts a specified number of characters from the beginning of text strings (like Excel LEFT function).

    Parameters:
        df_name (str): Name of the CSV file to load.
        text_column (str): Column containing text to extract from.
        num_chars (int): Number of characters to extract from the left.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        Dict[str, Any]: {
            "operation": "left",
            "results": {
                "extracted_values": list[str],
                "message": str
            },
            "formula": "=LEFT(text, num_chars)"
        }

    Errors:
        Returns an error message in results if the file can't be read or the column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in the dataset.")

        # Extract left characters
        extracted_values = df[text_column].astype(str).apply(lambda x: x[:num_chars]).tolist()

        return {
            "operation": "left",
            "results": {
                "extracted_values": extracted_values,
                "message": f"Successfully extracted {num_chars} characters from the left of values in '{text_column}'.",
            },
            "formula": f"=LEFT({text_column}, {num_chars})",
        }
    except Exception as e:
        return {"operation": "left", "results": {"error": str(e)}, "formula": f"=LEFT({text_column}, {num_chars})"}


def right_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, text_column: str, num_chars: int
) -> dict[str, Any]:
    """
    Extracts a specified number of characters from the end of text strings (like Excel RIGHT function).

    Parameters:
        df_name (str): Name of the CSV file to load.
        text_column (str): Column containing text to extract from.
        num_chars (int): Number of characters to extract from the right.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        Dict[str, Any]: {
            "operation": "right",
            "results": {
                "extracted_values": list[str],
                "message": str
            },
            "formula": "=RIGHT(text, num_chars)"
        }

    Errors:
        Returns an error message in results if the file can't be read or the column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in the dataset.")

        # Extract right characters
        extracted_values = (
            df[text_column].astype(str).apply(lambda x: x[-num_chars:] if len(x) >= num_chars else x).tolist()
        )

        return {
            "operation": "right",
            "results": {
                "extracted_values": extracted_values,
                "message": f"Successfully extracted {num_chars} characters from the right of values in '{text_column}'.",
            },
            "formula": f"=RIGHT({text_column}, {num_chars})",
        }
    except Exception as e:
        return {
            "operation": "right",
            "results": {"error": str(e)},
            "formula": f"=RIGHT({text_column}, {num_chars})",
        }


def mid_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, text_column: str, start_num: int, num_chars: int
) -> dict[str, Any]:
    """
    Extracts a specified number of characters from the middle of text strings (like Excel MID function).

    Parameters:
        df_name (str): Name of the CSV file to load.
        text_column (str): Column containing text to extract from.
        start_num (int): Starting position (1-based index as in Excel).
        num_chars (int): Number of characters to extract.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        Dict[str, Any]: {
            "operation": "mid",
            "results": {
                "extracted_values": list[str],
                "message": str
            },
            "formula": "=MID(text, start_num, num_chars)"
        }

    Errors:
        Returns an error message in results if the file can't be read or the column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in the dataset.")

        # Extract middle characters (adjust for 0-based indexing in Python vs 1-based in Excel)
        extracted_values = (
            df[text_column]
            .astype(str)
            .apply(
                lambda x: x[start_num - 1 : start_num - 1 + num_chars]
                if start_num > 0 and start_num <= len(x)
                else ""
            )
            .tolist()
        )

        return {
            "operation": "mid",
            "results": {
                "extracted_values": extracted_values,
                "message": f"Successfully extracted {num_chars} characters starting at position {start_num} from values in '{text_column}'.",
            },
            "formula": f"=MID({text_column}, {start_num}, {num_chars})",
        }
    except Exception as e:
        return {
            "operation": "mid",
            "results": {"error": str(e)},
            "formula": f"=MID({text_column}, {start_num}, {num_chars})",
        }


def len_formula_tool(ctx: RunContext[AgentDeps], df_name: str, text_column: str) -> dict[str, Any]:
    """
    Counts the number of characters in text strings (like Excel LEN function).

    Parameters:
        df_name (str): Name of the CSV file to load.
        text_column (str): Column containing text to count characters in.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        Dict[str, Any]: {
            "operation": "len",
            "results": {
                "character_counts": list[int],
                "message": str
            },
            "formula": "=LEN(text)"
        }

    Errors:
        Returns an error message in results if the file can't be read or the column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in the dataset.")

        # Count characters
        character_counts = df[text_column].astype(str).apply(len).tolist()

        return {
            "operation": "len",
            "results": {
                "character_counts": character_counts,
                "message": f"Successfully counted characters in values from '{text_column}'.",
            },
            "formula": f"=LEN({text_column})",
        }
    except Exception as e:
        return {"operation": "len", "results": {"error": str(e)}, "formula": f"=LEN({text_column})"}


def find_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, text_column: str, find_text: str, start_num: int = 1
) -> dict[str, Any]:
    """
    Locates the position of one text string within another (case-sensitive, like Excel FIND function).

    Parameters:
        df_name (str): Name of the CSV file to load.
        text_column (str): Column containing text to search within.
        find_text (str): Text to find within the column values.
        start_num (int, optional): Position to start searching from (1-based index as in Excel). Default is 1.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        Dict[str, Any]: {
            "operation": "find",
            "results": {
                "positions": list[int | None],
                "message": str
            },
            "formula": "=FIND(find_text, within_text, [start_num])"
        }

    Errors:
        Returns an error message in results if the file can't be read or the column is missing.
        Returns None for positions where the text was not found (Excel would return #VALUE!).
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in the dataset.")

        # Find text positions (adjust for 0-based indexing in Python vs 1-based in Excel)
        def find_position(text):
            text = str(text)
            adjusted_start = start_num - 1
            if adjusted_start < 0 or adjusted_start >= len(text):
                return None
            pos = text.find(find_text, adjusted_start)
            return pos + 1 if pos >= 0 else None  # Return 1-based position or None if not found

        positions = df[text_column].apply(find_position).tolist()

        formula = f'=FIND("{find_text}", {text_column}' + (f", {start_num}" if start_num > 1 else "") + ")"

        return {
            "operation": "find",
            "results": {
                "positions": positions,
                "message": f"Successfully found positions of '{find_text}' in '{text_column}'.",
            },
            "formula": formula,
        }
    except Exception as e:
        return {
            "operation": "find",
            "results": {"error": str(e)},
            "formula": f'=FIND("{find_text}", {text_column})',
        }


def search_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, text_column: str, find_text: str, start_num: int = 1
) -> dict[str, Any]:
    """
    Locates the position of one text string within another (case-insensitive, like Excel SEARCH function).

    Parameters:
        df_name (str): Name of the CSV file to load.
        text_column (str): Column containing text to search within.
        find_text (str): Text to find within the column values.
        start_num (int, optional): Position to start searching from (1-based index as in Excel). Default is 1.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        Dict[str, Any]: {
            "operation": "search",
            "results": {
                "positions": list[int | None],
                "message": str
            },
            "formula": "=SEARCH(find_text, within_text, [start_num])"
        }

    Errors:
        Returns an error message in results if the file can't be read or the column is missing.
        Returns None for positions where the text was not found (Excel would return #VALUE!).
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in the dataset.")

        # Find text positions (case-insensitive, adjust for 0-based indexing in Python vs 1-based in Excel)
        def search_position(text):
            text = str(text).lower()
            find_text_lower = find_text.lower()
            adjusted_start = start_num - 1
            if adjusted_start < 0 or adjusted_start >= len(text):
                return None
            pos = text.find(find_text_lower, adjusted_start)
            return pos + 1 if pos >= 0 else None  # Return 1-based position or None if not found

        positions = df[text_column].apply(search_position).tolist()

        formula = f'=SEARCH("{find_text}", {text_column}' + (f", {start_num}" if start_num > 1 else "") + ")"

        return {
            "operation": "search",
            "results": {
                "positions": positions,
                "message": f"Successfully searched for positions of '{find_text}' in '{text_column}' (case-insensitive).",
            },
            "formula": formula,
        }
    except Exception as e:
        return {
            "operation": "search",
            "results": {"error": str(e)},
            "formula": f'=SEARCH("{find_text}", {text_column})',
        }


def replace_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, text_column: str, start_num: int, num_chars: int, new_text: str
) -> dict[str, Any]:
    """
    Replaces a specific portion of text with new text (like Excel REPLACE function).

    Parameters:
        df_name (str): Name of the CSV file to load.
        text_column (str): Column containing text to modify.
        start_num (int): Starting position for replacement (1-based index as in Excel).
        num_chars (int): Number of characters to replace.
        new_text (str): Text to insert as replacement.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        Dict[str, Any]: {
            "operation": "replace",
            "results": {
                "replaced_values": list[str],
                "message": str
            },
            "formula": "=REPLACE(old_text, start_num, num_chars, new_text)"
        }

    Errors:
        Returns an error message in results if the file can't be read or the column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in the dataset.")

        # Replace text portions
        def replace_text(text):
            text = str(text)
            adjusted_start = start_num - 1  # Adjust for 0-based indexing
            if adjusted_start < 0 or adjusted_start > len(text):
                return text  # Return original if start position is invalid
            return text[:adjusted_start] + new_text + text[adjusted_start + num_chars :]

        replaced_values = df[text_column].apply(replace_text).tolist()

        return {
            "operation": "replace",
            "results": {
                "replaced_values": replaced_values,
                "message": f"Successfully replaced {num_chars} characters starting at position {start_num} with '{new_text}' in '{text_column}'.",
            },
            "formula": f'=REPLACE({text_column}, {start_num}, {num_chars}, "{new_text}")',
        }
    except Exception as e:
        return {
            "operation": "replace",
            "results": {"error": str(e)},
            "formula": f'=REPLACE({text_column}, {start_num}, {num_chars}, "{new_text}")',
        }


def substitute_formula_tool(
    ctx: RunContext[AgentDeps],
    df_name: str,
    text_column: str,
    old_text: str,
    new_text: str,
    instance_num: int | None = None,
) -> dict[str, Any]:
    """
    Replaces specific text with new text (like Excel SUBSTITUTE function).

    Parameters:
        df_name (str): Name of the CSV file to load.
        text_column (str): Column containing text to modify.
        old_text (str): Text to be replaced.
        new_text (str): Text to insert as replacement.
        instance_num (int, optional): Which occurrence to replace (if omitted, replaces all occurrences).
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        Dict[str, Any]: {
            "operation": "substitute",
            "results": {
                "substituted_values": list[str],
                "message": str
            },
            "formula": "=SUBSTITUTE(text, old_text, new_text, [instance_num])"
        }

    Errors:
        Returns an error message in results if the file can't be read or the column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in the dataset.")

        # Substitute text
        def substitute_text(text):
            text = str(text)
            if instance_num is None:
                # Replace all occurrences
                return text.replace(old_text, new_text)
            else:
                # Replace specific occurrence
                parts = text.split(old_text)
                if len(parts) <= instance_num:
                    return text  # Not enough occurrences found
                result = old_text.join(parts[:instance_num]) + new_text + old_text.join(parts[instance_num:])
                return result

        substituted_values = df[text_column].apply(substitute_text).tolist()

        formula = (
            f'=SUBSTITUTE({text_column}, "{old_text}", "{new_text}"'
            + (f", {instance_num}" if instance_num is not None else "")
            + ")"
        )

        return {
            "operation": "substitute",
            "results": {
                "substituted_values": substituted_values,
                "message": f"Successfully substituted '{old_text}' with '{new_text}' in '{text_column}'.",
            },
            "formula": formula,
        }
    except Exception as e:
        return {
            "operation": "substitute",
            "results": {"error": str(e)},
            "formula": f'=SUBSTITUTE({text_column}, "{old_text}", "{new_text}")',
        }
