from pathlib import Path
from typing import Any

import pandas as pd


def preview_csv(
    df: Path | pd.DataFrame, top_n_categorical: int = 5, sample_size_for_type_inference: int = 1000
) -> dict[str, Any]:
    """
    Generates a comprehensive summary of a Pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to summarize.
        top_n_categorical (int): The number of most frequent values to show for categorical columns.
        sample_size_for_type_inference (int): Number of non-null rows to sample for more robust type inference especially for 'object' columns.
    Returns:
        dict: A dictionary containing the summary information.
    """
    file_name = ""
    if isinstance(df, Path):
        file_name = df.name
        df = pd.read_csv(df)  # type: ignore
    if df.empty:
        return {
            "file_level_summary": {
                "file_name": file_name,
                "shape": (0, 0),
                "memory_usage_mb": 0.0,
                "total_rows": 0,
                "total_columns": 0,
                "notes": "DataFrame is empty.",
            },
            "column_summaries": [],
        }

    summary = {}

    # 1. File-Level Summary
    total_rows, total_columns = df.shape

    summary["file_level_summary"] = {
        "shape": (total_rows, total_columns),
        "total_rows": total_rows,
        "total_columns": total_columns,
    }

    # 2. Column-Specific Summaries
    column_summaries = []
    for col_name in df.columns:
        col_data = df[col_name]  # type: ignore
        col_summary = {}

        col_summary["column_name"] = col_name
        col_summary["original_dtype"] = str(col_data.dtype)

        # Missing values
        missing_count = col_data.isnull().sum()  # type: ignore
        col_summary["missing_values_count"] = int(missing_count)
        col_summary["missing_values_percentage"] = (
            round((missing_count / total_rows) * 100, 2) if total_rows > 0 else 0
        )

        # Unique values
        try:
            # nunique() can be slow on very large object columns with many uniques.
            # If performance is an issue on huge datasets, consider sampling or alternative methods.
            unique_count = col_data.nunique()
            col_summary["unique_values_count"] = int(unique_count)
        except Exception:  # Handle cases like lists in cells which are not hashable
            col_summary["unique_values_count"] = "Error calculating (possibly unhashable type)"

        # Attempt more robust type inference for object columns
        inferred_type = str(col_data.dtype)
        if col_data.dtype == "object" and not col_data.isnull().all():  # type: ignore
            # Try to infer if it's numeric, datetime, or truly categorical
            sample_data = col_data.dropna().sample(  # type: ignore
                min(len(col_data.dropna()), sample_size_for_type_inference),  # type: ignore
                random_state=42,  # type: ignore
            )
            try:
                pd.to_numeric(sample_data)  # type: ignore
                inferred_type = "numeric (inferred from object)"
            except (ValueError, TypeError):
                try:
                    pd.to_datetime(sample_data)  # type: ignore
                    inferred_type = "datetime (inferred from object)"
                except (ValueError, TypeError):
                    inferred_type = "categorical/string (inferred from object)"
        elif pd.api.types.is_datetime64_any_dtype(col_data):  # type: ignore
            inferred_type = "datetime"
        elif pd.api.types.is_numeric_dtype(col_data):  # type: ignore
            inferred_type = "numeric"
        elif pd.api.types.is_bool_dtype(col_data):  # type: ignore
            inferred_type = "boolean"

        col_summary["inferred_best_dtype"] = inferred_type

        # Type-specific stats
        if "numeric" in inferred_type and not col_data.isnull().all():  # type: ignore
            # Ensure data is actually numeric before describing
            numeric_col_data = pd.to_numeric(col_data, errors="coerce").dropna()  # type: ignore
            if not numeric_col_data.empty:
                desc = numeric_col_data.describe()  # type: ignore
                col_summary["numeric_stats"] = {
                    "mean": round(desc.get("mean", float("nan")), 3),  # type: ignore
                    "std_dev": round(desc.get("std", float("nan")), 3),  # type: ignore
                    "min": round(desc.get("min", float("nan")), 3),  # type: ignore
                    "25th_percentile": round(desc.get("25%", float("nan")), 3),  # type: ignore
                    "median_50th_percentile": round(desc.get("50%", float("nan")), 3),  # type: ignore
                    "75th_percentile": round(desc.get("75%", float("nan")), 3),  # type: ignore
                    "max": round(desc.get("max", float("nan")), 3),  # type: ignore
                }
        elif "datetime" in inferred_type and not col_data.isnull().all():  # type: ignore
            datetime_col_data = pd.to_datetime(col_data, errors="coerce").dropna()  # type: ignore
            if not datetime_col_data.empty:
                col_summary["datetime_stats"] = {
                    "min_date": str(datetime_col_data.min()),  # type: ignore
                    "max_date": str(datetime_col_data.max()),  # type: ignore
                }
        elif "boolean" in inferred_type and not col_data.isnull().all():  # type: ignore
            col_summary["boolean_stats"] = col_data.dropna().astype(bool).value_counts().to_dict()  # type: ignore
        elif "categorical/string" in inferred_type and not col_data.isnull().all():  # type: ignore
            # For object/categorical columns
            if unique_count <= total_rows * 0.8 and unique_count > 0:  # type: ignore
                counts = col_data.value_counts(normalize=False).head(top_n_categorical)
                col_summary["categorical_stats"] = {
                    "top_n_values": {str(k): int(v) for k, v in counts.items()},
                    "is_highly_cardinal": unique_count > 50  # type: ignore
                    and unique_count / total_rows > 0.1,  # type: ignore
                }
            else:  # Likely free text or IDs
                col_summary["categorical_stats"] = {
                    "note": "High cardinality or mostly unique string values. Not showing top N.",
                    "is_highly_cardinal": True,
                }
        elif col_data.isnull().all():  # type: ignore
            col_summary["note"] = "Column is entirely empty (all NaN)."

        column_summaries.append(col_summary)  # type: ignore

    summary["column_summaries"] = column_summaries
    return summary  # type: ignore
