#!/usr/bin/env python3
"""
Test script to preview the enhanced csv_summary function with subscriptions.csv
"""

import json
import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent / "src"))

from operate_ai.cfo_graph import csv_summary


def test_csv_summary():
    """Test the enhanced csv_summary function with subscriptions.csv"""

    # Path to the subscriptions CSV file
    csv_file = Path("operateai_scenario1_data/subscriptions.csv")

    if not csv_file.exists():
        print(f"❌ File not found: {csv_file}")
        return

    print(f"🔍 Analyzing CSV file: {csv_file}")
    print("=" * 60)

    try:
        # Run the enhanced csv_summary function
        metadata = csv_summary(csv_file, top_n_categorical=10, high_cardinality_threshold=50)
        print(metadata)

        # Print file-level summary
        print("📊 FILE SUMMARY:")
        file_summary = metadata["file_level_summary"]
        print(f"  • File: {file_summary['file_name']}")
        print(f"  • Shape: {file_summary['shape']}")
        print(f"  • Total rows: {file_summary['total_rows']:,}")
        print(f"  • Total columns: {file_summary['total_columns']}")
        print(f"  • Status: {file_summary['status']}")
        print(f"  • Analysis time: {file_summary['analysis_timestamp']}")

        # Print overall quality assessment
        if "overall_quality" in metadata:
            print("\n🎯 OVERALL DATA QUALITY:")
            quality = metadata["overall_quality"]
            print(f"  • Overall completeness: {quality['overall_completeness']}%")
            print(f"  • Total cells: {quality['total_cells']:,}")
            print(f"  • Missing cells: {quality['missing_cells']:,}")
            print("  • Column types:")
            print(f"    - Numeric: {quality['numeric_columns']}")
            print(f"    - Categorical: {quality['categorical_columns']}")
            print(f"    - DateTime: {quality['datetime_columns']}")

        # Print column summaries
        print("\n📋 COLUMN ANALYSIS:")
        for i, col in enumerate(metadata["column_summaries"], 1):
            print(f"\n  {i}. Column: {col['column_name']}")
            print(f"     • Type: {col['original_dtype']} → {col.get('inferred_best_dtype', 'unknown')}")

            if "data_quality" in col:
                dq = col["data_quality"]
                print(f"     • Completeness: {dq['completeness_percentage']}%")
                print(f"     • Unique values: {dq['unique_count']:,}")
                print(f"     • Uniqueness: {dq['uniqueness_percentage']}%")

                if "flags" in dq:
                    print(f"     • Quality flags: {', '.join(dq['flags'])}")

            # Show type-specific stats
            if "numeric_stats" in col:
                stats = col["numeric_stats"]
                print(f"     • Range: {stats['min']} to {stats['max']}")
                print(f"     • Mean: {stats['mean']}")
                if stats.get("coefficient_of_variation"):
                    print(f"     • CV: {stats['coefficient_of_variation']}")

            elif "categorical_stats" in col:
                stats = col["categorical_stats"]
                print(f"     • Unique values: {stats['unique_values']}")
                if "top_values" in stats and stats["top_values"]:
                    top_3 = stats["top_values"][:3]
                    top_str = ", ".join([f"{v['value']} ({v['count']})" for v in top_3])
                    print(f"     • Top values: {top_str}")

            elif "datetime_stats" in col:
                stats = col["datetime_stats"]
                print(f"     • Date range: {stats['min_date']} to {stats['max_date']}")
                if "date_range_days" in stats and stats["date_range_days"]:
                    print(f"     • Span: {stats['date_range_days']} days")

        # Print correlation analysis
        if "cross_column_analysis" in metadata:
            print("\n🔗 CORRELATION ANALYSIS:")
            corr_analysis = metadata["cross_column_analysis"]
            print(f"  • Found {corr_analysis['correlation_count']} meaningful correlations")

            for corr in corr_analysis["correlations"][:5]:  # Show top 5
                print(f"    - {corr['column1']} ↔ {corr['column2']}: {corr['correlation']} ({corr['strength']})")

        print("\n" + "=" * 60)
        print("✅ Analysis completed successfully!")

        # Optionally save full metadata to JSON for detailed inspection
        output_file = Path("subscriptions_metadata.json")
        with open(output_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"📝 Full metadata saved to: {output_file}")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_csv_summary()
