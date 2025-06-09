"""
LTV Analysis for Scenario 1
Calculates customer LTV for Jan 2023 to Dec 2023 by subscription types, plan types, industry, and channel.
Following requirements from 1.txt
"""

import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = Path("operateai_scenario1_data")
PROFIT_MARGIN = 0.75  # 75% profit margin assumption
YEARS_TO_CHURN_IF_ZERO = 5  # Assume 5 years if 0% churn
ANALYSIS_START = "2023-01-01"
ANALYSIS_END = "2023-12-31"

# Industry standard CAC estimates by channel (if needed)
CAC_ESTIMATES = {
    "Paid Search": 800,
    "Social Media": 600,
    "Email": 250,
    "Affiliate": 500,
    "Content": 400,
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all necessary datasets"""
    customers_df = pd.read_csv(DATA_DIR / "customers.csv")
    subscriptions_df = pd.read_csv(DATA_DIR / "subscriptions.csv")
    orders_df = pd.read_csv(DATA_DIR / "orders.csv")
    marketing_spend_df = pd.read_csv(DATA_DIR / "marketing_spend.csv")

    # Convert date columns
    customers_df["AcquisitionDate"] = pd.to_datetime(customers_df["AcquisitionDate"])
    subscriptions_df["StartDate"] = pd.to_datetime(subscriptions_df["StartDate"])
    subscriptions_df["EndDate"] = pd.to_datetime(subscriptions_df["EndDate"], errors="coerce")
    orders_df["OrderDate"] = pd.to_datetime(orders_df["OrderDate"])
    marketing_spend_df["Date"] = pd.to_datetime(marketing_spend_df["Date"])

    return customers_df, subscriptions_df, orders_df, marketing_spend_df


def get_initial_subscribers(subscriptions_df: pd.DataFrame, customers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get customers who were active subscribers at the beginning of Jan 2023
    and their initial subscription/plan types
    """
    jan_2023_start = pd.Timestamp(ANALYSIS_START)

    # Find customers with active subscriptions on Jan 1, 2023
    active_jan_2023 = subscriptions_df[
        (subscriptions_df["StartDate"] <= jan_2023_start)
        & ((subscriptions_df["EndDate"] >= jan_2023_start) | subscriptions_df["EndDate"].isna())
    ]

    # Get the first (initial) subscription for each customer to determine their initial plan/type
    customer_initial_subs = subscriptions_df.sort_values("StartDate").groupby("CustomerID").first().reset_index()

    # Merge with customer data to get segments
    initial_customers = active_jan_2023.merge(customers_df, on="CustomerID", how="left")
    initial_customers = initial_customers.merge(
        customer_initial_subs[["CustomerID", "PlanName", "SubscriptionType"]].rename(
            columns={"PlanName": "PlanName_Initial", "SubscriptionType": "SubscriptionType_Initial"}
        ),
        on="CustomerID",
        how="left",
    )

    return initial_customers


def calculate_arpu_by_segment(
    orders_df: pd.DataFrame, customers_df: pd.DataFrame, subscriptions_df: pd.DataFrame, segment_type: str
) -> pd.DataFrame:
    """
    Calculate ARPU by segment for 2023
    segment_type: 'plan', 'subscription_type', 'industry', 'channel', or 'total'
    """
    # Filter orders for 2023
    orders_2023 = orders_df[(orders_df["OrderDate"] >= ANALYSIS_START) & (orders_df["OrderDate"] <= ANALYSIS_END)]

    # Get initial customer segments
    initial_customers = get_initial_subscribers(subscriptions_df, customers_df)
    customer_segments = initial_customers[
        ["CustomerID", "IndustrySegment", "AcquisitionChannel", "PlanName_Initial", "SubscriptionType_Initial"]
    ].drop_duplicates()

    # Merge orders with customer segments
    orders_with_segments = orders_2023.merge(customer_segments, on="CustomerID", how="left")

    if segment_type == "total":
        # Total ARPU
        total_revenue = orders_with_segments["Amount"].sum()
        unique_customers = orders_with_segments["CustomerID"].nunique()
        arpu = total_revenue / unique_customers if unique_customers > 0 else 0
        return pd.DataFrame(
            {
                "Segment": ["Total"],
                "Total_Revenue": [total_revenue],
                "Unique_Customers": [unique_customers],
                "ARPU": [arpu],
            }
        )

    elif segment_type == "plan":
        segment_col = "PlanName_Initial"
    elif segment_type == "subscription_type":
        segment_col = "SubscriptionType_Initial"
    elif segment_type == "industry":
        segment_col = "IndustrySegment"
    elif segment_type == "channel":
        segment_col = "AcquisitionChannel"
    else:
        raise ValueError(f"Unknown segment_type: {segment_type}")

    # Calculate ARPU by segment
    segment_stats = (
        orders_with_segments.groupby(segment_col).agg({"Amount": "sum", "CustomerID": "nunique"}).reset_index()
    )

    segment_stats.columns = ["Segment", "Total_Revenue", "Unique_Customers"]
    segment_stats["ARPU"] = segment_stats["Total_Revenue"] / segment_stats["Unique_Customers"]

    return segment_stats


def calculate_churn_rate_by_segment(
    subscriptions_df: pd.DataFrame, customers_df: pd.DataFrame, segment_type: str
) -> pd.DataFrame:
    """
    Calculate churn rate by segment for customers active at start of 2023
    """
    initial_customers = get_initial_subscribers(subscriptions_df, customers_df)

    # Count customers at beginning of period by segment
    if segment_type == "total":
        initial_count = len(initial_customers["CustomerID"].unique())
        segment_counts = pd.DataFrame({"Segment": ["Total"], "Initial_Customers": [initial_count]})
    elif segment_type == "plan":
        segment_counts = initial_customers.groupby("PlanName_Initial")["CustomerID"].nunique().reset_index()
        segment_counts.columns = ["Segment", "Initial_Customers"]
    elif segment_type == "subscription_type":
        segment_counts = (
            initial_customers.groupby("SubscriptionType_Initial")["CustomerID"].nunique().reset_index()
        )
        segment_counts.columns = ["Segment", "Initial_Customers"]
    elif segment_type == "industry":
        segment_counts = initial_customers.groupby("IndustrySegment")["CustomerID"].nunique().reset_index()
        segment_counts.columns = ["Segment", "Initial_Customers"]
    elif segment_type == "channel":
        segment_counts = initial_customers.groupby("AcquisitionChannel")["CustomerID"].nunique().reset_index()
        segment_counts.columns = ["Segment", "Initial_Customers"]

    # Find churned customers (those who had active subscriptions in Jan 2023 but churned during 2023)
    jan_2023_start = pd.Timestamp(ANALYSIS_START)
    dec_2023_end = pd.Timestamp(ANALYSIS_END)

    initial_customer_ids = set(initial_customers["CustomerID"].unique())

    # Find customers who churned during 2023
    churned_customers = set()
    for customer_id in initial_customer_ids:
        customer_subs = subscriptions_df[subscriptions_df["CustomerID"] == customer_id]
        # Check if customer's latest subscription ended during 2023
        latest_sub = customer_subs.sort_values("StartDate").iloc[-1]
        if (
            pd.notna(latest_sub["EndDate"])
            and latest_sub["EndDate"] <= dec_2023_end
            and latest_sub["EndDate"] >= jan_2023_start
        ):
            churned_customers.add(customer_id)

    # Calculate churned customers by segment
    churned_data = initial_customers[initial_customers["CustomerID"].isin(churned_customers)]

    if segment_type == "total":
        churned_count = len(churned_customers)
        churn_counts = pd.DataFrame({"Segment": ["Total"], "Churned_Customers": [churned_count]})
    elif segment_type == "plan":
        churn_counts = churned_data.groupby("PlanName_Initial")["CustomerID"].nunique().reset_index()
        churn_counts.columns = ["Segment", "Churned_Customers"]
    elif segment_type == "subscription_type":
        churn_counts = churned_data.groupby("SubscriptionType_Initial")["CustomerID"].nunique().reset_index()
        churn_counts.columns = ["Segment", "Churned_Customers"]
    elif segment_type == "industry":
        churn_counts = churned_data.groupby("IndustrySegment")["CustomerID"].nunique().reset_index()
        churn_counts.columns = ["Segment", "Churned_Customers"]
    elif segment_type == "channel":
        churn_counts = churned_data.groupby("AcquisitionChannel")["CustomerID"].nunique().reset_index()
        churn_counts.columns = ["Segment", "Churned_Customers"]

    # Merge and calculate churn rate
    churn_stats = segment_counts.merge(churn_counts, on="Segment", how="left")
    churn_stats["Churned_Customers"] = churn_stats["Churned_Customers"].fillna(0)
    churn_stats["Churn_Rate"] = churn_stats["Churned_Customers"] / churn_stats["Initial_Customers"]

    return churn_stats


def calculate_ltv_by_segment(
    orders_df: pd.DataFrame, subscriptions_df: pd.DataFrame, customers_df: pd.DataFrame, segment_type: str
) -> pd.DataFrame:
    """
    Calculate LTV by segment using formula: LTV = (ARPU / Churn Rate) × Profit Margin
    """
    arpu_stats = calculate_arpu_by_segment(orders_df, customers_df, subscriptions_df, segment_type)
    churn_stats = calculate_churn_rate_by_segment(subscriptions_df, customers_df, segment_type)

    # Merge ARPU and churn data
    ltv_stats = arpu_stats.merge(churn_stats, on="Segment", how="outer")

    # Handle zero churn rate (assume 5-year customer lifetime)
    ltv_stats["Effective_Churn_Rate"] = ltv_stats["Churn_Rate"].apply(
        lambda x: 1 / (YEARS_TO_CHURN_IF_ZERO * 12) if x == 0 else x  # Convert to monthly churn
    )

    # Calculate LTV
    ltv_stats["LTV"] = (ltv_stats["ARPU"] / ltv_stats["Effective_Churn_Rate"]) * PROFIT_MARGIN

    # Round values
    ltv_stats["ARPU"] = ltv_stats["ARPU"].round(2)
    ltv_stats["Churn_Rate"] = ltv_stats["Churn_Rate"].round(4)
    ltv_stats["LTV"] = ltv_stats["LTV"].round(2)

    return ltv_stats


def calculate_cac_by_channel(marketing_spend_df: pd.DataFrame, customers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate CAC by acquisition channel for 2023
    """
    # Marketing spend for 2023
    marketing_2023 = marketing_spend_df[
        (marketing_spend_df["Date"] >= ANALYSIS_START) & (marketing_spend_df["Date"] <= ANALYSIS_END)
    ]

    # Acquisitions for 2023
    acquisitions_2023 = customers_df[
        (customers_df["AcquisitionDate"] >= ANALYSIS_START) & (customers_df["AcquisitionDate"] <= ANALYSIS_END)
    ]

    # Calculate spend by channel
    channel_spend = marketing_2023.groupby("Channel")["Spend"].sum().reset_index()

    # Calculate acquisitions by channel
    channel_acquisitions = acquisitions_2023["AcquisitionChannel"].value_counts().reset_index()
    channel_acquisitions.columns = ["Channel", "Acquisitions"]

    # Merge and calculate CAC
    cac_stats = channel_spend.merge(channel_acquisitions, on="Channel", how="left")
    cac_stats["Acquisitions"] = cac_stats["Acquisitions"].fillna(0)
    cac_stats["CAC"] = cac_stats.apply(
        lambda row: row["Spend"] / row["Acquisitions"]
        if row["Acquisitions"] > 0
        else CAC_ESTIMATES.get(row["Channel"], 0),
        axis=1,
    )

    cac_stats["CAC"] = cac_stats["CAC"].round(2)

    return cac_stats


def calculate_cac_ltv_analysis(
    orders_df: pd.DataFrame,
    subscriptions_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    marketing_spend_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate CAC to LTV analysis by channel
    """
    ltv_by_channel = calculate_ltv_by_segment(orders_df, subscriptions_df, customers_df, "channel")
    cac_by_channel = calculate_cac_by_channel(marketing_spend_df, customers_df)

    # Merge LTV and CAC data
    cac_ltv_stats = ltv_by_channel.merge(
        cac_by_channel[["Channel", "CAC"]], left_on="Segment", right_on="Channel", how="left"
    )

    # Calculate LTV/CAC ratio
    cac_ltv_stats["LTV_CAC_Ratio"] = cac_ltv_stats["LTV"] / cac_ltv_stats["CAC"]

    # Categorize ratio
    cac_ltv_stats["Category"] = cac_ltv_stats["LTV_CAC_Ratio"].apply(
        lambda x: "Excellent" if x >= 3.0 else "Good" if x >= 2.0 else "Break-even" if x >= 1.0 else "Unprofitable"
    )

    cac_ltv_stats["LTV_CAC_Ratio"] = cac_ltv_stats["LTV_CAC_Ratio"].round(2)

    return cac_ltv_stats


def format_currency(value: float) -> str:
    """Format value as currency"""
    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage"""
    return f"{value:.4%}"


def generate_summary_tables():
    """Generate all required summary tables"""
    print("Loading data...")
    customers_df, subscriptions_df, orders_df, marketing_spend_df = load_data()

    print("Calculating LTV metrics...")

    # 1. Total Customer LTV
    print("\n=== 1. TOTAL CUSTOMER LTV ===")
    total_ltv = calculate_ltv_by_segment(orders_df, subscriptions_df, customers_df, "total")
    print(total_ltv.to_string(index=False))

    # 2. Customer LTV by Plan
    print("\n=== 2. CUSTOMER LTV BY PLAN ===")
    plan_ltv = calculate_ltv_by_segment(orders_df, subscriptions_df, customers_df, "plan")
    print(plan_ltv.to_string(index=False))

    # 3. Customer LTV by Industry
    print("\n=== 3. CUSTOMER LTV BY INDUSTRY ===")
    industry_ltv = calculate_ltv_by_segment(orders_df, subscriptions_df, customers_df, "industry")
    print(industry_ltv.to_string(index=False))

    # 4. Customer LTV by Channel
    print("\n=== 4. CUSTOMER LTV BY CHANNEL ===")
    channel_ltv = calculate_ltv_by_segment(orders_df, subscriptions_df, customers_df, "channel")
    print(channel_ltv.to_string(index=False))

    # 5. CAC to LTV Analysis
    print("\n=== 5. CAC TO LTV ANALYSIS BY CHANNEL ===")
    cac_ltv_analysis = calculate_cac_ltv_analysis(orders_df, subscriptions_df, customers_df, marketing_spend_df)
    print(cac_ltv_analysis[["Segment", "LTV", "CAC", "LTV_CAC_Ratio", "Category"]].to_string(index=False))

    # Save results to CSV
    print("\nSaving results to CSV files...")
    total_ltv.to_csv("total_ltv_analysis.csv", index=False)
    plan_ltv.to_csv("plan_ltv_analysis.csv", index=False)
    industry_ltv.to_csv("industry_ltv_analysis.csv", index=False)
    channel_ltv.to_csv("channel_ltv_analysis.csv", index=False)
    cac_ltv_analysis.to_csv("cac_ltv_analysis.csv", index=False)

    print("Analysis complete! CSV files saved.")

    return {
        "total_ltv": total_ltv,
        "plan_ltv": plan_ltv,
        "industry_ltv": industry_ltv,
        "channel_ltv": channel_ltv,
        "cac_ltv_analysis": cac_ltv_analysis,
    }


if __name__ == "__main__":
    results = generate_summary_tables()
