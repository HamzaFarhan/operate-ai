from pathlib import Path

import pandas as pd

# Load the generated data
data_dir = Path("operateai_scenario1_data")

# Load relevant datasets
subscriptions_df = pd.read_csv(data_dir / "subscriptions.csv")
orders_df = pd.read_csv(data_dir / "orders.csv")

# Convert date columns to datetime
subscriptions_df["StartDate"] = pd.to_datetime(subscriptions_df["StartDate"])
subscriptions_df["EndDate"] = pd.to_datetime(subscriptions_df["EndDate"])
orders_df["OrderDate"] = pd.to_datetime(orders_df["OrderDate"])

print("=== MRR Calculation for December 2024 ===\n")

# Step 1: Find active customers as of Dec 2024
dec_2024 = pd.Timestamp("2024-12-31")
print(f"Calculating MRR as of: {dec_2024.strftime('%Y-%m-%d')}")

# Active subscriptions are those that:
# - Started before or on Dec 31, 2024
# - Either have no end date (still active) OR ended after Dec 31, 2024
active_subs = subscriptions_df[
    (subscriptions_df["StartDate"] <= dec_2024)
    & ((subscriptions_df["EndDate"].isna()) | (subscriptions_df["EndDate"] > dec_2024))
]

print(f"\nActive subscriptions as of Dec 2024: {len(active_subs)}")

# Get unique active customers
active_customers = active_subs["CustomerID"].unique()
print(f"Unique active customers: {len(active_customers)}")

# Step 2: Calculate ARPU for December 2024
# Get orders from December 2024 for active customers
dec_orders = orders_df[
    (orders_df["OrderDate"].dt.year == 2024)
    & (orders_df["OrderDate"].dt.month == 12)
    & (orders_df["CustomerID"].isin(active_customers))
]

print(f"\nOrders in December 2024 from active customers: {len(dec_orders)}")
total_revenue_dec = dec_orders["Amount"].sum()
print(f"Total revenue in December 2024: ${total_revenue_dec:,.2f}")

# Calculate ARPU
if len(active_customers) > 0:
    arpu = total_revenue_dec / len(active_customers)
    print(f"ARPU for December 2024: ${arpu:.2f}")

    # Calculate MRR
    mrr = len(active_customers) * arpu
    print("\nMRR Calculation:")
    print(f"Customer Count: {len(active_customers)}")
    print(f"ARPU: ${arpu:.6f}")
    print(f"MRR = {len(active_customers)} × ${arpu:.6f} = ${mrr:.2f}")
else:
    print("No active customers found!")

# Additional breakdown by subscription type and plan
print("\n=== Breakdown by Plan and Subscription Type ===")
active_subs_with_orders = active_subs.merge(
    dec_orders.groupby("CustomerID")["Amount"].sum().reset_index(), on="CustomerID", how="left"
).fillna(0)

breakdown = (
    active_subs_with_orders.groupby(["PlanName", "SubscriptionType"])
    .agg({"CustomerID": "count", "Amount": ["sum", "mean"]})
    .round(2)
)

breakdown.columns = ["Customer_Count", "Total_Revenue", "Avg_Revenue_Per_Customer"]
print(breakdown)

print("\n=== Summary ===")
print("Expected MRR: $13,770.00")
print(f"Calculated MRR: ${mrr:.2f}")
print(f"Difference: ${abs(13770.0 - mrr):.2f}")
