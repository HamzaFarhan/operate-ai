import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dateutil import relativedelta
from faker import Faker

# Initialize Faker
fake = Faker()

# --- Configuration ---
NUM_CUSTOMERS = 200
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2025, 1, 1)
OUTPUT_DIR = Path("operateai_scenario1_data")

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Customer segmentation weights from scenario_1.md
GEOGRAPHY_WEIGHTS = {"North America": 0.60, "Europe": 0.25, "Asia-Pacific": 0.15}
INDUSTRY_WEIGHTS = {"Tech": 0.35, "Healthcare": 0.25, "Retail": 0.20, "Education": 0.10, "Other": 0.10}
CUSTOMER_TYPE_WEIGHTS = {"SMBs": 0.70, "Mid-Market": 0.25, "Enterprise": 0.05}
ACQUISITION_CHANNEL_WEIGHTS = {"Paid Ads": 0.50, "Organic Search": 0.30, "Email Campaigns": 0.20}

# --- Generate Customers ---
customers_data = []
for _ in range(NUM_CUSTOMERS):
    customer_id = str(uuid.uuid4())
    customer_name = fake.name()
    company_name = fake.company()
    email = fake.email()

    # Assign segments based on weights
    geography = random.choices(list(GEOGRAPHY_WEIGHTS.keys()), weights=list(GEOGRAPHY_WEIGHTS.values()), k=1)[0]
    industry = random.choices(list(INDUSTRY_WEIGHTS.keys()), weights=list(INDUSTRY_WEIGHTS.values()), k=1)[0]
    customer_type = random.choices(
        list(CUSTOMER_TYPE_WEIGHTS.keys()), weights=list(CUSTOMER_TYPE_WEIGHTS.values()), k=1
    )[0]
    acquisition_channel = random.choices(
        list(ACQUISITION_CHANNEL_WEIGHTS.keys()), weights=list(ACQUISITION_CHANNEL_WEIGHTS.values()), k=1
    )[0]

    # Generate random acquisition date within the specified range
    total_days = (END_DATE - START_DATE).days
    random_days = random.randint(0, total_days)
    acquisition_date = START_DATE + timedelta(days=random_days)

    customers_data.append(
        {
            "CustomerID": customer_id,
            "CustomerName": customer_name,
            "CompanyName": company_name,
            "Email": email,
            "Geography": geography,
            "IndustrySegment": industry,
            "CustomerType": customer_type,
            "AcquisitionDate": acquisition_date.strftime("%Y-%m-%d"),  # Format date as string
            "AcquisitionChannel": acquisition_channel,
        }
    )

customers_df = pd.DataFrame(customers_data)

# --- Save Customers CSV ---
customers_csv_path = OUTPUT_DIR / "customers.csv"
customers_df.to_csv(customers_csv_path, index=False)

print(f"Generated {len(customers_df)} customers and saved to {customers_csv_path}")


# --- Subscription Configuration ---
PLANS = {
    "Basic": {"Monthly": 75, "Annual": 50},
    "Pro": {"Monthly": 135, "Annual": 120},
    "Enterprise": {"Monthly": 315, "Annual": 300},
}
PLAN_NAMES = list(PLANS.keys())
SUBSCRIPTION_TYPES = ["Monthly", "Annual"]

# Probabilities for simulation
INITIAL_PLAN_WEIGHTS = {"Basic": 0.6, "Pro": 0.3, "Enterprise": 0.1}
INITIAL_TYPE_WEIGHTS = {"Monthly": 0.7, "Annual": 0.3}
CHURN_RATE_MONTHLY = 0.05  # 5% monthly churn probability
CHURN_RATE_ANNUAL = 0.02  # 2% effective monthly churn for annual (lower overall)
UPGRADE_PROBABILITY = 0.08  # Chance to upgrade per period
DOWNGRADE_PROBABILITY = 0.03  # Chance to downgrade per period

# --- Generate Subscriptions ---
subscriptions_data = []
# Ensure AcquisitionDate is datetime for comparison
customers_df["AcquisitionDate"] = pd.to_datetime(customers_df["AcquisitionDate"])

for _, customer in customers_df.iterrows():
    customer_id = customer["CustomerID"]
    acquisition_date = customer["AcquisitionDate"]
    # Start subscription sometime shortly after acquisition
    current_date = acquisition_date + timedelta(days=random.randint(1, 7))
    # Initial plan choice
    current_plan_index = PLAN_NAMES.index(
        random.choices(PLAN_NAMES, weights=list(INITIAL_PLAN_WEIGHTS.values()), k=1)[0]
    )
    current_sub_type = random.choices(SUBSCRIPTION_TYPES, weights=list(INITIAL_TYPE_WEIGHTS.values()), k=1)[0]

    active_subscription = True
    while active_subscription and current_date < END_DATE:
        subscription_id = str(uuid.uuid4())
        plan_name = PLAN_NAMES[current_plan_index]
        start_date = current_date

        # Calculate end date and price
        if current_sub_type == "Annual":
            end_date = start_date + timedelta(days=365)
            monthly_price = PLANS[plan_name]["Annual"]
            # Effective monthly churn probability for annual subscribers
            churn_prob = CHURN_RATE_ANNUAL
            period_months = 12
        else:  # Monthly
            end_date = start_date + timedelta(days=30)  # Approximate month duration
            monthly_price = PLANS[plan_name]["Monthly"]
            churn_prob = CHURN_RATE_MONTHLY
            period_months = 1

        end_date = min(end_date, END_DATE)

        status = "Active"  # Default status for the current period

        # Determine the start date for the *next* potential subscription period
        next_date = end_date + timedelta(days=1)

        # Simulate events only if the current period ends *before* the global END_DATE
        if end_date < END_DATE:
            # Check for churn first
            # Scale churn probability by the length of the subscription period
            if random.random() < churn_prob * period_months:
                status = "Churned"
                active_subscription = False  # Stop simulation for this customer
                # Churn occurs sometime *during* the period, not exactly at the end
                end_date = current_date + timedelta(days=random.randint(5, period_months * 30 - 1))
                # Ensure churn date doesn't exceed global END_DATE
                if end_date > END_DATE:
                    end_date = END_DATE
                next_date = None  # No next subscription if churned
            else:
                # If not churned, check for Upgrade/Downgrade
                action_rand = random.random()
                # Upgrade if possible
                if action_rand < UPGRADE_PROBABILITY and current_plan_index < len(PLAN_NAMES) - 1:
                    status = "Upgraded"
                    current_plan_index += 1  # Set plan for the *next* period
                # Downgrade if possible
                elif action_rand < (UPGRADE_PROBABILITY + DOWNGRADE_PROBABILITY) and current_plan_index > 0:
                    status = "Downgraded"
                    current_plan_index -= 1  # Set plan for the *next* period
                # Else: If no upgrade/downgrade, the status remains "Active"
                # and the plan carries over to the next period (implicit renewal)

        else:  # Current subscription period reaches or exceeds global END_DATE
            status = "Active"  # Customer is still active at the end of the simulation period
            active_subscription = False  # Stop simulating for this customer
            next_date = None  # No subscription events after global END_DATE

        subscriptions_data.append(
            {
                "SubscriptionID": subscription_id,
                "CustomerID": customer_id,
                "PlanName": plan_name,
                "SubscriptionType": current_sub_type,
                "StartDate": start_date.strftime("%Y-%m-%d"),
                "EndDate": end_date.strftime("%Y-%m-%d") if end_date else None,
                "Status": status,  # Status reflects outcome *at the end* of this period
                "MonthlyPrice": monthly_price,
            }
        )

        # Move to the next period or stop
        if next_date:
            current_date = next_date
        else:
            break  # Stop simulation loop if churned or reached end date

subscriptions_df = pd.DataFrame(subscriptions_data)

# --- Save Subscriptions CSV ---
subscriptions_csv_path = OUTPUT_DIR / "subscriptions.csv"
subscriptions_df.to_csv(subscriptions_csv_path, index=False)

print(f"Generated {len(subscriptions_df)} subscription records and saved to {subscriptions_csv_path}")


# --- Order Configuration ---
# Define COGS margins per plan (example values, adjust as needed)
COGS_MARGINS = {
    "Basic": 0.20,  # 20% COGS for Basic
    "Pro": 0.18,  # 18% COGS for Pro
    "Enterprise": 0.15,  # 15% COGS for Enterprise
}
DEFAULT_COGS_MARGIN = 0.20  # Fallback if plan name isn't found

# --- Generate Orders ---
orders_data = []
# Ensure dates are datetime objects (should already be done from subscriptions)
subscriptions_df["StartDate"] = pd.to_datetime(subscriptions_df["StartDate"])
subscriptions_df["EndDate"] = pd.to_datetime(subscriptions_df["EndDate"], errors="coerce")

for _, sub in subscriptions_df.iterrows():
    customer_id = sub["CustomerID"]
    subscription_id = sub["SubscriptionID"]
    plan_name = sub["PlanName"]
    sub_type = sub["SubscriptionType"]
    monthly_price = sub["MonthlyPrice"]
    start_date = sub["StartDate"]
    # Use the actual end date for the subscription period, or global END_DATE if still active/not churned early
    end_date = sub["EndDate"] if pd.notna(sub["EndDate"]) else END_DATE  # type: ignore

    # Determine billing frequency and amount per billing cycle
    if sub_type == "Annual":
        billing_period_delta = relativedelta.relativedelta(years=1)
        billing_amount = monthly_price * 12
    else:  # Monthly
        billing_period_delta = relativedelta.relativedelta(months=1)
        billing_amount = monthly_price

    # Get the correct COGS margin for this plan
    cogs_margin = COGS_MARGINS.get(str(plan_name), DEFAULT_COGS_MARGIN)
    profit_multiplier = 1 - cogs_margin

    # Generate orders for each billing cycle within the subscription period
    current_billing_date = pd.to_datetime(start_date)
    while current_billing_date <= pd.to_datetime(end_date):
        # Only generate an order if the billing date is on or before the subscription's actual end date
        # This handles cases where a subscription ends mid-cycle (e.g., churn)
        if current_billing_date <= pd.to_datetime(end_date):
            profit = billing_amount * profit_multiplier
            orders_data.append(
                {
                    "OrderID": str(uuid.uuid4()),
                    "CustomerID": customer_id,
                    "SubscriptionID": subscription_id,
                    "OrderDate": current_billing_date.strftime("%Y-%m-%d"),
                    "PlanName": plan_name,  # Added Plan Name
                    "SubscriptionType": sub_type,  # Added Subscription Type
                    "Amount": round(billing_amount, 2),  # Revenue for this billing cycle
                    "Profit": round(profit, 2),  # Profit for this billing cycle
                }
            )

        # Move to the next potential billing date
        # Using simple timedelta; for perfect month/year alignment, dateutil.relativedelta is better
        current_billing_date += billing_period_delta

orders_df = pd.DataFrame(orders_data)

# --- Save Orders CSV ---
orders_csv_path = OUTPUT_DIR / "orders.csv"
# Sort orders chronologically for better readability (optional)
orders_df = orders_df.sort_values(by=["CustomerID", "OrderDate"])
orders_df.to_csv(orders_csv_path, index=False)

print(f"Generated {len(orders_df)} order records (incl. recurring) and saved to {orders_csv_path}")


# --- Marketing Spend Configuration ---
MARKETING_CHANNELS_SPEND = {
    "Paid Search": {"base_monthly": 5000, "variability": 0.2},  # Base spend $5k +/- 20%
    "Social Media Ads": {"base_monthly": 3000, "variability": 0.25},  # Base spend $3k +/- 25%
    "Email Marketing": {"base_monthly": 500, "variability": 0.1},  # Base spend $500 +/- 10% (platform costs etc.)
    "Affiliate Marketing": {"base_monthly": 1000, "variability": 0.3},  # Base spend $1k +/- 30% (commissions vary)
    "Content Marketing": {
        "base_monthly": 1500,
        "variability": 0.15,
    },  # Base spend $1.5k +/- 15% (freelancers, tools)
}
# Simulate slight growth in marketing budget over time
ANNUAL_BUDGET_GROWTH_RATE = 0.10  # 10% increase per year

# --- Generate Marketing Spend ---
marketing_spend_data = []
num_years = (END_DATE.year - START_DATE.year) + (
    END_DATE.month > START_DATE.month or (END_DATE.month == START_DATE.month and END_DATE.day >= START_DATE.day)
)

current_month_date = START_DATE.replace(day=1)  # Start from the first day of the start month

while current_month_date < END_DATE:
    # Calculate years passed for budget growth adjustment
    years_passed = (current_month_date.year - START_DATE.year) + (
        current_month_date.month - START_DATE.month
    ) / 12.0
    budget_multiplier = (1 + ANNUAL_BUDGET_GROWTH_RATE) ** years_passed

    for channel, config in MARKETING_CHANNELS_SPEND.items():
        base_spend = config["base_monthly"] * budget_multiplier
        variability = config["variability"]
        # Calculate spend for the month with random variation
        spend = base_spend * (1 + random.uniform(-variability, variability))
        # Ensure spend is not negative
        spend = max(0, spend)

        marketing_spend_data.append(
            {
                "Date": current_month_date.strftime("%Y-%m-%d"),  # Record spend for the start of the month
                "Channel": channel,
                "Spend": round(spend, 2),
            }
        )

    # Move to the next month
    current_month_date += relativedelta.relativedelta(months=1)

marketing_spend_df = pd.DataFrame(marketing_spend_data)

# --- Save Marketing Spend CSV ---
marketing_spend_csv_path = OUTPUT_DIR / "marketing_spend.csv"
marketing_spend_df.to_csv(marketing_spend_csv_path, index=False)

print(f"Generated {len(marketing_spend_df)} marketing spend records and saved to {marketing_spend_csv_path}")


# --- Usage Metrics Configuration ---
# Define baseline usage patterns (can be refined later, e.g., based on plan)
BASE_MONTHLY_LOGINS = 20
LOGIN_VARIABILITY = 0.5  # +/- 50%
BASE_MONTHLY_FEATURES = 50
FEATURES_VARIABILITY = 0.4  # +/- 40%

# --- Generate Usage Metrics ---
usage_metrics_data = []

# Ensure subscription dates are datetime objects (should already be done)
# subscriptions_df['StartDate'] = pd.to_datetime(subscriptions_df['StartDate'])
# subscriptions_df['EndDate'] = pd.to_datetime(subscriptions_df['EndDate'], errors='coerce')

# Generate a range of first days of months for the simulation period
start_month = START_DATE.replace(day=1)
end_month = END_DATE.replace(day=1)
current_month_iter = start_month

while current_month_iter < end_month:
    month_start_date = current_month_iter
    month_end_date = current_month_iter + relativedelta.relativedelta(months=1) - timedelta(days=1)

    # Find subscriptions active during this month
    # A subscription is active if:
    # - It started before or on the month's end date AND
    # - It ended on or after the month's start date OR has no end date (NaT)
    active_subs_this_month = subscriptions_df[
        (subscriptions_df["StartDate"] <= month_end_date)
        & ((subscriptions_df["EndDate"] >= month_start_date) | pd.isna(subscriptions_df["EndDate"]))
    ]

    # Get unique active customer IDs for the month
    active_customer_ids = active_subs_this_month["CustomerID"].unique()  # type: ignore

    for customer_id in active_customer_ids:
        # Simulate logins for the month
        logins = int(BASE_MONTHLY_LOGINS * (1 + random.uniform(-LOGIN_VARIABILITY, LOGIN_VARIABILITY)))
        logins = max(0, logins)  # Ensure non-negative

        # Simulate feature usage
        features_used = int(
            BASE_MONTHLY_FEATURES * (1 + random.uniform(-FEATURES_VARIABILITY, FEATURES_VARIABILITY))
        )
        features_used = max(0, features_used)  # Ensure non-negative

        usage_metrics_data.append(
            {
                "UsageID": str(uuid.uuid4()),
                "CustomerID": customer_id,
                "Date": month_start_date.strftime("%Y-%m-%d"),  # Usage recorded for the start of the month
                "Logins": logins,
                "FeaturesUsedCount": features_used,
            }
        )

    # Move to the next month
    current_month_iter += relativedelta.relativedelta(months=1)

usage_metrics_df = pd.DataFrame(usage_metrics_data)

# --- Save Usage Metrics CSV ---
usage_metrics_csv_path = OUTPUT_DIR / "usage_metrics.csv"
usage_metrics_df = usage_metrics_df.sort_values(by=["CustomerID", "Date"])  # Sort for readability
usage_metrics_df.to_csv(usage_metrics_csv_path, index=False)

print(f"Generated {len(usage_metrics_df)} usage metrics records and saved to {usage_metrics_csv_path}")


# --- Support Tickets Configuration ---
TICKET_PROBABILITY_PER_CUSTOMER_MONTH = 0.05  # 5% chance an active customer logs a ticket each month
ISSUE_TYPES = ["Billing", "Technical", "How-to", "Account Management"]
ISSUE_TYPE_WEIGHTS = [0.3, 0.4, 0.2, 0.1]  # Billing 30%, Tech 40%, How-to 20%, Account 10%
RESOLUTION_STATUSES = ["Resolved", "Open", "Escalated", "Closed - No Action"]
RESOLUTION_STATUS_WEIGHTS = [0.7, 0.1, 0.05, 0.15]  # Resolved 70%, Open 10%, Escalated 5%, Closed 15%

# --- Generate Support Tickets ---
support_tickets_data = []

# Re-use the monthly iteration logic
start_month = START_DATE.replace(day=1)
end_month = END_DATE.replace(day=1)
current_month_iter = start_month

while current_month_iter < end_month:
    month_start_date = current_month_iter
    month_end_date = current_month_iter + relativedelta.relativedelta(months=1) - timedelta(days=1)

    # Find subscriptions active during this month (same logic as usage metrics)
    active_subs_this_month = subscriptions_df[
        (subscriptions_df["StartDate"] <= month_end_date)
        & ((subscriptions_df["EndDate"] >= month_start_date) | pd.isna(subscriptions_df["EndDate"]))
    ]
    active_customer_ids = active_subs_this_month["CustomerID"].unique()  # type: ignore

    for customer_id in active_customer_ids:
        # Decide if this customer creates a ticket this month
        if random.random() < TICKET_PROBABILITY_PER_CUSTOMER_MONTH:
            # Generate a random date within the month for the ticket
            days_in_month = (month_end_date - month_start_date).days + 1
            ticket_date = month_start_date + timedelta(days=random.randint(0, days_in_month - 1))

            # Randomly select issue type and status based on weights
            issue_type = random.choices(ISSUE_TYPES, weights=ISSUE_TYPE_WEIGHTS, k=1)[0]
            resolution_status = random.choices(RESOLUTION_STATUSES, weights=RESOLUTION_STATUS_WEIGHTS, k=1)[0]

            support_tickets_data.append(
                {
                    "TicketID": str(uuid.uuid4()),
                    "CustomerID": customer_id,
                    "TicketDate": ticket_date.strftime("%Y-%m-%d"),
                    "IssueType": issue_type,
                    "ResolutionStatus": resolution_status,
                }
            )

    # Move to the next month
    current_month_iter += relativedelta.relativedelta(months=1)

support_tickets_df = pd.DataFrame(support_tickets_data)

# --- Save Support Tickets CSV ---
support_tickets_csv_path = OUTPUT_DIR / "support_tickets.csv"
support_tickets_df = support_tickets_df.sort_values(by=["CustomerID", "TicketDate"])  # Sort
support_tickets_df.to_csv(support_tickets_csv_path, index=False)

print(f"Generated {len(support_tickets_df)} support ticket records and saved to {support_tickets_csv_path}")


# --- Operational Costs Configuration ---
OPERATIONAL_COST_CATEGORIES = {
    "SEO": {"base_monthly": 2500, "variability": 0.1},  # SEO agency/tools cost +/- 10%
    "Software/Tools": {"base_monthly": 1200, "variability": 0.05},  # General SaaS tools, etc. +/- 5%
    "Salaries": {"base_monthly": 50000, "variability": 0.02},  # Example salaries +/- 2%
    "Rent/Utilities": {"base_monthly": 8000, "variability": 0.05},  # Example rent/utils +/- 5%
}
# Simulate potential slight increase in operational costs over time
ANNUAL_OPEX_GROWTH_RATE = 0.05  # 5% increase per year

# --- Generate Operational Costs ---
operational_costs_data = []

# Re-use the monthly iteration logic
start_month = START_DATE.replace(day=1)
end_month = END_DATE.replace(day=1)
current_month_iter = start_month

while current_month_iter < end_month:
    # Calculate years passed for cost growth adjustment
    years_passed = (current_month_iter.year - START_DATE.year) + (
        current_month_iter.month - START_DATE.month
    ) / 12.0
    cost_multiplier = (1 + ANNUAL_OPEX_GROWTH_RATE) ** years_passed

    for category, config in OPERATIONAL_COST_CATEGORIES.items():
        base_cost = config["base_monthly"] * cost_multiplier
        variability = config["variability"]
        # Calculate cost for the month with random variation
        cost = base_cost * (1 + random.uniform(-variability, variability))
        cost = max(0, cost)  # Ensure non-negative

        operational_costs_data.append(
            {
                "CostID": str(uuid.uuid4()),
                "Date": current_month_iter.strftime("%Y-%m-%d"),  # Cost recorded for the start of the month
                "CostCategory": category,
                "Amount": round(cost, 2),
                # "Description": f"{category} cost for {current_month_iter.strftime('%Y-%m')}" # Optional description
            }
        )

    # Move to the next month
    current_month_iter += relativedelta.relativedelta(months=1)

operational_costs_df = pd.DataFrame(operational_costs_data)

# --- Save Operational Costs CSV ---
operational_costs_csv_path = OUTPUT_DIR / "operational_costs.csv"
operational_costs_df = operational_costs_df.sort_values(by=["Date", "CostCategory"])  # Sort
operational_costs_df.to_csv(operational_costs_csv_path, index=False)

print(f"Generated {len(operational_costs_df)} operational cost records and saved to {operational_costs_csv_path}")
