import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from faker import Faker

fake = Faker()

# --- Configuration based on scenario_1.md ---

CUSTOMER_COUNT = 200
SIMULATION_START_DATE = date(2022, 1, 1)
SIMULATION_END_DATE = date(2024, 12, 31)
SIMULATION_DAYS = (SIMULATION_END_DATE - SIMULATION_START_DATE).days

PLANS = {
    "Basic": {"monthly": 75, "annual": 50 * 12, "mrr": 75, "arr": 50 * 12},
    "Pro": {"monthly": 135, "annual": 120 * 12, "mrr": 135, "arr": 120 * 12},
    "Enterprise": {"monthly": 315, "annual": 300 * 12, "mrr": 315, "arr": 300 * 12},
}

GEOGRAPHY_DIST = {"North America": 0.60, "Europe": 0.25, "Asia-Pacific": 0.15}
INDUSTRY_DIST = {"Tech": 0.35, "Healthcare": 0.25, "Retail": 0.20, "Education": 0.10, "Other": 0.10}
CUSTOMER_TYPE_DIST = {"SMBs": 0.70, "Mid-Market": 0.25, "Enterprise": 0.05}
ACQUISITION_CHANNEL_DIST = {"Paid Ads": 0.50, "Organic Search": 0.30, "Email Campaigns": 0.20}
SUBSCRIPTION_TYPE_DIST = {"annual": 0.65, "monthly": 0.35}
INITIAL_PLAN_DIST = {"Basic": 0.50, "Pro": 0.40, "Enterprise": 0.10}

# Simplified CAC per channel
CAC_BY_CHANNEL = {"Paid Ads": 250, "Organic Search": 50, "Email Campaigns": 150}

# Churn/Upgrade/Downgrade dynamics (simplified probabilities per month)
MONTHLY_CHURN_RATE = 0.02  # Overall base rate
MONTHLY_UPGRADE_RATE = 0.03
MONTHLY_DOWNGRADE_RATE = 0.01


def _get_random_item(dist: dict) -> str:
    """Selects a random item based on distribution probabilities."""
    items, weights = zip(*dist.items())
    return random.choices(items, weights=weights, k=1)[0]


def generate_saas_data(output_dir: str | Path = "data") -> None:
    """Generate realistic SaaS customer and subscription data based on scenario_1.md."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    customers_data = []
    subscription_events = []
    event_id_counter = 1

    print(f"Simulating {CUSTOMER_COUNT} customers from {SIMULATION_START_DATE} to {SIMULATION_END_DATE}...")

    for i in range(CUSTOMER_COUNT):
        customer_id = f"CUST_{1000 + i}"
        # Simulate acquisition date randomly within the simulation period
        acquisition_offset = random.randint(0, SIMULATION_DAYS // 2)  # Acquire in first half generally
        acquisition_date = SIMULATION_START_DATE + timedelta(days=acquisition_offset)

        # Assign customer attributes
        geography = _get_random_item(GEOGRAPHY_DIST)
        industry = _get_random_item(INDUSTRY_DIST)
        customer_type = _get_random_item(CUSTOMER_TYPE_DIST)
        acquisition_channel = _get_random_item(ACQUISITION_CHANNEL_DIST)
        initial_plan = _get_random_item(INITIAL_PLAN_DIST)
        initial_sub_type = _get_random_item(SUBSCRIPTION_TYPE_DIST)
        cac = CAC_BY_CHANNEL[acquisition_channel]

        customers_data.append(
            {
                "customer_id": customer_id,
                "acquisition_date": acquisition_date,
                "acquisition_channel": acquisition_channel,
                "geography": geography,
                "industry": industry,
                "customer_type": customer_type,
                "initial_plan": initial_plan,
                "initial_subscription_type": initial_sub_type,
                "cac": cac,
            }
        )

        # --- Simulate Customer Lifecycle Events ---
        current_plan = initial_plan
        current_sub_type = initial_sub_type
        current_date = acquisition_date
        is_active = True

        # Initial signup event
        mrr = PLANS[current_plan]["mrr"] if current_sub_type == "monthly" else 0
        arr = PLANS[current_plan]["arr"] if current_sub_type == "annual" else 0
        subscription_events.append(
            {
                "event_id": event_id_counter,
                "customer_id": customer_id,
                "event_date": current_date,
                "event_type": "signup",
                "previous_plan": None,
                "new_plan": current_plan,
                "subscription_type": current_sub_type,
                "mrr": mrr,
                "arr": arr,
            }
        )
        event_id_counter += 1

        # Simulate monthly events until simulation end or churn
        while is_active and current_date < SIMULATION_END_DATE:
            # Move to the start of the next month for simplicity
            next_month_start = (current_date.replace(day=1) + timedelta(days=32)).replace(day=1)
            if next_month_start > SIMULATION_END_DATE:
                break
            current_date = next_month_start

            # Renewal Check (simplified: assume renewal happens monthly/annually)
            # Actual renewals would depend on the initial sub_type and date

            # Check for Churn first
            churn_roll = random.random()
            # Adjust churn based on factors (e.g., Enterprise less likely, Retail more likely)
            churn_modifier = 1.0
            if customer_type == "Enterprise":
                churn_modifier *= 0.5
            if industry == "Retail":
                churn_modifier *= 1.5

            if churn_roll < (MONTHLY_CHURN_RATE * churn_modifier):
                is_active = False
                subscription_events.append(
                    {
                        "event_id": event_id_counter,
                        "customer_id": customer_id,
                        "event_date": current_date,
                        "event_type": "churn",
                        "previous_plan": current_plan,
                        "new_plan": None,
                        "subscription_type": None,
                        "mrr": -PLANS[current_plan]["mrr"] if current_sub_type == "monthly" else 0,
                        "arr": -PLANS[current_plan]["arr"] if current_sub_type == "annual" else 0,
                    }
                )
                event_id_counter += 1
                continue  # Stop simulating events for this customer

            # Check for Upgrade (cannot upgrade from Enterprise)
            upgrade_roll = random.random()
            if current_plan != "Enterprise" and upgrade_roll < MONTHLY_UPGRADE_RATE:
                old_plan = current_plan
                plan_levels = list(PLANS.keys())
                current_index = plan_levels.index(current_plan)
                new_plan = plan_levels[current_index + 1]
                current_plan = new_plan
                # Keep subscription type for simplicity on upgrade/downgrade

                mrr_change = (
                    (PLANS[new_plan]["mrr"] - PLANS[old_plan]["mrr"]) if current_sub_type == "monthly" else 0
                )
                arr_change = (
                    (PLANS[new_plan]["arr"] - PLANS[old_plan]["arr"]) if current_sub_type == "annual" else 0
                )

                subscription_events.append(
                    {
                        "event_id": event_id_counter,
                        "customer_id": customer_id,
                        "event_date": current_date,
                        "event_type": "upgrade",
                        "previous_plan": old_plan,
                        "new_plan": new_plan,
                        "subscription_type": current_sub_type,
                        "mrr": mrr_change,
                        "arr": arr_change,
                    }
                )
                event_id_counter += 1
                continue  # Assume only one event per month

            # Check for Downgrade (cannot downgrade from Basic)
            downgrade_roll = random.random()
            if current_plan != "Basic" and downgrade_roll < MONTHLY_DOWNGRADE_RATE:
                old_plan = current_plan
                plan_levels = list(PLANS.keys())
                current_index = plan_levels.index(current_plan)
                new_plan = plan_levels[current_index - 1]
                current_plan = new_plan

                mrr_change = (
                    (PLANS[new_plan]["mrr"] - PLANS[old_plan]["mrr"]) if current_sub_type == "monthly" else 0
                )
                arr_change = (
                    (PLANS[new_plan]["arr"] - PLANS[old_plan]["arr"]) if current_sub_type == "annual" else 0
                )

                subscription_events.append(
                    {
                        "event_id": event_id_counter,
                        "customer_id": customer_id,
                        "event_date": current_date,
                        "event_type": "downgrade",
                        "previous_plan": old_plan,
                        "new_plan": new_plan,
                        "subscription_type": current_sub_type,
                        "mrr": mrr_change,
                        "arr": arr_change,
                    }
                )
                event_id_counter += 1
                continue  # Assume only one event per month

    # --- Create DataFrames and Save ---
    customers_df = pd.DataFrame(customers_data)
    customers_df["acquisition_date"] = pd.to_datetime(customers_df["acquisition_date"])

    events_df = pd.DataFrame(subscription_events)
    events_df["event_date"] = pd.to_datetime(events_df["event_date"])

    customers_df.to_csv(output_path / "customers.csv", index=False, date_format="%Y-%m-%d")
    events_df.to_csv(output_path / "subscription_events.csv", index=False, date_format="%Y-%m-%d")

    # --- Clean up old files ---
    (output_path / "financial_records.json").unlink(missing_ok=True)
    (output_path / "employee_metrics.csv").unlink(missing_ok=True)
    (output_path / "quarterly_report.md").unlink(missing_ok=True)

    print(f"Generated SaaS data: {len(customers_df)} customers, {len(events_df)} events.")
    print(f"Output files: customers.csv, subscription_events.csv in '{output_path.resolve()}'")


if __name__ == "__main__":
    generate_saas_data()
