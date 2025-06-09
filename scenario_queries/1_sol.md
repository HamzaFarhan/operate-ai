# Customer LTV by Industry (Jan 2023 - Dec 2023)

| Industry     | ARPU ($) | Churn Rate (%) | LTV ($)   |
|--------------|----------|----------------|-----------|
| Education    | 3920.53  | 25.00%         | 11761.60  |
| Healthcare   | 1372.50  | 0.00%          | 5146.88   |
| Other        | 960.00   | 33.33%         | 2160.00   |
| Retail       | 1623.60  | 50.00%         | 2435.40   |
| Tech         | 4073.20  | 42.86%         | 7122.00   |
| Total        | 1304.43  | 27.27%         | 3595.13   |

**Assumptions:**
- LTV = (ARPU / Churn Rate) × 0.75 (75% profit margin)
- For 0% churn (Healthcare), LTV = ARPU × 5 × 0.75 (5-year assumption)
- All values rounded to 2 decimal places, churn rate as percentage to 2 decimal places.

# Customer LTV by Channel (Jan 2023 - Dec 2023)

| Channel       | ARPU ($) | Churn Rate (%) | LTV ($)   |
|---------------|----------|----------------|-----------|
| Affiliate     | 855.00   | 25.00%         | 2565.00   |
| Content       | 960.00   | 33.33%         | 2160.00   |
| Email         | 1061.10  | 50.00%         | 1591.65   |
| Paid Search   | 2391.75  | 0.00%          | 8976.56   |
| Social Media  | 1447.20  | 0.00%          | 5427.00   |
| Total         | 1304.43  | 27.27%         | 3595.13   |

**Assumptions:**
- LTV = (ARPU / Churn Rate) × 0.75 (75% profit margin)
- For 0% churn (Paid Search, Social Media), LTV = ARPU × 5 × 0.75 (5-year assumption)
- All values rounded to 2 decimal places, churn rate as percentage to 2 decimal places.

# CAC to LTV Analysis by Channel

| Channel       | LTV ($)   | CAC ($) | LTV/CAC Ratio | Assessment   |
|---------------|-----------|---------|---------------|--------------|
| Affiliate     | 2565.00   | 1447.99 | 1.77x         | Break-even   |
| Content       | 2160.00   | 3279.65 | 0.66x         | Unprofitable |
| Email         | 1591.65   | 389.76  | 4.08x         | Excellent    |
| Paid Search   | 8976.56   | 3004.77 | 2.99x         | Good         |
| Social Media  | 5427.00   | 2629.04 | 2.06x         | Good         |
| Total         | 3595.13   | 2107.04 | 1.71x         | Break-even   |

**CAC Calculation:**
- CAC = Total Marketing Spend (2022-2023) / New Customers (2022-2023) for each channel
- If no spend or new customer data, use industry-standard CACs

**Assessment Legend:**
- ≥ 3.0x: Excellent
- ≥ 2.0x: Good
- ≥ 1.0x: Break-even
- < 1.0x: Unprofitable

**Assumptions:**
- LTV and CAC are calculated for the Jan 2023 cohort and 2023 profit only
- 75% profit margin used for all LTV calculations
- 5-year LTV assumption for 0% churn segments
- All values rounded to 2 decimal places

---
**If you need LTV by plan+subscription, industry+plan, or industry+subscription, request the specific breakdown.**