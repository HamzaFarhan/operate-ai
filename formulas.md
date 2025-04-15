Below is a comprehensive list of Excel formulas and functions that many financial planners, analysts, and modelers commonly employ. These formulas range from basic arithmetic calculations to complex financial models. Each function is designed to work independently or be nested within other formulas when building multi-layered models for tasks such as forecasting, budgeting, and valuation.

Below you will find a categorized listing along with brief descriptions and common examples. (Keep in mind that this list covers many—but not every—Excel formula used in financial planning and analysis, and you may tailor it further to the specific needs of your analysis.)

────────────────────────
**1. Basic Arithmetic & Aggregation**  
These functions are the building blocks for financial summaries and aggregations.

• **SUM**  
  - **Purpose:** Add up a range of numbers.  
  - **Example:** `=SUM(A1:A10)`

• **AVERAGE**  
  - **Purpose:** Calculate the mean of a dataset.  
  - **Example:** `=AVERAGE(B1:B10)`

• **MIN** and **MAX**  
  - **Purpose:** Identify the smallest or largest numbers.  
  - **Examples:** `=MIN(C1:C10)`, `=MAX(C1:C10)`

• **PRODUCT**  
  - **Purpose:** Multiply values together.  
  - **Example:** `=PRODUCT(D1:D4)`

────────────────────────
**2. Conditional Aggregation & Counting**  
These functions allow you to work with data subsets based on specific criteria.

• **SUMIF** / **SUMIFS**  
  - **Purpose:** Sum numbers that meet one or more conditions.  
  - **Examples:**  
    `=SUMIF(range, criteria)`  
    `=SUMIFS(sum_range, criteria_range1, criteria1, [criteria_range2, criteria2], …)`

• **COUNTIF** / **COUNTIFS**  
  - **Purpose:** Count the number of cells that meet specified criteria.  
  - **Examples:**  
    `=COUNTIF(range, criteria)`  
    `=COUNTIFS(range1, criteria1, [range2, criteria2], …)`

• **AVERAGEIF** / **AVERAGEIFS**  
  - **Purpose:** Calculate the average of cells that fulfill the given criteria.  
  - **Examples:**  
    `=AVERAGEIF(range, criteria, [average_range])`  
    `=AVERAGEIFS(average_range, criteria_range1, criteria1, …)`

────────────────────────
**3. Lookup & Reference Functions**  
These are invaluable when you need to retrieve data from a table or array dynamically.

• **VLOOKUP** and **HLOOKUP**  
  - **Purpose:** Search for a value in a vertical or horizontal range.  
  - **Examples:**  
    `=VLOOKUP(lookup_value, table_array, col_index, [range_lookup])`  
    `=HLOOKUP(lookup_value, table_array, row_index, [range_lookup])`

• **INDEX** and **MATCH**  
  - **Purpose:** A powerful combination for more flexible lookups; INDEX returns a value at a given position, while MATCH finds the relative position of an item.  
  - **Examples:**  
    `=INDEX(return_range, MATCH(lookup_value, lookup_range, 0))`

• **XLOOKUP**  
  - **Purpose:** A modern, more flexible lookup function available in Excel 365 replacing VLOOKUP/HLOOKUP.  
  - **Example:** `=XLOOKUP(lookup_value, lookup_array, return_array, [if_not_found])`

• **OFFSET** and **INDIRECT**  
  - **Purpose:** Create dynamic ranges or references based on changing data.  
  - **Examples:**  
    `=OFFSET(reference, rows, cols, [height], [width])`  
    `=INDIRECT(ref_text)`

• **CHOOSE**  
  - **Purpose:** Return a value from a list based on a given index.  
  - **Example:** `=CHOOSE(index_num, value1, value2, …)`

────────────────────────
**4. Logical & Error-Handling Functions**  
These functions help structure decision-making processes and manage errors gracefully.

• **IF**  
  - **Purpose:** Return different values depending on whether a condition is met.  
  - **Example:** `=IF(A1 > 100, "Above Budget", "Within Budget")`

• **IFERROR** and **IFNA**  
  - **Purpose:** Return a specified value if a formula results in an error (including #N/A).  
  - **Examples:**  
    `=IFERROR(formula, alternative_value)`  
    `=IFNA(formula, alternative_value)`

• **IFS**  
  - **Purpose:** Test multiple conditions without nesting several IF statements.  
  - **Example:** `=IFS(A1>100, "High", A1>50, "Medium", TRUE, "Low")`

• **AND, OR, NOT**  
  - **Purpose:** Combine or modify logical conditions.  
  - **Examples:**  
    `=AND(condition1, condition2)`  
    `=OR(condition1, condition2)`  
    `=NOT(condition)`

────────────────────────
**5. Financial Functions**  
These functions are specifically tailored to perform time value of money calculations and asset valuations.

• **NPV (Net Present Value)**  
  - **Purpose:** Calculate the present value of a series of cash flows at a constant discount rate.  
  - **Example:** `=NPV(discount_rate, cash_flow1, cash_flow2, …)`

• **IRR (Internal Rate of Return)**  
  - **Purpose:** Determine the discount rate that makes the net present value (NPV) of cash flows zero.  
  - **Example:** `=IRR(cash_flow_range)`

• **XNPV** and **XIRR**  
  - **Purpose:** Handle cash flows that occur at irregular intervals by considering specific dates.  
  - **Examples:**  
    `=XNPV(discount_rate, cash_flow_range, date_range)`  
    `=XIRR(cash_flow_range, date_range)`

• **PMT (Payment)**  
  - **Purpose:** Calculate the payment for a loan based on constant payments and a constant interest rate.  
  - **Example:** `=PMT(interest_rate, number_of_periods, present_value)`

• **IPMT (Interest Payment)** and **PPMT (Principal Payment)**  
  - **Purpose:** Determine the interest or principal portion for a specific period of a loan payment.  
  - **Examples:**  
    `=IPMT(interest_rate, period, number_of_periods, present_value)`  
    `=PPMT(interest_rate, period, number_of_periods, present_value)`

• **PV (Present Value)** and **FV (Future Value)**  
  - **Purpose:** Compute the present or future value of an investment given a constant interest rate.  
  - **Examples:**  
    `=PV(interest_rate, number_of_periods, payment, [future_value])`  
    `=FV(interest_rate, number_of_periods, payment, [present_value])`

• **NPER (Number of Periods)** and **RATE**  
  - **Purpose:** Determine the duration of an investment or the interest rate per period for an annuity.  
  - **Examples:**  
    `=NPER(interest_rate, payment, present_value)`  
    `=RATE(number_of_periods, payment, present_value)`

• **CUMIPMT** and **CUMPRINC**  
  - **Purpose:** Calculate cumulative interest and principal payments over a range of periods.  
  - **Examples:**  
    `=CUMIPMT(interest_rate, number_of_periods, present_value, start_period, end_period, type)`  
    `=CUMPRINC(interest_rate, number_of_periods, present_value, start_period, end_period, type)`

• **Depreciation Functions (SLN, SYD, DDB, DB)**  
  - **Purpose:** Compute different methods of asset depreciation, such as straight-line, sum-of-years’ digits, double-declining balance, or fixed depreciation.  
  - **Examples:**  
    `=SLN(cost, salvage, life)`  
    `=SYD(cost, salvage, life, period)`  
    `=DDB(cost, salvage, life, period, [factor])`

• **Bond Pricing Functions (PRICE, YIELD, DURATION)**  
  - **Purpose:** Help price bonds and evaluate yield and duration measures.  
  - **Examples:**  
    `=PRICE(settlement, maturity, rate, yield, redemption, frequency, [basis])`  
    `=YIELD(settlement, maturity, rate, price, redemption, frequency, [basis])`  
    `=DURATION(settlement, maturity, coupon, yield, frequency, [basis])`

────────────────────────
**6. Date & Time Functions**  
Essential for forecasting, scheduling cash flows, and working with time series data.

• **TODAY** and **NOW**  
  - **Purpose:** Return the current date or the current date and time.  
  - **Examples:** `=TODAY()`, `=NOW()`

• **DATE, YEAR, MONTH, DAY**  
  - **Purpose:** Construct dates and extract components (year, month, day) from a given date.  
  - **Example:** `=DATE(2025, 4, 15)`

• **EDATE** and **EOMONTH**  
  - **Purpose:** Calculate dates a given number of months before or after a specified date, or to find the end of the month.  
  - **Examples:**  
    `=EDATE(start_date, months)`  
    `=EOMONTH(start_date, months)`

• **DATEDIF** and **YEARFRAC**  
  - **Purpose:** Determine the difference between dates.  
  - **Examples:**  
    `=DATEDIF(start_date, end_date, "unit")`  
    `=YEARFRAC(start_date, end_date)`

• **WORKDAY** and **NETWORKDAYS**  
  - **Purpose:** Return a future or past date excluding weekends and optionally holidays; count the working days between two dates.  
  - **Examples:**  
    `=WORKDAY(start_date, days, [holidays])`  
    `=NETWORKDAYS(start_date, end_date, [holidays])`

────────────────────────
**7. Text & Data Management Functions**  
Useful for generating labels, combining text, and cleaning up data reports.

• **CONCAT** and **CONCATENATE**  
  - **Purpose:** Merge text strings together.  
  - **Examples:**  
    `=CONCAT(text1, text2, …)`  
    `=CONCATENATE(text1, text2, …)`

• **TEXT**  
  - **Purpose:** Format numbers or dates as text with a specified format.  
  - **Example:** `=TEXT(A1, "0.00%")`

• **LEFT, RIGHT, MID**  
  - **Purpose:** Extract portions of a text string.  
  - **Examples:** `=LEFT(text, num_chars)`, `=MID(text, start_num, num_chars)`

• **LEN**  
  - **Purpose:** Count the number of characters in a text string.  
  - **Example:** `=LEN(text)`

• **FIND** and **SEARCH**  
  - **Purpose:** Locate one text string within another.  
  - **Examples:** `=FIND(find_text, within_text)` (case-sensitive) and `=SEARCH(find_text, within_text)` (not case-sensitive)

• **REPLACE** and **SUBSTITUTE**  
  - **Purpose:** Replace a portion of a text string with another text string.  
  - **Examples:** `=REPLACE(old_text, start_num, num_chars, new_text)` and `=SUBSTITUTE(text, old_text, new_text)`

────────────────────────
**8. Statistical & Trend Analysis Functions**  
These functions support forecasting and risk analysis by uncovering trends and relationships in data.

• **STDEV.P / STDEV.S**  
  - **Purpose:** Calculate the standard deviation for a full population or a sample.  
  - **Examples:** `=STDEV.P(data_range)`, `=STDEV.S(data_range)`

• **VAR.P / VAR.S**  
  - **Purpose:** Calculate variance for a population or sample.  
  - **Examples:** `=VAR.P(data_range)`, `=VAR.S(data_range)`

• **MEDIAN** and **MODE**  
  - **Purpose:** Determine the middle value or most frequently occurring value in a dataset.  
  - **Examples:** `=MEDIAN(data_range)`, `=MODE(data_range)`

• **CORREL** and **COVARIANCE.P / COVARIANCE.S**  
  - **Purpose:** Measure the relationship or covariance between two datasets.  
  - **Examples:** `=CORREL(range1, range2)`, `=COVARIANCE.P(range1, range2)`

• **TREND** and **FORECAST (or FORECAST.LINEAR)**  
  - **Purpose:** Predict future values based on historical data trends.  
  - **Examples:**  
    `=TREND(known_y’s, [known_x’s], [new_x’s])`  
    `=FORECAST(new_x, known_y’s, known_x’s)`

• **GROWTH**  
  - **Purpose:** Forecast exponential growth trends.  
  - **Example:** `=GROWTH(known_y’s, [known_x’s], [new_x’s])`

────────────────────────
**9. Array and Dynamic Spill Functions (Modern Excel)**  
These functions help in performing calculations across ranges and enabling dynamic results.

• **UNIQUE**  
  - **Purpose:** Extract a list of unique values from a range.  
  - **Example:** `=UNIQUE(range)`

• **SORT** and **SORTBY**  
  - **Purpose:** Sort data or arrays dynamically.  
  - **Examples:** `=SORT(range)`, `=SORTBY(array, by_array)`

• **FILTER**  
  - **Purpose:** Return only those records that meet specified conditions.  
  - **Example:** `=FILTER(range, condition)`

• **SEQUENCE**  
  - **Purpose:** Generate a list of sequential numbers in an array format.  
  - **Example:** `=SEQUENCE(rows, [columns], [start], [step])`

• **RAND** and **RANDBETWEEN**  
  - **Purpose:** Generate random numbers for simulations or stochastic analyses.  
  - **Examples:** `=RAND()` for a random number between 0 and 1, and `=RANDBETWEEN(lower, upper)` for a random integer between two values.

────────────────────────
**10. Additional Useful Functions**  
These functions further aid in analysis, documentation, or advanced computations.

• **FORMULATEXT**  
  - **Purpose:** Returns the formula in a referenced cell as text, which can help in auditing or documentation.  
  - **Example:** `=FORMULATEXT(A1)`

• **TRANSPOSE**  
  - **Purpose:** Converts rows to columns or vice versa, useful for rearranging data.  
  - **Example:** `=TRANSPOSE(A1:B10)`

────────────────────────
**Combining Functions in FP&A Models**

Financial models often require blending these functions. For example:  
  • A discounted cash flow (DCF) model might nest **XNPV** and **XIRR** within a broader framework that uses **IF** and **SUMIFS** to filter and calculate cash flows by period.  
  • Budget models can employ **SUMIFS** for aggregating expenses by category, combined with **VLOOKUP** or **INDEX/MATCH** to pull in actuals versus forecasts.  
  • Trend analysis might use **FORECAST.LINEAR** together with **STDEV.S** to assess volatility and expected returns.

Each formula listed here is designed to work on its own or be integrated into larger, more complex formulas, making Excel a powerful environment for comprehensive financial planning and analysis.

This collection should provide a solid foundation to structure a wide variety of FP&A tasks.