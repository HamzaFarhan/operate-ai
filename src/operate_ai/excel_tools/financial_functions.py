import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic_ai import RunContext
from scipy.optimize import newton, root_scalar

from operate_ai.excel_tools import AgentDeps


def npv_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, discount_rate: float, cash_flow_columns: list[str]
) -> dict[str, Any]:
    """
    Calculates the Net Present Value (NPV) of a series of cash flows at a constant discount rate (like Excel NPV).

    Parameters:
        df_name (str): Name of the CSV file to load.
        discount_rate (float): The discount rate to apply to the cash flows.
        cash_flow_columns (list[str]): List of columns representing cash flows in order.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "npv",
            "results": { "npv": float, "message": str },
            "formula": "=NPV(discount_rate, cash_flow1, cash_flow2, …)"
        }

    Errors:
        Returns an error message in results if the file can't be read or a column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)
        for col in cash_flow_columns:
            if col not in df.columns:
                raise ValueError(f"Cash flow column '{col}' not found.")
        # Assume each column is a period, take the first row for each
        cash_flows = [float(df[col].iloc[0]) for col in cash_flow_columns]
        npv = sum(cf / (1 + discount_rate) ** i for i, cf in enumerate(cash_flows, 1))
        return {
            "operation": "npv",
            "results": {
                "npv": npv,
                "message": f"NPV for columns {cash_flow_columns} at rate {discount_rate} is {npv}.",
            },
            "formula": f"=NPV({discount_rate}, {', '.join(cash_flow_columns)})",
        }
    except Exception as e:
        return {
            "operation": "npv",
            "results": {"error": str(e)},
            "formula": f"=NPV({discount_rate}, {', '.join(cash_flow_columns)})",
        }


def irr_formula_tool(ctx: RunContext[AgentDeps], df_name: str, cash_flow_column: str) -> dict[str, Any]:
    """
    Determines the Internal Rate of Return (IRR) that makes the NPV of cash flows zero (like Excel IRR).

    Parameters:
        df_name (str): Name of the CSV file to load.
        cash_flow_column (str): Column representing the cash flows.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "irr",
            "results": { "irr": float, "message": str },
            "formula": "=IRR(cash_flow_range)"
        }

    Errors:
        Returns an error message in results if the file can't be read or a column is missing, or if IRR cannot be determined.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)
        if cash_flow_column not in df.columns:
            raise ValueError(f"Cash flow column '{cash_flow_column}' not found.")
        try:
            cash_flows = [float(x) for x in df[cash_flow_column]]
        except Exception as e:
            raise ValueError(f"Non-numeric value in cash flow column: {e}")
        if not cash_flows or not isinstance(cash_flows[0], (int, float)) or cash_flows[0] >= 0:
            raise ValueError("First cash flow must be negative (investment outlay).")

        # IRR: find r such that sum(cf / (1 + r)**i) = 0
        def npv_func(r):
            return sum(cf / (1 + r) ** i for i, cf in enumerate(cash_flows, 0))

        try:
            irr = newton(npv_func, 0.1, maxiter=100)
        except RuntimeError:
            # Try bracketed root finding if newton fails
            try:
                sol = root_scalar(npv_func, bracket=[-0.999, 10], method="bisect")
                if not sol.converged:
                    raise ValueError("IRR root finding did not converge.")
                irr = sol.root
            except Exception as e2:
                return {
                    "operation": "irr",
                    "results": {"error": f"IRR could not be determined: {e2}"},
                    "formula": f"=IRR({cash_flow_column})",
                }
        return {
            "operation": "irr",
            "results": {"irr": irr, "message": f"IRR for column {cash_flow_column} is {irr}."},
            "formula": f"=IRR({cash_flow_column})",
        }
    except Exception as e:
        return {
            "operation": "irr",
            "results": {"error": str(e)},
            "formula": f"=IRR({cash_flow_column})",
        }


def xnpv_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, discount_rate: float, cash_flow_column: str, date_column: str
) -> dict[str, Any]:
    """
    Calculates the Net Present Value for cash flows occurring at irregular intervals (like Excel XNPV).

    Parameters:
        df_name (str): Name of the CSV file to load.
        discount_rate (float): The discount rate to apply to the cash flows.
        cash_flow_column (str): Column representing the cash flows.
        date_column (str): Column representing the dates of the cash flows.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "xnpv",
            "results": { "xnpv": float, "message": str },
            "formula": "=XNPV(discount_rate, cash_flow_range, date_range)"
        }

    Errors:
        Returns an error message in results if the file can't be read or a column is missing.
    """

    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)
        if cash_flow_column not in df.columns:
            raise ValueError(f"Cash flow column '{cash_flow_column}' not found.")
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found.")
        cash_flows = df[cash_flow_column].astype(float).tolist()
        dates = pd.to_datetime(df[date_column])
        if len(cash_flows) != len(dates):
            raise ValueError("Cash flows and dates must have the same length.")
        t0 = dates.iloc[0]
        xnpv = sum(cf / (1 + discount_rate) ** ((d - t0).days / 365.0) for cf, d in zip(cash_flows, dates))
        return {
            "operation": "xnpv",
            "results": {
                "xnpv": xnpv,
                "message": f"XNPV for {cash_flow_column} at rate {discount_rate} is {xnpv}.",
            },
            "formula": f"=XNPV({discount_rate}, {cash_flow_column}, {date_column})",
        }
    except Exception as e:
        return {
            "operation": "xnpv",
            "results": {"error": str(e)},
            "formula": f"=XNPV({discount_rate}, {cash_flow_column}, {date_column})",
        }


def xirr_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, cash_flow_column: str, date_column: str
) -> dict[str, Any]:
    """
    Calculates the Internal Rate of Return for cash flows at irregular intervals (like Excel XIRR).

    Parameters:
        df_name (str): Name of the CSV file to load.
        cash_flow_column (str): Column representing the cash flows.
        date_column (str): Column representing the dates of the cash flows.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "xirr",
            "results": { "xirr": float, "message": str },
            "formula": "=XIRR(cash_flow_range, date_range)"
        }

    Errors:
        Returns an error message in results if the file can't be read or a column is missing.
    """

    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        df = pd.read_csv(file_path)
        if cash_flow_column not in df.columns:
            raise ValueError(f"Cash flow column '{cash_flow_column}' not found.")
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found.")
        cash_flows = df[cash_flow_column].astype(float).tolist()
        dates = pd.to_datetime(df[date_column])
        if len(cash_flows) != len(dates):
            raise ValueError("Cash flows and dates must have the same length.")
        t0 = dates.iloc[0]

        def xnpv_func(rate):
            return sum(cf / (1 + rate) ** ((d - t0).days / 365.0) for cf, d in zip(cash_flows, dates))

        # Initial guess: 0.1
        xirr = newton(xnpv_func, 0.1)
        return {
            "operation": "xirr",
            "results": {"xirr": xirr, "message": f"XIRR for {cash_flow_column} is {xirr}."},
            "formula": f"=XIRR({cash_flow_column}, {date_column})",
        }
    except Exception as e:
        return {
            "operation": "xirr",
            "results": {"error": str(e)},
            "formula": f"=XIRR({cash_flow_column}, {date_column})",
        }


def pmt_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, interest_rate: float, number_of_periods: int, present_value: float
) -> dict[str, Any]:
    """
    Calculates the payment for a loan based on constant payments and interest rate (like Excel PMT).

    Parameters:
        df_name (str): Name of the CSV file to load.
        interest_rate (float): The interest rate per period.
        number_of_periods (int): Number of payment periods.
        present_value (float): Present value (principal) of the loan.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "pmt",
            "results": { "payment": float, "message": str },
            "formula": "=PMT(interest_rate, number_of_periods, present_value)"
        }

    Errors:
        Returns an error message in results if the file can't be read or a column is missing.
    """

    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        pd.read_csv(file_path)  # Just to check file exists/valid
        # PMT formula: P = (r*PV) / (1 - (1 + r) ** -n)
        r = interest_rate
        n = number_of_periods
        pv = present_value
        if r == 0:
            payment = -pv / n
        else:
            payment = -pv * r / (1 - (1 + r) ** -n)
        return {
            "operation": "pmt",
            "results": {"payment": payment, "message": f"PMT for PV={pv}, r={r}, n={n} is {payment}."},
            "formula": f"=PMT({interest_rate}, {number_of_periods}, {present_value})",
        }
    except Exception as e:
        return {
            "operation": "pmt",
            "results": {"error": str(e)},
            "formula": f"=PMT({interest_rate}, {number_of_periods}, {present_value})",
        }


def ipmt_formula_tool(
    ctx: RunContext[AgentDeps],
    df_name: str,
    interest_rate: float,
    period: int,
    number_of_periods: int,
    present_value: float,
) -> dict[str, Any]:
    """
    Determines the interest portion of a loan payment for a specific period (like Excel IPMT).
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        pd.read_csv(file_path)  # Just to check file exists/valid
        r = interest_rate
        n = number_of_periods
        pv = present_value
        p = period
        if p < 1 or p > n:
            raise ValueError("Period must be between 1 and number_of_periods.")
        if r == 0:
            interest_payment = 0.0
        else:
            # Interest payment for period p
            # IPMT = (remaining principal at p-1) * r
            principal_before = pv * (1 + r) ** (p - 1) - (
                (-pv * r / (1 - (1 + r) ** -n)) * ((1 + r) ** (p - 1) - 1) / r
            )
            interest_payment = principal_before * r
        return {
            "operation": "ipmt",
            "results": {
                "interest_payment": interest_payment,
                "message": f"Interest payment at period {p} is {interest_payment}.",
            },
            "formula": f"=IPMT({r}, {p}, {n}, {pv})",
        }
    except Exception as e:
        return {
            "operation": "ipmt",
            "results": {"error": str(e)},
            "formula": f"=IPMT({interest_rate}, {period}, {number_of_periods}, {present_value})",
        }


def ppmt_formula_tool(
    ctx: RunContext[AgentDeps],
    df_name: str,
    interest_rate: float,
    period: int,
    number_of_periods: int,
    present_value: float,
) -> dict[str, Any]:
    """
    Determines the principal portion of a loan payment for a specific period (like Excel PPMT).
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        pd.read_csv(file_path)  # Just to check file exists/valid
        r = interest_rate
        n = number_of_periods
        pv = present_value
        p = period
        if p < 1 or p > n:
            raise ValueError("Period must be between 1 and number_of_periods.")
        if r == 0:
            payment = -pv / n
            principal_payment = payment
        else:
            payment = -pv * r / (1 - (1 + r) ** -n)
            # Principal payment = payment - interest payment
            principal_before = pv * (1 + r) ** (p - 1) - (payment * ((1 + r) ** (p - 1) - 1) / r)
            interest_payment = principal_before * r
            principal_payment = payment - interest_payment
        return {
            "operation": "ppmt",
            "results": {
                "principal_payment": principal_payment,
                "message": f"Principal payment at period {p} is {principal_payment}.",
            },
            "formula": f"=PPMT({r}, {p}, {n}, {pv})",
        }
    except Exception as e:
        return {
            "operation": "ppmt",
            "results": {"error": str(e)},
            "formula": f"=PPMT({interest_rate}, {period}, {number_of_periods}, {present_value})",
        }


def pv_formula_tool(
    ctx: RunContext[AgentDeps],
    df_name: str,
    interest_rate: float,
    number_of_periods: int,
    payment: float,
    future_value: float = 0.0,
) -> dict[str, Any]:
    """
    Computes the present value of an investment given a constant interest rate (like Excel PV).
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        pd.read_csv(file_path)  # Just to check file exists/valid
        r = interest_rate
        n = number_of_periods
        fv = future_value
        pmt = payment
        if r == 0:
            present_value = -pmt * n - fv
        else:
            present_value = -pmt * (1 - (1 + r) ** -n) / r - fv / (1 + r) ** n
        return {
            "operation": "pv",
            "results": {"present_value": present_value, "message": f"Present value is {present_value}."},
            "formula": f"=PV({r}, {n}, {pmt}, {fv})",
        }
    except Exception as e:
        return {
            "operation": "pv",
            "results": {"error": str(e)},
            "formula": f"=PV({interest_rate}, {number_of_periods}, {payment}, {future_value})",
        }


def fv_formula_tool(
    ctx: RunContext[AgentDeps],
    df_name: str,
    interest_rate: float,
    number_of_periods: int,
    payment: float,
    present_value: float = 0.0,
) -> dict[str, Any]:
    """
    Computes the future value of an investment given a constant interest rate (like Excel FV).
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        pd.read_csv(file_path)  # Just to check file exists/valid
        r = interest_rate
        n = number_of_periods
        pmt = payment
        pv = present_value
        if r == 0:
            future_value = -pv - pmt * n
        else:
            future_value = -pv * (1 + r) ** n - pmt * ((1 + r) ** n - 1) / r
        return {
            "operation": "fv",
            "results": {"future_value": future_value, "message": f"Future value is {future_value}."},
            "formula": f"=FV({r}, {n}, {pmt}, {pv})",
        }
    except Exception as e:
        return {
            "operation": "fv",
            "results": {"error": str(e)},
            "formula": f"=FV({interest_rate}, {number_of_periods}, {payment}, {present_value})",
        }


def nper_formula_tool(
    ctx: RunContext[AgentDeps],
    df_name: str,
    interest_rate: float,
    payment: float,
    present_value: float,
    future_value: float = 0.0,
) -> dict[str, Any]:
    """
    Computes the number of periods for an investment based on periodic, constant payments and a constant interest rate (like Excel NPER).
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        pd.read_csv(file_path)  # Just to check file exists/valid
        r = interest_rate
        pmt = payment
        pv = present_value
        fv = future_value
        if r == 0:
            if pmt == 0:
                raise ValueError("Payment cannot be zero if interest rate is zero.")
            number_of_periods = -(pv + fv) / pmt
        else:
            # Excel NPER: n = [ln((pmt/r + fv) / (pmt/r + pv))] / ln(1 + r)
            try:
                nper_num = pmt / r + fv
                nper_den = pmt / r + pv
                if nper_den == 0 or nper_num <= 0 or nper_den <= 0:
                    raise ValueError("Invalid parameters for log.")
                number_of_periods = (np.log(nper_num) - np.log(nper_den)) / np.log(1 + r)
            except Exception as e:
                raise ValueError(f"Error computing NPER: {e}")
        return {
            "operation": "nper",
            "results": {
                "number_of_periods": number_of_periods,
                "message": f"Number of periods is {number_of_periods}.",
            },
            "formula": f"=NPER({r}, {pmt}, {pv}, {fv})",
        }
    except Exception as e:
        return {
            "operation": "nper",
            "results": {"error": str(e)},
            "formula": f"=NPER({interest_rate}, {payment}, {present_value}, {future_value})",
        }


def rate_formula_tool(
    ctx: RunContext[AgentDeps],
    df_name: str,
    number_of_periods: int,
    payment: float,
    present_value: float,
    future_value: float = 0.0,
) -> dict[str, Any]:
    """
    Computes the interest rate per period of an annuity (like Excel RATE).
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        pd.read_csv(file_path)  # Just to check file exists/valid
        n = number_of_periods
        pmt = payment
        pv = present_value
        fv = future_value

        def rate_func(r):
            if r == 0:
                return pv + pmt * n + fv
            return pv * (1 + r) ** n + pmt * ((1 + r) ** n - 1) / r + fv

        try:
            rate = newton(rate_func, 0.1, maxiter=100)
        except RuntimeError:
            try:
                sol = root_scalar(rate_func, bracket=[-0.999, 10], method="bisect")
                if not sol.converged:
                    raise ValueError("RATE root finding did not converge.")
                rate = sol.root
            except Exception as e2:
                return {
                    "operation": "rate",
                    "results": {"error": f"RATE could not be determined: {e2}"},
                    "formula": f"=RATE({n}, {pmt}, {pv}, {fv})",
                }
        return {
            "operation": "rate",
            "results": {"rate": rate, "message": f"Rate per period is {rate}."},
            "formula": f"=RATE({n}, {pmt}, {pv}, {fv})",
        }
    except Exception as e:
        return {
            "operation": "rate",
            "results": {"error": str(e)},
            "formula": f"=RATE({number_of_periods}, {payment}, {present_value}, {future_value})",
        }


def cumipmt_formula_tool(
    ctx: RunContext[AgentDeps],
    df_name: str,
    interest_rate: float,
    number_of_periods: int,
    present_value: float,
    start_period: int,
    end_period: int,
    payment_type: int,
) -> dict[str, Any]:
    """
    Calculates cumulative interest payments over a range of periods (like Excel CUMIPMT).
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        pd.read_csv(file_path)  # Just to check file exists/valid
        r = interest_rate
        n = number_of_periods
        pv = present_value
        start = start_period
        end = end_period
        typ = payment_type
        if r == 0:
            raise ValueError("Interest rate must not be zero for CUMIPMT.")
        if start < 1 or end > n or start > end:
            raise ValueError("Periods must be between 1 and number_of_periods, and start_period <= end_period.")
        if typ not in (0, 1):
            raise ValueError("payment_type must be 0 (end) or 1 (beginning) of period.")
        payment = -pv * r / (1 - (1 + r) ** -n)
        cumulative_interest = 0.0
        for p in range(start, end + 1):
            if typ == 0:
                principal_before = pv * (1 + r) ** (p - 1) - (payment * ((1 + r) ** (p - 1) - 1) / r)
            else:
                principal_before = (
                    (pv * (1 + r) ** (p - 2) - (payment * ((1 + r) ** (p - 2) - 1) / r)) if p > 1 else pv
                )
            interest_payment = principal_before * r
            cumulative_interest += interest_payment
        return {
            "operation": "cumipmt",
            "results": {
                "cumulative_interest": -abs(cumulative_interest),
                "message": f"Cumulative interest paid from period {start} to {end} is {-abs(cumulative_interest)}.",
            },
            "formula": f"=CUMIPMT({r}, {n}, {pv}, {start}, {end}, {typ})",
        }
    except Exception as e:
        return {
            "operation": "cumipmt",
            "results": {"error": str(e)},
            "formula": f"=CUMIPMT({interest_rate}, {number_of_periods}, {present_value}, {start_period}, {end_period}, {payment_type})",
        }


def cumprinc_formula_tool(
    ctx: RunContext[AgentDeps],
    df_name: str,
    interest_rate: float,
    number_of_periods: int,
    present_value: float,
    start_period: int,
    end_period: int,
    payment_type: int,
) -> dict[str, Any]:
    """
    Calculates cumulative principal payments over a range of periods (like Excel CUMPRINC).
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        pd.read_csv(file_path)  # Just to check file exists/valid
        r = interest_rate
        n = number_of_periods
        pv = present_value
        start = start_period
        end = end_period
        typ = payment_type
        if r == 0:
            raise ValueError("Interest rate must not be zero for CUMPRINC.")
        if start < 1 or end > n or start > end:
            raise ValueError("Periods must be between 1 and number_of_periods, and start_period <= end_period.")
        if typ not in (0, 1):
            raise ValueError("payment_type must be 0 (end) or 1 (beginning) of period.")
        payment = -pv * r / (1 - (1 + r) ** -n)
        cumulative_principal = 0.0
        for p in range(start, end + 1):
            if typ == 0:
                principal_before = pv * (1 + r) ** (p - 1) - (payment * ((1 + r) ** (p - 1) - 1) / r)
            else:
                principal_before = (
                    (pv * (1 + r) ** (p - 2) - (payment * ((1 + r) ** (p - 2) - 1) / r)) if p > 1 else pv
                )
            interest_payment = principal_before * r
            principal_payment = payment - interest_payment
            cumulative_principal += principal_payment
        return {
            "operation": "cumprinc",
            "results": {
                "cumulative_principal": -abs(cumulative_principal),
                "message": f"Cumulative principal paid from period {start} to {end} is {-abs(cumulative_principal)}.",
            },
            "formula": f"=CUMPRINC({r}, {n}, {pv}, {start}, {end}, {typ})",
        }
    except Exception as e:
        return {
            "operation": "cumprinc",
            "results": {"error": str(e)},
            "formula": f"=CUMPRINC({interest_rate}, {number_of_periods}, {present_value}, {start_period}, {end_period}, {payment_type})",
        }


def sln_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, cost: float, salvage: float, life: int
) -> dict[str, Any]:
    """
    Computes straight-line depreciation of an asset (like Excel SLN).

    Parameters:
        df_name (str): Name of the CSV file to load.
        cost (float): Initial cost of the asset.
        salvage (float): Value at the end of depreciation.
        life (int): Useful life of the asset.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "sln",
            "results": { "depreciation": float, "message": str },
            "formula": "=SLN(cost, salvage, life)"
        }

    Errors:
        Returns an error message in results if the file can't be read or a column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        pd.read_csv(file_path)  # Just to check file exists/valid
        if life <= 0:
            raise ValueError("Life must be positive.")
        if cost < salvage:
            raise ValueError("Cost must be greater than or equal to salvage value.")
        depreciation = (cost - salvage) / life
        return {
            "operation": "sln",
            "results": {
                "depreciation": depreciation,
                "message": f"Straight-line depreciation per period is {depreciation}.",
            },
            "formula": f"=SLN({cost}, {salvage}, {life})",
        }
    except Exception as e:
        return {
            "operation": "sln",
            "results": {"error": str(e)},
            "formula": f"=SLN({cost}, {salvage}, {life})",
        }


def syd_formula_tool(
    ctx: RunContext[AgentDeps], df_name: str, cost: float, salvage: float, life: int, period: int
) -> dict[str, Any]:
    """
    Computes sum-of-years’ digits depreciation of an asset (like Excel SYD).

    Parameters:
        df_name (str): Name of the CSV file to load.
        cost (float): Initial cost of the asset.
        salvage (float): Value at the end of depreciation.
        life (int): Useful life of the asset.
        period (int): Period for which depreciation is calculated.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "syd",
            "results": { "depreciation": float, "message": str },
            "formula": "=SYD(cost, salvage, life, period)"
        }

    Errors:
        Returns an error message in results if the file can't be read or a column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        pd.read_csv(file_path)  # Just to check file exists/valid
        if life <= 0:
            raise ValueError("Life must be positive.")
        if period < 1 or period > life:
            raise ValueError("Period must be between 1 and useful life.")
        if cost < salvage:
            raise ValueError("Cost must be greater than or equal to salvage value.")
        syd = life * (life + 1) / 2
        depreciation = ((cost - salvage) * (life - period + 1)) / syd
        return {
            "operation": "syd",
            "results": {
                "depreciation": depreciation,
                "message": f"SYD depreciation for period {period} is {depreciation}.",
            },
            "formula": f"=SYD({cost}, {salvage}, {life}, {period})",
        }
    except Exception as e:
        return {
            "operation": "syd",
            "results": {"error": str(e)},
            "formula": f"=SYD({cost}, {salvage}, {life}, {period})",
        }


def ddb_formula_tool(
    ctx: RunContext[AgentDeps],
    df_name: str,
    cost: float,
    salvage: float,
    life: int,
    period: int,
    factor: float = 2.0,
) -> dict[str, Any]:
    """
    Computes double-declining balance depreciation of an asset (like Excel DDB).

    Parameters:
        df_name (str): Name of the CSV file to load.
        cost (float): Initial cost of the asset.
        salvage (float): Value at the end of depreciation.
        life (int): Useful life of the asset.
        period (int): Period for which depreciation is calculated.
        factor (float, optional): Acceleration factor. Default is 2.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "ddb",
            "results": { "depreciation": float, "message": str },
            "formula": "=DDB(cost, salvage, life, period, [factor])"
        }

    Errors:
        Returns an error message in results if the file can't be read or a column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        pd.read_csv(file_path)  # Just to check file exists/valid
        if life <= 0:
            raise ValueError("Life must be positive.")
        if period < 1 or period > life:
            raise ValueError("Period must be between 1 and useful life.")
        if cost < salvage:
            raise ValueError("Cost must be greater than or equal to salvage value.")
        if factor <= 0:
            raise ValueError("Factor must be positive.")
        depreciation = 0.0
        book_value = cost
        for p in range(1, period + 1):
            dep = min(book_value * factor / life, book_value - salvage) if (book_value - salvage) > 0 else 0
            if p == period:
                depreciation = dep
            book_value -= dep
        return {
            "operation": "ddb",
            "results": {
                "depreciation": depreciation,
                "message": f"DDB depreciation for period {period} is {depreciation}.",
            },
            "formula": f"=DDB({cost}, {salvage}, {life}, {period}, {factor})",
        }
    except Exception as e:
        return {
            "operation": "ddb",
            "results": {"error": str(e)},
            "formula": f"=DDB({cost}, {salvage}, {life}, {period}, {factor})",
        }


def price_formula_tool(
    ctx: RunContext[AgentDeps],
    df_name: str,
    settlement: str,
    maturity: str,
    rate: float,
    yield_: float,
    redemption: float,
    frequency: int,
    basis: int = 0,
) -> dict[str, Any]:
    """
    Calculates the price of a bond (like Excel PRICE).

    Parameters:
        df_name (str): Name of the CSV file to load.
        settlement (str): Settlement date (YYYY-MM-DD).
        maturity (str): Maturity date (YYYY-MM-DD).
        rate (float): Coupon rate.
        yield_ (float): Yield to maturity.
        redemption (float): Redemption value per $100 face value.
        frequency (int): Number of coupon payments per year.
        basis (int, optional): Day count basis. Default is 0.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "price",
            "results": { "price": float, "message": str },
            "formula": "=PRICE(settlement, maturity, rate, yield, redemption, frequency, [basis])"
        }

    Errors:
        Returns an error message in results if the file can't be read or a column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        pd.read_csv(file_path)  # Just to check file exists/valid
        # For simplicity, use a basic bond price formula (annual coupon, no day count adjustment)
        # price = sum(C/(1+y)^t) + redemption/(1+y)^N
        import datetime

        settlement_dt = datetime.datetime.strptime(settlement, "%Y-%m-%d")
        maturity_dt = datetime.datetime.strptime(maturity, "%Y-%m-%d")
        N = int(
            (maturity_dt.year - settlement_dt.year) * frequency
            + (maturity_dt.month - settlement_dt.month) / (12 / frequency)
        )
        if N <= 0:
            raise ValueError("Maturity must be after settlement.")
        C = rate * 100 / frequency
        y = yield_
        price = sum(C / (1 + y / frequency) ** t for t in range(1, N + 1)) + redemption / (1 + y / frequency) ** N
        return {
            "operation": "price",
            "results": {
                "price": price,
                "message": f"Bond price is {price}.",
            },
            "formula": f"=PRICE({settlement}, {maturity}, {rate}, {yield_}, {redemption}, {frequency}, {basis})",
        }
    except Exception as e:
        return {
            "operation": "price",
            "results": {"error": str(e)},
            "formula": f"=PRICE({settlement}, {maturity}, {rate}, {yield_}, {redemption}, {frequency}, {basis})",
        }


def yield_formula_tool(
    ctx: RunContext[AgentDeps],
    df_name: str,
    settlement: str,
    maturity: str,
    rate: float,
    price: float,
    redemption: float,
    frequency: int,
    basis: int = 0,
) -> dict[str, Any]:
    """
    Calculates the yield of a bond (like Excel YIELD).

    Parameters:
        df_name (str): Name of the CSV file to load.
        settlement (str): Settlement date (YYYY-MM-DD).
        maturity (str): Maturity date (YYYY-MM-DD).
        rate (float): Coupon rate.
        price (float): Price of the bond.
        redemption (float): Redemption value per $100 face value.
        frequency (int): Number of coupon payments per year.
        basis (int, optional): Day count basis. Default is 0.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "yield",
            "results": { "yield": float, "message": str },
            "formula": "=YIELD(settlement, maturity, rate, price, redemption, frequency, [basis])"
        }

    Errors:
        Returns an error message in results if the file can't be read or a column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        pd.read_csv(file_path)  # Just to check file exists/valid
        # To solve for yield, use a root-finding method so that price_formula(y) = price

        settlement_dt = datetime.datetime.strptime(settlement, "%Y-%m-%d")
        maturity_dt = datetime.datetime.strptime(maturity, "%Y-%m-%d")
        N = int(
            (maturity_dt.year - settlement_dt.year) * frequency
            + (maturity_dt.month - settlement_dt.month) / (12 / frequency)
        )
        if N <= 0:
            raise ValueError("Maturity must be after settlement.")
        C = rate * 100 / frequency

        def price_for_y(y):
            return (
                sum(C / (1 + y / frequency) ** t for t in range(1, N + 1)) + redemption / (1 + y / frequency) ** N
            )

        def f(y):
            return price_for_y(y) - price

        y0 = rate  # initial guess
        ytm = newton(f, y0)
        return {
            "operation": "yield",
            "results": {
                "yield": ytm,
                "message": f"Yield to maturity is {ytm}.",
            },
            "formula": f"=YIELD({settlement}, {maturity}, {rate}, {price}, {redemption}, {frequency}, {basis})",
        }
    except Exception as e:
        return {
            "operation": "yield",
            "results": {"error": str(e)},
            "formula": f"=YIELD({settlement}, {maturity}, {rate}, {price}, {redemption}, {frequency}, {basis})",
        }


def duration_formula_tool(
    ctx: RunContext[AgentDeps],
    df_name: str,
    settlement: str,
    maturity: str,
    coupon: float,
    yield_: float,
    redemption: float,
    frequency: int,
    basis: int = 0,
) -> dict[str, Any]:
    """
    Calculates the duration of a bond (like Excel DURATION).

    Parameters:
        df_name (str): Name of the CSV file to load.
        settlement (str): Settlement date (YYYY-MM-DD).
        maturity (str): Maturity date (YYYY-MM-DD).
        coupon (float): Coupon rate.
        yield_ (float): Yield to maturity.
        frequency (int): Number of coupon payments per year.
        basis (int, optional): Day count basis. Default is 0.
        ctx (RunContext): Runtime context containing the data directory path.

    Returns:
        dict[str, Any]: {
            "operation": "duration",
            "results": { "duration": float, "message": str },
            "formula": "=DURATION(settlement, maturity, coupon, yield, redemption, frequency, [basis])"
        }

    Errors:
        Returns an error message in results if the file can't be read or a column is missing.
    """
    try:
        file_path = Path(ctx.deps.data_dir) / df_name
        pd.read_csv(file_path)  # Just to check file exists/valid
        settlement_dt = datetime.datetime.strptime(settlement, "%Y-%m-%d")
        maturity_dt = datetime.datetime.strptime(maturity, "%Y-%m-%d")
        N = int(
            (maturity_dt.year - settlement_dt.year) * frequency
            + (maturity_dt.month - settlement_dt.month) / (12 / frequency)
        )
        if N <= 0:
            raise ValueError("Maturity must be after settlement.")
        C = coupon * 100 / frequency
        y = yield_
        pv_coupons = [(t / frequency) * C / (1 + y / frequency) ** t for t in range(1, N + 1)]
        pv_redemption = (N / frequency) * redemption / (1 + y / frequency) ** N
        duration = (sum(pv_coupons) + pv_redemption) / (
            sum(C / (1 + y / frequency) ** t for t in range(1, N + 1)) + redemption / (1 + y / frequency) ** N
        )
        return {
            "operation": "duration",
            "results": {
                "duration": duration,
                "message": f"Macaulay duration is {duration} years.",
            },
            "formula": f"=DURATION({settlement}, {maturity}, {coupon}, {yield_}, {frequency}, {basis})",
        }
    except Exception as e:
        return {
            "operation": "duration",
            "results": {"error": str(e)},
            "formula": f"=DURATION({settlement}, {maturity}, {coupon}, {yield_}, {frequency}, {basis})",
        }
