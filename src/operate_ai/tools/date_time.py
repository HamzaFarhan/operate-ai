from datetime import datetime
from typing import Any

from dateutil.relativedelta import relativedelta


def today_formula_tool() -> dict[str, Any]:
    """
    Returns the current date.

    Returns:
        dict[str, Any]: {
            "operation": "today",
            "results": { "date": str },
            "formula": "=TODAY()"
        }
    """
    return {"operation": "today", "results": {"date": str(datetime.now().date())}, "formula": "=TODAY()"}


def now_formula_tool() -> dict[str, Any]:
    """
    Returns the current date and time.

    Returns:
        dict[str, Any]: {
            "operation": "now",
            "results": { "datetime": str },
            "formula": "=NOW()"
        }
    """
    return {"operation": "now", "results": {"datetime": str(datetime.now())}, "formula": "=NOW()"}


def date_formula_tool(year: int, month: int, day: int) -> dict[str, Any]:
    """
    Constructs a date from year, month, and day.

    Parameters:
        year (int): The year.
        month (int): The month.
        day (int): The day.

    Returns:
        dict[str, Any]: {
            "operation": "date",
            "results": { "date": str },
            "formula": "=DATE(year, month, day)"
        }
    """
    return {
        "operation": "date",
        "results": {"date": str(datetime(year, month, day))},
        "formula": f"=DATE({year}, {month}, {day})",
    }


def year_formula_tool(date_str: str) -> dict[str, Any]:
    """
    Extracts the year from a given date string (YYYY-MM-DD).

    Parameters:
        date_str (str): The date string.

    Returns:
        dict[str, Any]: {
            "operation": "year",
            "results": { "year": int },
            "formula": "=YEAR(date)"
        }
    """
    return {
        "operation": "year",
        "results": {"year": int(datetime.strptime(date_str, "%Y-%m-%d").year)},
        "formula": f"=YEAR({date_str})",
    }


def month_formula_tool(date_str: str) -> dict[str, Any]:
    """
    Extracts the month from a given date string (YYYY-MM-DD).

    Parameters:
        date_str (str): The date string.

    Returns:
        dict[str, Any]: {
            "operation": "month",
            "results": { "month": int },
            "formula": "=MONTH(date)"
        }
    """
    return {
        "operation": "month",
        "results": {"month": int(datetime.strptime(date_str, "%Y-%m-%d").month)},
        "formula": f"=MONTH({date_str})",
    }


def day_formula_tool(date_str: str) -> dict[str, Any]:
    """
    Extracts the day from a given date string (YYYY-MM-DD).

    Parameters:
        date_str (str): The date string.

    Returns:
        dict[str, Any]: {
            "operation": "day",
            "results": { "day": int },
            "formula": "=DAY(date)"
        }
    """
    return {
        "operation": "day",
        "results": {"day": int(datetime.strptime(date_str, "%Y-%m-%d").day)},
        "formula": f"=DAY({date_str})",
    }


def edate_formula_tool(start_date: str, months: int) -> dict[str, Any]:
    """
    Returns the date that is a specified number of months before or after a start date.

    Parameters:
        start_date (str): The start date string (YYYY-MM-DD).
        months (int): Number of months to add (can be negative).

    Returns:
        dict[str, Any]: {
            "operation": "edate",
            "results": { "date": str },
            "formula": "=EDATE(start_date, months)"
        }
    """
    return {
        "operation": "edate",
        "results": {"date": str(datetime.strptime(start_date, "%Y-%m-%d") + relativedelta(months=months))},
        "formula": f"=EDATE({start_date}, {months})",
    }


def eomonth_formula_tool(start_date: str, months: int) -> dict[str, Any]:
    """
    Returns the serial number for the last day of the month that is the indicated number of months before or after start_date.

    Parameters:
        start_date (str): The start date string (YYYY-MM-DD).
        months (int): Number of months to add (can be negative).

    Returns:
        dict[str, Any]: {
            "operation": "eomonth",
            "results": { "date": str },
            "formula": "=EOMONTH(start_date, months)"
        }
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        # Move to the first of the target month
        target = start + relativedelta(months=months + 1, day=1)
        # Subtract one day to get last day of previous month (target month)
        last_day = target - relativedelta(days=1)
        date_str = last_day.strftime("%Y-%m-%d")
    except Exception:
        date_str = None
    return {
        "operation": "eomonth",
        "results": {"date": date_str},
        "formula": f"=EOMONTH({start_date}, {months})",
    }


def datedif_formula_tool(start_date: str, end_date: str, unit: str) -> dict[str, Any]:
    """
    Returns the difference between two dates in the specified unit (e.g., "Y", "M", "D").

    Parameters:
        start_date (str): The start date string (YYYY-MM-DD).
        end_date (str): The end date string (YYYY-MM-DD).
        unit (str): The unit for the difference ("Y", "M", "D").

    Returns:
        dict[str, Any]: {
            "operation": "datedif",
            "results": { "difference": Any },
            "formula": "=DATEDIF(start_date, end_date, unit)"
        }
    """
    try:
        d1 = datetime.strptime(start_date, "%Y-%m-%d")
        d2 = datetime.strptime(end_date, "%Y-%m-%d")
        if unit == "Y":
            diff = d2.year - d1.year - ((d2.month, d2.day) < (d1.month, d1.day))
        elif unit == "M":
            diff = (d2.year - d1.year) * 12 + d2.month - d1.month - (d2.day < d1.day)
        elif unit == "D":
            diff = (d2 - d1).days
        else:
            diff = None
    except Exception:
        diff = None
    return {
        "operation": "datedif",
        "results": {"difference": diff},
        "formula": f"=DATEDIF({start_date}, {end_date}, {unit})",
    }


def yearfrac_formula_tool(start_date: str, end_date: str) -> dict[str, Any]:
    """
    Returns the year fraction between two dates.

    Parameters:
        start_date (str): The start date string (YYYY-MM-DD).
        end_date (str): The end date string (YYYY-MM-DD).

    Returns:
        dict[str, Any]: {
            "operation": "yearfrac",
            "results": { "fraction": float },
            "formula": "=YEARFRAC(start_date, end_date)"
        }
    """
    try:
        d1 = datetime.strptime(start_date, "%Y-%m-%d")
        d2 = datetime.strptime(end_date, "%Y-%m-%d")
        fraction = (d2 - d1).days / 365.0
    except Exception:
        fraction = None
    return {
        "operation": "yearfrac",
        "results": {"fraction": fraction},
        "formula": f"=YEARFRAC({start_date}, {end_date})",
    }


def workday_formula_tool(start_date: str, days: int, holidays: list[str] | None = None) -> dict[str, Any]:
    """
    Returns the date after a given number of working days, excluding weekends and optionally holidays.

    Parameters:
        start_date (str): The start date string (YYYY-MM-DD).
        days (int): Number of working days to add.
        holidays (list[str], optional): List of holiday date strings.

    Returns:
        dict[str, Any]: {
            "operation": "workday",
            "results": { "date": str },
            "formula": "=WORKDAY(start_date, days, [holidays])"
        }
    """
    try:
        current = datetime.strptime(start_date, "%Y-%m-%d")
        n = abs(days)
        direction = 1 if days >= 0 else -1
        holidays_set = set(holidays or [])
        count = 0
        while count < n:
            current += relativedelta(days=direction)
            if current.weekday() < 5 and current.strftime("%Y-%m-%d") not in holidays_set:
                count += 1
        date_str = current.strftime("%Y-%m-%d")
    except Exception:
        date_str = None
    return {
        "operation": "workday",
        "results": {"date": date_str},
        "formula": f"=WORKDAY({start_date}, {days}, {holidays})",
    }


def networkdays_formula_tool(start_date: str, end_date: str, holidays: list[str] | None = None) -> dict[str, Any]:
    """
    Returns the number of working days between two dates, excluding weekends and optionally holidays.

    Parameters:
        start_date (str): The start date string (YYYY-MM-DD).
        end_date (str): The end date string (YYYY-MM-DD).
        holidays (list[str], optional): List of holiday date strings.

    Returns:
        dict[str, Any]: {
            "operation": "networkdays",
            "results": { "count": int },
            "formula": "=NETWORKDAYS(start_date, end_date, [holidays])"
        }
    """
    try:
        d1 = datetime.strptime(start_date, "%Y-%m-%d")
        d2 = datetime.strptime(end_date, "%Y-%m-%d")
        holidays_set = set(holidays or [])
        count = 0
        current = d1
        while current <= d2:
            if current.weekday() < 5 and current.strftime("%Y-%m-%d") not in holidays_set:
                count += 1
            current += relativedelta(days=1)
    except Exception:
        count = None
    return {
        "operation": "networkdays",
        "results": {"count": count},
        "formula": f"=NETWORKDAYS({start_date}, {end_date}, {holidays})",
    }
