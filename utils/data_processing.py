"""
data_processing.py
------------------
Everything related to reading and cleaning the uploaded CSV file:
  1. load_csv()        - read the file (auto-detects the separator)
  2. detect_columns()  - guess the date / time / consumption columns and unit
  3. prepare_data()    - clean the data and convert consumption to kWh
  4. make_daily()      - add up consumption per day (used for analysis + ML)
"""
import io
import re
import warnings

import pandas as pd

# Text values that usually mean "no data"
MISSING_MARKERS = ["?", "NA", "N/A", "NaN", "nan", "null", "NULL", "None", "-", ""]

# Units the user can choose from
UNIT_OPTIONS = ["kWh", "Wh", "MWh", "kW", "W"]
ENERGY_TO_KWH = {"kWh": 1.0, "Wh": 0.001, "MWh": 1000.0}   # energy units -> kWh
POWER_TO_KW = {"kW": 1.0, "W": 0.001}                      # power units  -> kW

DATE_HINTS = ["datetime", "timestamp", "date_time", "date", "time", "day"]
VALUE_HINTS = ["kwh", "energy", "consumption", "usage", "global_active_power",
               "active_power", "power", "load", "units", "wh", "kw"]


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------
def load_csv(file_obj):
    """Read an uploaded CSV (file object or bytes) into a DataFrame."""
    raw = file_obj.read() if hasattr(file_obj, "read") else file_obj
    if not raw:
        raise ValueError("The uploaded file is empty.")

    # Guess the separator (, ; tab |) from the header line
    sample = raw[:20000].decode("utf-8", errors="ignore")
    lines = sample.splitlines()
    first_line = lines[0] if lines else ""
    sep = max(",;\t|", key=first_line.count)
    if first_line.count(sep) == 0:
        sep = ","

    df = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=encoding,
                             na_values=MISSING_MARKERS, low_memory=False,
                             on_bad_lines="skip")
            break
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            raise ValueError(f"Could not read this file as a CSV: {exc}")
    if df is None or df.empty:
        raise ValueError("The CSV file has no data rows.")

    df.columns = [str(c).strip() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# 2. Column detection
# ---------------------------------------------------------------------------
_TIME_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?(\.\d+)?$")


def _is_time_like(series):
    """True for columns that only hold a time of day such as 17:24:00."""
    if pd.api.types.is_numeric_dtype(series):
        return False
    s = series.dropna().astype(str).str.strip().head(50)
    return len(s) > 0 and s.str.match(_TIME_RE).mean() >= 0.8


def _is_date_like(series):
    """True if most sample values can be parsed as dates."""
    if pd.api.types.is_numeric_dtype(series):
        return False
    s = series.dropna().astype(str).str.strip().head(50)
    if len(s) == 0:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(s, errors="coerce", dayfirst=guess_dayfirst(series))
    return parsed.notna().mean() >= 0.8


def guess_dayfirst(series):
    """Guess whether dates look like DD/MM/YYYY (True) or MM/DD/YYYY (False)."""
    s = series.dropna().astype(str).str.strip().head(500)
    parts = s.str.extract(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.]\d{2,4}").dropna().astype(int)
    if parts.empty:
        return False                      # ISO style (2024-01-31): no ambiguity
    if (parts[0] > 12).any():
        return True
    if (parts[1] > 12).any():
        return False
    return True                           # ambiguous -> day first (common in India)


def numeric_columns(df, exclude=()):
    """Columns that contain (mostly) numbers."""
    result = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].notna().any():
                result.append(col)
        else:
            coerced = pd.to_numeric(df[col].head(1000), errors="coerce")
            if coerced.notna().mean() >= 0.8:
                result.append(col)
    return result


def _find_by_hints(columns, hints):
    for hint in hints:
        for col in columns:
            if hint in str(col).lower():
                return col
    return None


def guess_unit(value_col):
    """Guess the unit from the column name (the user can always change it)."""
    name = str(value_col).lower()
    if "mwh" in name:
        return "MWh"
    if "kwh" in name:
        return "kWh"
    if re.search(r"(^|[^a-z])wh($|[^a-z])", name):
        return "Wh"
    if "kw" in name:
        return "kW"
    if "watt" in name or re.search(r"(^|[^a-z])w($|[^a-z])", name):
        return "W"
    if "power" in name:
        return "kW"
    return "kWh"


def detect_columns(df):
    """Return a dict with the best guess for date, time and value columns + unit."""
    cols = list(df.columns)
    time_like = [c for c in cols if _is_time_like(df[c])]
    date_candidates = [c for c in cols if c not in time_like and _is_date_like(df[c])]

    date_col = _find_by_hints(date_candidates, DATE_HINTS) or (date_candidates[0] if date_candidates else None)
    time_col = time_like[0] if (time_like and date_col is not None) else None

    numeric = numeric_columns(df, exclude=[date_col, time_col])
    value_col = _find_by_hints(numeric, VALUE_HINTS) or (numeric[0] if numeric else None)

    return {
        "date_col": date_col,
        "time_col": time_col,
        "value_col": value_col,
        "unit": guess_unit(value_col) if value_col else "kWh",
        "dayfirst": guess_dayfirst(df[date_col]) if date_col else False,
    }


# ---------------------------------------------------------------------------
# 3. Cleaning + unit conversion
# ---------------------------------------------------------------------------
def prepare_data(df, date_col, time_col, value_col, unit, dayfirst):
    """
    Clean the data and convert consumption to kWh.

    Returns (clean, report):
      clean  - DataFrame indexed by timestamp with one column 'kwh'
      report - dict with counts of removed rows, time interval, warnings ...
    """
    if date_col is None or value_col is None:
        raise ValueError("Please select a date column and a consumption column.")

    report = {"rows_read": len(df), "warnings": []}

    # --- build timestamps ---
    if time_col and time_col != "None":
        ts_text = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()
    else:
        ts_text = df[date_col].astype(str).str.strip()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        timestamps = pd.to_datetime(ts_text, errors="coerce", dayfirst=dayfirst)

    values = pd.to_numeric(df[value_col], errors="coerce")

    # --- remove invalid records (step by step, so we can report each) ---
    valid_date = timestamps.notna()
    report["invalid_dates"] = int((~valid_date).sum())
    if valid_date.sum() == 0:
        raise ValueError(f"Could not read any dates in column '{date_col}'. "
                         "Choose another date column or change the day-first option.")

    data = pd.DataFrame({"timestamp": timestamps[valid_date], "value": values[valid_date]})
    report["missing_values"] = int(data["value"].isna().sum())
    data = data.dropna(subset=["value"])
    if data.empty:
        raise ValueError(f"Column '{value_col}' has no numeric values. Choose another consumption column.")

    report["negative_values"] = int((data["value"] < 0).sum())
    data = data[data["value"] >= 0]

    data = data.sort_values("timestamp", kind="stable")
    dup = data["timestamp"].duplicated(keep="first")
    report["duplicate_timestamps"] = int(dup.sum())
    data = data[~dup]

    if len(data) < 2:
        raise ValueError("Fewer than 2 valid records remain after cleaning.")
    data = data.set_index("timestamp")

    # --- time interval between records ---
    interval = data.index.to_series().diff().dropna().median()
    interval_hours = interval.total_seconds() / 3600
    report["interval_hours"] = interval_hours
    if interval_hours < 23:
        report["resolution"] = "sub-daily"
    elif interval_hours <= 36:
        report["resolution"] = "daily"
    else:
        report["resolution"] = "coarse"

    # --- convert to kWh ---
    if unit in ENERGY_TO_KWH:
        kwh = data["value"] * ENERGY_TO_KWH[unit]
    elif unit in POWER_TO_KW:
        # Power (kW / W) is a RATE. Energy = power x time of each reading.
        kwh = data["value"] * POWER_TO_KW[unit] * interval_hours
        report["warnings"].append(
            f"'{unit}' is a power unit, so each reading was multiplied by the typical "
            f"time between readings ({interval_hours:.3f} h) to get kWh.")
    else:
        raise ValueError(f"Unknown unit: {unit}")

    clean = pd.DataFrame({"kwh": kwh})
    report["rows_used"] = len(clean)
    report["start"] = clean.index.min()
    report["end"] = clean.index.max()

    lost = report["rows_read"] - report["rows_used"]
    if report["rows_read"] and lost / report["rows_read"] > 0.2:
        report["warnings"].append(
            f"{lost / report['rows_read']:.0%} of the rows were removed during cleaning. "
            "Please check that the selected columns are correct.")
    if report["resolution"] == "coarse":
        report["warnings"].append(
            f"Records are about {interval_hours / 24:.0f} days apart, so daily statistics "
            "and prediction are not available for this dataset.")
    return clean, report


# ---------------------------------------------------------------------------
# 4. Daily totals
# ---------------------------------------------------------------------------
def make_daily(clean, resolution):
    """
    Add consumption per day. Days with (almost) no readings are set to NaN
    instead of 0 so that missing data is never treated as 'zero usage'.

    Returns (daily_series, info) or (None, {}) when data is coarser than daily.
    """
    if resolution == "coarse":
        return None, {}

    grouped = clean["kwh"].resample("D").agg(["sum", "count"])
    expected = grouped.loc[grouped["count"] > 0, "count"].median()
    complete = grouped["count"] >= 0.8 * expected

    daily = grouped["sum"].where(complete)
    daily.name = "kwh"
    info = {
        "incomplete_days": int(((grouped["count"] > 0) & ~complete).sum()),
        "missing_days": int((grouped["count"] == 0).sum()),
    }
    return daily, info
