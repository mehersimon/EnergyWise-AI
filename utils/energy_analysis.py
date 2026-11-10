"""
energy_analysis.py
------------------
Calculates the statistics shown on the Dashboard and Energy Analysis pages.
Every number comes from the uploaded dataset - nothing is assumed.
"""
import numpy as np
import pandas as pd

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def analyse(clean, daily, resolution, interval_hours):
    """
    clean  : DataFrame (timestamp index, column 'kwh')
    daily  : Series of kWh per day (NaN = missing day) or None
    Returns a dictionary of results.
    """
    a = {
        "resolution": resolution,
        "n_records": len(clean),
        "start": clean.index.min(),
        "end": clean.index.max(),
        "total_kwh": float(clean["kwh"].sum()),     # total uses every valid record
    }

    d = daily.dropna() if daily is not None else None
    a["n_days"] = len(d) if d is not None else 0

    # ---------------- daily statistics ----------------
    a["avg_daily"] = a["max_daily"] = a["min_daily"] = None
    a["max_day"] = a["min_day"] = None
    a["daily_std"] = a["daily_cv"] = None
    a["trend_pct_30d"] = a["trend_label"] = None
    a["high_threshold"] = None
    a["high_days"] = pd.Series(dtype=float)
    a["high_share"] = 0.0
    a["weekday_profile"] = None
    a["weekday_avg"] = a["weekend_avg"] = None

    if d is not None and len(d) >= 1:
        a["avg_daily"] = float(d.mean())
        a["max_daily"], a["max_day"] = float(d.max()), d.idxmax()
        a["min_daily"], a["min_day"] = float(d.min()), d.idxmin()

    if d is not None and len(d) >= 14:
        a["daily_std"] = float(d.std())
        a["daily_cv"] = a["daily_std"] / a["avg_daily"] if a["avg_daily"] else None

        # High-usage days: more than average + 1.5 x standard deviation
        a["high_threshold"] = a["avg_daily"] + 1.5 * a["daily_std"]
        a["high_days"] = d[d > a["high_threshold"]].sort_values(ascending=False)
        a["high_share"] = float(a["high_days"].sum() / d.sum()) if d.sum() else 0.0

        # Trend: straight line through the daily values (slope = kWh change per day)
        x = (d.index - d.index[0]).days.to_numpy(dtype=float)
        slope = np.polyfit(x, d.to_numpy(), 1)[0]
        a["trend_pct_30d"] = float(slope * 30 / a["avg_daily"] * 100) if a["avg_daily"] else 0.0
        # Only call it a trend if the line really fits the data (correlation >= 0.2);
        # otherwise the up/down movement is just random day-to-day noise.
        corr = np.corrcoef(x, d.to_numpy())[0, 1] if d.std() > 0 else 0.0
        if abs(corr) < 0.2:
            a["trend_label"] = "Stable"
        elif a["trend_pct_30d"] > 3:
            a["trend_label"] = "Increasing"
        elif a["trend_pct_30d"] < -3:
            a["trend_label"] = "Decreasing"
        else:
            a["trend_label"] = "Stable"

        # Weekday pattern (needs at least 2 weeks of data)
        if len(d) >= 14:
            a["weekday_profile"] = d.groupby(d.index.dayofweek).mean()
            weekend = d[d.index.dayofweek >= 5]
            weekday = d[d.index.dayofweek < 5]
            if len(weekend) >= 2 and len(weekday) >= 2:
                a["weekend_avg"], a["weekday_avg"] = float(weekend.mean()), float(weekday.mean())

    # ---------------- monthly statistics ----------------
    monthly = clean["kwh"].resample("MS").sum().to_frame("kwh")
    if d is not None and len(d) > 0:
        # A month counts as "complete" when at least 90% of its days have data.
        # (A partly covered first/last month fails this test automatically.)
        days_with_data = d.groupby(d.index.to_period("M")).count()
        monthly["complete"] = [
            days_with_data.get(m.to_period("M"), 0) >= 0.9 * m.days_in_month
            for m in monthly.index
        ]
    else:
        monthly["complete"] = True        # coarse data: every record is one period
    a["monthly"] = monthly
    complete = monthly[monthly["complete"]]
    a["avg_monthly"] = float(complete["kwh"].mean()) if len(complete) >= 1 else None
    a["n_complete_months"] = len(complete)
    a["peak_month"] = complete["kwh"].idxmax() if len(complete) >= 2 else None
    a["peak_month_kwh"] = float(complete["kwh"].max()) if len(complete) >= 2 else None
    a["low_month"] = complete["kwh"].idxmin() if len(complete) >= 2 else None
    a["low_month_kwh"] = float(complete["kwh"].min()) if len(complete) >= 2 else None

    # ---------------- single-record extremes (used when no daily data) ----------------
    a["max_record"], a["max_record_time"] = float(clean["kwh"].max()), clean["kwh"].idxmax()
    a["min_record"], a["min_record_time"] = float(clean["kwh"].min()), clean["kwh"].idxmin()

    # ---------------- hourly pattern (only for hourly-or-finer data) ----------------
    a["hourly_profile"] = None
    a["peak_hour"] = None
    a["night_ratio"] = None
    if resolution == "sub-daily" and interval_hours <= 1.0001:
        hourly = clean["kwh"].resample("h").agg(["sum", "count"])
        hourly = hourly.loc[hourly["count"] > 0, "sum"]
        if len(hourly) >= 24:
            profile = hourly.groupby(hourly.index.hour).mean()
            a["hourly_profile"] = profile
            a["peak_hour"] = int(profile.idxmax())
            night = profile[profile.index.isin(range(0, 6))]
            if len(night) and profile.mean() > 0:
                a["night_ratio"] = float(night.mean() / profile.mean())

    return a
