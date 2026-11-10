"""
recommendations.py
------------------
Rule-based recommendations. Every recommendation is triggered by a number
calculated from the uploaded dataset and shows that number as "evidence".

NOTE: The dataset only has total (aggregate) consumption, so we never claim
to know how much a specific appliance uses.
"""

HOURS = {h: f"{h:02d}:00-{(h + 1) % 24:02d}:00" for h in range(24)}


def _rec(title, detail, evidence, priority):
    return {"title": title, "detail": detail, "evidence": evidence, "priority": priority}


def generate(a, rate):
    """a = result of energy_analysis.analyse(); rate = Rs per kWh. Returns a list of dicts."""
    recs = []

    # 1. Unusually high usage days
    if a["high_threshold"] is not None and len(a["high_days"]) > 0:
        top_day = a["high_days"].index[0].strftime("%d %b %Y")
        share = a["high_share"] * 100
        recs.append(_rec(
            "Monitor periods with unusually high usage",
            "A few days used far more electricity than normal. Check what was different on "
            "those days (guests, extra cooling/heating, equipment left running) and try to avoid repeating it.",
            f"{len(a['high_days'])} days were above {a['high_threshold']:.1f} kWh/day "
            f"(average + 1.5 x standard deviation). They make up {share:.1f}% of total consumption. "
            f"Highest: {top_day} with {a['high_days'].iloc[0]:.1f} kWh.",
            "High" if share >= 10 else "Medium"))

    # 2. Trend
    if a["trend_label"] == "Increasing":
        recs.append(_rec(
            "Daily consumption is trending upward",
            "Your usage is gradually rising. Review recent changes in your routine or equipment "
            "and keep checking the daily trend so the increase does not continue unnoticed.",
            f"The trend line shows about {a['trend_pct_30d']:+.1f}% change per 30 days.",
            "High"))
    elif a["trend_label"] == "Decreasing":
        recs.append(_rec(
            "Consumption is trending downward - keep it up",
            "Your usage is falling over time. Continue the habits that are working and keep "
            "monitoring the daily trend.",
            f"The trend line shows about {a['trend_pct_30d']:+.1f}% change per 30 days.",
            "Low"))
    elif a["trend_label"] == "Stable":
        recs.append(_rec(
            "Keep monitoring daily consumption trends",
            "Your usage is steady. Checking the daily chart regularly helps you notice any "
            "sudden increase early.",
            f"The trend is stable ({a['trend_pct_30d']:+.1f}% per 30 days).",
            "Low"))

    # 3. Peak hour
    if a["peak_hour"] is not None:
        recs.append(_rec(
            "Reduce usage during your peak hour",
            "Try to shift flexible activities away from this period and avoid running many "
            "high-power devices at the same time.",
            f"Highest average hourly consumption is between {HOURS[a['peak_hour']]} "
            f"({a['hourly_profile'].max():.2f} kWh per hour on average).",
            "Medium"))

    # 4. Night-time (possible always-on loads)
    if a["night_ratio"] is not None and a["night_ratio"] >= 0.5:
        recs.append(_rec(
            "Check for devices left on at night",
            "Night-time usage is high compared with the rest of the day. This can indicate "
            "devices running or on standby when nobody needs them. Switch off what you can "
            "before sleeping. (Appliance-level usage cannot be identified from this dataset.)",
            f"Average consumption between 00:00 and 06:00 is {a['night_ratio'] * 100:.0f}% of the "
            f"average hourly consumption over the whole day.",
            "High" if a["night_ratio"] >= 0.7 else "Medium"))

    # 5. Weekend vs weekday
    if a["weekday_avg"] and a["weekend_avg"]:
        diff = (a["weekend_avg"] - a["weekday_avg"]) / a["weekday_avg"] * 100
        if diff >= 15:
            recs.append(_rec(
                "Weekend consumption is higher than weekdays",
                "More time at home usually means more usage. Plan weekend activities so that "
                "unnecessary appliances are not left running.",
                f"Weekend average {a['weekend_avg']:.1f} kWh/day vs weekday average "
                f"{a['weekday_avg']:.1f} kWh/day ({diff:+.0f}%).",
                "Medium"))
        elif diff <= -15:
            recs.append(_rec(
                "Weekday consumption is higher than weekends",
                "Check whether devices stay on while the house is empty during working hours.",
                f"Weekday average {a['weekday_avg']:.1f} kWh/day vs weekend average "
                f"{a['weekend_avg']:.1f} kWh/day ({diff:+.0f}%).",
                "Medium"))

    # 6. Seasonal peak
    if a["peak_month"] is not None and a["low_month_kwh"]:
        ratio = a["peak_month_kwh"] / a["low_month_kwh"]
        if ratio >= 1.3:
            recs.append(_rec(
                "Plan for your seasonal peak",
                "Usage changes a lot between months. Prepare for the highest month by servicing "
                "equipment, using efficient settings, and avoiding wasteful usage.",
                f"Highest month: {a['peak_month'].strftime('%b %Y')} ({a['peak_month_kwh']:.0f} kWh), "
                f"lowest: {a['low_month'].strftime('%b %Y')} ({a['low_month_kwh']:.0f} kWh) - "
                f"{ratio:.1f}x difference.",
                "Medium"))

    # 7. Irregular usage
    if a["daily_cv"] is not None and a["daily_cv"] > 0.35:
        recs.append(_rec(
            "Daily usage fluctuates a lot",
            "Large day-to-day swings make bills hard to predict. Try to build a steadier routine "
            "and find what causes the peaks.",
            f"Daily consumption varies by {a['daily_cv'] * 100:.0f}% of its average "
            f"(standard deviation {a['daily_std']:.1f} kWh).",
            "Low"))

    # 8. Always-shown general recommendations (with numbers from the dataset)
    if a["avg_daily"] is not None:
        recs.append(_rec(
            "Improve energy efficiency",
            "Use LED lighting, energy-rated (BEE star) appliances and regular maintenance. "
            "Efficient equipment gives the same comfort with fewer units.",
            f"Your average is {a['avg_daily']:.1f} kWh/day, about ₹{a['avg_daily'] * rate:,.0f} per day "
            f"at ₹{rate:g}/kWh.",
            "Low"))
    saved_10 = a["total_kwh"] * 0.10
    recs.append(_rec(
        "Reduce unnecessary electricity consumption",
        "Switch off lights and devices when not in use and avoid standby usage. "
        "Even small habits add up over time.",
        f"A 10% reduction would save about {saved_10:,.0f} kWh (₹{saved_10 * rate:,.0f}) "
        f"over this dataset's period (hypothetical estimate).",
        "Medium"))

    order = {"High": 0, "Medium": 1, "Low": 2}
    recs.sort(key=lambda r: order[r["priority"]])
    return recs
