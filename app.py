"""
ENERGYWISE AI - AI-Based Energy Consumption Prediction and Saving Recommendation System
Run with:  streamlit run app.py
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import data_processing as dp
from utils import energy_analysis as ea
from utils import prediction as pr
from utils import recommendations as rc

GREEN, NAVY, GREY = "#16A34A", "#0B1F3A", "#64748B"
PAGES = ["Dashboard", "Upload Dataset", "Energy Analysis",
         "Consumption Prediction", "Energy-Saving Recommendations"]
COST_NOTE = ("This is an estimate only. Actual electricity bills may include fixed charges, "
             "taxes and tariff slabs.")

st.set_page_config(page_title="ENERGYWISE AI", page_icon="⚡", layout="wide")

# ---------------------------------------------------------------------------
# Styling (green / white / dark navy)
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
[data-testid="stSidebar"] {{ background: {NAVY}; }}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] span {{ color: #FFFFFF; }}
[data-testid="stSidebar"] input {{ color: {NAVY}; }}
.banner {{ background: {NAVY}; padding: 18px 24px; border-radius: 12px; margin-bottom: 18px; }}
.banner h1 {{ color: #FFFFFF; margin: 0; font-size: 1.9rem; }}
.banner h1 span {{ color: #4ADE80; }}
.banner p {{ color: #CBD5E1; margin: 4px 0 0 0; }}
.metric-card {{ background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 6px solid {GREEN};
    border-radius: 10px; padding: 14px 16px; margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(11,31,58,.08); }}
.metric-label {{ color: {GREY}; font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; }}
.metric-value {{ color: {NAVY}; font-size: 1.55rem; font-weight: 700; line-height: 1.3; }}
.metric-sub {{ color: {GREY}; font-size: .8rem; }}
.rec-card {{ background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px;
    padding: 14px 18px; margin-bottom: 12px; }}
.rec-card h4 {{ color: {NAVY}; margin: 0 0 6px 0; }}
.rec-card p {{ color: #334155; margin: 0 0 6px 0; }}
.rec-card .evidence {{ color: {GREY}; font-size: .88rem; }}
.pill {{ display:inline-block; padding: 1px 10px; border-radius: 999px; font-size: .75rem;
    font-weight: 600; color: #FFFFFF; margin-left: 8px; vertical-align: middle; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def card(label, value, sub=""):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>',
                unsafe_allow_html=True)


def fmt(x, digits=1):
    return "N/A" if x is None else f"{x:,.{digits}f}"


def date_text(ts):
    return "N/A" if ts is None else pd.Timestamp(ts).strftime("%d %b %Y")


def style(fig, title="", yaxis="kWh"):
    fig.update_layout(title=title, template="plotly_white", height=360,
                      margin=dict(l=10, r=10, t=50, b=10), yaxis_title=yaxis,
                      legend=dict(orientation="h", x=1, xanchor="right", y=1.0, yanchor="bottom"), colorway=[GREEN, NAVY, "#F59E0B"])
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch")


def go_to(page):
    st.session_state["page"] = page


def header():
    st.markdown('<div class="banner"><h1>⚡ ENERGYWISE <span>AI</span></h1>'
                '<p>AI-Based Energy Consumption Prediction and Saving Recommendation System</p></div>',
                unsafe_allow_html=True)


def need_data():
    """Return the processed result, or show a hint and return None."""
    res = st.session_state.get("res")
    if res is None:
        st.info("No dataset loaded yet. Please upload your electricity consumption CSV first.")
        st.button("Go to Upload Dataset", on_click=go_to, args=("Upload Dataset",))
    return res


@st.cache_data(show_spinner="Training the Random Forest model...")
def cached_prediction(daily, horizon):
    return pr.train_and_forecast(daily, horizon)


WIDGET_KEYS = {"date_col": "w_date", "time_col": "w_time", "value_col": "w_value",
               "unit": "w_unit", "dayfirst": "w_dayfirst"}


def seed_widgets(cfg, cols, force=False):
    """
    Put the detected / previously chosen values into the widgets' own state.
    (Streamlit forgets widget values while another page is open, so we restore them
    from cfg. force=True is used when a new file is uploaded.)
    """
    defaults = {
        "date_col": cfg["date_col"] if cfg["date_col"] in cols else cols[0],
        "time_col": cfg["time_col"] if cfg["time_col"] in cols else "None",
        "value_col": cfg["value_col"] if cfg["value_col"] in cols else cols[0],
        "unit": cfg["unit"],
        "dayfirst": cfg["dayfirst"],
    }
    for name, widget_key in WIDGET_KEYS.items():
        if force or widget_key not in st.session_state:
            st.session_state[widget_key] = defaults[name]


# ---------------------------------------------------------------------------
# Page: Upload Dataset
# ---------------------------------------------------------------------------
def page_upload():
    st.subheader("Upload Dataset")
    st.write("Upload a CSV file containing a date/time column and an electricity consumption column.")

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is not None:
        file_id = f"{uploaded.name}-{uploaded.size}"
        if st.session_state.get("file_id") != file_id:          # a new file was chosen
            try:
                raw = dp.load_csv(uploaded)
            except ValueError as err:
                st.error(str(err))
                return
            cfg_new = dp.detect_columns(raw)
            seed_widgets(cfg_new, list(raw.columns), force=True)
            st.session_state.update(file_id=file_id, file_name=uploaded.name, raw=raw,
                                    cfg=cfg_new, res=None, res_key=None)

    raw = st.session_state.get("raw")
    if raw is None:
        st.markdown("**Expected format** (example):")
        st.code("Date,Time,Energy_kWh\n01/01/2024,00:00:00,0.42\n01/01/2024,01:00:00,0.38", language="text")
        st.caption("Supported units: kWh, Wh, MWh, and power in kW / W (converted to kWh using the time "
                   "between readings). Hourly, daily or minute-level data all work. "
                   "At least 60 days of data are needed for prediction.")
        return

    cfg = st.session_state["cfg"]
    st.success(f"Loaded **{st.session_state['file_name']}**")

    c1, c2 = st.columns(2)
    with c1:
        card("Records (rows)", f"{len(raw):,}")
    with c2:
        card("Columns", f"{raw.shape[1]}")

    st.markdown("##### Dataset preview")
    st.dataframe(raw.head(10), width="stretch")

    st.markdown("##### Missing values")
    missing = raw.isna().sum()
    miss_df = pd.DataFrame({"Column": missing.index, "Missing": missing.values,
                            "Percent": (missing.values / len(raw) * 100).round(2)})
    if missing.sum() == 0:
        st.write("No missing values found.")
    else:
        st.dataframe(miss_df[miss_df["Missing"] > 0], hide_index=True, width="stretch")
        st.caption("Rows with a missing date or consumption value are removed during cleaning.")

    # ---- column selection (pre-filled with automatic detection) ----
    st.markdown("##### Column selection")
    if cfg["date_col"] is None or cfg["value_col"] is None:
        st.warning("Automatic detection could not find every column. Please select them below.")
    else:
        st.caption("Columns were detected automatically. Please check them and confirm the unit.")

    cols = list(raw.columns)
    seed_widgets(cfg, cols)
    c1, c2, c3, c4 = st.columns(4)
    cfg["date_col"] = c1.selectbox("Date (or date-time) column", cols, key="w_date")
    cfg["time_col"] = c2.selectbox("Separate time column (optional)", ["None"] + cols, key="w_time")
    cfg["value_col"] = c3.selectbox("Consumption column", cols, key="w_value")
    cfg["unit"] = c4.selectbox("Unit of consumption column", dp.UNIT_OPTIONS, key="w_unit")
    cfg["dayfirst"] = st.checkbox("Dates are written day-first (DD/MM/YYYY)", key="w_dayfirst")
    if cfg["unit"] in dp.POWER_TO_KW:
        st.info(f"{cfg['unit']} is a power unit. It will be converted to kWh using the time between readings.")

    # ---- process (only when the selection changed) ----
    key = (st.session_state["file_id"], cfg["date_col"], cfg["time_col"],
           cfg["value_col"], cfg["unit"], cfg["dayfirst"])
    if st.session_state.get("res_key") != key:
        try:
            with st.spinner("Cleaning and analysing the dataset..."):
                clean, report = dp.prepare_data(raw, cfg["date_col"], cfg["time_col"],
                                                cfg["value_col"], cfg["unit"], cfg["dayfirst"])
                daily, info = dp.make_daily(clean, report["resolution"])
                analysis = ea.analyse(clean, daily, report["resolution"], report["interval_hours"])
            st.session_state["res"] = dict(clean=clean, report=report, daily=daily,
                                           daily_info=info, analysis=analysis)
        except ValueError as err:
            st.session_state["res"] = None
            st.session_state["res_key"] = None
            st.error(str(err))
            return
        st.session_state["res_key"] = key

    res = st.session_state["res"]
    rep = res["report"]
    st.markdown("##### Cleaning report")
    st.table(pd.DataFrame({
        "Check": ["Rows read", "Rows with invalid dates (removed)", "Rows with missing consumption (removed)",
                  "Rows with negative consumption (removed)", "Duplicate timestamps (removed)",
                  "Rows used", "Date range", "Typical time between readings"],
        "Result": [f"{rep['rows_read']:,}", f"{rep['invalid_dates']:,}", f"{rep['missing_values']:,}",
                   f"{rep['negative_values']:,}", f"{rep['duplicate_timestamps']:,}", f"{rep['rows_used']:,}",
                   f"{date_text(rep['start'])} to {date_text(rep['end'])}",
                   f"{rep['interval_hours']:.3f} hours ({rep['resolution']})"],
    }))
    for w in rep["warnings"]:
        st.warning(w)
    info = res["daily_info"]
    if info.get("incomplete_days") or info.get("missing_days"):
        st.caption(f"{info['incomplete_days']} incomplete days and {info['missing_days']} days without data "
                   "are excluded from daily statistics (never treated as zero usage).")
    st.success("Dataset is ready. Open the Dashboard to see the results.")
    st.button("Go to Dashboard", on_click=go_to, args=("Dashboard",))


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------
def daily_chart(res, title):
    a, daily = res["analysis"], res["daily"]
    fig = go.Figure()
    if daily is not None:
        d = daily.dropna()
        fig.add_trace(go.Scatter(x=d.index, y=d.values, mode="lines", name="Daily kWh",
                                 line=dict(color=GREEN, width=1.5)))
        if len(d) >= 7:
            fig.add_trace(go.Scatter(x=d.index, y=d.rolling(7).mean(), mode="lines",
                                     name="7-day average", line=dict(color=NAVY, width=2.5)))
        if a["high_threshold"] is not None:
            fig.add_hline(y=a["high_threshold"], line_dash="dot", line_color="#DC2626",
                          annotation_text="High-usage level")
    else:
        clean = res["clean"]
        fig.add_trace(go.Bar(x=clean.index, y=clean["kwh"], name="kWh per record", marker_color=GREEN))
    return style(fig, title)


def page_dashboard(rate):
    st.subheader("Dashboard")
    res = need_data()
    if res is None:
        return
    a = res["analysis"]
    cost = a["total_kwh"] * rate

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Total consumption", f"{fmt(a['total_kwh'], 1)} kWh",
             f"{date_text(a['start'])} - {date_text(a['end'])}")
    with c2:
        card("Average daily", f"{fmt(a['avg_daily'], 2)} kWh" if a["avg_daily"] is not None else "N/A",
             f"{a['n_days']} complete days")
    with c3:
        card("Estimated cost", f"₹{cost:,.0f}", f"at ₹{rate:g} per kWh")
    with c4:
        card("Records", f"{a['n_records']:,}", f"{res['report']['resolution']} data")
    st.caption(COST_NOTE)

    show(daily_chart(res, "Daily electricity consumption"))

    c1, c2, c3 = st.columns(3)
    with c1:
        card("Highest consumption day",
             f"{fmt(a['max_daily'], 1)} kWh" if a["max_daily"] is not None else f"{fmt(a['max_record'], 2)} kWh",
             date_text(a["max_day"] if a["max_daily"] is not None else a["max_record_time"]))
    with c2:
        trend = a["trend_label"] or "N/A"
        card("Consumption trend", trend,
             f"{a['trend_pct_30d']:+.1f}% per 30 days" if a["trend_pct_30d"] is not None else "needs 14+ days")
    with c3:
        if res["daily"] is not None:
            result = cached_prediction(res["daily"], 7)
            if result["ok"]:
                total7 = result["forecast"]["predicted_kwh"].sum()
                card("Predicted next 7 days", f"{total7:,.1f} kWh", f"≈ ₹{total7 * rate:,.0f}")
            else:
                card("Prediction", "Not available", "see Consumption Prediction page")
        else:
            card("Prediction", "Not available", "data is coarser than daily")

    st.markdown("##### Top recommendations")
    for r in rc.generate(a, rate)[:3]:
        recommendation_card(r)
    st.button("See all recommendations and the What-If calculator", on_click=go_to,
              args=("Energy-Saving Recommendations",))


# ---------------------------------------------------------------------------
# Page: Energy Analysis
# ---------------------------------------------------------------------------
def page_analysis():
    st.subheader("Energy Analysis")
    res = need_data()
    if res is None:
        return
    a = res["analysis"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Total consumption", f"{fmt(a['total_kwh'], 1)} kWh")
    with c2:
        card("Average daily", f"{fmt(a['avg_daily'], 2)} kWh")
    with c3:
        card("Average monthly", f"{fmt(a['avg_monthly'], 1)} kWh",
             f"{a['n_complete_months']} complete months" if a["avg_monthly"] is not None
             else "needs at least 1 complete month")
    with c4:
        card("Records", f"{a['n_records']:,}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Maximum (daily)", f"{fmt(a['max_daily'], 1)} kWh", date_text(a["max_day"]))
    with c2:
        card("Minimum (daily)", f"{fmt(a['min_daily'], 1)} kWh", date_text(a["min_day"]))
    with c3:
        card("Consumption trend", a["trend_label"] or "N/A",
             f"{a['trend_pct_30d']:+.1f}% per 30 days" if a["trend_pct_30d"] is not None else "needs 14+ days")
    with c4:
        period = []
        if a["max_day"] is not None:
            period.append(f"Day: {date_text(a['max_day'])}")
        if a["peak_month"] is not None:
            period.append(f"Month: {a['peak_month'].strftime('%b %Y')}")
        if a["peak_hour"] is not None:
            period.append(f"Hour: {a['peak_hour']:02d}:00")
        card("Highest consumption period", "<br>".join(period) if period else "N/A",
             f"single-record max: {a['max_record']:.2f} kWh")

    show(daily_chart(res, "Daily consumption with high-usage level"))

    left, right = st.columns(2)
    with left:
        m = a["monthly"]
        if len(m) >= 2:
            colors = [GREEN if ok else "#CBD5E1" for ok in m["complete"]]
            fig = go.Figure(go.Bar(x=m.index.strftime("%b %Y"), y=m["kwh"], marker_color=colors))
            show(style(fig, "Monthly consumption (grey = incomplete month)"))
        else:
            st.info("Monthly chart needs data from at least 2 months.")
    with right:
        if a["weekday_profile"] is not None:
            wp = a["weekday_profile"]
            fig = go.Figure(go.Bar(x=[ea.WEEKDAY_NAMES[i][:3] for i in wp.index], y=wp.values,
                                   marker_color=GREEN))
            show(style(fig, "Average consumption by day of week", "kWh per day"))
        else:
            st.info("Day-of-week chart needs at least 14 days of data.")

    if a["hourly_profile"] is not None:
        hp = a["hourly_profile"]
        fig = go.Figure(go.Bar(x=[f"{h:02d}:00" for h in hp.index], y=hp.values, marker_color=NAVY))
        show(style(fig, "Average consumption by hour of day", "kWh per hour"))

    if len(a["high_days"]) > 0:
        st.markdown("##### Highest-consumption days")
        table = a["high_days"].head(10).rename("kWh").reset_index()
        table.columns = ["Date", "Consumption (kWh)"]
        table["Date"] = table["Date"].dt.strftime("%d %b %Y")
        table["Consumption (kWh)"] = table["Consumption (kWh)"].round(2)
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption(f"Days above {a['high_threshold']:.1f} kWh (average + 1.5 x standard deviation).")
    elif a["high_threshold"] is not None:
        st.success("No unusually high-consumption days were found.")


# ---------------------------------------------------------------------------
# Page: Consumption Prediction
# ---------------------------------------------------------------------------
def page_prediction(rate):
    st.subheader("Consumption Prediction (Random Forest Regression)")
    res = need_data()
    if res is None:
        return
    if res["daily"] is None:
        st.error("Prediction is not available: the records in this dataset are spaced more than a day "
                 "apart, so there is no daily history to learn from.")
        return

    horizon = st.slider("Days to predict", 7, 30, 14)
    result = cached_prediction(res["daily"], horizon)
    if not result["ok"]:
        st.error(f"This dataset is not suitable for prediction. {result['reason']}")
        return
    for w in result["warnings"]:
        st.warning(w)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("MAE", f"{result['mae']:.2f} kWh", "average error per day")
    with c2:
        card("RMSE", f"{result['rmse']:.2f} kWh", "penalises large errors")
    with c3:
        card("R² score", f"{result['r2']:.3f}", "1.0 = perfect, 0 = no better than the average")
    with c4:
        card("Baseline MAE", f"{result['baseline_mae']:.2f} kWh", "rule: tomorrow = today")
    st.caption(f"Trained on the first {result['n_train']} days and tested on the last {result['n_test']} days "
               "(time-based split, so no future data is used in training).")

    t = result["test"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t.index, y=t["actual"], name="Actual", mode="lines+markers",
                             line=dict(color=NAVY)))
    fig.add_trace(go.Scatter(x=t.index, y=t["predicted"], name="Predicted", mode="lines+markers",
                             line=dict(color=GREEN, dash="dash")))
    show(style(fig, "Actual vs predicted daily consumption (test period)"))

    f = result["forecast"]
    hist = result["history"].tail(60)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Recent actual", line=dict(color=NAVY)))
    fig.add_trace(go.Scatter(x=f["date"], y=f["predicted_kwh"], name="Forecast",
                             mode="lines+markers", line=dict(color=GREEN)))
    show(style(fig, f"Forecast for the next {horizon} days"))

    total = f["predicted_kwh"].sum()
    c1, c2 = st.columns(2)
    with c1:
        card(f"Predicted total ({horizon} days)", f"{total:,.1f} kWh")
    with c2:
        card("Estimated cost of forecast", f"₹{total * rate:,.0f}", f"at ₹{rate:g} per kWh")
    st.caption("Forecasts are made one day at a time, so uncertainty grows for later days. " + COST_NOTE)

    table = f.copy()
    table["date"] = table["date"].dt.strftime("%d %b %Y")
    table["predicted_kwh"] = table["predicted_kwh"].round(2)
    table.columns = ["Date", "Predicted consumption (kWh)"]
    st.dataframe(table, hide_index=True, width="stretch")

    imp = result["importances"]
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=GREEN))
    show(style(fig, "Which features does the model rely on most?", ""))

    with st.expander("How does this model work? (simple explanation)"):
        st.markdown("""
1. The data is added up to **one value per day**.
2. For every day, features are created from the **past only**: yesterday's usage, usage 2 and 7 days ago,
   the average of the previous 7 days, day of week, month and day of month.
3. A **Random Forest** (200 decision trees) learns the link between these features and the day's usage.
   The final answer is the **average of all trees**.
4. The model is tested on the **last 20% of days**, which it never saw during training.
5. For the future forecast it predicts tomorrow, then uses that prediction to predict the next day, and so on.
""")


# ---------------------------------------------------------------------------
# Page: Recommendations (+ What-If calculator)
# ---------------------------------------------------------------------------
PRIORITY_COLOR = {"High": "#DC2626", "Medium": "#F59E0B", "Low": GREEN}


def recommendation_card(r):
    st.markdown(f'<div class="rec-card"><h4>{r["title"]}'
                f'<span class="pill" style="background:{PRIORITY_COLOR[r["priority"]]}">{r["priority"]}</span></h4>'
                f'<p>{r["detail"]}</p><div class="evidence"><b>Based on your data:</b> {r["evidence"]}</div></div>',
                unsafe_allow_html=True)


def page_recommendations(rate):
    st.subheader("Energy-Saving Recommendations")
    res = need_data()
    if res is None:
        return
    a = res["analysis"]

    st.caption("Recommendations are generated from patterns found in your dataset. The dataset contains "
               "total consumption only, so individual appliance usage cannot be identified.")
    for r in rc.generate(a, rate):
        recommendation_card(r)

    st.markdown("---")
    st.markdown("#### What-If Energy Saving Calculator")
    pct = st.select_slider("Target reduction in consumption (%)", options=[5, 10, 15, 20], value=10)
    current = a["total_kwh"]
    target = current * (1 - pct / 100)
    saved = current - target

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Current consumption", f"{current:,.1f} kWh", "total in dataset")
    with c2:
        card("Target consumption", f"{target:,.1f} kWh", f"{pct}% lower")
    with c3:
        card("Expected units saved", f"{saved:,.1f} kWh")
    with c4:
        card("Estimated money saved", f"₹{saved * rate:,.0f}", f"at ₹{rate:g} per kWh")
    if a["avg_monthly"] is not None:
        per_month = a["avg_monthly"] * pct / 100
        st.write(f"On a monthly basis this is about **{per_month:,.1f} kWh (₹{per_month * rate:,.0f}) per month**.")
    st.info("These are hypothetical savings estimates, not guaranteed results. " + COST_NOTE)


# ---------------------------------------------------------------------------
# Main: sidebar navigation
# ---------------------------------------------------------------------------
def main():
    st.session_state.setdefault("page", PAGES[0])
    st.sidebar.markdown("## ⚡ ENERGYWISE AI")
    page = st.sidebar.radio("Navigation", PAGES, key="page")
    st.sidebar.markdown("---")
    rate = st.sidebar.number_input("Electricity rate (₹ per kWh)", min_value=0.0, value=7.0, step=0.5,
                                   help="Default is ₹7 per unit. Change it to match your tariff.")

    header()
    if page == "Dashboard":
        page_dashboard(rate)
    elif page == "Upload Dataset":
        page_upload()
    elif page == "Energy Analysis":
        page_analysis()
    elif page == "Consumption Prediction":
        page_prediction(rate)
    else:
        page_recommendations(rate)


main()
