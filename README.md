# ⚡ ENERGYWISE AI
**AI-Based Energy Consumption Prediction and Saving Recommendation System**

Upload an electricity consumption CSV and the app automatically cleans it, analyses it, predicts future
consumption with **Random Forest Regression**, estimates the bill and gives data-based saving tips.

Built with: Python, Streamlit, Pandas, NumPy, Scikit-learn, Plotly (no database, no API, no login).

---

## 1. Installation

1. Install **Python 3.9 or newer** (https://www.python.org).
2. Open a terminal inside the `ENERGYWISE-AI` folder.
3. (Recommended) create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate          (Windows)
   source venv/bin/activate       (Mac / Linux)
   ```
4. Install the libraries:
   ```
   pip install -r requirements.txt
   ```

## 2. Run the app
```
streamlit run app.py
```
The browser opens at http://localhost:8501.

## 3. How to upload your dataset
1. Open **Upload Dataset** in the sidebar and click *Browse files*.
2. Choose your `.csv` file.
3. The app shows a preview, record/column counts and missing values.
4. Check the automatically detected **date column**, optional **time column**, **consumption column** and
   **unit**. If something is wrong, change it in the dropdowns.
5. Open **Dashboard** to see the results. Set your tariff in the sidebar (default ₹7 per kWh).

**Your CSV needs:** one date (or date-time) column, optionally a separate time column, and one numeric
consumption column. Examples that work:

| Date | Time | Global_active_power |   (kW, minute data, `;` separated, `?` = missing)
|---|---|---|
| 16/12/2006 | 17:24:00 | 4.216 |

| Datetime | Energy_kWh |   (hourly kWh)
|---|---|
| 2024-01-01 00:00 | 0.42 |

**Units:** kWh, Wh, MWh are converted directly. kW and W are *power*, so each reading is multiplied by the
typical time between readings (e.g. 1 minute = 1/60 hour) to get kWh.

**Needs at least 60 days of data for prediction.** Shorter or monthly datasets still get analysis,
cost and recommendations, but the app clearly says prediction is not possible (no fake results).

## 4. Pages
| Page | What it shows |
|---|---|
| Dashboard | Summary cards, estimated cost, daily chart, 7-day forecast, top recommendations |
| Upload Dataset | Upload, preview, missing values, column/unit selection, cleaning report |
| Energy Analysis | Total / average / max / min, monthly average, trend, highest period, charts |
| Consumption Prediction | Random Forest results (MAE, RMSE, R²), actual vs predicted, future forecast |
| Energy-Saving Recommendations | Data-based tips + What-If calculator (5/10/15/20 %) |

## 5. Project structure
```
ENERGYWISE-AI/
├── app.py                      Streamlit dashboard (pages + charts)
├── requirements.txt
├── README.md
├── .streamlit/config.toml      light theme + green accent colour
├── dataset/                    put your CSV files here (optional)
├── models/                     trained model is saved here (energywise_rf.joblib)
└── utils/
    ├── data_processing.py      read CSV, detect columns, clean, convert units to kWh
    ├── energy_analysis.py      statistics, trend, peak periods, high-usage days
    ├── prediction.py           Random Forest features, training, forecast
    └── recommendations.py      rule-based tips using the analysis results
```

## 6. How the calculations work
* **Total consumption** = sum of all valid records in kWh.
* **Average daily** = mean of the daily totals (days with too few readings are excluded, never counted as 0).
* **Average monthly** = mean of monthly totals, only for months where at least 90 % of days have data.
* **Trend** = straight line through daily values; reported as % change per 30 days (marked *Stable* if the
  line does not fit the data).
* **High-usage day** = a day above *average + 1.5 × standard deviation*.
* **Estimated cost** = Total consumption (kWh) × rate. This is only an estimate; real bills include fixed
  charges, taxes and tariff slabs.
* **What-If** = Target = Current × (1 − reduction %); Units saved = Current − Target; Money saved = Units saved × rate.
  These are hypothetical estimates, not guaranteed results.

---

## 7. Machine Learning explanation (for viva)

**Problem type:** Regression – predict a number (kWh used tomorrow) from past data. This is time-series
forecasting solved as a supervised regression problem.

**Algorithm: Random Forest Regressor**
* A Random Forest is a collection of many decision trees (we use 200).
* Each tree is trained on a random sample of the data and random subsets of features, so each tree is
  slightly different.
* For a new day, every tree gives a prediction and the forest returns the **average** – this reduces
  over-fitting compared with a single tree.
* Why Random Forest? Simple, works well on small tabular data, needs no feature scaling, handles
  non-linear patterns and gives feature importance.

**Features (all from the past only):**
`lag_1` (yesterday), `lag_2`, `lag_7` (same weekday last week), `rolling_mean_7` (average of previous 7 days),
`day_of_week`, `month`, `day_of_month`.

**Avoiding data leakage:** the data is split **by time**, not randomly – the first 80 % of days train
the model and the last 20 % (minimum 14 days) test it. Every feature of a day uses only earlier days, so
the model never sees the future.

**Metrics (calculated on the unseen test days):**
* **MAE** – average absolute error in kWh (easy to understand).
* **RMSE** – like MAE but punishes big mistakes more.
* **R²** – how much of the variation is explained (1 = perfect, 0 = same as predicting the average,
  negative = worse than the average).
* **Baseline MAE** – error of the naive rule "tomorrow = today". A useful model should beat it.

**Forecasting the future:** the final model is re-trained on all days and predicts day by day; each
prediction becomes the "yesterday" for the next day, so uncertainty grows for later days.

**Likely viva questions**
1. *Why not a random train/test split?* It would let the model learn from future days → unrealistic accuracy.
2. *What if R² is low?* The app warns the user; daily usage may be mostly random, or there is too little data.
3. *Why is at least 60 days required?* The features need 7 days of history and we need enough days to train and test.
4. *Is accuracy hard-coded?* No – every metric is computed from the uploaded data.
5. *Can it tell which appliance uses most?* No – the dataset only has total consumption, so the app never claims that.
6. *Why kW needs conversion?* kW is a rate of use; energy (kWh) = power (kW) × time (hours).

## 8. Limitations
* Predicts **daily** consumption only; accuracy depends on the dataset.
* With less than one year of data, yearly seasonality cannot be learned.
* Bill estimate uses one flat rate (no slabs/fixed charges).
* Recommendations are rule-based and use aggregate consumption only.
