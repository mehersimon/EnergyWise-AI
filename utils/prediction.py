"""
prediction.py
-------------
Random Forest Regression to predict DAILY electricity consumption (kWh).

How it works (simple version for viva):
  1. Add up the uploaded data to one value per day.
  2. Create features from the PAST: yesterday's usage, usage 2 and 7 days ago,
     the average of the last 7 days, day of week, month and day of month.
  3. Split by TIME: the first 80% of days train the model, the last 20% test it.
     (No random shuffling, so the model never sees the future while training.)
  4. Train a Random Forest, measure MAE / RMSE / R2 on the test days.
  5. Re-train on all days and predict the next N days one day at a time.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = ["lag_1", "lag_2", "lag_7", "rolling_mean_7",
            "day_of_week", "month", "day_of_month"]
MIN_ROWS = 60                                   # minimum usable days
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "energywise_rf.joblib"


def make_features(series):
    """Build the feature table. Every feature uses only days BEFORE the target day."""
    df = pd.DataFrame({"kwh": series})
    df["lag_1"] = series.shift(1)
    df["lag_2"] = series.shift(2)
    df["lag_7"] = series.shift(7)
    df["rolling_mean_7"] = series.shift(1).rolling(7).mean()
    df["day_of_week"] = series.index.dayofweek
    df["month"] = series.index.month
    df["day_of_month"] = series.index.day
    return df.dropna()


def _new_model():
    return RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                 random_state=42, n_jobs=-1)


def train_and_forecast(daily, horizon=14):
    """
    daily   : Series of kWh per day (DatetimeIndex, NaN where data is missing)
    horizon : number of future days to predict
    Returns a dict. result["ok"] is False (with result["reason"]) when the
    dataset is not suitable - we never fake predictions.
    """
    s = daily.copy()
    first, last = s.first_valid_index(), s.last_valid_index()
    if first is None:
        return {"ok": False, "reason": "No complete days of data were found."}
    s = s.loc[first:last]                      # drop incomplete days at the edges

    n_missing_before = int(s.isna().sum())
    s = s.interpolate(limit=3, limit_area="inside")   # fill only short gaps (<=3 days)

    feats = make_features(s)
    if len(feats) < MIN_ROWS:
        return {"ok": False,
                "reason": f"Only {len(feats)} usable days after preparing the features. "
                          f"At least {MIN_ROWS} days of data are needed for a reliable prediction."}

    # ---- time-based split (no shuffling => no future data leaks into training) ----
    n_test = max(14, int(len(feats) * 0.2))
    n_train = len(feats) - n_test
    if n_train < 40:
        return {"ok": False, "reason": "Not enough days left for training after the time-based split."}
    train, test = feats.iloc[:n_train], feats.iloc[n_train:]

    model = _new_model()
    model.fit(train[FEATURES], train["kwh"])
    pred = model.predict(test[FEATURES])

    mae = mean_absolute_error(test["kwh"], pred)
    rmse = float(np.sqrt(mean_squared_error(test["kwh"], pred)))
    r2 = r2_score(test["kwh"], pred)
    baseline_mae = mean_absolute_error(test["kwh"], test["lag_1"])   # "same as yesterday"

    warnings_list = []
    if len(feats) < 365:
        warnings_list.append("The dataset covers less than one year, so the model cannot learn "
                             "yearly (seasonal) patterns. Treat predictions as rough estimates.")
    if r2 < 0.3:
        warnings_list.append("R² on the test period is low: the daily usage in this dataset is hard "
                             "to predict from past usage. The forecast is not very reliable.")
    if mae >= baseline_mae:
        warnings_list.append("The model is not more accurate than the simple rule 'tomorrow = today'.")

    # ---- re-train on all data and predict the future day by day ----
    final_model = _new_model()
    final_model.fit(feats[FEATURES], feats["kwh"])
    try:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(final_model, MODEL_PATH)
    except OSError:
        pass                                    # saving is optional

    values = s.tolist()
    date = s.index[-1]
    rows = []
    for _ in range(int(horizon)):
        date = date + pd.Timedelta(days=1)
        x = pd.DataFrame([{
            "lag_1": values[-1], "lag_2": values[-2], "lag_7": values[-7],
            "rolling_mean_7": float(np.mean(values[-7:])),
            "day_of_week": date.dayofweek, "month": date.month, "day_of_month": date.day,
        }])[FEATURES]
        y = float(final_model.predict(x)[0])
        values.append(y)                        # the prediction becomes tomorrow's "yesterday"
        rows.append({"date": date, "predicted_kwh": y})

    importances = pd.Series(final_model.feature_importances_, index=FEATURES).sort_values()

    return {
        "ok": True,
        "n_days": len(feats), "n_train": n_train, "n_test": n_test,
        "n_missing_before": n_missing_before,
        "mae": float(mae), "rmse": rmse, "r2": float(r2), "baseline_mae": float(baseline_mae),
        "test": pd.DataFrame({"actual": test["kwh"], "predicted": pred}, index=test.index),
        "history": s,
        "forecast": pd.DataFrame(rows),
        "importances": importances,
        "warnings": warnings_list,
    }
