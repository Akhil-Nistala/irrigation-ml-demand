"""Showcase analyses on real NASA POWER weather, for the end-of-course presentation.

Same 7 inputs, chronological 80/20 split, Random Forest settings and LSTM architecture and training protocol
as src/no_lag_experiment.py (the LSTM class and training loop are copied from there unchanged). It answers
four practical questions on the 400 real-weather test days (20 May 2023 - 22 Jun 2024):

  1. Water budget  : how close is the predicted irrigation volume, per season and over the whole test period?
  2. Error by season: where in the year do the models make their errors?
  3. Cheap sensors : how much accuracy is lost without wind and solar sensors, or with only a rain gauge and a thermometer
                     (mean, or daily max/min)? Baseline: FAO-56's own temperature-only method (Hargreaves, raw and calibrated).
  4. History length: how many years of records does a model need?

Deployed predictions are clipped at zero (demand cannot be negative). Deep models: 3 seeds, MAE reported as the mean
of the per-seed MAEs (as in no_lag_experiment.py); daily series and water budgets use the seed-averaged prediction.

    python src/showcase_analysis.py

Writes report/showcase_test_predictions.csv, report/showcase_summary.json and models/random_forest_nasa.joblib
(the Random Forest served by the web app).
"""
import json
import pathlib
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor

ROOT = pathlib.Path(__file__).resolve().parents[1]
RANDOM_STATE, SEEDS, WINDOW = 42, [0, 1, 2], 7
torch.set_num_threads(2)

FEATURES = ["rainfall_mm", "temperature_C", "humidity_percent", "wind_speed_mps",
            "solar_radiation_MJ_m2_day", "crop_coefficient", "growth_stage"]
TARGET = "irrigation_demand_mm"
SUBSETS = {
    "All 7 inputs": FEATURES,
    "No wind or solar sensor": ["rainfall_mm", "temperature_C", "humidity_percent", "crop_coefficient", "growth_stage"],
    "Rain gauge + max/min thermometer": ["rainfall_mm", "tmax_C", "tmin_C", "crop_coefficient", "growth_stage"],
    "Rain gauge + thermometer": ["rainfall_mm", "temperature_C", "crop_coefficient", "growth_stage"],
}
HISTORY_DAYS = [365, 730, 1095, 1600]
SEASON = {3: "Pre-monsoon", 4: "Pre-monsoon", 5: "Pre-monsoon", 6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
          10: "Post-monsoon and winter", 11: "Post-monsoon and winter", 12: "Post-monsoon and winter",
          1: "Post-monsoon and winter", 2: "Post-monsoon and winter"}

df = pd.read_csv(ROOT / "data/irrigation_dataset_nasa_power.csv", parse_dates=["date"])
# daily max/min temperature from the same NASA POWER download (used only by the cheap-sensor scenario): the daily
# range Tmax - Tmin tracks cloudiness, which is why FAO-56 estimates missing radiation from it (Hargreaves)
raw = pd.read_csv(ROOT / "data/nasa_power_nagpur_raw.csv", parse_dates=["date"])
df = df.merge(raw[["date", "T2M_MAX", "T2M_MIN"]].rename(columns={"T2M_MAX": "tmax_C", "T2M_MIN": "tmin_C"}), on="date", how="left")
assert df[["tmax_C", "tmin_C"]].notna().all().all()
split = int(len(df) * 0.8)
test = df.iloc[split:].reset_index(drop=True)
y_test = test[TARGET].values
mae = lambda y, p: float(mean_absolute_error(y, p))  # noqa: E731


def random_forest():
    return RandomForestRegressor(n_estimators=300, max_depth=10, random_state=RANDOM_STATE, n_jobs=2)


# ---------------- LSTM, identical to src/no_lag_experiment.py ----------------
class LSTM(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.lstm = nn.LSTM(n_features, 48, batch_first=True)
        self.head = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.head(self.lstm(x)[0][:, -1]).squeeze(-1)


def lstm_runs(cols, history=split):
    """Train the LSTM on the `history` days before the test period (3 seeds); return per-seed test predictions."""
    X = df[cols].values.astype(np.float32)
    idx = np.arange(WINDOW - 1, len(df))
    win = np.stack([X[i - WINDOW + 1:i + 1] for i in idx])
    y = df[TARGET].values.astype(np.float32)[idx]
    tr = (idx < split) & (idx >= split - history)
    te = idx >= split
    first = max(0, split - history)
    mu, sd = X[first:split].mean(0), X[first:split].std(0) + 1e-6
    ymu, ysd = y[tr].mean(), y[tr].std()
    win_s = (win - mu) / sd
    out = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        Xtr, ytr = torch.tensor(win_s[tr]), torch.tensor((y[tr] - ymu) / ysd)
        nval = int(len(Xtr) * 0.1)
        Xfit, yfit, Xval, yval = Xtr[:-nval], ytr[:-nval], Xtr[-nval:], ytr[-nval:]
        model = LSTM(len(cols))
        opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
        best, best_state, wait = np.inf, None, 0
        for epoch in range(400):
            model.train()
            perm = torch.randperm(len(Xfit))
            for b in range(0, len(Xfit), 64):
                j = perm[b:b + 64]
                opt.zero_grad()
                nn.functional.mse_loss(model(Xfit[j]), yfit[j]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                v = nn.functional.mse_loss(model(Xval), yval).item()
            if v < best - 1e-5:
                best, best_state, wait = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= 30:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            out.append(model(torch.tensor(win_s[te])).numpy() * ysd + ymu)
    return out


def lstm_summary(runs):
    clipped = [np.clip(p, 0, None) for p in runs]
    return dict(MAE=float(np.mean([mae(y_test, p) for p in clipped])), MAE_std=float(np.std([mae(y_test, p) for p in clipped])),
                MAE_unclipped=float(np.mean([mae(y_test, p) for p in runs])), prediction=np.mean(clipped, axis=0), seeds=clipped)


def rf_fit_predict(cols, history=split):
    rows = slice(max(0, split - history), split)
    m = random_forest().fit(df[cols].iloc[rows], df[TARGET].iloc[rows])
    return m, np.clip(m.predict(test[cols]), 0, None)


t0 = time.time()
# ---------------- base models on the 7 inputs ----------------
preds = {}
for name, m in [("Linear Regression", LinearRegression()), ("Decision Tree", DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE))]:
    m.fit(df[FEATURES].iloc[:split], df[TARGET].iloc[:split])
    preds[name] = np.clip(m.predict(test[FEATURES]), 0, None)
rf, preds["Random Forest"] = rf_fit_predict(FEATURES)
joblib.dump(rf, ROOT / "models/random_forest_nasa.joblib", compress=3)
base_lstm = lstm_summary(lstm_runs(FEATURES))
preds["LSTM"] = base_lstm["prediction"]
print(f"base: RF MAE {mae(y_test, preds['Random Forest']):.4f}, LSTM MAE {base_lstm['MAE']:.4f} "
      f"(unclipped {base_lstm['MAE_unclipped']:.4f}) [{time.time() - t0:.0f}s]", flush=True)

pred_df = pd.DataFrame({"date": test.date.dt.strftime("%Y-%m-%d"), "actual": y_test, **{k: v for k, v in preds.items()}})
pred_df.to_csv(ROOT / "report/showcase_test_predictions.csv", index=False, float_format="%.4f")

# ---------------- 1-2. water budget and error by season ----------------
seed_mae = lambda m: float(np.mean([mae(y_test[m], p[m]) for p in base_lstm["seeds"]]))  # noqa: E731
season = test.date.dt.month.map(SEASON)
order = ["Monsoon", "Post-monsoon and winter", "Pre-monsoon"]
budget, season_mae = {}, {}
for s in order:
    m = (season == s).values
    budget[s] = dict(days=int(m.sum()), actual_mm=float(y_test[m].sum()), **{k: float(v[m].sum()) for k, v in preds.items()})
    season_mae[s] = {k: mae(y_test[m], v[m]) for k, v in preds.items()} | {"LSTM": seed_mae(m)}
budget["Whole test period"] = dict(days=len(y_test), actual_mm=float(y_test.sum()), **{k: float(v.sum()) for k, v in preds.items()})
season_mae["Whole test period"] = {k: mae(y_test, v) for k, v in preds.items()} | {"LSTM": base_lstm["MAE"]}
peak = y_test >= np.quantile(y_test, 0.9)
peak_mae = {k: mae(y_test[peak], v[peak]) for k, v in preds.items()} | {"LSTM": seed_mae(peak)}

weekly = pred_df.assign(date=pd.to_datetime(pred_df.date)).set_index("date").resample("W-MON", label="left", closed="left").mean()

# ---------------- 3. cheap sensors ----------------
sensors = {}
for name, cols in SUBSETS.items():
    p_rf = preds["Random Forest"] if name == "All 7 inputs" else rf_fit_predict(cols)[1]
    l = base_lstm if name == "All 7 inputs" else lstm_summary(lstm_runs(cols))
    sensors[name] = dict(inputs=cols, RF=mae(y_test, p_rf), LSTM=l["MAE"], LSTM_std=l["MAE_std"])
    print("sensors", name, {k: round(v, 3) for k, v in sensors[name].items() if isinstance(v, float)}, f"[{time.time() - t0:.0f}s]", flush=True)

# FAO-56's own answer to missing sensors (Eq. 52, Hargreaves): ET0 from Tmax, Tmin and Ra only, then the same Kc,
# effective-rainfall and efficiency steps. "raw" is the textbook equation; "calibrated" fits ET0_PM = a + b * ET0_HS on the
# training days, as FAO-56 recommends for local use, so it draws on the same training data as the models.
sys.path.insert(0, str(ROOT / "src"))
import generate_dataset as G  # noqa: E402
Ra = G.extraterrestrial_radiation(df.date.dt.day_of_year.values, G.LATITUDE_DEG)
et0_hs = 0.0023 * ((df.tmax_C + df.tmin_C) / 2 + 17.8) * np.sqrt(np.clip(df.tmax_C - df.tmin_C, 0, None)) * 0.408 * Ra
slope, intercept = np.polyfit(et0_hs[:split], df.et0_mm[:split], 1)
hs_demand = lambda et0: (np.clip(et0 * df.crop_coefficient - df.effective_rainfall_mm, 0, None) / G.IRRIGATION_EFFICIENCY).values  # noqa: E731
hargreaves = dict(raw=mae(y_test, hs_demand(et0_hs)[split:]), calibrated=mae(y_test, hs_demand(intercept + slope * et0_hs)[split:]),
                  calibration=dict(slope=float(slope), intercept=float(intercept)), inputs=SUBSETS["Rain gauge + max/min thermometer"])
print("hargreaves", {k: v for k, v in hargreaves.items() if k != "inputs"}, flush=True)

# ---------------- 4. history length ----------------
history = {}
for n in HISTORY_DAYS:
    p_rf = preds["Random Forest"] if n == split else rf_fit_predict(FEATURES, n)[1]
    l = base_lstm if n == split else lstm_summary(lstm_runs(FEATURES, n))
    history[n] = dict(years=round(n / 365.25, 1), RF=mae(y_test, p_rf), LSTM=l["MAE"], LSTM_std=l["MAE_std"])
    print("history", n, {k: round(v, 3) for k, v in history[n].items()}, f"[{time.time() - t0:.0f}s]", flush=True)

summary = dict(
    test_period=[pred_df.date.iloc[0], pred_df.date.iloc[-1]],
    note="Predictions clipped at zero. Deep-model MAE = mean of 3 per-seed MAEs.",
    base_mae={k: mae(y_test, v) for k, v in preds.items()} | {"LSTM": base_lstm["MAE"], "LSTM_unclipped": base_lstm["MAE_unclipped"]},
    water_budget_mm=budget, season_mae=season_mae,
    peak_days=dict(threshold_mm=float(np.quantile(y_test, 0.9)), days=int(peak.sum()), mae=peak_mae),
    weekly=dict(week_start=[d.strftime("%Y-%m-%d") for d in weekly.index], **{c: weekly[c].round(3).tolist() for c in weekly.columns}),
    cheap_sensors=sensors, hargreaves_baseline=hargreaves, history_length=history,
)
(ROOT / "report/showcase_summary.json").write_text(json.dumps(summary, indent=2))
print(f"done in {time.time() - t0:.0f}s")
