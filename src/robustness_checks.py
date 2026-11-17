"""Robustness checks prepared for the viva, on the same data and setup as Parts 2-3.

  1. Gradient Boosting   : was in the very first version (best by a hair) and later dropped; re-run on the final 7-input setups.
  2. Random Forest tuning: does the RF result depend on our fixed settings (300 trees, depth 10)?
  3. Split choice        : a random 80/20 split against our chronological split, to show the leakage it causes.
  4. Rolling test years  : expanding-window evaluation on three successive 400-day test blocks (RF, LR, LSTM).
  5. LSTM size           : 24, 48 (ours) and 96 hidden units.
  6. Physics audit       : (a) effect of the vapour-pressure shortcut in our Penman-Monteith code (ea from e(Tmean)
                           instead of FAO-56 Eq. 19's mean of e(Tmax), e(Tmin)); (b) share of rain the daily FAO/AGLW-type
                           formula (earlier misnamed USDA-SCS) counts as effective (it gives zero below 16.7 mm/day).

    python src/robustness_checks.py        -> report/robustness_checks.json
"""
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import generate_dataset as G  # noqa: E402

RS, SEEDS, WINDOW = 42, [0, 1, 2], 7
torch.set_num_threads(2)
F7 = ["rainfall_mm", "temperature_C", "humidity_percent", "wind_speed_mps", "solar_radiation_MJ_m2_day", "crop_coefficient", "growth_stage"]
F8 = F7[:5] + ["previous_irrigation_mm"] + F7[5:]
T = "irrigation_demand_mm"
syn = pd.read_csv(ROOT / "data/irrigation_dataset.csv", parse_dates=["date"])
nasa = pd.read_csv(ROOT / "data/irrigation_dataset_nasa_power.csv", parse_dates=["date"])
split = 1600
mae = lambda y, p: float(mean_absolute_error(y, p))  # noqa: E731
out, t0 = {}, time.time()


def rf(**kw):
    return RandomForestRegressor(**{"n_estimators": 300, "max_depth": 10, "random_state": RS, "n_jobs": 2, **kw})


def fit_score(model, d, feats, tr=slice(0, split), te=slice(split, None)):
    model.fit(d[feats].iloc[tr], d[T].iloc[tr])
    p = model.predict(d[feats].iloc[te])
    return dict(MAE=mae(d[T].iloc[te], p), R2=float(r2_score(d[T].iloc[te], p)))


# 1. Gradient Boosting (sklearn defaults, as in the first version; and a slower-learning variant)
gb_default = lambda: GradientBoostingRegressor(random_state=RS)  # noqa: E731
gb_slow = lambda: GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=RS)  # noqa: E731
out["gradient_boosting"] = {
    "Part 1 setup (synthetic, 8 inputs)": dict(GB=fit_score(gb_default(), syn, F8), RF=fit_score(rf(), syn, F8)),
    "synthetic, 7 inputs": dict(GB=fit_score(gb_default(), syn, F7), GB_500=fit_score(gb_slow(), syn, F7), RF=fit_score(rf(), syn, F7)),
    "NASA POWER, 7 inputs": dict(GB=fit_score(gb_default(), nasa, F7), GB_500=fit_score(gb_slow(), nasa, F7), RF=fit_score(rf(), nasa, F7)),
}
print("gb", json.dumps(out["gradient_boosting"]), f"[{time.time() - t0:.0f}s]", flush=True)

# 2. Random Forest settings on real weather
grid = {}
for depth in [6, 10, 15, None]:
    for n in [100, 300, 1000]:
        grid[f"depth {depth}, {n} trees"] = fit_score(rf(max_depth=depth, n_estimators=n), nasa, F7)["MAE"]
out["rf_grid_nasa_mae"] = grid
print("rf grid", grid, f"[{time.time() - t0:.0f}s]", flush=True)

# 3. Random split against chronological split (real weather, RF and LR)
tr_i, te_i = train_test_split(np.arange(len(nasa)), test_size=0.2, random_state=RS, shuffle=True)
out["split_choice"] = {
    "chronological": dict(RF=fit_score(rf(), nasa, F7)["MAE"], LR=fit_score(LinearRegression(), nasa, F7)["MAE"]),
    "random": dict(RF=fit_score(rf(), nasa, F7, tr_i, te_i)["MAE"], LR=fit_score(LinearRegression(), nasa, F7, tr_i, te_i)["MAE"]),
    "random, with yesterday's demand": dict(RF=fit_score(rf(), nasa, F8, tr_i, te_i)["MAE"]),
}
print("split", out["split_choice"], f"[{time.time() - t0:.0f}s]", flush=True)


# 4-5. LSTM helpers (same architecture and training loop as src/no_lag_experiment.py, hidden size as a parameter)
class LSTM(nn.Module):
    def __init__(self, f, hidden=48):
        super().__init__()
        self.lstm = nn.LSTM(f, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.head(self.lstm(x)[0][:, -1]).squeeze(-1)


def lstm_mae(d, tr_start, tr_end, te_end, hidden=48):
    X = d[F7].values.astype(np.float32)
    y = d[T].values.astype(np.float32)
    tr = np.arange(max(WINDOW - 1, tr_start), tr_end)
    te = np.arange(tr_end, te_end)
    win = lambda ends: np.stack([X[i - WINDOW + 1:i + 1] for i in ends])  # noqa: E731
    wtr, wte = win(tr), win(te)
    mu, sd = wtr.reshape(-1, 7).mean(0), wtr.reshape(-1, 7).std(0) + 1e-6
    ymu, ysd = y[tr].mean(), y[tr].std()
    maes = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        Xt, yt = torch.tensor((wtr - mu) / sd), torch.tensor((y[tr] - ymu) / ysd)
        nval = int(len(Xt) * 0.1)
        Xf, yf, Xv, yv = Xt[:-nval], yt[:-nval], Xt[-nval:], yt[-nval:]
        m = LSTM(7, hidden)
        opt = torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=1e-4)
        best, state, wait = np.inf, None, 0
        for _ in range(400):
            m.train()
            perm = torch.randperm(len(Xf))
            for b in range(0, len(Xf), 64):
                j = perm[b:b + 64]
                opt.zero_grad()
                nn.functional.mse_loss(m(Xf[j]), yf[j]).backward()
                opt.step()
            m.eval()
            with torch.no_grad():
                v = nn.functional.mse_loss(m(Xv), yv).item()
            if v < best - 1e-5:
                best, state, wait = v, {k: t.clone() for k, t in m.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= 30:
                    break
        m.load_state_dict(state)
        m.eval()
        with torch.no_grad():
            p = m(torch.tensor((wte - mu) / sd)).numpy() * ysd + ymu
        maes.append(mae(y[te], np.clip(p, 0, None)))
    return float(np.mean(maes)), float(np.std(maes))


# 4. rolling test years: train on everything before each 400-day block
folds = {}
for start in [800, 1200, 1600]:
    end = start + 400
    tr, te = slice(0, start), slice(start, end)
    folds[f"{nasa.date.iloc[start]:%d %b %Y} - {nasa.date.iloc[end - 1]:%d %b %Y}"] = dict(
        train_days=start, LR=fit_score(LinearRegression(), nasa, F7, tr, te)["MAE"], RF=fit_score(rf(), nasa, F7, tr, te)["MAE"],
        LSTM=lstm_mae(nasa, 0, start, end)[0])
    print("fold", start, folds[list(folds)[-1]], f"[{time.time() - t0:.0f}s]", flush=True)
out["rolling_test_years_nasa"] = folds

# 5. LSTM size
out["lstm_hidden_nasa"] = {str(h): dict(zip(["MAE", "MAE_std"], lstm_mae(nasa, 0, split, len(nasa), h))) for h in [24, 48, 96]}
print("lstm size", out["lstm_hidden_nasa"], f"[{time.time() - t0:.0f}s]", flush=True)

# 6a. vapour-pressure shortcut: recompute real-weather ET0 both ways
raw = pd.read_csv(ROOT / "data/nasa_power_nagpur_raw.csv", index_col="date", parse_dates=True).reindex(nasa.date)
Tm, Tx, Tn, RH = raw.T2M.values, raw.T2M_MAX.values, raw.T2M_MIN.values, raw.RH2M.values
Ra = G.extraterrestrial_radiation(nasa.date.dt.day_of_year.values, G.LATITUDE_DEG)
et0_code = np.clip(G.fao56_penman_monteith(Tm, Tx, Tn, RH, raw.WS2M.values, raw.ALLSKY_SFC_SW_DWN.values, Ra, G.ELEVATION_M), 0, None)
# FAO-56 Eq. 19: ea = RHmean/100 * (e(Tmax) + e(Tmin)) / 2. Shift Tmean inside a copy of the formula's ea term only.
es_pair = (G.saturation_vapor_pressure(Tx) + G.saturation_vapor_pressure(Tn)) / 2
ea_code, ea_eq19 = G.saturation_vapor_pressure(Tm) * RH / 100, es_pair * RH / 100
# ET0 with the Eq. 19 ea: same equation, ea swapped (RH rescaled so that e(Tmean) * RH' = es_pair * RH)
et0_eq19 = np.clip(G.fao56_penman_monteith(Tm, Tx, Tn, RH * es_pair / G.saturation_vapor_pressure(Tm), raw.WS2M.values,
                                           raw.ALLSKY_SFC_SW_DWN.values, Ra, G.ELEVATION_M), 0, None)
dem = lambda e0: np.clip(e0 * nasa.crop_coefficient - nasa.effective_rainfall_mm, 0, None) / G.IRRIGATION_EFFICIENCY  # noqa: E731
out["vapour_pressure_audit_nasa"] = dict(
    mean_ea_code_kPa=float(ea_code.mean()), mean_ea_eq19_kPa=float(ea_eq19.mean()),
    mean_et0_code=float(et0_code.mean()), mean_et0_eq19=float(et0_eq19.mean()),
    et0_relative_difference=float(et0_code.mean() / et0_eq19.mean() - 1),
    mean_demand_code=float(dem(et0_code).mean()), mean_demand_eq19=float(dem(et0_eq19).mean()),
    correlation_et0=float(np.corrcoef(et0_code, et0_eq19)[0, 1]))
print("ea audit", out["vapour_pressure_audit_nasa"], flush=True)

# 6b. effective rainfall at a daily step
def pe_audit(d):
    P = d.rainfall_mm.values
    rain = P > 1
    return dict(rain_days=int(rain.sum()), rain_days_with_zero_effective=int((rain & (d.effective_rainfall_mm.values == 0)).sum()),
                share_of_rain_counted_effective=float(d.effective_rainfall_mm.sum() / P.sum()),
                threshold_mm=10 / 0.6)
out["effective_rainfall_audit"] = dict(synthetic=pe_audit(syn), nasa=pe_audit(nasa))
print("pe audit", out["effective_rainfall_audit"], flush=True)

(ROOT / "report/robustness_checks.json").write_text(json.dumps(out, indent=2))
print(f"done in {time.time() - t0:.0f}s")
