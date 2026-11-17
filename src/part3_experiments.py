"""Part 3 experiments: the open questions listed at the end of Part 2, tested on real NASA POWER weather.

  A. New site        IIT Hyderabad (17.59 N, 78.12 E), same dates, maize calendar and FAO-56 pipeline. Models trained at
                     Nagpur are tested at Hyderabad without retraining, and compared with models trained at Hyderabad.
  B. New crop        cotton at Nagpur (FAO-56 calendar 30/50/60/55 days, Kc 0.35 -> 1.15 -> 0.60). Maize-trained models
                     are tested on cotton days, and compared with models trained on cotton.
  C. Days ahead      predict the demand h = 1, 3 or 7 days ahead from the weather up to today plus the crop stage on the
                     target day (known from the sowing date). Baselines: today's FAO-56 demand carried forward
                     (persistence), and the same-day model, which is what a perfect weather forecast would give.
  D. Noisier records train on targets with extra noise (sd 0.5 and 1.0 mm/day, about 10% and 20% of mean demand), as
                     measured irrigation would be, and score against the noise-free FAO-56 demand.

Same 7 inputs, chronological 80/20 split (last 400 days = test), Random Forest settings and LSTM architecture and
training loop as src/no_lag_experiment.py; deep models use 3 seeds and report the mean of the per-seed MAEs.

    python src/part3_experiments.py

Writes data/nasa_power_hyderabad_raw.csv, data/irrigation_dataset_hyderabad.csv, data/irrigation_dataset_cotton.csv
and report/part3_summary.json.
"""
import json
import pathlib
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import generate_dataset as G  # noqa: E402  (FAO-56 functions and constants)

ROOT = pathlib.Path(__file__).resolve().parents[1]
RANDOM_STATE, SEEDS, WINDOW = 42, [0, 1, 2], 7
torch.set_num_threads(2)
FEATURES = ["rainfall_mm", "temperature_C", "humidity_percent", "wind_speed_mps",
            "solar_radiation_MJ_m2_day", "crop_coefficient", "growth_stage"]
TARGET = "irrigation_demand_mm"
MAIZE = dict(lengths=(20, 35, 40, 30), kc=(0.30, 1.20, 0.60))
COTTON = dict(lengths=(30, 50, 60, 55), kc=(0.35, 1.15, 0.60))
HYD_LAT, HYD_LON = 17.59, 78.12
PARAMS = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "ALLSKY_SFC_SW_DWN", "PRECTOTCORR"]
DATES = pd.date_range(G.START_DATE, periods=G.N_DAYS, freq="D")
mae = lambda y, p: float(mean_absolute_error(y, p))  # noqa: E731
t0 = time.time()


# ---------------- datasets ----------------
def download(lat, lon, path):
    """NASA POWER daily point data, AG community; returns (raw frame, elevation in m from the API)."""
    url = ("https://power.larc.nasa.gov/api/temporal/daily/point?parameters=" + ",".join(PARAMS) +
           f"&community=AG&longitude={lon}&latitude={lat}&start={DATES[0]:%Y%m%d}&end={DATES[-1]:%Y%m%d}&format=JSON")
    with urllib.request.urlopen(url, timeout=120) as r:
        js = json.load(r)
    raw = pd.DataFrame({k: pd.Series(v) for k, v in js["properties"]["parameter"].items()})
    raw.index = pd.to_datetime(raw.index, format="%Y%m%d")
    raw.index.name = "date"
    raw.to_csv(path)
    return raw, float(js["geometry"]["coordinates"][2])


def calendar(n, lengths, kc):
    """Kc and growth stage for back-to-back seasons (FAO-56 four-stage curve), same rule as generate_dataset.py."""
    l1, l2, l3, l4 = lengths
    ini, mid, end = kc
    d = np.arange(n) % sum(lengths)
    k = np.select([d < l1, d < l1 + l2, d < l1 + l2 + l3],
                  [ini, ini + (d - l1) / l2 * (mid - ini), mid], mid + (d - l1 - l2 - l3) / l4 * (end - mid))
    stage = 1 + (d >= l1).astype(int) + (d >= l1 + l2) + (d >= l1 + l2 + l3)
    return k, stage


def build(raw, lat, elevation, crop):
    raw = raw.reindex(DATES).mask(raw <= -998).interpolate(limit_direction="both")
    T, Tmax, Tmin = raw.T2M.values, raw.T2M_MAX.values, raw.T2M_MIN.values
    RH, u2, Rs, P = raw.RH2M.values, raw.WS2M.values, raw.ALLSKY_SFC_SW_DWN.values, raw.PRECTOTCORR.values
    Ra = G.extraterrestrial_radiation(DATES.dayofyear.values, lat)
    et0 = np.clip(G.fao56_penman_monteith(T, Tmax, Tmin, RH, u2, Rs, Ra, elevation), 0.0, None)
    kc, stage = calendar(G.N_DAYS, **crop)
    etc, pe = et0 * kc, G.effective_rainfall_scs(P)
    gross = np.clip(etc - pe, 0, None) / G.IRRIGATION_EFFICIENCY
    demand = np.clip(gross + np.random.default_rng(G.RANDOM_STATE).normal(0, 0.18, G.N_DAYS), 0, None)
    return pd.DataFrame({"date": DATES, "rainfall_mm": P.round(2), "temperature_C": T.round(2), "humidity_percent": RH.round(2),
                         "wind_speed_mps": u2.round(2), "solar_radiation_MJ_m2_day": Rs.round(2), "crop_coefficient": kc.round(3),
                         "growth_stage": stage, "et0_mm": et0.round(3), "etc_mm": etc.round(3), "effective_rainfall_mm": pe.round(3),
                         "clean_demand_mm": gross.round(3), TARGET: demand.round(3)})


nag_raw = pd.read_csv(ROOT / "data/nasa_power_nagpur_raw.csv", index_col="date", parse_dates=True)
nag = build(nag_raw, G.LATITUDE_DEG, G.ELEVATION_M, MAIZE)
committed = pd.read_csv(ROOT / "data/irrigation_dataset_nasa_power.csv")
assert np.abs(nag[TARGET].values - committed[TARGET].values).max() < 2e-3, "builder must reproduce the Part 2 dataset"
hyd_path = ROOT / "data/nasa_power_hyderabad_raw.csv"
if hyd_path.exists():
    hyd_raw, hyd_elev = pd.read_csv(hyd_path, index_col="date", parse_dates=True), 565.46
else:
    hyd_raw, hyd_elev = download(HYD_LAT, HYD_LON, hyd_path)
hyd = build(hyd_raw, HYD_LAT, hyd_elev, MAIZE)
cot = build(nag_raw, G.LATITUDE_DEG, G.ELEVATION_M, COTTON)
hyd.drop(columns="clean_demand_mm").to_csv(ROOT / "data/irrigation_dataset_hyderabad.csv", index=False)
cot.drop(columns="clean_demand_mm").to_csv(ROOT / "data/irrigation_dataset_cotton.csv", index=False)
split = int(G.N_DAYS * 0.8)
print(f"datasets ready [{time.time() - t0:.0f}s]", flush=True)


# ---------------- models ----------------
def random_forest(X, y):
    return RandomForestRegressor(n_estimators=300, max_depth=10, random_state=RANDOM_STATE, n_jobs=2).fit(X, y)


class LSTM(nn.Module):  # identical to src/no_lag_experiment.py
    def __init__(self, n_features):
        super().__init__()
        self.lstm = nn.LSTM(n_features, 48, batch_first=True)
        self.head = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.head(self.lstm(x)[0][:, -1]).squeeze(-1)


def windows(X, ends):
    return np.stack([X[i - WINDOW + 1:i + 1] for i in ends]).astype(np.float32)


def lstm(win_tr, y_tr, tests):
    """Train 3 seeds on (win_tr, y_tr); return, for each test window set, the list of per-seed predictions (clipped at 0)."""
    f = win_tr.shape[-1]
    mu, sd = win_tr.reshape(-1, f).mean(0), win_tr.reshape(-1, f).std(0) + 1e-6
    ymu, ysd = y_tr.mean(), y_tr.std()
    out = [[] for _ in tests]
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        Xtr, ytr = torch.tensor((win_tr - mu) / sd), torch.tensor(((y_tr - ymu) / ysd).astype(np.float32))
        nval = int(len(Xtr) * 0.1)
        Xfit, yfit, Xval, yval = Xtr[:-nval], ytr[:-nval], Xtr[-nval:], ytr[-nval:]
        model = LSTM(f)
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
            for k, w in enumerate(tests):
                out[k].append(np.clip(model(torch.tensor((w - mu) / sd)).numpy() * ysd + ymu, 0, None))
    return out


def seed_mae(y, runs):
    return float(np.mean([mae(y, p) for p in runs])), float(np.std([mae(y, p) for p in runs]))


tr_end = np.arange(WINDOW - 1, split)          # windows ending on training days
te_end = np.arange(split, G.N_DAYS)            # windows ending on the 400 test days
Xn, Xh, Xc = (d[FEATURES].values for d in (nag, hyd, cot))
yn, yh, yc = (d[TARGET].values for d in (nag, hyd, cot))

# ---------------- A + B: new site and new crop ----------------
rf_n = random_forest(nag[FEATURES].iloc[:split], yn[:split])
rf_h = random_forest(hyd[FEATURES].iloc[:split], yh[:split])
rf_c = random_forest(cot[FEATURES].iloc[:split], yc[:split])
pred = lambda m, d: np.clip(m.predict(d[FEATURES].iloc[split:]), 0, None)  # noqa: E731
ln_n, ln_h_transfer, ln_c_transfer = lstm(windows(Xn, tr_end), yn[tr_end], [windows(Xn, te_end), windows(Xh, te_end), windows(Xc, te_end)])
(ln_h_local,) = lstm(windows(Xh, tr_end), yh[tr_end], [windows(Xh, te_end)])
(ln_c_local,) = lstm(windows(Xc, tr_end), yc[tr_end], [windows(Xc, te_end)])
site = {
    "Nagpur (trained here)": dict(RF=mae(yn[split:], pred(rf_n, nag)), LSTM=seed_mae(yn[split:], ln_n)[0]),
    "Hyderabad, Nagpur model": dict(RF=mae(yh[split:], pred(rf_n, hyd)), LSTM=seed_mae(yh[split:], ln_h_transfer)[0]),
    "Hyderabad, retrained": dict(RF=mae(yh[split:], pred(rf_h, hyd)), LSTM=seed_mae(yh[split:], ln_h_local)[0]),
}
crop = {
    "Maize (trained on maize)": site["Nagpur (trained here)"],
    "Cotton, maize model": dict(RF=mae(yc[split:], pred(rf_n, cot)), LSTM=seed_mae(yc[split:], ln_c_transfer)[0]),
    "Cotton, retrained": dict(RF=mae(yc[split:], pred(rf_c, cot)), LSTM=seed_mae(yc[split:], ln_c_local)[0]),
}
print("site", site, "\ncrop", crop, f"[{time.time() - t0:.0f}s]", flush=True)
yr = G.N_DAYS / 365.25
context = dict(
    hyderabad=dict(lat=HYD_LAT, lon=HYD_LON, elevation_m=hyd_elev, mean_demand=float(yh.mean()), annual_rain_mm=float(hyd.rainfall_mm.sum() / yr),
                   mean_et0=float(hyd.et0_mm.mean()), mean_temp=float(hyd.temperature_C.mean()), test_mean_demand=float(yh[split:].mean())),
    nagpur=dict(mean_demand=float(yn.mean()), annual_rain_mm=float(nag.rainfall_mm.sum() / yr), mean_et0=float(nag.et0_mm.mean()),
                mean_temp=float(nag.temperature_C.mean()), test_mean_demand=float(yn[split:].mean())),
    cotton=dict(mean_demand=float(yc.mean()), test_mean_demand=float(yc[split:].mean()), cycle_days=sum(COTTON["lengths"]), kc=COTTON["kc"]),
    hyderabad_monthly_demand=hyd.groupby(hyd.date.dt.month)[TARGET].mean().round(3).tolist(),
    nagpur_monthly_demand=nag.groupby(nag.date.dt.month)[TARGET].mean().round(3).tolist(),
)

# ---------------- C: days ahead ----------------
ahead = {}
for h in [1, 3, 7]:
    # inputs: weather and crop for the 7 days up to today (t), plus Kc and stage on the target day (t + h)
    kc_t = np.stack([nag.crop_coefficient.values, nag.growth_stage.values], 1)
    tr_t = np.arange(WINDOW - 1, split - h)              # today t such that the target t + h is a training day
    te_t = np.arange(split - h, G.N_DAYS - h)            # targets are exactly the 400 test days
    Xh_ = np.concatenate([Xn, np.zeros((G.N_DAYS, 2))], 1)
    def win_h(ts):  # noqa: E306
        w = windows(Xh_, ts)
        w[:, :, -2:] = kc_t[ts + h][:, None, :]
        return w
    rf = random_forest(np.hstack([Xn[tr_t], kc_t[tr_t + h]]), yn[tr_t + h])
    p_rf = np.clip(rf.predict(np.hstack([Xn[te_t], kc_t[te_t + h]])), 0, None)
    (runs,) = lstm(win_h(tr_t), yn[tr_t + h], [win_h(te_t)])
    ahead[h] = dict(RF=mae(yn[split:], p_rf), LSTM=seed_mae(yn[split:], runs)[0], persistence=mae(yn[split:], yn[te_t]))
    print("ahead", h, ahead[h], f"[{time.time() - t0:.0f}s]", flush=True)
ahead[0] = dict(RF=site["Nagpur (trained here)"]["RF"], LSTM=site["Nagpur (trained here)"]["LSTM"], persistence=None)

# ---------------- D: noisier records ----------------
clean = nag.clean_demand_mm.values
noisy = {}
for sd in [0.18, 0.5, 1.0]:
    y = yn if sd == 0.18 else np.clip(clean + np.random.default_rng(7).normal(0, sd, G.N_DAYS), 0, None)
    rf = random_forest(nag[FEATURES].iloc[:split], y[:split])
    (runs,) = lstm(windows(Xn, tr_end), y[tr_end], [windows(Xn, te_end)])
    noisy[sd] = dict(RF=mae(clean[split:], pred(rf, nag)), LSTM=seed_mae(clean[split:], runs)[0],
                     record_error=mae(clean[split:], y[split:]))
    print("noise", sd, noisy[sd], f"[{time.time() - t0:.0f}s]", flush=True)

summary = dict(note="MAE in mm/day on the 400 test days (20 May 2023 - 22 Jun 2024); predictions clipped at zero; LSTM = mean of 3 per-seed MAEs. "
                    "Noise experiment scored against the noise-free FAO-56 demand.",
               new_site=site, new_crop=crop, days_ahead={str(k): v for k, v in sorted(ahead.items())},
               noisy_records={str(k): v for k, v in noisy.items()}, context=context)
(ROOT / "report/part3_summary.json").write_text(json.dumps(summary, indent=2))
print(f"done in {time.time() - t0:.0f}s")
