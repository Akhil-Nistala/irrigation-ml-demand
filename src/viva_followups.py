"""Checks prompted by the mock-viva examiner's questions (real Nagpur weather unless stated).

  1. Persistence baseline ("today = yesterday") on both datasets          (P1-06)
  2. How often the synthetic 8 mm/day ET0 cap binds                        (P1-08)
  3. Dates of the deep models' early-stopping validation block            (P2-05)
  4. Block-bootstrap confidence interval for Random Forest minus LSTM MAE (P2-06)
  5. A physics-informed linear baseline (Kc x weather interactions)        (P2-07)
  6. Linear Regression's negative predictions, recounted                  (P2-08)
  7. Adding daily Tmax and Tmin as inputs (MLP, LSTM, Random Forest)       (P2-02, P2-03)
  8. A climatology baseline for the days-ahead test                        (P3-05)
  9. FAO-56 Penman-Monteith with its own missing-data rules                (P3-10)
 10. A test period nobody has looked at: NASA POWER after 22 Jun 2024      (X-03)
 Also: a seasonal clock (Ra) as an extra input and LSTM windows of 1, 3, 7 days (P2-02); a calendar-aware climatology (P3-05).

    python src/viva_followups.py      -> report/viva_followups.json
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import generate_dataset as G  # noqa: E402

RS, SEEDS, WINDOW, SPLIT = 42, [0, 1, 2], 7, 1600
torch.set_num_threads(2)
F7 = ["rainfall_mm", "temperature_C", "humidity_percent", "wind_speed_mps", "solar_radiation_MJ_m2_day", "crop_coefficient", "growth_stage"]
T = "irrigation_demand_mm"
mae = lambda y, p: float(mean_absolute_error(y, p))  # noqa: E731
syn = pd.read_csv(ROOT / "data/irrigation_dataset.csv", parse_dates=["date"])
nasa = pd.read_csv(ROOT / "data/irrigation_dataset_nasa_power.csv", parse_dates=["date"])
raw = pd.read_csv(ROOT / "data/nasa_power_nagpur_raw.csv", parse_dates=["date"])
nasa = nasa.merge(raw[["date", "T2M_MAX", "T2M_MIN"]].rename(columns={"T2M_MAX": "tmax_C", "T2M_MIN": "tmin_C"}), on="date")
out, t0 = {}, time.time()
rf = lambda: RandomForestRegressor(n_estimators=300, max_depth=10, random_state=RS, n_jobs=2)  # noqa: E731


class MLP(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(f, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x[:, -1]).squeeze(-1)


class LSTM(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.lstm = nn.LSTM(f, 48, batch_first=True)
        self.head = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.head(self.lstm(x)[0][:, -1]).squeeze(-1)


def deep(cls, X, y, tr_end, te_end, X_te=None, y_te=None, window=WINDOW):
    """Train on windows ending before tr_end, test on windows ending in [tr_end, te_end) (or on X_te, y_te); mean MAE of 3 seeds."""
    f = X.shape[1]
    win = lambda A, ends: np.stack([A[i - window + 1:i + 1] for i in ends]).astype(np.float32)  # noqa: E731
    tr = np.arange(WINDOW - 1, tr_end)
    wtr = win(X, tr)
    if X_te is None:
        te = np.arange(tr_end, te_end)
        wte, yte = win(X, te), y[te]
    else:
        te = np.arange(window - 1, len(X_te))
        wte, yte = win(X_te, te), y_te[te]
    mu, sd = wtr.reshape(-1, f).mean(0), wtr.reshape(-1, f).std(0) + 1e-6
    ymu, ysd = y[tr].mean(), y[tr].std()
    res = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        Xt, yt = torch.tensor((wtr - mu) / sd), torch.tensor(((y[tr] - ymu) / ysd).astype(np.float32))
        nval = int(len(Xt) * 0.1)
        Xf, yf, Xv, yv = Xt[:-nval], yt[:-nval], Xt[-nval:], yt[-nval:]
        m = cls(f)
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
            res.append(mae(yte, np.clip(m(torch.tensor((wte - mu) / sd)).numpy() * ysd + ymu, 0, None)))
    return float(np.mean(res)), float(np.std(res))


# 1. persistence: predict today's demand with yesterday's
out["persistence_mae"] = {n: mae(d[T].values[SPLIT:], d[T].values[SPLIT - 1:-1]) for n, d in [("synthetic", syn), ("nasa", nasa)]}

# 2. synthetic ET0 cap
cap = syn.et0_mm >= 7.999
out["synthetic_et0_cap"] = dict(train_days_at_cap=int(cap.iloc[:SPLIT].sum()), test_days_at_cap=int(cap.iloc[SPLIT:].sum()),
                                test_days_at_cap_apr_may_2024=int((cap & syn.date.between("2024-04-01", "2024-05-31")).sum()),
                                plateau_mm=8 * 1.2 / 0.75)
print("1-2", out["persistence_mae"], out["synthetic_et0_cap"], flush=True)

# 3. validation block of the deep models: last 10% of the 1,594 training windows
n_tr = SPLIT - (WINDOW - 1)
first_val = SPLIT - int(n_tr * 0.1)
out["validation_block"] = dict(start=str(nasa.date.iloc[first_val].date()), end=str(nasa.date.iloc[SPLIT - 1].date()), days=SPLIT - first_val)

# 4. moving-block bootstrap of the MAE difference (Random Forest minus seed-averaged LSTM), 14-day blocks
pr = pd.read_csv(ROOT / "report/showcase_test_predictions.csv")
e_rf, e_ls = np.abs(pr["Random Forest"] - pr.actual).values, np.abs(pr.LSTM - pr.actual).values
rng, n, blk = np.random.default_rng(RS), len(pr), 14
diffs, rels = [], []
for _ in range(10000):
    starts = rng.integers(0, n - blk + 1, size=int(np.ceil(n / blk)))
    idx = np.concatenate([np.arange(s, s + blk) for s in starts])[:n]
    diffs.append(e_rf[idx].mean() - e_ls[idx].mean())
    rels.append(1 - e_ls[idx].mean() / e_rf[idx].mean())
out["bootstrap_rf_minus_lstm"] = dict(point=float(e_rf.mean() - e_ls.mean()), ci95=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                                      relative_point=float(1 - e_ls.mean() / e_rf.mean()),
                                      relative_ci95=[float(np.percentile(rels, 2.5)), float(np.percentile(rels, 97.5))],
                                      share_of_days_lstm_closer=float((e_ls < e_rf).mean()), block_days=blk, resamples=10000)
print("3-4", out["validation_block"], out["bootstrap_rf_minus_lstm"], flush=True)


# 5. physics-informed linear baseline
def phys(d):
    k = d.crop_coefficient
    X = pd.DataFrame({"kc": k, "kc_rs": k * d.solar_radiation_MJ_m2_day, "kc_t": k * d.temperature_C, "kc_rh": k * d.humidity_percent,
                      "kc_u2": k * d.wind_speed_mps, "rain": d.rainfall_mm})
    X2 = X.assign(scs_rain=G.effective_rainfall_scs(d.rainfall_mm.values))
    return X, X2


lin = {}
for name, d in [("synthetic", syn), ("nasa", nasa)]:
    X, X2 = phys(d)
    y = d[T].values
    lin[name] = {"Kc x weather": mae(y[SPLIT:], LinearRegression().fit(X[:SPLIT], y[:SPLIT]).predict(X[SPLIT:])),
                 "Kc x weather + SCS rain term": mae(y[SPLIT:], LinearRegression().fit(X2[:SPLIT], y[:SPLIT]).predict(X2[SPLIT:])),
                 "plain 7 inputs": mae(y[SPLIT:], LinearRegression().fit(d[F7][:SPLIT], y[:SPLIT]).predict(d[F7][SPLIT:]))}
out["physics_informed_linear"] = lin

# 6. Linear Regression negatives
out["lr_negative_predictions"] = {n: int((LinearRegression().fit(d[F7][:SPLIT], d[T][:SPLIT]).predict(d[F7][SPLIT:]) < 0).sum())
                                  for n, d in [("synthetic", syn), ("nasa", nasa)]}
print("5-6", lin, out["lr_negative_predictions"], flush=True)

# 7. adding Tmax and Tmin
F9 = F7 + ["tmax_C", "tmin_C"]
y = nasa[T].values.astype(np.float32)
nine = {}
for label, feats in [("7 inputs", F7), ("9 inputs (+Tmax, Tmin)", F9)]:
    X = nasa[feats].values.astype(np.float32)
    nine[label] = dict(RF=mae(y[SPLIT:], rf().fit(nasa[feats][:SPLIT], y[:SPLIT]).predict(nasa[feats][SPLIT:])),
                       MLP=deep(MLP, X, y, SPLIT, len(nasa))[0], LSTM=deep(LSTM, X, y, SPLIT, len(nasa))[0])
    print("7", label, nine[label], f"[{time.time() - t0:.0f}s]", flush=True)
out["tmax_tmin_inputs"] = nine

# 7b. a seasonal clock: extraterrestrial radiation Ra (latitude + day of year), which enters Penman-Monteith through Rs/Rso
nasa["ra_MJ"] = G.extraterrestrial_radiation(nasa.date.dt.day_of_year.values, G.LATITUDE_DEG)
clock = {}
for label, feats in [("7 inputs + Ra", F7 + ["ra_MJ"]), ("9 inputs + Ra", F9 + ["ra_MJ"])]:
    X = nasa[feats].values.astype(np.float32)
    clock[label] = dict(RF=mae(y[SPLIT:], rf().fit(nasa[feats][:SPLIT], y[:SPLIT]).predict(nasa[feats][SPLIT:])),
                        MLP=deep(MLP, X, y, SPLIT, len(nasa))[0], LSTM=deep(LSTM, X, y, SPLIT, len(nasa))[0])
    print("7b", label, clock[label], f"[{time.time() - t0:.0f}s]", flush=True)
out["seasonal_clock_inputs"] = clock

# 7c. does the LSTM need the past week at all? Same LSTM, windows of 1 and 3 days (same training days as the 7-day run)
X7 = nasa[F7].values.astype(np.float32)
out["lstm_window_days"] = {str(w): deep(LSTM, X7, y, SPLIT, len(nasa), window=w)[0] for w in [1, 3, 7]}
print("7c", out["lstm_window_days"], f"[{time.time() - t0:.0f}s]", flush=True)

# 8. climatology for days ahead: mean training demand for each day of year, smoothed over 15 days
doy = nasa.date.dt.day_of_year.values
clim = pd.Series(y[:SPLIT]).groupby(doy[:SPLIT]).mean().reindex(range(1, 367)).interpolate(limit_direction="both")
clim_s = pd.concat([clim.iloc[-7:], clim, clim.iloc[:7]]).rolling(15, center=True).mean().iloc[7:-7]
out["climatology_mae"] = dict(day_of_year_mean=mae(y[SPLIT:], clim.loc[doy[SPLIT:]].values),
                              smoothed_15_day=mae(y[SPLIT:], clim_s.loc[doy[SPLIT:]].values))
# 8b. a fairer climatology that knows the crop calendar: typical ET0 for the day of year x Kc on the day, typical effective rain
et0_c = nasa.et0_mm[:SPLIT].groupby(doy[:SPLIT]).mean().reindex(range(1, 367)).interpolate(limit_direction="both")
pe_c = nasa.effective_rainfall_mm[:SPLIT].groupby(doy[:SPLIT]).mean().reindex(range(1, 367)).interpolate(limit_direction="both")
kc_te = nasa.crop_coefficient.values[SPLIT:]
out["climatology_mae"]["typical_et0_x_kc_minus_typical_pe"] = mae(y[SPLIT:], np.clip(et0_c.loc[doy[SPLIT:]].values * kc_te - pe_c.loc[doy[SPLIT:]].values, 0, None) / G.IRRIGATION_EFFICIENCY)
out["climatology_mae"]["typical_et0_x_kc_no_rain"] = mae(y[SPLIT:], et0_c.loc[doy[SPLIT:]].values * kc_te / G.IRRIGATION_EFFICIENCY)
print("8", out["climatology_mae"], flush=True)

# 9. FAO-56 missing-data Penman-Monteith from a rain gauge + max/min thermometer:
#    Rs = 0.16 sqrt(Tmax - Tmin) Ra (Eq. 50, interior site), ea = e(Tmin) (Eq. 48), u2 = 2 m/s (Ch. 3 default)
Tx, Tn = nasa.tmax_C.values, nasa.tmin_C.values
Tm = (Tx + Tn) / 2
Ra = G.extraterrestrial_radiation(doy, G.LATITUDE_DEG)
Rs = 0.16 * np.sqrt(np.clip(Tx - Tn, 0, None)) * Ra
rh_equiv = 100 * G.saturation_vapor_pressure(Tn) / G.saturation_vapor_pressure(Tm)  # makes the code's ea = e(Tmin)
et0_md = np.clip(G.fao56_penman_monteith(Tm, Tx, Tn, rh_equiv, np.full_like(Tm, 2.0), Rs, Ra, G.ELEVATION_M), 0, None)
dem = lambda e0: (np.clip(e0 * nasa.crop_coefficient - nasa.effective_rainfall_mm, 0, None) / G.IRRIGATION_EFFICIENCY).values  # noqa: E731
b, a = np.polyfit(et0_md[:SPLIT], nasa.et0_mm[:SPLIT], 1)
out["fao56_missing_data_pm"] = dict(raw=mae(y[SPLIT:], dem(et0_md)[SPLIT:]), calibrated_on_training_days=mae(y[SPLIT:], dem(a + b * et0_md)[SPLIT:]))
print("9", out["fao56_missing_data_pm"], flush=True)

# 10. an untouched period: NASA POWER for Nagpur after 22 Jun 2024, same FAO-56 pipeline; models trained on all 2,000 days
new_path = ROOT / "data/nasa_power_nagpur_after_2024-06-22_raw.csv"
params = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "ALLSKY_SFC_SW_DWN", "PRECTOTCORR"]
if not new_path.exists():
    url = ("https://power.larc.nasa.gov/api/temporal/daily/point?parameters=" + ",".join(params) +
           "&community=AG&longitude=79.09&latitude=21.15&start=20240623&end=20261110&format=JSON")
    with urllib.request.urlopen(url, timeout=180) as r:
        p = json.load(r)["properties"]["parameter"]
    nr = pd.DataFrame({k: pd.Series(v) for k, v in p.items()})
    nr.index = pd.to_datetime(nr.index, format="%Y%m%d")
    nr.index.name = "date"
    nr.to_csv(new_path)
nr = pd.read_csv(new_path, index_col="date", parse_dates=True)
nr = nr[(nr > -998).all(axis=1)]                                          # drop days NASA has not filled yet
nr = nr.loc[: nr.index[np.argmax(np.diff(nr.index.values).astype("timedelta64[D]").astype(int) > 1)] if (np.diff(nr.index.values).astype("timedelta64[D]").astype(int) > 1).any() else nr.index[-1]]
days = np.arange(G.N_DAYS, G.N_DAYS + len(nr))                            # the crop calendar carries on from day 2,000
cyc = days % G.CYCLE_LENGTH
kc = np.array([G.crop_coefficient_and_stage(c)[0] for c in cyc])
st = np.array([G.crop_coefficient_and_stage(c)[1] for c in cyc])
et0_n = np.clip(G.fao56_penman_monteith(nr.T2M.values, nr.T2M_MAX.values, nr.T2M_MIN.values, nr.RH2M.values, nr.WS2M.values,
                                        nr.ALLSKY_SFC_SW_DWN.values, G.extraterrestrial_radiation(nr.index.day_of_year.values, G.LATITUDE_DEG), G.ELEVATION_M), 0, None)
pe_n = G.effective_rainfall_scs(nr.PRECTOTCORR.values)
y_n = np.clip(np.clip(et0_n * kc - pe_n, 0, None) / G.IRRIGATION_EFFICIENCY + np.random.default_rng(2024).normal(0, 0.18, len(nr)), 0, None)
new = pd.DataFrame({"rainfall_mm": nr.PRECTOTCORR.values, "temperature_C": nr.T2M.values, "humidity_percent": nr.RH2M.values,
                    "wind_speed_mps": nr.WS2M.values, "solar_radiation_MJ_m2_day": nr.ALLSKY_SFC_SW_DWN.values,
                    "crop_coefficient": kc, "growth_stage": st, T: y_n})
both = pd.concat([nasa[F7 + [T]], new[F7 + [T]]], ignore_index=True)      # so the first new days have a 7-day window
Xb, yb = both[F7].values.astype(np.float32), both[T].values.astype(np.float32)
unseen = dict(start=str(nr.index[0].date()), end=str(nr.index[-1].date()), days=len(nr), mean_demand=float(y_n.mean()),
              LR=mae(y_n, np.clip(LinearRegression().fit(nasa[F7], nasa[T]).predict(new[F7]), 0, None)),
              RF=mae(y_n, np.clip(rf().fit(nasa[F7], nasa[T]).predict(new[F7]), 0, None)),
              LSTM=deep(LSTM, Xb, yb, G.N_DAYS, len(both))[0],
              persistence=mae(y_n[1:], y_n[:-1]))
out["unseen_period"] = unseen
print("10", unseen, f"[{time.time() - t0:.0f}s]", flush=True)

(ROOT / "report/viva_followups.json").write_text(json.dumps(out, indent=2))
print(f"done in {time.time() - t0:.0f}s")
