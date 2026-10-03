"""Experiment 3 dataset: REAL daily weather from NASA POWER for Nagpur + the same FAO-56 target.

Weather (observed / satellite-derived, NASA POWER Agroclimatology community, daily point API):
    T2M, T2M_MAX, T2M_MIN  -> temperature_C (mean), plus Tmax/Tmin for Penman-Monteith
    RH2M                   -> humidity_percent
    WS2M                   -> wind_speed_mps (2 m height, as FAO-56 expects)
    ALLSKY_SFC_SW_DWN      -> solar_radiation_MJ_m2_day
    PRECTOTCORR            -> rainfall_mm (bias-corrected; IMERG-based)
Same location (21.15 N, 79.09 E, ~ Nagpur), same 2,000 days from 2019-01-01, same columns as data/irrigation_dataset.csv.

The target is computed with the same FAO-56 pipeline as src/generate_dataset.py (Penman-Monteith ET0,
maize Kc calendar, USDA-SCS effective rainfall, 75% efficiency, N(0, 0.18) noise, seed 42), so what changes versus the
synthetic dataset is the weather. Two deliberate differences: real daily Tmax/Tmin replace the synthetic monthly diurnal
range, and ET0 is not capped at 8 mm/day (real pre-monsoon days exceed it).

    python src/nasa_power_dataset.py          # downloads once to data/nasa_power_nagpur_raw.csv, then builds the dataset
"""
import json
import pathlib
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import generate_dataset as G  # noqa: E402  (FAO-56 functions and constants)

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data/nasa_power_nagpur_raw.csv"
OUT = ROOT / "data/irrigation_dataset_nasa_power.csv"
LAT, LON = G.LATITUDE_DEG, 79.09
DATES = pd.date_range(G.START_DATE, periods=G.N_DAYS, freq="D")
PARAMS = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "ALLSKY_SFC_SW_DWN", "PRECTOTCORR"]


def download():
    url = ("https://power.larc.nasa.gov/api/temporal/daily/point?parameters=" + ",".join(PARAMS) +
           f"&community=AG&longitude={LON}&latitude={LAT}&start={DATES[0]:%Y%m%d}&end={DATES[-1]:%Y%m%d}&format=JSON")
    with urllib.request.urlopen(url, timeout=120) as r:
        p = json.load(r)["properties"]["parameter"]
    raw = pd.DataFrame({k: pd.Series(v) for k, v in p.items()})
    raw.index = pd.to_datetime(raw.index, format="%Y%m%d")
    raw.index.name = "date"
    raw.to_csv(RAW)
    return raw


raw = pd.read_csv(RAW, index_col="date", parse_dates=True) if RAW.exists() else download()
raw = raw.reindex(DATES)
missing = int((raw.isna() | (raw <= -998)).sum().sum())
raw = raw.mask(raw <= -998).interpolate(limit_direction="both")   # POWER uses -999 for missing values

T, Tmax, Tmin = raw.T2M.values, raw.T2M_MAX.values, raw.T2M_MIN.values
RH, u2, Rs, P = raw.RH2M.values, raw.WS2M.values, raw.ALLSKY_SFC_SW_DWN.values, raw.PRECTOTCORR.values

Ra = G.extraterrestrial_radiation(DATES.dayofyear.values, LAT)
et0_raw = G.fao56_penman_monteith(T, Tmax, Tmin, RH, u2, Rs, Ra, G.ELEVATION_M)
# The synthetic pipeline clipped ET0 to 1-8 mm/day. With real weather ~10% of days (Apr-Jun heat) exceed 8, which is
# physically plausible for Nagpur, so the real target is NOT clipped (only kept non-negative).
et0 = np.clip(et0_raw, 0.0, None)
kc_stage = [G.crop_coefficient_and_stage(d) for d in np.arange(G.N_DAYS) % G.CYCLE_LENGTH]
kc = np.array([k for k, _ in kc_stage])
stage = np.array([s for _, s in kc_stage])
etc = et0 * kc
pe = G.effective_rainfall_scs(P)
gross = np.clip(etc - pe, 0, None) / G.IRRIGATION_EFFICIENCY
noise = np.random.default_rng(G.RANDOM_STATE).normal(0, 0.18, G.N_DAYS)
demand = np.clip(gross + noise, 0, None)
prev = np.roll(demand, 1)
prev[0] = demand.mean()

df = pd.DataFrame({
    "date": DATES, "rainfall_mm": P.round(2), "temperature_C": T.round(2), "humidity_percent": RH.round(2),
    "wind_speed_mps": u2.round(2), "solar_radiation_MJ_m2_day": Rs.round(2), "previous_irrigation_mm": prev.round(3),
    "crop_coefficient": kc.round(3), "growth_stage": stage, "et0_mm": et0.round(3), "etc_mm": etc.round(3),
    "effective_rainfall_mm": pe.round(3), "irrigation_demand_mm": demand.round(3),
})
df.to_csv(OUT, index=False)

# ---------------- real vs synthetic weather: summary + figure ----------------
syn = pd.read_csv(ROOT / "data/irrigation_dataset.csv", parse_dates=["date"])
cols = ["rainfall_mm", "temperature_C", "humidity_percent", "wind_speed_mps", "solar_radiation_MJ_m2_day", "et0_mm",
        "irrigation_demand_mm"]
summary = pd.DataFrame({
    "synthetic_mean": syn[cols].mean(), "nasa_mean": df[cols].mean(),
    "synthetic_std": syn[cols].std(), "nasa_std": df[cols].std(),
    "lag1_autocorr_synthetic": [syn[c].autocorr(1) for c in cols], "lag1_autocorr_nasa": [df[c].autocorr(1) for c in cols],
})
summary.loc["annual_rainfall_mm", ["synthetic_mean", "nasa_mean"]] = [syn.rainfall_mm.sum() / (G.N_DAYS / 365.25),
                                                                       df.rainfall_mm.sum() / (G.N_DAYS / 365.25)]
summary.to_csv(ROOT / "report/nasa_vs_synthetic_weather.csv")
meta = dict(source="NASA POWER daily point API, community AG", latitude=LAT, longitude=LON, start=str(DATES[0].date()),
            end=str(DATES[-1].date()), days=G.N_DAYS, parameters=PARAMS, missing_values_interpolated=missing,
            et0_days_above_synthetic_cap_8mm=int((et0_raw > 8).sum()), et0_max_mm=float(et0_raw.max()),
            rain_days_over_1mm=int((P > 1).sum()), max_daily_rain_mm=float(P.max()))
(ROOT / "report/nasa_power_dataset_meta.json").write_text(json.dumps(meta, indent=2))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
GREY, TEAL = "#9AA5A8", "#0B4F6C"
fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
monthly = lambda d, c, f="mean": d.groupby(d.date.dt.month)[c].agg(f)  # noqa: E731
panels = [("rainfall_mm", "Rainfall (mm/month, avg year)", "sum"), ("temperature_C", "Mean temperature (°C)", "mean"),
          ("humidity_percent", "Relative humidity (%)", "mean"), ("wind_speed_mps", "Wind speed (m/s)", "mean"),
          ("solar_radiation_MJ_m2_day", "Solar radiation (MJ/m²/day)", "mean"), ("irrigation_demand_mm", "Irrigation demand (mm/day)", "mean")]
years = G.N_DAYS / 365.25
for ax, (c, title, f) in zip(axes.flat, panels):
    s, n = monthly(syn, c, f), monthly(df, c, f)
    if f == "sum":
        s, n = s / years, n / years
    ax.plot(s.index, s.values, "o-", color=GREY, lw=2, label="Synthetic (Exp. 1-2)")
    ax.plot(n.index, n.values, "o-", color=TEAL, lw=2, label="NASA POWER (real)")
    ax.set_title(title)
    ax.set_xticks(range(1, 13), list("JFMAMJJASOND"))
axes[0, 0].legend(frameon=False)
plt.tight_layout()
plt.savefig(ROOT / "figures/13_nasa_vs_synthetic_weather.png", dpi=150)
plt.close()

print(json.dumps(meta, indent=2))
print(summary.round(2).to_string())
