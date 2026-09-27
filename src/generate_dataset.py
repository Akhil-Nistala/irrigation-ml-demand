"""
Synthetic daily irrigation-water-demand dataset generator.

Physical basis
--------------
- Reference evapotranspiration (ET0): FAO-56 full Penman-Monteith equation
  (Allen et al., 1998, FAO Irrigation & Drainage Paper No. 56, Ch. 4, Eq. 6),
  using the simplified (mean-RH-only) actual-vapour-pressure form (Eq. 19)
  since only a single daily mean humidity is assumed available.
- Crop evapotranspiration: ETc = ET0 x Kc (FAO-56, Eq. 58), with Kc for maize
  (Zea mays, grain) taken from FAO-56 Table 12 and the four-stage crop
  calendar of FAO-56 Table 11 (subhumid climate, arid/semi-arid irrigation
  practice): Kc_ini=0.30, Kc_mid=1.20, Kc_end=0.60, stage lengths
  20 / 35 / 40 / 30 days (initial / development / mid-season / late-season).
- Effective rainfall: USDA Soil Conservation Service empirical formula
  (USDA-SCS, 1970; also reproduced in FAO Irrigation & Drainage Paper 25,
  Doorenbos & Pruitt, 1977):
      Pe = 0.6P - 10   for P <= 75 (mm/period)
      Pe = 0.8P - 25   for P  > 75 (mm/period)
  NOTE: this formula was developed for MONTHLY/decadal rainfall totals. It is
  applied here at a DAILY time step as a deliberate simplifying assumption
  (clearly documented) so that small, mostly-evaporated showers contribute
  ~zero effective rainfall while large storms are partially discounted for
  runoff losses -- the qualitative behaviour irrigation engineers expect.
- Net irrigation requirement = max(0, ETc - Pe)
- Gross irrigation requirement = Net irrigation requirement / Ei,
  Ei = 0.75 (representative surface/sprinkler irrigation efficiency).

Climate generation
-------------------
Weather is generated from a MONTHLY CLIMATOLOGY table loosely calibrated to a
representative central-Indian, subtropical-monsoon location (~21 N latitude,
~310 m elevation -- the approximate coordinates of Nagpur, Maharashtra) so
that the seasonal cycle (pre-monsoon summer, monsoon, winter) and the
FAO-56 physics stay mutually consistent. Day-to-day persistence is added
with an AR(1) process so consecutive days are correlated (as in real
weather), and rainfall is modelled as a rain-day probability plus a Gamma-
distributed amount when it rains, with an occasional heavy-storm tail during
the monsoon months.

THIS IS SYNTHETIC DATA. It is generated using FAO-56-based relationships
calibrated to realistic Indian agricultural weather ranges. It is NOT
observed data from any real farm, field trial, or meteorological station.

Reproducible: random_state = 42 (see RNG below).
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

N_DAYS = 2000                     # ~5.5 years of daily data
START_DATE = "2019-01-01"

LATITUDE_DEG = 21.15               # ~ Nagpur, Maharashtra, India
ELEVATION_M = 310.0
ALBEDO = 0.23                      # FAO-56 reference crop albedo
SIGMA = 4.903e-9                   # Stefan-Boltzmann const, MJ K^-4 m^-2 day^-1
IRRIGATION_EFFICIENCY = 0.75

# Maize (FAO-56 Table 11 / Table 12) crop calendar, single representative cycle
STAGE_LENGTHS = {"initial": 20, "development": 35, "mid": 40, "late": 30}
CYCLE_LENGTH = sum(STAGE_LENGTHS.values())  # 125 days
KC_INI, KC_MID, KC_END = 0.30, 1.20, 0.60

# --- Monthly climatology (approximate, representative central-Indian station) ---
# T: mean/std daily mean temperature (C); DIURNAL: Tmax-Tmin range (C)
# RH: mean/std relative humidity (%); WIND: mean/std wind speed (m/s)
# KT: mean/std clearness index (Rs/Ra), used to derive solar radiation
# RAIN_P: probability a day is a rain day; RAIN_MEAN: mean rainfall (mm) when it rains
MONTHLY = {
    1:  dict(T=17, Tsd=2.5, DIURNAL=13, RH=55, RHsd=10, WIND=1.4, WINDsd=0.4, KT=0.62, KTsd=0.06, RAIN_P=0.04, RAIN_MEAN=6),
    2:  dict(T=20, Tsd=2.5, DIURNAL=13, RH=48, RHsd=10, WIND=1.7, WINDsd=0.4, KT=0.64, KTsd=0.06, RAIN_P=0.05, RAIN_MEAN=6),
    3:  dict(T=26, Tsd=2.5, DIURNAL=13, RH=38, RHsd=9,  WIND=2.1, WINDsd=0.5, KT=0.66, KTsd=0.05, RAIN_P=0.05, RAIN_MEAN=7),
    4:  dict(T=31, Tsd=2.2, DIURNAL=13, RH=32, RHsd=8,  WIND=2.4, WINDsd=0.5, KT=0.65, KTsd=0.05, RAIN_P=0.06, RAIN_MEAN=9),
    5:  dict(T=35, Tsd=2.2, DIURNAL=13, RH=30, RHsd=7,  WIND=2.9, WINDsd=0.6, KT=0.66, KTsd=0.05, RAIN_P=0.10, RAIN_MEAN=11),
    6:  dict(T=33, Tsd=2.5, DIURNAL=9,  RH=55, RHsd=12, WIND=3.9, WINDsd=0.9, KT=0.55, KTsd=0.08, RAIN_P=0.38, RAIN_MEAN=20),
    7:  dict(T=29, Tsd=2.0, DIURNAL=6,  RH=80, RHsd=7,  WIND=4.1, WINDsd=0.9, KT=0.40, KTsd=0.08, RAIN_P=0.60, RAIN_MEAN=26),
    8:  dict(T=28, Tsd=2.0, DIURNAL=6,  RH=83, RHsd=7,  WIND=3.7, WINDsd=0.9, KT=0.38, KTsd=0.07, RAIN_P=0.58, RAIN_MEAN=23),
    9:  dict(T=28, Tsd=2.0, DIURNAL=8,  RH=75, RHsd=8,  WIND=2.9, WINDsd=0.7, KT=0.48, KTsd=0.07, RAIN_P=0.38, RAIN_MEAN=18),
    10: dict(T=27, Tsd=2.5, DIURNAL=11, RH=58, RHsd=10, WIND=1.9, WINDsd=0.5, KT=0.60, KTsd=0.06, RAIN_P=0.12, RAIN_MEAN=12),
    11: dict(T=22, Tsd=2.5, DIURNAL=12, RH=55, RHsd=10, WIND=1.4, WINDsd=0.4, KT=0.62, KTsd=0.06, RAIN_P=0.03, RAIN_MEAN=6),
    12: dict(T=18, Tsd=2.5, DIURNAL=13, RH=58, RHsd=10, WIND=1.3, WINDsd=0.4, KT=0.61, KTsd=0.06, RAIN_P=0.03, RAIN_MEAN=5),
}


def ar1_series(n, phi, sd, seed_rng):
    """Zero-mean AR(1) noise series with lag-1 correlation `phi` and marginal std `sd`."""
    innov_sd = sd * np.sqrt(max(1e-6, 1 - phi ** 2))
    noise = seed_rng.normal(0, innov_sd, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + noise[i]
    return x


def extraterrestrial_radiation(doy, lat_deg):
    """FAO-56 Eq. 21-25: extraterrestrial radiation Ra (MJ/m2/day)."""
    phi = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * doy)
    decl = 0.409 * np.sin(2 * np.pi / 365 * doy - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(decl), -1, 1))
    Gsc = 0.0820
    Ra = (24 * 60 / np.pi) * Gsc * dr * (
        ws * np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.sin(ws)
    )
    return Ra


def saturation_vapor_pressure(T):
    """FAO-56 Eq. 11 (kPa), T in C."""
    return 0.6108 * np.exp(17.27 * T / (T + 237.3))


def slope_svp_curve(T):
    """FAO-56 Eq. 13 (kPa/C)."""
    return 4098 * saturation_vapor_pressure(T) / (T + 237.3) ** 2


def psychrometric_constant(elevation_m):
    """FAO-56 Eq. 7-8 (kPa/C)."""
    P = 101.3 * ((293 - 0.0065 * elevation_m) / 293) ** 5.26
    return 0.000665 * P


def fao56_penman_monteith(Tmean, Tmax, Tmin, RHmean, u2, Rs, Ra, elevation_m):
    """FAO-56 reference evapotranspiration ET0 (mm/day), Eq. 6."""
    Rso = (0.75 + 2e-5 * elevation_m) * Ra
    ratio = np.clip(Rs / np.maximum(Rso, 1e-6), 0.3, 1.0)

    es = (saturation_vapor_pressure(Tmax) + saturation_vapor_pressure(Tmin)) / 2
    ea = saturation_vapor_pressure(Tmean) * RHmean / 100  # Eq. 19 (mean-RH form)

    Rns = (1 - ALBEDO) * Rs
    TmaxK, TminK = Tmax + 273.16, Tmin + 273.16
    Rnl = SIGMA * ((TmaxK ** 4 + TminK ** 4) / 2) * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0))) * (
        1.35 * ratio - 0.35
    )
    Rn = Rns - Rnl

    delta = slope_svp_curve(Tmean)
    gamma = psychrometric_constant(elevation_m)
    G = 0.0  # soil heat flux, ~0 for daily time step (FAO-56 Sec. 3.5.1)

    numerator = 0.408 * delta * (Rn - G) + gamma * (900 / (Tmean + 273)) * u2 * (es - ea)
    denominator = delta + gamma * (1 + 0.34 * u2)
    return numerator / denominator


def crop_coefficient_and_stage(day_in_cycle):
    """Return (Kc, growth_stage) for a 0-indexed day within the 125-day maize cycle."""
    d = day_in_cycle
    L_ini, L_dev, L_mid, L_late = (
        STAGE_LENGTHS["initial"], STAGE_LENGTHS["development"],
        STAGE_LENGTHS["mid"], STAGE_LENGTHS["late"],
    )
    if d < L_ini:
        return KC_INI, 1
    d -= L_ini
    if d < L_dev:
        frac = d / L_dev
        return KC_INI + frac * (KC_MID - KC_INI), 2
    d -= L_dev
    if d < L_mid:
        return KC_MID, 3
    d -= L_mid
    frac = d / L_late
    return KC_MID + frac * (KC_END - KC_MID), 4


def effective_rainfall_scs(P):
    """USDA-SCS empirical effective-rainfall formula, applied at daily step (see module docstring)."""
    Pe = np.where(P > 75, 0.8 * P - 25, 0.6 * P - 10)
    Pe = np.clip(Pe, 0, None)
    return np.minimum(Pe, P)


def generate_dataset(n_days=N_DAYS, start_date=START_DATE, seed=RANDOM_STATE):
    local_rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    months = dates.month
    doy = dates.dayofyear.values

    months_arr = np.array([MONTHLY[m] for m in months])
    T_mean_m = np.array([MONTHLY[m]["T"] for m in months])
    T_sd_m = np.array([MONTHLY[m]["Tsd"] for m in months])
    diurnal_m = np.array([MONTHLY[m]["DIURNAL"] for m in months])
    RH_mean_m = np.array([MONTHLY[m]["RH"] for m in months])
    RH_sd_m = np.array([MONTHLY[m]["RHsd"] for m in months])
    wind_mean_m = np.array([MONTHLY[m]["WIND"] for m in months])
    wind_sd_m = np.array([MONTHLY[m]["WINDsd"] for m in months])
    kt_mean_m = np.array([MONTHLY[m]["KT"] for m in months])
    kt_sd_m = np.array([MONTHLY[m]["KTsd"] for m in months])
    rain_p_m = np.array([MONTHLY[m]["RAIN_P"] for m in months])
    rain_mean_m = np.array([MONTHLY[m]["RAIN_MEAN"] for m in months])

    # --- Temperature: monthly mean + AR(1) persistent anomaly ---
    temp_anom = ar1_series(n_days, phi=0.75, sd=1.0, seed_rng=local_rng)
    temperature = np.clip(T_mean_m + temp_anom + local_rng.normal(0, 0.6, n_days), 15, 42)
    Tmax = temperature + diurnal_m / 2
    Tmin = temperature - diurnal_m / 2

    # --- Rainfall: rain-day Bernoulli + Gamma amount, heavy-storm tail in monsoon ---
    is_rain = local_rng.random(n_days) < rain_p_m
    gamma_shape = 1.6
    rain_amt = local_rng.gamma(shape=gamma_shape, scale=rain_mean_m / gamma_shape, size=n_days)
    heavy_storm = (local_rng.random(n_days) < 0.03) & (months >= 6) & (months <= 9)
    rain_amt = np.where(heavy_storm, rain_amt + local_rng.uniform(30, 45, n_days), rain_amt)
    rainfall = np.clip(np.where(is_rain, rain_amt, 0.0), 0, 80)

    # --- Clearness index -> solar radiation via extraterrestrial radiation Ra ---
    Ra = extraterrestrial_radiation(doy, LATITUDE_DEG)
    kt_noise = ar1_series(n_days, phi=0.5, sd=1.0, seed_rng=local_rng)
    kt = kt_mean_m + kt_sd_m * kt_noise
    kt = np.where(is_rain, kt - 0.12, kt)  # extra cloud attenuation on rain days
    kt = np.clip(kt, 0.20, 0.80)
    solar_radiation = np.clip(kt * Ra, 8, 30)

    # --- Relative humidity: monthly mean + AR(1) + rain-day boost + dryness on clear days ---
    rh_anom = ar1_series(n_days, phi=0.7, sd=1.0, seed_rng=local_rng)
    humidity = RH_mean_m + RH_sd_m * rh_anom - 25 * (kt - kt_mean_m)
    humidity = np.where(is_rain, humidity + 18, humidity)
    humidity = np.clip(humidity, 25, 95)

    # --- Wind speed: monthly mean + AR(1), mild boost on rain/storm days ---
    wind_anom = ar1_series(n_days, phi=0.6, sd=1.0, seed_rng=local_rng)
    wind_speed = wind_mean_m + wind_sd_m * wind_anom
    wind_speed = np.where(is_rain, wind_speed + 0.4, wind_speed)
    wind_speed = np.clip(wind_speed, 0.5, 6.0)

    # --- ET0 via FAO-56 Penman-Monteith ---
    et0 = fao56_penman_monteith(temperature, Tmax, Tmin, humidity, wind_speed, solar_radiation, Ra, ELEVATION_M)
    et0 = np.clip(et0, 1.0, 8.0)

    # --- Crop coefficient / growth stage (continuous back-to-back maize cycles) ---
    day_in_cycle = np.arange(n_days) % CYCLE_LENGTH
    kc_stage = [crop_coefficient_and_stage(d) for d in day_in_cycle]
    crop_coefficient = np.array([x[0] for x in kc_stage])
    growth_stage = np.array([x[1] for x in kc_stage])

    # --- FAO-56 crop water balance ---
    etc = et0 * crop_coefficient
    effective_rainfall = effective_rainfall_scs(rainfall)
    net_irrigation = np.clip(etc - effective_rainfall, 0, None)
    gross_irrigation = net_irrigation / IRRIGATION_EFFICIENCY

    # small measurement-like noise so the target isn't a deterministic function of X
    irrigation_demand = gross_irrigation + local_rng.normal(0, 0.18, n_days)
    irrigation_demand = np.clip(irrigation_demand, 0, None)

    previous_irrigation = np.roll(irrigation_demand, 1)
    previous_irrigation[0] = irrigation_demand.mean()

    df = pd.DataFrame({
        "date": dates,
        "rainfall_mm": rainfall.round(2),
        "temperature_C": temperature.round(2),
        "humidity_percent": humidity.round(2),
        "wind_speed_mps": wind_speed.round(2),
        "solar_radiation_MJ_m2_day": solar_radiation.round(2),
        "previous_irrigation_mm": previous_irrigation.round(3),
        "crop_coefficient": crop_coefficient.round(3),
        "growth_stage": growth_stage,
        "et0_mm": et0.round(3),
        "etc_mm": etc.round(3),
        "effective_rainfall_mm": effective_rainfall.round(3),
        "irrigation_demand_mm": irrigation_demand.round(3),
    })
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "../data/irrigation_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df.head(10).to_string(index=False))
    print("\nSummary statistics:")
    print(df.describe().T)
