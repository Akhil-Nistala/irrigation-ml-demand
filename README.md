# Machine-Learning-Based Prediction of Irrigation Water Demand

A Civil Engineering (Water Resources Engineering) term project that predicts daily irrigation water demand from weather and crop data, using a synthetic dataset grounded in **FAO-56** crop-water-balance methodology and evaluated with standard scikit-learn regressors.

> **Data provenance.** All data in this project is **synthetically generated using FAO-56-based relationships calibrated to realistic Indian agricultural weather ranges**. It is *not* real observed data from any farm, field trial, or meteorological station — see [Physical basis](#physical-basis) below for exactly how it was constructed and why that makes it more than arbitrary random numbers.

---

## Project structure

```
irrigation-ml-demand/
├── README.md                          <- you are here
├── irrigation_ml_project.ipynb        <- full pipeline: dataset generation + EDA + training, executed
├── model_training.ipynb               <- modelling stage only: load dataset, train, evaluate, executed
├── src/
│   └── generate_dataset.py            <- FAO-56-based synthetic dataset generator (standalone, reproducible)
├── data/
│   └── irrigation_dataset.csv         <- generated dataset (2,000 daily rows)
├── models/
│   ├── random_forest.joblib           <- trained Random Forest model
│   ├── linear_regression.joblib       <- trained Linear Regression model
│   ├── best_model_gradient_boosting.joblib   <- best model by R² (auto-selected)
│   └── metrics.json                   <- MAE / RMSE / R² for every model
├── figures/                           <- 9 PNG plots (see below)
└── report/
    ├── term_project_report.md         <- full written term-project report (Sections 1-10)
    └── model_comparison.csv           <- raw results table (MAE, RMSE, R², actual/predicted mean)
```

## Quick start

Everything is already generated and committed to this folder (dataset, trained models, figures, executed notebooks), so you can just **read** `model_training.ipynb`, `irrigation_ml_project.ipynb`, or `report/term_project_report.md` directly. To reproduce from scratch:

```bash
cd irrigation-ml-demand

# 1. Regenerate the synthetic dataset (writes data/irrigation_dataset.csv)
cd src
python generate_dataset.py
cd ..

# 2. Re-run model training + evaluation end-to-end (all training now lives in the notebook)
jupyter nbconvert --to notebook --execute --inplace model_training.ipynb

# 3. (Optional) re-run the full combined pipeline notebook (generation + EDA + training)
jupyter nbconvert --to notebook --execute --inplace irrigation_ml_project.ipynb
```

Everything uses `random_state=42`, so re-running produces **bit-for-bit identical** results.

### Dependencies

```
numpy, pandas, matplotlib, seaborn, scikit-learn, joblib, jupyter, nbconvert
```

(Already installed in this environment. If setting up elsewhere: `pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter nbconvert nbformat`.)

---

## What problem this solves

Irrigation scheduling needs a daily answer to "how much water does the crop need today?" The classical answer (FAO-56) computes this from a chain of physical equations requiring five separate weather variables plus knowledge of the crop's growth stage. This project asks whether a machine-learning model can learn to predict that same quantity directly from the raw weather + crop inputs — offering a faster, more easily deployable alternative for on-farm or canal-scheduling decision support, while remaining interpretable and physically checkable.

## Physical basis

Rather than generating random numbers, the dataset follows the internationally standard **FAO Irrigation and Drainage Paper No. 56** methodology (Allen et al., 1998):

1. **Reference evapotranspiration (ET0)** — computed with the **full FAO-56 Penman-Monteith equation**, not a shortcut approximation. This requires:
   - Vapour pressure deficit, from temperature and relative humidity
   - The aerodynamic (wind) term
   - **Net radiation**, itself derived from generated solar radiation together with latitude/day-of-year-dependent extraterrestrial radiation (Ra) and clear-sky radiation (Rso) — the standard FAO-56 Chapter 3 radiation procedure
   - Standard psychrometric constant / saturation-vapour-pressure-curve-slope formulas (elevation-dependent)
2. **Crop evapotranspiration:** `ETc = ET0 × Kc`, using **FAO-56 Table 12** crop coefficients for **maize** and the **FAO-56 Table 11** four-stage crop calendar (initial / development / mid-season / late-season — 20/35/40/30 days, Kc 0.30 → 1.20 → 1.20 → 0.60).
3. **Effective rainfall** — the USDA Soil Conservation Service formula (`Pe = 0.6P−10` or `0.8P−25`), applied at a daily step as a documented simplification (the formula was originally derived for monthly totals, but its qualitative behaviour — small showers ≈ 0% effective, large storms partially discounted — is exactly right at any timescale).
4. **Net → gross irrigation requirement:** `Net = max(0, ETc − Pe)`; `Gross = Net / Ei`, with irrigation efficiency `Ei = 0.75`.

**Why this isn't "just random data":** weather is generated with realistic monthly climatology (calibrated to a representative central-Indian, subtropical-monsoon location, ~21°N/~310m elevation) plus **AR(1) day-to-day autocorrelation** (so heatwaves/wet spells persist for several days, as real weather does), and rainfall is a proper stochastic process (rain-day probability × Gamma-distributed amount, with an occasional monsoon heavy-storm tail) — not i.i.d. noise. The target variable is then computed by literally running the FAO-56 equations on this generated weather, plus a small final noise term so the ML task is genuinely non-trivial (not a lookup table).

Full derivation, all formulas, and inline citations are in the docstring at the top of `src/generate_dataset.py`.

## Dataset

- **2,000 daily records**, 2019-01-01 to 2024-06-22 (~5.5 years)
- **Columns:** `date`, `rainfall_mm`, `temperature_C`, `humidity_percent`, `wind_speed_mps`, `solar_radiation_MJ_m2_day`, `previous_irrigation_mm`, `crop_coefficient`, `growth_stage`, `et0_mm`, `etc_mm`, `effective_rainfall_mm`, `irrigation_demand_mm`
- The intermediate physical columns (`et0_mm`, `etc_mm`, `effective_rainfall_mm`) are kept in the CSV for transparency/auditability but are **excluded from the ML feature set** — the model only sees the same primary inputs a real farmer/scheduler would have.

**Feature set used for ML (X):** rainfall, temperature, humidity, wind speed, solar radiation, previous day's irrigation demand, crop coefficient, growth stage.
**Target (y):** `irrigation_demand_mm` (gross irrigation requirement, mm/day).

## Modelling approach

- **Split:** chronological 80/20 — train on the first 1,600 days, test on the *last* 400 days (no shuffling). This matters: it evaluates genuine future-day forecasting rather than interpolation between temporally-neighbouring (and therefore correlated) days.
- **Models:** Linear Regression, Decision Tree Regressor, Random Forest Regressor, and Gradient Boosting Regressor (bonus comparison). No deep learning, per the intended undergraduate scope.
- **Metrics:** MAE, RMSE, R², plus actual-vs-predicted mean demand (to check for systematic bias).

## Results

| Model | MAE (mm/day) | RMSE (mm/day) | R² |
|---|---|---|---|
| Linear Regression | 0.717 | 1.012 | 0.924 |
| Decision Tree | 0.656 | 0.999 | 0.926 |
| **Random Forest** | **0.420** | **0.639** | **0.970** |
| Gradient Boosting | 0.410 | 0.616 | 0.972 |

Random Forest / Gradient Boosting clearly beat the linear baseline, as expected — irrigation demand depends on nonlinear interactions between weather, crop coefficient, and rainfall (the Penman-Monteith equation itself is nonlinear). Linear Regression's actual-vs-predicted plot shows it occasionally predicts **physically impossible negative demand**, a direct illustration of its limitation. Full discussion, feature-importance interpretation, and a physical-reasonableness check (seasonal pattern, bias check) are in `report/term_project_report.md`, Sections 5–7.

## Figures (in `figures/`)

| File | Contents |
|---|---|
| `01_temperature_vs_demand.png` | Temperature vs irrigation demand (coloured by growth stage) |
| `02_rainfall_vs_demand.png` | Rainfall vs irrigation demand |
| `03_solar_radiation_vs_demand.png` | Solar radiation vs irrigation demand |
| `04_crop_coefficient_vs_demand.png` | Crop coefficient (Kc) vs irrigation demand |
| `05_actual_vs_predicted.png` | Actual vs predicted demand — Linear Regression & Random Forest, side by side |
| `06_model_comparison.png` | Bar charts of MAE / RMSE / R² across all 4 models |
| `07_timeseries_demand.png` | Time series of actual vs Random-Forest-predicted demand over the test period |
| `08_feature_importance_rf.png` | Random Forest feature importance ranking |
| `09_monthly_avg_demand.png` | Monthly average irrigation demand (seasonal pattern check) |

## Key finding worth knowing before you present this

`previous_irrigation_mm` (yesterday's demand) dominates Random Forest feature importance (~0.71 of total). This is *physically expected* — weather is autocorrelated day-to-day, so yesterday's demand is already a strong proxy for today's (a classic **persistence effect** in hydro-meteorological forecasting) — but it does mean part of the reported accuracy reflects this persistence rather than the model learning the full weather→ET0→demand chain from scratch. This is called out explicitly in the report's Discussion and Limitations sections (Sections 7–8) — worth mentioning proactively in a viva/presentation rather than waiting to be asked.

## Reproducibility

Everything uses `random_state=42` (or an equivalent seeded `numpy.random.default_rng(42)`). Re-running `generate_dataset.py` followed by re-executing `model_training.ipynb` (or `irrigation_ml_project.ipynb`) reproduces identical numbers every time.

## References

1. Allen, R.G., Pereira, L.S., Raes, D., Smith, M. (1998). *Crop Evapotranspiration – Guidelines for Computing Crop Water Requirements.* FAO Irrigation and Drainage Paper No. 56.
2. Doorenbos, J., Pruitt, W.O. (1977). *Guidelines for Predicting Crop Water Requirements.* FAO Irrigation and Drainage Paper No. 24/25.
3. USDA Soil Conservation Service (1970). *Irrigation Water Requirements.* Technical Release No. 21.
4. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.

Full reference list with context: `report/term_project_report.md`, Section 10.
