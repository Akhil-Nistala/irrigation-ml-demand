# Explainer: Everything Behind This Project, From First Principles

This document explains **every concept, formula, and code decision** used in this
project — assuming no prior background in either irrigation engineering or machine
learning. Read it top to bottom, or jump to a section using the table of contents.

## Table of contents

1. [What problem are we actually solving?](#1-what-problem-are-we-actually-solving)
2. [The water-balance idea, in plain terms](#2-the-water-balance-idea-in-plain-terms)
3. [What is FAO-56?](#3-what-is-fao-56)
4. [Reference evapotranspiration (ET0) — the FAO-56 Penman-Monteith equation, term by term](#4-reference-evapotranspiration-et0--the-fao-56-penman-monteith-equation-term-by-term)
5. [Net radiation (Rn) — where the energy comes from](#5-net-radiation-rn--where-the-energy-comes-from)
6. [Crop coefficient (Kc) and crop evapotranspiration (ETc)](#6-crop-coefficient-kc-and-crop-evapotranspiration-etc)
7. [Effective rainfall, and net vs. gross irrigation requirement](#7-effective-rainfall-and-net-vs-gross-irrigation-requirement)
8. [Why synthetic data, and why it isn't "just random numbers"](#8-why-synthetic-data-and-why-it-isnt-just-random-numbers)
9. [Dataset generation, code section by code section](#9-dataset-generation-code-section-by-code-section)
10. [From physics to a machine-learning problem](#10-from-physics-to-a-machine-learning-problem)
11. [What is supervised machine learning, and what is "training"?](#11-what-is-supervised-machine-learning-and-what-is-training)
12. [Why a chronological train/test split, not a random one](#12-why-a-chronological-traintest-split-not-a-random-one)
13. [Model 1: Linear Regression](#13-model-1-linear-regression)
14. [Model 2: Decision Tree Regressor](#14-model-2-decision-tree-regressor)
15. [Model 3: Random Forest Regressor](#15-model-3-random-forest-regressor)
16. [Evaluation metrics, explained one by one](#16-evaluation-metrics-explained-one-by-one)
17. [Feature importance — three complementary metrics, explained](#17-feature-importance--three-complementary-metrics-explained)
18. [Reading the actual results](#18-reading-the-actual-results)
19. [Glossary (quick lookup)](#19-glossary-quick-lookup)

---

## 1. What problem are we actually solving?

A farmer (or an irrigation-canal operator, or an automated drip-irrigation controller)
needs to decide, **every single day**: *how much water should I apply to this field
today?*

Apply too little, and the crop is water-stressed, which reduces yield. Apply too
much, and you waste a scarce resource, may leach fertilizer out of the root zone,
and may waterlog the soil.

The "correct" answer depends on:
- **How thirsty the atmosphere is today** (hot, sunny, windy, dry air pulls more
  water out of the soil/plant than cool, cloudy, still, humid air).
- **How thirsty the crop itself is at its current growth stage** (a tiny seedling
  needs far less water than a fully-grown, actively photosynthesizing plant).
- **How much water it already got for free** (rainfall).

Civil/agricultural engineers have a standard, internationally-adopted method for
turning those three things into a number: the **FAO-56 crop water balance**
(explained in detail below). This project:

1. Builds a dataset of daily weather + crop information using this method (Sections 3–9).
2. Asks: *can a machine-learning model learn to predict the final answer (irrigation
   demand) directly from the raw weather + crop inputs*, without explicitly running
   the FAO-56 equations every time? (Sections 10–17)
3. Checks whether the answer it learns actually makes engineering sense (Section 18).

---

## 2. The water-balance idea, in plain terms

Think of the root zone of soil under a crop as a **bucket** that the plant drinks
from:

- Water enters the bucket from **rainfall** and **irrigation**.
- Water leaves the bucket through **evapotranspiration** — a combination of
  *evaporation* (water turning to vapour directly from wet soil/leaf surfaces) and
  *transpiration* (water drawn up through plant roots and released as vapour
  through leaf pores, as a side effect of photosynthesis). These two are almost
  always lumped together as one term, **ET**, because in a cropped field it is hard
  to measure them separately and, from a water-budget point of view, both simply
  remove water from the bucket.

If more water leaves via ET than enters via rain, the bucket is running low, and
**irrigation must make up the difference**. That single sentence is the entire
logic of this project — everything below is about calculating "how much ET" and
"how much of the rain actually counted."

---

## 3. What is FAO-56?

**FAO-56** refers to *FAO Irrigation and Drainage Paper No. 56: "Crop
Evapotranspiration – Guidelines for Computing Crop Water Requirements"* (Allen,
Pereira, Raes & Smith, 1998), published by the **Food and Agriculture Organization
of the United Nations (FAO)**.

It is, essentially, **the international engineering standard** for estimating crop
water requirements. Before it, dozens of competing ET-estimation formulas existed
(Blaney-Criddle, Hargreaves, Thornthwaite, Penman's original 1948 formula, etc.),
each calibrated to different climates and giving inconsistent results. FAO-56 did
two things that made it the standard everyone now uses:

1. It **standardized the reference surface** — instead of measuring "evaporation
   from whatever crop happens to be there," it defines evapotranspiration from a
   precisely specified **hypothetical reference crop**: a well-watered, actively
   growing grass surface of uniform height (0.12 m), fixed surface resistance
   (70 s/m), and albedo 0.23. This reference quantity is called **ET0** (Section 4).
2. It provides **standardized crop coefficients (Kc)** — published, tabulated
   multipliers (Section 6) that convert this one universal reference number into
   an estimate for *any specific crop, at any specific growth stage*, without
   needing a new formula for every crop.

So the FAO-56 workflow is always the same two steps:

```
ET0  (reference evapotranspiration — depends only on WEATHER)
  ×
Kc   (crop coefficient — depends only on CROP TYPE and GROWTH STAGE)
  =
ETc  (crop evapotranspiration — the actual water use of THIS crop, right now)
```

This is exactly the structure this project implements in code
(`src/generate_dataset.py`, functions `fao56_penman_monteith()` and
`crop_coefficient_and_stage()`).

---

## 4. Reference evapotranspiration (ET0) — the FAO-56 Penman-Monteith equation, term by term

This is the single most important equation in the whole project. It is implemented
in `fao56_penman_monteith()` in `src/generate_dataset.py`. The equation (FAO-56,
Eq. 6):

$$ET_0 = \frac{0.408\,\Delta\,(R_n - G) + \gamma\,\dfrac{900}{T+273}\,u_2\,(e_s - e_a)}{\Delta + \gamma\,(1 + 0.34\,u_2)}$$

Read it as **"energy term" + "aerodynamic term," divided by a normalizing
denominator.** Let's unpack every single symbol:

| Symbol | Name | Meaning | Where in code |
|---|---|---|---|
| $R_n$ | Net radiation | The net energy (in MJ/m²/day) actually absorbed at the crop surface after accounting for reflected sunlight and radiated heat loss. This is the "fuel" that powers evaporation. | `Rn = Rns - Rnl`, computed inside the same function (see Section 5) |
| $G$ | Soil heat flux density | Energy that goes into (or comes out of) heating the soil itself, rather than evaporating water. At a **daily** timestep this is conventionally taken as ≈0, since day-to-day soil temperature change averages out (FAO-56 §3.5.1). | `G = 0.0` |
| $T$ | Mean daily air temperature (°C) | Used in two places: it appears directly in the aerodynamic term, and it's needed to compute $\Delta$ and vapour pressures. | `temperature_C` column |
| $u_2$ | Wind speed at 2 m height (m/s) | Wind physically carries water-vapour-laden air away from the leaf surface, replacing it with drier air, so more wind → faster ET (up to a point). | `wind_speed_mps` column |
| $e_s$ | Saturation vapour pressure (kPa) | The maximum amount of water vapour the air *could* hold at the current temperature, if it were 100% saturated. Air holds exponentially more moisture as it gets warmer — this is why $e_s$ uses an exponential (Tetens-type) formula. | `saturation_vapor_pressure(T)` |
| $e_a$ | Actual vapour pressure (kPa) | How much water vapour the air *actually* holds right now. | `es(Tmean) * RH/100` (Eq. 19, the simplified mean-RH form used when only a daily-mean humidity is available) |
| $e_s - e_a$ | Vapour pressure deficit (kPa) | The "dryness" of the air — how much more moisture it *could* absorb. This is the direct driving force of evaporation: air with a big deficit pulls water out of the crop fast; air already near saturation (e.g. during monsoon) pulls very little. | computed inline |
| $\Delta$ | Slope of the saturation-vapour-pressure curve (kPa/°C) | How steeply $e_s$ rises with temperature at the current temperature — a purely mathematical derivative of the $e_s(T)$ curve, needed to properly weight the energy term. | `slope_svp_curve(T)` |
| $\gamma$ | Psychrometric constant (kPa/°C) | Relates air temperature to atmospheric pressure's effect on evaporation; it depends on elevation (air pressure drops with altitude, which changes the constant slightly). | `psychrometric_constant(elevation_m)` |
| $0.408$, $900$, $0.34$ | Unit-conversion constants | These are not arbitrary — they come from converting energy units (MJ/m²/day) into equivalent water-depth units (mm/day), and from the standard aerodynamic resistance assumptions FAO-56 bakes into the "simplified" form of the equation so you don't need to compute aerodynamic/surface resistances separately. | built into the formula as written |

**Intuition for the two halves of the numerator:**
- The **first half**, $0.408\,\Delta\,(R_n-G)$, is the **"radiation-driven" or
  "energy" term** — how much evaporation the available sunlight/net-radiation
  energy alone could sustain.
- The **second half**, $\gamma\,\frac{900}{T+273}\,u_2\,(e_s-e_a)$, is the
  **"aerodynamic" or "wind/dryness" term** — how much *extra* evaporation happens
  because the air is dry and windy enough to keep whisking moisture away.

ET0 is therefore high when **both** halves are large: hot, sunny, windy, and dry —
exactly the "high demand" scenario the project's synthetic data is built to
reproduce (Section 8).

---

## 5. Net radiation (Rn) — where the energy comes from

$R_n$ itself isn't measured directly in this project (no pyranometer/net radiometer
data exists) — it's *derived* from solar radiation, following FAO-56 Chapter 3.
This derivation is the most elaborate part of the code, so here it is broken down
fully (function `fao56_penman_monteith`, using helper `extraterrestrial_radiation`):

### 5.1 Extraterrestrial radiation ($R_a$)

This is the solar radiation that would arrive at the *top of the atmosphere* above
a given point on Earth, on a given day of the year — a purely astronomical quantity
depending only on **latitude** and **day-of-year** (nothing about weather at all).
It's the theoretical ceiling: whatever actually reaches the ground surface ($R_s$,
below) can never exceed a fraction of this. Computed as (FAO-56 Eqs. 21–25, function
`extraterrestrial_radiation`):

- $d_r$ — the *inverse relative distance* between Earth and Sun (the Earth's orbit
  is a slight ellipse, so it's marginally closer to the Sun on some days of the
  year than others, changing the radiation received by a percent or two).
- $\delta$ — the *solar declination angle*: the angle of the Sun above the equator,
  which is what actually produces seasons (in June the Northern Hemisphere tilts
  toward the Sun, giving a positive declination and longer, more direct sunlight
  days; in December it's negative).
- $\omega_s$ — the *sunset hour angle*: derived from latitude and declination, this
  determines day length.
- These combine into $R_a$ via a standard astronomical formula.

In the code, `LATITUDE_DEG = 21.15` (chosen to represent Nagpur, Maharashtra — a
real central-Indian location used **only** to anchor the seasonal cycle to
something geographically sensible, not because real Nagpur weather records were
used).

### 5.2 Clear-sky radiation ($R_{so}$)

$$R_{so} = (0.75 + 2\times10^{-5} z)\, R_a$$

This estimates how much of $R_a$ would actually reach the ground **on a perfectly
cloudless day** at elevation $z$ (in metres) — the atmosphere itself absorbs and
scatters some sunlight even with zero clouds, and thinner air at higher elevations
lets slightly more through, hence the small positive elevation term.

### 5.3 Actual solar radiation ($R_s$)

On a real (possibly cloudy) day, $R_s < R_{so}$. In this project, $R_s$ is
generated using a **clearness index** $K_t = R_s / R_a$ (see Section 9.3) — sampled
per day from a season-dependent distribution and multiplied by that day's $R_a$.
This is the value stored in `solar_radiation_MJ_m2_day`.

### 5.4 Net shortwave radiation ($R_{ns}$)

$$R_{ns} = (1-\alpha)\,R_s, \qquad \alpha = 0.23$$

$\alpha$ (albedo) is the fraction of incoming sunlight the reference grass surface
reflects straight back to the sky without absorbing (0.23 is the standard FAO-56
reference-crop value — real grass typically reflects about 23% of incoming solar
radiation). The rest, $R_{ns}$, is absorbed as usable energy.

### 5.5 Net longwave radiation ($R_{nl}$)

$$R_{nl} = \sigma \left[\frac{T_{max,K}^4 + T_{min,K}^4}{2}\right]\,(0.34 - 0.14\sqrt{e_a})\,(1.35\,R_s/R_{so} - 0.35)$$

Every object above absolute zero radiates heat outward as infrared ("longwave")
radiation — this is the Stefan-Boltzmann law ($\sigma = 4.903\times10^{-9}$
MJ K⁻⁴ m⁻² day⁻¹ is the Stefan-Boltzmann constant in these units). The surface
loses energy this way constantly, day and night. Two corrections are applied to
the raw blackbody radiation:
- **Humidity correction** $(0.34 - 0.14\sqrt{e_a})$: humid air itself radiates heat
  back down toward the surface (the "greenhouse" effect of water vapour), so moist
  air reduces the *net* longwave loss.
- **Cloudiness correction** $(1.35\,R_s/R_{so} - 0.35)$: clouds trap outgoing
  longwave radiation and radiate much of it back down, so a cloudy day (low
  $R_s/R_{so}$ ratio) loses much less net longwave energy than a clear one.

### 5.6 Net radiation

$$R_n = R_{ns} - R_{nl}$$

i.e., (energy absorbed from sunlight) minus (energy lost as outgoing heat
radiation) = the net energy actually available to power evapotranspiration.

---

## 6. Crop coefficient (Kc) and crop evapotranspiration (ETc)

$ET_0$ (Sections 4–5) tells us how thirsty the *atmosphere* is, completely
independent of what's actually growing in the field. To get the water use of a
**specific crop**, FAO-56 multiplies by a **crop coefficient**:

$$ET_c = ET_0 \times K_c$$

$K_c$ is not a fixed number for a crop — it changes dramatically as the crop grows,
because a tiny seedling covers almost none of the ground (most ET0-driving energy
just evaporates bare soil, weakly) versus a mature, full-canopy crop (which
transpires vigorously through a huge total leaf area). FAO-56 (Table 11) divides
the growing season into **four stages**, and (Table 12) publishes tabulated $K_c$
values per crop per stage. This project models **maize** (Zea mays):

| Stage | Code | Typical duration | $K_c$ |
|---|---|---|---|
| **Initial** | 1 | 20 days | 0.30 (flat — mostly bare/wet soil evaporation, seedling barely transpires) |
| **Development** | 2 | 35 days | rises linearly 0.30 → 1.20 (canopy is closing in, transpiration ramps up) |
| **Mid-season** | 3 | 40 days | 1.20 (flat — full canopy cover, peak water use, *higher than the reference grass* because a mature maize canopy is aerodynamically rougher and transpires more per unit area) |
| **Late-season** | 4 | 30 days | falls linearly 1.20 → 0.60 (crop is senescing/drying down toward harvest, stomata closing, less transpiration) |

This is exactly what `crop_coefficient_and_stage()` implements: it tracks which day
of a 125-day cycle (20+35+40+30) the simulation is on, and returns the
correspondingly interpolated $K_c$ and an integer growth-stage code (1–4), which
become the `crop_coefficient` and `growth_stage` columns/features.

---

## 7. Effective rainfall, and net vs. gross irrigation requirement

### 7.1 Effective rainfall ($P_e$)

Not every millimetre of rain that falls ends up available to the crop's roots.
Some is intercepted by leaves and evaporates without ever reaching the soil; some
runs off the surface during intense storms without infiltrating; some percolates
below the root zone. The fraction that *does* end up usable is called **effective
rainfall**.

This project uses the classic **USDA Soil Conservation Service (SCS) empirical
formula** (function `effective_rainfall_scs`):

$$P_e = 0.6P - 10 \quad (P \le 75\text{ mm}), \qquad P_e = 0.8P - 25 \quad (P > 75\text{ mm}), \qquad P_e = \max(0, P_e)$$

Notice the behaviour this produces: for a small rainfall (say $P=10$mm),
$P_e = 0.6(10)-10 = -4 \rightarrow$ clipped to **0** — a light shower is assumed to
be lost almost entirely to evaporation/interception before it does the crop any
good. For a large rainfall (say $P=50$mm), $P_e = 0.6(50)-10 = 20$mm — only 40% of
it counted as effective (the rest assumed lost to runoff/percolation). This
"diminishing, then partial credit" shape is exactly the real-world behaviour
engineers rely on this formula for.

*(Honest caveat, stated clearly in the code and report: this formula was
originally developed for **monthly or ten-day period** totals, not single days. It
is applied here at a daily step as a deliberate, documented simplification — chosen
because its qualitative shape is still correct at any timescale, even though the
exact numbers wouldn't survive a rigorous monthly-vs-daily statistical comparison.)*

### 7.2 Net irrigation requirement

$$\text{Net irrigation requirement} = \max(0,\ ET_c - P_e)$$

This is simply: *however much water the crop needs ($ET_c$), minus however much it
already got for free from effective rainfall ($P_e$)*. If rainfall alone already
covers or exceeds the crop's need (common during monsoon), the net requirement is
zero — no irrigation needed that day.

### 7.3 Gross irrigation requirement (irrigation efficiency)

No real irrigation system delivers 100% of the water it draws from a canal/well
directly to the root zone — some evaporates in transit, some percolates past the
root zone at the field edges, some is lost to poor field leveling, etc. **Irrigation
efficiency** ($E_i$) captures this:

$$\text{Gross irrigation requirement} = \frac{\text{Net irrigation requirement}}{E_i}$$

This project uses $E_i = 0.75$ (75%), a realistic value for a reasonably
well-managed surface or sprinkler system (the realistic engineering range is
roughly 70–80%). This is the **final target variable**, `irrigation_demand_mm` — it
represents how much water actually needs to be *withdrawn* (from a canal, well, or
tank) to deliver the crop's net requirement, accounting for system losses.

---

## 8. Why synthetic data, and why it isn't "just random numbers"

No real farm-level dataset with all eight required daily variables (rainfall,
temperature, humidity, wind, solar radiation, previous demand, Kc, growth stage)
was available for this term project. Two options existed: fabricate arbitrary
random numbers (bad — the ML models would "succeed" at learning meaningless noise,
and nothing about the results would be trustworthy or explainable), or generate
data that **obeys the same physical laws real weather and real crop water use
obey**, so that the numbers are internally consistent and the resulting ML problem
is a faithful stand-in for the real one.

This project does the latter. Concretely, "physically meaningful" means:

- Weather variables aren't drawn independently each day from a fixed distribution —
  they follow a **seasonal cycle** (Section 9.1) and have **day-to-day persistence**
  (Section 9.2), exactly like real weather (a heatwave lasts several days, not one
  isolated day).
- Solar radiation is **derived from astronomy** (Section 5.1–5.3), not sampled out
  of thin air, so it is automatically consistent with the time of year and cannot
  exceed its physical ceiling.
- The **target variable is computed by literally running the FAO-56 equations**
  (Sections 4–7) on the generated weather — it is not an independently-sampled
  number that merely *correlates* with the inputs.
- The only "unphysical" convenience taken is a small final Gaussian noise term
  added to the computed demand (Section 10), so that the ML task is a genuine
  (imperfectly predictable) regression problem rather than an exact, noiseless
  deterministic function the model could memorize perfectly.

---

## 9. Dataset generation, code section by code section

All of this lives in `src/generate_dataset.py`. Here is what each part does and, importantly, *why*.

### 9.1 Monthly climatology table (`MONTHLY` dict)

A dictionary keyed by month number (1–12), each holding mean/standard-deviation
parameters for temperature, humidity, wind, a solar "clearness index," and
rain-day probability/mean-rainfall-when-wet. These were **hand-set to
approximate a real central-Indian, subtropical-monsoon climate** (values loosely
representative of a location like Nagpur), capturing three broad seasons:

- **Pre-monsoon summer (March–May):** hottest, driest, highest solar clearness —
  this is where the highest irrigation demand should occur (see Section 19).
- **Monsoon (June–October):** much higher rain probability and rainfall amount,
  higher humidity, lower solar clearness (cloud cover) — irrigation demand should
  drop sharply here.
- **Winter (November–February):** cooler, drier, moderate solar radiation —
  moderate irrigation demand.

These aren't arbitrary: they were deliberately tuned (see the commit history / the
"tuning" step described in the project report) so the *final computed irrigation
demand* would fall in the physically expected 0–8 mm/day range for the *bulk* of
days, with only genuine extreme-weather days exceeding it — exactly as real-world
agronomic guidance describes.

### 9.2 AR(1) autoregressive noise (`ar1_series`)

**What is an AR(1) process?** "AR(1)" stands for "autoregressive, order 1." It's
the simplest possible model of a quantity that is *correlated with its own recent
past*:

$$x_t = \phi\, x_{t-1} + \varepsilon_t$$

Today's value ($x_t$) is some fraction $\phi$ (the "persistence" or
"autocorrelation" coefficient, between 0 and 1) of yesterday's value, plus a fresh
random shock $\varepsilon_t$. If $\phi=0$, every day is completely independent
random noise (unrealistic — real weather doesn't "forget" yesterday instantly). If
$\phi$ is close to 1, today's value is almost identical to yesterday's, changing
only slowly (very "sticky" weather). This project uses values like $\phi=0.75$ for
temperature, $\phi=0.7$ for humidity, etc. — chosen to give a few days of
realistic persistence (a hot spell "remembers" it was hot yesterday and is likely
to still be somewhat hot today) without becoming so sticky that the weather never
changes.

This directly produces the **month-to-month persistence** you'd see in a real
temperature or humidity time series — and the same persistence shows up directly
in the final target variable, which is exactly why `previous_irrigation_mm`
turns out to be such a strong predictor (Section 17).

### 9.3 Solar radiation via the clearness index

Rather than sampling solar radiation independently, the code samples a
**clearness index** $K_t = R_s/R_a$ per day (with its own AR(1) persistence, and
reduced on rain days to represent cloud cover), then multiplies by that day's
astronomically-computed $R_a$ (Section 5.1). This guarantees solar radiation is
always physically consistent with the calendar date and never exceeds a sensible
physical ceiling — a much stronger physical anchor than sampling it independently
would give.

### 9.4 Rainfall (Bernoulli + Gamma distribution)

Real daily rainfall isn't a smooth bell-curve — most days have **exactly zero**
rain, and the days that do have rain show a **right-skewed** distribution (many
light showers, a few heavy downpours). This is modeled with two steps:

1. **Is it a rain day at all?** A weighted coin-flip (Bernoulli trial) using
   `rain_p_m`, the month-specific rain probability (e.g. ~5% in December, ~55–60%
   in July/August).
2. **If yes, how much rain?** Drawn from a **Gamma distribution** — a standard
   statistical distribution for modeling positive, right-skewed quantities (it
   naturally produces "many small values, a long tail of large ones," which is
   exactly rainfall's real-world shape, unlike a symmetric Normal/Gaussian
   distribution which could even produce nonsensical negative rainfall).

An additional small probability of an extreme "heavy storm" event (adding 30–45mm
on top) is included during monsoon months, capturing the occasional severe
downpour real monsoon seasons produce — capped overall at 80mm/day per the
project's realistic-range specification.

### 9.5 Temperature, humidity, wind

Each is generated as: **monthly mean** (from the climatology table) **+ AR(1)
persistent anomaly** (Section 9.2) **+ small daily noise**, then clipped to the
physically realistic range specified for the project (e.g. temperature 15–42°C).
Humidity additionally gets a boost on rain days (rain is usually accompanied by
higher humidity) and a reduction on especially clear/dry days (tied to the
clearness index) — capturing the real negative correlation between "how sunny/dry
a day is" and "how humid it is."

### 9.6 Putting it together: ET0 → ETc → effective rainfall → demand

Once daily temperature, humidity, wind, and solar radiation exist, the code calls
`fao56_penman_monteith(...)` (Sections 4–5) to get `et0_mm`, multiplies by the
day's crop coefficient (Section 6) to get `etc_mm`, computes `effective_rainfall_mm`
(Section 7.1), and finally computes net and gross irrigation requirement
(Sections 7.2–7.3) — stored as `irrigation_demand_mm`.

### 9.7 Final noise and the lag feature

A small Gaussian noise term ($\sigma=0.18$mm) is added to the final demand (then
clipped at zero, since negative water demand is meaningless) — this is what
prevents the ML models from *exactly* reverse-engineering the deterministic
formula and instead forces them to learn an approximate, statistical relationship,
just as they would have to on real, imperfectly-measured data.

`previous_irrigation_mm` is simply **yesterday's** `irrigation_demand_mm` value,
shifted forward by one day (`np.roll(..., 1)`), with the very first day backfilled
with the overall mean (since there is no "day zero" to look back to). This gives
the model an "antecedent condition" feature, similar to how a real irrigation
scheduler might use yesterday's known water use as one input to today's estimate.

---

## 10. From physics to a machine-learning problem

At this point we have a dataset where, in principle, `irrigation_demand_mm` could
be recomputed **exactly** from the other columns by re-running the FAO-56 formulas
— except that:
- The ML models are **not given** the intermediate physical columns (`et0_mm`,
  `etc_mm`, `effective_rainfall_mm`) as inputs — only the *primary* weather/crop
  variables a real user would actually have on hand (rainfall, temperature,
  humidity, wind, solar radiation, previous demand, crop coefficient, growth
  stage).
- A small noise term (Section 9.7) was added to the target, so it isn't a perfectly
  noiseless function of the inputs anyway.

This turns "recompute a known formula" into "**learn an unknown function from
examples**" — which is exactly what supervised machine learning does (Section 11).

---

## 11. What is supervised machine learning, and what is "training"?

**Supervised learning** means: you have a set of examples, each with some **input
features** ($X$ — here, the 8 weather/crop columns) and a known correct **output
answer** ($y$ — here, `irrigation_demand_mm`). You give an algorithm many such
$(X, y)$ pairs, and it searches for a mathematical function $f$ such that
$f(X) \approx y$ as closely as possible across all the examples. This process of
searching for the best $f$ is called **training** or **fitting** the model
(`model.fit(X_train, y_train)` in the code).

Once trained, the model can be given a **brand-new** $X$ it has never seen before
(the test set) and produce a prediction $\hat{y} = f(X)$. How close $\hat{y}$ comes
to the true (but, in a real deployment, initially unknown) $y$ is what the
**evaluation metrics** (Section 16) measure.

Different algorithms search for $f$ in different ways and with different
assumptions about what shape $f$ is allowed to take — that's exactly what
distinguishes Linear Regression from a Decision Tree from a Random Forest
(Sections 13–15).

---

## 12. Why a chronological train/test split, not a random one

Normally in ML, you'd randomly shuffle your examples and put, say, 80% in a
training set and 20% in a test set. **This project deliberately does NOT do
that.** Instead, the **first 1,600 days** (chronologically) are the training set,
and the **last 400 days** are the test set (function `chronological_split`).

**Why this matters here specifically:** the weather (and therefore the target) is
strongly **autocorrelated day-to-day** (Section 9.2 — an AR(1) process). If you
randomly shuffled the data, a training-set day might sit literally one day before
or after a test-set day, meaning the model could partly "cheat" by exploiting
near-duplicate neighbouring conditions rather than genuinely learning the
underlying weather→demand relationship. A chronological split ensures the test
set is **entirely in the future relative to training** — exactly the situation a
real deployed model would face (you can only ever train on the past to predict a
day that hasn't happened yet). This is standard practice for **any** time-series /
forecasting-style ML problem, not specific to irrigation.

---

## 13. Model 1: Linear Regression

**The idea:** assume the output is a straight-line (technically, a flat
*hyperplane* in 8-dimensional feature space) combination of the inputs:

$$\hat{y} = b_0 + b_1 x_1 + b_2 x_2 + \dots + b_8 x_8$$

Each $b_i$ (a "coefficient" or "weight") says: *"holding everything else fixed,
increasing this one feature by one unit changes the prediction by exactly $b_i$,
always, everywhere."* Training a linear regression means finding the $b_i$ values
that minimize the total squared prediction error across all training examples (a
method called **Ordinary Least Squares**) — this has a closed-form mathematical
solution (no trial-and-error search needed), which is part of why linear
regression is so fast and simple.

**How it's used here:** `sklearn.linear_model.LinearRegression()`, fit directly on
the 8 raw features, no additional configuration needed.

**Why it's included:** as the **simplest possible baseline**. It's fast,
completely interpretable (you can literally read off "each mm of rainfall reduces
predicted demand by $b_i$ mm"), and gives a floor to compare fancier models
against. Its key limitation, visible directly in this project's results (Section
18): it assumes a *constant, additive* relationship everywhere, so it cannot
represent the genuine **nonlinear interactions** in the underlying physics (e.g.
the fact that wind speed's effect on ET0 depends on how dry the air already is,
or that Kc's effect switches abruptly between growth stages) — and it can predict
physically impossible **negative** irrigation demand, since nothing in its
mathematics stops it from doing so.

---

## 14. Model 2: Decision Tree Regressor

**The idea:** instead of one global formula, repeatedly ask **yes/no questions**
about the features to split the data into smaller and smaller groups, until each
final group ("leaf") is fairly uniform, then predict the **average target value**
within whichever leaf a new example falls into.

For example, a (simplified, illustrative) tree might learn:
```
Is crop_coefficient > 0.9?
├── No  → Is rainfall_mm > 5? → ...
└── Yes → Is temperature_C > 30?
          ├── No  → predict 4.2 mm/day
          └── Yes → predict 7.8 mm/day
```

**How the splits are chosen during training:** at each step, the algorithm tries
every possible "is this feature above/below this threshold?" question across all
features and picks whichever single split reduces the **variance** of the target
within the resulting two groups the most (i.e., makes each group as internally
consistent/homogeneous as possible). It repeats this recursively.

**How it's used here:**
`sklearn.tree.DecisionTreeRegressor(max_depth=8, random_state=42)`. `max_depth=8`
caps how many yes/no questions deep the tree can go — without this limit, a tree
can keep splitting until every single training example has its own leaf,
memorizing the training data perfectly but generalizing terribly to new data (a
problem called **overfitting**).

**Strength over linear regression:** can capture genuine nonlinear thresholds and
interactions (e.g. "high demand *only if* both mid-season Kc AND low rainfall").
**Weakness:** a single tree's predictions are a step-function (constant within each
leaf) and can be unstable — a small change in the training data can produce a
noticeably different tree.

---

## 15. Model 3: Random Forest Regressor

**The idea:** train **many** decision trees (here, 300 — `n_estimators=300`), each
on a **different random bootstrap sample** of the training data (sampling rows
*with replacement*, so each tree sees a slightly different, overlapping subset)
and, at each split, each tree is only allowed to consider a **random subset of
features** rather than all of them. Then, to make a final prediction, **average**
the predictions of all 300 trees together.

**Why this works better than one tree ("bagging"):** any single decision tree is
prone to overfitting its particular training sample's quirks (high "variance" in
statistical terms). But if you train 300 *different* trees (each seeing slightly
different data and feature subsets, so each overfits in a *different, uncorrelated*
way) and average them, the individual overfitting mistakes tend to cancel out,
while the genuine underlying signal (which all the trees pick up on, since it's
really there in the data) reinforces itself. This general technique — training
many randomized versions of a model and averaging — is called **bagging**
(bootstrap aggregating), and Random Forest is bagging specifically applied to
decision trees, with the added twist of randomizing which features each split is
allowed to consider (which further de-correlates the individual trees).

**How it's used here:**
`sklearn.ensemble.RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)`.
`n_jobs=-1` just means "use all available CPU cores to train the 300 trees in
parallel," a computational convenience with no effect on the model's predictions.

**Why it performed best in this project:** exactly because irrigation demand
genuinely depends on nonlinear interactions between weather, Kc, and rainfall (a
direct consequence of the nonlinear Penman-Monteith physics used to generate the
data) — precisely the kind of relationship tree-based models can represent and
linear regression cannot.

*(This project deliberately keeps the model set to these three — Linear
Regression, Decision Tree, Random Forest — rather than also including a
boosting-based ensemble, so that every result and every figure can be attributed
to one of exactly three well-understood, clearly-differentiated approaches.)*

---

## 16. Evaluation metrics, explained one by one

After training, each model predicts `irrigation_demand_mm` for the **400 held-out
test days** it never saw during training. We then need a way to numerically score
"how good were these predictions?" This project reports four things per model:

### 16.1 MAE — Mean Absolute Error

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|$$

For every test day, take the absolute difference between the true demand $y_i$ and
the predicted demand $\hat{y}_i$ (so being off by +2mm or −2mm both count as "2"),
then average across all 400 test days. **Units: mm/day, same as the target** — so
an MAE of 0.42 mm/day (Random Forest's result) means "on average, this model's
daily prediction is off by 0.42 mm, in either direction." It's easy to interpret
directly and is **not disproportionately punished by a few large errors** — every
day's error contributes to the average in direct proportion to its size.

### 16.2 RMSE — Root Mean Squared Error

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

Similar idea, but each error is **squared** before averaging (then square-rooted
at the end to bring the units back to mm/day). Squaring means a single day where
the model is very wrong (say, off by 5mm) contributes *much* more to RMSE (25,
after squaring) than five days each off by 1mm (1 each, summing to only 5) — even
though both scenarios have the same *total* absolute error. **RMSE is therefore
more sensitive to occasional large errors ("outliers") than MAE.** This project's
results show RMSE consistently larger than MAE for every model (e.g. Random
Forest: MAE 0.420 vs RMSE 0.639) — this is mathematically guaranteed to always be
true (RMSE ≥ MAE), and the *size of the gap* tells you something extra: a bigger
gap between RMSE and MAE means the model has a few days with noticeably larger
errors than its "typical" day, rather than uniformly-sized errors everywhere.

### 16.3 R² — Coefficient of Determination

$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

The denominator, $\sum_i(y_i-\bar{y})^2$, measures how much the *true* test-set
values vary around their own mean $\bar{y}$ — i.e., how hard the prediction
problem inherently is (if all test days had identical demand, this would be zero
and R² would be undefined). The numerator is the model's actual total squared
error. R² asks: **"what fraction of the natural variation in the target did this
model successfully explain, compared to just always guessing the average?"**

- $R^2 = 1$: perfect predictions.
- $R^2 = 0$: the model is no better than always predicting the mean value.
- $R^2 < 0$: the model is *worse* than just guessing the mean every time (a sign
  of a genuinely bad model — though this project's models are all comfortably
  positive and high: 0.92–0.97).

This project's Random Forest R² of 0.970 means the model explains 97% of the
day-to-day variation in irrigation demand across the test period — a strong
result, though remember (Section 17 / the report's Discussion) that a large chunk
of this is attributable to the `previous_irrigation_mm` persistence feature rather
than the model deeply "understanding" the weather-to-demand physics from scratch.

### 16.4 Actual mean vs. predicted mean

Simply $\bar{y}$ (average true demand across the 400 test days) versus $\bar{\hat{y}}$
(average predicted demand). This isn't a measure of day-to-day accuracy at all —
it's a **systematic bias check**: even a model with mediocre day-to-day accuracy
could still get the *overall average right by luck/cancellation*, or a model could
be *consistently* over- or under-predicting even while tracking day-to-day
fluctuations reasonably (a systematic offset). This project's results show all
three models' predicted means within 1–2% of the true mean (5.277 mm/day) —
i.e., **no model shows a systematic over- or under-prediction bias**; whatever
error exists is in day-to-day precision, captured instead by MAE/RMSE/R².

---

## 17. Feature importance — three complementary metrics, explained

"Which feature mattered most for the prediction?" sounds like a simple question,
but a **single** importance measure can be misleading — different measures can
disagree, and each has its own blind spot. This project therefore computes and
cross-checks **three independent metrics** (`report/feature_importance_summary.csv`,
figure `07_feature_importance_comparison.png`), rather than reporting just one.

### 17.1 Correlation with target

The simplest possible measure: the **Pearson correlation coefficient** between
one feature column and the target column, computed across the whole dataset,
completely ignoring every other feature. It ranges from $-1$ (perfect negative
linear relationship) to $+1$ (perfect positive linear relationship), with 0
meaning no *linear* relationship at all. It's a fast, easy-to-understand
sanity-check baseline — but it has two blind spots: it only detects *linear*
relationships (it would score a strong U-shaped relationship as ≈0, even though
that's a very real, very strong pattern), and it doesn't account for other
features already explaining the same thing (multicollinearity).

### 17.2 Random Forest MDI importance

Already explained in the original version of this section: **Mean Decrease in
Impurity**. For the trained Random Forest, `rf.feature_importances_` gives a
score for each input feature — every time *any* tree, at *any* split, uses a
particular feature, that split reduces the variance of the target within the
resulting groups by some amount; summing this reduction across every split in
every one of the 300 trees (normalized to sum to 1) gives each feature's MDI
importance. This is computed purely from **training-set** tree-building
behaviour.

**Caveat:** MDI importance can be biased toward features with many possible
distinct values (continuous features get more opportunities to find a good
split than a 4-category feature like `growth_stage`), and — because it's
computed from the training data the trees were built on — it doesn't directly
tell you how much that feature actually helps predict *new* data.

### 17.3 Permutation importance

The most direct answer to "what feature was actually important **for
prediction**": take the already-trained Random Forest and the **held-out test
set** (400 days the model has never seen). For one feature at a time:
1. Shuffle just that one column's values across the test rows (breaking any
   real relationship it has with the target, while keeping every other column
   untouched).
2. Feed this scrambled test set back through the trained model and measure how
   much the R² score (Section 16.3) *drops* compared to the unscrambled test
   set.
3. Repeat the shuffle 30 times (each with a different random shuffle) and
   report the average drop and its standard deviation — repeating gives both a
   more reliable estimate and an honest sense of how *consistent* that
   feature's importance is.

A feature whose shuffling barely changes R² wasn't really needed for
prediction; a feature whose shuffling collapses R² was critical. Because this
is measured **on the test set**, using the **already-fixed, already-trained**
model, it directly answers "how much does the model's real-world prediction
accuracy depend on this feature" — unlike MDI, which only describes how the
trees were built.

**In code:** `sklearn.inspection.permutation_importance(rf, X_test, y_test, n_repeats=30, random_state=42, scoring="r2")`.

### 17.4 Putting the three together

| Feature | Correlation with target | RF MDI importance | RF permutation importance (mean ± std) |
|---|---|---|---|
| `previous_irrigation_mm` | 0.840 | 0.713 | **0.579 ± 0.040** |
| `crop_coefficient` | 0.551 | 0.107 | 0.222 ± 0.014 |
| `rainfall_mm` | −0.404 | 0.106 | 0.170 ± 0.016 |
| `solar_radiation_MJ_m2_day` | 0.658 | 0.033 | 0.058 ± 0.008 |
| `temperature_C` | 0.360 | 0.016 | 0.030 ± 0.003 |
| `humidity_percent` | −0.563 | 0.012 | 0.026 ± 0.003 |
| `wind_speed_mps` | −0.024 | 0.012 | 0.019 ± 0.002 |
| `growth_stage` | 0.385 | 0.001 | ~0.000 |

**Reading this table:**

- `previous_irrigation_mm` tops **all three** metrics — because yesterday's
  demand, thanks to weather autocorrelation (Section 9.2), is an extremely
  effective single number for separating "this is a high-demand stretch of
  days" from "this is a low-demand stretch." This is a genuine feature of the
  data-generating process, not a modelling artefact.
- `crop_coefficient` and `rainfall_mm` are next on every metric — matching
  their *direct*, non-ET0-mediated role in the FAO-56 water balance
  ($ET_c = ET_0 \times K_c$; $\text{Net} = ET_c - P_e$).
- **MDI and permutation importance agree closely here**, and that agreement is
  itself useful evidence: since the two methods are computed in completely
  different ways (training-time split quality vs. test-time accuracy impact),
  their agreement means the ranking isn't an artefact of either method's
  particular blind spots.
- The **correlation column tells a slightly different, still-consistent
  story**: notice `solar_radiation_MJ_m2_day` actually has a *higher raw
  correlation* (0.658) than `crop_coefficient` (0.551) or `rainfall_mm`
  (−0.404), yet ranks *below* them in both RF-based importance measures. This
  is a good illustration of why correlation alone can mislead: solar radiation
  is correlated with the target partly *because* it's correlated with the
  season (and therefore indirectly with rainfall and Kc, which also vary by
  season) — once the Random Forest already has `crop_coefficient` and
  `rainfall_mm` available, solar radiation's *additional, independent*
  predictive contribution is smaller than its raw correlation alone would
  suggest.
- The permutation-importance **standard deviations** (from the 30 repeats)
  show `previous_irrigation_mm`'s effect is both large and highly consistent,
  while several weaker weather features have importances close to (or
  overlapping) zero once repeat-to-repeat variability is accounted for — their
  individual contribution, given the stronger features are already present, is
  genuinely small, not just small-but-uncertain.
- `growth_stage`'s near-zero importance on **every** metric is not a failure —
  `crop_coefficient` already encodes the same underlying information (and does
  so more precisely, as a continuous value), so the model correctly treats
  `growth_stage` as redundant.

---

## 18. Reading the actual results

(Full numbers: `report/model_comparison.csv` and `report/feature_importance_summary.csv`; full discussion: `report/term_project_report.md`, Sections 5–7.)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.717 | 1.012 | 0.924 |
| Decision Tree | 0.656 | 0.999 | 0.926 |
| Random Forest | 0.420 | 0.639 | 0.970 |

Tying every concept above together:

- **Linear Regression's** relatively high error (and its physically-impossible
  negative predictions, visible in `05_actual_vs_predicted.png`) is the direct,
  expected consequence of Section 13's limitation: it cannot represent the
  genuinely nonlinear Penman-Monteith-driven relationship between weather and
  demand.
- **Random Forest's** much lower error is the direct, expected consequence of
  Section 15: tree ensembles *can* represent nonlinear thresholds and
  interactions, and this project's target variable was *constructed* using a
  nonlinear physical equation — so of course a model family capable of
  nonlinearity does better.
- The **monthly average demand chart** (`08_monthly_avg_demand.png`) peaking in
  March–May and collapsing in the monsoon is a direct, physically-expected
  consequence of Section 9.1's climatology (hot/dry pre-monsoon → high ET0, low
  rain → high demand; monsoon → high rain, high humidity, cloudier → low ET0
  and lots of effective rainfall → low or zero demand).
- **Feature importance's** ranking (previous demand ≫ Kc ≈ rainfall > weather
  variables) directly reflects Section 17's explanation, and matches which
  variables enter the FAO-56 water balance most *directly* (Kc and rainfall,
  Section 6–7) versus only *indirectly* through ET0 (temperature, humidity, wind,
  solar radiation, Section 4) — and this ranking is **confirmed across all
  three independent importance metrics**, not just one.

Nothing in the results is a coincidence — every pattern traces back to a specific
modelling or physical choice documented in this file.

---

## 19. Glossary (quick lookup)

| Term | One-line meaning |
|---|---|
| **Evapotranspiration (ET)** | Combined water loss from soil evaporation + plant transpiration |
| **ET0** | Reference evapotranspiration — atmospheric evaporative demand, crop-independent |
| **ETc** | Crop evapotranspiration — actual water use of a specific crop |
| **Kc** | Crop coefficient — multiplier converting ET0 to ETc for a specific crop/stage |
| **FAO-56** | The international standard method/publication for computing crop water requirements |
| **Penman-Monteith equation** | The physics-based formula FAO-56 uses to compute ET0 |
| **Net radiation (Rn)** | Net energy available at the surface to power evapotranspiration |
| **Albedo** | Fraction of incoming sunlight reflected rather than absorbed |
| **Vapour pressure deficit** | How much drier the air is than fully saturated — direct driver of evaporation |
| **Effective rainfall** | The portion of rainfall actually usable by the crop, net of losses |
| **Irrigation efficiency** | Fraction of water withdrawn that actually reaches the crop's root zone |
| **Net irrigation requirement** | Water the crop needs, minus what rainfall already provided |
| **Gross irrigation requirement** | Net requirement, inflated to account for delivery-system losses |
| **AR(1) process** | A simple statistical model where today's value partly repeats yesterday's, plus new randomness |
| **Feature (X)** | An input variable given to a machine-learning model |
| **Target (y)** | The output variable the model is trying to predict |
| **Training** | The process of a model learning parameters from example data |
| **Overfitting** | A model memorizing training-data quirks instead of learning general patterns, hurting new-data performance |
| **Bagging** | Training many models on different random bootstrap samples and averaging their predictions (what Random Forest does) |
| **MAE** | Average absolute prediction error, in the target's own units |
| **RMSE** | Like MAE but penalizes large errors more heavily |
| **R²** | Fraction of the target's natural variation explained by the model |
| **Feature importance** | A score showing how much a model relied on each input feature — this project uses three independent kinds (see below) |
| **Correlation (Pearson)** | Simplest importance-adjacent measure: raw linear association between one feature and the target, ignoring all other features |
| **MDI importance** (Mean Decrease in Impurity) | Tree-based importance: how much a feature reduced target variance across all training-set tree splits |
| **Permutation importance** | Shuffle one feature in the test set and measure the resulting drop in test-set R² — the most direct "did this feature help prediction" measure |
