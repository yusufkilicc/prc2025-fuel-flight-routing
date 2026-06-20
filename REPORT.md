# Case Study — Fuel-Burn Prediction & Route Decision Support

A walkthrough of the problem, the data, the modeling choices, how they were
validated, what was learned, and — importantly — where the approach stops being
trustworthy.

---

## 1. Problem

The [PRC Data Challenge 2025](https://ansperformance.eu/study/data-challenge/dc2025/)
asks for a model that predicts the **fuel burned (kg) over time intervals of a
flight** from open trajectory data (ADS-B + ACARS). It is a regression problem on
real, noisy, gappy data.

Beyond the leaderboard, the goal here was to extend the predictor into a **decision
tool**: given two airports and an aircraft type, score candidate routes by fuel,
CO₂, time and cost.

## 2. Data

| Source | Scale | Notes |
|---|---|---|
| Flight list | 11 037 train flights | metadata: aircraft type, O/D, takeoff/landing |
| Trajectories | ~25 k rows/flight | ADS-B (dense) + ACARS (sparse, carries mach) |
| Fuel labels | 131 530 intervals | kg burned between consecutive ACARS reports |
| Airports | 8 787 | ICAO, lon/lat, elevation |

**Exploratory findings that shaped everything (`src/eda_phase0.py`):**

1. **`fuel_kg` is near-perfectly log-normal** (log-skew 0.08 vs raw 4.31) → the model
   trains on `log(fuel)`.
2. **Fuel rate is physically consistent**: climb 1.29 > cruise 0.63 > descent 0.38 kg/s;
   A320neo 0.34 → A350 1.69 → 777-300ER 2.17 kg/s.
3. **The dominant structural fact — "ADS-B darkness":** ~35 % of intervals contain *zero*
   ADS-B points (oceanic cruise, outside ground-receiver range), yet they hold **~52 %
   of all fuel mass**. A model built only on observed telemetry has *no features* for
   half the target. Of these dark intervals, ~92 % still contain an ACARS point (mach),
   and ~59 % are bracketed by ADS-B (altitude interpolable); only ~7 % are fully blind.

This single fact drove the architecture: **physics is not optional**.

## 3. Features

`src/feature_pipeline.py` builds one row per interval by **fusing three sources**
(not just binning ADS-B):

- ADS-B aggregates: altitude (mean/start/end/Δ/max), vertical-rate, ground speed,
  in-interval great-circle distance, flight phase;
- **ACARS mach** (rescues dark intervals — ended up the #5 split feature);
- **altitude interpolation** across coverage gaps from bracketing ADS-B;
- flight-level context: aircraft type, O/D great-circle distance, elapsed fraction.

## 4. Model — physics + ML hybrid

`src/openap_baseline.py` runs [OpenAP](https://github.com/junzis/openap) per interval:
`FuelFlow.enroute(mass, tas, alt, vs)` integrated over the interval, with **mass
estimated** from a load factor and propagated forward as fuel burns. Missing-altitude
mid-flight intervals are imputed to cruise altitude, lifting OpenAP coverage from
80 % → 93 %.

`src/train_hybrid.py` then trains LightGBM with the OpenAP estimate as a feature
(feature-augmented) and, alternatively, as an explicit residual target.

## 5. Validation

- **GroupKFold on `flight_id`** so intervals of the same flight never split across
  folds (prevents leakage).
- A separate **time-based holdout** is warranted (train ≈ April, rank ≈ September,
  final ≈ October 2025) and is the next rigor step.

| Model | OOF RMSE (kg) | MAE |
|---|---|---|
| Naïve median | 953 | — |
| LightGBM baseline | 258 | 82 |
| **OpenAP hybrid** | **252** | 81 |

**Segment breakdown** (where error lives):

| Segment | n | RMSE |
|---|---|---|
| ADS-B present | 85 656 | 178 |
| Dark (0 ADS-B) | 45 874 | 351 |
| Fully blind | 25 985 | 367 |

## 6. What was learned (the honest part)

- **OpenAP's gain on the *prediction* task is small (~2 %).** A gradient-boosted tree
  with rich telemetry already learns a physics-like mapping, so the physics signal
  largely *overlaps* existing features. **SHAP** reframes this positively: the OpenAP
  estimate is the **#1 driver** (mean|SHAP| 0.62, 3× the next). Physics isn't
  marginal — its information was just redundant with what the tree already had.
- **The dark-segment error (~350 kg) is largely irreducible.** We don't observe what
  the aircraft did there, and mass is uncertain. This is a *missing-observation*
  problem, not a modeling one.
- **OpenAP's real payoff is the decision layer**, where there is no telemetry at all.

## 7. Decision layer & product

`src/route_decision.py` scores **hypothetical** routes with OpenAP as the backbone:
a great-circle path is given a climb/cruise/descent profile, each segment integrated
with mass propagation, descent modeled near-idle, and a small empirical calibration
applied. Three illustrative scenarios are compared — *great-circle*,
*wind/fuel-optimal* (step-climb + tailwind), *published-ATC* (airspace dogleg) — and
ranked by a weighted multi-criteria score.

`web/web_app.py` (FastAPI + MapLibre + Tailwind) presents this on a **3D globe**:
pick origin/destination/aircraft, see the routes, KPI cards, a scenario comparison,
a SHAP "what drives fuel" panel, and a visible model-limits box.

Mass is estimated **distance-aware** (2-pass): a nominal pass estimates trip fuel,
then takeoff mass = OEW + payload + trip-fuel (capped at MTOW), so longer routes are
correctly heavier. Calibration is **per aircraft class** (wide/narrow/regional), and a
**mass-sensitivity band** (load factor 0.50–0.72) is shown as the UI uncertainty range.

Physics sanity check vs real-world block fuel (within ~15 %, labelled illustrative):

| Route | Aircraft | Engine est. | Real-world ~ |
|---|---|---|---|
| AMS→JFK | A350-900 | ~45 t | 38–42 t |
| IST→JFK | A350-900 | ~62 t | 50–55 t |
| SAL→IAD | A320neo | ~5.4 t | 5–6 t |
| KUL→AMS | 787-9 | ~70 t | 75–80 t |

## 8. Productization (post-review hardening)

A full code review (`docs/REVIEW.md`) drove a hardening pass: the submission now uses
the **hybrid** model (not the baseline); both UIs share one mass assumption; the engine
got distance-aware mass, per-class calibration and a visible uncertainty band; the map
is antimeridian-safe; the web app has error handling, request cancellation, a legend,
accessibility affordances and shareable URLs. A **pytest suite (11 tests) + CI**, a
**Dockerfile**, and **one-command setup/run** scripts were added. Optuna tuning +
chronological-holdout tooling lives in `src/tune.py`.

## 9. Limitations & next steps

- Mass is estimated, not known — still the dominant source of error (now distance-aware,
  but no real load/payload data).
- No real weather/wind grid or ATC route data; those scenarios are illustrative.
- Quantile band is under-calibrated (~72 % vs target 80 %).
- The official challenge metric was not independently confirmed.

**Remaining next steps:** a plan-derivable ("set B") ML residual on top of OpenAP for the
UI; real ERA5 wind; offline asset vendoring; mobile QA.

---

*Code: MIT (see LICENSE). Data: PRC Data Challenge 2025, distributed separately (see DATA.md).
Physics: OpenAP by Junzi Sun.*
