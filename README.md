# Glovo Hourly Demand Forecasting

**21DM011 — Intelligent Data Development · Final Project**
Barcelona School of Economics · Data Science Methodology (DSDM)

Forecast hourly delivery-order demand for one Glovo city (Barcelona) **one week
(168 hours) ahead**, choose a champion model through an honest backtest, and turn
the forecast into a concrete operational decision — a courier-staffing roster.

---

## 1. The problem

Glovo plans each city on a **slot-based system of 24 hourly slots per day**.
Planning is refreshed **weekly**: every Sunday after 23:59 the team must hand
operations a forecast for **all 168 hours** of the coming week (Mon 00:00 →
Sun 23:00). That forecast drives courier headcount, and the cost is asymmetric:

- **Over-estimate** → idle couriers and wasted spend.
- **Under-estimate** → delivery delays and poor service.

So the modelling task is a fixed-horizon problem: predict the next **168 hourly
observations**, and then size staffing against that prediction.

The grading metric is **SMAPE**; **MASE** is the sanity check against the
seasonal-naive baseline, and **MSE** tracks the expensive peak-hour misses.

---

## 2. The data

| | |
|---|---|
| City | Barcelona (BCN), single city |
| Granularity | Hourly orders |
| Span | Feb 2021 → Jan 2022 (~50 weeks) |
| Files | [`data/train_data.csv`](data/train_data.csv), [`data/test_data_mock.csv`](data/test_data_mock.csv) |

Key data facts that shaped the modelling:

- **~32% of hours are exactly zero** — the platform is shut overnight (00:00–05:00
  always; 23:00 almost always). This envelope is *deterministic*, so the models
  force those hours to 0 in post-processing.
- A handful of **missing overnight hours** are imputed onto a gap-free hourly grid
  (time-interpolation → hour-of-week median → 0), because lags and seasonal
  differences need a regular index.

---

## 3. What the analysis found (EDA)

1. **Seasonality is the signal.** Two cycles — a daily shape (lunch peak ~13:00,
   bigger dinner peak ~21:00) and a weekly shape (Friday/weekend lift) — combine
   into a stable **168-cell weekly fingerprint** (hour × weekday). ACF ≈ 0.88 at
   lags 24 and 168 confirms it.
2. **The level grows** through 2021 (weekly totals ~11k → ~16.5k), while the
   seasonal *shape* stays stable. So forecasts must be rescaled to the **recent**
   level, not a year-ago level.
3. **Non-stationary in levels**, confirmed with paired **ADF + KPSS** tests; a
   seasonal difference at lag 168 removes the dominant structure.
4. **Peaks are spiky and SMAPE punishes noisy low-volume "ramp" hours** (06:00–08:00),
   where a tiny absolute error is a large *percentage* error. This is why smooth
   global models and variance-chasing ML can *lose* to a robust seasonal estimate.

---

## 4. Methodology

**Validation that imitates production.** A **rolling-origin (walk-forward)
backtest over the last 12 weeks**: for week *k*, train on all hours strictly
before it and forecast its 168 hours — exactly the weekly planning refresh. A
shuffled split would leak the future and overstate accuracy.

**Models compared** (deliberately different families):

| Family | Member |
|---|---|
| Naive (baseline) | Seasonal naive *t−168* |
| Linear / harmonic | Fourier (24h + 168h) + trend, ridge |
| Classical | SARIMA (2,0,1)(1,0,1)[24] + weekly Fourier |
| Gradient boosting | XGBoost, 17 engineered features |
| **Robust seasonal (champion)** | Trend-adjusted seasonal **median** profile |

---

## 5. The champion model & results

**Trend-adjusted seasonal median profile.** For each of the 168 hours-of-week,
take the **median of the last 10 realizations** (robust to one-off spikes), then
**rescale the whole week** by the ratio of the most recent week's level to the
profile's level (tracks the growth trend); finally clip ≥ 0 and force dead night
hours to 0.

It encodes the three EDA findings directly — *seasonal memory* (median profile),
*robustness* (median not mean), *level-awareness* (trend rescale) — and **has
nothing it can overfit**, which is why it generalizes where the 17-feature
booster does not.

> **Result:** across the 12-week rolling backtest the champion beats the
> seasonal-naive baseline on SMAPE (with **MASE < 1**) and also beats harmonic
> regression, SARIMA, and XGBoost. *Performance ≫ complexity.*

The champion ships with a **90% empirical prediction interval**, built
distribution-free from backtest residuals grouped by hour of day (wide at the
noisy dinner peak, tight in the calm morning).

---

## 6. From forecast to decision: courier staffing

A forecast is only useful if it changes a decision. The project builds a
**courier-staffing MILP** (OR-Tools + SCIP):

- **Decision:** integer shift-starts; a courier works a contiguous **6-hour shift**,
  so neighbouring hours are coupled (real rostering, not 168 independent ceilings).
- **Service constraint:** each courier serves a fixed capacity of orders/hour;
  unmet demand is penalised.
- **Objective:** `min  wage·(courier-hours) + penalty·(unmet orders)` — the exact
  over-vs-under-staffing trade-off from the brief.

This traces the **service-vs-cost frontier** the business actually chooses on, and
a **chance-constrained** variant sizes against the 90% interval rather than the mean.

---

## 7. Robustness & extensions

Each is implemented and *measured* on the same backtest — and reported honestly,
including where the extra machinery does not pay off:

- **Holiday / event effects** — anomaly detector + Barcelona/Catalonia holiday
  calendar + holiday-aware champion variant.
- **Interval calibration** — out-of-sample coverage of the 90% band (≈ 85–90%).
- **Chance-constrained staffing** — buffer against the band, not the point forecast.
- **Ramp-hour analysis** — a daypart decomposition pinpointing the intermittent
  morning hours as the residual-error frontier.

**Productionization (MLOps/DataOps):** scheduled weekly orchestration after the
Sunday data lands, data contracts / freshness & anomaly gates, deploy-code-not-model,
and drift monitoring on rolling SMAPE/MASE.

---

## 8. Repository structure

| Path | Contents |
|---|---|
| [`1_3_framing_data_eda.py`](1_3_framing_data_eda.py) | **Part 1** — Business framing · Data quality · EDA (§1–3) |
| [`4_6_methodology_results_champion.py`](4_6_methodology_results_champion.py) | **Part 2** — Methodology · Results · Champion model (§4–6) |
| [`7_9_submission_staffing_robustness.py`](7_9_submission_staffing_robustness.py) | **Part 3** — Submission · Staffing MILP · Robustness · Conclusion (§7–11 + Appendix) |
| [`prod_deployment_considerations.pdf`](prod_deployment_considerations.pdf) | Production/deployment write-up |
| [`data/`](data/) | `train_data.csv`, `test_data_mock.csv` |

Each of the three numbered files is **fully self-contained** — it recomputes every
figure, statistic and backtest from `data/` and imports no project modules.

---

## 9. How to run

The project is built as **[marimo](https://marimo.io)** notebooks. Each file
declares its own dependencies inline (PEP 723), so you can open it directly:

```bash
# open any part as an interactive notebook
marimo edit 1_3_framing_data_eda.py
marimo edit 4_6_methodology_results_champion.py
marimo edit 7_9_submission_staffing_robustness.py
```

**Scientific-Python stack:** `numpy`, `pandas`, `plotly`, `statsmodels`,
`scikit-learn`, `xgboost`, `ortools`.

**Compute notes:** Part 1 ~1 min (MSTL is the slow step); Part 2 ~15–25 min (four
models × 12 folds + SARIMA + a feature-engineering study); Part 3 ~2–4 min (the
MILP solves in seconds).

**Optional environment overrides:** `IDD_DATA_DIR`, `IDD_TRAIN_FILE`,
`IDD_TEST_FILE`, `IDD_OUTPUT_DIR`.

---

## 10. Conclusion

**Ship the trend-adjusted seasonal median profile.** This series is dominated by
a stable weekly fingerprint plus a slow growth trend; there is little extra
structure for ML or ARIMA to exploit, and SMAPE punishes the noise they add in
the low-volume ramp hours. The simple model is robust, level-aware, fast to
retrain, and **defensible hour-by-hour to operations** — the "number we can defend
for every hour of the week."

*Genuinely remaining work:* a calibrated quantile-regression interval, a dedicated
intermittent-demand model (Croston/SBA) for the ramp hours, richer event features
(weather, promotions, sport), and a stochastic joint forecast-and-roster.

---

## 11. Team

| Contributor | |
|---|---|
| **Elvis Casco** | `ecasco1@gmail.com` |
| **Erika Blanco** (`YazBlanco` / `eybf`) | `erika.blanco@bse.eu` |
| **María Victoria Suriel** | `mariavsurieln@gmail.com` |

*Course: 21DM011 Intelligent Data Development — BSE Data Science Methodology.*
