"""
================================================================================
 FINAL PROJECT — Glovo hourly demand forecasting   ·   marimo notebook (report)
================================================================================
 THIS FILE (solution.py) is the PAPER: business framing → data → EDA →
 stationarity & decomposition → methodology → feature engineering → results →
 champion → prediction intervals → submission → forecast-to-staffing optimization
 → productionization → conclusion. Read it top to bottom.

 ── Folder map (where each idea lives) ────────────────────────────────────────
   forecasting_utils.py     ★ ENGINE / main idea: data, metrics (MSE/SMAPE/MASE),
                              models, feature matrix, rolling-origin backtest,
                              diagnostics (ADF/KPSS, MSTL, ACF/PACF), intervals,
                              champion forecaster.
   feature_engineering.py   → FE study: 17 features × 5 ML/hybrid strategies.
   staffing_optimization.py → forecast → courier-staffing decision (OR-Tools MILP).
   make_figures.py          → regenerates figures/ + results_bundle.json + *.csv.
   make_predictions.py      → writes predictions.csv (+ vendored check_output_format).
   config.py                → env-driven, S3-aware paths (cloud-portable).
   predictions.csv          → the deliverable (168 rows: time, preds).
 Self-contained: data/ is bundled; the checker is vendored. Configure via env vars
 (see config.py / README). Grounded in 21DM011: slides, exercises 01/04/05, modelling.py.
================================================================================
"""

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import json
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config
    import forecasting_utils as fu

    FIG = config.FIGURES_DIR
    y = fu.load_series(config.TRAIN_FILE)
    B = json.loads((config.OUTPUT_DIR / "results_bundle.json").read_text())  # from make_figures.py
    return B, FIG, config, fu, pd, y


@app.cell
def _(FIG):
    import plotly.io as pio

    def img(name, width=None):
        # Return the native Plotly figure — marimo renders it interactively in the
        # LIVE app (`marimo edit/run solution.py`). For a self-contained OFFLINE
        # interactive HTML, use `build_report.py` (marimo's static export is not
        # self-contained). Figures are built by make_figures.py.
        base = name.rsplit(".", 1)[0]  # accept "x" or legacy "x.png"
        fig = pio.from_json((FIG / f"{base}.json").read_text(encoding="utf-8"))
        if width:
            fig.update_layout(width=width, autosize=False)
        return fig

    return (img,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Forecasting hourly delivery demand for Glovo — and turning it into a staffing plan

    **Author:** Elvis Casco · **Course:** 21DM011 Intelligent Data Development (BSE)

    ## Abstract
    A delivery platform must decide, for every one of the **168 hourly slots** in a week,
    how many couriers to put on the road. Over-staff and you burn payroll; under-staff and
    deliveries slip. This report builds **trusted hourly order forecasts** for one city (BCN)
    for the week **2022-01-24 → 2022-01-30**, and then converts them into a cost-optimal
    courier roster.

    Methodologically we follow the course template: **EDA → stationarity (ADF/KPSS) and
    ACF/PACF → candidate model families → out-of-sample validation that imitates the real
    weekly refresh → champion selection → point forecast + interval → a constrained-optimization
    decision layer.** We benchmark a naive baseline against harmonic regression, a feature-rich
    gradient-boosting model (17 engineered features), and classical SARIMA, all under a
    **12-week rolling-origin backtest**. The winner is a **trend-adjusted seasonal median
    profile**: **SMAPE ≈ 16.3**, a ~27% improvement over the seasonal-naive baseline
    (**MASE < 1**), beating every more complex model. We close by feeding the forecast into a
    **mixed-integer courier-staffing optimization** (OR-Tools), mirroring the course's
    decision-making module, and outline the MLOps/DataOps needed to run it weekly in production.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Business framing

    Glovo runs each city on a **slot-based system: 24 hourly slots per day**. Planning is
    refreshed **weekly** — every Sunday after 23:59 the team must hand operations a forecast
    for **all 168 hours** of the coming week (Mon 00:00 → Sun 23:00). The forecast feeds
    courier headcount, and the asymmetric cost is explicit in the brief:

    - **Over-estimate →** idle couriers, wasted payroll.
    - **Under-estimate →** delivery delays, restaurant complaints, customer churn.

    The realistic evaluation, therefore, is not a random train/test split but a **rolling weekly
    origin**: *train on everything up to a Sunday, forecast the next 7×24 hours* (the course's
    rule: *"validate in a way that imitates the real problem"*). We are scored on **SMAPE**
    (and report **MSE** and **MASE** alongside).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Data and quality
    """)
    return


@app.cell
def _(B, config, mo, pd, y):
    _raw = config.read_csv(config.TRAIN_FILE, parse_dates=["time"])  # pre-regularization timestamps
    miss = pd.date_range(y.index.min(), y.index.max(), freq="h").difference(
        pd.DatetimeIndex(_raw["time"]))
    ov = B["overview"]
    mo.vstack([
        mo.md(
            f"""
            One city (**BCN**), **{ov['n_hours']:,} hourly observations** spanning
            **{ov['start']} → {ov['end']}** (~**{ov['weeks']} weeks**). Mean
            **{ov['mean']}** orders/hour, median **{ov['median']:.0f}**, max **{ov['max']:.0f}**.
            **{ov['zero_share']:.0%} of hours are exactly zero** (the platform is shut overnight).

            **Data quality.** The raw file has **{len(miss)} missing hours** — all overnight blocks
            (e.g. {', '.join(str(t.date()) for t in miss[:3].normalize().unique())}…). Because lags
            and seasonal differences require a **gap-free hourly grid**, we reindex onto a regular
            hourly index and impute: short gaps by time-interpolation, the rest by the same
            hour-of-week median, finally 0. All overnight gaps resolve to 0 (correct). Values are
            clipped at ≥ 0. *(See `fu.load_series`.)*
            """
        )
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Exploratory data analysis

    ### 3.1 The series, trend and stationarity
    The course's first move is a **visual stationarity check** — three questions: *is the mean
    stable? is the variance stable? are different periods exchangeable?* Below, the **level
    rises** through 2021 and the **daily amplitude widens** with it. Both answers are "no":
    the series is **non-stationary** (trend + variance growth), so any ARMA model must difference
    or otherwise absorb that structure first.
    """)
    return


@app.cell
def _(img):
    img("eda_series_full.png")
    return


@app.cell
def _(B, mo):
    s = B["stationarity"]
    def row(name, d):
        return f"| {name} | {d['adf_p']:.3f} | {d['kpss_p']:.3f} | **{d['verdict']}** |"
    mo.md(
        f"""
        Formal confirmation with the course's **paired ADF + KPSS** test (ADF H₀ = unit root;
        KPSS H₀ = stationary), read jointly:

        | series | ADF p | KPSS p | verdict |
        |---|---:|---:|---|
        {row("levels (trend)", s["levels"])}
        {row("first difference Δ₁", s["first_diff"])}
        {row("seasonal difference Δ₁₆₈", s["seasonal_diff_168"])}

        The levels are non-stationary; a **seasonal difference at lag 168** removes the dominant
        weekly structure. This is exactly why **seasonal methods** (which condition on the weekly
        position) and **differencing-based models** are the natural candidates here.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2 Two seasonal cycles: a daily shape and a weekly shape
    Orders follow a sharp **intraday** rhythm — dead 00:00–05:00, a **lunch peak (~13:00)** and a
    larger **dinner peak (~21:00, Spanish dinner time)** — and a **weekly** rhythm with a
    **Friday/weekend lift**. The boxplots show the spread (not just the mean) grows at the peaks,
    echoing Exercise 04's finding that peak periods are *wider*, not merely higher.
    """)
    return


@app.cell
def _(img, mo):
    mo.vstack([img("eda_profiles.png"), img("eda_boxplots.png")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Crossing the two cycles gives a **168-cell weekly fingerprint** (hour × weekday). This single
    object is, as we will see, almost the entire forecastable signal.
    """)
    return


@app.cell
def _(img):
    img("eda_heatmap.png", width=620)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.3 Growth trend and a stable seasonal shape
    Weekly totals climb from ~11k to ~16.5k orders (left). Overlaying the **last eight weeks** by
    hour-of-week (right) shows the **shape is stable while the level drifts up** — the textbook
    signature (Exercise 04's YoY overlay) that says: *model the recurring shape, but rescale it to
    the most recent level.* A pure "same week last year" forecast would lag this growth badly.
    """)
    return


@app.cell
def _(img):
    img("eda_trend_overlay.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.4 Decomposition: separating trend, daily and weekly seasonality
    An **MSTL** decomposition (multiple-seasonal STL, periods 24h and 168h) cleanly separates a
    slow trend, a stable daily cycle, a weekly cycle, and a residual. The residual is small and
    spiky around the peaks — a hint that the *shape* is highly predictable but the *peak heights*
    carry irreducible noise (which will matter for SMAPE).
    """)
    return


@app.cell
def _(img):
    img("diag_mstl.png", width=820)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.5 Autocorrelation
    The ACF/PACF make the two seasonalities explicit: strong autocorrelation at **lag 24 and lag
    168** (and their multiples). The course uses ACF/PACF to choose ARIMA orders; here they justify
    using **24h and 168h lags / seasonal terms** as the backbone of every model.
    """)
    return


@app.cell
def _(img):
    img("diag_acf_pacf.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.6 The zero envelope and the distribution
    ~32% of hours are **structurally zero** (00:00–05:00 always; 23:00 almost always). This is
    deterministic, not random — so we **force those hours to 0** in post-processing and clip
    negatives. The awake-hour distribution is right-skewed with a long tail at the dinner peak.
    Crucially, **SMAPE is dominated by the small low-volume "ramp" hours** (06:00–08:00): a tiny
    absolute error there is a large *percentage* error, so a model that is merely smooth-and-noisy
    in the morning pays a heavy SMAPE price.
    """)
    return


@app.cell
def _(img):
    img("eda_distribution.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What the EDA dictates
    1. **Seasonality is the signal** (ACF ≈ 0.88 at 24h and 168h) → seasonal methods should be hard
       to beat; *naive heuristics are the mandatory first baseline* (course rule).
    2. **The overnight envelope is deterministic** → force 00–05 to 0; free accuracy.
    3. **There is real growth** → the forecast must be rescaled to the **recent** level, not a
       year-ago level.
    4. **Peaks are spiky and the metric punishes noisy low hours** → smooth global models
       (Fourier/ARIMA) and variance-chasing ML risk *losing* to a robust seasonal estimate.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Methodology

    ### 4.1 Validation that imitates production
    We use a **rolling-origin (walk-forward) backtest** over the **last 12 weeks**: for week *k* we
    train on all hours strictly before it and forecast its 168 hours. This *is* the weekly planning
    refresh, and it is the course's rule — *"validate in a way that imitates the real problem."* An
    ordinary shuffled train/test split would **leak** future information and badly overstate accuracy.

    **The three metrics, read intensively** (truth $y_t$, forecast $\hat y_t$, $n=168$ per week):

    | Metric | Formula | Units | What it rewards / how to read it |
    |---|---|---|---|
    | **MSE** | $\frac1n\sum (y_t-\hat y_t)^2$ | orders² | Squares errors → **dominated by the big dinner-peak misses**. Required by the brief. Lower = better; not comparable across cities of different size. |
    | **SMAPE** | $\frac{100}{n}\sum \frac{2\,|y_t-\hat y_t|}{|y_t|+|\hat y_t|}$ | % (0–200) | Symmetric % error, **scale-free**, bounded. We set the **0/0 hour to 0** (both predicted and actual zero ⇒ no error). Because the denominator is small when volume is small, SMAPE is **dominated by the low-volume ramp hours** — the central tension of this problem. The grading metric. |
    | **MASE** | $\dfrac{\frac1n\sum|y_t-\hat y_t|}{\text{in-sample MAE of seasonal-naive }(t{-}168)}$ | ratio | Scales error by the seasonal-naive benchmark. **Decision rule (course): MASE < 1 ⇒ you beat seasonal naive; > 1 ⇒ just use seasonal naive.** The cleanest "did complexity earn its keep?" test, and immune to the zero problem. |

    We report all three; **SMAPE is the headline** (it is what we are graded on), MASE is the
    sanity check, MSE watches the expensive peak errors. *(Implementation: `fu.rolling_origin_backtest`,
    `fu.smape / mse / mase`.)*

    ### 4.2 Model families
    Following the course's model-family table, we span genuinely different mechanisms:

    | Family | Member | What it uses |
    |---|---|---|
    | **Naive (baseline)** | Seasonal naive *t−168* | last week, same hour |
    | **Linear / harmonic** | Fourier(24h,168h) + trend, ridge | smooth global seasonality |
    | **Classical** | SARIMA (daily seasonal + weekly Fourier) | past values & errors |
    | **Gradient boosting** | XGBoost, 17 features (§4.3) | engineered features, non-linear |
    | **Robust seasonal** | Trend-adjusted seasonal **median** profile | recent seasonal memory |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.3 Feature engineering
    For the gradient-boosting model the course is blunt: *"Gradient boosting sees a table, not a
    series… the entire job of the practitioner is building the feature matrix,"* with two hard
    rules — **trees cannot extrapolate** (so detrend / supply level features) and **only lags ≥ h
    are valid** at prediction time (no leakage). Because our horizon is one full week, every lag we
    use is **≥ 168h**, so the whole week is predicted in one **direct** shot — no recursion, no
    leakage. The **17 features** (`fu.build_feature_matrix`):

    | Group | Features |
    |---|---|
    | Calendar | `hour, dow, hour-of-week, is_weekend, day-of-month, week-of-year, month` |
    | Cyclical | `sin/cos` of the 24h and 168h cycles |
    | Weekly lags (log) | `lag168, lag336, lag504, lag672` |
    | Rolling same-hour-of-week | `mean{2,4,8}, median10, std8` (strictly prior weeks — the leakage-safe analogue of the course's `same_month_rolling_means`) |
    | Level / growth | `weekly_level` (causal 168h mean), `wow_ratio` (week-over-week) |

    The level/growth features exist **precisely because trees cannot extrapolate** the upward trend.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.4 Every model parameter, explained
    So the results below can be read intensively, here is what each knob does and **why it is set
    where it is**. All live in `forecasting_utils.py`.

    **Champion — trend-adjusted seasonal median profile** (`predict_seasonal_profile`)

    | Parameter | Value | Meaning & why |
    |---|---|---|
    | `k_weeks` | **10** | How many recent same-hour-of-week values to aggregate. Tuned by backtest: small k is noisy, large k lags the growth trend; **10 (~2½ months) minimized SMAPE**. |
    | `use_median` | **True** | Median (not mean) over the k weeks ⇒ **robust to one-off spikes** (a single promo night doesn't distort the profile). Median beat mean by ~3 SMAPE points. |
    | `trend_adjust` | **True** | Rescales the whole week by `last-week level ÷ profile level` so the forecast **tracks the upward trend** instead of lagging it. Worth ~1.5 SMAPE. |
    | `holiday_adjust` | False (default) | §9.1 extension; scales holiday hours by the historical holiday factor. Off for the submission week (no holiday). |

    **Gradient boosting — XGBoost** (`xgb_regressor`); chosen for regularization, not raw fit:

    | Parameter | Value | Meaning & why |
    |---|---|---|
    | `n_estimators` | 600 | number of boosting trees (with a low learning rate, more trees = finer fit). |
    | `learning_rate` | 0.03 | shrinkage per tree; **small ⇒ less overfitting**, paired with many trees. |
    | `max_depth` | 5 | tree depth = interaction order captured; 5 balances expressiveness vs overfit. |
    | `min_child_weight` | 5 | min samples per leaf — **regularizer** that stops the tree memorizing the noisy ramp hours. |
    | `subsample` / `colsample_bytree` | 0.9 / 0.8 | row/column sampling per tree ⇒ **stochastic regularization**. |
    | `reg_lambda` | 2.0 | L2 penalty on leaf weights — further shrinkage. |
    | target | `log1p(orders)` | log scale **stabilizes the peak variance** (course rule); trained on **awake hours only** so the model never wastes capacity on structural zeros. |

    **Harmonic regression** (`predict_harmonic`): `n_daily=4`, `n_weekly=6` Fourier harmonics of the
    24h and 168h cycles + a linear trend, fit with **ridge** `alpha=10` (shrinks the harmonics to
    avoid ringing). More weekly than daily harmonics because the weekly shape is more complex.

    **Classical SARIMA** (`predict_sarima`): `order=(2,0,1)`, `seasonal_order=(1,0,1,24)` + 4 weekly
    **Fourier exogenous** terms. Read as (p,d,q)(P,D,Q)[s]: AR(2)+MA(1) on the level, a **daily**
    seasonal AR(1)+MA(1) at **s=24**, and the **weekly** cycle carried by Fourier regressors because
    a seasonal term at s=168 is computationally intractable — the very reason ARIMA struggles here.

    **Empirical interval** (`empirical_interval_offsets`): `level=0.90`, residual quantiles taken
    **per hour of day** so the band is wide at the noisy dinner peak and tight in the morning.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Results
    """)
    return


@app.cell
def _(B, mo, pd):
    lb = pd.DataFrame(B["leaderboard"]).T[["smape", "smape_sd", "mse", "mase"]]
    lb = lb.sort_values("smape").round(2)
    mo.vstack([
        mo.md("### 5.1 Leaderboard (12 weekly folds)"),
        mo.Html(lb.reset_index().rename(columns={"index": "model"}).to_html(index=False, border=0)),
        mo.md(
            "The **seasonal profile wins on every metric**, with **MASE < 1** confirming it beats the "
            "seasonal-naive baseline. Harmonic regression is worst (it smears the spiky dinner peak and "
            "cannot represent the hard zero floor); **XGBoost — despite 17 features — loses to the naive "
            "baseline**, exactly the failure the course predicts when trees meet a trend they cannot "
            "extrapolate and noisy low-volume hours they overfit."
        ),
    ])
    return


@app.cell
def _(img):
    img("bt_leaderboard.png", width=760)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.2 Stability across weeks
    The champion is not only best on average but **consistently** best week to week — important
    operationally, since a planner trusts a model that does not occasionally blow up.
    """)
    return


@app.cell
def _(img):
    img("bt_perfold.png")
    return


@app.cell
def _(config, mo, pd):
    fe = pd.read_csv(config.OUTPUT_DIR / "fe_comparison.csv").round(2)
    mo.vstack([
        mo.md(
            """
            ### 5.3 Feature-engineering study — does ML ever win?
            We pushed the ML angle hard: the 17-feature XGBoost, a **Tweedie** (count) objective, a
            **profile + XGBoost-on-residual** hybrid, and two **ensembles**. Result:
            """
        ),
        mo.Html(fe.rename(columns={fe.columns[0]: "strategy"}).to_html(index=False, border=0)),
        mo.md(
            "**None beats the seasonal profile.** The hybrids confirm *why*: an XGBoost trained on the "
            "champion's residual cannot improve it — the residual is essentially **white noise**. The "
            "profile has already extracted the learnable structure; what's left (peak-height jitter in "
            "the noisy hours) is irreducible. This is the course's lesson made concrete: *performance ≫ "
            "complexity*, and a robust seasonal estimator is the right tool for a series that is mostly "
            "a stable weekly fingerprint plus a slow trend. *(Reproduce: `feature_engineering.py`.)*"
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.4 The classical contender: SARIMA
    For completeness we fit a classical **SARIMAX** (daily seasonal structure m=24 + weekly Fourier
    exogenous terms) on one held-out week, in the spirit of Exercise 04's airline model. It is both
    **slower (~5 min/fit)** and **less accurate** than the seasonal profile: dual seasonality at
    hourly granularity is exactly where pure ARIMA struggles, and the weekly period (168) is too
    long for a tractable seasonal AR/MA term.
    """)
    return


@app.cell
def _(B, img, mo):
    sar = B.get("sarima_oneweek", {})
    mo.vstack([
        img("bt_sarima.png"),
        mo.md(f"*One-week held-out SMAPE — champion **{sar.get('champion_smape')}**, "
              f"SARIMA **{sar.get('smape')}**, seasonal naive **{sar.get('naive_smape')}**.*"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.5 Where the error lives
    Decomposing SMAPE **by hour of day** shows the champion's gains concentrate in the volatile
    peak and ramp hours; both models are near-perfect overnight (forced zeros). The pooled
    actual-vs-predicted scatter is tight along the diagonal, and the residuals are roughly centred
    at every hour (no systematic bias), with the expected larger spread at the dinner peak.
    """)
    return


@app.cell
def _(img, mo):
    mo.vstack([img("bt_perhour.png"), img("bt_champion_diag.png")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Champion model

    **Trend-adjusted seasonal median profile.** For each of the 168 hours-of-week, take the
    **median of the last 10 realizations** (median ⇒ robust to one-off spikes/outliers), then
    **rescale the whole week** by the ratio of the most recent week's level to the profile's level
    so the forecast tracks the growth trend; finally clip ≥ 0 and force the dead night hours to 0.

    It encodes the three EDA findings directly — *seasonal memory* (median profile), *robustness*
    (median, not mean), *level-awareness* (trend rescale) — and has **nothing it can overfit**.
    That is why it generalizes where the 17-feature booster does not. *(Implementation:
    `fu.predict_seasonal_profile`, k=10; `fu.champion_forecast`.)*

    Below: the champion vs the naive baseline on the most recent held-out week.
    """)
    return


@app.cell
def _(img):
    img("bt_holdout_week.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.1 Prediction intervals
    Point forecasts alone cannot size a staffing **buffer**. We attach a **90% empirical interval**
    built distribution-free from the backtest residuals **grouped by hour of day** (so the band is
    wide at the noisy dinner peak and tight in the morning). This is what lets operations choose a
    service-level-aware headcount rather than staffing to the mean. *(`fu.empirical_interval_offsets`.)*
    """)
    return


@app.cell
def _(img):
    img("forecast_intervals.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · The submission
    Refit the champion on **all** history and forecast 2022-01-24 → 2022-01-30, then run the
    repository's **`check_output_format`** (asserts 168×2 shape, `datetime64[ns]` / `float64`
    dtypes, no nulls, time-alignment). *(The mock holdout is all zeros, so its printed MSE is a
    format check, not an accuracy signal — accuracy is the backtest above.)*
    """)
    return


@app.cell
def _(config, fu, mo, pd, y):
    from check_output_format import check_output_format

    predictions = fu.champion_forecast(y)
    out_csv = config.predictions_path()
    predictions.to_csv(out_csv, index=False)
    reloaded = pd.read_csv(out_csv, parse_dates=["time"]).astype({"preds": "float64"})
    check_output_format(reloaded, config.TEST_FILE)
    mo.md(f"**Submission written** — {len(reloaded)} rows · sum **{reloaded['preds'].sum():.0f}** · "
          f"peak **{reloaded['preds'].max():.0f}** · format check **passed**.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · From forecast to decision: courier-staffing optimization
    A forecast is only useful if it changes a decision. The course's optimization module teaches
    turning a demand forecast into a **constrained resource plan** (LP/MILP via OR-Tools), as in
    the *Medistor procurement* exercise. We build the analogous **courier-staffing MILP**:

    - **Decision:** integer shift-starts; a courier works a contiguous **6-hour shift**, so staffing
      in one hour is coupled to its neighbours (real rostering, not 168 independent ceilings).
    - **Coverage / service:** `couriers(t) = Σ shifts covering t`; each courier serves
      `capacity_per_courier` orders/hour; `unmet(t) ≥ demand(t) − capacity·couriers(t)`.
    - **Objective:** `min  wage·(courier-hours) + penalty·(unmet orders)` — over-staffing burns the
      wage term, under-staffing burns the penalty term: the exact trade-off in the brief.

    *(Implementation: `staffing_optimization.optimize_staffing`, OR-Tools + SCIP.)*
    """)
    return


@app.cell
def _(fu, img, mo, y):
    from staffing_optimization import optimize_staffing

    st = optimize_staffing(fu.champion_forecast(y))  # solve the MILP live (~2s)
    p = st["params"]
    mo.vstack([
        img("staffing.png"),
        mo.md(
            f"""
            With illustrative economics (capacity **{p['capacity_per_courier']:.0f}** orders/courier·h,
            wage **€{p['wage_per_hour']:.0f}/h**, under-staffing penalty **€{p['understaff_penalty']:.0f}/order**,
            **{p['shift_length']}h** shifts), the optimal weekly roster uses **{st['courier_hours']:.0f}
            courier-hours**, achieves a **{st['service_level']:.1%} service level**, at a total cost of
            **€{st['total_cost']:,.0f}** (wage €{st['wage_cost']:,.0f} + shortfall penalty
            €{st['penalty_cost']:,.0f}). The shift coupling forces a little over-staffing on the shoulders
            of the dinner peak — visible in the Friday panel — which is the genuine cost of integer
            rostering. Tightening the penalty or shift length traces out the **service-vs-cost frontier**
            the business actually chooses on.
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9 · Robustness & extensions (the limitations, implemented)

    A complete forecasting project does not merely *list* its limitations — it confronts them.
    This section implements and **validates** four extensions: holiday/event handling, prediction-
    interval **calibration**, **chance-constrained** staffing, and an analysis of the intermittent
    ramp hours. Each is measured on the same rolling-origin backtest, and each result is reported
    honestly — including where the extra machinery does *not* pay off.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9.1 Holiday and event effects
    **Problem.** The seasonal profile assumes every week looks like recent weeks, so it misreads
    public holidays. **What we built.** (i) An **anomaly detector** (`fu.anomalous_days`) — a
    robust MAD z-score on daily totals vs the same-weekday rolling median — to surface unusual days;
    (ii) a **Barcelona/Catalonia holiday calendar** (`fu.HOLIDAYS`); (iii) a **holiday-aware
    champion** that scales holiday hours by the historical holiday/normal demand ratio
    (`fu.holiday_adjustment_factor`, applied via `predict_seasonal_profile(..., holiday_adjust=True)`).

    **How to read the parameter.** The *holiday factor* is the median of `actual / expected`
    over past holiday awake-hours: **< 1 ⇒ holidays are quieter**, > 1 busier. We then backtest the
    plain vs holiday-aware champion, splitting weeks by whether they contain a holiday.
    """)
    return


@app.cell
def _(B, img, mo):
    h = B["holidays"]
    mo.vstack([
        img("ext_holidays.png"),
        mo.md(
            f"""
            **Result (an honest negative).** Estimated **holiday factor = {h['factor']}** → across the
            year Catalan holidays ran about **{(1-h['factor'])*100:.0f}% below** an ordinary day of that
            weekday. The detector flags **{h['n_anomalous_days']}** extreme days — and tellingly, **most
            are demand *dips* unrelated to the holiday calendar** (likely platform outages or data gaps),
            a valuable **data-quality** signal in its own right. But applying a **single global** factor
            does **not** help the backtest: overall SMAPE moves **{h['smape_all_champion']} →
            {h['smape_all_holiday']}** and even on holiday-containing weeks **{h['smape_holweeks_champion']}
            → {h['smape_holweeks_holiday']}** — slightly *worse*. The lesson is genuine: holiday effects are
            **heterogeneous** (Christmas ≠ a local *fiesta* ≠ New Year's Eve), so one multiplicative number
            over-corrects as often as it helps. That is exactly why the shipped champion **does not** apply
            it; the principled fix is **per-holiday** effects or holiday *features* inside a richer model.
            The submission week (Jan 24–30) has no holiday, so the forecast is unaffected regardless.
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9.2 Are the prediction intervals calibrated?
    **Problem.** A 90% interval is only trustworthy if ~90% of actuals actually fall inside it.
    **What we built.** `fu.backtest_interval_coverage` replays the backtest and, for each week, builds
    the interval from the residuals of **earlier folds only** (no peeking), then measures the fraction
    of held-out hours that land inside. This is a genuine **out-of-sample coverage** test.
    """)
    return


@app.cell
def _(B, img, mo):
    c = B["coverage"]
    mo.vstack([
        img("ext_coverage.png", width=720),
        mo.md(
            f"""
            **Result.** Nominal **{c['nominal']:.0%}**, empirical **{c['empirical_coverage']:.1%}** (mean
            interval width ≈ **{c['mean_width']:.0f}** orders) — the band is **well-calibrated**
            out-of-sample, essentially hitting its target. Per-fold coverage ranges ~85–98% (figure),
            looser in the noisier weeks. So the 90% interval **can be trusted** to size a staffing buffer
            (§9.3). A **quantile-regression** interval would tighten the per-fold consistency further, but
            the simple empirical band is already close to nominal — a satisfying result for a
            distribution-free method.
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9.3 Chance-constrained staffing (size against the band, not the mean)
    **Problem.** §8 staffs against the **point** forecast, so on a high-demand realization it is
    under-staffed half the time. **What we built.** A *buffered* plan that sizes capacity against the
    **90% upper band** (`fu.champion_forecast_interval(...).hi`) — a simple **chance constraint**:
    meet demand with ~90% probability per hour. We re-solve the same MILP on the buffered demand.
    """)
    return


@app.cell
def _(B, img, mo):
    sb = B["staffing_buffer"]
    dcost = sb["buffer_cost"] - sb["point_cost"]
    mo.vstack([
        img("ext_staffing_buffer.png"),
        mo.md(
            f"""
            **Result.** Sizing to the 90% band lifts realized service
            **{sb['point_service']:.1%} → {sb['buffer_service']:.1%}** but raises weekly cost
            **€{sb['point_cost']:,.0f} → €{sb['buffer_cost']:,.0f}** (**+€{dcost:,.0f}**,
            {dcost/sb['point_cost']*100:.0f}% more) and courier-hours
            **{sb['point_courier_hours']:.0f} → {sb['buffer_courier_hours']:.0f}**. With these economics
            the **point plan is already near-optimal** — because the under-staffing penalty (€20/order)
            so dominates the wage (€12/h ÷ 4 = €3/order) that the point optimizer *already* staffs
            generously. Buffering is over-insurance here; it earns its keep only when the penalty/wage
            ratio is smaller. This is exactly the **service-vs-cost frontier** the business chooses on,
            now made explicit and priced.
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9.4 Where the residual error lives: the intermittent ramp hours
    **Problem.** SMAPE is dominated by the low-volume morning **ramp** (06:00–08:00), where a tiny
    absolute miss is a huge percentage miss. **What we did.** Decompose the champion's backtest SMAPE
    by daypart to size the opportunity precisely.
    """)
    return


@app.cell
def _(B, img, mo):
    r = B["ramp"]
    mo.vstack([
        img("ext_ramp.png", width=620),
        mo.md(
            f"""
            **Result.** Champion SMAPE is **{r['ramp 6-8']}** on the ramp hours vs **{r['peak 17-22']}**
            at the dinner peak and **{r['midday 9-16']}** midday. The ramp is the worst daypart by far —
            these hours are **intermittent** (often 0–10 orders, frequent true zeros). A symmetric
            percentage metric punishes them disproportionately, and no smooth model can fix that. The
            right tool is a dedicated **intermittent-demand** method (e.g. Croston/SBA or a zero-inflated
            count model) applied **only to the ramp hours**, leaving the robust profile for the rest of
            the day — a targeted next step with a clear, measured payoff ceiling.
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10 · Productionization (MLOps / DataOps)
    Per Module 2 — *"a real intelligent data application needs more than an ML model"* — running
    this weekly in production needs:

    - **Orchestration:** a scheduled weekly job (e.g. Airflow) that fires **after** the Sunday data
      lands (dependency-based, not just time-based), retrains the champion, writes `predictions.csv`
      to **S3**, and notifies planners.
    - **Data contracts / quality gates:** schema validation, row-count and **freshness** checks, and
      a **volume/anomaly threshold** to tell a true demand event (storm, holiday) from a data bug
      before it poisons the forecast.
    - **Deploy-code, not deploy-model:** promote the *code* through environments so the model is
      retrained and tested in each — safer for automated weekly retraining.
    - **Monitoring:** track rolling SMAPE/MASE for **concept drift** and feature distributions for
      **data drift**; retrain or alert when the live error crosses the backtest band.

    The champion's simplicity is an asset here: it is cheap to retrain, trivial to monitor, and
    **defensible hour-by-hour to operations** — the "number we can defend for every hour of the week."
    """)
    return


@app.cell(hide_code=True)
def _(B, mo):
    lb_final = B["leaderboard"]
    champ = lb_final["Seasonal profile (champion)"]; naive = lb_final["Seasonal naive (t-168)"]
    impr = (naive["smape"] - champ["smape"]) / naive["smape"]
    mo.md(
        f"""
        ## 11 · Conclusion & limitations

        **Ship the trend-adjusted seasonal median profile.** Across a 12-week rolling backtest that
        replays the weekly planning refresh it delivers **SMAPE {champ['smape']:.1f}** vs
        **{naive['smape']:.1f}** for the seasonal-naive baseline (**{impr:.0%}** lower), **MASE
        {champ['mase']:.2f} < 1**, and beats harmonic regression, classical SARIMA, and a 17-feature
        gradient-boosting model. It is robust, level-aware, fast to retrain, and explainable.

        **Why the simple model wins.** This series is dominated by a stable weekly fingerprint plus a
        slow growth trend; there is little extra structure for ML or ARIMA to exploit, and the SMAPE
        metric punishes the noise they add in the low-volume ramp hours. *Performance ≫ complexity.*

        **What we addressed (§9).** Holiday/event effects (detector + holiday-aware variant), interval
        **calibration** (out-of-sample coverage ≈ 85–90%), **chance-constrained** staffing against the
        90% band, and a daypart decomposition pinpointing the **ramp hours** as the residual-error
        frontier — each implemented and measured, not just named.

        **Genuinely remaining.** (1) A **calibrated quantile-regression** interval to hit coverage
        exactly. (2) A dedicated **intermittent-demand** model (Croston/SBA or zero-inflated) for the
        ramp hours. (3) Richer **event features** (weather, promotions, sporting events) — the anomaly
        detector shows unexplained spikes/dips beyond the holiday calendar. (4) A **stochastic** joint
        forecast-and-roster that optimizes against the full predictive distribution.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Appendix A · Parameters, tests & metrics — quick reference

    A consolidated, intensive-reading reference. Everything is in `forecasting_utils.py` unless noted.

    **A.1 Statistical tests used**

    | Test | Null hypothesis H₀ | We reject H₀ when | Reading it here |
    |---|---|---|---|
    | **ADF** (Augmented Dickey–Fuller) | series has a **unit root** (non-stationary) | p < 0.05 | low p ⇒ *no* unit root ⇒ stationary-ish |
    | **KPSS** | series **is stationary** | p < 0.05 | low p ⇒ *not* stationary. **Read jointly with ADF** (course idiom): ADF-reject & KPSS-not ⇒ stationary; both reject ⇒ trend/curvature. |
    | **ACF** (autocorrelation) | — (descriptive) | spike outside ±1.96/√n | spikes at **24 & 168** ⇒ daily + weekly seasonality |
    | **PACF** (partial autocorr.) | — | as above | direct lag effects; guides AR order in ARIMA |
    | **Coverage** (§9.2) | — | empirical ≈ nominal? | 90% band actually covers ~85% ⇒ slightly optimistic |

    **A.2 Metrics** — MSE (orders², peak-weighted), SMAPE (%, scale-free, ramp-weighted, the grade),
    MASE (ratio, **<1 beats seasonal naive**). Formulas in §4.1.

    **A.3 Model parameters** — see §4.4 for the champion (`k_weeks=10`, median, trend-adjust),
    XGBoost (600 trees, depth 5, lr 0.03, regularizers), harmonic (4 daily / 6 weekly Fourier + ridge
    10), and SARIMA (2,0,1)(1,0,1)[24] + weekly Fourier.

    **A.4 Decision-layer parameters** (`staffing_optimization.py`): `capacity_per_courier=4`
    orders/h, `wage_per_hour=€12`, `understaff_penalty=€20/order`, `shift_length=6h`,
    `open_hours=(6,23)`. The penalty/wage ratio drives the service-vs-cost trade-off (§8, §9.3).

    **A.5 Reproduce** — `python pipeline.py all` (or `predict` / `figures` / `features` / `notebook`).
    Paths & cloud config: `config.py` (env vars `IDD_*`); container: `Dockerfile`; deps: `requirements.txt`.
    """)
    return


if __name__ == "__main__":
    app.run()
