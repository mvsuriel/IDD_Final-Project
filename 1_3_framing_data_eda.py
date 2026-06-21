# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.3",
#     "numpy>=2.4.6",
#     "pandas>=3.0.3",
#     "plotly>=6.8.0",
#     "statsmodels==0.14.6",
# ]
# ///
"""
================================================================================
 PART 1 / 3 — Framing · Data quality · Exploratory data analysis
================================================================================
 Glovo hourly demand forecasting — 21DM011 Final Project, Elvis Casco.

 SELF-CONTAINED. This file needs nothing but `data/train_data.csv` and the
 scientific-Python stack (numpy, pandas, plotly, statsmodels). It does NOT read
 results_bundle.json or figures/, and does NOT import the project modules —
 every number and every chart is recomputed from the raw data when you open it.
 (~1 min: the MSTL decomposition is the slow step.)

   §1  Business framing
   §2  Data and quality
   §3  Exploratory data analysis
        3.1 Series, trend & stationarity (ADF / KPSS)
        3.2 Two seasonal cycles (daily + weekly)
        3.3 Growth trend & stable seasonal shape
        3.4 MSTL decomposition
        3.5 Autocorrelation (ACF / PACF)
        3.6 Zero envelope & distribution
        →  What the EDA dictates

 Run:  marimo edit 1_3_framing_data_eda.py
 Optional env overrides: IDD_DATA_DIR, IDD_TRAIN_FILE (default ./data/train_data.csv).
 Reading order (each file is independent): 4_6_methodology_results_champion.py,
 then 7_9_submission_staffing_robustness.py.
================================================================================
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # ------------------------------------------------------------------ #
    # Data loading + diagnostics
    # ------------------------------------------------------------------ #
    import os
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HORIZON = 168                     # 7 days x 24 h = one planning week
    DEAD_HOURS = (0, 1, 2, 3, 4, 5)   # structurally-zero overnight hours
    DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    # course palette
    C_HIST, C_FC, C_PEAK, C_ALT, C_GREY = "#4C72B0", "#D55E00", "#C44E52", "#55A868", "#7f7f7f"

    _HERE = Path(__file__).resolve().parent
    _DATA_DIR = Path(os.environ.get("IDD_DATA_DIR", str(_HERE / "data")))
    TRAIN_FILE = os.environ.get("IDD_TRAIN_FILE", str(_DATA_DIR / "train_data.csv"))

    def hour_of_week(idx):
        """0..167: Monday 00:00 -> 0, Sunday 23:00 -> 167."""
        return (idx.dayofweek * 24 + idx.hour).to_numpy()

    def load_series(train_csv):
        """Load the training CSV as a *regular* gap-free hourly Series of orders,
        imputing the handful of missing hours (interp -> hour-of-week median -> 0)
        and clipping at 0 — lags and seasonal differences need a gap-free grid."""
        df = pd.read_csv(train_csv, parse_dates=["time"]).sort_values("time")
        s = df.set_index("time")["orders"].astype(float)
        grid = pd.date_range(s.index.min(), s.index.max(), freq="h")
        s = s.reindex(grid)
        how = hour_of_week(s.index)
        s = s.interpolate("time", limit=3)
        how_median = pd.Series(s.groupby(how).transform("median").values, index=s.index)
        s = s.fillna(how_median).fillna(0.0).clip(lower=0)
        s.name = "orders"
        return s

    def stationarity_report(series, regression="c"):
        """Paired ADF + KPSS test with the course's joint verdict. ADF H0 = unit
        root; KPSS H0 = stationary — read together."""
        from statsmodels.tsa.stattools import adfuller, kpss

        v = series.dropna().to_numpy(float)
        adf_p = float(adfuller(v, regression="ct" if regression == "ct" else "c")[1])
        with np.errstate(all="ignore"):
            kpss_p = float(kpss(v, regression="ct" if regression == "ct" else "c", nlags="auto")[1])
        adf_stat = adf_p < 0.05      # rejects unit root
        kpss_stat = kpss_p >= 0.05   # fails to reject stationarity
        if adf_stat and kpss_stat:
            verdict = "stationary"
        elif not adf_stat and not kpss_stat:
            verdict = "unit root (difference it)"
        elif adf_stat and not kpss_stat:
            verdict = "trend-stationary / curvature"
        else:
            verdict = "inconclusive"
        return {"adf_p": adf_p, "kpss_p": kpss_p, "verdict": verdict}

    def mstl_decompose(series, periods=(24, 168)):
        from statsmodels.tsa.seasonal import MSTL

        return MSTL(series, periods=periods).fit()

    def acf_pacf(series, nlags=60):
        from statsmodels.tsa.stattools import acf, pacf

        v = series.dropna().to_numpy(float)
        a = acf(v, nlags=nlags, fft=True)
        p = pacf(v, nlags=nlags, method="ywm")
        ci = 1.96 / np.sqrt(len(v))
        return a, p, ci

    def style(fig, height=380):
        fig.update_layout(template="plotly_white", height=height,
                          margin=dict(l=60, r=20, t=50, b=40),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                          font=dict(size=12))
        return fig

    y = load_series(TRAIN_FILE)
    H = HORIZON
    return (
        C_ALT,
        C_HIST,
        C_PEAK,
        DEAD_HOURS,
        DOW,
        H,
        TRAIN_FILE,
        acf_pacf,
        go,
        make_subplots,
        mstl_decompose,
        np,
        pd,
        stationarity_report,
        style,
        y,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part 1 / 3 — Framing, data and EDA

    This file covers sections **1, 2 and 3** of the final project and is fully
    self-contained. It recomputes every figure and statistic from
    `data/train_data.csv`. The methodology, backtest and champion model live in
    `4_6_methodology_results_champion.py`; the submission, staffing optimisation
    and robustness extensions live in `7_9_submission_staffing_robustness.py`.
    Each of the three files runs independently.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Business framing

    Glovo runs each city on a **slot-based system: 24 hourly slots per day**. Planning is
    refreshed **weekly**: every Sunday after 23:59 the team must hand operations a forecast
    for **all 168 hours** of the coming week (Mon 00:00 → Sun 23:00). The forecast feeds
    courier headcount, and the asymmetric cost:

    - **Over-estimate →** idle couriers and wasted resources.
    - **Under-estimate →** delivery delays.


    Because planning is performed on a weekly basis, the forecasting task is to predict the next 168 hourly observations.

    *The forecasting methodology and evaluation protocol are presented in Part 2; here, we focus on understanding the seasonality and stationarity characteristics of the series.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Data and quality
    """)
    return


@app.cell
def _(TRAIN_FILE, mo, pd, y):
    _raw = pd.read_csv(TRAIN_FILE, parse_dates=["time"])
    _miss = pd.date_range(y.index.min(), y.index.max(), freq="h").difference(
        pd.DatetimeIndex(_raw["time"]))
    _n = len(y)
    _mean = round(float(y.mean()), 1)
    _median = float(y.median())
    _max = float(y.max())
    _zero = float((y == 0).mean())
    _weeks = round(_n / 168, 1)
    mo.md(
        f"""
        One city (**BCN**), **{_n:,} hourly observations** spanning
        **{y.index.min()} → {y.index.max()}** (~**{_weeks} weeks**). Mean
        **{_mean}** orders/hour, median **{_median:.0f}**, max **{_max:.0f}**.
        **{_zero:.0%} of hours are exactly zero** (the platform is shut overnight).

        **Data quality.** The raw file has **{len(_miss)} missing hours** — all overnight blocks
        (e.g. {', '.join(str(t.date()) for t in _miss[:3].normalize().unique())}…). Because lags
        and seasonal differences require a **gap-free hourly grid**, we reindex onto a regular
        hourly index and impute: short gaps by time-interpolation, the rest by the same
        hour-of-week median, finally 0. All overnight gaps resolve to 0 (correct). Values are
        clipped at ≥ 0. *(See `load_series` in the engine cell at the top.)*
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Exploratory data analysis

    ### 3.1 The series, trend and stationarity
    The first approach is a **visual stationarity check**. We aim to answer three questions: *is the mean
    stable? is the variance stable? are different periods exchangeable?*

    Below, we observe that the **level increases throughout 2021**, and the **seasonal fluctuations appear to become larger over time**. This indicates that the mean is not stable and suggests heteroscedasticity, implying that the **series is non-stationary in levels**.
    """)
    return


@app.cell
def _(C_HIST, C_PEAK, go, style, y):
    _fig = go.Figure()
    _fig.add_scatter(x=y.index, y=y.values, name="hourly orders",
                     line=dict(color=C_HIST, width=0.6), opacity=0.6)
    _fig.add_scatter(x=y.index, y=y.rolling(168).mean(), name="7-day rolling mean",
                     line=dict(color=C_PEAK, width=2))
    _fig.update_layout(title="Hourly orders, Feb 2021 – Jan 2022 (level rises, daily band widens)",
                       yaxis_title="orders", hovermode="x unified")
    style(_fig, 380)
    return


@app.cell
def _(mo, stationarity_report, y):
    _s = {
        "levels": stationarity_report(y, regression="ct"),
        "first_diff": stationarity_report(y.diff().dropna(), regression="c"),
        "seasonal_diff_168": stationarity_report(y.diff(168).dropna(), regression="c"),
    }

    def _row(name, d):
        return f"| {name} | {d['adf_p']:.3f} | {d['kpss_p']:.3f} | **{d['verdict']}** |"

    mo.md(
        f"""
        Formal confirmation with the course's **paired ADF + KPSS** test (ADF H₀ = unit root;
        KPSS H₀ = stationary), read jointly:

        | series | ADF p | KPSS p | verdict |
        |---|---:|---:|---|
        {_row("levels (trend)", _s["levels"])}
        {_row("first difference Δ₁", _s["first_diff"])}
        {_row("seasonal difference Δ₁₆₈", _s["seasonal_diff_168"])}

        The levels are non-stationary: a seasonal difference at lag 168 substantially removes the 
        dominant weekly seasonal structure and yields a series that appears stationary according 
        to both ADF and KPSS. These results suggest that **seasonal methods** and models that incorporate 
        **differencing** are particularly appropriate candidates for forecasting this series.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2 Two seasonal cycles: a daily shape and a weekly shape
    Orders follow a sharp **intraday** rhythm: dead 00:00–05:00, a **lunch peak (~13:00)** and a
    larger **dinner peak (~21:00, Spanish dinner time)**. Also we observe a **weekly** rhythm with a
    **Friday/weekend lift**. The boxplots show the spread (not just the mean) grows at the peaks, finding that peak periods are *wider*, not only higher.
    """)
    return


@app.cell
def _(C_HIST, DOW, make_subplots, mo, style, y):
    _bh = y.groupby(y.index.hour).mean()
    _bd = y.groupby(y.index.dayofweek).mean()
    _f1 = make_subplots(rows=1, cols=2, subplot_titles=("Mean orders by hour of day", "Mean orders by day of week"))
    _f1.add_scatter(x=_bh.index, y=_bh.values, mode="lines+markers", line=dict(color=C_HIST), row=1, col=1, showlegend=False)
    _f1.add_vrect(x0=-0.5, x1=5.5, fillcolor="grey", opacity=0.12, line_width=0, row=1, col=1)
    _f1.add_bar(x=DOW, y=_bd.values, marker_color="#DD8452", row=1, col=2, showlegend=False)
    _f1.update_xaxes(title_text="hour", row=1, col=1)

    _f2 = make_subplots(rows=1, cols=2, subplot_titles=("Orders by hour (distribution)", "Orders by weekday (distribution)"))
    _f2.add_box(x=y.index.hour, y=y.values, marker_color=C_HIST, row=1, col=1, showlegend=False, boxpoints=False)
    _f2.add_box(x=[DOW[d] for d in y.index.dayofweek], y=y.values, marker_color="#DD8452", row=1, col=2, showlegend=False, boxpoints=False)
    _f2.update_xaxes(title_text="hour", row=1, col=1)
    mo.vstack([style(_f1), style(_f2)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Crossing the two cycles gives a **168-cell weekly fingerprint** (hour × weekday). This single
    object is, as we will see, almost the entire forecastable signal.
    """)
    return


@app.cell
def _(DOW, go, pd, style, y):
    _piv = (pd.DataFrame({"o": y.values, "h": y.index.hour, "d": y.index.dayofweek})
            .groupby(["h", "d"])["o"].mean().unstack("d"))
    _fig = go.Figure(go.Heatmap(z=_piv.values, x=DOW, y=list(_piv.index), colorscale="Viridis", colorbar_title="orders"))
    _fig.update_layout(title="Weekly fingerprint: mean orders by hour × weekday", xaxis_title="weekday", yaxis_title="hour")
    style(_fig, 460)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.3 Growth trend and a stable seasonal shape
    Weekly totals climb from ~11k to ~16.5k orders (left). Overlaying the **last eight weeks** by
    hour-of-week (right) shows the **shape is stable while the level drifts up**. With this we are modeling the recurring shape, but rescaling it to
    the most recent level.
    """)
    return


@app.cell
def _(C_ALT, C_HIST, H, make_subplots, style, y):
    _wk = y.resample("W").sum()
    _fig = make_subplots(rows=1, cols=2, subplot_titles=("Weekly total orders (growth trend)",
                                                         "Last 8 weeks overlaid by hour-of-week"))
    _fig.add_scatter(x=_wk.index, y=_wk.values, mode="lines+markers", line=dict(color=C_ALT), row=1, col=1, showlegend=False)
    for _i in range(8, 0, -1):
        _seg = y.iloc[-_i * H:len(y) - (_i - 1) * H]
        _fig.add_scatter(x=list(range(H)), y=_seg.values, line=dict(color=C_HIST, width=0.8),
                         opacity=0.35 + 0.06 * (8 - _i), row=1, col=2, showlegend=False)
    _fig.update_xaxes(title_text="hour of week (0=Mon 00:00)", row=1, col=2)
    style(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.4 Decomposition: separating trend, daily and weekly seasonality
    An **MSTL** decomposition (multiple-seasonal STL, periods 24h and 168h) cleanly separates a
    slow trend, a stable daily cycle, a weekly cycle, and a residual. The residual is small and
    spiky around the peaks, which hints us that the *shape* is highly predictable but the *peak heights*
    carry irreducible noise (This is important for SMAPE).
    """)
    return


@app.cell
def _(C_HIST, make_subplots, mstl_decompose, style, y):
    _res = mstl_decompose(y)
    _seas = _res.seasonal
    _comps = [("observed", y.values), ("trend", _res.trend.values),
              ("seasonal 24h", _seas.iloc[:, 0].values), ("seasonal 168h", _seas.iloc[:, 1].values),
              ("residual", _res.resid.values)]
    _fig = make_subplots(rows=5, cols=1, shared_xaxes=True, subplot_titles=[c[0] for c in _comps], vertical_spacing=0.03)
    for _i, _cv in enumerate(_comps, 1):
        _fig.add_scatter(x=y.index, y=_cv[1], line=dict(color=C_HIST, width=0.6), row=_i, col=1, showlegend=False)
    _fig.update_layout(title="MSTL decomposition (daily 24h + weekly 168h seasonality)")
    style(_fig, 720)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.5 Autocorrelation
    The ACF reveals strong seasonal structure with significant peaks at **lags 24 and 168**, corresponding to daily and weekly cycles. The PACF suggests that part of this **seasonal dependence persists** after controlling for shorter lags. These findings support the use of 24-hour and 168-hour lag features and seasonal terms in subsequent forecasting models.
    """)
    return


@app.cell
def _(C_ALT, C_HIST, C_PEAK, acf_pacf, make_subplots, style, y):
    _a, _p, _ci = acf_pacf(y, nlags=180)
    _fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF (spikes at 24, 168)", "PACF"))
    _fig.add_bar(x=list(range(len(_a))), y=_a, marker_color=C_HIST, row=1, col=1, showlegend=False)
    _fig.add_bar(x=list(range(len(_p))), y=_p, marker_color=C_ALT, row=1, col=2, showlegend=False)
    for _col in (1, 2):
        for _sgn in (_ci, -_ci):
            _fig.add_hline(y=_sgn, line=dict(color=C_PEAK, dash="dash", width=1), row=1, col=_col)
        _fig.update_xaxes(title_text="lag (h)", row=1, col=_col)
    style(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.6 The zero envelope and the distribution
    ~32% of hours are **structurally zero** (00:00–05:00 always; 23:00 almost always). This is
    deterministic, not random, so we **force those hours to 0** in post-processing and clip
    negatives. The awake-hour distribution is right-skewed with a long tail at the dinner peak.
    Crucially, **SMAPE is dominated by the small low-volume "ramp" hours** (06:00–08:00): a tiny
    absolute error there is a large *percentage* error, so a model that is merely smooth-and-noisy
    in the morning pays a heavy SMAPE price.
    """)
    return


@app.cell
def _(C_ALT, C_HIST, DEAD_HOURS, make_subplots, np, style, y):
    _awake = y[~np.isin(y.index.hour, DEAD_HOURS)]
    _zsh = (y == 0).groupby(y.index.hour).mean()
    _fig = make_subplots(rows=1, cols=2, subplot_titles=("Distribution of orders (awake hours)",
                                                         "Share of hours that are exactly zero"))
    _fig.add_histogram(x=_awake.values, nbinsx=60, marker_color=C_HIST, row=1, col=1, showlegend=False)
    _fig.add_bar(x=_zsh.index, y=_zsh.values, marker_color=C_ALT, row=1, col=2, showlegend=False)
    _fig.update_xaxes(title_text="orders", row=1, col=1)
    _fig.update_xaxes(title_text="hour", row=1, col=2)
    style(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What the EDA dictates
    1. **Seasonality is the signal** (ACF ≈ 0.88 at 24h and 168h) → seasonal methods should be hard
       to beat; *naive heuristics are the mandatory first baseline*.
    2. **The overnight envelope is deterministic** → force 00–05 to 0; free accuracy.
    3. **There is real growth** → the forecast must be rescaled to the **recent** level, not a
       year-ago level.
    4. **Peaks are spiky and the metric punishes noisy low hours** → smooth global models
       (Fourier/ARIMA) and variance-chasing ML risk *losing* to a robust seasonal estimate.

    *Continue in `4_6_methodology_results_champion.py` for the methodology, the 12-week
    rolling-origin backtest, and the champion selection.*
    """)
    return


if __name__ == "__main__":
    app.run()
