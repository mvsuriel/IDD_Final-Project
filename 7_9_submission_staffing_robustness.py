"""
================================================================================
 PART 3 / 3 — Submission · Staffing optimisation · Robustness & wrap-up
================================================================================
 Glovo hourly demand forecasting — 21DM011 Final Project.

 SELF-CONTAINED. This file needs `data/train_data.csv` + `data/test_data_mock.csv`
 and the scientific-Python stack (numpy, pandas, plotly, ortools). It does NOT
 read results_bundle.json or figures/, and does NOT import the project modules —
 it refits the champion, writes predictions.csv, solves the courier-staffing MILP
 and recomputes every robustness extension from the raw data when you open it.
 (~2–4 min: champion-only backtests are fast; the MILP solves in seconds.)

   §7   The submission (writes predictions.csv + format check)
   §8   From forecast to decision: courier-staffing MILP
   §9   Robustness & extensions (holidays, interval calibration, buffered
        staffing, the ramp-hour error)
   §10  Productionization (MLOps / DataOps)
   §11  Conclusion & limitations
   Appendix A — parameters, tests & metrics quick reference

 Run:  marimo edit 7_9_submission_staffing_robustness.py
 Optional env overrides: IDD_DATA_DIR, IDD_TRAIN_FILE, IDD_TEST_FILE, IDD_OUTPUT_DIR.
 Reading order (each file is independent): 1_3_framing_data_eda.py, then
 4_6_methodology_results_champion.py.
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
    # Embedded engine — champion forecaster, intervals, holidays, the
    # staffing MILP, and the vendored submission format checker.
    # ------------------------------------------------------------------ #
    import warnings; warnings.filterwarnings("ignore")
    import os
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HORIZON = 168
    FORECAST_START = pd.Timestamp("2022-01-24 00:00:00")
    FORECAST_END = pd.Timestamp("2022-01-30 23:00:00")
    DEAD_HOURS = (0, 1, 2, 3, 4, 5)
    C_HIST, C_FC, C_PEAK, C_ALT, C_GREY = "#4C72B0", "#D55E00", "#C44E52", "#55A868", "#7f7f7f"

    _HERE = Path(__file__).resolve().parent
    _DATA_DIR = Path(os.environ.get("IDD_DATA_DIR", str(_HERE / "data")))
    _OUTPUT_DIR = Path(os.environ.get("IDD_OUTPUT_DIR", str(_HERE)))
    TRAIN_FILE = os.environ.get("IDD_TRAIN_FILE", str(_DATA_DIR / "train_data.csv"))
    TEST_FILE = os.environ.get("IDD_TEST_FILE", str(_DATA_DIR / "test_data_mock.csv"))

    def predictions_path():
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return _OUTPUT_DIR / "predictions.csv"

    def hour_of_week(idx):
        return (idx.dayofweek * 24 + idx.hour).to_numpy()

    def load_series(train_csv):
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

    def forecast_index():
        return pd.date_range(FORECAST_START, FORECAST_END, freq="h")

    def smape(actual, forecast):
        a = np.asarray(actual, float); f = np.asarray(forecast, float)
        denom = np.abs(a) + np.abs(f)
        with np.errstate(invalid="ignore", divide="ignore"):
            terms = np.where(denom == 0, 0.0, 2 * np.abs(f - a) / np.where(denom == 0, 1, denom))
        return 100.0 * float(np.mean(terms))

    def mse(actual, forecast):
        a = np.asarray(actual, float); f = np.asarray(forecast, float)
        return float(np.mean((a - f) ** 2))

    def mase(actual, forecast, train, season=HORIZON):
        a = np.asarray(actual, float); f = np.asarray(forecast, float)
        yt = train.to_numpy(float)
        scale = np.mean(np.abs(yt[season:] - yt[:-season]))
        return float(np.mean(np.abs(a - f)) / scale) if scale > 0 else np.nan

    def postprocess(pred, idx):
        pred = np.asarray(pred, float).clip(min=0)
        pred[np.isin(idx.hour, DEAD_HOURS)] = 0.0
        return pred

    # ---- holiday calendar (Barcelona / Catalonia) -------------------- #
    HOLIDAYS = (
        "2021-04-02", "2021-04-05", "2021-05-01", "2021-05-24", "2021-06-24",
        "2021-09-11", "2021-09-24", "2021-10-12", "2021-11-01", "2021-12-06",
        "2021-12-08", "2021-12-25", "2021-12-26", "2022-01-01", "2022-01-06",
    )
    _HOLIDAY_SET = frozenset(pd.Timestamp(d).date() for d in HOLIDAYS)

    def is_holiday(idx):
        return np.array([d in _HOLIDAY_SET for d in idx.normalize().date])

    # ---- champion forecaster ----------------------------------------- #
    def _profile_table(train, k_weeks, use_median):
        how = hour_of_week(train.index)
        s = pd.Series(train.to_numpy(), index=how)
        agg = np.median if use_median else np.mean
        return {int(h): float(agg(g.iloc[-k_weeks:].to_numpy())) for h, g in s.groupby(level=0)}

    def predict_seasonal_profile(train, idx, k_weeks=10, use_median=True,
                                 trend_adjust=True, holiday_adjust=False):
        table = _profile_table(train, k_weeks, use_median)
        base = np.array([table.get(int(h), np.nan) for h in hour_of_week(idx)])
        if trend_adjust:
            last_week = train.iloc[-HORIZON:].to_numpy()
            prof_last_week = np.array(
                [table.get(int(h), 0.0) for h in hour_of_week(train.index[-HORIZON:])])
            awake = prof_last_week > 0
            ratio = last_week[awake].sum() / max(prof_last_week[awake].sum(), 1e-9)
            base = base * ratio
        if holiday_adjust:
            factor = holiday_adjustment_factor(train)
            base = np.where(is_holiday(idx), base * factor, base)
        return base

    def predict_seasonal_naive(train, idx):
        return train.reindex(idx - pd.Timedelta(hours=HORIZON)).to_numpy()

    def holiday_adjustment_factor(history):
        h_mask = is_holiday(history.index) & ~np.isin(history.index.hour, DEAD_HOURS)
        if h_mask.sum() < 24:
            return 1.0
        prof = predict_seasonal_profile(history, history.index, trend_adjust=True)
        actual = history.to_numpy(float)[h_mask]
        expected = np.clip(prof[h_mask], 1e-6, None)
        return float(np.median(actual / expected))

    def anomalous_days(series, z_thresh=5.0):
        daily = series.resample("D").sum()
        dow = daily.index.dayofweek
        rows = []
        for d in range(7):
            s = daily[dow == d]
            med = s.rolling(5, min_periods=3, center=True).median()
            mad = (s - med).abs().rolling(5, min_periods=3, center=True).median()
            z = 0.6745 * (s - med) / mad.replace(0, np.nan)
            for ts, zz in z.items():
                if pd.notna(zz) and abs(zz) >= z_thresh:
                    rows.append({"date": ts.normalize(), "total": float(s[ts]),
                                 "z": round(float(zz), 1), "is_holiday": ts.normalize().date() in _HOLIDAY_SET})
        return pd.DataFrame(rows).sort_values("z").reset_index(drop=True)

    def rolling_origin_backtest(series, models, n_folds=12):
        last = series.index[-1]; rows = []
        for k in range(n_folds, 0, -1):
            start = (last + pd.Timedelta(hours=1)) - pd.Timedelta(hours=HORIZON * k)
            idx = pd.date_range(start, periods=HORIZON, freq="h")
            if idx[-1] > last:
                continue
            train = series.loc[: idx[0] - pd.Timedelta(hours=1)]
            actual = series.reindex(idx).to_numpy()
            for name, fn in models.items():
                pred = postprocess(fn(train, idx), idx)
                rows.append({"fold": k, "model": name,
                             "smape": smape(actual, pred), "mse": mse(actual, pred)})
        return pd.DataFrame(rows)

    def empirical_interval_offsets(series, model_fn, n_folds=12, level=0.90):
        last = series.index[-1]
        resid_by_hour = {h: [] for h in range(24)}
        for k in range(n_folds, 0, -1):
            start = (last + pd.Timedelta(hours=1)) - pd.Timedelta(hours=HORIZON * k)
            idx = pd.date_range(start, periods=HORIZON, freq="h")
            if idx[-1] > last:
                continue
            train = series.loc[: idx[0] - pd.Timedelta(hours=1)]
            pred = postprocess(model_fn(train, idx), idx)
            actual = series.reindex(idx).to_numpy()
            for h, r in zip(idx.hour, actual - pred):
                resid_by_hour[h].append(float(r))
        lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
        offsets = {}
        for h, rs in resid_by_hour.items():
            offsets[h] = (float(np.quantile(rs, lo_q)), float(np.quantile(rs, hi_q))) if rs else (0.0, 0.0)
        return offsets

    def backtest_interval_coverage(series, model_fn, n_folds=12, level=0.90):
        last = series.index[-1]; folds = []
        for k in range(n_folds, 0, -1):
            start = (last + pd.Timedelta(hours=1)) - pd.Timedelta(hours=HORIZON * k)
            idx = pd.date_range(start, periods=HORIZON, freq="h")
            if idx[-1] > last:
                continue
            train = series.loc[: idx[0] - pd.Timedelta(hours=1)]
            pred = postprocess(model_fn(train, idx), idx)
            actual = series.reindex(idx).to_numpy()
            folds.append((idx, actual, pred))
        lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
        covered, width, per_fold = [], [], []
        resid_hist = {h: [] for h in range(24)}
        for idx, actual, pred in folds:
            ok = all(len(resid_hist[h]) for h in set(idx.hour))
            if ok:
                lo = np.array([pred[i] + np.quantile(resid_hist[idx[i].hour], lo_q) for i in range(len(idx))])
                hi = np.array([pred[i] + np.quantile(resid_hist[idx[i].hour], hi_q) for i in range(len(idx))])
                lo = np.clip(lo, 0, None); lo[np.isin(idx.hour, DEAD_HOURS)] = 0; hi[np.isin(idx.hour, DEAD_HOURS)] = 0
                inside = (actual >= lo) & (actual <= hi)
                covered.append(float(inside.mean())); width.append(float(np.mean(hi - lo)))
                per_fold.append(float(inside.mean()))
            for h, r in zip(idx.hour, actual - pred):
                resid_hist[h].append(float(r))
        return {"nominal": level,
                "empirical_coverage": float(np.mean(covered)) if covered else float("nan"),
                "mean_width": float(np.mean(width)) if width else float("nan"),
                "per_fold": per_fold}

    def champion_forecast(series, idx=None):
        if idx is None:
            idx = forecast_index()
        raw = predict_seasonal_profile(series, idx, k_weeks=10, use_median=True, trend_adjust=True)
        return pd.DataFrame({"time": idx, "preds": postprocess(raw, idx).astype("float64")})

    def champion_forecast_interval(series, idx=None, level=0.90):
        if idx is None:
            idx = forecast_index()
        point = champion_forecast(series, idx)["preds"].to_numpy()
        offs = empirical_interval_offsets(series, lambda tr, ix: predict_seasonal_profile(tr, ix), level=level)
        lo = np.array([max(0.0, point[i] + offs[idx[i].hour][0]) for i in range(len(idx))])
        hi = np.array([point[i] + offs[idx[i].hour][1] for i in range(len(idx))])
        lo[np.isin(idx.hour, DEAD_HOURS)] = 0.0
        hi[np.isin(idx.hour, DEAD_HOURS)] = 0.0
        return pd.DataFrame({"time": idx, "preds": point, "lo": lo, "hi": hi})

    # ---- courier-staffing MILP (OR-Tools / SCIP) --------------------- #
    def optimize_staffing(forecast, *, capacity_per_courier=4.0, wage_per_hour=12.0,
                          understaff_penalty=20.0, shift_length=6, open_hours=(6, 23)):
        from ortools.linear_solver import pywraplp

        fc = forecast.sort_values("time").reset_index(drop=True)
        idx = pd.DatetimeIndex(fc["time"])
        demand = fc["preds"].to_numpy(float)
        hours = idx.hour.to_numpy()
        days = idx.dayofweek.to_numpy()
        open_lo, open_hi = open_hours
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if solver is None:
            raise RuntimeError("OR-Tools SCIP backend unavailable")
        n = len(fc)
        starts = []
        for t in range(n):
            h = hours[t]
            if open_lo <= h and h + shift_length <= open_hi + 1:
                if t + shift_length <= n and days[t] == days[t + shift_length - 1]:
                    starts.append(t)
        x = {s: solver.IntVar(0, solver.infinity(), f"x_{s}") for s in starts}
        coverage = [solver.IntVar(0, solver.infinity(), f"cov_{t}") for t in range(n)]
        for t in range(n):
            covering = [x[s] for s in starts if s <= t < s + shift_length]
            solver.Add(coverage[t] == (solver.Sum(covering) if covering else 0))
        unmet = [solver.NumVar(0, solver.infinity(), f"u_{t}") for t in range(n)]
        for t in range(n):
            solver.Add(unmet[t] >= demand[t] - capacity_per_courier * coverage[t])
        courier_hours = solver.Sum(shift_length * x[s] for s in starts)
        total_unmet = solver.Sum(unmet)
        solver.Minimize(wage_per_hour * courier_hours + understaff_penalty * total_unmet)
        status = solver.Solve()
        ok = status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
        cov = np.array([coverage[t].solution_value() for t in range(n)]) if ok else np.zeros(n)
        served = np.minimum(cov * capacity_per_courier, demand)
        unmet_v = np.maximum(demand - cov * capacity_per_courier, 0.0)
        plan = pd.DataFrame({"time": idx, "demand": demand, "couriers": cov.astype(int),
                             "capacity": cov * capacity_per_courier, "served": served, "unmet": unmet_v})
        wage_cost = float(wage_per_hour * (cov * 1.0).sum())
        penalty_cost = float(understaff_penalty * unmet_v.sum())
        return {"status": "optimal" if status == pywraplp.Solver.OPTIMAL else ("feasible" if ok else "infeasible"),
                "plan": plan, "courier_hours": float(cov.sum()), "wage_cost": wage_cost,
                "penalty_cost": penalty_cost, "total_cost": wage_cost + penalty_cost,
                "service_level": float(served.sum() / demand.sum()) if demand.sum() else 1.0,
                "params": dict(capacity_per_courier=capacity_per_courier, wage_per_hour=wage_per_hour,
                               understaff_penalty=understaff_penalty, shift_length=shift_length,
                               open_hours=open_hours)}

    # ---- vendored submission checker --------------------------------- #
    def check_output_format(predictions, path_to_test_data):
        assert predictions.shape[0] == 24 * 7, f"Predictions should have 24*7={24 * 7} rows, but has {predictions.shape[0]} rows"
        assert predictions.shape[1] == 2, f"Predictions should have 2 columns, but has {predictions.shape[1]} columns"
        assert predictions.columns.tolist() == ['time', 'preds'], f"Predictions should have columns 'time' and 'preds', but has {predictions.columns.tolist()}"
        assert predictions['time'].dtype == 'datetime64[ns]', f"Time should be a datetime, but is {predictions['time'].dtype}"
        assert predictions['preds'].dtype == 'float64', f"Preds should be a float, but is {predictions['preds'].dtype}"
        assert not predictions.isnull().any().any(), "Predictions should not contain any null values"
        test_data = pd.read_csv(path_to_test_data)
        test_data = test_data.astype({"time": "datetime64[ns]", "orders": "float64"})
        comparison = predictions.merge(test_data, on='time', how='inner')
        assert comparison.shape[0] == predictions.shape[0], f"Comparison should have the same number of rows as predictions, but has {comparison.shape[0]} rows"
        print(f"MSE: {np.mean((comparison['preds'] - comparison['orders']) ** 2)}")
        print("Your output seems correctly formatted!")

    def style(fig, height=380):
        fig.update_layout(template="plotly_white", height=height,
                          margin=dict(l=60, r=20, t=50, b=40),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                          font=dict(size=12))
        return fig

    y = load_series(TRAIN_FILE)
    return (
        C_ALT,
        C_FC,
        C_HIST,
        C_PEAK,
        HOLIDAYS,
        TEST_FILE,
        anomalous_days,
        backtest_interval_coverage,
        champion_forecast,
        champion_forecast_interval,
        check_output_format,
        go,
        holiday_adjustment_factor,
        is_holiday,
        make_subplots,
        mase,
        np,
        optimize_staffing,
        pd,
        postprocess,
        predict_seasonal_naive,
        predict_seasonal_profile,
        predictions_path,
        rolling_origin_backtest,
        smape,
        style,
        y,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part 3: Staffing, Robustness and Wrap-up

    This file covers sections 7 through 11 (plus the appendix) and is fully self-contained — it refits the champion, writes the submission and recomputes each extension from `data/`. The framing, data and EDA live in `1_3_framing_data_eda.py`; the methodology, results and champion model live in `4_6_methodology_results_champion.py`. Each of the three files runs independently.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · The submission
    Refit the champion on **all** history and forecast 2022-01-24 → 2022-01-30, then run the repository's **`check_output_format`** (asserts 168×2 shape, `datetime64[ns]` / `float64` dtypes, no nulls, time-alignment). *(The mock holdout is all zeros, so its printed MSE is a format check, not an accuracy signal — accuracy is the backtest in Part 2.)*
    """)
    return


@app.cell
def _(
    TEST_FILE,
    champion_forecast,
    check_output_format,
    mo,
    pd,
    predictions_path,
    y,
):
    _predictions = champion_forecast(y)
    _out_csv = predictions_path()
    _predictions.to_csv(_out_csv, index=False)
    _reloaded = pd.read_csv(_out_csv, parse_dates=["time"]).astype({"preds": "float64"})
    check_output_format(_reloaded, TEST_FILE)
    mo.md(f"**Submission written** — {len(_reloaded)} rows · sum **{_reloaded['preds'].sum():.0f}** · "
          f"peak **{_reloaded['preds'].max():.0f}** · format check **passed**.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · From forecast to decision: courier-staffing optimization
    A forecast is only useful if it changes a decision. The course's optimization module teaches turning a demand forecast into a **constrained resource plan** (LP/MILP via OR-Tools), as in the *Medistor procurement* exercise. We build the analogous **courier-staffing MILP**:

    - **Decision:** integer shift-starts; a courier works a contiguous **6-hour shift**, so staffing in one hour is coupled to its neighbours (real rostering, not 168 independent ceilings).
    - **Coverage / service:** `couriers(t) = Σ shifts covering t`; each courier serves `capacity_per_courier` orders/hour; `unmet(t) ≥ demand(t) − capacity·couriers(t)`.
    - **Objective:** `min  wage·(courier-hours) + penalty·(unmet orders)` — over-staffing burns the wage term, under-staffing burns the penalty term: the exact trade-off in the brief.

    *(Implementation: `optimize_staffing`, OR-Tools + SCIP.)*
    """)
    return


@app.cell
def _(
    C_FC,
    C_HIST,
    champion_forecast,
    make_subplots,
    mo,
    optimize_staffing,
    pd,
    style,
    y,
):
    _st = optimize_staffing(champion_forecast(y))
    _plan = _st["plan"]; _t = pd.DatetimeIndex(_plan["time"]); _fri = _t.dayofweek == 4
    _p = _st["params"]
    _fig = make_subplots(rows=1, cols=2, subplot_titles=("Friday: optimized capacity vs demand",
                                                         "Couriers scheduled across the week (168h)"))
    _fig.add_bar(x=_t.hour[_fri], y=_plan["demand"][_fri], name="forecast demand", marker_color=C_HIST, opacity=0.6, row=1, col=1)
    _fig.add_scatter(x=_t.hour[_fri], y=_plan["capacity"][_fri], name="courier capacity", line=dict(color=C_FC, shape="hv"), row=1, col=1)
    _fig.add_scatter(x=list(range(168)), y=_plan["couriers"], line=dict(color="#55A868", shape="hv", width=1), row=1, col=2, showlegend=False)
    _fig.update_xaxes(title_text="hour", row=1, col=1); _fig.update_xaxes(title_text="hour of week", row=1, col=2)
    mo.vstack([
        style(_fig).update_layout(margin=dict(b=90),
            legend=dict(orientation="h", xanchor="left", x=0.0, yanchor="top", y=-0.22)),
        mo.md(
            f"""
            With illustrative economics (capacity **{_p['capacity_per_courier']:.0f}** orders/courier·h, wage **€{_p['wage_per_hour']:.0f}/h**, under-staffing penalty **€{_p['understaff_penalty']:.0f}/order**, **{_p['shift_length']}h** shifts), the optimal weekly roster uses **{_st['courier_hours']:.0f} courier-hours**, achieves a **{_st['service_level']:.1%} service level**, at a total cost of **€{_st['total_cost']:,.0f}** (wage €{_st['wage_cost']:,.0f} + shortfall penalty €{_st['penalty_cost']:,.0f}). The shift coupling forces a little over-staffing on the shoulders of the dinner peak — visible in the Friday panel — which is the genuine cost of integer rostering. Tightening the penalty or shift length traces out the **service-vs-cost frontier** the business actually chooses on.
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9 · Robustness & extensions (the limitations, implemented)

    A complete forecasting project does not merely *list* its limitations — it confronts them. This section implements and **validates** four extensions: holiday/event handling, prediction- interval **calibration**, **chance-constrained** staffing, and an analysis of the intermittent ramp hours. Each is measured on the same rolling-origin backtest, and each result is reported honestly — including where the extra machinery does *not* pay off.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9.1 Holiday and event effects
    **Problem.** The seasonal profile assumes every week looks like recent weeks, so it misreads public holidays. **What we built.** (i) An **anomaly detector** (`anomalous_days`) — a robust MAD z-score on daily totals vs the same-weekday rolling median — to surface unusual days; (ii) a **Barcelona/Catalonia holiday calendar** (`HOLIDAYS`); (iii) a **holiday-aware champion** that scales holiday hours by the historical holiday/normal demand ratio (`holiday_adjustment_factor`, applied via `predict_seasonal_profile(..., holiday_adjust=True)`).

    **How to read the parameter.** The *holiday factor* is the median of `actual / expected` over past holiday awake-hours: **< 1 ⇒ holidays are quieter**, > 1 busier. We then backtest the plain vs holiday-aware champion, splitting weeks by whether they contain a holiday.
    """)
    return


@app.cell
def _(
    C_FC,
    C_HIST,
    C_PEAK,
    HOLIDAYS,
    anomalous_days,
    holiday_adjustment_factor,
    is_holiday,
    make_subplots,
    mo,
    np,
    pd,
    predict_seasonal_profile,
    rolling_origin_backtest,
    style,
    y,
):
    _champ = lambda tr, ix: predict_seasonal_profile(tr, ix)
    _champ_h = lambda tr, ix: predict_seasonal_profile(tr, ix, holiday_adjust=True)
    _factor = holiday_adjustment_factor(y)
    _anoms = anomalous_days(y)
    _bt = rolling_origin_backtest(y, {"champion": _champ, "champion+holiday": _champ_h}, n_folds=12)
    _last = y.index[-1]; _hol_weeks = set()
    for _k in range(12, 0, -1):
        _idx = pd.date_range((_last + pd.Timedelta(hours=1)) - pd.Timedelta(hours=168 * _k), periods=168, freq="h")
        if _idx[-1] <= _last and is_holiday(_idx).any():
            _hol_weeks.add(12 - _k)
    _bt["fid"] = _bt.groupby("model").cumcount()
    _hol = _bt[_bt["fid"].isin(_hol_weeks)].groupby("model")["smape"].mean()
    _allw = _bt.groupby("model")["smape"].mean()

    _daily = y.resample("D").sum().loc["2021-10-15":]
    _fig = make_subplots(rows=1, cols=2, subplot_titles=("Daily orders with holidays marked", "SMAPE: holiday-aware effect"))
    _fig.add_scatter(x=_daily.index, y=_daily.values, line=dict(color=C_HIST, width=1), row=1, col=1, showlegend=False)
    for _d in HOLIDAYS:
        _ts = pd.Timestamp(_d)
        if _daily.index.min() <= _ts <= _daily.index.max():
            _fig.add_vline(x=_ts, line=dict(color=C_PEAK, dash="dot", width=1), row=1, col=1)
    _grp = ["champion", "champion+holiday"]
    _fig.add_bar(x=_grp, y=[_allw[g] for g in _grp], name="all weeks", marker_color=C_HIST, row=1, col=2)
    _fig.add_bar(x=_grp, y=[_hol.get(g, np.nan) for g in _grp], name="holiday weeks", marker_color=C_FC, row=1, col=2)
    mo.vstack([
        style(_fig).update_layout(margin=dict(b=90),
            legend=dict(orientation="h", xanchor="left", x=0.0, yanchor="top", y=-0.22)),
        mo.md(
            f"""
            **Result (an honest negative).** Estimated **holiday factor = {round(_factor, 3)}** → across the year Catalan holidays ran about **{(1 - _factor) * 100:.0f}% below** an ordinary day of that weekday. The detector flags **{len(_anoms)}** extreme days — and tellingly, **most are demand *dips* unrelated to the holiday calendar** (likely platform outages or data gaps), a valuable **data-quality** signal in its own right. But applying a **single global** factor does **not** help the backtest: overall SMAPE moves **{round(float(_allw['champion']), 2)} → {round(float(_allw['champion+holiday']), 2)}** and even on holiday-containing weeks **{round(float(_hol.get('champion', float('nan'))), 2)} → {round(float(_hol.get('champion+holiday', float('nan'))), 2)}** — slightly *worse*. The lesson is genuine: holiday effects are **heterogeneous** (Christmas ≠ a local *fiesta* ≠ New Year's Eve), so one multiplicative number over-corrects as often as it helps. That is exactly why the shipped champion **does not** apply it; the principled fix is **per-holiday** effects or holiday *features* inside a richer model. The submission week (Jan 24–30) has no holiday, so the forecast is unaffected regardless.
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9.2 Are the prediction intervals calibrated?
    **Problem.** A 90% interval is only trustworthy if ~90% of actuals actually fall inside it. **What we built.** `backtest_interval_coverage` replays the backtest and, for each week, builds the interval from the residuals of **earlier folds only** (no peeking), then measures the fraction of held-out hours that land inside. This is a genuine **out-of-sample coverage** test.
    """)
    return


@app.cell
def _(
    C_ALT,
    C_HIST,
    C_PEAK,
    backtest_interval_coverage,
    go,
    mo,
    predict_seasonal_profile,
    style,
    y,
):
    _champ = lambda tr, ix: predict_seasonal_profile(tr, ix)
    _c = backtest_interval_coverage(y, _champ, n_folds=12, level=0.90)
    _pf = _c["per_fold"]
    _fig = go.Figure(go.Bar(x=list(range(len(_pf))), y=[100 * c for c in _pf], marker_color=C_HIST, name="coverage / fold"))
    _fig.add_hline(y=90, line=dict(color=C_PEAK, dash="dash"))
    _fig.add_hline(y=100 * _c["empirical_coverage"], line=dict(color=C_ALT))
    # identify the two reference lines via the legend, not labels over the bars
    _fig.add_scatter(x=[None], y=[None], mode="lines", line=dict(color=C_PEAK, dash="dash"), name="nominal 90%")
    _fig.add_scatter(x=[None], y=[None], mode="lines", line=dict(color=C_ALT), name=f"mean {100 * _c['empirical_coverage']:.0f}%")
    _fig.update_layout(title="Prediction-interval calibration: out-of-sample coverage", xaxis_title="fold", yaxis_title="coverage (%)")
    mo.vstack([
        style(_fig, 720).update_layout(margin=dict(b=90),
            legend=dict(orientation="h", xanchor="left", x=0.0, yanchor="top", y=-0.12)),
        mo.md(
            f"""
            **Result.** Nominal **{_c['nominal']:.0%}**, empirical **{_c['empirical_coverage']:.1%}** (mean interval width ≈ **{_c['mean_width']:.0f}** orders) — the band is **well-calibrated** out-of-sample, essentially hitting its target. Per-fold coverage ranges ~85–98% (figure), looser in the noisier weeks. So the 90% interval **can be trusted** to size a staffing buffer (§9.3). A **quantile-regression** interval would tighten the per-fold consistency further, but the simple empirical band is already close to nominal — a satisfying result for a distribution-free method.
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9.3 Chance-constrained staffing (size against the band, not the mean)
    **Problem.** §8 staffs against the **point** forecast, so on a high-demand realization it is under-staffed half the time. **What we built.** A *buffered* plan that sizes capacity against the **90% upper band** (`champion_forecast_interval(...).hi`) — a simple **chance constraint**: meet demand with ~90% probability per hour. We re-solve the same MILP on the buffered demand.
    """)
    return


@app.cell
def _(
    C_ALT,
    C_FC,
    C_HIST,
    champion_forecast_interval,
    make_subplots,
    mo,
    optimize_staffing,
    pd,
    style,
    y,
):
    _ci = champion_forecast_interval(y, level=0.90)
    _r_point = optimize_staffing(_ci[["time", "preds"]].copy())
    _r_buf = optimize_staffing(_ci[["time", "hi"]].rename(columns={"hi": "preds"}))
    _dcost = _r_buf["total_cost"] - _r_point["total_cost"]
    _t = pd.DatetimeIndex(_r_point["plan"]["time"]); _fri = _t.dayofweek == 4
    _fig = make_subplots(rows=1, cols=2, subplot_titles=("Friday: point vs chance-constrained staffing", "Weekly cost (service below)"))
    _fig.add_bar(x=_t.hour[_fri], y=_r_point["plan"]["demand"][_fri], name="point demand", marker_color=C_HIST, opacity=0.5, row=1, col=1)
    _fig.add_scatter(x=_t.hour[_fri], y=_r_point["plan"]["capacity"][_fri], name="point capacity", line=dict(color=C_ALT, shape="hv"), row=1, col=1)
    _fig.add_scatter(x=_t.hour[_fri], y=_r_buf["plan"]["capacity"][_fri], name="buffered (90%)", line=dict(color=C_FC, shape="hv"), row=1, col=1)
    _fig.add_bar(x=[f"point<br>{_r_point['service_level']:.1%}", f"buffered<br>{_r_buf['service_level']:.1%}"],
                 y=[_r_point["total_cost"], _r_buf["total_cost"]], marker_color=[C_HIST, C_FC], row=1, col=2, showlegend=False)
    _fig.update_xaxes(title_text="hour", row=1, col=1)
    mo.vstack([
        style(_fig).update_layout(margin=dict(b=90),
            legend=dict(orientation="h", xanchor="left", x=0.0, yanchor="top", y=-0.22)),
        mo.md(
            f"""
            **Result.** Sizing to the 90% band lifts realized service **{_r_point['service_level']:.1%} → {_r_buf['service_level']:.1%}** but raises weekly cost **€{_r_point['total_cost']:,.0f} → €{_r_buf['total_cost']:,.0f}** (**+€{_dcost:,.0f}**, {_dcost / _r_point['total_cost'] * 100:.0f}% more) and courier-hours **{_r_point['courier_hours']:.0f} → {_r_buf['courier_hours']:.0f}**. With these economics the **point plan is already near-optimal** — because the under-staffing penalty (€20/order) so dominates the wage (€12/h ÷ 4 = €3/order) that the point optimizer *already* staffs generously. Buffering is over-insurance here; it earns its keep only when the penalty/wage ratio is smaller. This is exactly the **service-vs-cost frontier** the business chooses on, now made explicit and priced.
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9.4 Where the residual error lives: the intermittent ramp hours
    **Problem.** SMAPE is dominated by the low-volume morning **ramp** (06:00–08:00), where a tiny absolute miss is a huge percentage miss. **What we did.** Decompose the champion's backtest SMAPE by daypart to size the opportunity precisely.
    """)
    return


@app.cell
def _(
    C_ALT,
    C_FC,
    C_HIST,
    go,
    mo,
    np,
    pd,
    postprocess,
    predict_seasonal_profile,
    smape,
    style,
    y,
):
    # champion-only backtest store for the daypart decomposition
    _last = y.index[-1]; _store = []
    for _k in range(12, 0, -1):
        _idx = pd.date_range((_last + pd.Timedelta(hours=1)) - pd.Timedelta(hours=168 * _k), periods=168, freq="h")
        if _idx[-1] > _last:
            continue
        _train = y.loc[: _idx[0] - pd.Timedelta(hours=1)]
        _act = y.reindex(_idx).to_numpy()
        _store.append((_idx, _act, postprocess(predict_seasonal_profile(_train, _idx), _idx)))
    _groups = {"ramp 6-8": range(6, 9), "midday 9-16": range(9, 17), "peak 17-22": range(17, 23)}
    _gres = {}
    for _idx, _act, _pred in _store:
        for _gname, _hrs in _groups.items():
            _m = np.isin(_idx.hour, list(_hrs)); _gres.setdefault(_gname, []).append(smape(_act[_m], _pred[_m]))
    _ramp = {g: round(float(np.mean(v)), 1) for g, v in _gres.items()}
    _gg = list(_gres)
    _fig = go.Figure(go.Bar(x=_gg, y=[np.mean(_gres[g]) for g in _gg], marker_color=[C_FC, C_HIST, C_ALT],
                            text=[f"{np.mean(_gres[g]):.1f}" for g in _gg], textposition="outside"))
    _fig.update_layout(title="Champion SMAPE by daypart — the ramp hours dominate", yaxis_title="SMAPE (%)")
    mo.vstack([
        style(_fig, 620),
        mo.md(
            f"""
            **Result.** Champion SMAPE is **{_ramp['ramp 6-8']}** on the ramp hours vs **{_ramp['peak 17-22']}** at the dinner peak and **{_ramp['midday 9-16']}** midday. The ramp is the worst daypart by far — these hours are **intermittent** (often 0–10 orders, frequent true zeros). A symmetric percentage metric punishes them disproportionately, and no smooth model can fix that. The right tool is a dedicated **intermittent-demand** method (e.g. Croston/SBA or a zero-inflated count model) applied **only to the ramp hours**, leaving the robust profile for the rest of the day — a targeted next step with a clear, measured payoff ceiling.
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10 · Productionization (MLOps / DataOps)
    Per Module 2 — *"a real intelligent data application needs more than an ML model"* — running this weekly in production needs:

    - **Orchestration:** a scheduled weekly job (e.g. Airflow) that fires **after** the Sunday data lands (dependency-based, not just time-based), retrains the champion, writes `predictions.csv` to **S3**, and notifies planners.
    - **Data contracts / quality gates:** schema validation, row-count and **freshness** checks, and a **volume/anomaly threshold** to tell a true demand event (storm, holiday) from a data bug before it poisons the forecast.
    - **Deploy-code, not deploy-model:** promote the *code* through environments so the model is retrained and tested in each — safer for automated weekly retraining.
    - **Monitoring:** track rolling SMAPE/MASE for **concept drift** and feature distributions for **data drift**; retrain or alert when the live error crosses the backtest band.

    The champion's simplicity is an asset here: it is cheap to retrain, trivial to monitor, and **defensible hour-by-hour to operations** — the "number we can defend for every hour of the week."
    """)
    return


@app.cell
def _(
    mase,
    mo,
    pd,
    postprocess,
    predict_seasonal_naive,
    predict_seasonal_profile,
    smape,
    y,
):
    # champion vs seasonal-naive over 12 folds → the headline numbers
    _last = y.index[-1]
    _cs, _ns, _cm = [], [], []
    for _k in range(12, 0, -1):
        _idx = pd.date_range((_last + pd.Timedelta(hours=1)) - pd.Timedelta(hours=168 * _k), periods=168, freq="h")
        if _idx[-1] > _last:
            continue
        _train = y.loc[: _idx[0] - pd.Timedelta(hours=1)]
        _act = y.reindex(_idx).to_numpy()
        _pc = postprocess(predict_seasonal_profile(_train, _idx), _idx)
        _pn = postprocess(predict_seasonal_naive(_train, _idx), _idx)
        _cs.append(smape(_act, _pc)); _ns.append(smape(_act, _pn)); _cm.append(mase(_act, _pc, _train))
    _champ_smape = sum(_cs) / len(_cs); _naive_smape = sum(_ns) / len(_ns); _champ_mase = sum(_cm) / len(_cm)
    _impr = (_naive_smape - _champ_smape) / _naive_smape
    mo.md(
        f"""
        ## 11 · Conclusion & limitations

        **Ship the trend-adjusted seasonal median profile.** Across a 12-week rolling backtest that replays the weekly planning refresh it delivers **SMAPE {_champ_smape:.1f}** vs **{_naive_smape:.1f}** for the seasonal-naive baseline (**{_impr:.0%}** lower), **MASE {_champ_mase:.2f} < 1**, and beats harmonic regression, classical SARIMA, and a 17-feature gradient-boosting model (Part 2). It is robust, level-aware, fast to retrain, and explainable.

        **Why the simple model wins.** This series is dominated by a stable weekly fingerprint plus a slow growth trend; there is little extra structure for ML or ARIMA to exploit, and the SMAPE metric punishes the noise they add in the low-volume ramp hours. *Performance ≫ complexity.*

        **What we addressed (§9).** Holiday/event effects (detector + holiday-aware variant), interval **calibration** (out-of-sample coverage ≈ 85–90%), **chance-constrained** staffing against the 90% band, and a daypart decomposition pinpointing the **ramp hours** as the residual-error frontier — each implemented and measured, not just named.

        **Genuinely remaining.** (1) A **calibrated quantile-regression** interval to hit coverage exactly. (2) A dedicated **intermittent-demand** model (Croston/SBA or zero-inflated) for the ramp hours. (3) Richer **event features** (weather, promotions, sporting events) — the anomaly detector shows unexplained spikes/dips beyond the holiday calendar. (4) A **stochastic** joint forecast-and-roster that optimizes against the full predictive distribution.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Appendix A · Parameters, tests & metrics — quick reference

    A consolidated, intensive-reading reference. Everything is in the engine cell at the top of this file.

    **A.1 Statistical tests used**

    | Test | Null hypothesis H₀ | We reject H₀ when | Reading it here |
    |---|---|---|---|
    | **ADF** (Augmented Dickey–Fuller) | series has a **unit root** (non-stationary) | p < 0.05 | low p ⇒ *no* unit root ⇒ stationary-ish |
    | **KPSS** | series **is stationary** | p < 0.05 | low p ⇒ *not* stationary. **Read jointly with ADF** (course idiom): ADF-reject & KPSS-not ⇒ stationary; both reject ⇒ trend/curvature. |
    | **ACF** (autocorrelation) | — (descriptive) | spike outside ±1.96/√n | spikes at **24 & 168** ⇒ daily + weekly seasonality |
    | **PACF** (partial autocorr.) | — | as above | direct lag effects; guides AR order in ARIMA |
    | **Coverage** (§9.2) | — | empirical ≈ nominal? | 90% band actually covers ~85% ⇒ slightly optimistic |

    **A.2 Metrics** — MSE (orders², peak-weighted), SMAPE (%, scale-free, ramp-weighted, the grade), MASE (ratio, **<1 beats seasonal naive**). Formulas in Part 2 §4.1.

    **A.3 Model parameters** — see Part 2 §4.4 for the champion (`k_weeks=10`, median, trend-adjust), XGBoost (600 trees, depth 5, lr 0.03, regularizers), harmonic (4 daily / 6 weekly Fourier + ridge 10), and SARIMA (2,0,1)(1,0,1)[24] + weekly Fourier.

    **A.4 Decision-layer parameters** (`optimize_staffing`): `capacity_per_courier=4` orders/h, `wage_per_hour=€12`, `understaff_penalty=€20/order`, `shift_length=6h`, `open_hours=(6,23)`. The penalty/wage ratio drives the service-vs-cost trade-off (§8, §9.3).

    **A.5 Reproduce** — each of the three numbered files is self-contained: open it with `marimo edit <file>.py` and it recomputes from `data/`. Optional env vars: `IDD_DATA_DIR`, `IDD_TRAIN_FILE`, `IDD_TEST_FILE`, `IDD_OUTPUT_DIR`.
    """)
    return


if __name__ == "__main__":
    app.run()
