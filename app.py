"""Streamlit app: Explainable Next-Day Extreme Heat Prediction for Chennai.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
# pandas 2.3 + numpy 2.5 emit a harmless internal DeprecationWarning on Timestamp arithmetic.
warnings.filterwarnings("ignore", message="The 'generic' unit for NumPy timedelta")

from heatpred import config as C  # noqa: E402
from heatpred.data import build_request_url  # noqa: E402
from heatpred.heat import hot_probability  # noqa: E402
from heatpred.explain import explain_row, group_contributions, linear_coefficients, plain_language  # noqa: E402
from heatpred.live import load_combined  # noqa: E402
from heatpred.predict import PredictionUnavailable, latest_issue_date, predict_for_target_date  # noqa: E402

st.set_page_config(page_title="Chennai Next-Day Heat Predictor", page_icon="🌡️", layout="wide")

# Palette: validated categorical slots (blue / orange / aqua) + diverging red-blue for effects.
PRED_COLOR, THR_COLOR, ACT_COLOR, OBS_COLOR = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
UP_COLOR, DOWN_COLOR = "#e34948", "#2a78d6"


# ---------------------------------------------------------------------------
# Loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(C.MODEL_FILE)
    meta = json.loads(C.METADATA_FILE.read_text())
    thr = pd.read_csv(C.THRESHOLD_CSV, index_col="doy").iloc[:, 0]
    bt = pd.read_csv(C.BACKTEST_CSV, parse_dates=["issue_date", "target_date"])
    ref = pd.read_csv(C.REFERENCE_CSV, index_col="target_doy")
    return model, meta, thr, bt, ref


@st.cache_data(ttl=3 * 3600, show_spinner="Fetching the latest NASA POWER data…")
def load_data(_cache_key: str):
    daily, status = load_combined(use_live=True)
    return daily, status


def threshold_for(thr: pd.Series, day: pd.Timestamp) -> float:
    return float(thr.loc[pd.Timestamp(day).dayofyear])


def split_of(day: pd.Timestamp) -> str:
    day = pd.Timestamp(day)
    if day <= pd.Timestamp(C.TRAIN_END):
        return "train"
    if day <= pd.Timestamp(C.VAL_END):
        return "validation"
    if day <= pd.Timestamp(C.STUDY_END):
        return "test"
    return "live"


SPLIT_NOTE = {
    "train": ("⚠️ This date is in the **training period (1981–2012)**. The deployed model learned "
              "from it, so the error here is optimistic. Use 2019 or later for a fair test."),
    "validation": ("⚠️ This date is in the **validation period (2013–2018)**, which the deployed model "
                   "was refitted on after model selection. The error here is optimistic."),
    "test": ("✅ This date is in the **held-out test period (2019–2025)**. The model never saw it "
             "during training, tuning or model selection: a fair check."),
    "live": ("✅ This date is **after the frozen study period (2026 onward)**: genuinely unseen data "
             "retrieved from NASA POWER after the model was built."),
}


# ---------------------------------------------------------------------------
# Reusable display pieces
# ---------------------------------------------------------------------------
def heat_badge(value: float, threshold: float) -> str:
    """Badge for an OBSERVED temperature."""
    if value > threshold:
        return f":red[**Unusually hot for Chennai**] (above {threshold:.1f} °C)"
    return f":green[**Within the usual range**] (at or below {threshold:.1f} °C)"


def hot_chance(pred: float, threshold: float, target: pd.Timestamp) -> float:
    sigma = {int(k): v for k, v in meta["heat_indicator"]["sigma_by_month"].items()}
    return float(hot_probability([pred], [threshold], [pd.Timestamp(target).month], sigma)[0])


def flag_badge(chance: float) -> str:
    cut = meta["heat_indicator"]["cutoff"]
    state = ":red[**Heat flag ON**]" if chance >= cut else ":green[**Heat flag off**]"
    return f"{state}: {chance:.0%} chance of an unusually hot day (the flag is raised at {cut:.0%} or more)"


def contribution_chart(table: pd.DataFrame):
    g = group_contributions(table)
    g["direction"] = np.where(g["contribution"] >= 0, "raises prediction", "lowers prediction")
    return alt.Chart(g).mark_bar().encode(
        x=alt.X("contribution:Q", title="Effect on predicted Tmax (°C)"),
        y=alt.Y("group:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["raises prediction", "lowers prediction"],
                                                       range=[UP_COLOR, DOWN_COLOR]), legend=None),
        tooltip=[alt.Tooltip("group:N"), alt.Tooltip("contribution:Q", format="+.2f")],
    ).properties(height=230)


def recent_chart(daily: pd.DataFrame, thr: pd.Series, target: pd.Timestamp, pred: float,
                 actual: float | None, days: int = 21):
    start = target - pd.Timedelta(days=days)
    obs = daily.loc[start: target - pd.Timedelta(days=1), ["tmax"]].reset_index()
    obs.columns = ["date", "value"]
    obs["series"] = "Observed Tmax (inputs)"
    pts = [{"date": target, "value": pred, "series": "Prediction"}]
    if actual is not None:
        pts.append({"date": target, "value": actual, "series": "Actual Tmax (not used)"})
    pts = pd.DataFrame(pts)
    dates = pd.date_range(start, target)
    t = pd.DataFrame({"date": dates, "value": [threshold_for(thr, d) for d in dates]})
    colors = alt.Scale(domain=["Observed Tmax (inputs)", "Prediction", "Actual Tmax (not used)"],
                       range=[OBS_COLOR, PRED_COLOR, ACT_COLOR])
    line = alt.Chart(obs).mark_line(point=True).encode(
        x=alt.X("date:T", title=None), y=alt.Y("value:Q", title="°C", scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", scale=colors, title=None), tooltip=["date:T", alt.Tooltip("value:Q", format=".1f")])
    points = alt.Chart(pts).mark_point(size=160, filled=True, shape="diamond").encode(
        x="date:T", y="value:Q", color=alt.Color("series:N", scale=colors, title=None),
        tooltip=["series:N", "date:T", alt.Tooltip("value:Q", format=".1f")])
    thr_line = alt.Chart(t).mark_line(strokeDash=[5, 4], color=THR_COLOR, opacity=0.7).encode(
        x="date:T", y="value:Q", tooltip=[alt.Tooltip("value:Q", title="Hot threshold", format=".1f")])
    return (thr_line + line + points).properties(height=300).configure_legend(orient="bottom")


def show_explanation(model, res: dict, thr: pd.Series, chance: float | None = None):
    base, table = explain_row(model, res["features"], reference)
    t_thr = threshold_for(thr, res["target_date"])
    normal = float(table.loc[table.feature == "clim_tmax_target", "value"].iloc[0]) \
        if (table.feature == "clim_tmax_target").any() else None
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("#### Why this prediction?")
        sd = meta["heat_indicator"]["sigma_by_month"][str(res["target_date"].month)]
        for line in plain_language(base, res["prediction"], table, t_thr, normal, chance, sd):
            st.markdown(line)
    with c2:
        st.markdown("#### Effect of each group of inputs")
        st.altair_chart(contribution_chart(table), width="stretch")
    with st.expander("Details: every input feature and its exact contribution"):
        show = table[["feature", "description", "group", "value", "normal for date", "contribution"]].copy()
        show[["value", "normal for date"]] = show[["value", "normal for date"]].round(2)
        show["contribution"] = show["contribution"].round(3)
        st.caption(f"Exact decomposition of the linear model: {base:.2f} °C (prediction for a normal day "
                   f"at this time of year, from average 1981–2018 conditions) + sum of contributions "
                   f"({table.contribution.sum():+.2f} °C) = {base + table.contribution.sum():.2f} °C. "
                   "Each contribution = coefficient × (today's standardised value − normal standardised value).")
        st.dataframe(show, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
try:
    model, meta, thr, backtest, reference = load_artifacts()
except FileNotFoundError:
    st.error("Model files are missing. Run `python scripts/train.py` (or the notebook) first.")
    st.stop()

cache_key = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")
daily, status = load_data(cache_key)

st.title("🌡️ Chennai Next-Day Maximum Temperature: Explainable Predictor")
st.caption("A student climate-change project using NASA POWER gridded data and an interpretable "
           "machine-learning model. **Not an official India Meteorological Department (IMD) forecast "
           "or heat-wave alert.**")

with st.sidebar:
    st.header("Data status")
    st.markdown(f"**Source in use:** {status.source}")
    st.markdown(f"**Latest complete day of data:** {status.last_complete_date:%d %b %Y}")
    if status.retrieved_utc:
        st.markdown(f"**Retrieved (UTC):** {status.retrieved_utc.replace('T', ' ')}")
    if status.message:
        st.warning(status.message)
    st.caption("NASA POWER publishes daily meteorology about 2–3 days after the fact, so the newest "
               "data are always a few days old. Very recent values may later be revised.")
    st.divider()
    st.markdown(f"**Model:** {meta['deployed_model']['name']}")
    tm = meta["test_metrics"][meta["deployed_model"]["name"]]
    st.markdown(f"**Test MAE (2019–2025):** {tm['MAE']:.2f} °C")
    st.caption(f"Location: {C.LATITUDE}° N, {C.LONGITUDE}° E (one ≈50 km NASA POWER grid cell).")

tab_next, tab_hist, tab_perf, tab_about = st.tabs(
    ["Next-day prediction", "Historical backtest", "Model performance", "Data, method & limitations"])

# ---------------------------------------------------------------------------
with tab_next:
    try:
        issue = latest_issue_date(daily)
        target = issue + pd.Timedelta(days=1)
        res = predict_for_target_date(model, daily, target)
        t_thr = threshold_for(thr, target)
        st.subheader(f"Prediction for {target:%A, %d %B %Y}")
        st.info(f"Based on NASA POWER data **through {issue:%d %B %Y}** (the latest complete day; "
                f"source: {status.source}). Because of the 2–3 day publishing delay this 'next day' "
                f"may already be in the past; it is the day after the latest available data, "
                f"not necessarily tomorrow.")
        chance = hot_chance(res["prediction"], t_thr, target)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted maximum temperature", f"{res['prediction']:.1f} °C")
        c2.metric("'Unusually hot' threshold for this date", f"{t_thr:.1f} °C",
                  help="90th percentile of 1981–2012 maximum temperatures within ±7 days of this "
                       "calendar day. Project-defined, not an official criterion.")
        c3.metric("Chance of an unusually hot day", f"{chance:.0%}",
                  help="Uses the model's typical error for this month (from 1981–2012 cross-validation).")
        c4.metric("Typical test-period error (MAE)", f"± {tm['MAE']:.2f} °C")
        st.markdown(flag_badge(chance))
        if res["actual"] is not None:
            st.markdown(f"Recorded value already available: **{res['actual']:.1f} °C** "
                        f"(error {res['prediction'] - res['actual']:+.2f} °C).")
        if res["n_missing_inputs"]:
            st.warning(f"{res['n_missing_inputs']} input value(s) were missing and replaced with "
                       "training medians; treat this prediction with extra caution.")
        st.altair_chart(recent_chart(daily, thr, target, res["prediction"], res["actual"]),
                        width="stretch")
        show_explanation(model, res, thr, chance)
    except PredictionUnavailable as err:
        st.error(f"A next-day prediction is not available right now: {err}")

# ---------------------------------------------------------------------------
with tab_hist:
    st.subheader("Pick a past date and compare the prediction with what actually happened")
    st.markdown("The prediction for the date you pick is made **only from data up to the day before**. "
                "The data are cut off at that day before any feature is calculated, so the actual "
                "temperature of the chosen date cannot influence its prediction.")
    first_target = daily.index.min() + pd.Timedelta(days=C.WARMUP_DAYS + 1)
    last_target = latest_issue_date(daily) + pd.Timedelta(days=1)
    examples = {
        "Choose a date yourself": None,
        "Hottest test-period day": backtest.loc[backtest.split == "test"].sort_values("actual_tmax").iloc[-1].target_date,
        "Largest test-period miss": backtest.loc[backtest.split == "test"].assign(a=lambda d: d.error.abs()).sort_values("a").iloc[-1].target_date,
        "A typical monsoon-season day (15 Aug 2023)": pd.Timestamp("2023-08-15"),
    }
    pick = st.radio("Quick examples", list(examples), horizontal=True)
    default = examples[pick] if examples[pick] is not None else pd.Timestamp("2024-05-31")
    chosen = st.date_input("Date to predict", value=default.date(), min_value=first_target.date(),
                           max_value=last_target.date(), format="DD/MM/YYYY",
                           disabled=examples[pick] is not None)
    target = pd.Timestamp(default if examples[pick] is not None else chosen)
    try:
        res = predict_for_target_date(model, daily, target)
        t_thr = threshold_for(thr, target)
        st.markdown(SPLIT_NOTE[split_of(target)])
        chance = hot_chance(res["prediction"], t_thr, target)
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Tmax", f"{res['prediction']:.1f} °C",
                  help=f"Made with data up to {res['issue_date']:%d %b %Y}")
        if res["actual"] is not None:
            err = res["prediction"] - res["actual"]
            c2.metric("Actual Tmax (NASA POWER)", f"{res['actual']:.1f} °C")
            c3.metric("Error (predicted − actual)", f"{err:+.2f} °C",
                      help="Positive = the model predicted too warm; negative = too cool.")
            c3.caption("The model was too warm." if err >= 0.05 else
                       "The model was too cool." if err <= -0.05 else "Almost exact.")
            st.markdown(f"Forecast: {flag_badge(chance)}  \n"
                        f"What happened: {heat_badge(res['actual'], t_thr)}")
        else:
            c2.metric("Actual Tmax", "not yet published")
            st.markdown(f"Forecast: {flag_badge(chance)}")
        if res["n_missing_inputs"]:
            st.warning(f"{res['n_missing_inputs']} input value(s) were missing and were filled "
                       "with training medians.")
        st.altair_chart(recent_chart(daily, thr, target, res["prediction"], res["actual"]),
                        width="stretch")
        show_explanation(model, res, thr, chance)
    except PredictionUnavailable as err:
        st.warning(str(err))

# ---------------------------------------------------------------------------
with tab_perf:
    st.subheader("How well does it work on years it never saw? (test period 2019–2025)")
    order = ["Persistence (tomorrow = today)", "Seasonal average", "Linear Regression", "SVR (RBF)", "Random Forest"]
    tmet = pd.DataFrame(meta["test_metrics"]).T.loc[order]
    tmet = tmet[["MAE", "MSE", "RMSE", "R2", "Bias (pred - actual)"]].astype(float).round(3)
    tmet.index = [f"{i} ★ deployed" if i == meta["deployed_model"]["name"] else i for i in tmet.index]
    st.dataframe(tmet, width="stretch")
    st.caption("MAE = average size of the error in °C (lower is better). MSE/RMSE penalise large misses. "
               "R² = share of day-to-day variation explained (1 is perfect). Every model was refitted on "
               "1981–2018 with settings chosen using 1981–2012 (cross-validation) and 2013–2018 (validation).")
    st.markdown(f"**Selection rule (fixed before the test period was scored):** {meta['selection_rule']}")
    vmet = pd.DataFrame(meta["validation_metrics"]).T.loc[order][["MAE", "RMSE", "R2"]].astype(float).round(3)
    with st.expander("Validation-period (2013–2018) scores used to choose the model"):
        st.dataframe(vmet, width="stretch")

    st.markdown("#### Performance on unusually hot days (actual Tmax above the seasonal 90th percentile)")
    hot = pd.DataFrame(meta["test_hot_day_metrics"]).T.loc[order].astype(float)
    hot = hot.rename(columns={"hot_days": "hot days", "MAE_on_hot_days": "MAE on hot days",
                              "Bias_on_hot_days": "bias on hot days"}).round(3)
    st.dataframe(hot, width="stretch")
    st.caption("These errors are measured only on days that turned out to be extreme, which favours "
               "'tomorrow = today' (it copies yesterday's heat). The regression models are pulled toward "
               "typical values and under-predict these peaks by about 1 °C: a real weakness of this "
               "design.")
    st.markdown("#### How well does the heat flag work? (test period)")
    flags = pd.DataFrame(meta["heat_indicator"]["test_flag_metrics"]).T.astype(float)
    flags = flags[["hot_days", "flagged_days", "hits", "missed", "false_alarms", "recall (hit rate)",
                   "precision", "F1"]].round(3)
    st.dataframe(flags, width="stretch")
    st.caption("Hit rate = share of actual hot days that were flagged; precision = share of flags that "
               "were correct; F1 combines both. The deployed flag uses the predicted chance with a cut-off "
               "chosen on 2013–2018. On the test years it catches more hot days than the plain "
               "'prediction above threshold' rule, but persistence still scores a slightly higher F1.")
    cal = pd.DataFrame(meta["heat_indicator"]["test_calibration"])
    with st.expander("Is the 'chance of an unusually hot day' honest? (calibration on the test period)"):
        st.dataframe(cal.round(3), hide_index=True, width="stretch")
        st.caption("If the chances are reliable, days given about 40 % should turn out hot about 40 % "
                   "of the time.")

    test_bt = backtest[backtest.split == "test"].copy()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Predicted vs actual (test period)")
        lo, hi = float(test_bt[["actual_tmax", "predicted_tmax"]].min().min()) - 0.5, \
            float(test_bt[["actual_tmax", "predicted_tmax"]].max().max()) + 0.5
        sc = alt.Chart(test_bt).mark_circle(size=12, opacity=0.35, color=PRED_COLOR).encode(
            x=alt.X("actual_tmax:Q", title="Actual Tmax (°C)", scale=alt.Scale(domain=[lo, hi])),
            y=alt.Y("predicted_tmax:Q", title="Predicted Tmax (°C)", scale=alt.Scale(domain=[lo, hi])),
            tooltip=["target_date:T", "actual_tmax:Q", "predicted_tmax:Q"])
        diag = alt.Chart(pd.DataFrame({"x": [lo, hi], "y": [lo, hi]})).mark_line(
            color="gray", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
        st.altair_chart((sc + diag).properties(height=330), width="stretch")
    with c2:
        st.markdown("#### Errors (residuals) by month")
        test_bt["month"] = test_bt.target_date.dt.month
        box = alt.Chart(test_bt).mark_boxplot(extent=1.5, color=PRED_COLOR).encode(
            x=alt.X("month:O", title="Month"), y=alt.Y("error:Q", title="Predicted − actual (°C)"))
        st.altair_chart(box.properties(height=330), width="stretch")

    st.markdown("#### Test-period time series")
    year = st.select_slider("Year", options=sorted(test_bt.target_date.dt.year.unique()), value=2024)
    yr = test_bt[test_bt.target_date.dt.year == year].melt(
        id_vars="target_date", value_vars=["actual_tmax", "predicted_tmax", "threshold_p90"],
        var_name="series", value_name="°C")
    yr["series"] = yr["series"].map({"actual_tmax": "Actual", "predicted_tmax": "Predicted",
                                     "threshold_p90": "Hot threshold (P90)"})
    ts = alt.Chart(yr).mark_line(strokeWidth=1.4).encode(
        x=alt.X("target_date:T", title=None), y=alt.Y("°C:Q", scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", scale=alt.Scale(domain=["Actual", "Predicted", "Hot threshold (P90)"],
                                                    range=[OBS_COLOR, PRED_COLOR, THR_COLOR]), title=None),
        strokeDash=alt.condition(alt.datum.series == "Hot threshold (P90)", alt.value([5, 4]), alt.value([1, 0])),
        tooltip=["target_date:T", "series:N", alt.Tooltip("°C:Q", format=".1f")])
    st.altair_chart(ts.properties(height=300).configure_legend(orient="bottom"), width="stretch")

    st.markdown("#### What the model relies on overall")
    if hasattr(model.named_steps["model"], "coef_"):
        coefs = linear_coefficients(model).head(12)
        ch = alt.Chart(coefs).mark_bar().encode(
            x=alt.X("coef_per_SD (°C):Q", title="Change in prediction for a typical (1 SD) increase, °C"),
            y=alt.Y("feature:N", sort=alt.EncodingSortField("coef_per_SD (°C)", op="max", order="descending"), title=None),
            color=alt.condition(alt.datum["coef_per_SD (°C)"] > 0, alt.value(UP_COLOR), alt.value(DOWN_COLOR)),
            tooltip=["feature", "group", alt.Tooltip("coef_per_SD (°C):Q", format="+.3f")])
        st.altair_chart(ch.properties(height=330), width="stretch")
        st.caption("Standardised coefficients of the linear model (12 largest). Many inputs are strongly "
                   "correlated (for example several recent-temperature features), so individual "
                   "coefficients can offset each other; the grouped explanation above is easier to read. "
                   "These are associations, not causes.")
    imp_path = C.OUTPUTS_DIR / "permutation_importance_test.csv"
    if imp_path.exists():
        with st.expander("Permutation importance on the test period"):
            st.dataframe(pd.read_csv(imp_path).head(15).round(3), hide_index=True, width="stretch")
            st.caption("How much the test MAE gets worse when one feature's values are shuffled. "
                       "Correlated features share importance, so each can look small on its own.")

    st.markdown("#### Example results from the test period")
    ex = pd.concat([
        test_bt.nlargest(5, "actual_tmax").assign(example="5 hottest days"),
        test_bt.assign(a=test_bt.error.abs()).nlargest(5, "a").assign(example="5 largest errors"),
    ])[["example", "target_date", "actual_tmax", "predicted_tmax", "error", "persistence", "threshold_p90"]]
    ex["target_date"] = ex["target_date"].dt.date
    st.dataframe(ex.rename(columns={"persistence": "tomorrow = today"}), hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
with tab_about:
    st.subheader("Data")
    dm = meta["data"]
    st.markdown(f"""
- **Source:** NASA Langley Research Center POWER Project, Daily API **{dm.get('api_version')}**,
  community `{C.POWER_COMMUNITY}`, time standard `{C.POWER_TIME_STANDARD}` (local solar time).
  Study snapshot retrieved {dm.get('retrieved_utc', '?').replace('T', ' ')} UTC.
- **Location:** {C.LATITUDE}° N, {C.LONGITUDE}° E. NASA POWER meteorology comes from the MERRA-2
  reanalysis (and GEOS-IT for the most recent weeks) on a ½° × ⅝° grid, roughly 55 × 68 km. A value
  represents a **grid-cell average**, so it can differ from readings at Chennai's IMD stations
  (Nungambakkam, Meenambakkam), especially on sea-breeze days. The grid cell's reported elevation is
  {dm.get('grid_cell_geometry', {}).get('coordinates', [None, None, '?'])[2]} m.
- **Variables:** daily max/min/mean temperature, dew point, relative and specific humidity,
  bias-corrected rainfall, wind speed and direction at 2 m, surface pressure.
- **Missing values:** the API uses −999; these become blanks. The 1981–2025 snapshot had none.
- **Reproducible request:**
""")
    st.code(build_request_url(C.STUDY_START, C.STUDY_END), language="text")
    st.subheader("Method in one paragraph")
    st.markdown(f"""
For every day D the model reads weather observed **on day D and earlier** (the last 7 days of maximum
temperature, 3- to 30-day averages, humidity, rain totals, wind direction expressed as land-wind vs
sea-breeze components, pressure and its change, and the calendar position of D+1) and predicts the
maximum temperature of D+1. Data are split **by date**: 1981–2012 training, 2013–2018 validation,
2019–2025 test. Seasonal normals, gap-filling medians and scaling are learned from training rows only
inside a scikit-learn pipeline. Linear Regression, Support Vector Regression and Random Forest were
tuned with time-ordered cross-validation inside 1981–2012 and compared with two baselines. The
deployed model is **{meta['deployed_model']['name']}**, refitted on 1981–2018.

**Historical backtest:** `outputs/backtest_predictions.csv` holds the deployed model's prediction for
every date in 1981–2025, generated by the notebook from each date's previous-day features. The
*Historical backtest* tab recomputes the same prediction live from data truncated at the day before
(an automated test checks both agree). Dates before 2019 were used in fitting and are labelled as such.

**Heat indicator:** {meta['heat_indicator']['definition']}. {meta['heat_indicator']['note']}

**Chance of an unusually hot day and the heat flag:** the model's errors in time-ordered
cross-validation on 1981–2012 have a known spread for each month (about ±0.6 °C in December, about
±1.6 °C in May–June). Treating the error as normally distributed gives the chance that the actual
value exceeds the threshold. The flag is raised when that chance is at least
{meta['heat_indicator']['cutoff']:.0%}, a cut-off chosen on the 2013–2018 validation years (best F1)
and frozen before the test years were scored.

**Design history (reported openly):** the first version had no long-term trend input and flagged
"hot" only when the predicted value itself crossed the threshold. On the test years it caught only
17 % of hot days. Two changes were then chosen *using the 2013–2018 validation years only*: a
calendar-based trend input and the probability-based flag. The test years were then scored a second
time. Both runs' logs are in `outputs/run_logs/`.
""")
    st.subheader("Limitations")
    st.markdown("""
- **Not an official forecast.** IMD heat-wave criteria use station observations and departures from
  normal. This tool uses gridded reanalysis data and its own percentile threshold.
- **Gridded, not station data.** NASA POWER smooths local effects such as the sea breeze and the urban
  heat island, so its values can differ from IMD station readings.
- **A statistical next-day model only.** It extrapolates recent weather and has no knowledge of
  approaching cyclones, sudden rain or monsoon onset beyond what yesterday's data imply. It tends to
  under-predict sudden jumps and over-predict sudden drops.
- **Data latency and revisions.** Data arrive 2–3 days late, and recent GEOS-IT values may later be
  replaced by MERRA-2 values.
- **No climate-change attribution.** This model cannot show that climate change caused any particular
  hot day. Explanations describe associations in historical data, not causes.
- **Trends in gridded reanalysis are uncertain.** Long-term trends in MERRA-2 for a single grid cell may
  not match trends at Chennai's IMD stations, which this project did not analyse.
""")
    st.subheader("What the 1981–2025 record for this grid cell shows")
    snap = daily.loc[C.STUDY_START:C.STUDY_END]
    rows = []
    for col, label, unit in [("tmax", "Daily maximum temperature", "°C"), ("tmin", "Daily minimum (night) temperature", "°C"),
                             ("tmean", "Daily mean temperature", "°C"), ("rh", "Relative humidity", "%"),
                             ("qv", "Specific humidity", "g/kg")]:
        annual = snap[col].groupby(snap.index.year).mean()
        slope = np.polyfit(annual.index, annual.values, 1)[0] * 10
        rows.append({"variable": label, "trend per decade": f"{slope:+.2f} {unit}",
                     "1981–2012 mean": f"{annual.loc[1981:2012].mean():.2f}",
                     "2013–2025 mean": f"{annual.loc[2013:2025].mean():.2f}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    exceed = (snap["tmax"].to_numpy() > thr.loc[snap.index.dayofyear].to_numpy())
    ex_year = pd.Series(exceed, index=snap.index).groupby(snap.index.year).mean() * 100
    st.markdown(f"""
In this NASA POWER cell, **daytime maximum temperature shows almost no linear trend** over 1981–2025, while
**night-time minimum temperature and humidity have risen**. Days above the 1981–2012 'unusually hot'
threshold made up {ex_year.loc[1981:2012].mean():.1f} % of days in 1981–2012 and
{ex_year.loc[2019:2025].mean():.1f} % in 2019–2025. Warmer, more humid nights matter for heat stress
even when daytime maxima do not rise. This is a description of one gridded dataset (ordinary least-squares
trend of annual means, no significance test), not a formal climate-trend or attribution analysis.
""")
    st.subheader("References")
    st.markdown("""
- NASA POWER Project, Daily API documentation: https://power.larc.nasa.gov/docs/services/api/temporal/daily/
- NASA POWER meteorology sources: https://power.larc.nasa.gov/docs/methodology/data/sources_meteorology/
- Gelaro, R. et al. (2017). The Modern-Era Retrospective Analysis for Research and Applications, Version 2 (MERRA-2). *J. Climate* 30, 5419–5454. https://doi.org/10.1175/JCLI-D-16-0758.1
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR* 12, 2825–2830.
- *These data were obtained from the NASA Langley Research Center POWER Project funded through the NASA Earth Science Division.*
""")
