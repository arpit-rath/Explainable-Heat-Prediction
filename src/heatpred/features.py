"""Leakage-safe feature engineering.

Convention
----------
Each row is indexed by the *issue date* D. Every feature on row D is computed
only from observations on day D or earlier. The target on row D is the
maximum temperature of day D+1 (the *target date*). Rolling windows end on D
(inclusive) and never look forward.

Statistics that are *learned* from data (seasonal climatology, imputation
medians, scaling) are not computed here: they live inside the scikit-learn
pipeline (``ClimatologyTransformer``, ``SimpleImputer``, ``StandardScaler``)
so they are fitted on training rows only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config as C

TARGET = "target_tmax_next_day"

# Human-readable grouping used by explanations in the app.
FEATURE_GROUPS: dict[str, str] = {}


def _group(name: str, *cols: str) -> list[str]:
    for c in cols:
        FEATURE_GROUPS[c] = name
    return list(cols)


RAW_FEATURES = (
    _group("Recent temperatures",
           *[f"tmax_lag{k}" for k in range(7)], "tmax_change_1d",
           "tmax_mean_3", "tmax_mean_7", "tmax_mean_14", "tmax_mean_30", "tmax_std_7",
           "tmin_lag0", "tmin_lag1", "tmin_lag2", "tmin_mean_7", "tmean_lag0", "dtr_lag0")
    + _group("Humidity & moisture", "rh_lag0", "rh_lag1", "rh_mean_7", "qv_lag0", "dewpoint_lag0")
    + _group("Rainfall", "precip_lag0", "precip_lag1", "precip_sum_3", "precip_sum_7",
             "precip_sum_30")
    + _group("Wind (sea breeze vs land wind)", "wind_speed_lag0", "wind_u_lag0", "wind_v_lag0",
             "wind_u_mean_3")
    + _group("Surface pressure", "pressure_lag0", "pressure_change_1d")
    + _group("Time of year", "target_doy_sin", "target_doy_cos")
    + _group("Slow drift across years", "years_since_1981")
)
# Columns consumed by ClimatologyTransformer (dropped after use).
HELPER_COLUMNS = ["issue_doy", "target_doy"]
# Columns added by ClimatologyTransformer.
CLIM_FEATURES = (_group("Time of year", "clim_tmax_target")
                 + _group("Recent temperatures", "tmax_anom_lag0", "tmax_mean_7_anom"))
FEATURE_GROUPS.update({"issue_doy": "Time of year", "target_doy": "Time of year"})
INPUT_COLUMNS = RAW_FEATURES + HELPER_COLUMNS
MODEL_FEATURES = RAW_FEATURES + CLIM_FEATURES

FEATURE_DESCRIPTIONS = {
    "tmax_lag0": "Maximum temperature on the issue day (D)",
    "tmax_change_1d": "Change in max temperature from D-1 to D",
    "tmax_mean_7": "Average max temperature over the last 7 days",
    "tmax_anom_lag0": "How far D's max temperature was above/below normal for its date",
    "tmax_mean_7_anom": "How far the last-7-day average max was above/below normal",
    "clim_tmax_target": "Normal max temperature for the target date (average over the years the model was fitted on)",
    "rh_lag0": "Relative humidity on day D",
    "precip_sum_3": "Total rain over the last 3 days",
    "wind_u_lag0": "West-to-east wind component on D (positive = wind from the land/west)",
    "wind_v_lag0": "South-to-north wind component on D (positive = wind from the south)",
    "pressure_change_1d": "Change in surface pressure from D-1 to D",
    "dtr_lag0": "Day-night temperature range on D",
    "tmean_lag0": "Mean temperature on day D",
    "tmin_lag0": "Minimum (night-time) temperature on day D",
    "qv_lag0": "Specific humidity (water vapour content) on day D",
    "dewpoint_lag0": "Dew point on day D",
    "years_since_1981": "Years since 1981 (lets the model absorb slow multi-year shifts in the record)",
    "target_doy_sin": "Time of year of the target date (seasonal cycle, sine part)",
    "target_doy_cos": "Time of year of the target date (seasonal cycle, cosine part)",
    "issue_doy": "Calendar day of D (used to compute seasonal normals)",
    "target_doy": "Calendar day of D+1 (used to look up the seasonal normal)",
    "tmax_mean_3": "Average max temperature over the last 3 days",
    "tmax_mean_14": "Average max temperature over the last 14 days",
    "tmax_mean_30": "Average max temperature over the last 30 days",
    "tmax_std_7": "Day-to-day variability of max temperature over the last 7 days",
    "tmin_lag1": "Minimum temperature on D-1",
    "tmin_lag2": "Minimum temperature on D-2",
    "tmin_mean_7": "Average minimum temperature over the last 7 days",
    "rh_lag1": "Relative humidity on D-1",
    "rh_mean_7": "Average relative humidity over the last 7 days",
    "precip_lag0": "Rain on day D",
    "precip_lag1": "Rain on D-1",
    "precip_sum_7": "Total rain over the last 7 days",
    "precip_sum_30": "Total rain over the last 30 days",
    "wind_speed_lag0": "Wind speed on day D",
    "wind_u_mean_3": "Average west-to-east wind component over the last 3 days (positive = land wind)",
    "pressure_lag0": "Surface pressure on day D",
}
for _k in range(1, 7):
    FEATURE_DESCRIPTIONS[f"tmax_lag{_k}"] = f"Maximum temperature {_k} day(s) before D"


def _rolling(s: pd.Series, window: int, how: str) -> pd.Series:
    r = s.rolling(window, min_periods=max(1, int(np.ceil(0.7 * window))))
    return getattr(r, how)()


def build_feature_table(daily: pd.DataFrame) -> pd.DataFrame:
    """Return one row per issue date D with features (<= D) and the target (D+1).

    ``daily`` must be indexed by date. Missing calendar days are inserted as
    NaN so that a lag of k rows is always a lag of k days.
    """
    d = daily.sort_index()
    d = d[~d.index.duplicated(keep="last")]
    d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"))
    d.index.name = "issue_date"

    # Wind components (meteorological convention: direction wind blows FROM).
    theta = np.deg2rad(d["wind_dir"])
    wind_u = -d["wind_speed"] * np.sin(theta)   # > 0 : westerly (from land, Chennai coast faces east)
    wind_v = -d["wind_speed"] * np.cos(theta)   # > 0 : southerly

    f = pd.DataFrame(index=d.index)
    for k in range(7):
        f[f"tmax_lag{k}"] = d["tmax"].shift(k)
    f["tmax_change_1d"] = d["tmax"] - d["tmax"].shift(1)
    for w in (3, 7, 14, 30):
        f[f"tmax_mean_{w}"] = _rolling(d["tmax"], w, "mean")
    f["tmax_std_7"] = _rolling(d["tmax"], 7, "std")

    for k in range(3):
        f[f"tmin_lag{k}"] = d["tmin"].shift(k)
    f["tmin_mean_7"] = _rolling(d["tmin"], 7, "mean")
    f["tmean_lag0"] = d["tmean"]
    f["dtr_lag0"] = d["tmax"] - d["tmin"]

    f["rh_lag0"] = d["rh"]
    f["rh_lag1"] = d["rh"].shift(1)
    f["rh_mean_7"] = _rolling(d["rh"], 7, "mean")
    f["qv_lag0"] = d["qv"]
    f["dewpoint_lag0"] = d["dewpoint"]

    f["precip_lag0"] = d["precip"]
    f["precip_lag1"] = d["precip"].shift(1)
    for w in (3, 7, 30):
        f[f"precip_sum_{w}"] = _rolling(d["precip"], w, "sum")

    f["wind_speed_lag0"] = d["wind_speed"]
    f["wind_u_lag0"] = wind_u
    f["wind_v_lag0"] = wind_v
    f["wind_u_mean_3"] = _rolling(wind_u, 3, "mean")

    f["pressure_lag0"] = d["pressure"]
    f["pressure_change_1d"] = d["pressure"] - d["pressure"].shift(1)

    target_date = d.index + pd.Timedelta(days=1)
    tdoy = target_date.dayofyear.to_numpy()
    f["target_doy_sin"] = np.sin(2 * np.pi * tdoy / 365.25)
    f["target_doy_cos"] = np.cos(2 * np.pi * tdoy / 365.25)
    # Slow drift term: known from the calendar alone, so it is available at prediction time.
    f["years_since_1981"] = (d.index.year - 1981) + (d.index.dayofyear - 1) / 365.25
    f["issue_doy"] = d.index.dayofyear
    f["target_doy"] = tdoy

    f["target_date"] = target_date
    f[TARGET] = d["tmax"].shift(-1)             # the ONLY column that looks forward
    f["n_missing_inputs"] = f[RAW_FEATURES].isna().sum(axis=1)
    return f


def drop_warmup(table: pd.DataFrame, days: int = C.WARMUP_DAYS) -> pd.DataFrame:
    """Drop the first rows whose rolling windows are incomplete."""
    return table.iloc[days:]


def assign_split(target_dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    t = pd.to_datetime(pd.Series(target_dates)).to_numpy()
    out = np.full(len(t), "live", dtype=object)
    out[t <= np.datetime64(C.STUDY_END)] = "test"
    out[t <= np.datetime64(C.VAL_END)] = "validation"
    out[t <= np.datetime64(C.TRAIN_END)] = "train"
    return out


def modelling_table(daily: pd.DataFrame) -> pd.DataFrame:
    """Feature table restricted to rows with a known target, labelled by split."""
    t = drop_warmup(build_feature_table(daily))
    t = t[t[TARGET].notna()].copy()
    t["split"] = assign_split(t["target_date"])
    return t


# ---------------------------------------------------------------------------
# Seasonal climatology & heat threshold (learned from training data only)
# ---------------------------------------------------------------------------
def _circular_window_stat(doy: np.ndarray, values: np.ndarray, half_width: int, stat) -> np.ndarray:
    """For each calendar day 1..366, apply ``stat`` to values within +/- half_width days."""
    out = np.full(367, np.nan)
    for day in range(1, 367):
        dist = np.abs(doy - day)
        dist = np.minimum(dist, 366 - dist)
        sel = values[(dist <= half_width) & ~np.isnan(values)]
        if sel.size:
            out[day] = stat(sel)
    return out


class ClimatologyTransformer(BaseEstimator, TransformerMixin):
    """Adds seasonal-normal features learned from the rows it is fitted on.

    fit(): builds a smoothed day-of-year mean of ``tmax_lag0`` (the issue
    day's observed Tmax, i.e. an *input*, not the target) using a +/-
    ``half_width`` day window.
    transform(): adds the normal Tmax for the target date and the issue-day
    anomalies, then drops the helper day-of-year columns.
    """

    def __init__(self, half_width: int = C.HOT_WINDOW_HALF_WIDTH):
        self.half_width = half_width

    def fit(self, X: pd.DataFrame, y=None):
        self.clim_ = _circular_window_stat(
            X["issue_doy"].to_numpy(int), X["tmax_lag0"].to_numpy(float), self.half_width, np.mean
        )
        self.feature_names_in_ = np.array(X.columns)
        kept = [c for c in X.columns if c not in HELPER_COLUMNS]
        self.feature_names_out_ = np.array(kept + CLIM_FEATURES)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        clim_today = self.clim_[X["issue_doy"].to_numpy(int)]
        X["clim_tmax_target"] = self.clim_[X["target_doy"].to_numpy(int)]
        X["tmax_anom_lag0"] = X["tmax_lag0"] - clim_today
        X["tmax_mean_7_anom"] = X["tmax_mean_7"] - clim_today
        return X.drop(columns=HELPER_COLUMNS)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


def seasonal_threshold(dates: pd.DatetimeIndex, tmax: np.ndarray,
                       q: float = C.HOT_PERCENTILE,
                       half_width: int = C.HOT_WINDOW_HALF_WIDTH) -> pd.Series:
    """Day-of-year percentile of Tmax (15-day window), indexed 1..366."""
    arr = _circular_window_stat(pd.DatetimeIndex(dates).dayofyear.to_numpy(), np.asarray(tmax, float),
                                half_width, lambda v: np.percentile(v, q))
    return pd.Series(arr[1:], index=pd.RangeIndex(1, 367, name="doy"), name=f"p{int(q)}_tmax")
