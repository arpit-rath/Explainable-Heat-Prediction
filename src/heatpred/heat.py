"""Probability that the target day is 'unusually hot for Chennai'.

The regression model gives a single best-guess temperature. Its errors are
roughly normal with a spread that depends on the season, so

    P(hot) = P(actual > threshold) = 1 - Phi((threshold - prediction) / sigma_month)

where sigma_month is the standard deviation of the model's out-of-fold errors
in time-series cross-validation on the TRAINING period, by target month. The
cut-off for raising the flag is chosen on the VALIDATION period by maximising
F1, then frozen before the test period is scored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone

from .features import TARGET
from .models import time_series_cv

CUTOFF_GRID = np.round(np.arange(0.10, 0.71, 0.05), 2)


def cv_residuals(estimator, X: pd.DataFrame, y: pd.Series, target_dates: pd.Series) -> pd.DataFrame:
    rows = []
    for tr, te in time_series_cv().split(X):
        m = clone(estimator).fit(X.iloc[tr], y.iloc[tr])
        rows.append(pd.DataFrame({"month": pd.DatetimeIndex(target_dates.iloc[te]).month,
                                  "residual": y.iloc[te].to_numpy() - m.predict(X.iloc[te])}))
    return pd.concat(rows, ignore_index=True)


def sigma_by_month(residuals: pd.DataFrame) -> dict[int, float]:
    s = residuals.groupby("month")["residual"].std()
    return {int(k): float(v) for k, v in s.items()}


def hot_probability(prediction, threshold, months, sigma: dict[int, float]) -> np.ndarray:
    sig = np.array([sigma[int(m)] for m in np.atleast_1d(months)])
    return 1.0 - norm.cdf((np.asarray(threshold, float) - np.asarray(prediction, float)) / sig)


def flag_scores(hot: np.ndarray, flagged: np.ndarray, prob: np.ndarray | None = None) -> dict:
    hot, flagged = np.asarray(hot, bool), np.asarray(flagged, bool)
    tp = int((hot & flagged).sum())
    out = {
        "hot_days": int(hot.sum()),
        "flagged_days": int(flagged.sum()),
        "hits": tp,
        "missed": int((hot & ~flagged).sum()),
        "false_alarms": int((~hot & flagged).sum()),
        "recall (hit rate)": tp / hot.sum() if hot.any() else np.nan,
        "precision": tp / flagged.sum() if flagged.any() else np.nan,
        "F1": 2 * tp / (hot.sum() + flagged.sum()) if (hot.sum() + flagged.sum()) else np.nan,
    }
    if prob is not None:
        out["Brier score"] = float(np.mean((np.asarray(prob, float) - hot) ** 2))
    return out


def choose_cutoff(prob: np.ndarray, hot: np.ndarray) -> tuple[float, pd.DataFrame]:
    table = pd.DataFrame([{"cutoff": c, **flag_scores(hot, prob >= c)} for c in CUTOFF_GRID])
    best = float(table.loc[table["F1"].idxmax(), "cutoff"])
    return best, table


def calibration_table(prob: np.ndarray, hot: np.ndarray) -> pd.DataFrame:
    bins = pd.cut(prob, [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0], include_lowest=True)
    df = pd.DataFrame({"prob": prob, "hot": np.asarray(hot, float)})
    return df.groupby(bins, observed=True).agg(days=("hot", "size"), mean_predicted_chance=("prob", "mean"),
                                               observed_frequency=("hot", "mean"))


def is_hot(table: pd.DataFrame, threshold: np.ndarray) -> np.ndarray:
    return table[TARGET].to_numpy() > threshold
