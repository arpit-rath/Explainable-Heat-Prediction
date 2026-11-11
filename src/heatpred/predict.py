"""Single-date prediction used by the app (historical backtest and next-day)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .features import INPUT_COLUMNS, RAW_FEATURES, build_feature_table

HISTORY_DAYS_NEEDED = C.WARMUP_DAYS + 5


class PredictionUnavailable(ValueError):
    """The requested date cannot be predicted from the data available."""


def features_for_issue_date(daily: pd.DataFrame, issue_date: pd.Timestamp) -> pd.DataFrame:
    """Feature row for issue date D, built from data on or before D ONLY.

    The daily table is truncated at D *before* features are computed, so the
    target day's observations are physically absent from the computation.
    """
    issue_date = pd.Timestamp(issue_date).normalize()
    window = daily.loc[issue_date - pd.Timedelta(days=HISTORY_DAYS_NEEDED): issue_date]
    if window.empty or issue_date not in window.index or window.loc[issue_date].isna().all():
        raise PredictionUnavailable(
            f"No NASA POWER observations are available for {issue_date.date()} "
            "(the day before the selected date), so no prediction can be made.")
    table = build_feature_table(window)
    return table.loc[[issue_date]]


def predict_for_target_date(model, daily: pd.DataFrame, target_date) -> dict:
    target_date = pd.Timestamp(target_date).normalize()
    issue_date = target_date - pd.Timedelta(days=1)
    earliest = daily.index.min() + pd.Timedelta(days=C.WARMUP_DAYS + 1)
    if target_date < earliest:
        raise PredictionUnavailable(
            f"Predictions start on {earliest.date()}: the model needs {C.WARMUP_DAYS} days of history.")
    row = features_for_issue_date(daily, issue_date)
    n_missing = int(row[RAW_FEATURES].isna().sum(axis=1).iloc[0])
    pred = float(model.predict(row[INPUT_COLUMNS])[0])
    actual = daily["tmax"].get(target_date, np.nan)
    return {
        "target_date": target_date,
        "issue_date": issue_date,
        "prediction": round(pred, 2),
        "actual": None if pd.isna(actual) else float(actual),
        "n_missing_inputs": n_missing,
        "features": row,
    }


def latest_issue_date(daily: pd.DataFrame) -> pd.Timestamp:
    """Most recent day on which every parameter is present (not a fill value)."""
    complete = daily.dropna(how="any")
    if complete.empty:
        raise PredictionUnavailable("No complete day of data is available.")
    return complete.index.max()
