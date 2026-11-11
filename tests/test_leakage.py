"""Leakage and correctness checks. Run with:  python -m pytest -q"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heatpred import config as C  # noqa: E402
from heatpred.features import (INPUT_COLUMNS, RAW_FEATURES, TARGET, ClimatologyTransformer,  # noqa: E402
                               assign_split, build_feature_table, seasonal_threshold)


def synthetic_daily(n_days: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", periods=n_days, freq="D")
    doy = idx.dayofyear.to_numpy()
    tmax = 32 + 4 * np.sin(2 * np.pi * (doy - 60) / 365) + rng.normal(0, 1, n_days)
    tmin = tmax - 7 + rng.normal(0, 0.5, n_days)
    return pd.DataFrame({
        "tmax": tmax, "tmin": tmin, "tmean": (tmax + tmin) / 2, "dewpoint": 22 + rng.normal(0, 1, n_days),
        "rh": 70 + rng.normal(0, 5, n_days), "qv": 16 + rng.normal(0, 1, n_days),
        "precip": rng.gamma(0.5, 4, n_days), "wind_speed": 3 + rng.normal(0, 0.5, n_days),
        "wind_dir": rng.uniform(0, 360, n_days), "pressure": 100.3 + rng.normal(0, 0.3, n_days),
    }, index=idx)


def test_features_do_not_change_when_future_changes():
    """Perturbing every value after day D must not change any feature on row D."""
    daily = synthetic_daily()
    base = build_feature_table(daily)
    cut = pd.Timestamp("2000-08-15")
    future = daily.copy()
    future.loc[future.index > cut] += 50.0          # wildly different future
    pert = build_feature_table(future)
    pd.testing.assert_frame_equal(base.loc[:cut, RAW_FEATURES], pert.loc[:cut, RAW_FEATURES])


def test_target_is_next_day_tmax_and_not_a_feature():
    daily = synthetic_daily()
    t = build_feature_table(daily)
    d = pd.Timestamp("2000-03-10")
    assert t.loc[d, TARGET] == pytest.approx(daily.loc[d + pd.Timedelta(days=1), "tmax"])
    assert t.loc[d, "tmax_lag0"] == pytest.approx(daily.loc[d, "tmax"])
    assert TARGET not in INPUT_COLUMNS
    # no feature equals the target column
    for col in RAW_FEATURES:
        assert not np.allclose(t[col].fillna(-1), t[TARGET].fillna(-1)), col


def test_missing_calendar_days_keep_lags_in_days():
    daily = synthetic_daily().drop(pd.Timestamp("2000-02-10"))
    t = build_feature_table(daily)
    assert np.isnan(t.loc[pd.Timestamp("2000-02-10"), "tmax_lag0"])
    assert np.isnan(t.loc[pd.Timestamp("2000-02-11"), "tmax_lag1"])


def test_split_assignment_by_target_date():
    s = assign_split(pd.to_datetime(["2012-12-31", "2013-01-01", "2018-12-31", "2019-01-01",
                                     "2025-12-31", "2026-01-01"]))
    assert list(s) == ["train", "validation", "validation", "test", "test", "live"]


def test_climatology_uses_only_fitted_rows():
    daily = synthetic_daily(800)
    t = build_feature_table(daily).iloc[30:]
    t = t[t[TARGET].notna()]
    first, second = t.iloc[:365], t.iloc[365:]
    a = ClimatologyTransformer().fit(first[INPUT_COLUMNS])
    shifted = first[INPUT_COLUMNS].copy()
    b = ClimatologyTransformer().fit(pd.concat([shifted, second[INPUT_COLUMNS]]))
    assert not np.allclose(a.clim_[1:], b.clim_[1:])  # depends on rows it is fitted on
    out = a.transform(second[INPUT_COLUMNS])
    assert "issue_doy" not in out and "clim_tmax_target" in out


def test_threshold_is_seasonal_percentile():
    idx = pd.date_range("2001-01-01", "2010-12-31", freq="D")
    vals = np.arange(len(idx), dtype=float)
    thr = seasonal_threshold(idx, vals)
    assert thr.index.min() == 1 and thr.index.max() == 366
    assert thr.notna().all()


@pytest.mark.skipif(not C.MODEL_FILE.exists(), reason="model not trained yet")
def test_app_prediction_matches_precomputed_backtest():
    """The app's on-the-fly prediction equals the documented precomputed backtest."""
    import joblib
    from heatpred import data as D
    from heatpred.predict import predict_for_target_date
    daily, _ = D.load_frame(C.SNAPSHOT_CSV)
    model = joblib.load(C.MODEL_FILE)
    bt = pd.read_csv(C.BACKTEST_CSV, parse_dates=["target_date"])
    for day in ["2019-05-20", "2021-01-15", "2024-06-01", "2025-12-31"]:
        res = predict_for_target_date(model, daily, pd.Timestamp(day))
        expected = bt.loc[bt.target_date == pd.Timestamp(day), "predicted_tmax"].item()
        assert res["prediction"] == pytest.approx(expected, abs=0.006)
        # actual value of the target day must not be in the features used
        assert res["issue_date"] == pd.Timestamp(day) - pd.Timedelta(days=1)


@pytest.mark.skipif(not C.MODEL_FILE.exists(), reason="model not trained yet")
def test_prediction_ignores_target_day_and_later_data():
    import joblib
    from heatpred import data as D
    from heatpred.predict import predict_for_target_date
    daily, _ = D.load_frame(C.SNAPSHOT_CSV)
    model = joblib.load(C.MODEL_FILE)
    day = pd.Timestamp("2024-05-10")
    a = predict_for_target_date(model, daily, day)["prediction"]
    tampered = daily.copy()
    tampered.loc[tampered.index >= day, :] = tampered.loc[tampered.index >= day, :] + 10
    b = predict_for_target_date(model, tampered, day)["prediction"]
    assert a == pytest.approx(b)


@pytest.mark.skipif(not C.REFERENCE_CSV.exists(), reason="model not trained yet")
def test_local_explanation_is_exact():
    """base + sum(contributions) reproduces the model's prediction exactly."""
    import joblib
    from heatpred import data as D
    from heatpred.explain import explain_row
    from heatpred.predict import predict_for_target_date
    daily, _ = D.load_frame(C.SNAPSHOT_CSV)
    model = joblib.load(C.MODEL_FILE)
    ref = pd.read_csv(C.REFERENCE_CSV, index_col="target_doy")
    for day in ["2019-05-20", "2023-11-20", "2024-12-31"]:
        res = predict_for_target_date(model, daily, pd.Timestamp(day))
        for reference in (ref, None):
            base, table = explain_row(model, res["features"], reference)
            assert base + table.contribution.sum() == pytest.approx(res["prediction"], abs=0.006)
