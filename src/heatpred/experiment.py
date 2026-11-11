"""End-to-end experiment, split into stages that the notebook calls one by one.

Stage order (the test period is only touched in ``final_test_evaluation``):
  1. ``prepare``                - snapshot -> leakage-safe feature table + splits
  2. ``fit_threshold``          - seasonal P90 heat threshold from TRAINING target days only
  3. ``tune_candidates``        - time-series CV inside the training period
  4. ``validation_comparison``  - fit on train, score on validation -> select model
  5. ``final_test_evaluation``  - refit on train+validation, score once on test
  6. ``save_artifacts``         - model, metadata, threshold, backtest predictions
"""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import heat as H
from .features import FEATURE_GROUPS, INPUT_COLUMNS, MODEL_FEATURES, TARGET, modelling_table, seasonal_threshold
from .models import (build_estimator, hot_day_metrics, regression_metrics, select_model, tune)

ML_MODELS = ["Linear Regression", "SVR (RBF)", "Random Forest"]
BASELINES = ["Persistence (tomorrow = today)", "Seasonal average"]
ALL_MODELS = BASELINES + ML_MODELS


@dataclass
class Experiment:
    daily: pd.DataFrame = None
    data_meta: dict = field(default_factory=dict)
    table: pd.DataFrame = None
    threshold: pd.Series = None
    tuning: dict = field(default_factory=dict)
    val_metrics: pd.DataFrame = None
    selected: str = None
    test_metrics: pd.DataFrame = None
    test_hot_metrics: pd.DataFrame = None
    final_models: dict = field(default_factory=dict)
    test_predictions: pd.DataFrame = None
    heat: dict = field(default_factory=dict)
    val_cutoff_table: pd.DataFrame = None
    val_calibration: pd.DataFrame = None
    test_flag_metrics: pd.DataFrame = None
    test_calibration: pd.DataFrame = None

    # -- convenience accessors ------------------------------------------------
    def part(self, *splits: str) -> pd.DataFrame:
        return self.table[self.table["split"].isin(splits)]

    def Xy(self, *splits: str):
        t = self.part(*splits)
        return t[INPUT_COLUMNS], t[TARGET]

    def threshold_for(self, target_dates) -> np.ndarray:
        doy = pd.DatetimeIndex(target_dates).dayofyear
        return self.threshold.loc[doy].to_numpy()


def prepare(force_download: bool = False) -> Experiment:
    exp = Experiment()
    exp.daily, exp.data_meta = D.load_or_download_snapshot(force=force_download)
    exp.table = modelling_table(exp.daily)
    return exp


def split_summary(exp: Experiment) -> pd.DataFrame:
    g = exp.table.groupby("split", sort=False)
    return pd.DataFrame({
        "rows": g.size(),
        "first target date": g["target_date"].min().dt.date,
        "last target date": g["target_date"].max().dt.date,
        "mean target Tmax (°C)": g[TARGET].mean().round(2),
    })


def fit_threshold(exp: Experiment) -> pd.Series:
    train = exp.part("train")
    exp.threshold = seasonal_threshold(train["target_date"], train[TARGET].to_numpy())
    return exp.threshold


def tune_candidates(exp: Experiment, verbose: bool = True) -> pd.DataFrame:
    X, y = exp.Xy("train")
    rows = []
    for name in ML_MODELS:
        res = tune(name, X, y)
        exp.tuning[name] = res
        rows.append({k: v for k, v in res.items() if k != "cv_results"})
        if verbose:
            print(f"{name:18s} CV MAE {res['cv_mae']:.3f} ± {res['cv_mae_std']:.3f} °C "
                  f"({res['n_candidates']} candidates, {res['seconds']} s) {res['best_params']}")
    return pd.DataFrame(rows).set_index("model")


def use_recorded_tuning(exp: Experiment, metadata_path=C.METADATA_FILE) -> pd.DataFrame:
    """Fast path for Colab: reuse hyperparameters found by a previous full search."""
    recorded = json.loads(open(metadata_path, encoding="utf-8").read())["tuning"]
    exp.tuning = {name: dict(recorded[name]) for name in ML_MODELS}
    return pd.DataFrame(exp.tuning.values()).set_index("model")


def _evaluate(models: dict, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds = pd.DataFrame({name: m.predict(X) for name, m in models.items()}, index=X.index)
    metrics = pd.DataFrame({name: regression_metrics(y, preds[name]) for name in preds}).T
    return metrics, preds


def validation_comparison(exp: Experiment) -> pd.DataFrame:
    Xtr, ytr = exp.Xy("train")
    Xva, yva = exp.Xy("validation")
    models = {name: build_estimator(name, exp.tuning.get(name, {}).get("best_params")).fit(Xtr, ytr)
              for name in ALL_MODELS}
    exp.val_metrics, val_preds = _evaluate(models, Xva, yva)
    exp.selected = select_model(exp.val_metrics)
    calibrate_heat_probability(exp, val_preds[exp.selected])
    return exp.val_metrics


def calibrate_heat_probability(exp: Experiment, val_pred: pd.Series) -> dict:
    """Error spread from training CV; flag cut-off from validation (max F1)."""
    train = exp.part("train")
    est = build_estimator(exp.selected, exp.tuning.get(exp.selected, {}).get("best_params"))
    resid = H.cv_residuals(est, train[INPUT_COLUMNS], train[TARGET], train["target_date"])
    sigma = H.sigma_by_month(resid)
    val = exp.part("validation")
    thr = exp.threshold_for(val["target_date"])
    prob = H.hot_probability(val_pred, thr, val["target_date"].dt.month, sigma)
    hot = val[TARGET].to_numpy() > thr
    cutoff, exp.val_cutoff_table = H.choose_cutoff(prob, hot)
    exp.val_calibration = H.calibration_table(prob, hot)
    exp.heat = {"sigma_by_month": sigma, "cutoff": cutoff,
                "validation_flag": H.flag_scores(hot, prob >= cutoff, prob),
                "validation_persistence_flag": H.flag_scores(hot, val["tmax_lag0"].to_numpy() > thr)}
    return exp.heat


def final_test_evaluation(exp: Experiment) -> pd.DataFrame:
    """Refit every model on train+validation with its tuned settings; score on test once."""
    Xfit, yfit = exp.Xy("train", "validation")
    Xte, yte = exp.Xy("test")
    exp.final_models = {name: build_estimator(name, exp.tuning.get(name, {}).get("best_params")).fit(Xfit, yfit)
                        for name in ALL_MODELS}
    exp.test_metrics, preds = _evaluate(exp.final_models, Xte, yte)
    test = exp.part("test")
    thr = exp.threshold_for(test["target_date"])
    exp.test_hot_metrics = pd.DataFrame({n: hot_day_metrics(yte, preds[n], thr) for n in preds}).T
    preds.insert(0, "actual", yte)
    preds.insert(0, "target_date", test["target_date"])
    preds["threshold_p90"] = thr
    prob = H.hot_probability(preds[exp.selected], thr, test["target_date"].dt.month, exp.heat["sigma_by_month"])
    preds["hot_probability"] = prob
    exp.test_predictions = preds
    hot = yte.to_numpy() > thr
    exp.test_flag_metrics = pd.DataFrame({
        "Persistence: today's Tmax above threshold": H.flag_scores(hot, Xte["tmax_lag0"].to_numpy() > thr),
        f"{exp.selected}: predicted Tmax above threshold": H.flag_scores(hot, preds[exp.selected].to_numpy() > thr),
        f"{exp.selected}: chance of hot day >= {exp.heat['cutoff']:.0%} (deployed flag)":
            H.flag_scores(hot, prob >= exp.heat["cutoff"], prob),
    }).T
    exp.test_calibration = H.calibration_table(prob, hot)
    return exp.test_metrics


def permutation_importance_table(exp: Experiment, name: str | None = None, n_repeats: int = 5) -> pd.DataFrame:
    """Model-agnostic global importance on the TEST period for the final model."""
    from sklearn.inspection import permutation_importance
    name = name or exp.selected
    model = exp.final_models[name]
    Xte, yte = exp.Xy("test")
    r = permutation_importance(model, Xte, yte, scoring="neg_mean_absolute_error",
                               n_repeats=n_repeats, random_state=C.RANDOM_SEED, n_jobs=1)
    out = pd.DataFrame({"feature": Xte.columns, "MAE increase (°C)": r.importances_mean,
                        "std": r.importances_std})
    return out.sort_values("MAE increase (°C)", ascending=False).reset_index(drop=True)


def backtest_frame(exp: Experiment) -> pd.DataFrame:
    """Predictions of the deployed model and baselines for every snapshot row."""
    X = exp.table[INPUT_COLUMNS]
    out = pd.DataFrame({
        "issue_date": exp.table.index.date,
        "target_date": exp.table["target_date"].dt.date,
        "split": exp.table["split"],
        "actual_tmax": exp.table[TARGET].round(2),
        "predicted_tmax": np.round(exp.final_models[exp.selected].predict(X), 2),
        "persistence": np.round(exp.final_models[BASELINES[0]].predict(X), 2),
        "seasonal_average": np.round(exp.final_models[BASELINES[1]].predict(X), 2),
        "threshold_p90": np.round(exp.threshold_for(exp.table["target_date"]), 2),
    })
    out["error"] = (out["predicted_tmax"] - out["actual_tmax"]).round(2)
    prob = H.hot_probability(out["predicted_tmax"], out["threshold_p90"],
                             exp.table["target_date"].dt.month, exp.heat["sigma_by_month"])
    out["hot_probability"] = np.round(prob, 3)
    out["hot_flag"] = prob >= exp.heat["cutoff"]
    out["actually_hot"] = out["actual_tmax"] > out["threshold_p90"]
    out["in_sample_for_deployed_model"] = out["split"].isin(["train", "validation"])
    return out


def _versions() -> dict:
    import sklearn, scipy, requests  # noqa: E401
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
            "joblib": joblib.__version__, "requests": requests.__version__}


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def save_artifacts(exp: Experiment, importance: pd.DataFrame | None = None) -> dict:
    C.MODELS_DIR.mkdir(exist_ok=True)
    C.OUTPUTS_DIR.mkdir(exist_ok=True)
    joblib.dump(exp.final_models[exp.selected], C.MODEL_FILE, compress=3)
    exp.threshold.round(3).to_csv(C.THRESHOLD_CSV)
    from .explain import seasonal_reference
    Xfit, _ = exp.Xy("train", "validation")
    seasonal_reference(exp.final_models[exp.selected], Xfit).to_csv(C.REFERENCE_CSV)
    bt = backtest_frame(exp)
    bt.to_csv(C.BACKTEST_CSV, index=False)
    exp.test_metrics.round(4).to_csv(C.OUTPUTS_DIR / "test_metrics.csv")
    exp.val_metrics.round(4).to_csv(C.OUTPUTS_DIR / "validation_metrics.csv")
    exp.test_hot_metrics.round(4).to_csv(C.OUTPUTS_DIR / "test_hot_day_metrics.csv")
    exp.test_flag_metrics.round(4).to_csv(C.OUTPUTS_DIR / "test_heat_flag_metrics.csv")
    exp.test_calibration.round(4).to_csv(C.OUTPUTS_DIR / "test_heat_probability_calibration.csv")
    if importance is not None:
        importance.round(4).to_csv(C.OUTPUTS_DIR / "permutation_importance_test.csv", index=False)

    meta = {
        "project": "Explainable Next-Day Extreme Heat Prediction for Chennai",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "location": {"name": C.LOCATION_NAME, "latitude": C.LATITUDE, "longitude": C.LONGITUDE},
        "data": exp.data_meta,
        "target": "Next-day (D+1) daily maximum 2 m air temperature, °C (NASA POWER T2M_MAX)",
        "splits_by_target_date": {
            "train": [C.STUDY_START, C.TRAIN_END], "validation": ["2013-01-01", C.VAL_END],
            "test": [C.TEST_START, C.STUDY_END], "live": f"after {C.STUDY_END} (never used for fitting)",
        },
        "rows_per_split": exp.table["split"].value_counts().to_dict(),
        "deployed_model": {
            "name": exp.selected,
            "fitted_on": "train + validation target dates (1981-2012 and 2013-2018)",
            "hyperparameters": exp.tuning.get(exp.selected, {}).get("best_params", {}),
        },
        "selection_rule": (f"Lowest validation MAE among ML candidates; a simpler model within "
                           f"{C.SIMPLICITY_TOLERANCE_C} °C of the best is preferred "
                           f"(simplicity order: {C.MODEL_COMPLEXITY_ORDER})."),
        "tuning": {n: {k: v for k, v in r.items() if k != "cv_results"} for n, r in exp.tuning.items()},
        "validation_metrics": exp.val_metrics.to_dict(orient="index"),
        "test_metrics": exp.test_metrics.to_dict(orient="index"),
        "test_hot_day_metrics": exp.test_hot_metrics.to_dict(orient="index"),
        "heat_indicator": {
            "definition": (f"Tmax above the {C.HOT_PERCENTILE}th percentile of training-period "
                           f"(1981-2012) Tmax within ±{C.HOT_WINDOW_HALF_WIDTH} days of the same "
                           f"calendar day"),
            "note": "Project-defined indicator. Not an IMD heat-wave criterion or alert.",
            "probability_method": ("P(hot) = 1 - Phi((threshold - prediction) / sigma_month); sigma_month = SD "
                                   "of out-of-fold time-series-CV errors on 1981-2012 by target month; "
                                   "flag cut-off chosen on 2013-2018 by maximum F1"),
            "sigma_by_month": exp.heat["sigma_by_month"],
            "cutoff": exp.heat["cutoff"],
            "validation_flag": exp.heat["validation_flag"],
            "validation_persistence_flag": exp.heat["validation_persistence_flag"],
            "test_flag_metrics": exp.test_flag_metrics.to_dict(orient="index"),
            "test_calibration": exp.test_calibration.reset_index(names="chance_bin").astype({"chance_bin": str}).to_dict(orient="records"),
        },
        "input_columns": INPUT_COLUMNS,
        "model_features": MODEL_FEATURES,
        "feature_groups": FEATURE_GROUPS,
        "random_seed": C.RANDOM_SEED,
        "versions": _versions(),
    }
    meta = _clean(meta)
    C.METADATA_FILE.write_text(json.dumps(meta, indent=2, default=str))
    return meta
