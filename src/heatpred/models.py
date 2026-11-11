"""Model pipelines, baselines, time-series tuning and metrics."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from . import config as C
from .features import INPUT_COLUMNS, ClimatologyTransformer


# ---------------------------------------------------------------------------
# Baselines (implemented as estimators so they share the evaluation code)
# ---------------------------------------------------------------------------
class PersistenceBaseline(BaseEstimator, RegressorMixin):
    """Tomorrow's maximum temperature equals today's."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.asarray(X["tmax_lag0"], float)


class SeasonalBaseline(BaseEstimator, RegressorMixin):
    """Tomorrow equals the training-period average Tmax for that calendar day."""

    def fit(self, X, y=None):
        self.clim_ = ClimatologyTransformer().fit(X)
        return self

    def predict(self, X):
        return self.clim_.transform(X)["clim_tmax_target"].to_numpy(float)


# ---------------------------------------------------------------------------
# Candidate ML pipelines. Every learned statistic is inside the pipeline, so
# it is re-fitted on the training part of each CV fold.
# ---------------------------------------------------------------------------
def make_pipeline(name: str) -> Pipeline:
    steps = [("climatology", ClimatologyTransformer()),
             ("impute", SimpleImputer(strategy="median"))]
    if name == "Linear Regression":
        steps += [("scale", StandardScaler()), ("model", LinearRegression())]
    elif name == "SVR (RBF)":
        steps += [("scale", StandardScaler()), ("model", SVR(kernel="rbf"))]
    elif name == "Random Forest":
        steps += [("model", RandomForestRegressor(random_state=C.RANDOM_SEED, n_jobs=-1))]
    else:
        raise ValueError(name)
    return Pipeline(steps).set_output(transform="pandas")


PARAM_SPACES = {
    "Linear Regression": None,  # no hyperparameters; CV score still reported
    # A first, smaller grid (C<=10, gamma in {scale, 0.005}) put the optimum on
    # the grid edge (C=10, gamma=0.005), so the grid was widened around it.
    "SVR (RBF)": {
        "model__C": [3.0, 10.0, 30.0],
        "model__epsilon": [0.3, 0.5],
        "model__gamma": [0.002, 0.005, 0.01],
    },
    "Random Forest": {
        "model__n_estimators": [200],
        "model__max_depth": [None, 12, 20],
        "model__min_samples_leaf": [1, 5, 10, 20],
        "model__max_features": [0.33, 0.5, 1.0],
    },
}
RF_SEARCH_ITER = 8


def time_series_cv() -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=C.CV_SPLITS, gap=C.CV_GAP_DAYS)


def tune(name: str, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Time-series CV on the training period only. Returns best params + CV MAE."""
    cv = time_series_cv()
    space = PARAM_SPACES[name]
    t0 = time.time()
    if space is None:
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(make_pipeline(name), X_train, y_train, cv=cv,
                                 scoring="neg_mean_absolute_error")
        return {"model": name, "best_params": {}, "cv_mae": float(-scores.mean()),
                "cv_mae_std": float(scores.std()), "n_candidates": 1,
                "seconds": round(time.time() - t0, 1)}
    if name == "Random Forest":
        search = RandomizedSearchCV(make_pipeline(name), space, n_iter=RF_SEARCH_ITER, cv=cv,
                                    scoring="neg_mean_absolute_error",
                                    random_state=C.RANDOM_SEED, n_jobs=1)
    else:
        search = GridSearchCV(make_pipeline(name), space, cv=cv,
                              scoring="neg_mean_absolute_error", n_jobs=-1)
    search.fit(X_train, y_train)
    best = search.best_index_
    return {"model": name, "best_params": search.best_params_,
            "cv_mae": float(-search.best_score_),
            "cv_mae_std": float(search.cv_results_["std_test_score"][best]),
            "n_candidates": int(len(search.cv_results_["params"])),
            "seconds": round(time.time() - t0, 1),
            "cv_results": pd.DataFrame(search.cv_results_)}


def build_estimator(name: str, params: dict | None = None):
    if name == "Persistence (tomorrow = today)":
        return PersistenceBaseline()
    if name == "Seasonal average":
        return SeasonalBaseline()
    est = make_pipeline(name)
    if params:
        est.set_params(**params)
    return est


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "Bias (pred - actual)": float(np.mean(y_pred - y_true)),
        "n": int(len(y_true)),
    }


def hot_day_metrics(y_true, y_pred, threshold) -> dict:
    """Errors on actually-hot days and detection skill for the heat indicator."""
    y_true, y_pred, thr = (np.asarray(a, float) for a in (y_true, y_pred, threshold))
    hot = y_true > thr
    flagged = y_pred > thr
    tp = int((hot & flagged).sum())
    out = {
        "hot_days": int(hot.sum()),
        "MAE_on_hot_days": float(np.mean(np.abs(y_pred[hot] - y_true[hot]))) if hot.any() else np.nan,
        "Bias_on_hot_days": float(np.mean(y_pred[hot] - y_true[hot])) if hot.any() else np.nan,
        "hit_rate (recall)": tp / hot.sum() if hot.any() else np.nan,
        "precision": tp / flagged.sum() if flagged.any() else np.nan,
        "false_alarms": int((~hot & flagged).sum()),
        "missed": int((hot & ~flagged).sum()),
    }
    return out


def select_model(val_table: pd.DataFrame) -> str:
    """Apply the pre-declared selection rule to validation MAE of ML candidates."""
    ml = val_table[val_table.index.isin(C.MODEL_COMPLEXITY_ORDER)]
    best_mae = ml["MAE"].min()
    for name in C.MODEL_COMPLEXITY_ORDER:          # simplest first
        if name in ml.index and ml.loc[name, "MAE"] <= best_mae + C.SIMPLICITY_TOLERANCE_C:
            return name
    return ml["MAE"].idxmin()


def X_of(table: pd.DataFrame) -> pd.DataFrame:
    return table[INPUT_COLUMNS]


def fit_clone(est, X, y):
    return clone(est).fit(X, y)
