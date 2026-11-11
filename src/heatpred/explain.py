"""Global and local explanations of the deployed (linear) model.

Local explanation - exact, no approximation
-------------------------------------------
For Linear Regression inside the pipeline the prediction is

    prediction = intercept + sum_j coef_j * z_j

where z_j is feature j after imputation and standardisation (fitted on
training data). For a reference input vector z_ref the same identity gives

    prediction = [intercept + sum_j coef_j * z_ref_j]  +  sum_j coef_j * (z_j - z_ref_j)
               =            base value                 +  contributions

We use as reference the *average 1981-2018 day at the same time of year*
(mean model inputs over target days within +/-7 calendar days). The base
value is then "what the model predicts on a normal day for this date", and
each contribution says how much today's departure from normal in that input
moved the prediction. The decomposition is exact because the model is linear.

These are statistical associations learned from historical data, not
physical cause-and-effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .features import FEATURE_DESCRIPTIONS, FEATURE_GROUPS, INPUT_COLUMNS


def _require_linear(pipeline):
    model = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        raise NotImplementedError("Exact local explanations are implemented for the linear model only.")
    return model


def seasonal_reference(pipeline, X_fit: pd.DataFrame, half_width: int = C.HOT_WINDOW_HALF_WIDTH) -> pd.DataFrame:
    """Mean model inputs (imputed, unscaled) for each target calendar day 1..366."""
    raw = pipeline[:-2].transform(X_fit[INPUT_COLUMNS])        # climatology + imputer
    doy = X_fit["target_doy"].to_numpy(int)
    rows = {}
    for day in range(1, 367):
        dist = np.abs(doy - day)
        dist = np.minimum(dist, 366 - dist)
        rows[day] = raw[dist <= half_width].mean()
    ref = pd.DataFrame(rows).T
    ref.index.name = "target_doy"
    return ref


def explain_row(pipeline, X_row: pd.DataFrame, reference: pd.DataFrame | None = None) -> tuple[float, pd.DataFrame]:
    """Return (base_value, per-feature contributions) for one row.

    With ``reference`` (from :func:`seasonal_reference`) contributions are
    relative to a normal day for the target date; without it, relative to the
    average training day (base value = intercept).
    """
    model = _require_linear(pipeline)
    scaler = pipeline.named_steps["scale"]
    raw = pipeline[:-2].transform(X_row[INPUT_COLUMNS])
    z = scaler.transform(raw).to_numpy()[0]
    names = list(raw.columns)
    if reference is not None:
        ref_raw = reference.loc[[int(X_row["target_doy"].iloc[0])], names]
        z_ref = scaler.transform(ref_raw).to_numpy()[0]
    else:
        z_ref = np.zeros_like(z)
    base = float(model.intercept_ + np.dot(model.coef_, z_ref))
    contrib = model.coef_ * (z - z_ref)
    ref_values = scaler.inverse_transform(z_ref.reshape(1, -1))[0]
    table = pd.DataFrame({
        "feature": names,
        "group": [FEATURE_GROUPS.get(n, "Other") for n in names],
        "description": [FEATURE_DESCRIPTIONS.get(n, n) for n in names],
        "value": raw.iloc[0].to_numpy(),
        "normal for date": ref_values,
        "contribution": contrib,
    })
    return base, table.sort_values("contribution", key=np.abs, ascending=False).reset_index(drop=True)


def group_contributions(table: pd.DataFrame) -> pd.DataFrame:
    g = table.groupby("group")["contribution"].sum().reset_index()
    return g.sort_values("contribution", key=np.abs, ascending=False).reset_index(drop=True)


def linear_coefficients(pipeline) -> pd.DataFrame:
    """Standardised coefficients: °C change in prediction per 1 training-SD change."""
    model = pipeline.named_steps["model"]
    scale = pipeline.named_steps["scale"]
    names = list(scale.feature_names_in_)
    out = pd.DataFrame({
        "feature": names,
        "group": [FEATURE_GROUPS.get(n, "Other") for n in names],
        "coef_per_SD (°C)": model.coef_,
        "training SD": scale.scale_,
        "coef_per_unit": model.coef_ / scale.scale_,
    })
    return out.sort_values("coef_per_SD (°C)", key=np.abs, ascending=False).reset_index(drop=True)


EXAMPLE_UNITS = {
    **{f"tmax_lag{k}": "°C" for k in range(7)}, "tmax_mean_3": "°C", "tmax_mean_7": "°C",
    "tmax_mean_14": "°C", "tmax_mean_30": "°C", "tmin_lag0": "°C", "tmin_lag1": "°C", "tmin_mean_7": "°C",
    "tmean_lag0": "°C", "dtr_lag0": "°C", "tmax_anom_lag0": "°C", "tmax_mean_7_anom": "°C",
    "dewpoint_lag0": "°C", "rh_lag0": "%", "rh_lag1": "%", "rh_mean_7": "%", "qv_lag0": "g/kg",
    "precip_lag0": "mm", "precip_lag1": "mm", "precip_sum_3": "mm", "precip_sum_7": "mm",
    "precip_sum_30": "mm", "wind_u_lag0": "m/s", "wind_v_lag0": "m/s", "wind_u_mean_3": "m/s",
    "wind_speed_lag0": "m/s", "pressure_lag0": "kPa", "years_since_1981": "years",
}


def _fmt(v: float) -> str:
    return f"{abs(v):.1f} °C"


def plain_language(base: float, prediction: float, table: pd.DataFrame,
                   threshold: float | None = None, clim_normal: float | None = None,
                   chance: float | None = None, error_sd: float | None = None,
                   top_n: int = 3) -> list[str]:
    """Short sentences for non-specialists, built from grouped contributions."""
    groups = group_contributions(table)
    lines = []
    if clim_normal is not None and not np.isnan(clim_normal):
        diff = prediction - clim_normal
        word = "warmer" if diff > 0 else "cooler"
        lines.append(f"The model expects **{prediction:.1f} °C**, which is {_fmt(diff)} {word} than "
                     f"the 1981–2018 average of {clim_normal:.1f} °C for this date.")
    lines.append(f"Starting point: on a normal day at this time of year (1981–2018 average conditions) "
                 f"the model would predict {base:.1f} °C. How the latest conditions differ from normal "
                 f"moves it to {prediction:.1f} °C:")
    for _, r in groups.head(top_n).iterrows():
        if abs(r.contribution) < 0.05:
            continue
        direction = "raised" if r.contribution > 0 else "lowered"
        line = f"- **{r.group}** {direction} the prediction by about {_fmt(r.contribution)}"
        if "normal for date" in table:
            top = table[table.group == r.group].iloc[0]          # table is sorted by |contribution|
            if top.feature in EXAMPLE_UNITS:
                desc = top.description[0].lower() + top.description[1:]
                line += (f" (largest single factor: {desc} was "
                         f"{top.value:.1f} {EXAMPLE_UNITS[top.feature]} vs a normal "
                         f"{top['normal for date']:.1f})")
        lines.append(line + ".")
    if threshold is not None and not np.isnan(threshold):
        side = "above" if prediction > threshold else "below"
        msg = (f"The best guess is {side} the project's 'unusually hot for Chennai' line of "
               f"{threshold:.1f} °C for this date.")
        if chance is not None:
            spread = f"about ±{error_sd:.1f} °C" if error_sd else "around a degree"
            msg += (f" Because the model's typical error in this month is {spread}, the chance that "
                    f"the actual value ends up above the line is about {chance:.0%}.")
        lines.append(msg)
    lines.append("These effects are patterns learned from past data (associations), not proof of cause.")
    return lines
