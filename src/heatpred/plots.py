"""Matplotlib figures for the notebook and the report (saved to outputs/figures)."""
from __future__ import annotations

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "font.size": 9.5,
    "legend.frameon": False, "lines.linewidth": 1.6, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def _save(fig, name: str):
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES_DIR / f"{name}.png")
    return fig


def eda_overview(daily: pd.DataFrame, threshold: pd.Series):
    """Seasonal cycle, annual trend and exceedance frequency of the training P90."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    doy = daily.index.dayofyear
    clim = daily.groupby(doy)["tmax"].agg(["mean", lambda s: s.quantile(0.1), lambda s: s.quantile(0.9)])
    ax = axes[0]
    ax.fill_between(clim.index, clim.iloc[:, 1], clim.iloc[:, 2], color=BLUE, alpha=0.15, lw=0,
                    label="10th–90th percentile (1981–2025)")
    ax.plot(clim.index, clim["mean"], color=BLUE, label="Mean Tmax (1981–2025)")
    ax.plot(threshold.index, threshold.values, color=ORANGE, ls="--", label="Hot threshold (train P90)")
    ax.set(title="Seasonal cycle of daily max temperature", xlabel="Day of year", ylabel="°C")
    ax.legend(loc="lower center", fontsize=8)

    ax = axes[1]
    trends = {}
    for col, color, name in [("tmax", RED, "Tmax (day)"), ("tmin", BLUE, "Tmin (night)")]:
        annual = daily[col].groupby(daily.index.year).mean()
        slope, icpt = np.polyfit(annual.index, annual.values, 1)
        trends[col] = slope * 10
        ax.plot(annual.index, annual.values, color=color, marker="o", ms=3, lw=1.2,
                label=f"{name}: trend {slope * 10:+.2f} °C/decade")
        ax.plot(annual.index, slope * annual.index + icpt, color=color, ls="--", lw=1)
    slope = trends["tmax"] / 10
    ax.set(title="Annual means of daily Tmax and Tmin", xlabel="Year", ylabel="°C")
    ax.legend(loc="center right")

    thr = threshold.loc[doy].to_numpy()
    exceed = pd.Series(daily["tmax"].to_numpy() > thr, index=daily.index).groupby(daily.index.year).mean() * 100
    ax = axes[2]
    colors = [BLUE if y <= int(C.TRAIN_END[:4]) else ORANGE for y in exceed.index]
    ax.bar(exceed.index, exceed.values, color=colors, width=0.8)
    ax.axhline(10, color=INK2, ls=":", lw=1)
    ax.set(title="% of days above the 1981–2012 P90 threshold", xlabel="Year", ylabel="% of days")
    ax.text(exceed.index.min(), 10.5, "10 % expected by construction (training years)", color=INK2, fontsize=8)
    fig.tight_layout()
    return _save(fig, "eda_overview"), {"trend_c_per_decade": slope * 10, "tmin_trend_c_per_decade": trends["tmin"],
                                        "exceedance_pct_by_year": exceed}


def correlation_with_target(table: pd.DataFrame, target: str, cols: list[str]):
    corr = table.loc[table["split"] == "train", cols + [target]].corr()[target].drop(target)
    corr = corr.reindex(corr.abs().sort_values().index)
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.barh(corr.index, corr.values, color=[RED if v > 0 else BLUE for v in corr.values], height=0.7)
    ax.set(title="Correlation with next-day Tmax (training period)", xlabel="Pearson r")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, "feature_correlations"), corr


def monthly_relationships(daily: pd.DataFrame):
    """Physical sanity checks: wind direction and rain vs Tmax."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    theta = np.deg2rad(daily["wind_dir"])
    u = -daily["wind_speed"] * np.sin(theta)
    ax = axes[0]
    sel = daily.index.month.isin([4, 5, 6])
    ax.scatter(u[sel], daily.loc[sel, "tmax"], s=4, alpha=0.3, color=BLUE)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set(title="Apr–Jun: land wind (from west) vs Tmax", xlabel="Wind u-component (m/s, >0 = from west)",
           ylabel="Tmax (°C)")
    ax = axes[1]
    rain_bins = pd.cut(daily["precip"], [-0.1, 0.5, 5, 20, 500], labels=["<0.5", "0.5–5", "5–20", ">20"])
    anom = daily["tmax"] - daily.groupby(daily.index.dayofyear)["tmax"].transform("mean")
    m = anom.groupby(rain_bins, observed=True).mean()
    ax.bar(m.index.astype(str), m.values, color=[RED if v > 0 else BLUE for v in m.values], width=0.6)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set(title="Same-day rain vs Tmax anomaly", xlabel="Rain (mm/day)", ylabel="Tmax anomaly (°C)")
    fig.tight_layout()
    return _save(fig, "physical_relationships")


def pred_vs_actual(preds: pd.DataFrame, model: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    y, p = preds["actual"], preds[model]
    hot = y > preds["threshold_p90"]
    ax = axes[0]
    ax.scatter(y[~hot], p[~hot], s=5, alpha=0.35, color=BLUE, label="Other days")
    ax.scatter(y[hot], p[hot], s=7, alpha=0.6, color=ORANGE, label="Actually 'unusually hot'")
    lo, hi = min(y.min(), p.min()) - 0.5, max(y.max(), p.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], color=INK2, ls="--", lw=1)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), title=f"{model}: predicted vs actual (test 2019–2025)",
           xlabel="Actual Tmax (°C)", ylabel="Predicted Tmax (°C)")
    ax.legend(loc="upper left")
    resid = p - y
    ax = axes[1]
    ax.hist(resid, bins=60, color=BLUE, alpha=0.85)
    ax.axvline(0, color=INK2, lw=1)
    ax.set(title=f"Residuals (mean {resid.mean():+.2f}, SD {resid.std():.2f} °C)",
           xlabel="Predicted − actual (°C)", ylabel="Days")
    ax = axes[2]
    month = pd.to_datetime(preds["target_date"]).dt.month
    data = [resid[month == m].to_numpy() for m in range(1, 13)]
    ax.boxplot(data, tick_labels=list("JFMAMJJASOND"), showfliers=False,
               medianprops={"color": ORANGE}, boxprops={"color": INK2}, whiskerprops={"color": INK2})
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set(title="Residuals by month", ylabel="Predicted − actual (°C)")
    fig.tight_layout()
    return _save(fig, "pred_vs_actual_residuals")


def residual_vs_change(preds: pd.DataFrame, model: str, tmax_today: pd.Series):
    """Errors are largest on days with large day-to-day change."""
    fig, ax = plt.subplots(figsize=(6, 4))
    change = preds["actual"] - tmax_today.to_numpy()
    resid = preds[model] - preds["actual"]
    ax.scatter(change, resid, s=4, alpha=0.3, color=BLUE)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set(title="Residual vs actual day-to-day change (test)", xlabel="Actual Tmax(D+1) − Tmax(D) (°C)",
           ylabel="Predicted − actual (°C)")
    fig.tight_layout()
    return _save(fig, "residual_vs_change")


def time_series(preds: pd.DataFrame, model: str, year: int):
    d = preds[pd.to_datetime(preds["target_date"]).dt.year == year]
    t = pd.to_datetime(d["target_date"])
    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.plot(t, d["actual"], color=INK2, label="Actual", lw=1.4)
    ax.plot(t, d[model], color=BLUE, label=f"Predicted ({model})", lw=1.4)
    ax.plot(t, d["threshold_p90"], color=ORANGE, ls="--", lw=1.2, label="Hot threshold (P90)")
    ax.set(title=f"Test period {year}: daily max temperature", ylabel="°C")
    ax.legend(loc="upper right", ncols=3)
    fig.tight_layout()
    return _save(fig, f"timeseries_{year}")


def bar_importance(df: pd.DataFrame, value: str, label: str, name: str, top: int = 15):
    d = df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(d["feature"], d[value], color=[RED if v > 0 else BLUE for v in d[value]], height=0.7)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set(title=label, xlabel=value)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, name)


def model_comparison(test_metrics: pd.DataFrame, selected: str):
    m = test_metrics["MAE"].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    colors = [ORANGE if i == selected else (MUTED if "Persist" in i or "Seasonal" in i else BLUE) for i in m.index]
    ax.barh(m.index, m.values, color=colors, height=0.6)
    for i, v in enumerate(m.values):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", color=INK, fontsize=9)
    ax.set(title="Test-period MAE (2019–2025), lower is better", xlabel="MAE (°C)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, m.max() * 1.15)
    fig.tight_layout()
    return _save(fig, "model_comparison_test_mae")
