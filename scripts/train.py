"""Reproduce the whole study headlessly (same stages as the notebook).

    python scripts/train.py            # full run incl. hyperparameter search
    python scripts/train.py --no-live  # skip refreshing data/recent_cache.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heatpred import config as C, experiment as E, live, plots  # noqa: E402
from heatpred.data import quality_report  # noqa: E402
from heatpred.explain import linear_coefficients  # noqa: E402
from heatpred.features import TARGET  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-live", action="store_true", help="do not refresh the bundled recent-data cache")
    ap.add_argument("--redownload", action="store_true", help="re-download the 1981-2025 snapshot")
    args = ap.parse_args(argv)

    exp = E.prepare(force_download=args.redownload)
    print("Data quality:", quality_report(exp.daily))
    print(E.split_summary(exp))

    E.fit_threshold(exp)
    plots.eda_overview(exp.daily, exp.threshold)

    print("\n== Time-series CV tuning (training period only) ==")
    print(E.tune_candidates(exp).drop(columns="best_params"))

    print("\n== Validation period 2013-2018 ==")
    print(E.validation_comparison(exp).round(3))
    print("Selected by rule:", exp.selected)
    print("Heat-probability error SD by month (training CV):",
          {k: round(v, 2) for k, v in exp.heat["sigma_by_month"].items()})
    print("Flag cut-off chosen on validation (max F1):", exp.heat["cutoff"])
    print(exp.val_cutoff_table.round(3).to_string(index=False))
    print("Validation calibration:"); print(exp.val_calibration.round(3))

    print("\n== Final test period 2019-2025 (models refitted on 1981-2018) ==")
    print(E.final_test_evaluation(exp).round(3))
    print(exp.test_hot_metrics.round(3))
    print(); print("== Heat flag on test ==")
    print(exp.test_flag_metrics.round(3))
    print("Test calibration:"); print(exp.test_calibration.round(3))

    imp = E.permutation_importance_table(exp)
    print(imp.head(10).round(3))
    plots.model_comparison(exp.test_metrics, exp.selected)
    plots.pred_vs_actual(exp.test_predictions, exp.selected)
    plots.time_series(exp.test_predictions, exp.selected, 2024)
    plots.residual_vs_change(exp.test_predictions, exp.selected, exp.part("test")["tmax_lag0"])
    plots.bar_importance(imp, "MAE increase (°C)", "Permutation importance (test period)", "permutation_importance")
    model = exp.final_models[exp.selected]
    if hasattr(model.named_steps["model"], "coef_"):
        plots.bar_importance(linear_coefficients(model), "coef_per_SD (°C)",
                             "Linear model: standardised coefficients", "linear_coefficients")
    meta = E.save_artifacts(exp, imp)
    print("Saved model:", C.MODEL_FILE, "| deployed:", meta["deployed_model"])

    if not args.no_live:
        try:
            recent, rmeta = live.fetch_recent()
            live.save_recent_cache(recent, rmeta)
            print("Recent cache saved through", recent.dropna().index.max().date())
        except Exception as err:  # noqa: BLE001
            print("Could not refresh recent cache:", err)


if __name__ == "__main__":
    main()
