# Explainable Next-Day Extreme Heat Prediction for Chennai

Predicts the **next day's maximum air temperature (Tmax) in Chennai** from weather observed up to and
including today. It flags days that would be *unusually hot for Chennai* and explains every prediction
in plain language. Data: NASA POWER Daily API (gridded MERRA-2 / GEOS-IT reanalysis). Interface: a
Streamlit app with a next-day prediction and an interactive historical backtest.

> College climate-change project. **Not an India Meteorological Department (IMD) forecast or heat-wave
> alert.** NASA POWER values are grid-cell averages (½° × ⅝°) and can differ from Chennai station readings.

## Results (held-out test period 2019–2025, 2,557 days)

| Model | MAE (°C) | MSE (°C²) | R² |
|---|---|---|---|
| Persistence (tomorrow = today) | 0.854 | 1.456 | 0.868 |
| Seasonal average (1981–2018) | 1.195 | 2.511 | 0.772 |
| **Linear Regression (deployed)** | **0.766** | **1.118** | **0.899** |
| SVR (RBF) | 0.741 | 1.069 | 0.903 |
| Random Forest | 0.754 | 1.090 | 0.901 |

* The deployed Linear Regression has a 10 % lower MAE than persistence. It was chosen by a rule fixed
  in advance: lowest validation MAE, but prefer a simpler model within 0.05 °C. SVR was 0.023 °C better
  on validation.
* **Weakness:** on the 192 test days that turned out unusually hot, every regression model
  under-predicts by about 1 °C. The probability-based heat flag catches 44 % of those days with 52 %
  precision (F1 0.47), while persistence scores F1 0.49. Details are in
  [reports/technical_report.md](reports/technical_report.md).

These numbers come from `outputs/test_metrics.csv`, `outputs/test_heat_flag_metrics.csv` and the run
logs in `outputs/run_logs/`.

## Repository layout

```
app.py                         Streamlit app (public interface)
notebooks/Chennai_Next_Day_Heat_Prediction.ipynb   Colab notebook: data → EDA → models → artifacts
scripts/train.py               same pipeline headless (python scripts/train.py)
src/heatpred/                  config, data download & QC, features, models, heat probability, explanations
data/                          frozen NASA POWER snapshot 1981–2025 + recent-data cache (with request metadata)
models/                        final_model.joblib, metadata.json, heat threshold, seasonal reference inputs
outputs/                       metrics CSVs, backtest_predictions.csv, figures/, run_logs/
reports/                       technical report draft, system diagram, demo script
tests/test_leakage.py          leakage / correctness tests (pytest)
```

## Run the notebook in Google Colab

1. Open `https://colab.research.google.com/github/arpit-rath/Explainable-Heat-Prediction/blob/main/notebooks/Chennai_Next_Day_Heat_Prediction.ipynb`
   (this link works once the repository is pushed and public). Alternatively, upload the `.ipynb`
   through *File → Upload notebook*.
2. *Runtime → Run all.* The first cell clones the repository and installs `requirements.txt`.
3. Settings in the first cell:
   * `RUN_TUNING = True` runs the full hyperparameter search, which is most of the run time. On a
     16-thread laptop it took about 1.5 min on mains power and about 26 min when throttled on battery;
     it has not been timed on Colab, whose free runtime has fewer cores. `False` reuses the
     hyperparameters recorded in `models/metadata.json`.
   * `DOWNLOAD_FRESH = True` re-downloads 1981–2025 from NASA POWER and reports any difference from
     the bundled snapshot. Results are computed on the bundled snapshot so they are reproducible.

## Run locally

```bash
python -m venv .venv
```
```bash
.venv/Scripts/python -m pip install -r requirements-dev.txt
```
(on macOS/Linux use `.venv/bin/python`.)

```bash
.venv/Scripts/python scripts/train.py
```
```bash
.venv/Scripts/python -m pytest -q
```
```bash
.venv/Scripts/python -m streamlit run app.py
```

`scripts/train.py` rebuilds every artifact in `models/` and `outputs/` and refreshes
`data/recent_cache.csv`. The run is deterministic (seed 42), and a re-run reproduced the logged test
metrics exactly.

## Data

* **Request** (exact, reproducible):
  `https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M_MAX,T2M_MIN,T2M,T2MDEW,RH2M,QV2M,PRECTOTCORR,WS2M,WD2M,PS&community=AG&longitude=80.2707&latitude=13.0827&start=19810101&end=20251231&format=JSON&time-standard=LST`
* API v2.10.3. Sources: MERRA-2 (snapshot), GEOS-IT (latest weeks). Fill value −999. Time standard
  LST. Units: °C, %, g/kg, mm/day, m/s, degrees, kPa.
* **Quality:** 16,436 rows × 10 columns, 1981-01-01 to 2025-12-31. No missing calendar days, no
  duplicate dates, no −999 values, nothing outside physical ranges, Tmin ≤ Tmean ≤ Tmax on every day.
* Retrieval timestamps are stored in `data/*.meta.json`.

## Leakage safeguards

* Row D uses observations up to and including D only; rolling windows end on D. The target is Tmax(D+1).
* Splits are by target date: train 1981–2012, validation 2013–2018, test 2019–2025. Data after 2025
  is used only by the app.
* The seasonal normal, imputation medians and scaling sit inside the scikit-learn pipeline, so they are
  fitted on training folds only. The heat threshold uses 1981–2012 target days only.
* Tuning: `TimeSeriesSplit(n_splits=5, gap=30)` inside 1981–2012. The heat-flag cut-off was chosen on
  2013–2018.
* The app truncates the data at the day before the selected date *before* computing features. Tests
  check that (a) altering future data never changes past features, (b) altering the target day does not
  change its prediction, (c) app predictions equal `outputs/backtest_predictions.csv`, and (d) local
  explanations sum exactly to the prediction.

**Historical predictions (`outputs/backtest_predictions.csv`)** are generated by the notebook or
`scripts/train.py`. For every target date in 1981–2025 the deployed model (fitted on 1981–2018)
predicts from that date's previous-day features. Rows before 2019 are therefore *in-sample* and are
flagged `in_sample_for_deployed_model=True`. The app labels them the same way.

## Deploy to Streamlit Community Cloud (not done yet)

1. Push this repository to GitHub (public, or private with Streamlit access).
2. At share.streamlit.io choose *Create app → from GitHub*, pick the repository and branch `main`,
   set the main file to `app.py`, and under *Advanced settings* choose **Python 3.12**.
3. Deploy. Dependencies come from `requirements.txt`; scikit-learn is pinned to 1.9.0 because the saved
   model was built with it. No secrets or uploads are needed: the snapshot, model and recent-data cache
   ship with the repository, and the app fetches new data from NASA POWER itself (cached for 3 hours,
   falling back to the bundled cache if the API is unavailable).

## Data citation

These data were obtained from the NASA Langley Research Center (LaRC) POWER Project, funded through the
NASA Earth Science Division (POWER Daily API v2.10.3, accessed 10 November 2026).
