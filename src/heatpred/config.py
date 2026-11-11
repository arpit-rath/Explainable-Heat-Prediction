"""Central configuration: location, data request, study periods and seeds.

Every other module and the notebook read their constants from here, so the
whole project can be reproduced by changing values in one place.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Location: Chennai (central city coordinates). NASA POWER returns the value
# of the MERRA-2 / GEOS-IT grid cell (0.5 deg lat x 0.625 deg lon) containing
# this point, NOT a station measurement.
# ---------------------------------------------------------------------------
LOCATION_NAME = "Chennai, Tamil Nadu, India"
LATITUDE = 13.0827
LONGITUDE = 80.2707

# ---------------------------------------------------------------------------
# NASA POWER Daily API request
# Docs: https://power.larc.nasa.gov/docs/services/api/temporal/daily/
# ---------------------------------------------------------------------------
POWER_DAILY_POINT_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_COMMUNITY = "AG"          # Agroclimatology community (units: degC, mm/day, m/s)
POWER_TIME_STANDARD = "LST"     # Local Solar Time (API default, set explicitly)
POWER_FILL_VALUE = -999.0       # Missing-value code reported in the response header

# NASA POWER parameter code -> project column name
POWER_PARAMETERS: dict[str, str] = {
    "T2M_MAX": "tmax",          # Maximum air temperature at 2 m (degC)
    "T2M_MIN": "tmin",          # Minimum air temperature at 2 m (degC)
    "T2M": "tmean",             # Mean air temperature at 2 m (degC)
    "T2MDEW": "dewpoint",       # Dew/frost point at 2 m (degC)
    "RH2M": "rh",               # Relative humidity at 2 m (%)
    "QV2M": "qv",               # Specific humidity at 2 m (g/kg)
    "PRECTOTCORR": "precip",    # Bias-corrected precipitation (mm/day)
    "WS2M": "wind_speed",       # Wind speed at 2 m (m/s)
    "WD2M": "wind_dir",         # Wind direction at 2 m (degrees, direction wind blows FROM)
    "PS": "pressure",           # Surface pressure (kPa)
}

UNITS: dict[str, str] = {
    "tmax": "°C", "tmin": "°C", "tmean": "°C", "dewpoint": "°C", "rh": "%",
    "qv": "g/kg", "precip": "mm/day", "wind_speed": "m/s", "wind_dir": "degrees",
    "pressure": "kPa",
}

# ---------------------------------------------------------------------------
# Study period and chronological split (assigned by TARGET date, i.e. the day
# whose maximum temperature is being predicted).
# ---------------------------------------------------------------------------
STUDY_START = "1981-01-01"      # first day of NASA POWER daily meteorology
STUDY_END = "2025-12-31"        # fixed end of the frozen modelling snapshot
TRAIN_END = "2012-12-31"        # train:      1981 .. 2012 (32 years)
VAL_END = "2018-12-31"          # validation: 2013 .. 2018 (6 years)
TEST_START = "2019-01-01"       # test:       2019 .. 2025 (7 years, untouched until selection)
# Anything after STUDY_END is "live" data fetched by the app: never used in
# training, tuning or model selection.

WARMUP_DAYS = 30                # longest rolling window; earlier rows are dropped

# ---------------------------------------------------------------------------
# Project-defined heat indicator ("unusually hot for Chennai")
# ---------------------------------------------------------------------------
HOT_PERCENTILE = 90             # percentile of Tmax
HOT_WINDOW_HALF_WIDTH = 7       # +/- days around each calendar day (15-day window)

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
CV_SPLITS = 5
CV_GAP_DAYS = 30                # gap between CV train and CV validation folds
# Selection rule (declared before looking at the test period): choose the
# candidate with the lowest validation MAE, unless a simpler/more
# interpretable candidate is within this tolerance, in which case prefer it.
SIMPLICITY_TOLERANCE_C = 0.05
MODEL_COMPLEXITY_ORDER = ["Linear Regression", "Random Forest", "SVR (RBF)"]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

SNAPSHOT_CSV = DATA_DIR / "chennai_nasa_power_daily_1981_2025.csv"
SNAPSHOT_META = DATA_DIR / "chennai_nasa_power_daily_1981_2025.meta.json"
RECENT_CACHE_CSV = DATA_DIR / "recent_cache.csv"
RECENT_CACHE_META = DATA_DIR / "recent_cache.meta.json"
MODEL_FILE = MODELS_DIR / "final_model.joblib"
METADATA_FILE = MODELS_DIR / "metadata.json"
THRESHOLD_CSV = MODELS_DIR / "hot_threshold_p90.csv"
REFERENCE_CSV = MODELS_DIR / "seasonal_reference_inputs.csv"
BACKTEST_CSV = OUTPUTS_DIR / "backtest_predictions.csv"
