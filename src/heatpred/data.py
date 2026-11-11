"""Download, clean and quality-check NASA POWER daily data for Chennai."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

from . import config as C


class PowerAPIError(RuntimeError):
    """Raised when the NASA POWER API cannot return usable data."""


def _yyyymmdd(day) -> str:
    return pd.Timestamp(day).strftime("%Y%m%d")


def build_request_params(start, end) -> dict[str, str]:
    """Exact query parameters sent to the NASA POWER Daily point endpoint."""
    return {
        "parameters": ",".join(C.POWER_PARAMETERS),
        "community": C.POWER_COMMUNITY,
        "longitude": f"{C.LONGITUDE}",
        "latitude": f"{C.LATITUDE}",
        "start": _yyyymmdd(start),
        "end": _yyyymmdd(end),
        "format": "JSON",
        "time-standard": C.POWER_TIME_STANDARD,
    }


def build_request_url(start, end) -> str:
    """Human-readable, reproducible request URL (commas left unescaped)."""
    return C.POWER_DAILY_POINT_URL + "?" + urlencode(build_request_params(start, end), safe=",")


def fetch_power_json(start, end, timeout: float = 90, retries: int = 3,
                     backoff: float = 5.0) -> dict:
    """Call the API with retries. Raises PowerAPIError on failure."""
    url = build_request_url(start, end)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                payload = resp.json()
                if "properties" not in payload:
                    raise PowerAPIError(f"Unexpected response: {str(payload)[:300]}")
                return payload
            detail = resp.text[:300] if "json" in resp.headers.get("content-type", "") else resp.reason
            last_err = PowerAPIError(f"HTTP {resp.status_code}: {detail}")
            if resp.status_code in (400, 422):      # validation error: retrying won't help
                break
        except (requests.RequestException, ValueError) as err:
            last_err = err
        if attempt < retries:
            time.sleep(backoff * attempt)
    raise PowerAPIError(f"NASA POWER request failed after {retries} attempt(s): {last_err}")


def payload_to_frame(payload: dict) -> pd.DataFrame:
    """Convert API JSON to a date-indexed DataFrame; fill values become NaN."""
    params = payload["properties"]["parameter"]
    fill = payload.get("header", {}).get("fill_value", C.POWER_FILL_VALUE)
    df = pd.DataFrame({code: pd.Series(values) for code, values in params.items()})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    df = df.replace(fill, np.nan).astype(float)
    df = df.rename(columns=C.POWER_PARAMETERS)
    ordered = [c for c in C.POWER_PARAMETERS.values() if c in df.columns]
    return df[ordered].sort_index()


def payload_metadata(payload: dict, start, end) -> dict:
    header = payload.get("header", {})
    return {
        "source": "NASA POWER Daily API",
        "api_version": header.get("api", {}).get("version"),
        "data_sources": header.get("sources"),
        "fill_value": header.get("fill_value"),
        "time_standard": header.get("time_standard"),
        "grid_cell_geometry": payload.get("geometry"),
        "parameters": payload.get("parameters"),
        "request_url": build_request_url(start, end),
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "requested_start": str(pd.Timestamp(start).date()),
        "requested_end": str(pd.Timestamp(end).date()),
    }


def download(start=C.STUDY_START, end=C.STUDY_END, **kw) -> tuple[pd.DataFrame, dict]:
    payload = fetch_power_json(start, end, **kw)
    return payload_to_frame(payload), payload_metadata(payload, start, end)


def save_frame(df: pd.DataFrame, meta: dict, csv_path: Path, meta_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, float_format="%.2f")
    meta_path.write_text(json.dumps(meta, indent=2))


def load_frame(csv_path: Path, meta_path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    meta = json.loads(meta_path.read_text()) if meta_path and meta_path.exists() else {}
    return df, meta


def load_or_download_snapshot(force: bool = False) -> tuple[pd.DataFrame, dict]:
    """Return the frozen 1981-2025 snapshot, downloading it if absent."""
    if C.SNAPSHOT_CSV.exists() and not force:
        return load_frame(C.SNAPSHOT_CSV, C.SNAPSHOT_META)
    df, meta = download(C.STUDY_START, C.STUDY_END)
    save_frame(df, meta, C.SNAPSHOT_CSV, C.SNAPSHOT_META)
    return df, meta


def complete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where every parameter is present (used to find the latest usable day)."""
    return df.dropna(how="any")


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------
PHYSICAL_LIMITS = {          # generous plausibility bounds for coastal Chennai
    "tmax": (10, 50), "tmin": (5, 40), "tmean": (5, 45), "dewpoint": (-5, 35),
    "rh": (0, 100), "qv": (0, 35), "precip": (0, 500), "wind_speed": (0, 40),
    "wind_dir": (0, 360), "pressure": (90, 105),
}


def quality_report(df: pd.DataFrame) -> dict:
    """Explicit data-quality summary used in the notebook and README."""
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    missing_dates = full_range.difference(df.index)
    invalid = {}
    for col, (lo, hi) in PHYSICAL_LIMITS.items():
        if col in df:
            s = df[col]
            invalid[col] = int(((s < lo) | (s > hi)).sum())
    report = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "first_date": str(df.index.min().date()),
        "last_date": str(df.index.max().date()),
        "expected_days_in_range": int(len(full_range)),
        "missing_calendar_dates": int(len(missing_dates)),
        "duplicate_dates": int(df.index.duplicated().sum()),
        "missing_values_per_column": {c: int(v) for c, v in df.isna().sum().items()},
        "out_of_physical_range_per_column": invalid,
        "tmin_greater_than_tmax": int((df["tmin"] > df["tmax"]).sum()),
        "tmean_outside_tmin_tmax": int(((df["tmean"] < df["tmin"] - 0.01) |
                                        (df["tmean"] > df["tmax"] + 0.01)).sum()),
    }
    return report


# Is each raw variable observed by the end of the issue day D (when the
# forecast for D+1 is made)? All are daily aggregates of day D or earlier.
FEATURE_AVAILABILITY = pd.DataFrame(
    [
        ("tmax", "Max temperature of day D and earlier days", "Yes - known after day D ends"),
        ("tmin", "Min temperature of day D and earlier", "Yes"),
        ("tmean", "Mean temperature of day D", "Yes"),
        ("dewpoint", "Dew point of day D", "Yes"),
        ("rh", "Relative humidity of day D and earlier", "Yes"),
        ("qv", "Specific humidity of day D", "Yes"),
        ("precip", "Rainfall of day D and earlier", "Yes"),
        ("wind_speed / wind_dir", "Wind of day D", "Yes"),
        ("pressure", "Surface pressure of day D and earlier", "Yes"),
        ("calendar", "Day-of-year of the target day D+1", "Yes - the calendar is always known"),
        ("tmax of D+1", "The TARGET", "No - never used as a feature"),
    ],
    columns=["variable", "used as", "available at prediction time?"],
)
