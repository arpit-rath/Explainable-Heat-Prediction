"""Combine the frozen snapshot with recent NASA POWER data for the app.

Order of preference for data after the snapshot ends:
  1. live request to the NASA POWER API (cached by the app for a few hours),
  2. the bundled ``data/recent_cache.csv`` (with its retrieval time shown),
  3. snapshot only (predictions limited to the snapshot period).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from . import config as C
from . import data as D


@dataclass
class DataStatus:
    source: str                    # "live API", "bundled cache", "snapshot only"
    retrieved_utc: str | None
    last_complete_date: pd.Timestamp
    message: str = ""


def _merge(snapshot: pd.DataFrame, recent: pd.DataFrame | None) -> pd.DataFrame:
    if recent is None or recent.empty:
        return snapshot
    recent = recent[recent.index > snapshot.index.max()]
    return pd.concat([snapshot, recent]).sort_index()


def fetch_recent(today: pd.Timestamp | None = None, timeout: float = 45) -> tuple[pd.DataFrame, dict]:
    today = pd.Timestamp(today or datetime.now(timezone.utc).date())
    start = pd.Timestamp(C.STUDY_END) + pd.Timedelta(days=1)
    return D.download(start, today, timeout=timeout, retries=2, backoff=2)


def save_recent_cache(df: pd.DataFrame, meta: dict) -> None:
    D.save_frame(df, meta, C.RECENT_CACHE_CSV, C.RECENT_CACHE_META)


def load_combined(use_live: bool = True) -> tuple[pd.DataFrame, DataStatus]:
    snapshot, snap_meta = D.load_frame(C.SNAPSHOT_CSV, C.SNAPSHOT_META)
    errors = []
    if use_live:
        try:
            recent, meta = fetch_recent()
            combined = _merge(snapshot, recent)
            last = D.complete_rows(combined).index.max()
            return combined, DataStatus("live NASA POWER API", meta["retrieved_utc"], last)
        except Exception as err:  # network down, API error, bad payload ...
            errors.append(f"Live NASA POWER request failed ({type(err).__name__}: {str(err)[:160]}).")
    if C.RECENT_CACHE_CSV.exists():
        recent, meta = D.load_frame(C.RECENT_CACHE_CSV, C.RECENT_CACHE_META)
        combined = _merge(snapshot, recent)
        last = D.complete_rows(combined).index.max()
        errors.append("Showing the copy of recent data bundled with the app instead.")
        return combined, DataStatus("bundled cache", meta.get("retrieved_utc"), last, " ".join(errors))
    last = D.complete_rows(snapshot).index.max()
    errors.append("Only the 1981–2025 snapshot is available.")
    return snapshot, DataStatus("snapshot only", snap_meta.get("retrieved_utc"), last, " ".join(errors))
