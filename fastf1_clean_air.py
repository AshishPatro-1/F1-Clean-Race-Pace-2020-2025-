from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleanAirConfig:
    clean_gap_s: float = 2.0
    drop_first_lap_in_stint: bool = True
    drop_last_lap_in_stint: bool = False


def _to_seconds(td: object) -> float | None:
    if td is None or (isinstance(td, float) and np.isnan(td)):
        return None
    if isinstance(td, timedelta):
        return td.total_seconds()
    if isinstance(td, pd.Timedelta):
        return td.total_seconds()
    return None


def _mad_based_outlier_mask(x: pd.Series, z: float = 4.5) -> pd.Series:
    """Returns True for non-outliers using a robust MAD rule."""
    x = x.astype(float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        return pd.Series([True] * len(x), index=x.index)
    modified_z = 0.6745 * (x - med) / mad
    return np.abs(modified_z) <= z


def compute_gap_to_ahead_seconds(laps: pd.DataFrame) -> pd.Series:
    """
    Compute per-lap gap to car ahead (in seconds) using FastF1 lap completion
    timestamps (`Time`) within each LapNumber, ordered by Position.

    Expected columns: LapNumber, Position, Time
    """
    if not {"LapNumber", "Position", "Time"}.issubset(laps.columns):
        return pd.Series([np.nan] * len(laps), index=laps.index, dtype="float64")

    def per_lap(df: pd.DataFrame) -> pd.Series:
        df2 = df.sort_values("Position")
        t = df2["Time"].map(_to_seconds).astype("float64")
        gap = t - t.shift(1)
        out = pd.Series(index=df2.index, data=gap.values, dtype="float64")
        return out.reindex(df.index)

    gb = laps.groupby("LapNumber", group_keys=False)
    try:
        return gb.apply(per_lap, include_groups=False)
    except TypeError:
        # pandas<2.2 fallback
        return gb.apply(per_lap)


def build_fastf1_race_level_features(
    laps: pd.DataFrame,
    *,
    year: int,
    event_name: str,
    round_number: int | None,
    dataset_name: str = "fastf1",
    config: CleanAirConfig = CleanAirConfig(),
) -> pd.DataFrame:
    """
    Returns driver-race rows with clean-air pace metrics.

    Required columns (best-effort; missing columns degrade gracefully):
    - Driver, Team, LapNumber, LapTime, Time, Position
    Optional: Stint, TyreLife, Compound, TrackStatus, IsAccurate, PitInTime, PitOutTime
    """
    df = laps.copy()

    # Normalize key timing columns
    if "LapTime" in df.columns:
        df["lap_time_s"] = df["LapTime"].map(_to_seconds).astype("float64")
    else:
        df["lap_time_s"] = np.nan

    df["gap_ahead_s"] = compute_gap_to_ahead_seconds(df)

    # Basic validity filters
    valid = df["lap_time_s"].notna()

    # FastF1 provides IsAccurate for many sessions; keep accurate only if present.
    if "IsAccurate" in df.columns:
        valid &= df["IsAccurate"].fillna(False)

    # Drop in/out laps when identifiable
    for col in ("PitInTime", "PitOutTime"):
        if col in df.columns:
            valid &= df[col].isna()

    # TrackStatus: '1' = green in FastF1; handle both int/str
    if "TrackStatus" in df.columns:
        ts = df["TrackStatus"].astype(str)
        valid &= ts == "1"

    df = df.loc[valid].copy()

    # Stint trimming and outlier filtering within (Driver, Stint)
    if "Stint" in df.columns:
        df["stint_lap_idx"] = df.groupby(["Driver", "Stint"])["LapNumber"].rank(method="first").astype(int)
        if config.drop_first_lap_in_stint:
            df = df.loc[df["stint_lap_idx"] > 1].copy()
        if config.drop_last_lap_in_stint:
            max_idx = df.groupby(["Driver", "Stint"])["stint_lap_idx"].transform("max")
            df = df.loc[df["stint_lap_idx"] < max_idx].copy()

        df["non_outlier"] = df.groupby(["Driver", "Stint"])["lap_time_s"].transform(
            lambda s: _mad_based_outlier_mask(s)
        )
        df = df.loc[df["non_outlier"].fillna(True)].copy()
    else:
        df["non_outlier"] = _mad_based_outlier_mask(df["lap_time_s"])
        df = df.loc[df["non_outlier"]].copy()

    # Clean air
    df["is_clean_air"] = df["gap_ahead_s"].fillna(np.inf) > float(config.clean_gap_s)
    # P1 has NaN gap; treat as clean
    if "Position" in df.columns:
        df.loc[df["Position"] == 1, "is_clean_air"] = True

    # Driver-level aggregates
    group_cols = ["Driver"]
    if "Team" in df.columns:
        group_cols.append("Team")

    def agg_driver(g: pd.DataFrame) -> pd.Series:
        all_laps = g["lap_time_s"].astype(float)
        clean = g.loc[g["is_clean_air"], "lap_time_s"].astype(float)
        return pd.Series(
            {
                "valid_laps_n": int(all_laps.notna().sum()),
                "clean_air_laps_n": int(clean.notna().sum()),
                "lap_time_median_s": float(np.nanmedian(all_laps)) if len(all_laps) else np.nan,
                "clean_air_lap_time_median_s": float(np.nanmedian(clean)) if len(clean) else np.nan,
                "clean_air_lap_time_p25_s": float(np.nanpercentile(clean, 25)) if len(clean) else np.nan,
                "clean_air_best5_median_s": float(np.nanmedian(np.sort(clean)[:5])) if len(clean) >= 5 else np.nan,
                "clean_gap_s": float(config.clean_gap_s),
            }
        )

    gb2 = df.groupby(group_cols, dropna=False)
    try:
        out = gb2.apply(agg_driver, include_groups=False).reset_index()
    except TypeError:
        out = gb2.apply(agg_driver).reset_index()
    out.insert(0, "dataset", dataset_name)
    out.insert(0, "round", round_number)
    out.insert(0, "event", event_name)
    out.insert(0, "year", int(year))
    return out

