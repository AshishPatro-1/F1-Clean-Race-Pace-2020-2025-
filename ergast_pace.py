from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests


ERGAST_BASE = "https://ergast.com/mrd/api/f1"


@dataclass(frozen=True)
class ErgastConfig:
    timeout_s: int = 8
    user_agent: str = "f1-clean-air-race-pace/1.0"


def _parse_lap_time_to_seconds(s: str) -> float | None:
    """
    Ergast lap time strings are usually 'M:SS.mmm' (e.g., '1:27.345').
    Sometimes hours appear rarely; we handle 'H:MM:SS.mmm' too.
    """
    if not s or not isinstance(s, str):
        return None
    parts = s.split(":")
    try:
        if len(parts) == 2:
            m = int(parts[0])
            sec = float(parts[1])
            return 60.0 * m + sec
        if len(parts) == 3:
            h = int(parts[0])
            m = int(parts[1])
            sec = float(parts[2])
            return 3600.0 * h + 60.0 * m + sec
    except ValueError:
        return None
    return None


def _get_json(url: str, *, cfg: ErgastConfig) -> dict:
    r = requests.get(url, headers={"User-Agent": cfg.user_agent}, timeout=cfg.timeout_s)
    r.raise_for_status()
    return r.json()


def list_races(year: int, *, cfg: ErgastConfig) -> pd.DataFrame:
    url = f"{ERGAST_BASE}/{year}.json?limit=1000"
    j = _get_json(url, cfg=cfg)
    races = j["MRData"]["RaceTable"]["Races"]
    rows = []
    for r in races:
        rows.append(
            {
                "year": int(year),
                "round": int(r["round"]),
                "raceName": r.get("raceName"),
                "circuitId": (r.get("Circuit") or {}).get("circuitId"),
                "date": r.get("date"),
            }
        )
    return pd.DataFrame(rows)


def fetch_race_laps(year: int, round_number: int, *, cfg: ErgastConfig) -> pd.DataFrame:
    """
    Fetch all lap times for a given race via paging.
    Endpoint: /{year}/{round}/laps.json?limit=2000&offset=...
    """
    limit = 2000
    offset = 0
    all_rows: list[dict] = []
    while True:
        url = f"{ERGAST_BASE}/{year}/{round_number}/laps.json?limit={limit}&offset={offset}"
        j = _get_json(url, cfg=cfg)
        mr = j["MRData"]
        total = int(mr["total"])
        races = mr["RaceTable"]["Races"]
        if not races:
            break
        laps = races[0].get("Laps", [])
        for lap in laps:
            lap_no = int(lap["number"])
            for t in lap.get("Timings", []):
                all_rows.append(
                    {
                        "year": int(year),
                        "round": int(round_number),
                        "lap": lap_no,
                        "driverId": t.get("driverId"),
                        "position": int(t.get("position")) if t.get("position") else None,
                        "lap_time_s": _parse_lap_time_to_seconds(t.get("time")),
                    }
                )
        offset += limit
        if offset >= total:
            break
    return pd.DataFrame(all_rows)


def build_ergast_race_level_features(
    *,
    year: int,
    event_name: str,
    round_number: int,
    laps: pd.DataFrame,
    dataset_name: str = "ergast",
) -> pd.DataFrame:
    """
    Ergast does not provide reliable per-lap gaps to the car ahead.
    We therefore produce race-level *pace-only* metrics per driver.
    """
    df = laps.copy()
    df = df.loc[df["lap_time_s"].notna()].copy()

    out = (
        df.groupby(["driverId"], dropna=False)["lap_time_s"]
        .agg(
            valid_laps_n="count",
            lap_time_median_s="median",
            lap_time_p25_s=lambda s: float(np.nanpercentile(s.astype(float), 25)) if len(s) else np.nan,
            best5_median_s=lambda s: float(np.nanmedian(np.sort(s.astype(float))[:5])) if len(s) >= 5 else np.nan,
        )
        .reset_index()
        .rename(columns={"driverId": "Driver"})
    )

    out.insert(0, "dataset", dataset_name)
    out.insert(0, "round", int(round_number))
    out.insert(0, "event", event_name)
    out.insert(0, "year", int(year))
    return out

