from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    df.to_csv(path, index=False)


FASTF1_OUTPUT_COLUMNS = [
    "year",
    "event",
    "round",
    "dataset",
    "Driver",
    "Team",
    "valid_laps_n",
    "clean_air_laps_n",
    "lap_time_median_s",
    "clean_air_lap_time_median_s",
    "clean_air_lap_time_p25_s",
    "clean_air_best5_median_s",
    "clean_gap_s",
    "error",
]

ERGAST_OUTPUT_COLUMNS = [
    "year",
    "event",
    "round",
    "dataset",
    "Driver",
    "valid_laps_n",
    "lap_time_median_s",
    "lap_time_p25_s",
    "best5_median_s",
    "error",
]


def _select_existing_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    keep = [c for c in cols if c in df.columns]
    return df.loc[:, keep].copy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build race-level pace features for 2020–2025.")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--clean-gap-s", type=float, default=2.0)
    parser.add_argument("--cache-dir", type=str, default=str(Path(__file__).parent / "cache"))
    parser.add_argument("--skip-ergast", action="store_true", help="Skip Ergast extraction (offline/blocked).")
    parser.add_argument(
        "--rounds",
        type=str,
        default="",
        help="Optional comma-separated round numbers to run (e.g. '1,2,3').",
    )
    args = parser.parse_args()

    root = Path(__file__).parent
    out_fastf1 = root / "output" / "fastf1"
    out_ergast = root / "output" / "ergast"
    out_comp = root / "output" / "comparison"

    # Lazy imports so running `--help` doesn't require deps.
    import fastf1

    from ergast_pace import ErgastConfig, build_ergast_race_level_features, fetch_race_laps
    from fastf1_clean_air import CleanAirConfig, build_fastf1_race_level_features

    fastf1.Cache.enable_cache(args.cache_dir)
    try:
        fastf1.set_log_level("WARNING")
    except Exception:
        pass

    all_rows: list[pd.DataFrame] = []

    for year in range(args.start_year, args.end_year + 1):
        # Use FastF1 schedule as the primary race list (works even if Ergast is blocked).
        # Persist schedules locally so we can continue even if rate-limited later.
        sched_cache = Path(args.cache_dir) / f"schedule_{year}.csv"
        try:
            sched = fastf1.get_event_schedule(year)
            _ensure_dir(sched_cache.parent)
            sched.to_csv(sched_cache, index=False)
        except Exception:
            if sched_cache.exists():
                sched = pd.read_csv(sched_cache)
            else:
                raise
        races = (
            sched.loc[sched["EventFormat"].astype(str).str.contains("conventional|sprint", case=False, na=False)]
            if "EventFormat" in sched.columns
            else sched
        )
        races = races.loc[races["RoundNumber"].notna()].copy()
        races["round"] = races["RoundNumber"].astype(int)
        races["raceName"] = races["EventName"].astype(str)
        races = races[["round", "raceName"]].drop_duplicates().sort_values("round").reset_index(drop=True)
        if args.rounds.strip():
            wanted = {int(x.strip()) for x in args.rounds.split(",") if x.strip()}
            races = races.loc[races["round"].isin(sorted(wanted))].reset_index(drop=True)
        fastf1_year_rows: list[pd.DataFrame] = []
        ergast_year_rows: list[pd.DataFrame] = []

        for _, r in tqdm(races.iterrows(), total=len(races), desc=f"{year}"):
            rnd = int(r["round"])
            event = str(r["raceName"])

            # ---- FastF1
            try:
                session = fastf1.get_session(year, rnd, "R")
                # Keep API calls low: we only need lap timing (no telemetry/weather/messages).
                session.load(laps=True, telemetry=False, weather=False, messages=False)
                laps = session.laps
                features = build_fastf1_race_level_features(
                    laps,
                    year=year,
                    event_name=event,
                    round_number=rnd,
                    config=CleanAirConfig(clean_gap_s=float(args.clean_gap_s)),
                )
                fastf1_year_rows.append(features)
            except Exception as e:
                # Keep going; record a minimal error row for traceability.
                fastf1_year_rows.append(
                    pd.DataFrame(
                        [
                            {
                                "year": year,
                                "event": event,
                                "round": rnd,
                                "dataset": "fastf1",
                                "error": str(e),
                            }
                        ]
                    )
                )

            # ---- Ergast (pace-only; optional)
            if not args.skip_ergast:
                try:
                    laps_e = fetch_race_laps(year, rnd, cfg=ErgastConfig())
                    features_e = build_ergast_race_level_features(
                        year=year,
                        event_name=event,
                        round_number=rnd,
                        laps=laps_e,
                    )
                    ergast_year_rows.append(features_e)
                except Exception as e:
                    ergast_year_rows.append(
                        pd.DataFrame(
                            [
                                {
                                    "year": year,
                                    "event": event,
                                    "round": rnd,
                                    "dataset": "ergast",
                                    "error": str(e),
                                }
                            ]
                        )
                    )

            # Write incremental snapshots so partial progress is preserved
            if fastf1_year_rows:
                snap = pd.concat(fastf1_year_rows, ignore_index=True)
                snap = _select_existing_columns(snap, FASTF1_OUTPUT_COLUMNS)
                _write_csv(snap, out_fastf1 / f"{year}.csv")
            if ergast_year_rows:
                snap_e = pd.concat(ergast_year_rows, ignore_index=True)
                snap_e = _select_existing_columns(snap_e, ERGAST_OUTPUT_COLUMNS)
                _write_csv(snap_e, out_ergast / f"{year}.csv")

        df_fastf1_year = pd.concat(fastf1_year_rows, ignore_index=True) if fastf1_year_rows else pd.DataFrame()
        df_ergast_year = pd.concat(ergast_year_rows, ignore_index=True) if ergast_year_rows else pd.DataFrame()

        _write_csv(_select_existing_columns(df_fastf1_year, FASTF1_OUTPUT_COLUMNS), out_fastf1 / f"{year}.csv")
        _write_csv(_select_existing_columns(df_ergast_year, ERGAST_OUTPUT_COLUMNS), out_ergast / f"{year}.csv")

        if not df_fastf1_year.empty:
            all_rows.append(df_fastf1_year)
        if not df_ergast_year.empty:
            all_rows.append(df_ergast_year)

    # Final combined file (all datasets, all years)
    df_all = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    _write_csv(df_all, out_comp / "all_years_comparison.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

