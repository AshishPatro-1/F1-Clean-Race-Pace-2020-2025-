## F1 Clean Air Race Pace 

This folder produces **race-level pace features** for seasons **2020–2025** using:

- **FastF1**: includes a **clean-air filter** (gap to car ahead at lap end \(>\) threshold).
- **Ergast**: provides lap times but **does not include per-lap gaps**, so outputs are **pace-only** (no clean-air).

Outputs:

- `output/fastf1/<year>.csv`
- `output/ergast/<year>.csv`
- `output/comparison/all_years_comparison.csv`

## Setup

From the main workspace root:

```powershell
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r "F1_Clean race pace\requirements.txt"
```

## Run

```powershell
.\.venv\Scripts\python.exe -u "F1_Clean race pace\run_all_years.py"
```

## Clean-air definition (FastF1)

On each lap number, we compute **gap-to-car-ahead** using the per-driver lap completion timestamp (`Time`) from FastF1:

- gap_ahead_s = Time(driver) - Time(position-1)
- clean if gap_ahead_s > `--clean-gap-s` (default 2.0s)

We also exclude:

- in/out laps (when identifiable)
- laps without a `LapTime`
- obvious neutralization laps (SC/VSC) when track status is available

You can tune filtering with CLI flags in `run_all_years.py`.
