from __future__ import annotations

from pathlib import Path

import pandas as pd


def _read_csv_if_nonempty(path: Path) -> pd.DataFrame | None:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None
        return pd.read_csv(path)
    except Exception:
        return None


def main() -> int:
    root = Path(__file__).parent
    out_dir = root / "output"

    frames: list[pd.DataFrame] = []
    for dataset in ("fastf1", "ergast"):
        ddir = out_dir / dataset
        if not ddir.exists():
            continue
        for p in sorted(ddir.glob("*.csv")):
            df = _read_csv_if_nonempty(p)
            if df is None or df.empty:
                continue
            if "dataset" not in df.columns:
                df.insert(0, "dataset", dataset)
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    comp_path = out_dir / "comparison" / "all_years_comparison.csv"
    comp_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(comp_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

