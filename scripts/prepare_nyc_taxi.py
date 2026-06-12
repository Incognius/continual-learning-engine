"""Offline preparation of NYC TLC yellow-taxi data into a clean streaming chunk.

Downloads the requested monthly parquet files, cleans the notoriously dirty raw
data, derives pickup-time features, computes the trip-duration target, sorts
chronologically, optionally subsamples, and writes a single cached parquet that
``NYCTaxiDataSource`` replays.

Usage:
    python scripts/prepare_nyc_taxi.py \
        --months 2023-01 2023-02 2023-03 \
        --rows-per-month 150000 \
        --out data/processed/nyc_taxi.parquet

Run once; the source then streams the cached file offline.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
RAW_DIR = Path("data/raw")

# Columns we actually need from the raw file (keeps memory down).
_USE_COLS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
]

DEFAULT_RUSH_HOURS = {7, 8, 9, 16, 17, 18, 19}


def download_month(month: str) -> Path:
    """Download one monthly parquet to data/raw (cached; skips if present)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"yellow_tripdata_{month}.parquet"
    dest = RAW_DIR / fname
    if dest.exists():
        print(f"  [cache] {fname} already downloaded")
        return dest
    url = f"{BASE_URL}/{fname}"
    print(f"  [get ] {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"  [ok  ] {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def clean_month(
    path: Path,
    month: str,
    min_duration_s: int,
    max_duration_s: int,
    rows: int | None,
    rush_hours: set[int],
) -> pd.DataFrame:
    """Load one month, clean it, and derive features + target."""
    df = pd.read_parquet(path, columns=_USE_COLS)
    n0 = len(df)

    pickup = pd.to_datetime(df["tpep_pickup_datetime"])
    dropoff = pd.to_datetime(df["tpep_dropoff_datetime"])
    duration = (dropoff - pickup).dt.total_seconds()

    year, mon = (int(x) for x in month.split("-"))
    month_start = pd.Timestamp(year=year, month=mon, day=1)
    month_end = month_start + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1)

    mask = (
        duration.between(min_duration_s, max_duration_s)
        & (df["passenger_count"].fillna(0) >= 1)
        & (df["trip_distance"] > 0)
        & (df["PULocationID"].notna())
        & (df["DOLocationID"].notna())
        & (df["RatecodeID"].notna())
        # Drop records whose pickup timestamp falls outside the file's month
        # (these are data-entry errors and would corrupt chronological replay).
        & (pickup >= month_start)
        & (pickup < month_end)
    )

    df = df[mask].copy()
    pickup = pickup[mask]
    df["pickup_datetime"] = pickup
    df["target_duration_s"] = duration[mask]

    # Pickup-time features only (no leakage).
    df["pickup_hour"] = pickup.dt.hour
    df["pickup_dayofweek"] = pickup.dt.dayofweek
    df["pickup_month"] = pickup.dt.month
    df["is_weekend"] = (pickup.dt.dayofweek >= 5).astype(int)
    df["is_rush_hour"] = pickup.dt.hour.isin(rush_hours).astype(int)
    df["passenger_count"] = df["passenger_count"].astype(int)
    df["pu_zone"] = df["PULocationID"].astype(int).astype(str)
    df["do_zone"] = df["DOLocationID"].astype(int).astype(str)
    df["ratecode"] = df["RatecodeID"].astype(int).astype(str)
    # Realized route length (miles -> km). NOTE: this is the *driven* distance and
    # is only known at dropoff, so it must NOT be used as a leakage-free feature
    # directly. It is retained so the source can expose it as a SIMULATED
    # production "planned route distance" feature (see config use_route_distance).
    df["route_distance_km"] = (df["trip_distance"] * 1.60934).clip(lower=0.0)

    out_cols = [
        "pickup_datetime",
        "target_duration_s",
        "pickup_hour",
        "pickup_dayofweek",
        "pickup_month",
        "is_weekend",
        "is_rush_hour",
        "passenger_count",
        "route_distance_km",
        "pu_zone",
        "do_zone",
        "ratecode",
    ]
    df = df[out_cols]

    if rows is not None and len(df) > rows:
        # Subsample but preserve temporal coverage by sampling then re-sorting.
        df = df.sample(n=rows, random_state=42)

    df = df.sort_values("pickup_datetime").reset_index(drop=True)
    print(
        f"  [clean] {month}: {n0:,} -> {len(df):,} rows kept "
        f"({100 * len(df) / max(n0, 1):.1f}%)"
    )
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--months", nargs="+", default=["2023-01", "2023-02", "2023-03"],
        help="Months in YYYY-MM format.",
    )
    parser.add_argument("--out", default="data/processed/nyc_taxi.parquet")
    parser.add_argument("--rows-per-month", type=int, default=150_000,
                        help="Subsample cap per month (0 = keep all).")
    parser.add_argument("--min-duration", type=int, default=60,
                        help="Drop trips shorter than this many seconds.")
    parser.add_argument("--max-duration", type=int, default=3 * 3600,
                        help="Drop trips longer than this many seconds.")
    args = parser.parse_args(argv)

    rows = None if args.rows_per_month == 0 else args.rows_per_month
    rush_hours = DEFAULT_RUSH_HOURS

    frames: list[pd.DataFrame] = []
    for month in args.months:
        print(f"[month] {month}")
        raw = download_month(month)
        frames.append(
            clean_month(raw, month, args.min_duration, args.max_duration, rows, rush_hours)
        )

    full = pd.concat(frames, ignore_index=True).sort_values("pickup_datetime")
    full = full.reset_index(drop=True)
    full.insert(0, "record_id", [f"taxi-{i}" for i in range(len(full))])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_path, index=False)

    print("-" * 60)
    print(f"[done ] {len(full):,} trips -> {out_path} "
          f"({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"[span ] {full['pickup_datetime'].min()}  ->  "
          f"{full['pickup_datetime'].max()}")
    print(f"[target] duration seconds | mean={full['target_duration_s'].mean():.0f}s "
          f"median={full['target_duration_s'].median():.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
