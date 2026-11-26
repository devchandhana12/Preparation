"""
NYC Yellow Taxi: trip duration (ETA) project
Extract -> Normalize -> Audit (look only) -> Clean (fixed rules) -> Save

Run:  python taxi_pipeline.py
"""
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# =====================================================================
# CONFIG: every decision lives here
# =====================================================================
FILE_PATH = "data/raw/yellow_tripdata_2026-03.parquet"
ZONE_MAP_FILE_PATH = "data/raw/taxi_zone_lookup.csv"
CLEAN_PATH = "data/clean/trips_2026-03.parquet"

EXPECTED_COLS = [
    "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "passenger_count", "trip_distance", "RatecodeID", "store_and_fwd_flag",
    "PULocationID", "DOLocationID", "payment_type", "fare_amount", "extra",
    "mta_tax", "tip_amount", "tolls_amount", "improvement_surcharge",
    "total_amount", "congestion_surcharge", "Airport_fee",
    "cbd_congestion_fee",
]

RENAME_COLS = {                      # applied AFTER lowercasing
    "pulocationid": "pickup_location_id",
    "dolocationid": "dropoff_location_id",
    "tpep_pickup_datetime": "pickup_datetime",
    "tpep_dropoff_datetime": "dropoff_datetime",
}

# What the model may use: only things known when the ride is booked, plus the target
KEEP_COLS = [
    "pickup_datetime",
    "pickup_location_id",
    "dropoff_location_id",
    "is_outside_nyc",
    "duration_minutes",              # target
]

UNKNOWN_LOC_ID = [264]               # no usable location -> drop
OUTSIDE_NYC_LOC_ID = [265]           # real place outside the city -> keep + flag

MONTH_START = pd.Timestamp("2026-03-01")
MONTH_END = pd.Timestamp("2026-04-01")   # exclusive
MIN_DURATION = 1                         # minutes; shorter = cancellations / bad timestamps
MAX_DURATION = 180                       # minutes; longer = meter left running

MONEY_COLS = ["fare_amount", "total_amount"]   # used to find bad rows, then dropped


# =====================================================================
# 1. EXTRACT
# =====================================================================
def extract_data(path: Path | str) -> pd.DataFrame:
    headers = pq.read_schema(path).names          # reads only the header, not 4M rows
    missing = set(EXPECTED_COLS) - set(headers)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return pq.read_table(path, columns=EXPECTED_COLS).to_pandas()


def zone_lookup_extract(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path)


# =====================================================================
# 2. NORMALIZE COLUMN NAMES
# =====================================================================
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df.rename(columns=RENAME_COLS)


# =====================================================================
# 3. TARGET: row-level, needed by both the audit and the cleaning
# =====================================================================
def add_duration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    minutes = df["dropoff_datetime"] - df["pickup_datetime"]
    df["duration_minutes"] = minutes.dt.total_seconds() / 60
    return df


# =====================================================================
# 4. AUDIT: look only, never changes df
# =====================================================================
def profile(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes("number")
    return pd.DataFrame({
        "dtypes": df.dtypes.astype(str),
        "null_%": (df.isna().mean() * 100).round(2),
        "min": numeric.min(),
        "max": numeric.max(),
        "zeros": (numeric == 0).sum(),
    }).sort_values("null_%", ascending=False)


def audit(df: pd.DataFrame) -> None:
    d = df["duration_minutes"]
    pickup = df["pickup_datetime"]
    pu, do = df["pickup_location_id"], df["dropoff_location_id"]

    print(f"rows: {len(df):,}")
    print(profile(df))
    print("\nnegative fare rows by payment_type:")
    print(df.loc[df["fare_amount"] < 0, "payment_type"].value_counts().sort_index())
    print(f"\nduration <= 0:            {(d <= 0).sum():,}")
    print(f"duration < {MIN_DURATION} min:         {((d > 0) & (d < MIN_DURATION)).sum():,}")
    print(f"duration > {MAX_DURATION} min:       {(d > MAX_DURATION).sum():,}")
    print(f"pickups outside March:    {((pickup < MONTH_START) | (pickup >= MONTH_END)).sum():,}")
    print(f"unknown location:         {(pu.isin(UNKNOWN_LOC_ID) | do.isin(UNKNOWN_LOC_ID)).sum():,}")
    print(f"outside NYC:              {(pu.isin(OUTSIDE_NYC_LOC_ID) | do.isin(OUTSIDE_NYC_LOC_ID)).sum():,}")
    print(f"exact duplicates:         {df.duplicated().sum():,}")
    print(f"passenger_count == 0:     {(df['passenger_count'] == 0).sum():,}")
    print(f"trip_distance == 0:       {(df['trip_distance'] == 0).sum():,}")


# =====================================================================
# 5. CLEAN: fixed rules only. Each step: build a "bad rows" mask, keep ~mask
# =====================================================================
def drop_rows(df: pd.DataFrame, bad: pd.Series, reason: str) -> pd.DataFrame:
    """Shared helper: drop the rows marked bad and report how many."""
    print(f"  {reason:<38} -{int(bad.sum()):>8,} rows")
    return df[~bad].copy()


def drop_negative_amounts(df: pd.DataFrame) -> pd.DataFrame:
    bad = (df[MONEY_COLS] < 0).any(axis=1)          # refunds, disputes, voids
    return drop_rows(df, bad, "negative fare/total amount")


def keep_month(df: pd.DataFrame) -> pd.DataFrame:
    p = df["pickup_datetime"]
    bad = (p < MONTH_START) | (p >= MONTH_END)
    return drop_rows(df, bad, "pickup outside March 2026")


def keep_valid_duration(df: pd.DataFrame) -> pd.DataFrame:
    d = df["duration_minutes"]
    bad = (d < MIN_DURATION) | (d > MAX_DURATION)
    return drop_rows(df, bad, f"duration outside {MIN_DURATION}-{MAX_DURATION} min")


def drop_unknown_locations(df: pd.DataFrame) -> pd.DataFrame:
    bad = (df["pickup_location_id"].isin(UNKNOWN_LOC_ID)
           | df["dropoff_location_id"].isin(UNKNOWN_LOC_ID))
    return drop_rows(df, bad, "unknown pickup/dropoff location")


def deduplicate_removal(df: pd.DataFrame) -> pd.DataFrame:
    bad = df.duplicated()                           # 0 in March, but future months may have some
    return drop_rows(df, bad, "exact duplicate rows")


def add_outside_nyc_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level feature: safe before the split."""
    df = df.copy()
    df["is_outside_nyc"] = (
        df["pickup_location_id"].isin(OUTSIDE_NYC_LOC_ID)
        | df["dropoff_location_id"].isin(OUTSIDE_NYC_LOC_ID)
    ).astype("int8")
    return df


def remove_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only what the model may use. Must run LAST: earlier steps need other columns."""
    return df[KEEP_COLS].copy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\nCLEAN  start: {len(df):,} rows")
    df = (df.pipe(normalize_cols)
          .pipe(add_duration)
          .pipe(drop_negative_amounts)
           .pipe(keep_month)
            .pipe(keep_valid_duration)
            .pipe(drop_unknown_locations)
            .pipe(deduplicate_removal)
            .pipe(add_outside_nyc_flag)
            .pipe(remove_cols))                     # LAST
    print(f"CLEAN  end:   {len(df):,} rows, columns: {df.columns.tolist()}")
    return df


# =====================================================================
# 6. SAVE
# =====================================================================
def save(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"saved -> {path}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    RUN_AUDIT = False            # True = print the full audit before cleaning

    df = extract_data(FILE_PATH)
    zones = zone_lookup_extract(ZONE_MAP_FILE_PATH)   # used later for features

    df = normalize_cols(df)
    df = add_duration(df)
    if RUN_AUDIT:
        audit(df)

    df = clean(df)               # every cleaning step runs here, once, in order
    save(df, CLEAN_PATH)