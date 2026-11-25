"""
Flight Fare: Extract -> Fix types -> Audit -> Clean -> Save
bronze (data/raw) -> silver (data/clean)

Run:  python audit.py                     (training data)
      import clean(df, training=False) from another script for test / new data
"""
from pathlib import Path

import pandas as pd

FILE_PATH = "data/raw/Data_Train.xlsx"
CLEAN_PATH = "data/clean/flights_clean.parquet"

EXPECTED_COLS = [
    "Airline", "Date_of_Journey", "Source", "Destination", "Route",
    "Dep_Time", "Arrival_Time", "Duration", "Total_Stops",
    "Additional_Info", "Price",
]

DATE_COLS = ["Date_of_Journey"]
TIME_COLS = ["Dep_Time", "Arrival_Time"]
TEXT_COLS = ["Airline", "Source", "Destination", "Route", "Additional_Info"]
NUMERICAL_COLS = ["Price"]
TARGET = "Price"

STOPS_MAP = {"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}
CITY_FIXES = {"New Delhi": "Delhi"}          # same airport (DEL)


# =====================================================================
# 1. EXTRACT
# =====================================================================
def extract_file(path: str, training: bool = True) -> pd.DataFrame:
    """training=False: new data has no target column, so don't expect it."""
    expected = EXPECTED_COLS if training else [c for c in EXPECTED_COLS if c != TARGET]
    headers = pd.read_excel(path, nrows=0).columns.to_list()
    missing = set(expected) - set(headers)
    if missing:
        raise ValueError(f"The following columns are missing {missing}")
    return pd.read_excel(path, usecols=expected)


# =====================================================================
# 2. FIX DATA TYPES
# =====================================================================
def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in DATE_COLS:
        df[col] = pd.to_datetime(df[col], format="%d/%m/%Y", errors="coerce")

    for col in TIME_COLS:
        t = pd.to_datetime(df[col].str[:5], format="%H:%M", errors="coerce")
        df[f"{col}_hour"] = t.dt.hour
        df[f"{col}_minute"] = t.dt.minute

    df["Total_Stops"] = df["Total_Stops"].map(STOPS_MAP)

    for col in [c for c in NUMERICAL_COLS if c in df.columns]:   # Price only exists in training
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.drop(columns=TIME_COLS)


# =====================================================================
# 3. AUDIT: look only, change nothing
# =====================================================================
def profile(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes("number")
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "null_%": (df.isna().mean() * 100).round(2),
        "unique": df.nunique(),
        "min": numeric.min(),
        "median": numeric.median(),
        "max": numeric.max(),
        "zeros": (numeric == 0).sum(),
    }).sort_values("null_%", ascending=False)


def auditing(df: pd.DataFrame) -> None:
    print(df.shape)
    print(profile(df))

    print("\nexact duplicate rows:", df.duplicated().sum())
    duplicated = df.duplicated(keep=False)
    print(df[duplicated].sort_values(by=list(df.columns)).head(6))

    for col in ["Airline", "Source", "Destination", "Additional_Info"]:
        print(f"\n{col}")
        print(df[col].value_counts(dropna=False))


# =====================================================================
# 4. CLEAN: fixed rules only (safe before the split)
# =====================================================================
def standardize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Same value must always look the same, BEFORE dedup."""
    df = df.copy()
    for col in TEXT_COLS:
        df[col] = df[col].str.strip()
    df["Destination"] = df["Destination"].replace(CITY_FIXES)
    df["Additional_Info"] = df["Additional_Info"].str.lower()   # "No Info" == "No info"
    return df


def parse_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Convert only complete duration values such as '2h 45m', '19h', or '5m' to minutes."""
    df = df.copy()
    duration = df["Duration"].astype("string").str.strip()  # Normalize duration values before validation.
    valid_duration = duration.str.fullmatch(r"(?:\d+h(?:\s+\d+m)?|\d+m)", na=False)  # Reject partial strings such as malformed duration text.
    hours = pd.to_numeric(duration.str.extract(r"(\d+)h")[0], errors="coerce").fillna(0)  # Extract a valid hour component.
    minutes = pd.to_numeric(duration.str.extract(r"(\d+)m")[0], errors="coerce").fillna(0)  # Extract a valid minute component.
    df["Duration_minutes"] = (hours * 60 + minutes).where(valid_duration)  # Preserve invalid values as missing for later imputation.
    return df.drop(columns="Duration")


def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    """No price = can't train on it. No stops/route = broken row."""
    before = len(df)
    df = df.dropna(subset=[TARGET, "Total_Stops", "Route"])
    print(f"drop_missing: {before} -> {len(df)} rows")
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Exact copies would leak between train and test."""
    before = len(df)
    df = df.drop_duplicates()
    print(f"deduplicate: {before} -> {len(df)} rows")
    return df


def clean(df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    """Fixes run on ALL data. Dropping rows only happens in training:
    in production every row is a request that needs a prediction."""
    df = df.pipe(standardize_text).pipe(parse_duration)
    if training:
        df = df.pipe(drop_missing).pipe(deduplicate)
    return df


# =====================================================================
# 5. SAVE
# =====================================================================
def save(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"saved {len(df)} rows x {df.shape[1]} cols -> {path}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    RUN_AUDIT = True             # set False once the audit is done

    df = extract_file(FILE_PATH, training=True)
    df = fix_data_types(df)
    if RUN_AUDIT:
        auditing(df)

    df = clean(df, training=True)
    print(profile(df))
    save(df, CLEAN_PATH)
