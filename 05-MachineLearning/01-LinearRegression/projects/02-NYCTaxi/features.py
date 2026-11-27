import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import TimeSeriesSplit

TRAIN_PATH = "data/clean/trips_2026-03.parquet"
TEST_PATH = "data/clean/trips_2026-04.parquet"
N_SPLITS = 5


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pickup_date = df["pickup_datetime"]
    df["pickup_hour"] = pickup_date.dt.hour.astype("int8")
    df["pickup_dayofweek"] = pickup_date.dt.dayofweek.astype("int8")
    df["is_weekend"] = df["pickup_dayofweek"].isin([5, 6]).astype("int8")
    return df


def show_folds(df: pd.DataFrame, cv) -> None:
    """Print what each fold trains and validates on. No model yet."""
    dates = df["pickup_datetime"]
    for i, (train_idx, valid_idx) in enumerate(cv.split(df), start=1):
        tr, va = dates.iloc[train_idx], dates.iloc[valid_idx]
        print(f"fold {i}: train {len(tr):>9,} rows ({tr.min():%d %b} to {tr.max():%d %b}) | "
              f"valid {len(va):>8,} rows ({va.min():%d %b} to {va.max():%d %b})")


if __name__ == "__main__":
    march = pq.read_table(TRAIN_PATH).to_pandas()
    april = pq.read_table(TEST_PATH).to_pandas()

    march = add_features(march)
    april = add_features(april)          # the final test set: untouched until the very end

    # TimeSeriesSplit splits by ROW POSITION, so the rows must be in time order
    march = march.sort_values("pickup_datetime").reset_index(drop=True)

    cv = TimeSeriesSplit(n_splits=N_SPLITS)
    show_folds(march, cv)