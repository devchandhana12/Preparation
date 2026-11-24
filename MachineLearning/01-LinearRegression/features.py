from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TRAIN_PATH="data/clean/flights_train.parquet"
TEST_PATH="data/clean/flights_test.parquet"
FEATURES_DIR="data/features"
PREPROCESSOR_PATH="models/preprocessor.joblib"

TARGET="Price"
NUM_COLS=[
    "Duration_minutes", "Total_Stops","Dep_Time_hour","Arrival_Time_hour","journey_day","journey_month","journey_weekday"
]

CAT_COLS=[
    "Airline","Source","Destination","Additional_Info"
]
VALID_SIZE=0.2
MIN_CATEGORY_ROWS=20

def add_features(df:pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date = df["Date_of_Journey"]
    df["journey_day"] = date.dt.day
    df["journey_month"] = date.dt.month
    df["journey_weekday"] = date.dt.dayofweek
    return df