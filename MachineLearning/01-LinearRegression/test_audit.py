import pandas as pd
from audit import extract_file, fix_data_types, profile, auditing, clean, save

FILE_PATH = "data/raw/Test_set.xlsx"
CLEAN_PATH = "data/clean/flights_test.parquet"

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    RUN_AUDIT = True

    df = extract_file(FILE_PATH, training=False)
    df = fix_data_types(df)
    if RUN_AUDIT:
        auditing(df)

    df = clean(df, training=False)
    print(profile(df))
    save(df, CLEAN_PATH)