from preprocess import clean, extract_data, save

FILE_PATH = "data/raw/yellow_tripdata_2026-04.parquet"

if __name__ == "__main__":
    df = extract_data(FILE_PATH)
    df = clean(df)
    save(df, "data/clean/trips_2026-04.parquet")