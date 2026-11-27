"""Feature creation shared by training and prediction."""  # Describe the module's responsibility.

import numpy as np  # Create cyclic time features.
import pandas as pd  # Transform tabular flight data.

CATEGORICAL_FEATURES = ["Airline", "Source", "Destination", "Journey_weekday"]  # Name columns encoded as categories.
NUMERIC_FEATURES = ["Total_Stops", "Duration_minutes", "Departure_sin", "Departure_cos", "Arrival_sin", "Arrival_cos"]  # Name continuous model inputs.
MIN_DURATION_MINUTES = 30  # Treat shorter itineraries as invalid for this dataset.


def make_features(frame: pd.DataFrame) -> pd.DataFrame:  # Build model inputs from a cleaned flight table.
    features = frame[["Airline", "Source", "Destination", "Total_Stops", "Duration_minutes"]].copy()  # Select raw model inputs.
    features["Journey_weekday"] = frame["Date_of_Journey"].dt.day_name()  # Extract the weekday from the journey date.
    features["Duration_minutes"] = features["Duration_minutes"].where(features["Duration_minutes"] >= MIN_DURATION_MINUTES)  # Mask implausibly short journeys.
    for source_prefix, feature_prefix in [("Dep_Time", "Departure"), ("Arrival_Time", "Arrival")]:  # Encode both clock times.
        minutes = frame[f"{source_prefix}_hour"] * 60 + frame[f"{source_prefix}_minute"]  # Convert a time into minutes since midnight.
        angle = 2 * np.pi * minutes / (24 * 60)  # Map one day onto a full circle.
        features[f"{feature_prefix}_sin"] = np.sin(angle)  # Preserve one cyclic coordinate.
        features[f"{feature_prefix}_cos"] = np.cos(angle)  # Preserve the other cyclic coordinate.
    return features[CATEGORICAL_FEATURES + NUMERIC_FEATURES]  # Return columns in a stable model-input order.
