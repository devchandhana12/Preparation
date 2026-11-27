"""Train, evaluate, save, and use the flight-fare linear-regression pipeline."""  # State the full script purpose.

import json  # Write evaluation metrics in a portable format.
from pathlib import Path  # Create paths relative to this script.

import joblib  # Persist the fitted sklearn pipeline.
import numpy as np  # Apply numeric validation and the nonnegative fare floor.
import pandas as pd  # Load, transform, and export tabular data.
from sklearn.base import clone  # Fit a fresh pipeline for each validation run.
from sklearn.compose import ColumnTransformer  # Apply different preprocessing to each feature type.
from sklearn.dummy import DummyRegressor  # Create the median-fare baseline.
from sklearn.impute import SimpleImputer  # Learn numeric missing-value replacements from training rows.
from sklearn.linear_model import LinearRegression  # Fit ordinary least squares.
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error  # Calculate regression metrics.
from sklearn.model_selection import GroupShuffleSplit  # Keep repeated itineraries together in one split.
from sklearn.pipeline import Pipeline  # Bundle preprocessing and estimation.
from sklearn.preprocessing import OneHotEncoder  # Encode categorical features.

from audit import clean, extract_file, fix_data_types  # Reuse the project's file validation and cleaning logic.
from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, make_features  # Reuse one feature schema for training and inference.

ROOT = Path(__file__).resolve().parent  # Resolve paths from this script's directory.
TRAIN_FILE = ROOT / "data" / "raw" / "Data_Train.xlsx"  # Locate the labeled workbook.
TEST_FILE = ROOT / "data" / "raw" / "Test_set.xlsx"  # Locate the unlabeled workbook.
MODEL_FILE = ROOT / "models" / "linear_regression_pipeline.joblib"  # Choose the persisted model location.
METRICS_FILE = ROOT / "models" / "evaluation_metrics.json"  # Choose the persisted metrics location.
PREDICTIONS_FILE = ROOT / "flight_fare_predictions.csv"  # Choose the prediction-export location.
TARGET = "Price"  # Name the supervised learning label.


def build_pipeline() -> Pipeline:  # Create an unfitted linear-regression pipeline.
    numeric_preprocessing = SimpleImputer(strategy="median", add_indicator=True)  # Fill missing numeric values using training medians.
    categorical_preprocessing = Pipeline([  # Define transformations for category columns.
        ("fill", SimpleImputer(strategy="constant", fill_value="Unknown")),  # Replace missing categories with one token.
        ("encode", OneHotEncoder(handle_unknown="ignore")),  # Encode categories and safely handle unseen inference values.
    ])  # Finish the categorical pipeline.
    preprocessing = ColumnTransformer([  # Assign transformations to named feature lists.
        ("numeric", numeric_preprocessing, NUMERIC_FEATURES),  # Transform continuous features.
        ("categorical", categorical_preprocessing, CATEGORICAL_FEATURES),  # Transform categorical features.
    ])  # Finish the feature preprocessor.
    return Pipeline([  # Return one object that prevents preprocessing leakage.
        ("preprocess", preprocessing),  # Fit transformations only on the supplied training rows.
        ("regressor", LinearRegression()),  # Learn least-squares coefficients from transformed inputs.
    ])  # Finish the model pipeline.


def score(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:  # Calculate interpretable regression metrics.
    return {  # Return one serializable metrics record.
        "mae_inr": float(mean_absolute_error(actual, predicted)),  # Report average absolute error in rupees.
        "rmse_inr": float(root_mean_squared_error(actual, predicted)),  # Penalize unusually large rupee errors more heavily.
        "r2": float(r2_score(actual, predicted)),  # Compare squared error with a mean-fare predictor.
        "bias_inr": float(np.mean(predicted - actual.to_numpy())),  # Positive values mean average overprediction.
    }  # Finish the metrics record.


def predict_nonnegative(model: Pipeline, features: pd.DataFrame) -> tuple[np.ndarray, int]:  # Predict fares and enforce the business lower bound.
    raw_predictions = model.predict(features)  # Ask the fitted pipeline for unconstrained linear predictions.
    if not np.isfinite(raw_predictions).all():  # Detect invalid values before exporting them.
        raise ValueError("The model generated a non-finite prediction.")  # Stop instead of producing a corrupt CSV.
    floored_rows = int((raw_predictions < 0).sum())  # Count values changed by the fare constraint.
    return np.maximum(raw_predictions, 0), floored_rows  # Return nonnegative fares and the diagnostic count.


def main() -> None:  # Execute the full reproducible training and inference workflow.
    raw_train = extract_file(TRAIN_FILE, training=True)  # Read the labeled workbook after validating headers.
    labeled = clean(fix_data_types(raw_train), training=True).reset_index(drop=True)  # Apply consistent cleaning and training-only row removal.
    if labeled[TARGET].isna().any() or labeled[TARGET].le(0).any():  # Validate the regression target before fitting.
        raise ValueError("The training target must contain only positive fares.")  # Reject invalid training labels.
    features = make_features(labeled)  # Build the feature matrix from cleaned labeled data.
    target = labeled[TARGET]  # Keep the label separate from model inputs.
    journey_month = labeled["Date_of_Journey"].dt.to_period("M")  # Extract the scheduled journey month.
    latest_month = journey_month.max()  # Reserve the latest available month for primary evaluation.
    time_train_index = np.flatnonzero(journey_month < latest_month)  # Select earlier months for fitting.
    time_valid_index = np.flatnonzero(journey_month == latest_month)  # Select the latest month for evaluation.
    itinerary_columns = ["Airline", "Date_of_Journey", "Source", "Destination", "Route", "Dep_Time_hour", "Dep_Time_minute"]  # Define an itinerary identity for grouped validation.
    itinerary_groups = labeled.groupby(itinerary_columns, dropna=False).ngroup()  # Assign one group identifier to each itinerary.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)  # Configure reproducible group-based validation.
    group_train_index, group_valid_index = next(splitter.split(features, target, groups=itinerary_groups))  # Split without placing one itinerary in both sets.
    splits = {"latest_month": (time_train_index, time_valid_index), "grouped_itineraries": (group_train_index, group_valid_index)}  # Name the two evaluation protocols.
    metrics = {"dataset": {"raw_training_rows": len(raw_train), "clean_training_rows": len(labeled), "latest_month": str(latest_month)}}  # Start a portable report.
    templates = {"median_baseline": DummyRegressor(strategy="median"), "linear_regression": build_pipeline()}  # Define comparison estimators.
    for split_name, (fit_index, validation_index) in splits.items():  # Evaluate both models on each validation protocol.
        metrics[split_name] = {}  # Create this split's report section.
        for model_name, template in templates.items():  # Fit each model separately on the same rows.
            fitted = clone(template).fit(features.iloc[fit_index], target.iloc[fit_index])  # Learn preprocessing and model parameters only from fit rows.
            predictions, floored_rows = predict_nonnegative(fitted, features.iloc[validation_index])  # Produce safe validation predictions.
            metrics[split_name][model_name] = {**score(target.iloc[validation_index], predictions), "train_rows": len(fit_index), "validation_rows": len(validation_index), "floored_rows": floored_rows}  # Save results and counts.
    final_model = build_pipeline().fit(features, target)  # Refit the chosen model using every labeled row.
    raw_test = extract_file(TEST_FILE, training=False)  # Read the test workbook without requiring a target column.
    test = clean(fix_data_types(raw_test), training=False)  # Apply the same fixed cleaning logic at inference time.
    if len(test) != len(raw_test) or not test.index.equals(raw_test.index):  # Guard against losing or reordering input rows.
        raise ValueError("Inference cleaning must preserve every test row and its original order.")  # Stop if source alignment is lost.
    test_features = make_features(test)  # Build the exact training feature schema for test rows.
    test_predictions, test_floored_rows = predict_nonnegative(final_model, test_features)  # Generate safe test predictions.
    prediction_export = pd.DataFrame({"excel_row": test.index + 2, "predicted_price_inr": test_predictions.round(2)})  # Pair each fare with its source Excel row.
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)  # Create the model directory when needed.
    joblib.dump(final_model, MODEL_FILE)  # Save the fitted pipeline including preprocessing vocabulary and medians.
    prediction_export.to_csv(PREDICTIONS_FILE, index=False, float_format="%.2f")  # Export one prediction per test row in workbook order.
    metrics["test_inference"] = {"prediction_rows": len(prediction_export), "floored_rows": test_floored_rows, "prediction_min_inr": float(prediction_export["predicted_price_inr"].min()), "prediction_max_inr": float(prediction_export["predicted_price_inr"].max())}  # Add export diagnostics.
    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")  # Save all numeric results for later comparison.
    print(json.dumps(metrics, indent=2))  # Print the report for command-line users.
    print(f"Saved model to {MODEL_FILE}")  # Report the persisted pipeline path.
    print(f"Saved predictions to {PREDICTIONS_FILE}")  # Report the prediction export path.


if __name__ == "__main__":  # Run the workflow only when this file is executed directly.
    main()  # Start training, evaluation, persistence, and prediction.
