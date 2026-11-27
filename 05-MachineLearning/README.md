# Machine Learning

Start with the [Linear Regression notes](01-LinearRegression/notes/notes.md) to learn how a model predicts a numeric target. Then use the flight-fare project to follow the full workflow, from inspecting raw data to evaluating a model and exporting predictions.

## 1. Predict flight fares with Linear Regression

Open the [flight-fare training notebook](01-LinearRegression/projects/01-FlightFare/train.ipynb) from its own folder in JupyterLab. It reads the bundled Excel workbooks and walks through data cleaning, feature engineering, baseline comparison, and error analysis.

1. Identify `Price` as the target and check which itinerary features are available when making a prediction.
2. Compare Linear Regression with a median-fare baseline on the latest labeled month. Use MAE to express the average absolute error in rupees, then compare RMSE and R².
3. Compare the time-based results with the grouped-itinerary results. Read the notebook's limitations and look for groups of flights where errors are larger.
4. Refit on all labeled rows and generate predictions for the unlabeled workbook. Compute performance metrics on the labeled holdouts, where actual fares are available.

Run the same workflow from the repository root with:

```sh
uv run python 05-MachineLearning/01-LinearRegression/projects/01-FlightFare/train.py
```

The script regenerates the [evaluation metrics](01-LinearRegression/projects/01-FlightFare/models/evaluation_metrics.json), the [predictions CSV](01-LinearRegression/projects/01-FlightFare/flight_fare_predictions.csv), and the fitted pipeline. Use the [interview and edge-case guide](01-LinearRegression/notes/Linear_Regression_Interview_Questions_and_Edge_Cases.pdf) to review the algorithm after the project.

## 2. Prepare NYC taxi data for time-based validation

The [taxi project](01-LinearRegression/projects/02-NYCTaxi/) is a follow-up exercise in progress. [Preprocessing](01-LinearRegression/projects/02-NYCTaxi/preprocess.py) audits and cleans raw trip records. [Feature engineering](01-LinearRegression/projects/02-NYCTaxi/features.py) builds pickup-time features and shows the training and validation dates for each March fold. The project reserves April for final evaluation once model training is added.

## Adding another algorithm

Give each algorithm a numbered module. Start with `notes/notes.md` for the problem, assumptions, math, and failure cases. Add each hands-on example under `projects/` and keep its data with it. Use a notebook or script to define the target, prepare the data, compare the model with a baseline on held-out examples, and explain the results. Include a PDF guide when one is ready.

Fit learned preprocessing, such as imputation or scaling, on the training split and apply the fitted transformations to held-out data.
