# Learn Data Science by Doing

Start with Python, learn to inspect and explain a dataset, then train and evaluate your first model. Work through the stages in order if you're new to data science. Each stage pairs existing material with something you can do on your own.

## Setup

Install [uv](https://docs.astral.sh/uv/). From the repository root, set up the project environment and launch JupyterLab:

```sh
uv sync --locked
uv run jupyter lab
```

The project pins Python 3.14 in [`.python-version`](.python-version). Open notebooks from their project folders in JupyterLab and run the cells from top to bottom. The Pandas and flight-fare notebooks read data bundled with the repo. The bike-sharing notebook downloads its dataset from Hugging Face on the first run.

## Learning path

| Stage | Start here | Try this |
| --- | --- | --- |
| [01 Python](01-Python/) | [Python notes](01-Python/notes/Python_for_ML_AI_Engineers.pdf) and [practice notebook](01-Python/practice/python_practice.ipynb) | Practice slicing, lists, and comprehensions. Write a function that cleans a list of values, including an empty input. |
| [02 Pandas](02-Pandas/) | [Pandas practice notebook](02-Pandas/pandas_practice.ipynb) | Load the salary CSV, inspect its columns, then filter, sort, and summarize the data. |
| [03 Statistics](03-Statistics/) | [Statistics notes](03-Statistics/notes/Statistics_for_Data_Science.pdf) and [interview field guide](03-Statistics/notes/Data_Science_Foundations_Interview_Field_Guide.pdf) | Calculate the mean and median salary. Add an extreme value and explain how each statistic changes. |
| [04 Exploratory data analysis (EDA)](04-EDA/) | [Bike-sharing notebook](04-EDA/01-BikeSharing/bike_sharing.ipynb) | Start with the notebook's duplicate-hour check. Inspect missing values, plot rides by hour, and write down what you found. |
| [05 Machine learning](05-MachineLearning/) | [Machine learning guide](05-MachineLearning/README.md) | Read the Linear Regression notes, then run the flight-fare notebook. Compare the model with the median baseline using the same labeled holdout. |
| [06 Applied AI](06-AppliedAI/) | [PDF RAG starter project](06-AppliedAI/01-PDF-RAG/) | The starter project contains a source PDF and a notebook stub. The planned walkthrough will cover answering questions using passages retrieved from a PDF. |

## Notes in progress

| Notes | Where to find them |
| --- | --- |
| EDA | Planned for [`04-EDA/`](04-EDA/), alongside the bike-sharing exercise. |
| Machine learning | [Linear Regression notes](05-MachineLearning/01-LinearRegression/notes/notes.md) are available. Additional algorithms will each get a module under [`05-MachineLearning/`](05-MachineLearning/). |
| AI | Planned for [`06-AppliedAI/`](06-AppliedAI/), alongside the PDF RAG starter project. |
