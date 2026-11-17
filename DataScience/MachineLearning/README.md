# Machine learning practice

Work through one algorithm at a time, starting with linear regression. Give each new algorithm its own numbered folder with the same structure.

| Folder inside each lesson | What goes there |
| --- | --- |
| `data/raw/` | The original dataset as downloaded. |
| `preparation/` | A notebook that walks through loading, checking, and cleaning the data, then saving the result. This is the ETL step. |
| `data/processed/` | The prepared data and the files you’ll load to train and evaluate the model. |
| `training/` | Your notebook for training and evaluating the model. |

The preparation notebook will explain which column to predict, what changed during cleaning, and which files to load. Any preprocessing that learns from the data, such as filling missing values with a median, will learn from the training split only and reuse those values for evaluation.
