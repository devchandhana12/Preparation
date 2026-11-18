# Machine Learning

Start with [Linear Regression](01-linear-regression/notes.md) or open the [traffic volume preparation notebook](01-linear-regression/preparation/prepare.ipynb) for the hands-on project. Give each algorithm a numbered folder such as `02-logistic-regression/`, with the notes, PDF, and hands-on work together.

| Path inside each algorithm folder | What goes there |
| --- | --- |
| `notes.md` | Theory, math, assumptions, diagnostics, and failure cases. |
| `guide.pdf` | The PDF guide for the algorithm. |
| `data/raw/` | The source dataset. Record where to get it in the preparation notebook. |
| `preparation/prepare.ipynb` | Inspect and clean the data, define the target, and prepare the train and evaluation splits. |
| `data/processed/` | Prepared data loaded by the training notebook. |
| `training/train.ipynb` | Train a baseline and the model, evaluate them on held-out data, and record the results. |

Run preparation before training. Fit any learned preprocessing, such as median imputation or scaling, on the training split and apply the fitted transformations to the held-out data.
