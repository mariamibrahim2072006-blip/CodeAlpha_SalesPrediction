# Advertising-based sales prediction

CodeAlpha DS task 4. Prepared for Mariam Ibrahim's review.

**Status:** Working review package, not yet submitted. Prepared with AI assistance. Mariam should run the project, review the choices and explain the results before it is represented as her completed internship work. No claim of independent authorship, internship completion or certification is made.

## What the project does

The 200-row Advertising.csv dataset is downloaded from the Kaggle source linked in the official task. Predict sales from TV, radio and newspaper advertising. The CSV index is excluded. All predictors and targets are numeric and checked for missing cells. An 80/20 split (160 training, 40 test) is used; a mean-sales baseline is included. Five-fold cross-validation reports training-only MAE, and a linear regression is evaluated once on the test data.

The data has no time index, target-segment field, or campaign ordering. This is a cross-sectional prediction exercise for comparable observations, not a demonstrated forecast of future periods. Predictor units are not defined on the Kaggle page, so no currency or monetary return is asserted. Raw regression coefficients are associations conditional on the other inputs, not evidence that spending causes sales. TV/radio associations are stronger than newspaper in this fit. A sensible next step is a controlled campaign experiment, not automatically reallocating a budget. Do not extrapolate beyond observed spend ranges. The source page lists its license as unknown; raw CSV data is not included in the review archive or proposed GitHub repository. Download access does not settle redistribution rights.

## Run it

Python 3.10.12 was used for the recorded run. Install Python first, then run from this folder:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell, instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
python -m unittest test_project.py
```

The first run of credit/sales needs internet to download the source data. Subsequent runs use the local cache. Iris and breast-cancer data are bundled with scikit-learn. Results are written under `results/`. No credentials are required. Nothing is sent or published by the scripts. The notebook needs Jupyter (`pip install notebook`) only if you want to open it interactively.

## Recorded held-out results

Seed: 42. Training rows: 160. Test rows: 40.

| Model | mae | rmse | r2 |
|---|---:|---:|---:|
| baseline | 4.9625 | 5.6315 | -0.0048 |
| linear | 1.4608 | 1.7816 | 0.8994 |


Binary classification precision, recall and F1 use class 1 (default or malignant). Iris metrics use macro averaging. Classification row indexes in `test_predictions.csv` refer to the cleaned analysis table, not the original file. Sales row indexes are original CSV row indexes. Exact train-only cross-validation means and standard deviations are in `results/metrics.json`. Data-quality counts are in `results/data_quality.json` for classification.

## Files

- `main.py`: complete source, including data loading and model evaluation.
- `walkthrough.ipynb`: executed notebook with explanation, code, recorded output and plots.
- `results/`: actual run metrics, test predictions and charts, not invented examples.
- `test_project.py`: automated output checks and a full rerun.
- `requirements.txt`: versions used in the verified run.

## Data and attribution

Source: https://www.kaggle.com/datasets/bumba5341/advertisingcsv

The task-linked Kaggle page credits uploader bumba5341. That page does not establish the original creator or redistribution rights. `source_sha256` records the exact downloaded CSV. Raw data is excluded from the archive.

## Review before publishing

Explain the target, the split, why the baseline matters, what leakage means, and one limitation. Re-run the code and check the outputs. A LinkedIn video/post and the official form are still required by CodeAlpha; neither is included as a completed step here.
