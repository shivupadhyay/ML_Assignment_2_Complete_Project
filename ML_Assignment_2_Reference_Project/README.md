# Breast Cancer Classification Benchmark

A reproducible machine learning project that compares five supervised classifiers on the Wisconsin Diagnostic Breast Cancer dataset and deploys an interactive Streamlit evaluator using pre-trained saved models.

## Overview

This project was designed to:

- train and compare five classifiers on the same held-out test set,
- evaluate them using six standard metrics: accuracy, ROC AUC, precision, recall, F1-score, and MCC,
- save fitted model pipelines for later deployment,
- expose a safe Streamlit app that loads saved models without retraining on uploaded test data.

## Project links

- GitHub repository: `https://github.com/shivupadhyay/ML_Assignment_2_Complete_Project/tree/main/ML_Assignment_2_Reference_Project`
- Streamlit app: `https://mlassignment2completeproject-makb5jpdg7ms4vgmek4jfb.streamlit.app/`

## Problem statement

Build and compare five supervised classifiers on one public classification dataset, assess the same held-out samples using accuracy, ROC AUC, precision, recall, F1-score, and Matthews correlation coefficient (MCC), and publish an interactive Streamlit evaluator that uses previously saved fitted models rather than fitting on uploaded test data.

## Dataset description

- Dataset: Wisconsin Diagnostic Breast Cancer (WDBC), distributed by scikit-learn and originally sourced from the UCI Machine Learning Repository.
- UCI original source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Instances: 569
- Input features: 30 numeric measurements
- Task: binary classification
- Class encoding: `target=0` means malignant, `target=1` means benign
- Data basis: 30 real-valued cell-nuclei measurements derived from digitized breast mass fine-needle aspirates
- Data split: stratified 80/20 train-test split with `random_state=42`
- Preprocessing: median imputation and StandardScaler inside individual pipelines, fitted only on training rows
- Files: `training_data.csv` and `test_data.csv` are generated from the published dataset; only test data should be uploaded to the app
- Important note: this is a benchmark project and is not intended for clinical diagnosis or medical use

## Models used and performance comparison

The table below reports metrics on the same untouched held-out 114-row test set, with class `1` (benign) treated as the positive class.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9035 | 0.9373 | 0.9420 | 0.9028 | 0.9220 | 0.7969 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Model-wise observations

| ML Model Name | Observation |
|---|---|
| Logistic Regression | Excellent linear baseline with the highest AUC and best overall balance across class metrics. Standardization supports stable optimization and interpretable coefficients. |
| Decision Tree | Lowest F1 and MCC on this split. It is easy to explain but less stable than the other models on this dataset partition. |
| KNN | Strong recall and F1 with standardized features, though prediction cost increases with the training set size. |
| Naive Bayes | Good AUC but comparatively weaker classification accuracy. Its Gaussian independence assumption may be less suitable for correlated tumor measurements. |
| Support Vector Machine | Ties Logistic Regression on accuracy, F1, and MCC; it learns nonlinear decision boundaries effectively when feature scaling is applied. |

### Overall winner

**Overall winner (tie-break: highest F1, then AUC): Logistic Regression.** Logistic Regression and SVM achieve the same test accuracy, F1, and MCC; Logistic Regression has slightly higher AUC on this split. These results depend on the fixed data split and chosen hyperparameters.

## Reproducible training and evaluation

1. Create a Python 3.11+ virtual environment: `python -m venv .venv`
2. Activate it:
   - Linux/macOS: `source .venv/bin/activate`
   - Windows PowerShell: `.venv\Scripts\Activate.ps1`
3. Install dependencies: `pip install -r requirements.txt`
4. Reproduce the saved models and generated data: `python train_models.py`
5. Launch the evaluator: `streamlit run app.py`
6. In the app, choose a model from the sidebar, inspect the bundled test data, and optionally upload `test_data.csv` for prediction and comparison
7. Capture a genuine screenshot from BITS Virtual Lab after successful execution

**Important:** `train_models.py` is the only code that calls `.fit()`. The Streamlit app loads the saved `.joblib` files and calls `.predict()` and `.predict_proba()` on the supplied test CSV. The model pipelines retain the training-fitted imputer and scaler, and no training occurs in the app itself.

## Repository structure

```text
ML_Assignment_2_Reference_Project/
├── app.py                       # Interactive evaluation dashboard
├── train_models.py              # Reproducible training and metrics generation
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
├── test_data.csv               # 114 held-out labeled test rows
├── training_data.csv           # 455 training rows used for reproducibility
├── evaluation_metrics.csv      # Recorded evaluation metrics
├── .gitignore
└── model/
    ├── metadata.json           # Feature names, target column, and model metadata
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

## Deploy on Streamlit Community Cloud

1. Create a public GitHub repository and commit the project, including all five `model/*.joblib` files, `test_data.csv`, and `requirements.txt`.
2. Visit https://streamlit.io/cloud and sign in with GitHub.
3. Choose **New app**, select the repository, set the branch to `main`, and set the entry file to `app.py`.
4. Deploy the app and verify that all five models are displayed correctly.
5. Test both the bundled CSV and an uploaded `test_data.csv` to confirm evaluation and confusion matrices render properly.
6. Update the README and the final PDF submission with the live URL and repository URL.

## Required PDF submission and validation

Submit one PDF in this order:

1. clickable GitHub repository link,
2. clickable live Streamlit app link,
3. your own screenshot of the project running in BITS Virtual Lab,
4. the final README content including dataset description, model metrics, observations, and winner.

A separate `submission_template.pdf` is included, but it must be replaced with your actual completed version and genuine screenshot.

### Pre-submit checks

- Both links publicly open successfully.
- All five trained artifact files are committed.
- Uploaded CSV includes the 30 original feature names plus `target`.
- No retraining occurs in the app.
- Screenshot clearly shows execution in BITS Virtual Lab.
- README and PDF match the final run and deployment state.

Submission deadline: **8 November 2026, 23:59** (per assignment sheet).

## Limitations and possible improvements

This project uses a single train/test split rather than nested cross-validation, and the reported metrics are based on one fixed held-out set. Model performance may fluctuate depending on the chosen split, and no hyperparameter tuning was performed on the held-out test data.

Possible extensions include:

- training-only cross-validation,
- per-class analysis emphasizing malignant recall,
- a model card or explainability panel,
- feature importance visualization and custom statistics for interpretation.

These improvements can provide deeper insight into model behavior while preserving the same evaluation discipline and no-leakage rule.
