## a. Problem statement

Build and compare five supervised classifiers on **one public classification dataset**, assess the same held-out samples using accuracy, ROC AUC, precision, recall, F1, and Matthews correlation coefficient (MCC), then publish an interactive Streamlit evaluator that uses **previously saved fitted models**, never fitting on uploaded test data.

## b. Dataset description

- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC), distributed by scikit-learn, originally sourced from the UCI Machine Learning Repository.
- **UCI original:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **Instances:** 569 (required: at least 500); **numeric input features:** 30 (required: at least 12).
- **Task:** Binary classification. `target=0` means malignant, `target=1` means benign (positive class in the evaluation metrics).
- **Data basis:** 30 real-valued cell-nuclei measurements derived from digitized images of breast mass fine-needle aspirates. No extra features are engineered.
- **Split:** stratified 80% training (455 samples) and 20% testing (114 samples), `random_state=42`.
- **Preprocessing:** median imputation, StandardScaler, then classifier; all fitted **only on training rows** via scikit-learn pipelines. No test leakage.
- **Data files:** `training_data.csv` and `test_data.csv` are generated from the published dataset. Only test data should be uploaded to the app.
- **Caution:** Benchmark exercise only; no clinical use or medical diagnosis is intended.

## c. GitHub Repository Link

**REPLACE WITH YOUR PUBLIC REPOSITORY URL:** `https://github.com/shivupadhyay/ML_Assignment_2_Complete_Project/tree/main/ML_Assignment_2_Reference_Project`

**Streamlit Community Cloud app:** `https://mlassignment2completeproject-makb5jpdg7ms4vgmek4jfb.streamlit.app/`.

## d. Models used and comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9035 | 0.9373 | 0.9420 | 0.9028 | 0.9220 | 0.7969 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

All metrics are computed on the same untouched held-out **114-row test set**, with class **1 (benign)** as positive. The ROC AUC uses predicted class probabilities rather than predicted class labels. Test-set results are not a substitute for external validation.

### Model-wise observations

| ML Model Name | Observation |
|---|---|
| Logistic Regression | Excellent linear baseline with the highest AUC (tie-best accuracy/F1/MCC). Standardization helps optimization and coefficients remain interpretable. |
| Decision Tree | Lowest F1 and MCC here; its single-tree partitions are easy to explain but are less stable on this particular held-out split. |
| KNN | Strong F1 and recall on standardized feature space, though prediction cost grows with training set size. |
| Naive Bayes | Good AUC but lower classification accuracy; Gaussian conditional-independence assumption may be imperfect for correlated tumor measurements. |
| Support Vector Machine | Ties Logistic Regression for accuracy, F1, and MCC. RBF kernel can learn nonlinear boundaries but needs feature scaling. |

**Overall winner (tie-break: highest F1, then AUC): Logistic Regression.** Logistic Regression and SVM achieve equal test F1, accuracy and MCC; Logistic Regression has slightly higher AUC on this split. These results depend on the fixed data split and hyperparameters. For real malignant-case sensitivity, also examine the classification report and confusion matrix: benign is the positive label here by dataset convention.

## e. Reproducible training and evaluation

1. Create a Python 3.11+ environment (preferably in the BITS Virtual Lab): `python -m venv .venv`.
2. Activate it: Linux/macOS `source .venv/bin/activate`; Windows PowerShell `.venv\Scripts\Activate.ps1`.
3. Install dependencies: `pip install -r requirements.txt`.
4. To independently reproduce saved models and CSV: `python train_models.py`.
5. Start the evaluator: `streamlit run app.py`.
6. Select a model from the left sidebar, inspect results on bundled test data, then upload **`test_data.csv`** using the CSV option; enable all-model comparison.
7. Capture **a genuine screenshot with BITS Virtual Lab visible** after executing the project there.

**Important:** `train_models.py` is the only code that calls `.fit()`. The Streamlit app loads `.joblib` files and calls `.predict()` and `.predict_proba()` on the supplied test CSV. Model pipelines preserve training-fitted imputers and scalers. Only load model files from your own trusted repository; joblib uses Python pickle serialization.

## f. Repository structure

```text
ML_Assignment_2_Reference_Project/
├── app.py                       # Interactive dashboard (no fitting)
├── train_models.py              # Reproducible dataset split and training
├── requirements.txt
├── README.md
├── test_data.csv                # 114 held-out labeled test rows
├── training_data.csv            # 455 training rows, reproducibility
├── evaluation_metrics.csv       # Real measured model comparison
├── .gitignore
└── model/
    ├── metadata.json           # Features, target, model filenames, split
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

## g. Deploy on Streamlit Community Cloud

1. Create your own GitHub repository; commit the project, including all five `model/*.joblib` artifacts, `test_data.csv`, and `requirements.txt`.
2. At https://streamlit.io/cloud, sign in with GitHub and select **New app**.
3. Choose the public repository, the `main` branch and entry point `app.py`, then deploy.
4. Open the live link in a fresh browser, test the bundled CSV and an upload of `test_data.csv`, and validate that five results and confusion matrices display.
5. Update this README and the submission PDF with the real live URL and repository URL.

## h. Required PDF submission and validation

Submit **one PDF** in this required order: (1) clickable GitHub repository link, (2) clickable live Streamlit app link, (3) **your own screenshot of execution in BITS Virtual Lab**, then (4) the full README content with dataset description, model metrics, model-wise observations and overall winner. A separate `submission_template.pdf` is included for the layout, but **you must replace its placeholders and insert your genuine lab screenshot**.

**Pre-submit checks:** verify both live links publicly open, all five trained artifact files are committed, test upload includes 30 original feature names plus `target`, no retraining occurs in the app, screenshot proves BITS Virtual Lab execution, and README/PDF match your actual final run. Submission deadline: **8 November 2026, 23:59 (per assignment sheet)**. The brief states **15 marks total (10 implementation/repository + 4 Streamlit + 1 lab screenshot)**.

## i. Limitations and possible independent improvements

This is a single train/test split, not nested cross-validation. Metrics are on a small test set; performance may fluctuate with the split. No hyperparameter tuning was performed on held-out test data. For an original extension, add training-only cross-validation, per-class comparison emphasizing malignant recall, a model-card page, or a custom feature significance visualization and explain your decisions in your own words.
