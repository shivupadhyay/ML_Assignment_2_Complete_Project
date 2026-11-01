"""Reproducible model training. Run on BITS Virtual Lab and inspect every result."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parent
TARGET = 'target'
RANDOM_SEED = 42

def calculate_metrics(y_true, predictions, probabilities):
    return {
        'Accuracy': float(accuracy_score(y_true, predictions)),
        'AUC': float(roc_auc_score(y_true, probabilities)),
        'Precision': float(precision_score(y_true, predictions, zero_division=0)),
        'Recall': float(recall_score(y_true, predictions, zero_division=0)),
        'F1': float(f1_score(y_true, predictions, zero_division=0)),
        'MCC': float(matthews_corrcoef(y_true, predictions)),
    }

def main():
    raw = load_breast_cancer(as_frame=True)
    frame = raw.frame.copy()
    X = frame.drop(columns=[TARGET])
    y = frame[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )
    ROOT.joinpath('model').mkdir(exist_ok=True)
    train_rows = X_train.copy()
    train_rows[TARGET] = y_train
    test_rows = X_test.copy()
    test_rows[TARGET] = y_test
    test_rows.to_csv(ROOT / 'test_data.csv', index=False)
    # Local training sample is saved to support reproducibility. Only test rows go to the UI.
    train_rows.to_csv(ROOT / 'training_data.csv', index=False)
    models = {
        'Logistic Regression': LogisticRegression(max_iter=3000, random_state=RANDOM_SEED),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=RANDOM_SEED),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Support Vector Machine': SVC(kernel='rbf', C=1.0, probability=True, random_state=RANDOM_SEED),
    }
    names = {'Logistic Regression':'logistic_regression', 'Decision Tree':'decision_tree',
             'KNN':'knn', 'Naive Bayes':'naive_bayes', 'Support Vector Machine':'svm'}
    rows=[]
    for label, estimator in models.items():
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', estimator),
        ])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        scores = pipeline.predict_proba(X_test)[:, list(pipeline.classes_).index(1)]
        rows.append({'ML Model Name': label, **calculate_metrics(y_test, pred, scores)})
        joblib.dump(pipeline, ROOT / 'model' / f'{names[label]}.joblib')
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / 'evaluation_metrics.csv', index=False)
    metadata = {
        'dataset':'UCI Wisconsin Diagnostic Breast Cancer (WDBC), via scikit-learn',
        'source':'https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic',
        'instances':len(frame), 'features':len(X.columns),
        'training_size':len(X_train),'test_size':len(X_test),
        'random_state':RANDOM_SEED, 'test_fraction':0.20,
        'target_column':TARGET,'positive_class':1,
        'class_0':'malignant','class_1':'benign',
        'feature_names':list(X.columns), 'model_files':names,
    }
    (ROOT/'model'/'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(result.to_string(index=False, float_format=lambda v:f'{v:.4f}'))
    print('\nWinner by test F1:', result.sort_values(['F1','AUC'],ascending=False).iloc[0]['ML Model Name'])

if __name__=='__main__':
    main()
