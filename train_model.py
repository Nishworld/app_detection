"""
train_model.py
--------------
Trains the core fraud classifier (XGBoost) plus a secondary anomaly
detector (Isolation Forest) for catching fraud patterns the supervised
model hasn't seen before.

Run:
    python train_model.py
    python train_model.py --data path/to/real_paysim.csv

Outputs (saved into ./artifacts/):
    xgb_fraud_model.joblib
    isolation_forest.joblib
    feature_columns.joblib
"""

import argparse
import os

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, classification_report,
                              precision_recall_curve, roc_auc_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from data_prep import load_and_engineer_features

ARTIFACT_DIR = "artifacts"


def train_xgboost(X_train, y_train):
    """
    Fraud data is heavily imbalanced (~1-2% positive class), so:
    - scale_pos_weight tells XGBoost to penalize missed fraud much more
      than false alarms, instead of just predicting "not fraud" always.
    - We optimize with logloss but will EVALUATE with PR-AUC/recall,
      because accuracy is meaningless on imbalanced data.
    """
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_isolation_forest(X_train, y_train):
    """
    Trained ONLY on legitimate transactions. Learns what 'normal' looks
    like, then flags anything that doesn't fit the pattern -- this is how
    you catch novel fraud types the supervised model has never seen.
    """
    legit_only = X_train[y_train == 0]
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.01,  # rough expected fraud rate
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(legit_only)
    return iso


def evaluate(model, X_test, y_test, model_name="Model"):
    proba = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)

    # pick a threshold that gives decent recall (catch fraud) while
    # keeping precision reasonable (don't annoy every legit customer)
    precisions, recalls, thresholds = precision_recall_curve(y_test, proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[max(best_idx - 1, 0)]

    preds = (proba >= best_threshold).astype(int)

    print(f"\n=== {model_name} evaluation ===")
    print(f"PR-AUC:  {pr_auc:.4f}   (this is the metric that matters most for fraud)")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Best threshold (by F1): {best_threshold:.4f}")
    print(classification_report(y_test, preds, digits=3))
    return best_threshold


def main(data_path=None):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    print("Loading and engineering features...")
    X, y, df = load_and_engineer_features(csv_path=data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Training XGBoost fraud classifier...")
    xgb_model = train_xgboost(X_train, y_train)
    threshold = evaluate(xgb_model, X_test, y_test, "XGBoost (supervised)")

    print("\nTraining Isolation Forest anomaly detector (unsupervised)...")
    iso_model = train_isolation_forest(X_train, y_train)
    # IsolationForest outputs -1 (anomaly) / 1 (normal) -- convert to 0/1 fraud-style label
    iso_preds_raw = iso_model.predict(X_test)
    iso_preds = (iso_preds_raw == -1).astype(int)
    print("Isolation Forest flagged", iso_preds.sum(), "anomalies out of", len(X_test),
          "| of those,", int(((iso_preds == 1) & (y_test == 1)).sum()), "were true fraud")

    joblib.dump(xgb_model, f"{ARTIFACT_DIR}/xgb_fraud_model.joblib")
    joblib.dump(iso_model, f"{ARTIFACT_DIR}/isolation_forest.joblib")
    joblib.dump(list(X.columns), f"{ARTIFACT_DIR}/feature_columns.joblib")
    joblib.dump(threshold, f"{ARTIFACT_DIR}/decision_threshold.joblib")

    print(f"\nSaved model artifacts to ./{ARTIFACT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None,
                         help="Path to real PaySim CSV. Omit to use synthetic data.")
    args = parser.parse_args()
    main(data_path=args.data)
