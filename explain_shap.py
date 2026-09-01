"""
explain_shap.py
---------------
Adds the "why was this flagged?" layer on top of your trained XGBoost
model, using SHAP. This is the piece that makes your project look
enterprise-ready: fraud teams legally/practically need to justify blocks,
not just get a black-box score.

Run:
    python explain_shap.py
(after you've run train_model.py at least once)
"""

import joblib
import numpy as np
import pandas as pd
import shap

from data_prep import load_and_engineer_features

ARTIFACT_DIR = "artifacts"


def load_artifacts():
    model = joblib.load(f"{ARTIFACT_DIR}/xgb_fraud_model.joblib")
    feature_cols = joblib.load(f"{ARTIFACT_DIR}/feature_columns.joblib")
    threshold = joblib.load(f"{ARTIFACT_DIR}/decision_threshold.joblib")
    return model, feature_cols, threshold


def explain_transaction(model, explainer, X_row: pd.DataFrame, feature_cols, top_n=3):
    """
    Returns a plain-English style list of the top_n features that pushed
    this specific transaction's fraud score up (or down).
    """
    shap_values = explainer(X_row[feature_cols])
    values = shap_values.values[0]
    contributions = list(zip(feature_cols, values, X_row[feature_cols].values[0]))

    # sort by absolute impact, biggest first
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    reasons = []
    for feat_name, shap_val, feat_value in contributions[:top_n]:
        direction = "increased" if shap_val > 0 else "decreased"
        reasons.append(
            f"{feat_name} = {feat_value:.2f} ({direction} fraud risk by {abs(shap_val):.3f})"
        )
    return reasons


def score_and_explain(transaction_features: pd.DataFrame):
    """
    Main function your API / Streamlit app should call.
    transaction_features: single-row DataFrame with the same columns
    produced by data_prep.engineer_features().

    Returns: dict with fraud_score, is_flagged, top_reasons
    """
    model, feature_cols, threshold = load_artifacts()
    explainer = shap.TreeExplainer(model)

    # make sure column order matches training
    X_row = transaction_features.reindex(columns=feature_cols, fill_value=0)

    fraud_score = float(model.predict_proba(X_row)[:, 1][0])
    is_flagged = fraud_score >= threshold
    reasons = explain_transaction(model, explainer, X_row, feature_cols)

    return {
        "fraud_score": round(fraud_score, 4),
        "is_flagged": bool(is_flagged),
        "threshold_used": round(float(threshold), 4),
        "top_reasons": reasons,
    }


if __name__ == "__main__":
    # demo: explain a handful of test transactions
    X, y, df = load_and_engineer_features()
    sample_idx = np.random.default_rng(0).choice(len(X), size=5, replace=False)

    for idx in sample_idx:
        row = X.iloc[[idx]]
        result = score_and_explain(row)
        print(f"\nTransaction #{idx} (actual label: {'FRAUD' if y.iloc[idx] == 1 else 'legit'})")
        print(f"  Fraud score: {result['fraud_score']}  |  Flagged: {result['is_flagged']}")
        for r in result["top_reasons"]:
            print(f"   - {r}")
