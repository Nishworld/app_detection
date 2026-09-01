"""
app.py
------
The demo you actually show judges. A simple Streamlit dashboard where you
enter (or pick a random) transaction and see:
  - fraud score
  - flagged / not flagged
  - WHY (SHAP reasons)
  - a "trained via federated learning across banks, no raw data shared"
    badge, so the privacy story is visible, not just theoretical

Run:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

from data_prep import load_and_engineer_features
from explain_shap import score_and_explain

st.set_page_config(page_title="Fraud Detection + Privacy Platform", layout="centered")

st.title("🔒 Privacy-Preserving Fraud Detection")



@st.cache_data
def get_sample_transactions():
    X, y, df = load_and_engineer_features()
    return X, y, df


X, y, df = get_sample_transactions()

st.subheader("Pick a transaction to score")

mode = st.radio("Source", ["Random sample from dataset", "Enter manually"], horizontal=True)

if mode == "Random sample from dataset":
    if st.button("🎲 Pull a random transaction"):
        idx = np.random.randint(0, len(X))
        st.session_state["row"] = X.iloc[[idx]]
        st.session_state["actual_label"] = y.iloc[idx]

    if "row" in st.session_state:
        row = st.session_state["row"]
        st.dataframe(row.T.rename(columns={row.index[0]: "value"}), use_container_width=True)

        result = score_and_explain(row)

        col1, col2 = st.columns(2)
        col1.metric("Fraud score", f"{result['fraud_score']*100:.1f}%")
        col2.metric("Decision", "🚨 FLAGGED" if result["is_flagged"] else "✅ Approved")

        actual = st.session_state.get("actual_label")
        if actual is not None:
            st.caption(f"Ground truth label for this sample: "
                       f"{'FRAUD' if actual == 1 else 'legit'}")

        st.markdown("**Why this decision:**")
        for reason in result["top_reasons"]:
            st.write(f"- {reason}")

else:
    st.write("Enter basic transaction details (rest are auto-filled with typical values):")
    amount = st.number_input("Amount", min_value=0.0, value=500.0)
    old_bal = st.number_input("Sender's balance before transaction", min_value=0.0, value=1000.0)
    txn_type = st.selectbox("Transaction type", ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"])

    if st.button("Score this transaction"):
        # build a minimal single-row frame matching engineered features,
        # filling anything not entered with dataset medians
        base = X.median(numeric_only=True).to_frame().T
        base["amount"] = amount
        base["oldbalanceOrg"] = old_bal
        base["newbalanceOrig"] = max(old_bal - amount, 0)
        for col in base.columns:
            if col.startswith("type_"):
                base[col] = 1 if col == f"type_{txn_type}" else 0

        result = score_and_explain(base)

        col1, col2 = st.columns(2)
        col1.metric("Fraud score", f"{result['fraud_score']*100:.1f}%")
        col2.metric("Decision", "🚨 FLAGGED" if result["is_flagged"] else "✅ Approved")

        st.markdown("**Why this decision:**")
        for reason in result["top_reasons"]:
            st.write(f"- {reason}")

st.divider()
st.caption(
    "Pipeline: XGBoost classifier + Isolation Forest anomaly detector + "
    "SHAP explainability, trained on data federated across simulated banks "
    "using Flower — no bank's raw transactions ever left its own partition."
)
