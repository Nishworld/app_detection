"""
api.py
------
This turns your fraud-detection ML into a proper backend SERVICE using
FastAPI -- the same architectural pattern real companies use.

Why this matters (say this to judges):
    Right now, your Streamlit app calls the model directly, in the same
    process. That's fine for a demo, but it's NOT how real enterprise
    systems work. In a real bank, the fraud model needs to be usable by
    MANY different systems at once -- the mobile app, the web portal, an
    internal ops dashboard, a partner's payment gateway -- all hitting
    the SAME model over a network request, not each having their own
    copy of the code.

    This file exposes your fraud model as a REST API: any system, in any
    programming language, can send a transaction over HTTP and get back
    a fraud score + explanation. Your Streamlit dashboard becomes just
    ONE of many possible "clients" of this API -- exactly like how a
    real bank's mobile app and web app both call the same backend.

Run:
    uvicorn api:app --reload --port 8000

Then test it at:
    http://localhost:8000/docs   (auto-generated interactive API docs)

Example request (what any external system would send):
    POST http://localhost:8000/score
    {
        "amount": 5000,
        "oldbalanceOrg": 5200,
        "newbalanceOrig": 0,
        "oldbalanceDest": 1000,
        "newbalanceDest": 6000,
        "type": "CASH_OUT"
    }
"""

from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from data_prep import load_and_engineer_features
from explain_shap import score_and_explain
from identity_layer import tokenize_identity

app = FastAPI(
    title="Enterprise Fraud Detection API",
    description=(
        "Privacy-preserving fraud scoring service. Accepts a transaction, "
        "returns a fraud score, decision, and explanation. Identity fields "
        "are tokenized before processing -- raw customer identity is never "
        "logged or stored by this service."
    ),
    version="1.0.0",
)


class TransactionRequest(BaseModel):
    """
    What an external system (bank app, payment gateway, ops dashboard)
    sends us to get a transaction scored.
    """
    amount: float = Field(..., description="Transaction amount", gt=0)
    oldbalanceOrg: float = Field(..., description="Sender's balance before the transaction")
    newbalanceOrig: float = Field(..., description="Sender's balance after the transaction")
    oldbalanceDest: float = Field(0, description="Receiver's balance before the transaction")
    newbalanceDest: float = Field(0, description="Receiver's balance after the transaction")
    type: str = Field(..., description="Transaction type", examples=["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"])
    account_id: Optional[str] = Field(None, description="Sender's raw account ID -- tokenized internally, never stored raw")


class FraudScoreResponse(BaseModel):
    """What we send back to whoever called us."""
    fraud_score: float
    is_flagged: bool
    threshold_used: float
    top_reasons: list[str]
    account_token: Optional[str] = None  # tokenized, never the raw ID


def build_feature_row(txn: TransactionRequest) -> pd.DataFrame:
    """
    Converts an incoming API request into the exact feature format the
    trained model expects -- reusing the SAME feature engineering used
    during training, so predictions stay consistent with how the model
    was built.
    """
    X_reference, _, _ = load_and_engineer_features()
    base = X_reference.median(numeric_only=True).to_frame().T

    base["amount"] = txn.amount
    base["oldbalanceOrg"] = txn.oldbalanceOrg
    base["newbalanceOrig"] = txn.newbalanceOrig
    base["oldbalanceDest"] = txn.oldbalanceDest
    base["newbalanceDest"] = txn.newbalanceDest

    for col in base.columns:
        if col.startswith("type_"):
            base[col] = 1 if col == f"type_{txn.type}" else 0

    # recompute the derived features so they match the new amount/balances
    base["orig_balance_delta"] = base["oldbalanceOrg"] - base["newbalanceOrig"]
    base["dest_balance_delta"] = base["newbalanceDest"] - base["oldbalanceDest"]
    base["orig_balance_error"] = base["orig_balance_delta"] - base["amount"]
    base["dest_balance_error"] = base["dest_balance_delta"] - base["amount"]
    base["amount_to_oldbalance_ratio"] = base["amount"] / (base["oldbalanceOrg"] + 1.0)
    base["drained_account"] = (base["newbalanceOrig"] < 1.0).astype(int)

    return base


@app.get("/")
def root():
    return {
        "service": "Enterprise Fraud Detection API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    """Standard health-check endpoint -- used by real deployment systems
    to verify the service is alive before routing traffic to it."""
    return {"status": "healthy"}


@app.post("/score", response_model=FraudScoreResponse)
def score_transaction(txn: TransactionRequest):
    """
    Main endpoint. Accepts a transaction, returns fraud score + explanation.

    Privacy note: if account_id is provided, it is tokenized (irreversibly
    hashed) before being used or returned -- the raw ID is never logged,
    stored, or included in the response.
    """
    try:
        feature_row = build_feature_row(txn)
        result = score_and_explain(feature_row)

        account_token = tokenize_identity(txn.account_id) if txn.account_id else None

        return FraudScoreResponse(
            fraud_score=result["fraud_score"],
            is_flagged=result["is_flagged"],
            threshold_used=result["threshold_used"],
            top_reasons=result["top_reasons"],
            account_token=account_token,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Run train_model.py before starting the API."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
