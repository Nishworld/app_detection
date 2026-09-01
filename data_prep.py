"""
data_prep.py
------------
Loads a PaySim-style transaction dataset (download from Kaggle:
"Synthetic Financial Datasets For Fraud Detection" / PaySim) and turns it
into ML-ready features.

If you don't have the CSV yet, this file can also GENERATE a synthetic
dataset that looks like PaySim, so you can start coding immediately and
swap in the real file later without changing any other code.

Expected raw columns (PaySim format):
    step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
    nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

Usage:
    python data_prep.py                # generates synthetic_transactions.csv
    from data_prep import load_and_engineer_features
    X, y, df = load_and_engineer_features("your_data.csv")
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 42


def generate_synthetic_paysim(n_rows: int = 50_000, fraud_rate: float = 0.012,
                               n_clients: int = 3, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Creates a synthetic PaySim-like dataset so you can build/test your whole
    pipeline before (or without) downloading the real Kaggle dataset.

    Adds a `client_id` column (0..n_clients-1) that simulates "different
    banks" -- this is what federated_train.py splits on later.
    """
    rng = np.random.default_rng(seed)

    n_fraud = int(n_rows * fraud_rate)
    n_legit = n_rows - n_fraud

    types = rng.choice(
        ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"],
        size=n_rows, p=[0.35, 0.20, 0.25, 0.15, 0.05]
    )

    amount = np.concatenate([
        rng.gamma(shape=2.0, scale=150, size=n_legit),      # legit: smaller, tighter spread
        rng.gamma(shape=2.0, scale=900, size=n_fraud),      # fraud: larger, more erratic
    ])
    old_bal_org = rng.uniform(0, 20_000, size=n_rows)
    # fraud transactions tend to drain the account close to zero
    drain_factor = np.concatenate([
        rng.uniform(0.05, 0.6, size=n_legit),
        rng.uniform(0.7, 1.0, size=n_fraud),
    ])
    new_bal_org = np.clip(old_bal_org - old_bal_org * drain_factor, 0, None)

    old_bal_dest = rng.uniform(0, 20_000, size=n_rows)
    new_bal_dest = old_bal_dest + amount

    step = rng.integers(1, 745, size=n_rows)  # PaySim uses "step" as an hour counter (1 month)

    is_fraud = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)]).astype(int)

    df = pd.DataFrame({
        "step": step,
        "type": types,
        "amount": amount,
        "nameOrig": [f"C{rng.integers(1e8, 1e9)}" for _ in range(n_rows)],
        "oldbalanceOrg": old_bal_org,
        "newbalanceOrig": new_bal_org,
        "nameDest": [f"M{rng.integers(1e8, 1e9)}" for _ in range(n_rows)],
        "oldbalanceDest": old_bal_dest,
        "newbalanceDest": new_bal_dest,
        "isFraud": is_fraud,
    })

    # shuffle rows so fraud isn't all clumped at the end
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # simulate "which bank" this transaction belongs to -- used for federated learning
    df["client_id"] = rng.integers(0, n_clients, size=len(df))

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turns raw transaction columns into model-ready numeric features.
    Keep this function separate from training so both train_model.py and
    federated_train.py can reuse identical feature logic.
    """
    df = df.copy()

    # --- balance-consistency features (very predictive for fraud) ---
    df["orig_balance_delta"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["dest_balance_delta"] = df["newbalanceDest"] - df["oldbalanceDest"]
    # a mismatch between amount sent and actual balance change is suspicious
    df["orig_balance_error"] = df["orig_balance_delta"] - df["amount"]
    df["dest_balance_error"] = df["dest_balance_delta"] - df["amount"]

    # --- ratio features ---
    df["amount_to_oldbalance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1.0)
    df["drained_account"] = (df["newbalanceOrig"] < 1.0).astype(int)

    # --- transaction type one-hot ---
    df = pd.get_dummies(df, columns=["type"], prefix="type")

    # --- per-account velocity feature: how many transactions has this
    #     origin account made so far in the dataset (simple proxy; in a real
    #     system you'd compute this from a rolling time window) ---
    df["orig_txn_count"] = df.groupby("nameOrig")["nameOrig"].transform("count")

    feature_cols = [c for c in df.columns if c not in
                    ["nameOrig", "nameDest", "isFraud", "isFlaggedFraud", "client_id"]]

    return df, feature_cols


def load_and_engineer_features(csv_path: str = None, n_clients: int = 3):
    """
    Main entry point. If csv_path is given, loads that file (must match
    PaySim column schema). Otherwise generates synthetic data.

    Returns: X (features), y (labels), df (full engineered dataframe,
    including client_id for federated_train.py)
    """
    if csv_path:
        df = pd.read_csv(csv_path)
        if "client_id" not in df.columns:
            rng = np.random.default_rng(RANDOM_STATE)
            df["client_id"] = rng.integers(0, n_clients, size=len(df))
    else:
        df = generate_synthetic_paysim(n_clients=n_clients)

    df, feature_cols = engineer_features(df)
    X = df[feature_cols]
    y = df["isFraud"]
    return X, y, df


if __name__ == "__main__":
    df = generate_synthetic_paysim()
    df.to_csv("synthetic_transactions.csv", index=False)
    print(f"Generated synthetic_transactions.csv with {len(df)} rows "
          f"({df['isFraud'].sum()} fraud, {df['isFraud'].mean()*100:.2f}% fraud rate)")
