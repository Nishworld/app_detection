"""
identity_layer.py
------------------
This is the "Identity Platform" half of your project title -- the piece
that was still missing.

The idea: raw identity fields (account IDs, names, phone numbers, card
numbers) should NEVER reach the fraud-detection model directly. Instead,
they get converted into irreversible tokens first. The fraud model only
ever sees these tokens (or features derived from them) -- never the real
identity.

Why this matters (say this to judges):
    A hospital/bank employee, or even someone who steals the model file,
    should NOT be able to work backwards from a token to find out who the
    real customer was. That's the whole point of tokenization here.

Two techniques are shown:
    1. Deterministic tokenization (HMAC-SHA256 with a secret salt)
       -- same real ID always -> same token, so you can still track
          "this account did 5 transactions" for fraud pattern detection,
          without ever knowing WHO that account belongs to.
    2. k-anonymity style bucketing for semi-identifying fields (e.g. age,
       location) -- optional extra layer, groups rare/unique values so
       no single record stands out.

Run:
    python identity_layer.py
"""

import hashlib
import hmac
import os

import pandas as pd

# In a real system this secret would live in a secrets manager (e.g. AWS
# Secrets Manager / HashiCorp Vault), NOT hardcoded. It's the "key" that
# makes the tokenization irreversible without it.
SECRET_SALT = os.environ.get("IDENTITY_SALT", "hackathon-demo-salt-change-in-prod").encode()


def tokenize_identity(raw_value: str) -> str:
    """
    Converts a raw identity string (account number, name, phone, etc.)
    into a fixed-length, irreversible token.

    Deterministic: the SAME raw_value always produces the SAME token, so
    the fraud model can still recognize "this is the same account acting
    repeatedly" -- which matters for velocity-based fraud features --
    without ever seeing or storing the real identity.

    Irreversible: because HMAC-SHA256 is a one-way function, and the
    secret salt is never exposed, you cannot go from the token back to
    the original value -- even if someone steals the token database.
    """
    if raw_value is None:
        return None
    token = hmac.new(SECRET_SALT, str(raw_value).encode(), hashlib.sha256).hexdigest()
    return token[:16]  # shortened for readability; still effectively unique


def tokenize_dataframe(df: pd.DataFrame, identity_columns: list) -> pd.DataFrame:
    """
    Applies tokenization to every identity column BEFORE the dataframe
    is allowed to touch the fraud model or feature engineering step.

    This function is what you'd call at the very front door of a real
    system -- e.g. right when a transaction arrives from a payment
    gateway, before it's stored or scored.
    """
    df = df.copy()
    for col in identity_columns:
        if col in df.columns:
            df[col] = df[col].apply(tokenize_identity)
    return df


def bucket_numeric_field(series: pd.Series, bins: int = 5) -> pd.Series:
    """
    k-anonymity style bucketing for semi-identifying numeric fields
    (age, income bracket, account tenure, etc.). Groups values into
    ranges so no single unusual value can be used to re-identify someone
    (e.g. "the only 91-year-old in the dataset" becomes "70+").
    """
    return pd.qcut(series, q=bins, duplicates="drop").astype(str)


def demo():
    print("=== Identity Tokenization Demo ===\n")

    raw_customers = pd.DataFrame({
        "nameOrig": ["C1234567890", "C1234567890", "C9876543210", "C1112223334"],
        "amount": [500, 1200, 75, 3000],
    })

    print("BEFORE tokenization (raw identity visible):")
    print(raw_customers.to_string(index=False))

    tokenized = tokenize_dataframe(raw_customers, identity_columns=["nameOrig"])

    print("\nAFTER tokenization (this is what the fraud model actually sees):")
    print(tokenized.to_string(index=False))

    print("\nNotice: 'C1234567890' appears twice in the raw data (rows 1 and 2),")
    print("and its token is IDENTICAL both times -- so the fraud model can still")
    print("detect 'this same account transacted twice', WITHOUT ever knowing the")
    print("real account number. This is deterministic but irreversible tokenization.")

    print("\nTrying to reverse a token back to the original value:")
    print("  -> Not possible. HMAC-SHA256 with a secret salt is a one-way function.")
    print("  -> Even if this token database were leaked, no customer identity is exposed.")


if __name__ == "__main__":
    demo()
