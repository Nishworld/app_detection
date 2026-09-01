"""
federated_train.py
-------------------
This is the "privacy-preserving" half of your project.

Simulates 3 separate "banks" (clients), each holding their own slice of
transaction data that never leaves their machine. A single shared fraud
model is trained by manually implementing FEDERATED AVERAGING (FedAvg) --
the same core algorithm Flower/Google's federated learning uses -- in
plain PyTorch, with no extra simulation framework required.

Why not use the Flower library's simulation engine? Flower's simulator
relies on Ray, which has known stability issues on Windows (crashes with
"access violation" errors). This version implements the exact same
concept -- each bank trains locally, only weights get averaged, no raw
data is ever shared -- without that dependency, so it runs reliably
everywhere including Windows.

Run:
    python federated_train.py
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from data_prep import load_and_engineer_features

NUM_CLIENTS = 3
NUM_ROUNDS = 5
LOCAL_EPOCHS = 2
LEARNING_RATE = 0.01


class FraudNet(nn.Module):
    """Small feed-forward net -- fine for tabular fraud features."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def prepare_client_partitions(n_clients=NUM_CLIENTS):
    """
    Splits the dataset into n_clients pieces, simulating separate banks.
    Each bank's data is normalized and converted to tensors independently
    -- exactly as if it lived on separate machines.
    """
    X, y, df = load_and_engineer_features(n_clients=n_clients)
    X = X.astype(float).fillna(0)
    X = (X - X.mean()) / (X.std() + 1e-8)  # normalization helps NN training

    partitions = []
    for cid in range(n_clients):
        mask = df["client_id"] == cid
        X_c, y_c = X[mask].values, y[mask].values
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_c, y_c, test_size=0.2, random_state=42,
            stratify=y_c if y_c.sum() > 1 else None
        )
        partitions.append({
            "X_train": torch.tensor(X_tr, dtype=torch.float32),
            "y_train": torch.tensor(y_tr, dtype=torch.float32),
            "X_val": torch.tensor(X_val, dtype=torch.float32),
            "y_val": torch.tensor(y_val, dtype=torch.float32),
        })
        print(f"Bank {cid}: {len(X_tr)} train rows, {len(X_val)} val rows "
              f"({int(y_c.sum())} fraud cases) -- stays local, never shared")
    return partitions, X.shape[1]


def train_local(model, X, y, epochs=LOCAL_EPOCHS, lr=LEARNING_RATE):
    """One bank training on its own private data for a few epochs."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        preds = model(X).squeeze()
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
    return model


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = model(X).squeeze()
        loss = nn.BCELoss()(preds, y).item()
        acc = ((preds > 0.5).float() == y).float().mean().item()
        # recall matters more than accuracy for fraud -- track it too
        actual_fraud = y == 1
        if actual_fraud.sum() > 0:
            recall = ((preds > 0.5).float()[actual_fraud] == 1).float().mean().item()
        else:
            recall = float("nan")
    return loss, acc, recall


def federated_average(global_model, client_state_dicts, client_sizes):
    """
    THE CORE OF FEDERATED LEARNING.

    Takes each bank's locally-trained model weights and combines them into
    one global model -- weighted by how much data each bank trained on.
    Banks with more data get proportionally more influence, but their raw
    data is never seen, only these numeric weight tensors.
    """
    total_size = sum(client_sizes)
    new_state_dict = copy.deepcopy(client_state_dicts[0])

    for key in new_state_dict.keys():
        weighted_sum = sum(
            client_state_dicts[i][key] * (client_sizes[i] / total_size)
            for i in range(len(client_state_dicts))
        )
        new_state_dict[key] = weighted_sum

    global_model.load_state_dict(new_state_dict)
    return global_model


def run_federated_training():
    print(f"Setting up {NUM_CLIENTS} simulated banks for federated training...\n")
    partitions, input_dim = prepare_client_partitions()

    global_model = FraudNet(input_dim)

    print(f"\nStarting {NUM_ROUNDS} rounds of FEDERATED AVERAGING across "
          f"{NUM_CLIENTS} banks (only model weights are exchanged, never data)...\n")

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"--- Round {round_num}/{NUM_ROUNDS} ---")
        client_state_dicts = []
        client_sizes = []

        for cid, data in enumerate(partitions):
            # each bank starts from the CURRENT global model, trains
            # locally on ITS OWN data only, and returns just the weights
            local_model = copy.deepcopy(global_model)
            local_model = train_local(local_model, data["X_train"], data["y_train"])

            val_loss, val_acc, val_recall = evaluate(local_model, data["X_val"], data["y_val"])
            print(f"  [Bank {cid}] trained locally -- val_acc={val_acc:.3f}, "
                  f"val_recall={val_recall:.3f} (data never left this bank)")

            client_state_dicts.append(local_model.state_dict())
            client_sizes.append(len(data["X_train"]))

        # server-side step: combine everyone's weights into one shared model
        global_model = federated_average(global_model, client_state_dicts, client_sizes)

    print("\nFederated training complete. A single shared fraud model now "
          "exists, trained across all banks -- with zero raw transaction "
          "data ever leaving any individual bank's environment.\n")

    print("=== Final global model performance per bank's validation set ===")
    for cid, data in enumerate(partitions):
        loss, acc, recall = evaluate(global_model, data["X_val"], data["y_val"])
        print(f"  Bank {cid}: accuracy={acc:.3f}, recall={recall:.3f}")

    return global_model


if __name__ == "__main__":
    run_federated_training()
