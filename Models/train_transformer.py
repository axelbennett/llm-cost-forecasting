import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Models.transformer import TransformerForecaster
from losses.pinball_loss import pinball_loss

# ── Load data ──────────────────────────────────
df = pd.read_csv("data/usage.csv")
df["tokens_norm"] = df.groupby("account_id")["tokens"].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-8)
)

# ── Dataset ────────────────────────────────────
class UsageDataset(Dataset):
    def __init__(self, df, window=28, horizon=7):
        self.samples = []
        for acc_id, group in df.groupby("account_id"):
            values = group["tokens_norm"].values
            for i in range(len(values) - window - horizon):
                x = values[i : i + window]
                y = values[i + window : i + window + horizon]
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), \
               torch.tensor(y, dtype=torch.float32)

# ── Split ──────────────────────────────────────
dataset = UsageDataset(df)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# ── Model ──────────────────────────────────────
model = TransformerForecaster()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
QUANTILES = [0.1, 0.5, 0.9]

best_val_loss = float('inf')
patience = 5
patience_counter = 0

# ── Training loop ──────────────────────────────
for epoch in range(30):
    model.train()
    train_loss = 0
    for x, y in train_loader:
        pred = model(x)
        loss = pinball_loss(pred, y, QUANTILES)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            pred = model(x)
            val_loss += pinball_loss(pred, y, QUANTILES).item()

    train_avg = train_loss / len(train_loader)
    val_avg = val_loss / len(val_loader)
    print(f"Epoch {epoch+1:02d} | Train: {train_avg:.4f} | Val: {val_avg:.4f}")

    # early stopping
    if val_avg < best_val_loss:
        best_val_loss = val_avg
        torch.save(model.state_dict(), "Models/transformer_model.pth")
        print(f"           ✓ saved best model")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

print("Done! Best val loss:", round(best_val_loss, 4))