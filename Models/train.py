import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ── 1. Load data ──────────────────────────────
df = pd.read_csv("data/usage.csv")

# ── 2. Normalize tokens per account ───────────
df["tokens_norm"] = df.groupby("account_id")["tokens"].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-8)
)

# ── 3. Dataset class ──────────────────────────
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

# ── 4. Split train/val ────────────────────────
dataset = UsageDataset(df)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# ── 5. Simple LSTM model ──────────────────────
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, horizon=7):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ── 6. Training loop ──────────────────────────
model = LSTMForecaster()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(20):
    model.train()
    train_loss = 0
    for x, y in train_loader:
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            pred = model(x)
            val_loss += loss_fn(pred, y).item()

    print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

print("Done! Model trained.")
torch.save(model.state_dict(), "models/lstm_model.pth")
print("Model saved to models/lstm_model.pth")
