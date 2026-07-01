import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# ── Load data ──────────────────────────────────
df = pd.read_csv("data/usage.csv")
df["tokens_norm"] = df.groupby("account_id")["tokens"].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-8)
)

# ── Rebuild model (same as train.py) ───────────
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, horizon=7):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ── Load saved weights ─────────────────────────
model = LSTMForecaster()
model.load_state_dict(torch.load("Models/lstm_model.pth"))
model.eval()

# ── Grab one account to visualize ─────────────
account = df[df["account_id"] == "acc_000"].copy()
values = account["tokens_norm"].values

window = 28
horizon = 7
x = torch.tensor(values[0:window], dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    pred = model(x).squeeze().numpy()

actual = values[window:window + horizon]

# ── Print results ──────────────────────────────
print("Day | Predicted | Actual")
print("-" * 30)
for i in range(horizon):
    print(f"Day {i+1:02d} | {pred[i]:+.4f}   | {actual[i]:+.4f}")

# ── Plot ───────────────────────────────────────
plt.figure(figsize=(10, 4))
plt.plot(range(window), values[:window], label="History", color="steelblue")
plt.plot(range(window, window + horizon), actual, label="Actual", color="green")
plt.plot(range(window, window + horizon), pred, label="Predicted", color="red", linestyle="--")
plt.axvline(x=window, color="gray", linestyle=":")
plt.legend()
plt.title("LSTM Forecast vs Actual — acc_000")
plt.xlabel("Day")
plt.ylabel("Tokens (normalized)")
plt.tight_layout()
plt.savefig("forecast_plot.png")
print("Plot saved to forecast_plot.png")