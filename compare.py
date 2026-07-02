import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Models.transformer import TransformerForecaster

# ── Load data ──────────────────────────────────
df = pd.read_csv("data/usage.csv")
df["tokens_norm"] = df.groupby("account_id")["tokens"].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-8)
)

# ── LSTM model ─────────────────────────────────
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, horizon=7):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ── Load both models ───────────────────────────
lstm = LSTMForecaster()
lstm.load_state_dict(torch.load(
    "Models/lstm_model.pth", weights_only=True))
lstm.eval()

transformer = TransformerForecaster()
transformer.load_state_dict(torch.load(
    "Models/transformer_model.pth", weights_only=True))
transformer.eval()

# ── Get one account ────────────────────────────
account = df[df["account_id"] == "acc_000"].copy()
values = account["tokens_norm"].values

window = 28
horizon = 7
x = torch.tensor(values[0:window], dtype=torch.float32).unsqueeze(0)
actual = values[window:window + horizon]

# ── Run both models ────────────────────────────
with torch.no_grad():
    lstm_pred = lstm(x).squeeze().numpy()
    transformer_pred = transformer(x).squeeze().numpy()

# transformer_pred shape: [7, 3] → p10, p50, p90
p10 = transformer_pred[:, 0]
p50 = transformer_pred[:, 1]
p90 = transformer_pred[:, 2]

# ── Print comparison table ─────────────────────
print(f"{'Day':<5} {'Actual':>8} {'LSTM':>8} {'p10':>8} {'p50':>8} {'p90':>8}")
print("-" * 50)
for i in range(horizon):
    print(f"Day {i+1:<2} {actual[i]:>8.3f} {lstm_pred[i]:>8.3f} "
          f"{p10[i]:>8.3f} {p50[i]:>8.3f} {p90[i]:>8.3f}")

# ── MAE comparison ─────────────────────────────
lstm_mae = np.mean(np.abs(lstm_pred - actual))
transformer_mae = np.mean(np.abs(p50 - actual))
print(f"\nLSTM MAE:        {lstm_mae:.4f}")
print(f"Transformer MAE: {transformer_mae:.4f}")
winner = "Transformer" if transformer_mae < lstm_mae else "LSTM"
print(f"Winner:          {winner}")

# ── Plot ───────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

days_hist = range(window)
days_fore = range(window, window + horizon)

# Top: LSTM
ax1.plot(days_hist, values[:window], color="steelblue", label="History")
ax1.plot(days_fore, actual, color="green", label="Actual", linewidth=2)
ax1.plot(days_fore, lstm_pred, color="red",
         linestyle="--", label="LSTM predicted")
ax1.axvline(x=window, color="gray", linestyle=":")
ax1.set_title("LSTM Forecast (point estimate)")
ax1.legend()
ax1.set_ylabel("Tokens (normalized)")

# Bottom: Transformer with uncertainty band
ax2.plot(days_hist, values[:window], color="steelblue", label="History")
ax2.plot(days_fore, actual, color="green", label="Actual", linewidth=2)
ax2.plot(days_fore, p50, color="purple",
         linestyle="--", label="Transformer p50")
ax2.fill_between(days_fore, p10, p90,
                 alpha=0.25, color="purple", label="p10–p90 band")
ax2.axvline(x=window, color="gray", linestyle=":")
ax2.set_title("Transformer Forecast (with uncertainty band)")
ax2.legend()
ax2.set_ylabel("Tokens (normalized)")
ax2.set_xlabel("Day")

plt.tight_layout()
plt.savefig("comparison_plot.png", dpi=150)
print("\nPlot saved to comparison_plot.png")