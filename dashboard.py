import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Models.transformer import TransformerForecaster

st.set_page_config(page_title="LLM Cost Forecasting", layout="wide")

# ── Load data ──────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/usage.csv")
    df["tokens_norm"] = df.groupby("account_id")["tokens"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    return df

# ── Load model ─────────────────────────────────
@st.cache_resource
def load_model():
    model = TransformerForecaster()
    model.load_state_dict(torch.load(
        "Models/transformer_model.pth", weights_only=True))
    model.eval()
    return model

df = load_data()
model = load_model()

# ── Sidebar ────────────────────────────────────
st.sidebar.title("LLM Cost Forecasting")
st.sidebar.markdown("Predicts next 7 days of token usage per account using a quantile transformer.")

account_ids = sorted(df["account_id"].unique())
selected = st.sidebar.selectbox("Select account", account_ids)
window = st.sidebar.slider("History window (days)", 14, 55, 28)

# ── Get account data ───────────────────────────
account = df[df["account_id"] == selected].copy()
values = account["tokens_norm"].values
raw_tokens = account["tokens"].values
plan = account["plan"].iloc[0]

if len(values) < window + 7:
    st.error("Not enough data for this account.")
    st.stop()

x = torch.tensor(
    values[0:window], dtype=torch.float32).unsqueeze(0)
actual = values[window:window + 7]
actual_raw = raw_tokens[window:window + 7]

# ── Run model ──────────────────────────────────
with torch.no_grad():
    pred = model(x).squeeze().numpy()

p10, p50, p90 = pred[:, 0], pred[:, 1], pred[:, 2]

# ── Header ─────────────────────────────────────
st.title(f"Account: {selected}")

col1, col2, col3 = st.columns(3)
col1.metric("Plan", plan.capitalize())
col2.metric("Avg daily tokens", f"{int(raw_tokens.mean()):,}")
col3.metric("Peak tokens", f"{int(raw_tokens.max()):,}")

st.divider()

# ── Forecast chart ─────────────────────────────
st.subheader("7-Day Token Usage Forecast")

fig, ax = plt.subplots(figsize=(12, 4))

days_hist = range(window)
days_fore = range(window, window + 7)

ax.plot(days_hist, values[:window],
        color="steelblue", label="History", linewidth=1.5)
ax.plot(days_fore, actual,
        color="green", label="Actual", linewidth=2)
ax.plot(days_fore, p50,
        color="purple", linestyle="--",
        label="Predicted (p50)", linewidth=2)
ax.fill_between(days_fore, p10, p90,
                alpha=0.2, color="purple",
                label="Uncertainty band (p10–p90)")
ax.axvline(x=window, color="gray",
           linestyle=":", linewidth=1)
ax.set_xlabel("Day")
ax.set_ylabel("Tokens (normalized)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

st.pyplot(fig)

# ── Forecast table ─────────────────────────────
st.subheader("Forecast breakdown")

table_data = []
for i in range(7):
    in_band = p10[i] <= actual[i] <= p90[i]
    table_data.append({
        "Day": f"Day +{i+1}",
        "Actual (norm)": round(float(actual[i]), 3),
        "p10": round(float(p10[i]), 3),
        "p50 (median)": round(float(p50[i]), 3),
        "p90": round(float(p90[i]), 3),
        "In band ✓": "✓" if in_band else "✗"
    })

table_df = pd.DataFrame(table_data)
st.dataframe(table_df, use_container_width=True)

# ── Calibration score ──────────────────────────
in_band_count = sum(
    1 for i in range(7) if p10[i] <= actual[i] <= p90[i])
coverage = in_band_count / 7

st.subheader("Model metrics")
m1, m2, m3 = st.columns(3)
mae = float(np.mean(np.abs(p50 - actual)))
m1.metric("MAE (normalized)", round(mae, 4))
m2.metric("Band coverage", f"{coverage:.0%}")
m3.metric("Days in band", f"{in_band_count}/7")

st.caption("Built with PyTorch · Quantile Transformer · Pinball Loss")