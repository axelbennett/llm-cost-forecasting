# llm-cost-forecasting
Forecasting LLM token usage/cost per account using a quantile transformer model
## Problem
Given an account's last 28 days of token usage (plus account metadata like plan tier and company size), predict the next 7 days of usage at the 10th, 50th, and 90th percentiles. This gives a realistic worst-case/likely-case/best-case range instead of a single guess.

## Approach
- **Baseline:** simple moving average / exponential smoothing
- **Classical:** per-account ARIMA / Prophet
- **Main model:** a small temporal transformer (PyTorch) with quantile output heads, trained with pinball loss for probabilistic forecasting
- **Comparison model:** an LSTM trained with standard MSE loss, used as an ablation to demonstrate the value of the transformer + quantile loss approach

## Data
Synthetic usage data generated to simulate realistic account behavior patterns (steady, growing, bursty, declining, seasonal), correlated with account metadata such as plan tier and company size.

## Evaluation
- MAE / RMSE for point forecasts
- Calibration of prediction intervals (does the true value fall inside the 80% interval ~80% of the time?)
- Business framing: percentage of cost-overrun accounts flagged before the overrun occurs

## Status
🚧 Work in progress — built as a portfolio project to demonstrate applied PyTorch, time series forecasting, and uncertainty-aware modeling for AI/ML engineering roles.

## Progress
- [x] Synthetic data generator (50 accounts × 90 days)
- [x] LSTM baseline model trained (20 epochs, MSE loss)
- [ ] Evaluation script + forecast visualization
- [ ] Temporal transformer with quantile output
- [ ] Pinball loss implementation
- [ ] Streamlit dashboard
- [ ] FastAPI endpoint + Docker