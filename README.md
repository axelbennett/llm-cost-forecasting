# LLM Cost Forecasting

Forecasting LLM token usage/cost per account for SaaS companies running 
LLM-powered features. Built with PyTorch — a quantile transformer that 
predicts the next 7 days of token usage as a calibrated uncertainty range 
(p10/p50/p90) rather than a single point estimate.

## Results

![Forecast Comparison](comparison_plot.png)

- Transformer MAE: **0.0982**
- LSTM baseline MAE: **0.2393**
- Improvement: **59% better accuracy**
- Band coverage: **7/7 days** inside p10-p90 interval


## Why this problem
Every SaaS company shipping LLM features now has unpredictable infrastructure 
costs. Token usage spikes with batch jobs, long prompts, and retry loops. 
Finance teams need calibrated forecasts — not just averages — to budget for 
worst-case exposure. This project builds that system.

## Architecture
- **Data:** 50 synthetic accounts × 90 days, 5 behavior archetypes 
  (steady, growing, bursty, declining, seasonal)
- **Baseline:** LSTM with MSE loss — single point forecast
- **Main model:** Temporal transformer with quantile output head, trained 
  with pinball loss for probabilistic forecasting
- **Output:** [7 days × 3 quantiles] — low / median / high per day

## Tech stack
Python · PyTorch · Streamlit · pandas · NumPy · matplotlib

## Setup
```bash
git clone https://github.com/axelbennett/llm-cost-forecasting.git
cd llm-cost-forecasting
python -m venv pytorch-env
pytorch-env\Scripts\activate  # Windows
pip install torch numpy pandas scikit-learn matplotlib streamlit
python data/generate_data.py
python Models/train_transformer.py
streamlit run dashboard.py

## Run the API locally

```bash
uvicorn api:app --reload
```
# API available at http://127.0.0.1:8000/docs

## Run with Docker

```bash
docker build -t llm-cost-forecasting .
docker run -p 8000:8000 llm-cost-forecasting
```
# API available at http://localhost:8000/docs
```

## Project structure
llm-cost-forecasting/
  data/
    generate_data.py       # synthetic data generator
  Models/
    transformer.py         # temporal transformer architecture
    train.py               # LSTM baseline training
    train_transformer.py   # transformer training with pinball loss
  losses/
    pinball_loss.py        # custom quantile loss function
  evaluate.py              # LSTM evaluation + forecast plot
  compare.py               # side by side model comparison
  dashboard.py             # streamlit forecast dashboard

  ## Progress
- [x] Synthetic data generator (50 accounts x 90 days)
- [x] LSTM baseline trained and evaluated
- [x] Temporal transformer with quantile output head
- [x] Custom pinball loss function
- [x] Model comparison — transformer wins by 59% MAE
- [x] Streamlit dashboard with live forecast band
- [x] FastAPI inference endpoint
- [x] Docker container
- [x] Deploy to Render (live URL)

## Status
🚧 Active build — portfolio project targeting ML/AI Engineer roles

## Live Demo
API: https://llm-cost-forecasting.onrender.com/docs



