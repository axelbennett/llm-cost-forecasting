from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Models.transformer import TransformerForecaster

app = FastAPI(
    title="LLM Cost Forecasting API",
    description="Predicts next 7 days of LLM token usage per account",
    version="1.0.0"
)

model = TransformerForecaster()
model.load_state_dict(torch.load(
    "Models/transformer_model.pth", weights_only=True))
model.eval()

class PredictRequest(BaseModel):
    account_id: str
    recent_tokens: List[float]

class DayForecast(BaseModel):
    day: int
    p10: float
    p50: float
    p90: float

class PredictResponse(BaseModel):
    account_id: str
    forecast: List[DayForecast]
    model_version: str = "transformer-v1"

@app.get("/")
def root():
    return {
        "name": "LLM Cost Forecasting API",
        "version": "1.0.0",
        "endpoints": ["/predict", "/health", "/docs"]
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.recent_tokens) != 28:
        raise HTTPException(
            status_code=400,
            detail=f"recent_tokens must be exactly 28 values, got {len(req.recent_tokens)}"
        )

    x = torch.tensor(
        req.recent_tokens, dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        pred = model(x).squeeze().numpy()

    forecast = [
        DayForecast(
            day=i + 1,
            p10=round(float(pred[i, 0]), 4),
            p50=round(float(pred[i, 1]), 4),
            p90=round(float(pred[i, 2]), 4),
        )
        for i in range(7)
    ]

    return PredictResponse(
        account_id=req.account_id,
        forecast=forecast
    )
