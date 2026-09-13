"""Standalone Mock Remote Model Server for demonstrating RemoteEndpointModelAdapter over HTTP."""

import os
import sys
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import uvicorn

# Ensure src and repo root are in sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(repo_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from ramma_backend.model_adapter import LocalPickleModelAdapter

app = FastAPI(
    title="RAMMA Mock Remote Model Server",
    description="Simulates a remote model inference server exposing POST /predict.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy loading of local adapter wrapping baseline classifier
_adapter = None


def get_adapter() -> LocalPickleModelAdapter:
    global _adapter
    if _adapter is None:
        model_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "models", "baseline_classifier.pkl")
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Baseline classifier model pickle not found at {model_path}")
        _adapter = LocalPickleModelAdapter(model_path=model_path)
    return _adapter


class PredictRequest(BaseModel):
    data: List[Dict[str, Any]]


@app.get("/")
def health_check():
    return {"status": "ok", "service": "RAMMA Mock Remote Model Server"}


@app.post("/predict")
def predict_endpoint(payload: PredictRequest):
    if not payload.data:
        raise HTTPException(status_code=400, detail="Input 'data' array cannot be empty.")

    try:
        adapter = get_adapter()
        df = pd.DataFrame(payload.data)
        predictions = adapter.predict(df).tolist()

        probas = adapter.predict_proba(df)
        probabilities = probas.tolist() if probas is not None else None

        return {
            "predictions": predictions,
            "probabilities": probabilities,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}") from e


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")
