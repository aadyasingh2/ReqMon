"""RAMMA Backend FastAPI Application entrypoint with CORS support and route configuration."""

import os
import sys
from typing import List

# Ensure root directory is in sys.path for data imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pydantic import BaseModel

from data.prepare_dataset import prepare_datasets_and_train_baseline
from ramma_backend.database import init_db
from ramma_backend.model_adapter import get_model_adapter
from ramma_backend.monitoring_runtime import generate_monitor_config, run_monitor
from ramma_backend.router import router as backend_router
from ramma_backend.threshold_engine import calibrate_threshold, compute_reference_metric_values
from ramma_nlp.contributors import rank_contributors
from ramma_nlp.explain import explain_violation
from ramma_nlp.interpreter import interpret_requirement
from ramma_nlp.schema import InterpretedRequirement

load_dotenv()

# Initialize backend database tables
init_db()

app = FastAPI(
    title="RAMMA NLP & Backend Service",
    description="API for interpreting and persisting ML monitoring requirements.",
)

# CORS configuration allowing React local development origins
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include backend DB management routes (/requirements, etc.)
app.include_router(backend_router)


class RequirementInput(BaseModel):
    text: str


@app.post("/interpret", response_model=InterpretedRequirement, tags=["Interpretation"])
def interpret_endpoint(input_data: RequirementInput) -> InterpretedRequirement:
    """FastAPI POST endpoint for interpreting natural-language monitoring requirements."""
    try:
        return interpret_requirement(input_data.text)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Interpretation error: {str(e)}"
        ) from e


class MonitorCheckInput(BaseModel):
    requirement_text: str
    model_source: str = "local:models/baseline_classifier.pkl"
    data_source: str = "clean"


class ContributorItem(BaseModel):
    feature: str
    psi: float


class MonitorCheckOutput(BaseModel):
    observed_value: float
    business_requirement_threshold: float
    operator: str
    business_requirement_violation: bool
    operational_drift_violation: bool
    explanation: str
    contributors: List[ContributorItem]


@app.post("/monitor/check", response_model=MonitorCheckOutput, tags=["Monitoring"])
def monitor_check_endpoint(input_data: MonitorCheckInput) -> MonitorCheckOutput:
    """Runs end-to-end ML monitoring check pipeline on target dataset (clean or drifted) for a given requirement and model source."""
    try:
        ref_path = "data/reference.csv"
        clean_path = "data/production_clean.csv"
        drifted_path = "data/production_drifted.csv"
        model_path = "models/baseline_classifier.pkl"

        if not (
            os.path.exists(ref_path)
            and os.path.exists(clean_path)
            and os.path.exists(drifted_path)
            and os.path.exists(model_path)
        ):
            prepare_datasets_and_train_baseline()

        ds_type = input_data.data_source.strip().lower()
        eval_path = clean_path if ds_type == "clean" else drifted_path

        adapter = get_model_adapter(input_data.model_source)
        interpreted = interpret_requirement(input_data.requirement_text)

        ref_df = pd.read_csv(ref_path)
        ref_metric_values = compute_reference_metric_values(ref_df, interpreted.metric, adapter)
        calib_res = calibrate_threshold(ref_metric_values, method="percentile", operator=interpreted.operator)

        monitor_config = generate_monitor_config(
            interpreted,
            calibrated_threshold=calib_res["calibrated_threshold"],
            baseline_performance=calib_res["mean"],
            threshold_source="calibrated",
        )

        eval_df = pd.read_csv(eval_path)
        eval_result = run_monitor(monitor_config, eval_df, adapter)
        exp_res = explain_violation(monitor_config, eval_result["observed_value"])
        contributors_raw = rank_contributors(ref_df, eval_df, top_n=4)

        contributors = [
            ContributorItem(feature=c["feature"], psi=float(c["psi"]))
            for c in contributors_raw
        ]

        return MonitorCheckOutput(
            observed_value=float(eval_result["observed_value"]),
            business_requirement_threshold=float(monitor_config["business_requirement_threshold"]),
            operator=str(monitor_config["operator"]),
            business_requirement_violation=bool(eval_result["business_requirement_violation"]),
            operational_drift_violation=bool(eval_result["operational_drift_violation"]),
            explanation=str(exp_res["explanation"]),
            contributors=contributors,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Monitor check error: {str(e)}"
        ) from e


