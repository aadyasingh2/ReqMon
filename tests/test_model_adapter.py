"""Tests for ModelAdapter implementations (LocalPickleModelAdapter and RemoteEndpointModelAdapter)."""

import os
import threading
import time
import numpy as np
import pandas as pd
import pytest
import uvicorn
from fastapi.testclient import TestClient

from ramma_backend.model_adapter import (
    LocalPickleModelAdapter,
    ModelAdapter,
    RemoteEndpointModelAdapter,
    get_model_adapter,
)
from scripts.mock_remote_model_server import app as server_app

MODEL_PATH = "models/baseline_classifier.pkl"
DATA_PATH = "data/production_clean.csv"


@pytest.fixture(scope="module")
def mock_server_port():
    """Runs mock_remote_model_server in a background thread for testing."""
    port = 5005
    config = uvicorn.Config(server_app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(1.0)  # Wait for server to start
    yield port
    server.should_exit = True
    thread.join(timeout=2.0)


def test_local_pickle_model_adapter():
    assert os.path.exists(MODEL_PATH)
    assert os.path.exists(DATA_PATH)

    adapter = LocalPickleModelAdapter(model_path=MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["is_fraud"]).select_dtypes(include=[np.number])

    preds = adapter.predict(X)
    probas = adapter.predict_proba(X)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(df)
    assert probas is not None
    assert probas.shape == (len(df), 2)


def test_model_adapter_equivalence(mock_server_port):
    """Proves LocalPickleModelAdapter and RemoteEndpointModelAdapter produce identical predictions."""
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["is_fraud"]).select_dtypes(include=[np.number])

    local_adapter = LocalPickleModelAdapter(model_path=MODEL_PATH)
    remote_adapter = RemoteEndpointModelAdapter(base_url=f"http://127.0.0.1:{mock_server_port}")

    local_preds = local_adapter.predict(X)
    remote_preds = remote_adapter.predict(X)

    np.testing.assert_array_equal(local_preds, remote_preds)

    local_probas = local_adapter.predict_proba(X)
    remote_probas = remote_adapter.predict_proba(X)

    assert local_probas is not None and remote_probas is not None
    np.testing.assert_allclose(local_probas, remote_probas, rtol=1e-5, atol=1e-5)


def test_remote_adapter_connection_error():
    """Confirms clear error message when remote endpoint is unreachable."""
    unreachable_adapter = RemoteEndpointModelAdapter(base_url="http://127.0.0.1:59999", timeout=1.0)
    df = pd.DataFrame({"feature1": [1.0, 2.0]})

    with pytest.raises(ConnectionError) as exc_info:
        unreachable_adapter.predict(df)

    assert "Failed to connect to remote model endpoint" in str(exc_info.value)


def test_get_model_adapter_factory():
    local_a = get_model_adapter("local")
    assert isinstance(local_a, LocalPickleModelAdapter)

    path_a = get_model_adapter(MODEL_PATH)
    assert isinstance(path_a, LocalPickleModelAdapter)

    remote_a = get_model_adapter("remote:http://localhost:5000")
    assert isinstance(remote_a, RemoteEndpointModelAdapter)
    assert remote_a.endpoint == "http://localhost:5000/predict"

    instance_a = get_model_adapter(local_a)
    assert instance_a is local_a


class DummyHighPerformanceModelAdapter(ModelAdapter):
    """Mock adapter simulating a high-performing model."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.ones(len(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        return np.column_stack([np.zeros(len(X)), np.ones(len(X))])


class DummyLowPerformanceModelAdapter(ModelAdapter):
    """Mock adapter simulating a lower-performing model."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([i % 2 for i in range(len(X))])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        p1 = np.array([0.6 if i % 2 == 1 else 0.4 for i in range(len(X))])
        return np.column_stack([1.0 - p1, p1])


def test_calibrate_threshold_differs_between_models():
    """Regression test confirming calibrate_threshold() with two different models produces different threshold values."""
    ref_df = pd.read_csv(DATA_PATH)
    high_adapter = DummyHighPerformanceModelAdapter()
    low_adapter = DummyLowPerformanceModelAdapter()

    from cli import compute_reference_metric_values
    from ramma_backend.threshold_engine import calibrate_threshold

    high_vals = compute_reference_metric_values(ref_df, "precision", high_adapter)
    low_vals = compute_reference_metric_values(ref_df, "precision", low_adapter)

    high_calib = calibrate_threshold(high_vals, method="percentile", operator=">=")
    low_calib = calibrate_threshold(low_vals, method="percentile", operator=">=")

    assert high_calib["mean"] != low_calib["mean"]
    assert high_calib["calibrated_threshold"] != low_calib["calibrated_threshold"]
    assert high_calib["calibrated_threshold"] > low_calib["calibrated_threshold"]
