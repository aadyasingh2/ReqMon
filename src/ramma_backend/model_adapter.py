"""Model Adapter module defining pluggable model sources (local pickle vs remote HTTP endpoint)."""

from abc import ABC, abstractmethod
import os
from typing import Any, Dict, Union
import joblib
import numpy as np
import pandas as pd
import requests


class ModelAdapter(ABC):
    """Abstract Base Class for model inference adapters."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates class predictions for input feature DataFrame."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        """Generates class prediction probabilities for input feature DataFrame, if supported."""
        pass


class LocalPickleModelAdapter(ModelAdapter):
    """Adapter for loading and running inference on local scikit-learn pickle models."""

    def __init__(self, model_path: str = "models/baseline_classifier.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        self.model_path = model_path
        self.model = joblib.load(model_path)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array(self.model.predict(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        if hasattr(self.model, "predict_proba"):
            try:
                probas = self.model.predict_proba(X)
                return np.array(probas)
            except Exception:
                return None
        return None


class RemoteEndpointModelAdapter(ModelAdapter):
    """Adapter for invoking a remote HTTP endpoint for model inference."""

    def __init__(self, base_url: str = "http://localhost:5000", timeout: float = 10.0):
        url = base_url.strip()
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"http://{url}"
        self.base_url = url.rstrip("/")
        self.endpoint = f"{self.base_url}/predict" if not self.base_url.endswith("/predict") else self.base_url
        self.timeout = timeout

    def _post_payload(self, X: pd.DataFrame) -> Dict[str, Any]:
        payload = {"data": X.to_dict(orient="records")}
        try:
            response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Failed to connect to remote model endpoint at '{self.endpoint}'. "
                f"Ensure the remote model server is running. Original error: {str(e)}"
            ) from e

        if response.status_code != 200:
            raise RuntimeError(
                f"Remote model endpoint at '{self.endpoint}' returned status code {response.status_code}. "
                f"Response body: {response.text}"
            )

        try:
            res_json = response.json()
        except Exception as e:
            raise ValueError(
                f"Remote model endpoint at '{self.endpoint}' returned invalid non-JSON payload."
            ) from e

        if "predictions" not in res_json:
            raise ValueError(
                f"Remote model endpoint JSON response missing required 'predictions' field. Received keys: {list(res_json.keys())}"
            )

        return res_json

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        res_json = self._post_payload(X)
        return np.array(res_json["predictions"])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        res_json = self._post_payload(X)
        if "probabilities" in res_json and res_json["probabilities"] is not None:
            return np.array(res_json["probabilities"])
        return None


def get_model_adapter(source: Union[str, ModelAdapter] = "models/baseline_classifier.pkl") -> ModelAdapter:
    """Factory function resolving a string specifier or ModelAdapter instance into a concrete ModelAdapter."""
    if isinstance(source, ModelAdapter):
        return source

    if not isinstance(source, str):
        raise TypeError(f"Expected model source to be a str or ModelAdapter, got {type(source)}")

    source_str = source.strip()

    if source_str.startswith("remote:"):
        url = source_str[len("remote:") :].strip()
        return RemoteEndpointModelAdapter(base_url=url)
    elif source_str.startswith("local:"):
        path = source_str[len("local:") :].strip()
        return LocalPickleModelAdapter(model_path=path)
    elif source_str == "local":
        return LocalPickleModelAdapter(model_path="models/baseline_classifier.pkl")
    else:
        # File path or fallback local model path
        return LocalPickleModelAdapter(model_path=source_str)
