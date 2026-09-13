"""Verification script for testing /interpret and /monitor/check API endpoints with clean and drifted data sources."""

import sys
import json
from fastapi.testclient import TestClient

from ramma_backend.main import app

client = TestClient(app)

def main():
    print("================================================================================")
    print("            VERIFYING API ENDPOINTS FOR FRONTEND WIRING (CLEAN vs DRIFTED)")
    print("================================================================================")

    # 1. Test POST /interpret (Unambiguous)
    print("\n[1] Testing POST /interpret (Unambiguous requirement)")
    res_interp = client.post("/interpret", json={"text": "Recall must remain above 93%"})
    assert res_interp.status_code == 200
    data_interp = res_interp.json()
    print("Response:\n", json.dumps(data_interp, indent=2))
    assert data_interp["metric"] == "recall"
    assert data_interp["operator"] == ">="
    assert data_interp["threshold"] == 0.93
    assert data_interp["is_ambiguous"] is False

    # 2. Test POST /interpret (Ambiguous)
    print("\n[2] Testing POST /interpret (Ambiguous requirement)")
    res_amb = client.post("/interpret", json={"text": "Is my precision good enough?"})
    assert res_amb.status_code == 200
    data_amb = res_amb.json()
    print("Response:\n", json.dumps(data_amb, indent=2))
    assert data_amb["is_ambiguous"] is True

    # 3. Test RANDOM_FOREST on CLEAN vs DRIFTED
    print("\n[3] Testing RANDOM_FOREST on PRODUCTION_CLEAN vs PRODUCTION_DRIFTED")
    res_rf_clean = client.post(
        "/monitor/check",
        json={
            "requirement_text": "Recall must remain above 93%",
            "model_source": "local:models/baseline_classifier.pkl",
            "data_source": "clean",
        },
    ).json()

    res_rf_drifted = client.post(
        "/monitor/check",
        json={
            "requirement_text": "Recall must remain above 93%",
            "model_source": "local:models/baseline_classifier.pkl",
            "data_source": "drifted",
        },
    ).json()

    print(f"  RANDOM_FOREST (CLEAN)   -> Observed: {res_rf_clean['observed_value']:.4f} | Biz Violation: {res_rf_clean['business_requirement_violation']} | Drift Violation: {res_rf_clean['operational_drift_violation']}")
    print(f"  RANDOM_FOREST (DRIFTED) -> Observed: {res_rf_drifted['observed_value']:.4f} | Biz Violation: {res_rf_drifted['business_requirement_violation']} | Drift Violation: {res_rf_drifted['operational_drift_violation']}")

    assert res_rf_clean["observed_value"] >= 0.93
    assert res_rf_clean["business_requirement_violation"] is False
    assert res_rf_clean["operational_drift_violation"] is False

    assert res_rf_drifted["observed_value"] < 0.90
    assert res_rf_drifted["business_requirement_violation"] is True
    assert res_rf_drifted["operational_drift_violation"] is True

    # 4. Test LOGISTIC_REG on CLEAN vs DRIFTED
    print("\n[4] Testing LOGISTIC_REG on PRODUCTION_CLEAN vs PRODUCTION_DRIFTED")
    res_lr_clean = client.post(
        "/monitor/check",
        json={
            "requirement_text": "Recall must remain above 93%",
            "model_source": "local:models/alt_classifier_logreg.pkl",
            "data_source": "clean",
        },
    ).json()

    res_lr_drifted = client.post(
        "/monitor/check",
        json={
            "requirement_text": "Recall must remain above 93%",
            "model_source": "local:models/alt_classifier_logreg.pkl",
            "data_source": "drifted",
        },
    ).json()

    print(f"  LOGISTIC_REG (CLEAN)   -> Observed: {res_lr_clean['observed_value']:.4f} | Biz Violation: {res_lr_clean['business_requirement_violation']} | Drift Violation: {res_lr_clean['operational_drift_violation']}")
    print(f"  LOGISTIC_REG (DRIFTED) -> Observed: {res_lr_drifted['observed_value']:.4f} | Biz Violation: {res_lr_drifted['business_requirement_violation']} | Drift Violation: {res_lr_drifted['operational_drift_violation']}")

    # 5. Test REMOTE_NODE on DRIFTED
    print("\n[5] Testing REMOTE_NODE on PRODUCTION_DRIFTED")
    res_remote = client.post(
        "/monitor/check",
        json={
            "requirement_text": "Recall must remain above 93%",
            "model_source": "remote:http://localhost:5000",
            "data_source": "drifted",
        },
    ).json()
    print(f"  REMOTE_NODE  (DRIFTED) -> Observed: {res_remote['observed_value']:.4f} | Biz Violation: {res_remote['business_requirement_violation']} | Drift Violation: {res_remote['operational_drift_violation']}")

    print("\n================================================================================")
    print("            ALL DATASET COMPARISON CHECKS PASSED SUCCESSFULLY!")
    print("================================================================================")

if __name__ == "__main__":
    main()
