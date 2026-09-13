"""Integration tests for ramma_backend database models and API endpoints."""

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ramma_backend.database import Base, get_db
from ramma_backend.main import app
from ramma_backend.models import Alert, Monitor, Requirement

# In-memory SQLite engine using StaticPool for thread-safe test sharing
SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_database():
    """Create fresh database tables before each test."""
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


def test_post_and_retrieve_requirement():
    """Test POSTing a requirement, interpreting it, saving to DB, and retrieving by ID."""
    req_text = "Recall must remain above 93%"

    # 1. POST /requirements
    response = client.post("/requirements", json={"text": req_text})
    assert response.status_code == 201
    data = response.json()

    assert "id" in data
    req_id = data["id"]
    assert data["raw_text"] == req_text
    assert data["metric"] == "recall"
    assert data["operator"] == ">="
    assert data["threshold"] == 0.93
    assert data["is_ambiguous"] is False

    # 2. GET /requirements/{id}
    get_response = client.get(f"/requirements/{req_id}")
    assert get_response.status_code == 200
    retrieved = get_response.json()

    assert retrieved["id"] == req_id
    assert retrieved["raw_text"] == req_text
    assert retrieved["metric"] == "recall"
    assert retrieved["threshold"] == 0.93


def test_list_requirements():
    """Test listing all requirements from the database."""
    req1 = "Recall must remain above 93%"
    req2 = "Precision for churn alerts must be at least 85%"

    res1 = client.post("/requirements", json={"text": req1})
    res2 = client.post("/requirements", json={"text": req2})
    assert res1.status_code == 201
    assert res2.status_code == 201

    list_response = client.get("/requirements")
    assert list_response.status_code == 200
    req_list = list_response.json()

    assert len(req_list) == 2
    metrics = [r["metric"] for r in req_list]
    assert "recall" in metrics
    assert "precision" in metrics


def test_get_nonexistent_requirement_404():
    """Test retrieving a non-existent requirement ID returns 404."""
    response = client.get("/requirements/9999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_database_models_relationships():
    """Test ORM relationships between Requirement, Monitor, and Alert tables."""
    db = TestingSessionLocal()

    # Create Requirement
    req = Requirement(
        raw_text="Fraud recall > 95%",
        metric="recall",
        operator=">",
        threshold=0.95,
        severity="critical",
        is_ambiguous=False,
    )
    db.add(req)
    db.commit()
    db.refresh(req)

    # Create Monitor linked to Requirement
    monitor = Monitor(requirement_id=req.id, config_json='{"check_interval": "1h"}')
    db.add(monitor)
    db.commit()
    db.refresh(monitor)

    # Create Alert linked to Monitor
    alert = Alert(
        monitor_id=monitor.id,
        observed_value=0.91,
        is_violation=True,
        explanation="Recall dropped to 91%",
        contributors_json='[{"feature": "transaction_amount", "psi": 0.45}]',
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)

    # Assert relationships
    assert len(req.monitors) == 1
    assert req.monitors[0].id == monitor.id
    assert len(monitor.alerts) == 1
    assert monitor.alerts[0].observed_value == 0.91

    db.close()


def test_cors_headers_on_api_endpoints():
    """Confirms cross-origin requests receive appropriate Access-Control-Allow-Origin headers."""
    origin = "http://localhost:5173"

    # GET /requirements with Origin header
    res_get = client.get("/requirements", headers={"Origin": origin})
    assert res_get.status_code == 200
    assert res_get.headers.get("access-control-allow-origin") == origin

    # POST /interpret with Origin header
    res_post = client.post("/interpret", json={"text": "Recall must remain above 93%"}, headers={"Origin": origin})
    assert res_post.status_code == 200
    assert res_post.headers.get("access-control-allow-origin") == origin

    # OPTIONS preflight request
    res_options = client.options(
        "/interpret",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert res_options.status_code == 200
    assert res_options.headers.get("access-control-allow-origin") == origin


def test_monitor_check_endpoint():
    """Test POST /monitor/check endpoint for real pipeline execution on clean and drifted datasets."""
    # 1. Clean dataset check (RandomForest recall PASS)
    res_clean = client.post(
        "/monitor/check",
        json={
            "requirement_text": "Recall must remain above 93%",
            "model_source": "local:models/baseline_classifier.pkl",
            "data_source": "clean",
        },
    )
    assert res_clean.status_code == 200
    data_clean = res_clean.json()

    assert data_clean["observed_value"] >= 0.93
    assert data_clean["business_requirement_violation"] is False
    assert data_clean["operational_drift_violation"] is False

    # 2. Drifted dataset check (RandomForest recall FAIL & DRIFT)
    res_drifted = client.post(
        "/monitor/check",
        json={
            "requirement_text": "Recall must remain above 93%",
            "model_source": "local:models/baseline_classifier.pkl",
            "data_source": "drifted",
        },
    )
    assert res_drifted.status_code == 200
    data_drifted = res_drifted.json()

    assert data_drifted["observed_value"] < 0.90
    assert data_drifted["business_requirement_violation"] is True
    assert data_drifted["operational_drift_violation"] is True



