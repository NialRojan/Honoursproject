from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_fixtures_endpoint_lists_fixtures():
    response = client.get("/fixtures")
    assert response.status_code == 200
    assert "precheck_consent" in response.json()["fixtures"]


def test_analyze_fixture_returns_valid_shape():
    response = client.post("/analyze", json={"fixture_id": "precheck_consent"})
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["risk_score"] <= 100
    assert isinstance(body["findings"], list)
    assert isinstance(body["screenshot_base64"], str) and len(body["screenshot_base64"]) > 0
    assert isinstance(body["elements"], list)


def test_analyze_missing_fixture_returns_error():
    response = client.post("/analyze", json={"fixture_id": "does_not_exist"})
    assert response.status_code == 404
