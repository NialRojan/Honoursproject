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


def test_analyze_path_traversal_fixture_id_returns_404():
    response = client.post("/analyze", json={"fixture_id": "../../../../etc/passwd"})
    assert response.status_code == 404


def test_analyze_rejects_both_url_and_fixture_id():
    response = client.post(
        "/analyze", json={"url": "https://example.com", "fixture_id": "precheck_consent"}
    )
    assert response.status_code == 400


def test_easy_cancel_flow_scores_lower_than_precheck_consent():
    """Regression guard: easy_cancel_flow is a clean/benign fixture and should
    not out-rank precheck_consent, which has a genuine pre-checked consent
    box flagged by both the structure and visual engines. Before the Fix 5
    container-tag exclusion (engines/visual.py's _build_item_list), the
    Visual engine produced a spurious nag_to_upgrade finding on
    easy_cancel_flow (its <body>/<form> inherited "Upgrade now" text from an
    unrelated sibling button), inverting this ranking (42 vs 39)."""
    easy = client.post("/analyze", json={"fixture_id": "easy_cancel_flow"}).json()
    precheck = client.post("/analyze", json={"fixture_id": "precheck_consent"}).json()
    assert easy["risk_score"] < precheck["risk_score"]


def test_analyze_rejects_file_scheme_url():
    """Security test: file:// URLs should be rejected with 400 to prevent
    local file read via file:///etc/passwd or similar."""
    response = client.post("/analyze", json={"url": "file:///etc/passwd"})
    assert response.status_code == 400
    body = response.json()
    assert "detail" in body
    # Ensure the response doesn't leak file contents
    assert "root:" not in body.get("detail", "")


def test_analyze_rejects_invalid_scheme_url():
    """Security test: non-http(s) schemes should be rejected with 400."""
    response = client.post("/analyze", json={"url": "ftp://example.com"})
    assert response.status_code == 400
    body = response.json()
    assert "detail" in body


def test_analyze_rejects_unparseable_url():
    """Malformed URLs without a proper scheme should be rejected with 400."""
    response = client.post("/analyze", json={"url": "not-a-valid-url"})
    assert response.status_code == 400
    body = response.json()
    assert "detail" in body


def test_analyze_rejects_malformed_ipv6_url():
    """Malformed IPv6 URLs should be rejected with 400, not crash with 500.
    Regression test for unhandled ValueError from urlparse on malformed input."""
    response = client.post("/analyze", json={"url": "http://[::1"})
    assert response.status_code == 400
    body = response.json()
    assert "detail" in body
