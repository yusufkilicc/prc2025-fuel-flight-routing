"""
API smoke tests via FastAPI TestClient (uses bundled demo data).
Skipped automatically if httpx/starlette TestClient is unavailable.
Run:  pytest -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web"))

try:
    from fastapi.testclient import TestClient
    import web_app
    client = TestClient(web_app.app)
except Exception as e:  # missing httpx, data, etc.
    client = None
    _reason = str(e)

pytestmark = pytest.mark.skipif(client is None, reason="TestClient/app unavailable")


def test_airports_endpoint():
    r = client.get("/api/airports")
    assert r.status_code == 200
    j = r.json()
    assert len(j["airports"]) > 50
    assert len(j["aircraft"]) > 5
    assert "shap" in j


def test_score_ok():
    r = client.get("/api/score", params={"o": "LTFM", "d": "KJFK", "ac": "A359"})
    assert r.status_code == 200
    j = r.json()
    assert len(j["scenarios"]) == 3
    assert all("fuel_lo" in s and "fuel_hi" in s for s in j["scenarios"])
    assert j["origin"]["iata"] == "IST"


def test_score_rejects_same_airport():
    r = client.get("/api/score", params={"o": "LTFM", "d": "LTFM", "ac": "A359"})
    assert r.status_code == 400


def test_index_serves_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "Route Comparison" in r.text
