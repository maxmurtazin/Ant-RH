from fastapi.testclient import TestClient


def test_cors_preflight_status_allows_localhost3000():
    from api.server import app

    client = TestClient(app)
    r = client.options(
        "/status",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r.status_code in (200, 204)
    assert "access-control-allow-origin" in {k.lower() for k in r.headers.keys()}


def test_frontend_polled_endpoints_return_json_200():
    from api.server import app

    client = TestClient(app)
    endpoints = [
        "/status",
        "/metrics/aco",
        "/metrics/aco/history",
        "/metrics/topological-lm",
        "/metrics/physics",
        "/health/gemma",
        "/system/metrics",
        "/jobs/summary",
        "/jobs/queue",
        "/experiment/progress",
    ]
    for ep in endpoints:
        r = client.get(ep)
        assert r.status_code == 200, f"{ep} -> {r.status_code} {r.text[:200]}"
        # Should be JSON for all these endpoints
        _ = r.json()

