from fastapi.testclient import TestClient


def test_required_endpoints_200():
    from api.server import app

    client = TestClient(app)
    required = [
        "/health",
        "/status",
        "/metrics/aco",
        "/metrics/aco/history",
        "/metrics/topological-lm",
        "/metrics/physics",
        "/system/metrics",
        "/jobs",
        "/jobs/summary",
        "/jobs/queue",
        "/experiment/progress",
        "/metrics/spectral/spacing",
        "/operator/analysis",
        "/docs/playbook",
        "/exports",
        "/runs/compare",
        "/debug/routes",
    ]

    for ep in required:
        r = client.get(ep)
        assert r.status_code == 200, f"{ep} -> {r.status_code} {r.text[:200]}"


def test_debug_routes_contains_jobs_paths():
    from api.server import app

    client = TestClient(app)
    r = client.get("/debug/routes")
    assert r.status_code == 200
    paths = r.json()
    assert "/jobs/summary" in paths
    assert "/jobs/queue" in paths

