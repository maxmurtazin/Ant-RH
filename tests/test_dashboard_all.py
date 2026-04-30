from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from api.server import app


REQUIRED_ROUTES = [
    "/health",
    "/debug/cors",
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
]


def test_required_routes_exist():
    client = TestClient(app)
    r = client.get("/debug/routes")
    assert r.status_code == 200
    routes = r.json()
    assert isinstance(routes, list)
    missing = [p for p in REQUIRED_ROUTES if p not in routes]
    assert not missing, f"missing routes: {missing}"


def test_required_routes_return_200():
    client = TestClient(app)
    for ep in REQUIRED_ROUTES:
        r = client.get(ep)
        assert r.status_code == 200, f"{ep} -> {r.status_code} {r.text[:200]}"


def test_cors_for_next():
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


def test_next_proxy_can_be_generated():
    # Ensure the Next.js same-origin proxy route exists in repo.
    p = Path(__file__).resolve().parents[1] / "web" / "app" / "api" / "backend" / "[...path]" / "route.ts"
    assert p.exists(), f"missing proxy route file: {p}"
    txt = p.read_text(encoding="utf-8", errors="replace")
    assert "process.env.API_BASE" in txt
    assert '"/api/backend"' not in txt  # should forward to backend base, not loop to itself


def test_aco_history_shape():
    client = TestClient(app)
    r = client.get("/metrics/aco/history")
    assert r.status_code == 200
    j = r.json()
    assert isinstance(j, dict)
    assert "points" in j
    assert "n" in j
    assert "source" in j
    points = j.get("points")
    if isinstance(points, list) and len(points) > 0 and isinstance(points[0], dict):
        p0 = points[0]
        assert "iter" in p0
        assert "best_loss" in p0
        assert "mean_loss" in p0


def test_jobs_shape():
    client = TestClient(app)

    r = client.get("/jobs/summary")
    assert r.status_code == 200
    j = r.json()
    assert isinstance(j, dict)
    assert "running_count" in j
    assert "latest_by_name" in j
    assert "total_jobs" in j

    r = client.get("/jobs/queue")
    assert r.status_code == 200
    j = r.json()
    assert isinstance(j, dict)
    assert "running" in j
    assert "queued" in j
    assert "paused" in j
    assert "done_recent" in j


def test_system_metrics_shape():
    client = TestClient(app)
    r = client.get("/system/metrics")
    assert r.status_code == 200
    j = r.json()
    assert isinstance(j, dict)
    assert ("cpu_percent" in j) or ("error" in j)
    assert ("memory_percent" in j) or ("error" in j)

