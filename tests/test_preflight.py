from scripts import preflight_dashboard as pf


def test_required_endpoints_include_jobs_and_routes():
    assert "/jobs/summary" in pf.REQUIRED_ENDPOINTS
    assert "/jobs/queue" in pf.REQUIRED_ENDPOINTS
    assert "/debug/routes" in pf.REQUIRED_ENDPOINTS

