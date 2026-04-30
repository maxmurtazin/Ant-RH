from __future__ import annotations

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from api.schemas import (
    AcoMetricsResponse,
    ErrorResponse,
    GemmaHelpRequest,
    GemmaHelpResponse,
    HealthResponse,
    RunResultResponse,
    RunStageRequest,
    StatusResponse,
    TopologicalLmMetricsResponse,
)
from api.utils import (
    ApiError,
    aco_metrics_from_history,
    is_local_client,
    read_csv_rows_rel_runs,
    read_json_rel_runs,
    read_text_rel_runs,
    run_gemma_help,
    run_make_target,
    topological_lm_metrics_from_eval_report,
)


app = FastAPI(title="Ant-RH Control API", version="0.1.0")


@app.exception_handler(ApiError)
def _api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(status_code=400, content=ErrorResponse(detail=str(exc)).model_dump())


@app.middleware("http")
async def _local_only(request: Request, call_next):
    client_host = getattr(getattr(request, "client", None), "host", None)
    if not is_local_client(client_host):
        return JSONResponse(status_code=403, content=ErrorResponse(detail="Localhost only.").model_dump())
    return await call_next(request)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    missing = []

    mem = read_text_rel_runs("gemma_project_memory.md", max_chars=20_000)
    if mem is None:
        missing.append("runs/gemma_project_memory.md")

    ga = read_json_rel_runs("gemma_analysis.json")
    if ga is None:
        missing.append("runs/gemma_analysis.json")

    osr = read_json_rel_runs("operator_stability_report.json")
    if osr is None:
        missing.append("runs/operator_stability_report.json")

    topo_md = read_text_rel_runs("topological_lm/report.md", max_chars=60_000)
    if topo_md is None:
        missing.append("runs/topological_lm/report.md")

    topo_eval = read_json_rel_runs("topological_lm/eval_report.json")
    if topo_eval is None:
        missing.append("runs/topological_lm/eval_report.json")

    # Compact extraction (avoid dumping entire files)
    aco = (ga or {}).get("aco", {}) if isinstance((ga or {}).get("aco"), dict) else {}
    aco_best_stats = aco.get("best_loss_stats", {}) if isinstance(aco.get("best_loss_stats"), dict) else {}
    aco_mean_stats = aco.get("mean_loss_stats", {}) if isinstance(aco.get("mean_loss_stats"), dict) else {}
    operator = (ga or {}).get("operator", {}) if isinstance((ga or {}).get("operator"), dict) else {}

    topo_metrics = topological_lm_metrics_from_eval_report(topo_eval or {})

    head = None
    if mem:
        head = "\n".join(mem.strip().splitlines()[:12]).strip()

    return StatusResponse(
        ok=True,
        missing=missing,
        gemma_main_issue=(ga or {}).get("main_issue") if isinstance(ga, dict) else None,
        gemma_learning=(ga or {}).get("learning") if isinstance(ga, dict) else None,
        aco_best_loss_last=(aco_best_stats.get("last") if isinstance(aco_best_stats, dict) else None),
        aco_best_loss_trend=(aco.get("best_loss_trend") if isinstance(aco, dict) else None),
        aco_mean_loss_last=(aco_mean_stats.get("last") if isinstance(aco_mean_stats, dict) else None),
        aco_mean_loss_trend=(aco.get("mean_loss_trend") if isinstance(aco, dict) else None),
        operator_spectral_loss=operator.get("spectral_loss") if isinstance(operator, dict) else None,
        operator_spacing_loss=operator.get("spacing_loss") if isinstance(operator, dict) else None,
        operator_total_loss=(osr or {}).get("total_loss") if isinstance(osr, dict) else None,
        operator_eigh_success=bool(operator.get("eigh_success")) if isinstance(operator, dict) and "eigh_success" in operator else None,
        topo_advantage_over_random=topo_metrics.get("advantage_over_random"),
        topo_random_mean_reward=topo_metrics.get("random_mean_reward"),
        topo_topological_lm_mean_reward=topo_metrics.get("topological_lm_mean_reward"),
        topo_unique_candidate_ratio=topo_metrics.get("unique_candidate_ratio"),
        topo_valid_braid_ratio=topo_metrics.get("valid_braid_ratio"),
        project_memory_head=head,
    )


@app.get("/metrics/aco", response_model=AcoMetricsResponse)
def metrics_aco() -> AcoMetricsResponse:
    rows = read_csv_rows_rel_runs("artin_aco_history.csv")
    m = aco_metrics_from_history(rows)
    return AcoMetricsResponse(**m)


@app.get("/metrics/topological-lm", response_model=TopologicalLmMetricsResponse)
def metrics_topological_lm() -> TopologicalLmMetricsResponse:
    report = read_json_rel_runs("topological_lm/eval_report.json") or {}
    m = topological_lm_metrics_from_eval_report(report)
    return TopologicalLmMetricsResponse(
        random_mean_reward=m.get("random_mean_reward"),
        topological_lm_mean_reward=m.get("topological_lm_mean_reward"),
        advantage_over_random=m.get("advantage_over_random"),
        unique_candidate_ratio=m.get("unique_candidate_ratio"),
        valid_braid_ratio=m.get("valid_braid_ratio"),
    )


@app.post("/gemma/help", response_model=GemmaHelpResponse)
def gemma_help(req: GemmaHelpRequest = Body(...)) -> GemmaHelpResponse:
    try:
        answer = run_gemma_help(req.question, voice=req.voice, timeout_s=300)
    except ApiError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return GemmaHelpResponse(answer=answer)


STAGE_TO_MAKE_TARGET = {
    "study": "study",
    "analyze": "analyze-gemma",
    "journal": "lab-journal",
    "docs": "docs",
    "topo-eval": "topo-eval",
    "topo-report": "topo-report",
}


@app.post("/run/stage", response_model=RunResultResponse)
def run_stage(req: RunStageRequest = Body(...)) -> RunResultResponse:
    target = STAGE_TO_MAKE_TARGET.get(req.stage)
    if not target:
        raise HTTPException(status_code=400, detail="Invalid stage.")
    res = run_make_target(target, timeout_s=300)
    return RunResultResponse(
        ok=(res.returncode == 0 and not res.timed_out),
        target=f"make {target}",
        returncode=res.returncode,
        duration_s=res.duration_s,
        timed_out=res.timed_out,
        stdout=res.stdout,
        stderr=res.stderr,
    )


@app.post("/docs/update", response_model=RunResultResponse)
def docs_update() -> RunResultResponse:
    res = run_make_target("docs", timeout_s=300)
    return RunResultResponse(
        ok=(res.returncode == 0 and not res.timed_out),
        target="make docs",
        returncode=res.returncode,
        duration_s=res.duration_s,
        timed_out=res.timed_out,
        stdout=res.stdout,
        stderr=res.stderr,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)

