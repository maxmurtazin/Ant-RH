from __future__ import annotations

import json
import csv
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.head("/")
def dashboard_head() -> FileResponse:
    # Some clients probe with HEAD; mirror GET behavior.
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/apple-touch-icon.png", include_in_schema=False)
def apple_touch_icon() -> Response:
    return Response(status_code=204)


@app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
def apple_touch_icon_precomposed() -> Response:
    return Response(status_code=204)


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


@app.get("/health/gemma")
def health_gemma() -> JSONResponse:
    path = Path("runs/gemma_health_check.json")
    try:
        if not path.exists():
            return JSONResponse(
                status_code=200,
                content={"status": "unknown", "message": "Run make gemma-health first."},
            )
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            return JSONResponse(status_code=200, content={"status": "unknown", "message": "Invalid health JSON."})
        return JSONResponse(status_code=200, content=loaded)
    except Exception:
        return JSONResponse(status_code=200, content={"status": "unknown", "message": "Failed to read health report."})


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
    physics = _physics_metrics()

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
        physics_self_adjoint_status=physics.get("self_adjoint_status"),
        physics_spectral_status=physics.get("spectral_status"),
        physics_otoc_indicator=physics.get("otoc_indicator"),
        physics_r_mean=physics.get("r_mean"),
        physics_source=physics.get("source"),
        project_memory_head=head,
    )


def _unknown_physics(source: str = "unknown") -> dict:
    return {
        "self_adjoint_status": "unknown",
        "self_adjoint_error": None,
        "spectral_status": "unknown",
        "otoc_indicator": "unknown",
        "r_mean": None,
        "spectrum_real": None,
        "spacing_std": None,
        "source": source,
    }


def _physics_metrics() -> dict:
    # Priority:
    # 1) runs/topological_lm/eval_report.json
    # 2) runs/operator_stability_report.json
    # 3) runs/operator_sensitivity_report.json

    # 1) TopologicalLM eval report (best available for r-statistic / otoc proxy)
    eval_report = read_json_rel_runs("topological_lm/eval_report.json")
    if isinstance(eval_report, dict):
        baselines = eval_report.get("baselines", {}) if isinstance(eval_report.get("baselines"), dict) else {}
        topo = baselines.get("TopologicalLM-only", {}) if isinstance(baselines.get("TopologicalLM-only"), dict) else {}
        dedup = topo.get("dedup", {}) if isinstance(topo.get("dedup"), dict) else {}
        raw = topo.get("raw", {}) if isinstance(topo.get("raw"), dict) else {}
        src = dedup if dedup else raw
        out = _unknown_physics("runs/topological_lm/eval_report.json")
        if isinstance(src, dict):
            out["self_adjoint_status"] = str(src.get("self_adjoint_status") or "unknown")
            out["spectral_status"] = str(src.get("spectral_status") or "unknown")
            out["otoc_indicator"] = str(src.get("otoc_indicator") or "unknown")
            out["r_mean"] = src.get("r_mean", None)

            # self_adjoint_error: estimate from top candidates (mean of finite values)
            cand = src.get("top_unique_candidates") or src.get("top_10_candidates") or []
            if isinstance(cand, list):
                vals = []
                for item in cand:
                    if not isinstance(item, dict):
                        continue
                    v = item.get("self_adjoint_error", None)
                    try:
                        if v is not None:
                            fv = float(v)
                            if fv == fv and abs(fv) != float("inf"):
                                vals.append(fv)
                    except Exception:
                        pass
                if vals:
                    out["self_adjoint_error"] = float(sum(vals) / len(vals))

            # spectrum_real / spacing_std are not explicitly logged in eval_report; infer minimally.
            if out["spectral_status"] == "ok":
                out["spectrum_real"] = True
            return out

    # 2) Operator stability report
    osr = read_json_rel_runs("operator_stability_report.json")
    if isinstance(osr, dict):
        out = _unknown_physics("runs/operator_stability_report.json")
        try:
            sym = osr.get("symmetry_error_after", None)
            fro = osr.get("fro_norm_after", None)
            if sym is not None and fro is not None:
                sym_f = float(sym)
                fro_f = float(fro)
                hdiff = sym_f / (fro_f + 1e-8)
                out["self_adjoint_error"] = float(hdiff)
                if hdiff < 1e-6:
                    out["self_adjoint_status"] = "ok"
                elif hdiff < 1e-3:
                    out["self_adjoint_status"] = "approx"
                else:
                    out["self_adjoint_status"] = "broken"
                out["spectrum_real"] = True
        except Exception:
            pass
        return out

    # 3) Operator sensitivity report (no detailed spectral structure; keep unknowns)
    sens = read_json_rel_runs("operator_sensitivity_report.json")
    if isinstance(sens, dict):
        return _unknown_physics("runs/operator_sensitivity_report.json")

    return _unknown_physics("none")


@app.get("/metrics/physics")
def metrics_physics() -> JSONResponse:
    return JSONResponse(status_code=200, content=_physics_metrics())


@app.get("/metrics/aco", response_model=AcoMetricsResponse)
def metrics_aco() -> AcoMetricsResponse:
    rows = read_csv_rows_rel_runs("artin_aco_history.csv")
    m = aco_metrics_from_history(rows)
    return AcoMetricsResponse(**m)


@app.get("/metrics/aco/history")
def metrics_aco_history(limit: int = Query(300, ge=1, le=50_000)) -> JSONResponse:
    """
    Return last N rows from runs/artin_aco_history.csv.

    Supports both CSV formats:
      - iter,best_loss,mean_loss
      - iter,best_loss,mean_loss,best_reward,mean_reward,reward_mode

    Missing columns become null. Malformed rows are skipped.
    """

    source = "runs/artin_aco_history.csv"
    path = Path(source)
    if not path.exists():
        return JSONResponse(status_code=200, content={"points": [], "n": 0, "source": source})

    def _to_float(v):
        if v is None:
            return None
        try:
            s = str(v).strip()
            if s == "" or s.lower() == "nan":
                return None
            return float(s)
        except Exception:
            return None

    def _to_int(v):
        if v is None:
            return None
        try:
            s = str(v).strip()
            if s == "" or s.lower() == "nan":
                return None
            return int(float(s))
        except Exception:
            return None

    def _is_numeric(s: str) -> bool:
        try:
            ss = str(s).strip()
            if ss == "" or ss.lower() == "nan":
                return False
            float(ss)
            return True
        except Exception:
            return False

    points = []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            # Detect header vs no-header by inspecting first non-empty row.
            first_row = None
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip() == "":
                    continue
                first_row = line
                # rewind to beginning for the actual parser
                f.seek(pos)
                break

            if first_row is None:
                return JSONResponse(status_code=200, content={"points": [], "n": 0, "source": source})

            first_cells = [c.strip() for c in first_row.split(",")]
            no_header = len(first_cells) >= 3 and _is_numeric(first_cells[0]) and _is_numeric(first_cells[1]) and _is_numeric(first_cells[2])

            if no_header:
                reader = csv.reader(f)
                for row in reader:
                    if not row or len(row) < 3:
                        continue
                    it = _to_int(row[0])
                    if it is None:
                        continue
                    best_loss = _to_float(row[1])
                    mean_loss = _to_float(row[2])
                    best_reward = _to_float(row[3]) if len(row) > 3 else None
                    mean_reward = _to_float(row[4]) if len(row) > 4 else None
                    reward_mode = (str(row[5]).strip() or None) if len(row) > 5 and row[5] is not None else None
                    points.append(
                        {
                            "iter": it,
                            "best_loss": best_loss,
                            "mean_loss": mean_loss,
                            "best_reward": best_reward,
                            "mean_reward": mean_reward,
                            "reward_mode": reward_mode,
                        }
                    )
            else:
                reader = csv.DictReader(f)
                for row in reader:
                    if not isinstance(row, dict):
                        continue
                    it = _to_int(row.get("iter"))
                    if it is None:
                        continue
                    reward_mode = row.get("reward_mode", None)
                    reward_mode = (str(reward_mode).strip() or None) if reward_mode is not None else None
                    points.append(
                        {
                            "iter": it,
                            "best_loss": _to_float(row.get("best_loss")),
                            "mean_loss": _to_float(row.get("mean_loss")),
                            "best_reward": _to_float(row.get("best_reward")),
                            "mean_reward": _to_float(row.get("mean_reward")),
                            "reward_mode": reward_mode,
                        }
                    )
    except Exception:
        # Be forgiving: never crash the dashboard due to malformed CSV.
        return JSONResponse(status_code=200, content={"points": [], "n": 0, "source": source})

    if len(points) > int(limit):
        points = points[-int(limit) :]

    return JSONResponse(status_code=200, content={"points": points, "n": len(points), "source": source})


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
        self_adjoint_status=m.get("self_adjoint_status"),
        spectral_status=m.get("spectral_status"),
        otoc_indicator=m.get("otoc_indicator"),
        r_mean=m.get("r_mean"),
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
    "gemma-health": "gemma-health",
    "stability": "stability",
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

