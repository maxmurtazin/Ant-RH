from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class StatusResponse(BaseModel):
    ok: bool = True
    missing: List[str] = Field(default_factory=list)

    gemma_main_issue: Optional[str] = None
    gemma_learning: Optional[bool] = None
    aco_best_loss_last: Optional[float] = None
    aco_best_loss_trend: Optional[str] = None
    aco_mean_loss_last: Optional[float] = None
    aco_mean_loss_trend: Optional[str] = None

    operator_spectral_loss: Optional[float] = None
    operator_spacing_loss: Optional[float] = None
    operator_total_loss: Optional[float] = None
    operator_eigh_success: Optional[bool] = None

    topo_advantage_over_random: Optional[float] = None
    topo_random_mean_reward: Optional[float] = None
    topo_topological_lm_mean_reward: Optional[float] = None
    topo_unique_candidate_ratio: Optional[float] = None
    topo_valid_braid_ratio: Optional[float] = None
    physics_self_adjoint_status: Optional[str] = None
    physics_spectral_status: Optional[str] = None
    physics_otoc_indicator: Optional[str] = None
    physics_r_mean: Optional[float] = None
    physics_source: Optional[str] = None

    project_memory_head: Optional[str] = None


class AcoMetricsResponse(BaseModel):
    best_loss: Optional[float] = None
    mean_loss: Optional[float] = None
    trend: Literal["increasing", "decreasing", "flat"] = "flat"
    n_rows: int = 0


class TopologicalLmMetricsResponse(BaseModel):
    random_mean_reward: Optional[float] = None
    topological_lm_mean_reward: Optional[float] = None
    advantage_over_random: Optional[float] = None
    unique_candidate_ratio: Optional[float] = None
    valid_braid_ratio: Optional[float] = None
    self_adjoint_status: Optional[str] = None
    spectral_status: Optional[str] = None
    otoc_indicator: Optional[str] = None
    r_mean: Optional[float] = None


class GemmaHelpRequest(BaseModel):
    question: str = Field(min_length=1)
    voice: bool = False


class GemmaHelpResponse(BaseModel):
    answer: str


AllowedStage = Literal["study", "analyze", "journal", "docs", "topo-eval", "topo-report", "gemma-health", "stability"]


class RunStageRequest(BaseModel):
    stage: AllowedStage


class RunResultResponse(BaseModel):
    ok: bool
    target: str
    returncode: int
    duration_s: float
    timed_out: bool = False
    stdout: str = ""
    stderr: str = ""


class ErrorResponse(BaseModel):
    detail: str
    extra: Optional[Dict[str, Any]] = None

