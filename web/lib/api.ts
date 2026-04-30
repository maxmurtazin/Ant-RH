export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8084";

export type StatusResponse = {
  ok: boolean;
  missing: string[];
  gemma_main_issue?: string | null;
  gemma_learning?: boolean | null;
  aco_best_loss_last?: number | null;
  aco_best_loss_trend?: string | null;
  aco_mean_loss_last?: number | null;
  aco_mean_loss_trend?: string | null;
  operator_spectral_loss?: number | null;
  operator_spacing_loss?: number | null;
  operator_total_loss?: number | null;
  operator_eigh_success?: boolean | null;
  topo_advantage_over_random?: number | null;
  topo_random_mean_reward?: number | null;
  topo_topological_lm_mean_reward?: number | null;
  topo_unique_candidate_ratio?: number | null;
  topo_valid_braid_ratio?: number | null;
  physics_self_adjoint_status?: string | null;
  physics_spectral_status?: string | null;
  physics_otoc_indicator?: string | null;
  physics_r_mean?: number | null;
  physics_source?: string | null;
  project_memory_head?: string | null;
};

export type AcoMetricsResponse = {
  best_loss: number | null;
  mean_loss: number | null;
  trend: "increasing" | "decreasing" | "flat";
  n_rows: number;
};

export type AcoHistoryPoint = {
  iter: number;
  best_loss: number | null;
  mean_loss: number | null;
  best_reward: number | null;
  mean_reward: number | null;
  reward_mode: string | null;
};

export type AcoHistoryResponse = { points: AcoHistoryPoint[]; n: number; source?: string };

export type TopoMetricsResponse = {
  random_mean_reward: number | null;
  topological_lm_mean_reward: number | null;
  advantage_over_random: number | null;
  unique_candidate_ratio: number | null;
  valid_braid_ratio: number | null;
  self_adjoint_status?: string | null;
  spectral_status?: string | null;
  otoc_indicator?: string | null;
  r_mean?: number | null;
};

export type PhysicsMetricsResponse = {
  self_adjoint_status: string;
  self_adjoint_error: number | null;
  spectral_status: string;
  otoc_indicator: string;
  r_mean: number | null;
  spectrum_real: boolean | null;
  spacing_std: number | null;
  source: string;
};

export type GemmaHealthResponse =
  | {
      overall_status: "ok" | "degraded" | "failed";
      checks: { name: string; status: string; latency_s: number; message: string }[];
    }
  | { status: "unknown"; message: string };

export type RunStageRequest = {
  stage:
    | "study"
    | "analyze"
    | "journal"
    | "docs"
    | "topo-eval"
    | "topo-report"
    | "gemma-health";
};

export type RunResultResponse = {
  ok: boolean;
  target: string;
  returncode: number;
  duration_s: number;
  timed_out: boolean;
  stdout: string;
  stderr: string;
};

export type GemmaHelpRequest = { question: string; voice: boolean };
export type GemmaHelpResponse = { answer: string };

export async function fetchJson(path: string, options?: RequestInit) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, { cache: "no-store", ...(options || {}) });
  if (!res.ok) {
    let body: any = null;
    try {
      body = await res.json();
    } catch {
      try {
        body = await res.text();
      } catch {
        body = null;
      }
    }
    const err: any = new Error(`${res.status} ${res.statusText}: ${url}`);
    err.status = res.status;
    err.body = body;
    throw err;
  }
  return await res.json();
}

export async function apiGet<T>(path: string): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, { method: "GET", cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${url}`);
  return (await res.json()) as T;
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  return (await fetchJson(path, {
    method: "POST",
    body: JSON.stringify(body ?? {}),
    headers: { "Content-Type": "application/json" },
  })) as T;
}

export function getStatus() {
  return apiGet<StatusResponse>("/status");
}

export function getAcoMetrics() {
  return apiGet<AcoMetricsResponse>("/metrics/aco");
}

export async function getAcoHistory(limit = 300) {
  const data = (await fetchJson(`/metrics/aco/history?limit=${limit}`)) as unknown;
  // Temporary debug
  // eslint-disable-next-line no-console
  console.log("ACO history response", data);
  if (Array.isArray(data)) return data as any[];
  if (data && typeof data === "object" && Array.isArray((data as any).points)) return (data as any).points as any[];
  return [];
}

export function getTopoMetrics() {
  return apiGet<TopoMetricsResponse>("/metrics/topological-lm");
}

export function getPhysicsMetrics() {
  return apiGet<PhysicsMetricsResponse>("/metrics/physics");
}

export function getGemmaHealth() {
  return apiGet<GemmaHealthResponse>("/health/gemma");
}

export type JobStatus = "queued" | "running" | "paused" | "done" | "failed" | "cancelled";

export type Job = {
  id: string;
  name: string;
  command: string[];
  status: JobStatus;
  started_at: string;
  ended_at: string | null;
  log_path: string;
  returncode: number | null;
  pid?: number;
};

export async function startJob(job: string, params?: Record<string, any>) {
  return (await fetchJson("/jobs/start", {
    method: "POST",
    body: JSON.stringify({ job, params: params || {} }),
    headers: { "Content-Type": "application/json" },
  })) as { job_id: string };
}

export async function getJobs() {
  return await apiGet<{ jobs: Job[] }>("/jobs");
}

export async function getJobsSummary() {
  return await apiGet<{
    running_count: number;
    latest_by_name: Record<string, any>;
    max_concurrent_jobs?: number;
    low_resource_mode?: boolean;
  }>("/jobs/summary");
}

export async function stopJob(jobId: string) {
  return await apiPost<{ status: string }>(`/jobs/${encodeURIComponent(jobId)}/stop`, {});
}

export async function resumeJob(jobId: string) {
  return await apiPost<{ status: string }>(`/jobs/${encodeURIComponent(jobId)}/resume`, {});
}

export async function pauseJob(jobId: string) {
  return await apiPost<{ status: string }>(`/jobs/${encodeURIComponent(jobId)}/pause`, {});
}

export type SystemMetrics = {
  cpu_percent: number;
  memory_percent: number;
  memory_used_gb: number;
  memory_total_gb: number;
  process_cpu_percent: number;
  process_memory_mb: number;
  gpu: { available: boolean; type: "mps" | "cuda" | "none"; usage_percent?: number | null; note?: string };
  timestamp: string;
};

export async function getSystemMetrics() {
  return await apiGet<SystemMetrics>("/system/metrics");
}

export type JobQueueResponse = {
  running: any[];
  queued: any[];
  done_recent: any[];
};

export async function getJobQueue() {
  return await apiGet<JobQueueResponse>("/jobs/queue");
}

export async function cancelJob(jobId: string) {
  return await apiPost<{ status: string }>(`/jobs/${encodeURIComponent(jobId)}/cancel`, {});
}

export async function moveJobUp(jobId: string) {
  return await apiPost<{ status: string }>(`/jobs/${encodeURIComponent(jobId)}/move-up`, {});
}

export async function moveJobDown(jobId: string) {
  return await apiPost<{ status: string }>(`/jobs/${encodeURIComponent(jobId)}/move-down`, {});
}

export type RunCompareRow = {
  id: string;
  label: string;
  timestamp: string;
  aco_best_loss: number | null;
  aco_mean_loss: number | null;
  topo_reward_mean: number | null;
  topo_advantage_over_random: number | null;
  self_adjoint_error: number | null;
  r_mean: number | null;
  operator_sensitivity: number | null;
  source: string;
};

export async function getRunComparison() {
  return await apiGet<{ runs: RunCompareRow[] }>("/runs/compare");
}

export async function saveScreenshot(name: string, image_base64: string) {
  return await apiPost<{ status: string; path: string }>(`/screenshots/save`, { name, image_base64 });
}

export async function getJob(jobId: string) {
  return await apiGet<Job>(`/jobs/${encodeURIComponent(jobId)}`);
}

export async function getJobLog(jobId: string, lines = 200) {
  const q = `?lines=${encodeURIComponent(String(lines))}`;
  return await apiGet<{ job_id: string; lines: string[] }>(`/jobs/${encodeURIComponent(jobId)}/log${q}`);
}

export type OperatorAnalysisResponse = {
  formula_tex: string | null;
  formula_text: string | null;
  pde_report_excerpt: string | null;
  active_terms: { term: string; coefficient: number; abs_coefficient: number }[];
  stability: { self_adjoint_error: number | null; eigh_success: boolean | null; spectral_loss: number | null };
  sensitivity: {
    operator_distance_mean: number | null;
    spectrum_distance_mean: number | null;
    loss_std: number | null;
    diagnosis: string | null;
  };
  structured_operator: {
    final_loss: number | null;
    spectral_loss: number | null;
    spacing_loss: number | null;
    top_weights: any[];
  };
  source_files: string[];
};

export async function getOperatorAnalysis() {
  return await apiGet<OperatorAnalysisResponse>("/operator/analysis");
}

export type TopoHistoryPoint = {
  iter: number;
  mean_reward: number | null;
  unique_candidate_ratio: number | null;
};

export type TopoHistoryResponse = {
  points: TopoHistoryPoint[];
  reward_samples: number[];
  n: number;
  source: string | null;
};

export async function getTopoHistory(limit = 300) {
  return await apiGet<TopoHistoryResponse>(`/metrics/topological-lm/history?limit=${encodeURIComponent(String(limit))}`);
}

export type PhysicsHistoryPoint = {
  iter: number;
  r_mean: number | null;
  self_adjoint_error: number | null;
};

export type PhysicsHistoryResponse = {
  points: PhysicsHistoryPoint[];
  n: number;
  source: string | null;
};

export async function getPhysicsHistory(limit = 300) {
  return await apiGet<PhysicsHistoryResponse>(`/metrics/physics/history?limit=${encodeURIComponent(String(limit))}`);
}

export type SpectralSpacingResponse = {
  hist_bins: number[];
  operator_spacing_hist: number[];
  zeta_spacing_hist: number[];
  gue_curve: number[];
  poisson_curve: number[];
  operator_r_mean: number | null;
  zeta_r_mean: number | null;
  source: string | null;
};

export async function getSpectralSpacing() {
  return await apiGet<SpectralSpacingResponse>("/metrics/spectral/spacing");
}

export function exportData() {
  window.open(`${API_BASE}/export`, "_blank");
}

export async function importData(file: File) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/import`, {
    method: "POST",
    body: form,
  });
  return await res.json();
}

export type ExportEntry = {
  id: string;
  timestamp: string;
  name: string;
  reason: string;
  path: string;
  size_bytes: number;
  files_count: number;
  metrics_summary?: { aco_best_loss?: number | null; topo_reward_mean?: number | null; spectral_loss?: number | null };
};

export async function getExports() {
  return (await apiGet<{ exports: ExportEntry[] }>("/exports")).exports || [];
}

export async function createExport(name: string, reason: string) {
  return await apiPost<ExportEntry>("/exports/create", { name, reason });
}

export function downloadExport(id: string) {
  window.open(`${API_BASE}/exports/${encodeURIComponent(id)}/download`, "_blank");
}

export async function deleteExport(id: string) {
  const url = `${API_BASE}/exports/${encodeURIComponent(id)}`;
  const res = await fetch(url, { method: "DELETE" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${url}`);
  return await res.json();
}

export type ExperimentProgressResponse = {
  steps: { id: string; title: string; status: "done" | "running" | "pending" | "failed"; eta_seconds: number; evidence: string; command: string }[];
  overall_progress: number;
  estimated_remaining_seconds: number;
};

export async function getExperimentProgress() {
  return await apiGet<ExperimentProgressResponse>("/experiment/progress");
}

export type NextStepRecommendation = {
  source: "gemma" | "rules";
  next_step: string;
  reason: string;
  command: string;
  dashboard_action: string;
  priority: "low" | "medium" | "high";
};

export async function getNextRecommendation(opts?: { use_gemma?: boolean }) {
  const useGemma = Boolean(opts?.use_gemma);
  const qs = useGemma ? "?use_gemma=true" : "";
  return await apiGet<NextStepRecommendation>(`/recommend/next-step${qs}`);
}

export async function startFullPipeline(options: { continue_on_failure: boolean; include_ppo: boolean }) {
  return await apiPost<{ job_id: string }>("/jobs/start-full-pipeline", options);
}

