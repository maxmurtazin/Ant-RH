const BASE = process.env.NEXT_PUBLIC_API_BASE?.trim() || "http://127.0.0.1:8084";
export const API_BASE = BASE;

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

export type AcoHistoryResponse = {
  points: AcoHistoryPoint[];
  n: number;
  source?: string;
};

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

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${BASE}${path}`;
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 12_000);
  try {
    const res = await fetch(url, { ...init, signal: controller.signal });
    if (!res.ok) {
      const txt = await res.text().catch(() => "");
      throw new Error(`${init?.method || "GET"} ${path} failed (${res.status}): ${txt}`);
    }
    return (await res.json()) as T;
  } finally {
    window.clearTimeout(timeout);
  }
}

export async function apiGet<T>(path: string): Promise<T> {
  return await fetchJson<T>(path, { method: "GET" });
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  return await fetchJson<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {})
  });
}

export function getStatus() {
  return apiGet<StatusResponse>("/status");
}

export function getAcoMetrics() {
  return apiGet<AcoMetricsResponse>("/metrics/aco");
}

export async function getAcoHistory(limit = 300) {
  return await fetchJson<AcoHistoryResponse>(`/metrics/aco/history?limit=${limit}`);
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

