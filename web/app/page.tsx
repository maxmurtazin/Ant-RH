"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AcoMetrics } from "@/components/AcoMetrics";
import AcoLiveCharts from "@/components/AcoLiveCharts";
import { AcoCharts } from "@/components/AcoCharts";
import { GemmaHealth } from "@/components/GemmaHealth";
import { CheckpointPanel } from "@/components/CheckpointPanel";
import { ImportExportPanel } from "@/components/ImportExportPanel";
import { JobControlPanel } from "@/components/JobControlPanel";
import { LogStream } from "@/components/LogStream";
import { OperatorAnalysis } from "@/components/OperatorAnalysis";
import { OperatorCharts } from "@/components/OperatorCharts";
import { PhysicsCharts } from "@/components/PhysicsCharts";
import { PhysicsDiagnostics } from "@/components/PhysicsDiagnostics";
import { QuickGuide } from "@/components/QuickGuide";
import { ProgressTracker } from "@/components/ProgressTracker";
import { RewardHistogram } from "@/components/RewardHistogram";
import { SpectralSpacingChart } from "@/components/SpectralSpacingChart";
import { TopologicalLmMetrics } from "@/components/TopologicalLmMetrics";
import { StatusBadge } from "@/components/StatusBadge";
import { LowResourceToggle } from "@/components/LowResourceToggle";
import { SystemMonitor } from "@/components/SystemMonitor";
import { JobQueuePanel } from "@/components/JobQueuePanel";
import { MultiRunComparison } from "@/components/MultiRunComparison";
import { ScreenshotButton } from "@/components/ScreenshotButton";
import { JobButton } from "@/components/JobButton";
import type {
  AcoMetricsResponse,
  GemmaHealthResponse,
  OperatorAnalysisResponse,
  PhysicsHistoryPoint,
  PhysicsMetricsResponse,
  SpectralSpacingResponse,
  StatusResponse,
  TopoHistoryPoint,
  TopoMetricsResponse
} from "@/lib/api";
import {
  API_BASE,
  DISPLAY_API_BASE,
  getJobQueue,
  getSystemMetrics,
  getRunComparison,
  createExport,
  exportData,
  getAcoHistory,
  getAcoMetrics,
  getGemmaHealth,
  getExperimentProgress,
  getJobsSummary,
  getJobs,
  getOperatorAnalysis,
  getPhysicsHistory,
  getPhysicsMetrics,
  getSpectralSpacing,
  getStatus,
  getTopoHistory,
  getTopoMetrics,
  startFullPipeline,
  startJob as startJobApi
} from "@/lib/api";

type EndpointKey = "status" | "aco" | "aco-history" | "topological-lm" | "physics" | "gemma-health";

function fmtTime(d: Date) {
  return d.toLocaleTimeString(undefined, { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export default function Page() {
  const [lowResourceMode, setLowResourceMode] = useState(false);
  const [logsEnabled, setLogsEnabled] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<string>("—");
  const [endpointErrors, setEndpointErrors] = useState<Partial<Record<EndpointKey, string>>>({});
  const [errors, setErrors] = useState<{ acoHistory?: string }>({});

  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [aco, setAco] = useState<AcoMetricsResponse | null>(null);
  const [topo, setTopo] = useState<TopoMetricsResponse | null>(null);
  const [physics, setPhysics] = useState<PhysicsMetricsResponse | null>(null);
  const [gemmaHealth, setGemmaHealth] = useState<GemmaHealthResponse | null>(null);
  const [acoHistory, setAcoHistory] = useState<any[]>([]);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [operatorAnalysis, setOperatorAnalysis] = useState<OperatorAnalysisResponse | null>(null);
  const [operatorError, setOperatorError] = useState<string>("");
  const [topoHistory, setTopoHistory] = useState<TopoHistoryPoint[]>([]);
  const [topoRewardSamples, setTopoRewardSamples] = useState<number[]>([]);
  const [physicsHistory, setPhysicsHistory] = useState<PhysicsHistoryPoint[]>([]);
  const [spectralSpacing, setSpectralSpacing] = useState<SpectralSpacingResponse | null>(null);
  const [progress, setProgress] = useState<any | null>(null);
  const [jobs, setJobs] = useState<any[]>([]);
  const [jobsSummary, setJobsSummary] = useState<any | null>(null);
  const [systemMetrics, setSystemMetrics] = useState<any | null>(null);
  const [jobQueue, setJobQueue] = useState<any | null>(null);
  const [runCompare, setRunCompare] = useState<any | null>(null);

  const refreshInFlightRef = useRef(false);

  useEffect(() => {
    try {
      const v = window.localStorage.getItem("ant_low_resource_mode");
      const on = String(v || "").trim().toLowerCase() === "true";
      setLowResourceMode(on);
      setLogsEnabled(!on);
    } catch {
      // ignore
    }
  }, []);

  const pollInterval = lowResourceMode ? 20_000 : 5_000;

  const summary = useMemo(() => {
    if (!status) return "—";
    const parts = [
      `gemma_learning=${String(status.gemma_learning ?? "—")}`,
      `main_issue=${String(status.gemma_main_issue ?? "—")}`,
      `operator_total_loss=${String(status.operator_total_loss ?? "—")}`,
      `topo_advantage=${String(status.topo_advantage_over_random ?? "—")}`
    ];
    return parts.join("\n");
  }, [status]);

  async function startJob(job: string, params: any = {}) {
    if (job === "checkpoint") {
      await createExport("manual_checkpoint", "manual");
      return;
    }
    if (job === "export") {
      exportData();
      return;
    }
    try {
      const result = await startJobApi(job, params);
      setActiveJobId(result.job_id);
    } catch (e: any) {
      if (e?.status === 429 || e?.status === 409) {
        // Surface a clearer message in the existing error box area.
        setEndpointErrors((prev) => ({ ...prev, status: "Job already running. Wait for it to finish." }));
      }
      throw e;
    }
  }

  async function startPipeline(opts: { continue_on_failure: boolean; include_ppo: boolean }) {
    const res = await startFullPipeline(opts);
    setActiveJobId(res.job_id);
  }

  const refreshAll = useCallback(async () => {
    if (refreshInFlightRef.current) return;
    refreshInFlightRef.current = true;
    setIsRefreshing(true);

    const nextErrors: Partial<Record<EndpointKey, string>> = {};
    let anySuccess = false;

    try {
      const results = await Promise.allSettled([
        getStatus(),
        getAcoMetrics(),
        getTopoMetrics(),
        getPhysicsMetrics(),
        getGemmaHealth()
      ]);

      const [st, a, t, p, gh] = results;

      if (st.status === "fulfilled") {
        anySuccess = true;
        setStatus(st.value);
      } else {
        nextErrors["status"] = st.reason instanceof Error ? st.reason.message : String(st.reason);
      }

      if (a.status === "fulfilled") {
        anySuccess = true;
        setAco(a.value);
      } else {
        nextErrors["aco"] = a.reason instanceof Error ? a.reason.message : String(a.reason);
      }

      try {
        const history = await getAcoHistory(300);
        setAcoHistory(Array.isArray(history) ? history : []);
        setErrors((prev) => ({ ...prev, acoHistory: undefined }));
      } catch (e) {
        setErrors((prev) => ({ ...prev, acoHistory: String(e) }));
        nextErrors["aco-history"] = e instanceof Error ? e.message : String(e);
      }

      // TopologicalLM history (best-effort; keep last good data on failure)
      try {
        const th = await getTopoHistory(300);
        setTopoHistory(Array.isArray(th.points) ? th.points : []);
        setTopoRewardSamples(Array.isArray(th.reward_samples) ? th.reward_samples : []);
      } catch (e) {
        nextErrors["topological-lm"] = nextErrors["topological-lm"] || (e instanceof Error ? e.message : String(e));
      }

      // Physics history (best-effort; keep last good data on failure)
      try {
        const ph = await getPhysicsHistory(300);
        setPhysicsHistory(Array.isArray(ph.points) ? ph.points : []);
      } catch (e) {
        nextErrors["physics"] = nextErrors["physics"] || (e instanceof Error ? e.message : String(e));
      }

      if (t.status === "fulfilled") {
        anySuccess = true;
        setTopo(t.value);
      } else {
        nextErrors["topological-lm"] = t.reason instanceof Error ? t.reason.message : String(t.reason);
      }

      if (p.status === "fulfilled") {
        anySuccess = true;
        setPhysics(p.value);
      } else {
        nextErrors["physics"] = p.reason instanceof Error ? p.reason.message : String(p.reason);
      }

      if (gh.status === "fulfilled") {
        anySuccess = true;
        setGemmaHealth(gh.value);
      } else {
        nextErrors["gemma-health"] = gh.reason instanceof Error ? gh.reason.message : String(gh.reason);
      }

      setEndpointErrors(nextErrors);
      if (anySuccess) setLastUpdated(fmtTime(new Date()));
    } finally {
      setIsRefreshing(false);
      refreshInFlightRef.current = false;
    }
  }, []);

  useEffect(() => {
    refreshAll();
    const id = window.setInterval(refreshAll, pollInterval);
    return () => window.clearInterval(id);
  }, [refreshAll, pollInterval]);

  useEffect(() => {
    let cancelled = false;
    async function refreshSummary() {
      try {
        const s = await getJobsSummary();
        if (!cancelled) setJobsSummary(s);
      } catch {
        // ignore
      }
    }
    refreshSummary();
    const id = window.setInterval(refreshSummary, pollInterval);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [pollInterval]);

  useEffect(() => {
    let cancelled = false;
    async function refreshSystem() {
      try {
        const d = await getSystemMetrics();
        if (!cancelled) setSystemMetrics(d);
      } catch (e: any) {
        if (!cancelled) setSystemMetrics({ error: "system_metrics_error", message: e?.message || String(e) });
      }
    }
    refreshSystem();
    const id = window.setInterval(refreshSystem, pollInterval);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [pollInterval]);

  useEffect(() => {
    let cancelled = false;
    async function refreshQueue() {
      try {
        const q = await getJobQueue();
        if (!cancelled) setJobQueue(q);
      } catch {
        // ignore
      }
    }
    refreshQueue();
    const id = window.setInterval(refreshQueue, lowResourceMode ? 5_000 : 3_000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [lowResourceMode]);

  useEffect(() => {
    let cancelled = false;
    async function refreshCompare() {
      try {
        const d = await getRunComparison();
        if (!cancelled) setRunCompare(d);
      } catch {
        // keep last good
      }
    }
    refreshCompare();
    const id = window.setInterval(refreshCompare, 20_000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function refreshJobs() {
      try {
        const res = await getJobs();
        if (!cancelled) setJobs(Array.isArray(res.jobs) ? res.jobs : []);
      } catch {
        // ignore
      }
    }
    refreshJobs();
    const id = window.setInterval(refreshJobs, pollInterval);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [pollInterval]);

  const runningJobs = useMemo(() => jobs.filter((j) => j && j.status === "running"), [jobs]);
  const anyRunning = runningJobs.length > 0;

  useEffect(() => {
    let cancelled = false;
    async function refreshProgress() {
      try {
        const d = await getExperimentProgress();
        if (!cancelled) setProgress(d);
      } catch {
        // keep last good
      }
    }
    refreshProgress();
    const id = window.setInterval(refreshProgress, pollInterval);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [pollInterval]);

  useEffect(() => {
    let cancelled = false;
    async function refreshSpectral() {
      try {
        const d = await getSpectralSpacing();
        if (!cancelled) setSpectralSpacing(d);
      } catch {
        // keep last good
      }
    }
    refreshSpectral();
    const id = window.setInterval(refreshSpectral, pollInterval);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [pollInterval]);

  useEffect(() => {
    let cancelled = false;
    async function refreshOperator() {
      try {
        const d = await getOperatorAnalysis();
        if (!cancelled) {
          setOperatorAnalysis(d);
          setOperatorError("");
        }
      } catch (e) {
        if (!cancelled) setOperatorError(e instanceof Error ? e.message : String(e));
      }
    }
    refreshOperator();
    const id = window.setInterval(refreshOperator, pollInterval);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [pollInterval]);

  const endpointBadges = useMemo(() => {
    const mk = (key: EndpointKey, hasData: boolean) => {
      const err = endpointErrors[key];
      if (err) return <StatusBadge key={key} tone="broken" label={`${key} error`} />;
      if (hasData) return <StatusBadge key={key} tone="ok" label={`${key} ok`} />;
      return <StatusBadge key={key} tone="unknown" label={`${key} —`} />;
    };
    return [
      mk("status", Boolean(status)),
      mk("aco", Boolean(aco)),
      mk("aco-history", Boolean(acoHistory.length)),
      mk("topological-lm", Boolean(topo)),
      mk("physics", Boolean(physics)),
      mk("gemma-health", Boolean(gemmaHealth))
    ];
  }, [endpointErrors, status, aco, acoHistory.length, topo, physics, gemmaHealth]);

  return (
    <div className="dashboardPage" id="dashboard-root">
      <header className="dashboard-header">
        <div>
          <h1 className="dashboardTitle">Ant-RH Control Dashboard</h1>
          <div className="header-meta mono muted">
            <span className="badge mono">Live: ON</span>
            <span className="badge mono">Backend target: {DISPLAY_API_BASE}</span>
            <span className="badge mono">Browser API route: {API_BASE}</span>
            <span className="badge mono">Last updated: {lastUpdated}</span>
          </div>
        </div>
        <div className="header-actions">
          {isRefreshing ? <span className="spinner" aria-label="Refreshing" /> : null}
          <LowResourceToggle
            enabled={lowResourceMode}
            onChange={(on) => {
              setLowResourceMode(on);
              setLogsEnabled(!on);
              try {
                window.localStorage.setItem("ant_low_resource_mode", on ? "true" : "false");
              } catch {
                // ignore
              }
            }}
          />
          <button className="btn btnPrimary" onClick={() => refreshAll()} disabled={isRefreshing}>
            Refresh
          </button>
        </div>
      </header>

      <div className="dashboard-grid">
        <section className="results-column resultsCol">
          <section className="results-section">
            <h2>Results &amp; Diagnostics</h2>
            <div className="results-grid">
              <section className="card result-card span-full">
                <div className="cardHeader">
                  <div className="cardTitle">Project Status</div>
                  <div className="cardHeaderRight">{endpointBadges}</div>
                </div>
                <div className="cardBody">
                  <div className="mono muted" style={{ marginTop: 8 }}>
                    ACO points: {acoHistory.length}
                    <br />
                    ACO error: {errors.acoHistory || "none"}
                  </div>
                  {Object.entries(endpointErrors).length ? (
                    <div className="errorBox mono">
                      {Object.entries(endpointErrors).map(([k, v]) => (
                        <div key={k}>
                          <strong>{k}</strong>: {v}
                        </div>
                      ))}
                    </div>
                  ) : null}
                  <div className="twoCol">
                    <div>
                      <div className="label">Current state</div>
                      <pre className="mono block">{summary}</pre>
                    </div>
                    <div>
                      <div className="label">Missing artifacts</div>
                      <pre className="mono block">{status?.missing?.length ? status.missing.join("\n") : "none"}</pre>
                    </div>
                  </div>
                  <div className="divider" />
                  <div className="mono muted">Auto-refresh every {lowResourceMode ? "20s" : "5s"}</div>
                </div>
              </section>

              <AcoMetrics data={aco} />
              <AcoLiveCharts points={acoHistory} />
              <TopologicalLmMetrics data={topo} />
              <PhysicsDiagnostics data={physics} />
              <AcoCharts points={acoHistory} />
              <OperatorCharts
                operatorDistanceMean={operatorAnalysis?.sensitivity?.operator_distance_mean ?? null}
                lossStd={operatorAnalysis?.sensitivity?.loss_std ?? null}
              />
              <SpectralSpacingChart data={spectralSpacing} />
              <RewardHistogram samples={topoRewardSamples} />
              <PhysicsCharts points={physicsHistory} />
              <OperatorAnalysis data={operatorAnalysis} />
              <MultiRunComparison runs={Array.isArray(runCompare?.runs) ? runCompare.runs : []} />
            </div>
          </section>
        </section>

        <aside className="controls-column controlsCol">
          <h2>Run Controls</h2>

          <div className="controls-grid">
            <div className="control-card control-card-wide">
              <section className="card">
                <div className="cardHeader">
                  <div className="cardTitle">Quick Actions</div>
                  <div className="hint">Most common runs</div>
                </div>
                <div className="cardBody">
                  <div className="action-row" style={{ flexWrap: "wrap" }}>
                    <JobButton label="Run ACO" jobName="aco" jobsSummary={jobsSummary} onRun={() => startJob("aco", { num_ants: 32, num_iters: 80, max_length: 6, max_power: 4, reward_mode: "rank" })} />
                    <JobButton label="Run Topo Eval" jobName="topo-eval" jobsSummary={jobsSummary} onRun={() => startJob("topo-eval", {})} />
                    <JobButton label="Run PDE" jobName="pde" jobsSummary={jobsSummary} onRun={() => startJob("pde", {})} />
                    <JobButton label="Run Sensitivity" jobName="sensitivity" jobsSummary={jobsSummary} onRun={() => startJob("sensitivity", {})} />
                    <button
                      className="btn btnPrimary"
                      onClick={() => startPipeline({ continue_on_failure: true, include_ppo: false })}
                      disabled={Boolean(jobsSummary?.running_count && jobsSummary.running_count > 0)}
                      title="Starts a sequential pipeline in backend"
                    >
                      Run Full Pipeline
                    </button>
                  </div>
                  {Boolean(jobsSummary?.running_count && jobsSummary.running_count > 0) ? (
                    <div className="mono muted" style={{ marginTop: 10 }}>
                      Pipeline disabled while jobs are running.
                    </div>
                  ) : null}
                </div>
              </section>
            </div>

            <div className="control-card control-card-wide">
              <QuickGuide onRunJob={startJob} jobsSummary={jobsSummary} />
            </div>

            <div className="control-card control-card-wide">
              <ProgressTracker data={progress} onRunJob={startJob} jobsSummary={jobsSummary} />
            </div>

            <div className="control-card">
              <JobControlPanel onJobStarted={setActiveJobId} disabled={anyRunning} jobsSummary={jobsSummary} />
            </div>

            <div className="control-card">
              <JobQueuePanel data={jobQueue} onRefresh={async () => setJobQueue(await getJobQueue())} />
            </div>

            <div className="control-card control-card-wide">
              <LogStream
                jobId={activeJobId}
                enabled={logsEnabled}
                disabledReason={lowResourceMode ? "Live logs disabled in Low Resource Mode." : "Live logs disabled."}
                onEnable={
                  lowResourceMode
                    ? () => {
                        setLogsEnabled(true);
                      }
                    : undefined
                }
              />
            </div>

            <div className="control-card">
              <GemmaHealth data={gemmaHealth} />
            </div>

            <div className="control-card">
              <section className="card">
                <div className="cardHeader">
                  <div className="cardTitle">System Monitor</div>
                  <div className="hint">CPU / RAM / Process / GPU</div>
                </div>
                <div className="cardBody">
                  <SystemMonitor data={systemMetrics} compact />
                </div>
              </section>
            </div>

            <div className="control-card">
              <CheckpointPanel />
            </div>

            <div className="control-card">
              <ImportExportPanel />
            </div>

            <div className="control-card">
              <section className="card">
                <div className="cardHeader">
                  <div className="cardTitle">Screenshots</div>
                  <div className="hint">Saved to runs/screenshots</div>
                </div>
                <div className="cardBody">
                  <div className="row" style={{ flexWrap: "wrap" }}>
                    <ScreenshotButton targetId="dashboard-root" name="dashboard" label="Save Dashboard" />
                    <ScreenshotButton targetId="aco-live-charts" name="aco_chart" label="Save ACO Chart" />
                    <ScreenshotButton targetId="spectral-spacing-chart" name="spectral_spacing" label="Save Spectral Spacing" />
                    <ScreenshotButton targetId="operator-analysis" name="operator_analysis" label="Save Operator Analysis" />
                    <ScreenshotButton targetId="multi-run-comparison" name="multi_run_comparison" label="Save Run Comparison" />
                  </div>
                </div>
              </section>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}

