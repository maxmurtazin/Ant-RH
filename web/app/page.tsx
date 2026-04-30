"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ActionPanel } from "@/components/ActionPanel";
import { AcoMetrics } from "@/components/AcoMetrics";
import AcoLiveCharts from "@/components/AcoLiveCharts";
import { AcoCharts } from "@/components/AcoCharts";
import { GemmaHealth } from "@/components/GemmaHealth";
import { GemmaHelp } from "@/components/GemmaHelp";
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
import { NextStepAdvisor } from "@/components/NextStepAdvisor";
import { FullPipelinePanel } from "@/components/FullPipelinePanel";
import { RewardHistogram } from "@/components/RewardHistogram";
import { SpectralSpacingChart } from "@/components/SpectralSpacingChart";
import { TopologicalLmMetrics } from "@/components/TopologicalLmMetrics";
import { StatusBadge } from "@/components/StatusBadge";
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

  const refreshInFlightRef = useRef(false);

  useEffect(() => {
    try {
      const v = window.localStorage.getItem("ant_rh_low_resource_mode");
      const on = v === "1" || v === "true";
      setLowResourceMode(on);
      setLogsEnabled(!on);
    } catch {
      // ignore
    }
  }, []);

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

  const refreshAll = useCallback(
    async (kind: "auto" | "manual" = "manual") => {
    if (refreshInFlightRef.current) return;
    refreshInFlightRef.current = true;
    setIsRefreshing(true);

    const nextErrors: Partial<Record<EndpointKey, string>> = {};
    let anySuccess = false;
    const includeHeavy = !lowResourceMode || kind === "manual";

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

      if (includeHeavy) {
        try {
          const history = await getAcoHistory(300);
          setAcoHistory(Array.isArray(history) ? history : []);
          setErrors((prev) => ({ ...prev, acoHistory: undefined }));
        } catch (e) {
          setErrors((prev) => ({ ...prev, acoHistory: String(e) }));
          nextErrors["aco-history"] = e instanceof Error ? e.message : String(e);
        }
      }

      // TopologicalLM history (best-effort; keep last good data on failure)
      if (includeHeavy) {
        try {
          const th = await getTopoHistory(300);
          setTopoHistory(Array.isArray(th.points) ? th.points : []);
          setTopoRewardSamples(Array.isArray(th.reward_samples) ? th.reward_samples : []);
        } catch (e) {
          nextErrors["topological-lm"] = nextErrors["topological-lm"] || (e instanceof Error ? e.message : String(e));
        }
      }

      // Physics history (best-effort; keep last good data on failure)
      if (includeHeavy) {
        try {
          const ph = await getPhysicsHistory(300);
          setPhysicsHistory(Array.isArray(ph.points) ? ph.points : []);
        } catch (e) {
          nextErrors["physics"] = nextErrors["physics"] || (e instanceof Error ? e.message : String(e));
        }
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
    },
    [lowResourceMode]
  );

  useEffect(() => {
    refreshAll("manual");
    const id = window.setInterval(() => refreshAll("auto"), lowResourceMode ? 20_000 : 5_000);
    return () => window.clearInterval(id);
  }, [refreshAll, lowResourceMode]);

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
    const id = window.setInterval(refreshSummary, lowResourceMode ? 10_000 : 3_000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [lowResourceMode]);

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
    const id = window.setInterval(refreshJobs, lowResourceMode ? 10_000 : 5_000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [lowResourceMode]);

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
    const id = window.setInterval(refreshProgress, lowResourceMode ? 20_000 : 5_000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [lowResourceMode]);

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
    const id = window.setInterval(refreshSpectral, lowResourceMode ? 60_000 : 20_000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [lowResourceMode]);

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
    const id = window.setInterval(refreshOperator, lowResourceMode ? 30_000 : 10_000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [lowResourceMode]);

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
    <div className="dashboardPage">
      <div className="dashboardHeader">
        <div className="dashboardHeaderLeft">
          <div className="dashboardTitle">Ant-RH Control Dashboard</div>
          <div className="mono muted dashboardSub">
            <span className="badge mono">Live: ON</span>
            <span className="badge mono">API base: {API_BASE}</span>
            <span className="badge mono">Last updated: {lastUpdated}</span>
          </div>
        </div>
        <div className="dashboardHeaderRight">
          {isRefreshing ? <span className="spinner" aria-label="Refreshing" /> : null}
          <label className="mono muted" style={{ display: "inline-flex", alignItems: "center", gap: 8, marginRight: 10 }}>
            <input
              type="checkbox"
              checked={lowResourceMode}
              onChange={(e) => {
                const on = e.target.checked;
                setLowResourceMode(on);
                setLogsEnabled(!on);
                try {
                  window.localStorage.setItem("ant_rh_low_resource_mode", on ? "1" : "0");
                } catch {
                  // ignore
                }
              }}
            />
            Low Resource Mode
          </label>
          <button className="btn btnPrimary" onClick={() => refreshAll("manual")} disabled={isRefreshing}>
            Refresh
          </button>
        </div>
      </div>

      <div className="dashboardGrid">
        <div className="resultsCol">
          <div className="sectionTitle">Results &amp; Diagnostics</div>

          <section className="card">
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
          <div className="mono muted">Auto-refresh every 5s</div>
        </div>
          </section>

      <AcoMetrics data={aco} />
      <AcoLiveCharts points={acoHistory} />
      <AcoCharts points={acoHistory} />
      <OperatorCharts
        operatorDistanceMean={operatorAnalysis?.sensitivity?.operator_distance_mean ?? null}
        lossStd={operatorAnalysis?.sensitivity?.loss_std ?? null}
      />
      <SpectralSpacingChart data={spectralSpacing} />
      <RewardHistogram samples={topoRewardSamples} />
      <PhysicsCharts points={physicsHistory} />
      <TopologicalLmMetrics data={topo} />
      <PhysicsDiagnostics data={physics} />
      <OperatorAnalysis data={operatorAnalysis} />
        </div>

        <div className="controlsCol">
          <div className="controlsSticky">
            <div className="sectionTitle">Run Controls</div>

            <QuickGuide onRunJob={startJob} jobsSummary={jobsSummary} />
            <NextStepAdvisor onRunJob={startJob} onStartFullPipeline={startPipeline} jobsSummary={jobsSummary} />
            <FullPipelinePanel onStart={startPipeline} activeJobId={activeJobId} jobsSummary={jobsSummary} />
            <ProgressTracker data={progress} onRunJob={startJob} jobsSummary={jobsSummary} />
            <JobControlPanel
              onJobStarted={setActiveJobId}
              disabled={anyRunning}
              runningJobs={runningJobs}
              onRefreshJobs={async () => {
                try {
                  const res = await getJobs();
                  setJobs(Array.isArray(res.jobs) ? res.jobs : []);
                } catch {
                  // ignore
                }
              }}
              jobsSummary={jobsSummary}
            />
            <ImportExportPanel />
            <CheckpointPanel />

            <section className="card">
              <div className="cardHeader">
                <div className="cardTitle">Operator Jobs</div>
                <div className="hint">Runs in backend job manager</div>
              </div>
              <div className="cardBody">
                <div className="action-row">
                  {/* keep as-is; JobControlPanel/QuickGuide have job-aware buttons */}
                  <button className="btn action-button" onClick={() => startJob("pde", {})} disabled={Boolean(jobsSummary?.latest_by_name?.pde?.status === "running")}>
                    Run PDE Discovery
                  </button>
                  <button className="btn action-button secondary" onClick={() => startJob("sensitivity", {})} disabled={Boolean(jobsSummary?.latest_by_name?.sensitivity?.status === "running")}>
                    Run Sensitivity Test
                  </button>
                  <button className="btn action-button secondary" onClick={() => startJob("stability", {})} disabled={Boolean(jobsSummary?.latest_by_name?.stability?.status === "running")}>
                    Run Stability Report
                  </button>
                </div>
                {operatorError ? <div className="errorBox mono" style={{ marginTop: 10 }}>{operatorError}</div> : null}
              </div>
            </section>

            <section className="card">
              <div className="cardHeader">
                <div className="cardTitle">Actions</div>
                <div className="hint">Whitelisted actions only</div>
              </div>
              <div className="cardBody">
                <ActionPanel onAfterAction={refreshAll} />
              </div>
            </section>

            <GemmaHealth data={gemmaHealth} />

            <section className="card">
              <div className="cardHeader">
                <div className="cardTitle">Gemma Help</div>
                <div className="hint">Local model; may be slow</div>
              </div>
              <div className="cardBody">
                <GemmaHelp />
              </div>
            </section>

            <LogStream jobId={activeJobId} />
          </div>
        </div>
      </div>
    </div>
  );
}

