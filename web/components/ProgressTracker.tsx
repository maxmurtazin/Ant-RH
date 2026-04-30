"use client";

import { MetricCard } from "@/components/MetricCard";
import { JobButton } from "@/components/JobButton";

type StepStatus = "done" | "running" | "pending" | "failed";

export type ExperimentProgress = {
  steps: { id: string; title: string; status: StepStatus; eta_seconds: number; evidence: string; command: string }[];
  overall_progress: number;
  estimated_remaining_seconds: number;
};

function fmtEta(sec: number) {
  const s = Math.max(0, Math.floor(sec || 0));
  if (s >= 3600) {
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    return `${h}h ${String(m).padStart(2, "0")}m`;
  }
  const m = Math.floor(s / 60);
  const ss = s % 60;
  return `${String(m).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
}

function statusLabel(st: StepStatus) {
  if (st === "done") return "✓ done";
  if (st === "running") return "running";
  if (st === "failed") return "failed";
  return "pending";
}

export function ProgressTracker(props: {
  data: ExperimentProgress | null;
  onRunJob: (job: string, params?: any) => Promise<void> | void;
  jobsSummary?: any;
}) {
  const steps = Array.isArray(props.data?.steps) ? props.data!.steps : [];
  const overall = Math.max(0, Math.min(1, Number(props.data?.overall_progress ?? 0)));
  const remaining = Number(props.data?.estimated_remaining_seconds ?? 0);

  const stepToJob: Record<string, string | null> = {
    health: "gemma-health",
    aco: "aco",
    analyze: "analyze",
    topo_lm: "topo-eval",
    ppo: "topo-ppo",
    physics: "stability",
    operator_analysis: "pde",
    checkpoint: "checkpoint",
    export: "export",
    setup_dashboard: null,
  };

  return (
    <MetricCard title="Experiment Progress" span2>
      <div className="progressBarOuter">
        <div className="progressBarInner" style={{ width: `${Math.round(overall * 100)}%` }} />
      </div>
      <div className="mono muted" style={{ marginTop: 8 }}>
        Overall: {(overall * 100).toFixed(0)}% · Remaining ETA: {fmtEta(remaining)}
      </div>

      <div className="divider" />

      <div className="progressList">
        {steps.map((s) => {
          const job = stepToJob[s.id] ?? null;
          const canRun = Boolean(job);
          const cls =
            s.status === "done"
              ? "progressStatus progressStatusDone"
              : s.status === "running"
                ? "progressStatus progressStatusRunning"
                : s.status === "failed"
                  ? "progressStatus progressStatusFailed"
                  : "progressStatus progressStatusPending";
          return (
            <div key={s.id} className="progressRow">
              <div className={cls}>{statusLabel(s.status)}</div>
              <div className="progressMain">
                <div className="progressTitle">{s.title}</div>
                <div className="mono muted progressMeta">
                  <span>cmd: {s.command}</span>
                  <span>·</span>
                  <span>evidence: {s.evidence}</span>
                  {s.status === "running" ? (
                    <>
                      <span>·</span>
                      <span>ETA {fmtEta(s.eta_seconds)}</span>
                    </>
                  ) : null}
                </div>
              </div>
              <div className="progressActions">
                {canRun && job ? (
                  <JobButton label={s.status === "done" ? "Re-run" : "Run"} jobName={job} jobsSummary={props.jobsSummary} onRun={() => props.onRunJob(job, {})} />
                ) : (
                  <button className="btn" disabled={true}>
                    Run
                  </button>
                )}
              </div>
            </div>
          );
        })}
        {!steps.length ? <div className="mono muted">No data yet</div> : null}
      </div>
    </MetricCard>
  );
}

