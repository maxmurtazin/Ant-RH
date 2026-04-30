"use client";

import { useMemo } from "react";

function fmtEta(sec?: number | null) {
  const s = Math.max(0, Math.floor(Number(sec || 0)));
  if (s >= 3600) {
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    return `${h}h ${String(m).padStart(2, "0")}m`;
  }
  const m = Math.floor(s / 60);
  const ss = s % 60;
  return `${String(m).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
}

export function JobButton(props: {
  label: string;
  jobName: string;
  onRun: () => void;
  jobsSummary?: any;
  className?: string;
}) {
  const job = props.jobsSummary?.latest_by_name?.[props.jobName] || null;
  const status = String(job?.status || "idle");
  const eta = typeof job?.eta_seconds === "number" ? job.eta_seconds : null;
  const runningCount = Number(props.jobsSummary?.running_count ?? 0);
  const maxConcurrent = Number(props.jobsSummary?.max_concurrent_jobs ?? 2);
  const globalCapReached = runningCount >= maxConcurrent;

  const { text, disabled, badge } = useMemo(() => {
    if (status === "running") {
      return {
        text: eta !== null ? `${props.label} running — ETA ${fmtEta(eta)}` : `${props.label} running…`,
        disabled: true,
        badge: null
      };
    }
    if (status === "paused") {
      return {
        text: `${props.label} paused`,
        disabled: true,
        badge: "paused"
      };
    }
    if (globalCapReached) {
      return {
        text: props.label,
        disabled: true,
        badge: "busy"
      };
    }
    if (status === "done") {
      return { text: props.label, disabled: false, badge: "done" };
    }
    if (status === "failed") {
      return { text: props.label, disabled: false, badge: "failed" };
    }
    return { text: props.label, disabled: false, badge: null };
  }, [status, eta, globalCapReached, props.label]);

  const cls =
    status === "running"
      ? "job-button running"
      : status === "paused"
        ? "job-button running"
      : status === "done"
        ? "job-button done"
        : status === "failed"
          ? "job-button failed"
          : "job-button";

  return (
    <div className="jobButtonWrap">
      <button className={`btn ${cls} ${props.className || ""}`} disabled={disabled} onClick={props.onRun}>
        {text}
      </button>
      {badge ? <span className={`job-badge ${badge}`}>{badge}</span> : null}
    </div>
  );
}

