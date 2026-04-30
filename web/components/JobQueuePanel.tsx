"use client";

import { MetricCard } from "@/components/MetricCard";
import { cancelJob, moveJobDown, moveJobUp, pauseJob, resumeJob, stopJob } from "@/lib/api";
import { useState } from "react";

function fmtEta(sec?: number | null) {
  const s = Math.max(0, Math.floor(Number(sec || 0)));
  if (!Number.isFinite(s)) return "—";
  const m = Math.floor(s / 60);
  const ss = s % 60;
  return `${String(m).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
}

export function JobQueuePanel(props: { data: { running: any[]; queued: any[]; done_recent: any[] } | null; onRefresh?: () => void }) {
  const [busy, setBusy] = useState(false);
  const d = props.data;

  async function act(fn: () => Promise<any>) {
    setBusy(true);
    try {
      await fn();
      props.onRefresh?.();
    } finally {
      setBusy(false);
    }
  }

  return (
    <MetricCard title="Job Queue" span2>
      {!d ? (
        <div className="mono muted">No queue data yet.</div>
      ) : (
        <div className="queue-panel">
          <div className="label">Running</div>
          {d.running?.length ? (
            d.running.map((j) => (
              <div key={String(j.id)} className={`queue-row queue-status-running`}>
                <div className="mono">
                  <div style={{ fontWeight: 750 }}>{String(j.name || j.job || j.id)}</div>
                  <div className="muted">
                    elapsed {fmtEta(j.elapsed_seconds)} · ETA {fmtEta(j.eta_seconds)}
                  </div>
                </div>
                <div className="action-row">
                  {String(j.status) === "paused" ? (
                    <button className="btn" disabled={busy} onClick={() => act(() => resumeJob(String(j.id)))}>
                      Resume
                    </button>
                  ) : (
                    <button className="btn" disabled={busy} onClick={() => act(() => pauseJob(String(j.id)))}>
                      Pause
                    </button>
                  )}
                  <button className="btn" disabled={busy} onClick={() => act(() => stopJob(String(j.id)))}>
                    Stop
                  </button>
                  <button className="btn" disabled={busy} onClick={() => act(() => cancelJob(String(j.id)))}>
                    Cancel
                  </button>
                </div>
              </div>
            ))
          ) : (
            <div className="mono muted">None</div>
          )}

          <div className="divider" />
          <div className="label">Queued</div>
          {d.queued?.length ? (
            d.queued.map((j, idx) => (
              <div key={String(j.id)} className={`queue-row queue-status-queued`}>
                <div className="mono">
                  <div style={{ fontWeight: 750 }}>
                    #{idx + 1} {String(j.name || j.job || j.id)}
                  </div>
                  <div className="muted">status: queued</div>
                </div>
                <div className="action-row">
                  <button className="btn" disabled={busy || idx === 0} onClick={() => act(() => moveJobUp(String(j.id)))}>
                    Up
                  </button>
                  <button
                    className="btn"
                    disabled={busy || idx === d.queued.length - 1}
                    onClick={() => act(() => moveJobDown(String(j.id)))}
                  >
                    Down
                  </button>
                  <button className="btn" disabled={busy} onClick={() => act(() => cancelJob(String(j.id)))}>
                    Cancel
                  </button>
                </div>
              </div>
            ))
          ) : (
            <div className="mono muted">Empty</div>
          )}

          <div className="divider" />
          <div className="label">Recent</div>
          {d.done_recent?.length ? (
            d.done_recent.slice(0, 8).map((j) => (
              <div key={String(j.id)} className={`queue-row queue-status-done`}>
                <div className="mono">
                  <div style={{ fontWeight: 750 }}>{String(j.name || j.job || j.id)}</div>
                  <div className="muted">
                    {String(j.status)} · duration {fmtEta(j.elapsed_seconds)}
                  </div>
                </div>
                <div className="mono muted">{String(j.ended_at || "—")}</div>
              </div>
            ))
          ) : (
            <div className="mono muted">None</div>
          )}
        </div>
      )}
    </MetricCard>
  );
}

