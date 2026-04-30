"use client";

import { useEffect, useState } from "react";
import { MetricCard } from "@/components/MetricCard";
import { getNextRecommendation, type NextStepRecommendation } from "@/lib/api";
import { JobButton } from "@/components/JobButton";

export function NextStepAdvisor(props: {
  onRunJob: (job: string, params?: any) => Promise<void> | void;
  onStartFullPipeline: (opts: { continue_on_failure: boolean; include_ppo: boolean }) => Promise<void> | void;
  disabled?: boolean;
  jobsSummary?: any;
  lowResourceMode?: boolean;
}) {
  const [rec, setRec] = useState<NextStepRecommendation | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [useGemma, setUseGemma] = useState(false);

  async function refresh() {
    setBusy(true);
    setError("");
    try {
      const r = await getNextRecommendation({ use_gemma: Boolean(useGemma) });
      setRec(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    if (props.lowResourceMode) return;
    refresh();
    const id = window.setInterval(refresh, 30_000);
    return () => window.clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.lowResourceMode, useGemma]);

  const pr = rec?.priority || "low";
  const prCls = pr === "high" ? "priority-high" : pr === "medium" ? "priority-medium" : "priority-low";

  async function runRecommended() {
    if (!rec) return;
    if (rec.dashboard_action === "full-pipeline") {
      await props.onStartFullPipeline({ continue_on_failure: true, include_ppo: false });
      return;
    }
    await props.onRunJob(rec.dashboard_action, {});
  }

  return (
    <MetricCard
      title="Gemma Next Step"
      span2
      right={
        <span className={`badge mono ${prCls}`}>
          {rec?.priority ?? "—"}
        </span>
      }
    >
      {error ? <div className="errorBox mono">{error}</div> : null}
      {!rec ? (
        <div className="mono muted">No recommendation yet.</div>
      ) : (
        <div className="mono">
          <div style={{ fontWeight: 750, marginBottom: 6 }}>{rec.next_step}</div>
          <div className="muted" style={{ marginBottom: 8 }}>
            {rec.reason}
          </div>
          <div className="muted">command: {rec.command}</div>
          <div className="muted">source: {rec.source}</div>
        </div>
      )}

      <div className="action-row" style={{ marginTop: 12 }}>
        <button className="btn" disabled={busy} onClick={refresh}>
          Refresh Recommendation
        </button>
        {props.lowResourceMode ? (
          <label className="mono muted" style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
            <input type="checkbox" checked={useGemma} onChange={(e) => setUseGemma(e.target.checked)} />
            use Gemma
          </label>
        ) : null}
        {rec ? (
          <JobButton
            label="Run Recommended Action"
            jobName={rec.dashboard_action === "full-pipeline" ? "full-pipeline" : String(rec.dashboard_action)}
            jobsSummary={props.jobsSummary}
            onRun={runRecommended}
          />
        ) : (
          <button className="btn btnPrimary" disabled={true}>
            Run Recommended Action
          </button>
        )}
      </div>
    </MetricCard>
  );
}

