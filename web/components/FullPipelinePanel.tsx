"use client";

import { useState } from "react";
import { MetricCard } from "@/components/MetricCard";
import { JobButton } from "@/components/JobButton";

export function FullPipelinePanel(props: {
  onStart: (opts: { continue_on_failure: boolean; include_ppo: boolean }) => Promise<void> | void;
  activeJobId: string | null;
  disabled?: boolean;
  jobsSummary?: any;
}) {
  const [includePpo, setIncludePpo] = useState(false);
  const [continueOnFailure, setContinueOnFailure] = useState(true);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  async function run() {
    setBusy(true);
    setMsg("");
    try {
      await props.onStart({ continue_on_failure: continueOnFailure, include_ppo: includePpo });
      setMsg("Started full pipeline job.");
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <MetricCard title="One-click Pipeline" span2>
      {msg ? <div className="errorBox mono">{msg}</div> : null}
      <div className="mono muted" style={{ marginBottom: 10 }}>
        This may take several minutes.
      </div>
      <div className="row">
        <label className="mono muted" style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 10 }}>
          <input type="checkbox" checked={includePpo} onChange={(e) => setIncludePpo(e.target.checked)} />
          include PPO
        </label>
        <label className="mono muted" style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 10 }}>
          <input type="checkbox" checked={continueOnFailure} onChange={(e) => setContinueOnFailure(e.target.checked)} />
          continue on failure
        </label>
      </div>

      <div className="action-row" style={{ marginTop: 12 }}>
        <JobButton label="Run Full Pipeline" jobName="full-pipeline" jobsSummary={props.jobsSummary} onRun={run} />
        {props.activeJobId ? <span className="badge mono">active job: {props.activeJobId}</span> : null}
      </div>
    </MetricCard>
  );
}

