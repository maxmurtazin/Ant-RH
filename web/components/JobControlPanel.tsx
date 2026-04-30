"use client";

import { useMemo, useState } from "react";
import { startJob } from "@/lib/api";
import { JobButton } from "@/components/JobButton";

type Props = {
  onJobStarted: (jobId: string) => void;
  disabled?: boolean;
  jobsSummary?: any;
};

function num(v: string) {
  const x = Number(v);
  return Number.isFinite(x) ? x : undefined;
}

export function JobControlPanel({
  onJobStarted,
  disabled,
  jobsSummary,
}: Props) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  // ACO params
  const [numAnts, setNumAnts] = useState("32");
  const [numIters, setNumIters] = useState("80");
  const [maxLength, setMaxLength] = useState("6");
  const [maxPower, setMaxPower] = useState("4");
  const [usePlanner, setUsePlanner] = useState(true);
  const [lambdaSelberg, setLambdaSelberg] = useState("");
  const [lambdaSpec, setLambdaSpec] = useState("");
  const [lambdaSpacing, setLambdaSpacing] = useState("");
  const [rewardMode, setRewardMode] = useState<"rank" | "raw" | "hybrid">("rank");

  const acoParams = useMemo(() => {
    const p: Record<string, any> = {
      num_ants: num(numAnts),
      num_iters: num(numIters),
      max_length: num(maxLength),
      max_power: num(maxPower),
      use_planner: usePlanner,
      reward_mode: rewardMode
    };
    const ls = num(lambdaSelberg);
    const lsp = num(lambdaSpec);
    const lspc = num(lambdaSpacing);
    if (ls !== undefined) p.lambda_selberg = ls;
    if (lsp !== undefined) p.lambda_spec = lsp;
    if (lspc !== undefined) p.lambda_spacing = lspc;
    // remove undefined
    for (const k of Object.keys(p)) if (p[k] === undefined) delete p[k];
    return p;
  }, [numAnts, numIters, maxLength, maxPower, usePlanner, lambdaSelberg, lambdaSpec, lambdaSpacing, rewardMode]);

  async function run(job: string, params?: Record<string, any>) {
    setBusy(true);
    setError("");
    try {
      const res = await startJob(job, params || {});
      onJobStarted(res.job_id);
    } catch (e) {
      const anyErr: any = e;
      if (anyErr?.status === 409) {
        setError("Job conflict: another job is already running or paused.");
      } else if (anyErr?.status === 429) {
        setError("Another job is already running.");
      } else {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="card">
      <div className="cardHeader">
        <div className="cardTitle">Job Control</div>
        <div className="hint">Launch jobs + parameters</div>
      </div>
      <div className="cardBody">
        {error ? <div className="errorBox mono">{error}</div> : null}

        <div className="twoCol">
          <div>
            <div className="label">ACO</div>
            <div className="row">
              <label className="mono muted" style={{ fontSize: 12 }}>
                num_ants
                <input className="input mono" value={numAnts} onChange={(e) => setNumAnts(e.target.value)} />
              </label>
              <label className="mono muted" style={{ fontSize: 12 }}>
                num_iters
                <input className="input mono" value={numIters} onChange={(e) => setNumIters(e.target.value)} />
              </label>
              <label className="mono muted" style={{ fontSize: 12 }}>
                max_length
                <input className="input mono" value={maxLength} onChange={(e) => setMaxLength(e.target.value)} />
              </label>
              <label className="mono muted" style={{ fontSize: 12 }}>
                max_power
                <input className="input mono" value={maxPower} onChange={(e) => setMaxPower(e.target.value)} />
              </label>
            </div>

            <div className="row" style={{ marginTop: 10 }}>
              <label className="mono muted" style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 10 }}>
                <input type="checkbox" checked={usePlanner} onChange={(e) => setUsePlanner(e.target.checked)} />
                use_planner
              </label>

              <label className="mono muted" style={{ fontSize: 12 }}>
                reward_mode
                <select className="input mono" value={rewardMode} onChange={(e) => setRewardMode(e.target.value as any)}>
                  <option value="rank">rank</option>
                  <option value="raw">raw</option>
                  <option value="hybrid">hybrid</option>
                </select>
              </label>
            </div>

            <div className="row" style={{ marginTop: 10 }}>
              <label className="mono muted" style={{ fontSize: 12 }}>
                lambda_selberg
                <input className="input mono" value={lambdaSelberg} onChange={(e) => setLambdaSelberg(e.target.value)} placeholder="(optional)" />
              </label>
              <label className="mono muted" style={{ fontSize: 12 }}>
                lambda_spec
                <input className="input mono" value={lambdaSpec} onChange={(e) => setLambdaSpec(e.target.value)} placeholder="(optional)" />
              </label>
              <label className="mono muted" style={{ fontSize: 12 }}>
                lambda_spacing
                <input className="input mono" value={lambdaSpacing} onChange={(e) => setLambdaSpacing(e.target.value)} placeholder="(optional)" />
              </label>
            </div>

            <div className="row" style={{ marginTop: 12 }}>
              <JobButton
                label="Run ACO"
                jobName="aco"
                jobsSummary={jobsSummary}
                onRun={() => run("aco", acoParams)}
              />
              <span className="hint mono">Params are validated server-side.</span>
            </div>
          </div>

          <div>
            <div className="label">TopologicalLM</div>
            <div className="row">
              <JobButton label="Train TopologicalLM" jobName="topo-train" jobsSummary={jobsSummary} onRun={() => run("topo-train")} />
              <JobButton label="Run Topo Eval" jobName="topo-eval" jobsSummary={jobsSummary} onRun={() => run("topo-eval")} />
              <JobButton label="Run PPO" jobName="topo-ppo" jobsSummary={jobsSummary} onRun={() => run("topo-ppo")} />
              <JobButton label="Run Topo Report" jobName="topo-report" jobsSummary={jobsSummary} onRun={() => run("topo-report")} />
            </div>

            <div className="divider" />

            <div className="label">Utility</div>
            <div className="row">
              <JobButton label="Run Study" jobName="study" jobsSummary={jobsSummary} onRun={() => run("study")} />
              <JobButton label="Run Docs" jobName="docs" jobsSummary={jobsSummary} onRun={() => run("docs")} />
              <JobButton label="Run Gemma Health" jobName="gemma-health" jobsSummary={jobsSummary} onRun={() => run("gemma-health")} />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

