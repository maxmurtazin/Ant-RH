"use client";

import { useEffect, useMemo, useState } from "react";
import { MetricCard } from "@/components/MetricCard";
import { API_BASE, createExport, fetchJson } from "@/lib/api";
import { JobButton } from "@/components/JobButton";

type Stage =
  | "setup"
  | "health"
  | "aco"
  | "analyze"
  | "topolm"
  | "ppo"
  | "physics"
  | "operator"
  | "checkpoint"
  | "export"
  | "iterate";

export function QuickGuide(props: {
  onRunJob: (job: string, params?: any) => Promise<void> | void;
  jobsSummary?: any;
}) {
  const [currentStage, setCurrentStage] = useState<Stage>("setup");
  const [playbook, setPlaybook] = useState<string>("");
  const [busy, setBusy] = useState<string>("");
  const [msg, setMsg] = useState<string>("");

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const res = (await fetchJson("/docs/playbook")) as any;
        if (!cancelled) setPlaybook(typeof res?.content === "string" ? res.content : "");
      } catch {
        // ignore
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const steps = useMemo(
    () => [
      { n: 0, id: "setup" as const, title: "Setup", desc: "make dashboard-next · open http://localhost:3000" },
      { n: 1, id: "health" as const, title: "Health", desc: "Run Gemma Health; verify llama-cli + models" },
      { n: 2, id: "aco" as const, title: "ACO", desc: "Baseline ACO; watch best_loss/mean_loss ↓" },
      { n: 3, id: "analyze" as const, title: "Analyze", desc: "Analyze + lab journal; review status/reports" },
      { n: 4, id: "topolm" as const, title: "TopoLM", desc: "Train + eval; check reward_mean + unique ratio" },
      { n: 5, id: "ppo" as const, title: "PPO (optional)", desc: "Run PPO; improve reward + diversity" },
      { n: 6, id: "physics" as const, title: "Physics", desc: "Self-adjoint ok; r_mean ~ 0.5 (GUE-like)" },
      { n: 7, id: "operator" as const, title: "Operator", desc: "Run PDE + sensitivity; confirm formula + signal" },
      { n: 8, id: "checkpoint" as const, title: "Checkpoint", desc: "Create a reproducible checkpoint zip" },
      { n: 9, id: "export" as const, title: "Export", desc: "Export ZIP for sharing/backup" },
      { n: 10, id: "iterate" as const, title: "Iterate", desc: "Loop: ACO → LM → PPO → PDE → Analysis" }
    ],
    []
  );

  async function run(job: string, stage: Stage, params?: any) {
    setBusy(job);
    setMsg("");
    setCurrentStage(stage);
    try {
      await props.onRunJob(job, params || {});
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy("");
    }
  }

  async function checkpoint() {
    setBusy("checkpoint");
    setMsg("");
    setCurrentStage("checkpoint");
    try {
      const entry = await createExport("manual_checkpoint", "manual");
      setMsg(`Checkpoint created: ${entry.id}`);
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy("");
    }
  }

  return (
    <MetricCard
      title="Quick Guide"
      span2
      right={
        <span className="badge mono" title="API base used by dashboard">
          {API_BASE}
        </span>
      }
    >
      {msg ? <div className="errorBox mono">{msg}</div> : null}

      <div className="quickGuideList">
        {steps.map((s) => (
          <div key={s.id} className={`quickGuideStep ${currentStage === s.id ? "quickGuideStepActive" : ""}`}>
            <div className="quickGuideNum mono">{s.n}</div>
            <div className="quickGuideMain">
              <div className="quickGuideTitle">{s.title}</div>
              <div className="quickGuideDesc mono muted">{s.desc}</div>
              {s.id === "health" ? (
                <div className="action-row" style={{ marginTop: 8 }}>
                  <JobButton label="Run Gemma Health" jobName="gemma-health" jobsSummary={props.jobsSummary} onRun={() => run("gemma-health", "health")} />
                </div>
              ) : null}
              {s.id === "aco" ? (
                <div className="action-row" style={{ marginTop: 8 }}>
                  <JobButton
                    label="Run ACO"
                    jobName="aco"
                    jobsSummary={props.jobsSummary}
                    onRun={() => run("aco", "aco", { num_ants: 32, num_iters: 80, max_length: 6, max_power: 4, reward_mode: "rank" })}
                  />
                </div>
              ) : null}
              {s.id === "topolm" ? (
                <div className="action-row" style={{ marginTop: 8 }}>
                  <JobButton label="Run Topo Train" jobName="topo-train" jobsSummary={props.jobsSummary} onRun={() => run("topo-train", "topolm")} />
                  <JobButton label="Run Topo Eval" jobName="topo-eval" jobsSummary={props.jobsSummary} onRun={() => run("topo-eval", "topolm")} />
                </div>
              ) : null}
              {s.id === "ppo" ? (
                <div className="action-row" style={{ marginTop: 8 }}>
                  <JobButton label="Run PPO" jobName="topo-ppo" jobsSummary={props.jobsSummary} onRun={() => run("topo-ppo", "ppo")} />
                </div>
              ) : null}
              {s.id === "operator" ? (
                <div className="action-row" style={{ marginTop: 8 }}>
                  <JobButton label="Run PDE" jobName="pde" jobsSummary={props.jobsSummary} onRun={() => run("pde", "operator")} />
                  <JobButton label="Run Sensitivity" jobName="sensitivity" jobsSummary={props.jobsSummary} onRun={() => run("sensitivity", "operator")} />
                </div>
              ) : null}
              {s.id === "checkpoint" ? (
                <div className="action-row" style={{ marginTop: 8 }}>
                  <button className="btn action-button" disabled={busy !== ""} onClick={checkpoint}>
                    Create Checkpoint
                  </button>
                </div>
              ) : null}
              {s.id === "export" ? (
                <div className="action-row" style={{ marginTop: 8 }}>
                  <button className="btn action-button" disabled={busy !== ""} onClick={() => window.open(`${API_BASE}/export`, "_blank")}>
                    Export ZIP
                  </button>
                </div>
              ) : null}
            </div>
          </div>
        ))}
      </div>

      {playbook ? (
        <details style={{ marginTop: 10 }}>
          <summary className="mono muted" style={{ cursor: "pointer" }}>
            full playbook (markdown)
          </summary>
          <pre className="console mono">{playbook}</pre>
        </details>
      ) : null}
    </MetricCard>
  );
}

