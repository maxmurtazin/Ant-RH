"use client";

import { useMemo, useState } from "react";
import type { RunResultResponse, RunStageRequest } from "@/lib/api";
import { apiPost } from "@/lib/api";

const allowedStages: RunStageRequest["stage"][] = [
  "study",
  "analyze",
  "journal",
  "docs",
  "topo-eval",
  "topo-report",
  "gemma-health"
];

export function ActionPanel({ onAfterAction }: { onAfterAction?: () => void }) {
  const [busy, setBusy] = useState(false);
  const [last, setLast] = useState<RunResultResponse | null>(null);
  const [err, setErr] = useState<string>("");

  const consoleText = useMemo(() => {
    if (err) return `error: ${err}`;
    if (!last) return "—";
    const meta = [
      `ok=${String(last.ok)}`,
      `target=${last.target}`,
      `rc=${String(last.returncode)}`,
      `duration_s=${String(last.duration_s)}`,
      `timed_out=${String(last.timed_out)}`
    ].join("\n");
    return [meta, "", "stdout:", last.stdout || "—", "", "stderr:", last.stderr || "—"].join("\n");
  }, [last, err]);

  async function run(stage: RunStageRequest["stage"]) {
    if (!allowedStages.includes(stage)) return;
    setBusy(true);
    setErr("");
    try {
      const res = await apiPost<RunResultResponse>("/run/stage", { stage });
      setLast(res);
      onAfterAction?.();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <div className="row">
        {allowedStages.map((s) => (
          <button key={s} className="btn" disabled={busy} onClick={() => void run(s)}>
            {busy ? "Running…" : `Run ${s}`}
          </button>
        ))}
      </div>
      <pre className="console mono">{consoleText}</pre>
    </div>
  );
}

