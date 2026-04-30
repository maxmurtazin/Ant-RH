"use client";

import { useMemo, useState } from "react";
import type { GemmaHelpResponse } from "@/lib/api";
import { apiPost } from "@/lib/api";

export function GemmaHelp() {
  const [question, setQuestion] = useState("");
  const [voice, setVoice] = useState(false);
  const [busy, setBusy] = useState(false);
  const [answer, setAnswer] = useState<string>("—");
  const [err, setErr] = useState<string>("");

  const output = useMemo(() => {
    if (err) return `error: ${err}`;
    return answer || "—";
  }, [answer, err]);

  async function ask() {
    const q = question.trim();
    if (!q) return;
    setBusy(true);
    setErr("");
    setAnswer("working…");
    try {
      const res = await apiPost<GemmaHelpResponse>("/gemma/help", { question: q, voice });
      setAnswer(res.answer || "—");
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      setAnswer("—");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <textarea
        className="textarea mono"
        rows={3}
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Ask a natural-language question…"
      />
      <div className="row" style={{ marginTop: 10, justifyContent: "space-between" }}>
        <label className="mono muted" style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <input type="checkbox" checked={voice} onChange={(e) => setVoice(e.target.checked)} />
          voice
        </label>
        <button className="btn btnPrimary" onClick={() => void ask()} disabled={busy}>
          {busy ? "Asking…" : "Ask Gemma"}
        </button>
      </div>
      <div className="label" style={{ marginTop: 10 }}>
        Answer
      </div>
      <pre className="console mono">{output}</pre>
    </div>
  );
}

