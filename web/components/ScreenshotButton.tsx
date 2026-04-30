"use client";

import { useState } from "react";
import html2canvas from "html2canvas";
import { saveScreenshot } from "@/lib/api";

export function ScreenshotButton(props: { targetId: string; name: string; label?: string }) {
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState<string>("");
  const [error, setError] = useState<string>("");

  async function run() {
    setBusy(true);
    setError("");
    setSaved("");
    try {
      const el = document.getElementById(props.targetId);
      if (!el) throw new Error(`Target not found: ${props.targetId}`);
      const canvas = await html2canvas(el, { backgroundColor: null, scale: 2 });
      const dataUrl = canvas.toDataURL("image/png");
      const res = await saveScreenshot(props.name, dataUrl);
      setSaved(res.path || "");
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="screenshotBtnWrap">
      <button className="btn btnSecondary screenshotBtn" disabled={busy} onClick={run}>
        {busy ? "Saving…" : props.label || "Save screenshot"}
      </button>
      {saved ? <div className="mono muted screenshotSaved">saved: {saved}</div> : null}
      {error ? <div className="mono screenshotError">{error}</div> : null}
    </div>
  );
}

