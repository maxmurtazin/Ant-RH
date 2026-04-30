"use client";

import { useState } from "react";
import { MetricCard } from "@/components/MetricCard";
import { exportData, importData } from "@/lib/api";

export function ImportExportPanel() {
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string>("");

  async function onUpload() {
    if (!file) return;
    setBusy(true);
    setMsg("");
    try {
      const res = await importData(file);
      setMsg(typeof res === "object" ? JSON.stringify(res) : String(res));
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <MetricCard title="Import / Export" span2>
      <div className="row">
        <button className="btn btnPrimary" onClick={() => exportData()}>
          Export (.zip)
        </button>
      </div>

      <div className="divider" />

      <div className="row">
        <input
          className="input mono"
          type="file"
          accept=".zip,application/zip"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
        <button className="btn" disabled={busy || !file} onClick={onUpload}>
          {busy ? "Uploading…" : "Import"}
        </button>
      </div>

      {msg ? (
        <pre className="console mono" style={{ marginTop: 12 }}>
          {msg}
        </pre>
      ) : (
        <div className="hint" style={{ marginTop: 10 }}>
          Import will copy allowed files into `runs/` (models excluded).
        </div>
      )}
    </MetricCard>
  );
}

