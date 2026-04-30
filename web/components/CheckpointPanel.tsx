"use client";

import { useEffect, useMemo, useState } from "react";
import { MetricCard } from "@/components/MetricCard";
import { createExport, deleteExport, downloadExport, getExports, type ExportEntry } from "@/lib/api";

function fmtBytes(n: number) {
  if (!Number.isFinite(n) || n <= 0) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let x = n;
  let i = 0;
  while (x >= 1024 && i < units.length - 1) {
    x /= 1024;
    i += 1;
  }
  return `${x.toFixed(1)} ${units[i]}`;
}

export function CheckpointPanel() {
  const [items, setItems] = useState<ExportEntry[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function refresh() {
    try {
      setError("");
      const ex = await getExports();
      setItems(Array.isArray(ex) ? ex : []);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  useEffect(() => {
    refresh();
    const id = window.setInterval(refresh, 20_000);
    return () => window.clearInterval(id);
  }, []);

  const sorted = useMemo(() => {
    const xs = Array.isArray(items) ? items.slice() : [];
    xs.sort((a, b) => String(b.timestamp || "").localeCompare(String(a.timestamp || "")));
    return xs;
  }, [items]);

  async function createManual() {
    setBusy(true);
    try {
      await createExport("manual_checkpoint", "manual");
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function onDelete(id: string) {
    setBusy(true);
    try {
      await deleteExport(id);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <MetricCard
      title="Checkpoints"
      span2
      right={
        <div className="action-row">
          <button className="btn btnPrimary" disabled={busy} onClick={createManual}>
            Create Manual Checkpoint
          </button>
          <button className="btn" disabled={busy} onClick={refresh}>
            Refresh
          </button>
        </div>
      }
    >
      {error ? <div className="errorBox mono">{error}</div> : null}
      {sorted.length === 0 ? (
        <div className="mono muted">No checkpoints yet.</div>
      ) : (
        <table className="table mono compactTable">
          <thead>
            <tr>
              <td className="k">timestamp</td>
              <td className="k">name</td>
              <td className="k">reason</td>
              <td className="k">size</td>
              <td className="k">metrics</td>
              <td className="v">actions</td>
            </tr>
          </thead>
          <tbody>
            {sorted.map((e) => (
              <tr key={e.id}>
                <td className="k">{e.timestamp || "—"}</td>
                <td className="k">{e.name}</td>
                <td className="k">{e.reason}</td>
                <td className="k">{fmtBytes(e.size_bytes)}</td>
                <td className="k">
                  {e.metrics_summary ? (
                    <>
                      aco_best_loss={String(e.metrics_summary.aco_best_loss ?? "—")}
                      <br />
                      topo_reward_mean={String(e.metrics_summary.topo_reward_mean ?? "—")}
                      <br />
                      spectral_loss={String(e.metrics_summary.spectral_loss ?? "—")}
                    </>
                  ) : (
                    "—"
                  )}
                </td>
                <td className="v">
                  <div className="action-row" style={{ justifyContent: "flex-end" }}>
                    <button className="btn" onClick={() => downloadExport(e.id)}>
                      Download
                    </button>
                    <button className="btn" disabled={busy} onClick={() => onDelete(e.id)}>
                      Delete
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </MetricCard>
  );
}

