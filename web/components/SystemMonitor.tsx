"use client";

import type { SystemMetrics } from "@/lib/api";

function UsageBar(props: { label: string; value: number; right?: string; tone?: "ok" | "warn" | "bad" }) {
  const pct = Math.max(0, Math.min(100, Number(props.value || 0)));
  const tone = props.tone || (pct >= 90 ? "bad" : pct >= 75 ? "warn" : "ok");
  return (
    <div className="usage-bar">
      <div className="usage-bar-row">
        <div className="mono muted">{props.label}</div>
        <div className="mono muted">{props.right || `${pct.toFixed(0)}%`}</div>
      </div>
      <div className={`usage-bar-outer ${tone}`}>
        <div className="usage-bar-inner" style={{ width: `${pct.toFixed(0)}%` }} />
      </div>
    </div>
  );
}

function CompactTile(props: { title: string; value: string; percent?: number | null }) {
  const pct = typeof props.percent === "number" && Number.isFinite(props.percent) ? Math.max(0, Math.min(100, props.percent)) : null;
  return (
    <div className="system-tile">
      <div className="mono muted system-tile-title">{props.title}</div>
      <div className="mono system-tile-value">{props.value}</div>
      {pct !== null ? (
        <div className="usage-bar-outer" style={{ height: 8, marginTop: 8 }}>
          <div className="usage-bar-inner" style={{ width: `${pct.toFixed(0)}%` }} />
        </div>
      ) : null}
    </div>
  );
}

export function SystemMonitor(props: { data: SystemMetrics | null; compact?: boolean }) {
  const d = props.data;
  const ramRight =
    d && d.memory_total_gb
      ? `${(d.memory_used_gb || 0).toFixed(1)} / ${(d.memory_total_gb || 0).toFixed(1)} GB`
      : "—";
  const gpuLabel =
    !d?.gpu
      ? "GPU: —"
      : d.gpu.type === "mps"
        ? d.gpu.available
          ? "GPU: MPS available"
          : "GPU: MPS not available"
        : d.gpu.type === "cuda"
          ? d.gpu.available
            ? "GPU: CUDA available"
            : "GPU: CUDA not available"
          : "GPU: none";

  if (props.compact) {
    if (!d) return <div className="system-monitor compact"><div className="mono muted">System: —</div></div>;
    if ((d as any).error) return <div className="system-monitor compact"><div className="mono muted">{String((d as any).message || (d as any).error)}</div></div>;
    return (
      <div className="system-monitor compact">
        <CompactTile title="CPU" value={`${Number(d.cpu_percent || 0).toFixed(0)}%`} percent={Number(d.cpu_percent || 0)} />
        <CompactTile title="RAM" value={ramRight} percent={Number(d.memory_percent || 0)} />
        <CompactTile title="Process" value={`${Number(d.process_cpu_percent || 0).toFixed(0)}% · ${(d.process_memory_mb || 0).toFixed(0)} MB`} percent={Number(d.process_cpu_percent || 0)} />
        <CompactTile title="GPU" value={gpuLabel.replace(/^GPU:\s*/, "")} percent={null} />
      </div>
    );
  }

  return (
    <section className="card span2 chart-card">
      <div className="cardHeader">
        <div className="cardTitle">System Monitor</div>
        <div className="hint">{d?.timestamp ? `updated ${String(d.timestamp)}` : "—"}</div>
      </div>
      <div className="cardBody">
        {!d ? (
          <div className="mono muted">No data yet.</div>
        ) : (d as any).error ? (
          <div className="errorBox mono">{String((d as any).message || (d as any).error)}</div>
        ) : (
          <div className="system-monitor">
            <UsageBar label="CPU" value={Number(d.cpu_percent || 0)} />
            <UsageBar label="RAM" value={Number(d.memory_percent || 0)} right={ramRight} />
            <UsageBar label="Process CPU" value={Number(d.process_cpu_percent || 0)} />
            <div className="row" style={{ justifyContent: "space-between", marginTop: 8 }}>
              <div className="mono muted">{gpuLabel}</div>
              <div className="mono muted">proc mem: {(d.process_memory_mb || 0).toFixed(0)} MB</div>
            </div>
            {d.gpu?.note ? <div className="mono muted" style={{ marginTop: 6 }}>{String(d.gpu.note)}</div> : null}
          </div>
        )}
      </div>
    </section>
  );
}

