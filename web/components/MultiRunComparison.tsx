"use client";

import { MetricCard } from "@/components/MetricCard";
import type { RunCompareRow } from "@/lib/api";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useMemo } from "react";

function fmt(x: any) {
  if (typeof x !== "number" || !Number.isFinite(x)) return "—";
  const ax = Math.abs(x);
  if (ax > 1e6 || (ax > 0 && ax < 1e-3)) return x.toExponential(3);
  return x.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function bestIndex(rows: RunCompareRow[], key: keyof RunCompareRow, mode: "min" | "max" | "absmin" | "target") {
  let best = -1;
  let bestScore: number | null = null;
  for (let i = 0; i < rows.length; i++) {
    const v = (rows[i] as any)[key];
    if (typeof v !== "number" || !Number.isFinite(v)) continue;
    let score = v;
    if (mode === "min") score = v;
    if (mode === "max") score = -v;
    if (mode === "absmin") score = Math.abs(v);
    if (mode === "target") score = Math.abs(v - 0.535);
    if (bestScore === null || score < bestScore) {
      bestScore = score;
      best = i;
    }
  }
  return best;
}

export function MultiRunComparison(props: { runs: RunCompareRow[] }) {
  const rows = Array.isArray(props.runs) ? props.runs : [];

  const best = useMemo(() => {
    return {
      aco_best_loss: bestIndex(rows, "aco_best_loss", "min"),
      topo_reward_mean: bestIndex(rows, "topo_reward_mean", "max"),
      self_adjoint_error: bestIndex(rows, "self_adjoint_error", "absmin"),
      r_mean: bestIndex(rows, "r_mean", "target"),
    };
  }, [rows]);

  const chartData = useMemo(() => {
    return rows.map((r) => ({
      id: r.id,
      label: (r.label || r.id).slice(0, 18),
      aco_best_loss: r.aco_best_loss,
      topo_reward_mean: r.topo_reward_mean,
      r_mean: r.r_mean,
      self_adjoint_error: r.self_adjoint_error,
    }));
  }, [rows]);

  return (
    <MetricCard title="Multi-run comparison" className="result-card span-full" id="multi-run-comparison">
        {!rows.length ? (
          <div className="mono muted">No runs yet.</div>
        ) : (
          <>
            <div className="multiRunGrid">
              {[
                { key: "aco_best_loss", label: "ACO best loss (lower better)" },
                { key: "topo_reward_mean", label: "Topo reward mean (higher better)" },
                { key: "r_mean", label: "r_mean (closer to 0.535 better)" },
                { key: "self_adjoint_error", label: "self_adjoint_error (closer to 0 better)" },
              ].map((c) => (
                <div key={c.key} className="miniChart">
                  <div className="mono muted" style={{ marginBottom: 6 }}>
                    {c.label}
                  </div>
                  <div style={{ height: 120 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
                        <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
                        <XAxis dataKey="label" tick={{ fill: "rgba(255,255,255,0.62)", fontSize: 10 }} stroke="rgba(255,255,255,0.12)" interval={0} />
                        <YAxis tick={{ fill: "rgba(255,255,255,0.62)", fontSize: 10 }} stroke="rgba(255,255,255,0.12)" width={44} />
                        <Tooltip
                          contentStyle={{ background: "rgba(12,18,26,0.92)", border: "1px solid rgba(255,255,255,0.10)" }}
                          labelStyle={{ color: "rgba(255,255,255,0.82)" }}
                          itemStyle={{ color: "rgba(255,255,255,0.82)" }}
                        />
                        <Bar dataKey={c.key} fill="rgba(96,165,250,0.72)" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              ))}
            </div>

            <div className="divider" />
            <div className="multiRunTableWrap">
              <div className="table-wrap">
                <table className="table mono compactTable multiRunTable">
                  <thead>
                    <tr>
                      <th>run</th>
                      <th>timestamp</th>
                      <th>aco_best_loss</th>
                      <th>aco_mean_loss</th>
                      <th>topo_reward_mean</th>
                      <th>advantage</th>
                      <th>self_adjoint_error</th>
                      <th>r_mean</th>
                      <th>operator_sensitivity</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((r, i) => (
                      <tr key={r.id} className={r.source === "current" ? "multiRunCurrent" : ""}>
                        <td>{r.label || r.id}</td>
                        <td className="muted">{(r.timestamp || "—").slice(0, 19)}</td>
                        <td className={i === best.aco_best_loss ? "bestCell" : ""}>{fmt(r.aco_best_loss)}</td>
                        <td>{fmt(r.aco_mean_loss)}</td>
                        <td className={i === best.topo_reward_mean ? "bestCell" : ""}>{fmt(r.topo_reward_mean)}</td>
                        <td>{fmt(r.topo_advantage_over_random)}</td>
                        <td className={i === best.self_adjoint_error ? "bestCell" : ""}>{fmt(r.self_adjoint_error)}</td>
                        <td className={i === best.r_mean ? "bestCell" : ""}>{fmt(r.r_mean)}</td>
                        <td>{fmt(r.operator_sensitivity)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
    </MetricCard>
  );
}

