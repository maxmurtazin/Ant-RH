"use client";

import { useMemo, useState } from "react";
import { MetricCard } from "@/components/MetricCard";
import type { OperatorAnalysisResponse } from "@/lib/api";

function fmt(x: unknown) {
  if (typeof x !== "number" || !Number.isFinite(x)) return "—";
  const abs = Math.abs(x);
  if (abs >= 1e6 || (abs > 0 && abs < 1e-3)) return x.toExponential(3);
  return x.toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
}

export function OperatorAnalysis(props: {
  data: OperatorAnalysisResponse | null;
  onRunJob?: (job: "pde" | "sensitivity" | "stability", params?: any) => Promise<void> | void;
}) {
  const d = props.data;
  const [_busy, setBusy] = useState<null | "pde" | "sensitivity" | "stability">(null);

  const terms = useMemo(() => (Array.isArray(d?.active_terms) ? d!.active_terms.slice(0, 30) : []), [d]);

  // Primary run controls are intentionally in the Run Controls column.
  // Keep this card mostly read-only.

  const formula = d?.formula_tex || d?.formula_text || null;

  return (
    <MetricCard
      title="Operator Formula / Analysis"
      className="result-card span-2"
      id="operator-analysis"
    >
      <div className="twoCol">
        <div>
          <div className="label">Formula</div>
          {formula ? (
            <pre className="formulaBox mono">{formula}</pre>
          ) : (
            <div className="mono muted">No formula discovered yet. Run make pde.</div>
          )}
        </div>

        <div>
          <div className="label">Stability</div>
          <table className="table mono compactTable">
            <tbody>
              <tr>
                <td className="k">self_adjoint_error</td>
                <td className="v">{fmt(d?.stability?.self_adjoint_error ?? null)}</td>
              </tr>
              <tr>
                <td className="k">eigh_success</td>
                <td className="v">{String(d?.stability?.eigh_success ?? "—")}</td>
              </tr>
              <tr>
                <td className="k">spectral_loss</td>
                <td className="v">{fmt(d?.stability?.spectral_loss ?? null)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="divider" />

      <div className="label">Active PDE terms</div>
      {terms.length ? (
        <table className="table mono compactTable">
          <thead>
            <tr>
              <td className="k">term</td>
              <td className="v">coefficient</td>
              <td className="v">|coefficient|</td>
            </tr>
          </thead>
          <tbody>
            {terms.map((t, i) => (
              <tr key={`${t.term}-${i}`}>
                <td className="k" style={{ width: "60%" }}>
                  {t.term}
                </td>
                <td className="v">{fmt(t.coefficient)}</td>
                <td className="v">{fmt(t.abs_coefficient)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div className="mono muted">No active terms yet.</div>
      )}

      <div className="divider" />

      <div className="twoCol">
        <div>
          <div className="label">Sensitivity</div>
          <table className="table mono compactTable">
            <tbody>
              <tr>
                <td className="k">operator_distance_mean</td>
                <td className="v">{fmt(d?.sensitivity?.operator_distance_mean ?? null)}</td>
              </tr>
              <tr>
                <td className="k">spectrum_distance_mean</td>
                <td className="v">{fmt(d?.sensitivity?.spectrum_distance_mean ?? null)}</td>
              </tr>
              <tr>
                <td className="k">loss_std</td>
                <td className="v">{fmt(d?.sensitivity?.loss_std ?? null)}</td>
              </tr>
              <tr>
                <td className="k">diagnosis</td>
                <td className="v">{d?.sensitivity?.diagnosis ?? "—"}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div>
          <div className="label">Structured Operator</div>
          <table className="table mono compactTable">
            <tbody>
              <tr>
                <td className="k">final_loss</td>
                <td className="v">{fmt(d?.structured_operator?.final_loss ?? null)}</td>
              </tr>
              <tr>
                <td className="k">spectral_loss</td>
                <td className="v">{fmt(d?.structured_operator?.spectral_loss ?? null)}</td>
              </tr>
              <tr>
                <td className="k">spacing_loss</td>
                <td className="v">{fmt(d?.structured_operator?.spacing_loss ?? null)}</td>
              </tr>
            </tbody>
          </table>
          <details>
            <summary className="mono muted" style={{ cursor: "pointer", marginTop: 10 }}>
              top weights
            </summary>
            <pre className="console mono">{d?.structured_operator?.top_weights ? JSON.stringify(d.structured_operator.top_weights, null, 2) : "—"}</pre>
          </details>
        </div>
      </div>

      <details>
        <summary className="mono muted" style={{ cursor: "pointer", marginTop: 10 }}>
          source files
        </summary>
        <pre className="console mono">{d?.source_files?.length ? d.source_files.join("\n") : "—"}</pre>
      </details>
    </MetricCard>
  );
}

