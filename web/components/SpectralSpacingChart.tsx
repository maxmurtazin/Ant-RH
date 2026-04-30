import { MetricCard } from "@/components/MetricCard";
import type { SpectralSpacingResponse } from "@/lib/api";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export function SpectralSpacingChart({ data }: { data: SpectralSpacingResponse | null }) {
  const bins = Array.isArray(data?.hist_bins) ? data!.hist_bins : [];
  const op = Array.isArray(data?.operator_spacing_hist) ? data!.operator_spacing_hist : [];
  const zz = Array.isArray(data?.zeta_spacing_hist) ? data!.zeta_spacing_hist : [];
  const gue = Array.isArray(data?.gue_curve) ? data!.gue_curve : [];
  const pois = Array.isArray(data?.poisson_curve) ? data!.poisson_curve : [];

  const rows =
    bins.length > 0
      ? bins.map((s, i) => ({
          s,
          operator: typeof op[i] === "number" ? op[i] : null,
          zeta: typeof zz[i] === "number" ? zz[i] : null,
          gue: typeof gue[i] === "number" ? gue[i] : null,
          poisson: typeof pois[i] === "number" ? pois[i] : null
        }))
      : [];

  const hasOperator = rows.some((r) => typeof r.operator === "number" && r.operator > 0);
  const hasZeta = rows.some((r) => typeof r.zeta === "number" && r.zeta > 0);

  return (
    <MetricCard
      title="GUE vs Zeta spacing overlay"
      span2
      right={
        <div className="row">
          <span className="badge mono">operator r̄≈{data?.operator_r_mean?.toFixed?.(3) ?? "—"}</span>
          <span className="badge mono">zeta r̄≈{data?.zeta_r_mean?.toFixed?.(3) ?? "—"}</span>
          <span className="badge mono">Poisson≈0.386</span>
          <span className="badge mono">GUE≈0.535</span>
        </div>
      }
    >
      {!hasOperator ? (
        <div className="mono muted">No data yet</div>
      ) : (
        <>
          <div className="mono muted" style={{ marginBottom: 8 }}>
            Source: {data?.source || "—"}
          </div>
          <div style={{ height: 240 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rows} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
                <XAxis
                  dataKey="s"
                  tick={{ fill: "rgba(255,255,255,0.62)", fontSize: 12 }}
                  stroke="rgba(255,255,255,0.12)"
                  label={{ value: "spacing s (normalized)", position: "insideBottom", offset: -2, fill: "rgba(255,255,255,0.48)" }}
                />
                <YAxis tick={{ fill: "rgba(255,255,255,0.62)", fontSize: 12 }} stroke="rgba(255,255,255,0.12)" width={56} />
                <Tooltip
                  contentStyle={{ background: "rgba(12,18,26,0.92)", border: "1px solid rgba(255,255,255,0.10)" }}
                  labelStyle={{ color: "rgba(255,255,255,0.82)" }}
                  itemStyle={{ color: "rgba(255,255,255,0.82)" }}
                />

                <Line
                  type="monotone"
                  dataKey="operator"
                  name="Current operator"
                  dot={false}
                  stroke="#5eead4"
                  strokeWidth={2}
                  isAnimationActive={false}
                />
                {hasZeta ? (
                  <Line type="monotone" dataKey="zeta" name="Zeta zeros" dot={false} stroke="#60a5fa" strokeWidth={2} isAnimationActive={false} />
                ) : null}
                <Line type="monotone" dataKey="gue" name="GUE/Wigner-Dyson" dot={false} stroke="#34d399" strokeWidth={2} isAnimationActive={false} />
                <Line type="monotone" dataKey="poisson" name="Poisson" dot={false} stroke="#fbbf24" strokeWidth={2} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </MetricCard>
  );
}

