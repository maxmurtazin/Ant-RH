import { MetricCard } from "@/components/MetricCard";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts";

export function PhysicsCharts({ points }: { points: any[] }) {
  const safe = Array.isArray(points) ? points : [];

  return (
    <MetricCard title="Physics Diagnostics (history)" span2>
      {safe.length === 0 ? (
        <div className="mono muted">No data yet</div>
      ) : (
        <div style={{ display: "grid", gap: 12 }}>
          <div style={{ height: 200 }}>
            <div className="label">r_mean over time</div>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={safe} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
                <XAxis dataKey="iter" tick={{ fill: "rgba(255,255,255,0.62)", fontSize: 12 }} stroke="rgba(255,255,255,0.12)" />
                <YAxis tick={{ fill: "rgba(255,255,255,0.62)", fontSize: 12 }} stroke="rgba(255,255,255,0.12)" width={56} />
                <Tooltip
                  labelFormatter={(l) => `iter ${l}`}
                  contentStyle={{ background: "rgba(12,18,26,0.92)", border: "1px solid rgba(255,255,255,0.10)" }}
                  labelStyle={{ color: "rgba(255,255,255,0.82)" }}
                  itemStyle={{ color: "rgba(255,255,255,0.82)" }}
                />
                <Line type="monotone" dataKey="r_mean" dot={false} stroke="#34d399" strokeWidth={2} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div style={{ height: 200 }}>
            <div className="label">self_adjoint_error over time</div>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={safe} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
                <XAxis dataKey="iter" tick={{ fill: "rgba(255,255,255,0.62)", fontSize: 12 }} stroke="rgba(255,255,255,0.12)" />
                <YAxis tick={{ fill: "rgba(255,255,255,0.62)", fontSize: 12 }} stroke="rgba(255,255,255,0.12)" width={56} />
                <Tooltip
                  labelFormatter={(l) => `iter ${l}`}
                  contentStyle={{ background: "rgba(12,18,26,0.92)", border: "1px solid rgba(255,255,255,0.10)" }}
                  labelStyle={{ color: "rgba(255,255,255,0.82)" }}
                  itemStyle={{ color: "rgba(255,255,255,0.82)" }}
                />
                <Line type="monotone" dataKey="self_adjoint_error" dot={false} stroke="#fb7185" strokeWidth={2} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </MetricCard>
  );
}

