import { MetricCard } from "@/components/MetricCard";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts";

export function AcoCharts({ points }: { points: any[] }) {
  const safe = Array.isArray(points) ? points : [];

  return (
    <MetricCard title="ACO Loss (history)" span2>
      {safe.length === 0 ? (
        <div className="mono muted">No data yet</div>
      ) : (
        <div style={{ height: 220 }}>
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
              <Line type="monotone" dataKey="best_loss" dot={false} stroke="#5eead4" strokeWidth={2} isAnimationActive={false} />
              <Line type="monotone" dataKey="mean_loss" dot={false} stroke="#60a5fa" strokeWidth={2} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </MetricCard>
  );
}

