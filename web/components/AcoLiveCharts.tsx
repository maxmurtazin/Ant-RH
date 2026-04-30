import { MetricCard } from "@/components/MetricCard";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

export default function AcoLiveCharts({ points }: { points: any[] }) {
  const safePoints = Array.isArray(points) ? points : [];
  const hasReward = safePoints.some((p) => typeof p?.best_reward === "number" || typeof p?.mean_reward === "number");

  return (
    <MetricCard title="ACO Live Charts" span2>
      <div className="mono muted" style={{ marginBottom: 8 }}>
        ACO points: {safePoints.length}
      </div>

      {safePoints.length === 0 ? (
        <div className="mono muted">No ACO history yet</div>
      ) : (
        <div style={{ display: "grid", gap: 12 }}>
          <div style={{ height: 220 }}>
            <div className="label">loss</div>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={safePoints} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
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

          {hasReward ? (
            <div style={{ height: 220 }}>
              <div className="label">reward</div>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={safePoints} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
                  <XAxis dataKey="iter" tick={{ fill: "rgba(255,255,255,0.62)", fontSize: 12 }} stroke="rgba(255,255,255,0.12)" />
                  <YAxis tick={{ fill: "rgba(255,255,255,0.62)", fontSize: 12 }} stroke="rgba(255,255,255,0.12)" width={56} />
                  <Tooltip
                    labelFormatter={(l) => `iter ${l}`}
                    contentStyle={{ background: "rgba(12,18,26,0.92)", border: "1px solid rgba(255,255,255,0.10)" }}
                    labelStyle={{ color: "rgba(255,255,255,0.82)" }}
                    itemStyle={{ color: "rgba(255,255,255,0.82)" }}
                  />
                  <Line type="monotone" dataKey="best_reward" dot={false} stroke="#34d399" strokeWidth={2} isAnimationActive={false} />
                  <Line type="monotone" dataKey="mean_reward" dot={false} stroke="#fbbf24" strokeWidth={2} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : null}
        </div>
      )}
    </MetricCard>
  );
}

