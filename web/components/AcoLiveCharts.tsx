import { MetricCard } from "@/components/MetricCard";
import { StatusBadge } from "@/components/StatusBadge";
import type { AcoHistoryPoint } from "@/lib/api";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

type NumOrNull = number | null;

function hasAny(points: AcoHistoryPoint[], keys: (keyof AcoHistoryPoint)[]) {
  return points.some((p) => keys.some((k) => typeof p[k] === "number" && Number.isFinite(p[k] as unknown as number)));
}

function compactNumber(x: unknown) {
  if (typeof x !== "number" || !Number.isFinite(x)) return "";
  const abs = Math.abs(x);
  if (abs >= 1e9) return `${(x / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${(x / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${(x / 1e3).toFixed(2)}K`;
  if (abs === 0) return "0";
  if (abs < 1e-3) return x.toExponential(2);
  if (abs < 1) return x.toFixed(4);
  return x.toFixed(3);
}

function yTick(v: NumOrNull) {
  return compactNumber(v);
}

function tooltipFormatter(value: unknown) {
  if (typeof value !== "number" || !Number.isFinite(value)) return ["—", ""];
  return [compactNumber(value), ""];
}

export function AcoLiveCharts(props: { points: AcoHistoryPoint[]; rewardMode?: string | null }) {
  const points = props.points || [];
  const hasLoss = hasAny(points, ["best_loss", "mean_loss"]);
  const hasReward = hasAny(points, ["best_reward", "mean_reward"]);

  const right = (
    <>
      {props.rewardMode ? <span className="badge mono">{props.rewardMode}</span> : null}
      <StatusBadge tone={points.length ? "ok" : "unknown"} label={points.length ? "LIVE" : "NO DATA"} />
    </>
  );

  return (
    <MetricCard title="ACO Live Charts" right={right} span2>
      {!points.length ? (
        <div className="mono muted">No ACO history yet</div>
      ) : (
        <div style={{ display: "grid", gap: 12 }}>
          {hasLoss ? (
            <div style={{ height: 220 }}>
              <div className="label">loss</div>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={points} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
                  <XAxis dataKey="iter" tick={{ fill: "rgba(255,255,255,0.62)", fontSize: 12 }} stroke="rgba(255,255,255,0.12)" />
                  <YAxis
                    tickFormatter={yTick}
                    tick={{ fill: "rgba(255,255,255,0.62)", fontSize: 12 }}
                    stroke="rgba(255,255,255,0.12)"
                    width={56}
                  />
                  <Tooltip
                    formatter={tooltipFormatter}
                    labelFormatter={(l) => `iter ${l}`}
                    contentStyle={{ background: "rgba(12,18,26,0.92)", border: "1px solid rgba(255,255,255,0.10)" }}
                    labelStyle={{ color: "rgba(255,255,255,0.82)" }}
                    itemStyle={{ color: "rgba(255,255,255,0.82)" }}
                  />
                  <Legend wrapperStyle={{ color: "rgba(255,255,255,0.62)" }} />
                  <Line type="monotone" dataKey="best_loss" name="best_loss" dot={false} stroke="#5eead4" strokeWidth={2} isAnimationActive={false} />
                  <Line type="monotone" dataKey="mean_loss" name="mean_loss" dot={false} stroke="#60a5fa" strokeWidth={2} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="mono muted">No loss columns found.</div>
          )}

          {hasReward ? (
            <div style={{ height: 220 }}>
              <div className="label">reward</div>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={points} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
                  <XAxis dataKey="iter" tick={{ fill: "rgba(255,255,255,0.62)", fontSize: 12 }} stroke="rgba(255,255,255,0.12)" />
                  <YAxis
                    tickFormatter={yTick}
                    tick={{ fill: "rgba(255,255,255,0.62)", fontSize: 12 }}
                    stroke="rgba(255,255,255,0.12)"
                    width={56}
                  />
                  <Tooltip
                    formatter={tooltipFormatter}
                    labelFormatter={(l) => `iter ${l}`}
                    contentStyle={{ background: "rgba(12,18,26,0.92)", border: "1px solid rgba(255,255,255,0.10)" }}
                    labelStyle={{ color: "rgba(255,255,255,0.82)" }}
                    itemStyle={{ color: "rgba(255,255,255,0.82)" }}
                  />
                  <Legend wrapperStyle={{ color: "rgba(255,255,255,0.62)" }} />
                  <Line type="monotone" dataKey="best_reward" name="best_reward" dot={false} stroke="#34d399" strokeWidth={2} isAnimationActive={false} />
                  <Line type="monotone" dataKey="mean_reward" name="mean_reward" dot={false} stroke="#fbbf24" strokeWidth={2} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : null}
        </div>
      )}
    </MetricCard>
  );
}

