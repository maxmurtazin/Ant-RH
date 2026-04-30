import { MetricCard } from "@/components/MetricCard";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts";

function buildHistogram(samples: number[], bins = 18) {
  const xs = samples.filter((x) => typeof x === "number" && Number.isFinite(x));
  if (xs.length === 0) return [];
  const lo = Math.min(...xs);
  const hi = Math.max(...xs);
  if (lo === hi) return [{ bin: `${lo.toFixed(3)}`, count: xs.length }];

  const step = (hi - lo) / bins;
  const counts = new Array(bins).fill(0);
  for (const x of xs) {
    const idx = Math.min(bins - 1, Math.max(0, Math.floor((x - lo) / step)));
    counts[idx] += 1;
  }
  return counts.map((c, i) => {
    const a = lo + i * step;
    const b = a + step;
    return { bin: `${a.toFixed(2)}–${b.toFixed(2)}`, count: c };
  });
}

export function RewardHistogram({ samples }: { samples: number[] }) {
  const safe = Array.isArray(samples) ? samples : [];
  const data = buildHistogram(safe, 18);

  return (
    <MetricCard title="TopologicalLM Reward Histogram" className="result-card span-1">
      {data.length === 0 ? (
        <div className="mono muted">No data yet</div>
      ) : (
        <div className="chart-shell">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 6, right: 8, bottom: 24, left: 0 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
              <XAxis
                dataKey="bin"
                interval={2}
                angle={-20}
                textAnchor="end"
                height={52}
                tick={{ fill: "rgba(255,255,255,0.62)", fontSize: 11 }}
                stroke="rgba(255,255,255,0.12)"
              />
              <YAxis tick={{ fill: "rgba(255,255,255,0.62)", fontSize: 12 }} stroke="rgba(255,255,255,0.12)" width={48} />
              <Tooltip
                contentStyle={{ background: "rgba(12,18,26,0.92)", border: "1px solid rgba(255,255,255,0.10)" }}
                labelStyle={{ color: "rgba(255,255,255,0.82)" }}
                itemStyle={{ color: "rgba(255,255,255,0.82)" }}
              />
              <Bar dataKey="count" fill="#fbbf24" isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </MetricCard>
  );
}

