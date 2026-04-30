import { MetricCard } from "@/components/MetricCard";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts";

export function OperatorCharts(props: { operatorDistanceMean: number | null; lossStd: number | null }) {
  const data = [
    { k: "operator_distance_mean", v: typeof props.operatorDistanceMean === "number" ? props.operatorDistanceMean : null },
    { k: "loss_std", v: typeof props.lossStd === "number" ? props.lossStd : null }
  ].filter((x) => x.v !== null);

  return (
    <MetricCard title="Operator Sensitivity (signal)" span2>
      {data.length === 0 ? (
        <div className="mono muted">No data yet</div>
      ) : (
        <div style={{ height: 220 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 6, right: 8, bottom: 10, left: 0 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
              <XAxis dataKey="k" tick={{ fill: "rgba(255,255,255,0.62)", fontSize: 11 }} stroke="rgba(255,255,255,0.12)" />
              <YAxis tick={{ fill: "rgba(255,255,255,0.62)", fontSize: 12 }} stroke="rgba(255,255,255,0.12)" width={56} />
              <Tooltip
                contentStyle={{ background: "rgba(12,18,26,0.92)", border: "1px solid rgba(255,255,255,0.10)" }}
                labelStyle={{ color: "rgba(255,255,255,0.82)" }}
                itemStyle={{ color: "rgba(255,255,255,0.82)" }}
              />
              <Bar dataKey="v" fill="#60a5fa" isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </MetricCard>
  );
}

