import { MetricCard } from "@/components/MetricCard";
import { StatusBadge } from "@/components/StatusBadge";
import type { AcoMetricsResponse } from "@/lib/api";

function toneFromTrend(trend?: string | null) {
  if (trend === "decreasing") return { tone: "ok" as const, label: "OK" };
  if (trend === "increasing") return { tone: "broken" as const, label: "WARNING" };
  if (trend === "flat") return { tone: "warn" as const, label: "FLAT" };
  return { tone: "unknown" as const, label: "UNKNOWN" };
}

export function AcoMetrics({ data }: { data: AcoMetricsResponse | null }) {
  const badge = toneFromTrend(data?.trend ?? null);
  return (
    <MetricCard title="ACO Metrics" right={<StatusBadge tone={badge.tone} label={badge.label} />}>
      <table className="table mono">
        <tbody>
          <tr>
            <td className="k">best_loss (last)</td>
            <td className="v">{data?.best_loss ?? "—"}</td>
          </tr>
          <tr>
            <td className="k">mean_loss (last)</td>
            <td className="v">{data?.mean_loss ?? "—"}</td>
          </tr>
          <tr>
            <td className="k">trend</td>
            <td className="v">{data?.trend ?? "—"}</td>
          </tr>
          <tr>
            <td className="k">n_rows</td>
            <td className="v">{data?.n_rows ?? "—"}</td>
          </tr>
        </tbody>
      </table>
    </MetricCard>
  );
}

