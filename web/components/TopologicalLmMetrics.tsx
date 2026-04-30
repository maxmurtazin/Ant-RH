import { MetricCard } from "@/components/MetricCard";
import { StatusBadge } from "@/components/StatusBadge";
import type { TopoMetricsResponse } from "@/lib/api";

function toneFromAdvantage(adv?: number | null) {
  if (typeof adv !== "number") return { tone: "unknown" as const, label: "UNKNOWN" };
  if (adv > 0) return { tone: "ok" as const, label: "OK" };
  return { tone: "warn" as const, label: "WARNING" };
}

export function TopologicalLmMetrics({ data }: { data: TopoMetricsResponse | null }) {
  const badge = toneFromAdvantage(data?.advantage_over_random ?? null);
  return (
    <MetricCard title="Topological LLM" right={<StatusBadge tone={badge.tone} label={badge.label} />}>
      <table className="table mono">
        <tbody>
          <tr>
            <td className="k">random mean_reward</td>
            <td className="v">{data?.random_mean_reward ?? "—"}</td>
          </tr>
          <tr>
            <td className="k">TopologicalLM mean_reward</td>
            <td className="v">{data?.topological_lm_mean_reward ?? "—"}</td>
          </tr>
          <tr>
            <td className="k">advantage_over_random</td>
            <td className="v">{data?.advantage_over_random ?? "—"}</td>
          </tr>
          <tr>
            <td className="k">unique_candidate_ratio</td>
            <td className="v">{data?.unique_candidate_ratio ?? "—"}</td>
          </tr>
          <tr>
            <td className="k">valid_braid_ratio</td>
            <td className="v">{data?.valid_braid_ratio ?? "—"}</td>
          </tr>
        </tbody>
      </table>
    </MetricCard>
  );
}

