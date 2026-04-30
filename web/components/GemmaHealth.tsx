import { MetricCard } from "@/components/MetricCard";
import { StatusBadge } from "@/components/StatusBadge";
import type { GemmaHealthResponse } from "@/lib/api";

function badgeFromOverall(overall?: string) {
  if (overall === "ok") return { tone: "ok" as const, label: "OK" };
  if (overall === "degraded") return { tone: "warn" as const, label: "WARNING" };
  if (overall === "failed") return { tone: "broken" as const, label: "BROKEN" };
  return { tone: "unknown" as const, label: "UNKNOWN" };
}

export function GemmaHealth({ data }: { data: GemmaHealthResponse | null }) {
  const overall =
    data && "overall_status" in data ? data.overall_status : data && "status" in data ? data.status : "unknown";
  const badge = badgeFromOverall(overall);
  const checks = data && "checks" in data ? data.checks : [];
  const byName = new Map(checks.map((c) => [c.name, c]));
  const s = (name: string) => byName.get(name)?.status ?? "—";

  return (
    <MetricCard title="Gemma Health" right={<StatusBadge tone={badge.tone} label={badge.label} />} className="compact-panel">
      <table className="table mono">
        <tbody>
          <tr>
            <td className="k">overall</td>
            <td className="v">{overall}</td>
          </tr>
          <tr>
            <td className="k">planner</td>
            <td className="v">{s("planner")}</td>
          </tr>
          <tr>
            <td className="k">analyzer</td>
            <td className="v">{s("analyzer")}</td>
          </tr>
          <tr>
            <td className="k">help</td>
            <td className="v">{s("help")}</td>
          </tr>
          <tr>
            <td className="k">journal</td>
            <td className="v">{s("lab_journal")}</td>
          </tr>
          <tr>
            <td className="k">docs</td>
            <td className="v">{s("docs_builder")}</td>
          </tr>
        </tbody>
      </table>
    </MetricCard>
  );
}

