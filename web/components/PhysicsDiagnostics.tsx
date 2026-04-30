import { MetricCard } from "@/components/MetricCard";
import { StatusBadge } from "@/components/StatusBadge";
import type { PhysicsMetricsResponse } from "@/lib/api";

function toneFromStatus(status?: string | null) {
  const s = (status || "").toLowerCase();
  if (s === "ok" || s === "chaotic") return { tone: "ok" as const, label: "OK" };
  if (s === "approx" || s === "intermediate" || s === "degenerate") return { tone: "warn" as const, label: "WARNING" };
  if (s === "broken") return { tone: "broken" as const, label: "BROKEN" };
  return { tone: "unknown" as const, label: "UNKNOWN" };
}

function fmtSci(x: number | null | undefined) {
  if (x === null || x === undefined) return "—";
  if (!Number.isFinite(x)) return "—";
  return x.toExponential(2);
}

export function PhysicsDiagnostics({ data }: { data: PhysicsMetricsResponse | null }) {
  const badge = toneFromStatus(data?.self_adjoint_status ?? null);
  return (
    <MetricCard
      title="Physics Diagnostics"
      right={
        <>
          <span className="badge mono">{data?.source ?? "—"}</span>
          <StatusBadge tone={badge.tone} label={badge.label} />
        </>
      }
    >
      <table className="table mono">
        <tbody>
          <tr>
            <td className="k">self_adjoint_status</td>
            <td className="v">{data?.self_adjoint_status ?? "unknown"}</td>
          </tr>
          <tr>
            <td className="k">self_adjoint_error</td>
            <td className="v">{fmtSci(data?.self_adjoint_error)}</td>
          </tr>
          <tr>
            <td className="k">spectral_status</td>
            <td className="v">{data?.spectral_status ?? "unknown"}</td>
          </tr>
          <tr>
            <td className="k">spectrum_real</td>
            <td className="v">
              {data?.spectrum_real === true ? "true" : data?.spectrum_real === false ? "false" : "—"}
            </td>
          </tr>
          <tr>
            <td className="k">spacing_std</td>
            <td className="v">{data?.spacing_std ?? "—"}</td>
          </tr>
          <tr>
            <td className="k">otoc_indicator</td>
            <td className="v">{data?.otoc_indicator ?? "unknown"}</td>
          </tr>
          <tr>
            <td className="k">r_mean</td>
            <td className="v">{data?.r_mean ?? "—"}</td>
          </tr>
        </tbody>
      </table>
    </MetricCard>
  );
}

