"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { ActionPanel } from "@/components/ActionPanel";
import { AcoMetrics } from "@/components/AcoMetrics";
import { AcoLiveCharts } from "@/components/AcoLiveCharts";
import { GemmaHealth } from "@/components/GemmaHealth";
import { GemmaHelp } from "@/components/GemmaHelp";
import { PhysicsDiagnostics } from "@/components/PhysicsDiagnostics";
import { TopologicalLmMetrics } from "@/components/TopologicalLmMetrics";
import { StatusBadge } from "@/components/StatusBadge";
import type { AcoHistoryPoint, AcoMetricsResponse, GemmaHealthResponse, PhysicsMetricsResponse, StatusResponse, TopoMetricsResponse } from "@/lib/api";
import { getAcoHistory, getAcoMetrics, getGemmaHealth, getPhysicsMetrics, getStatus, getTopoMetrics } from "@/lib/api";

type EndpointKey = "status" | "aco" | "aco-history" | "topological-lm" | "physics" | "gemma-health";

function fmtTime(d: Date) {
  return d.toLocaleTimeString(undefined, { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export default function Page() {
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<string>("—");
  const [endpointErrors, setEndpointErrors] = useState<Partial<Record<EndpointKey, string>>>({});

  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [aco, setAco] = useState<AcoMetricsResponse | null>(null);
  const [topo, setTopo] = useState<TopoMetricsResponse | null>(null);
  const [physics, setPhysics] = useState<PhysicsMetricsResponse | null>(null);
  const [gemmaHealth, setGemmaHealth] = useState<GemmaHealthResponse | null>(null);
  const [acoHistory, setAcoHistory] = useState<AcoHistoryPoint[]>([]);

  const refreshInFlightRef = useRef(false);

  const summary = useMemo(() => {
    if (!status) return "—";
    const parts = [
      `gemma_learning=${String(status.gemma_learning ?? "—")}`,
      `main_issue=${String(status.gemma_main_issue ?? "—")}`,
      `operator_total_loss=${String(status.operator_total_loss ?? "—")}`,
      `topo_advantage=${String(status.topo_advantage_over_random ?? "—")}`
    ];
    return parts.join("\n");
  }, [status]);

  async function refreshAll() {
    if (refreshInFlightRef.current) return;
    refreshInFlightRef.current = true;
    setIsRefreshing(true);

    const nextErrors: Partial<Record<EndpointKey, string>> = {};
    let anySuccess = false;

    try {
      const results = await Promise.allSettled([
        getStatus(),
        getAcoMetrics(),
        getAcoHistory(300),
        getTopoMetrics(),
        getPhysicsMetrics(),
        getGemmaHealth()
      ]);

      const [st, a, ah, t, p, gh] = results;

      if (st.status === "fulfilled") {
        anySuccess = true;
        setStatus(st.value);
      } else {
        nextErrors["status"] = st.reason instanceof Error ? st.reason.message : String(st.reason);
      }

      if (a.status === "fulfilled") {
        anySuccess = true;
        setAco(a.value);
      } else {
        nextErrors["aco"] = a.reason instanceof Error ? a.reason.message : String(a.reason);
      }

      if (ah.status === "fulfilled") {
        anySuccess = true;
        const pts = Array.isArray(ah.value?.points) ? ah.value.points : [];
        setAcoHistory(pts);
      } else {
        nextErrors["aco-history"] = ah.reason instanceof Error ? ah.reason.message : String(ah.reason);
      }

      if (t.status === "fulfilled") {
        anySuccess = true;
        setTopo(t.value);
      } else {
        nextErrors["topological-lm"] = t.reason instanceof Error ? t.reason.message : String(t.reason);
      }

      if (p.status === "fulfilled") {
        anySuccess = true;
        setPhysics(p.value);
      } else {
        nextErrors["physics"] = p.reason instanceof Error ? p.reason.message : String(p.reason);
      }

      if (gh.status === "fulfilled") {
        anySuccess = true;
        setGemmaHealth(gh.value);
      } else {
        nextErrors["gemma-health"] = gh.reason instanceof Error ? gh.reason.message : String(gh.reason);
      }

      setEndpointErrors(nextErrors);
      if (anySuccess) setLastUpdated(fmtTime(new Date()));
    } finally {
      setIsRefreshing(false);
      refreshInFlightRef.current = false;
    }
  }

  useEffect(() => {
    refreshAll();
    const id = window.setInterval(refreshAll, 5_000);
    return () => window.clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const endpointBadges = useMemo(() => {
    const mk = (key: EndpointKey, hasData: boolean) => {
      const err = endpointErrors[key];
      if (err) return <StatusBadge key={key} tone="broken" label={`${key} error`} />;
      if (hasData) return <StatusBadge key={key} tone="ok" label={`${key} ok`} />;
      return <StatusBadge key={key} tone="unknown" label={`${key} —`} />;
    };
    return [
      mk("status", Boolean(status)),
      mk("aco", Boolean(aco)),
      mk("aco-history", Boolean(acoHistory.length)),
      mk("topological-lm", Boolean(topo)),
      mk("physics", Boolean(physics)),
      mk("gemma-health", Boolean(gemmaHealth))
    ];
  }, [endpointErrors, status, aco, acoHistory.length, topo, physics, gemmaHealth]);

  return (
    <div className="grid">
      <section className="card span2">
        <div className="cardHeader">
          <div className="cardTitle">Project Status</div>
          <div className="cardHeaderRight">
            <span className="badge mono">Live: ON</span>
            <span className="badge mono">Last updated: {lastUpdated}</span>
            {isRefreshing ? <span className="spinner" aria-label="Refreshing" /> : null}
            <button className="btn btnPrimary" onClick={refreshAll} disabled={isRefreshing}>
              Refresh
            </button>
          </div>
        </div>
        <div className="cardBody">
          <div className="row">{endpointBadges}</div>
          {Object.entries(endpointErrors).length ? (
            <div className="errorBox mono">
              {Object.entries(endpointErrors).map(([k, v]) => (
                <div key={k}>
                  <strong>{k}</strong>: {v}
                </div>
              ))}
            </div>
          ) : null}
          <div className="twoCol">
            <div>
              <div className="label">Current state</div>
              <pre className="mono block">{summary}</pre>
            </div>
            <div>
              <div className="label">Missing artifacts</div>
              <pre className="mono block">{status?.missing?.length ? status.missing.join("\n") : "none"}</pre>
            </div>
          </div>
          <div className="divider" />
          <div className="mono muted">Auto-refresh every 5s</div>
        </div>
      </section>

      <AcoMetrics data={aco} />
      <AcoLiveCharts points={acoHistory} />
      <TopologicalLmMetrics data={topo} />
      <PhysicsDiagnostics data={physics} />
      <GemmaHealth data={gemmaHealth} />

      <section className="card span2">
        <div className="cardHeader">
          <div className="cardTitle">Actions</div>
          <div className="hint">Whitelisted actions only</div>
        </div>
        <div className="cardBody">
          <ActionPanel onAfterAction={refreshAll} />
        </div>
      </section>

      <section className="card span2">
        <div className="cardHeader">
          <div className="cardTitle">Gemma Help</div>
          <div className="hint">Local model; may be slow</div>
        </div>
        <div className="cardBody">
          <GemmaHelp />
        </div>
      </section>
    </div>
  );
}

