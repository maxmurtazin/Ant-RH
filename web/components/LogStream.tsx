"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { API_BASE } from "@/lib/api";

type StreamEvent =
  | { line: string }
  | { status: string; returncode?: number; error?: string }
  | { error: string };

export function LogStream(props: {
  jobId: string | null;
  enabled?: boolean;
  disabledReason?: string;
  onEnable?: () => void;
}) {
  const [lines, setLines] = useState<string[]>([]);
  const [status, setStatus] = useState<string>("—");
  const [error, setError] = useState<string>("");

  const boxRef = useRef<HTMLDivElement | null>(null);

  const streamUrl = useMemo(() => {
    if (!props.jobId) return null;
    return `${API_BASE}/jobs/${encodeURIComponent(props.jobId)}/stream`;
  }, [props.jobId]);

  useEffect(() => {
    setLines([]);
    setStatus("—");
    setError("");
    if (!streamUrl) return;
    if (props.enabled === false) return;

    const es = new EventSource(streamUrl);
    setStatus("connecting");

    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data) as StreamEvent;
        if (data && typeof data === "object" && "line" in data && typeof (data as any).line === "string") {
          const line = (data as any).line as string;
          setLines((prev) => {
            const next = prev.length > 2500 ? prev.slice(-2000) : prev.slice();
            next.push(line);
            return next;
          });
          return;
        }
        if (data && typeof data === "object" && "status" in data) {
          const st = String((data as any).status || "—");
          setStatus(st);
          const err = (data as any).error;
          if (st === "failed" && String(err || "") === "timeout") {
            setError("Health check timed out. See logs.");
          }
          return;
        }
        if (data && typeof data === "object" && "error" in data) {
          setError(String((data as any).error || "stream error"));
          return;
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    };

    es.onerror = () => {
      // Browser will auto-retry; keep visible state.
      setStatus("disconnected (retrying)");
    };

    return () => es.close();
  }, [streamUrl, props.enabled]);

  useEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [lines.length]);

  if (!props.jobId) {
    return (
      <section className="card span2">
        <div className="cardHeader">
          <div className="cardTitle">Logs</div>
          <div className="hint">Start a job to view logs</div>
        </div>
        <div className="cardBody">
          <div className="terminalBox mono muted">No active job.</div>
        </div>
      </section>
    );
  }

  if (props.enabled === false) {
    return (
      <section className="card span2">
        <div className="cardHeader">
          <div className="cardTitle">Logs</div>
          <div className="hint">Live stream disabled</div>
        </div>
        <div className="cardBody">
          <div className="low-resource-note mono muted">{props.disabledReason || "Live stream disabled."}</div>
          {props.onEnable ? (
            <div className="action-row" style={{ marginTop: 10 }}>
              <button className="btn" onClick={props.onEnable}>
                Enable logs for this job
              </button>
            </div>
          ) : null}
        </div>
      </section>
    );
  }

  return (
    <section className="card span2">
      <div className="cardHeader">
        <div className="cardTitle">Logs</div>
        <div className="cardHeaderRight">
          <span className="badge mono">job: {props.jobId}</span>
          <span className="badge mono">status: {status}</span>
        </div>
      </div>
      <div className="cardBody">
        {error ? <div className="errorBox mono">{error}</div> : null}
        <div ref={boxRef} className="terminalBox mono">
          {lines.length ? (
            lines.map((ln, i) => (
              <div key={i} className="terminalLine">
                {ln}
              </div>
            ))
          ) : (
            <div className="terminalLine muted">Waiting for output…</div>
          )}
        </div>
      </div>
    </section>
  );
}

