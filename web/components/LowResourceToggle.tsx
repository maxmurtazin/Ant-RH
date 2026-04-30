"use client";

import { useEffect } from "react";

const KEY = "ant_low_resource_mode";

function readStored(): boolean {
  try {
    const v = window.localStorage.getItem(KEY);
    if (v === null) return false;
    return String(v).trim().toLowerCase() === "true";
  } catch {
    return false;
  }
}

function writeStored(enabled: boolean) {
  try {
    window.localStorage.setItem(KEY, enabled ? "true" : "false");
  } catch {
    // ignore
  }
}

export function LowResourceToggle(props: { enabled: boolean; onChange: (enabled: boolean) => void }) {
  useEffect(() => {
    const stored = readStored();
    if (stored !== props.enabled) props.onChange(stored);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const on = Boolean(props.enabled);
  return (
    <div className={`low-resource-toggle ${on ? "on" : ""}`}>
      <button
        className="btn"
        onClick={() => {
          const next = !on;
          writeStored(next);
          props.onChange(next);
        }}
      >
        Low Resource Mode: {on ? "ON" : "OFF"}
      </button>
      {on ? <span className="low-resource-badge mono">Low load</span> : null}
    </div>
  );
}

