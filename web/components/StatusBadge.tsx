export type BadgeTone = "ok" | "warn" | "broken" | "unknown";

export function StatusBadge(props: { label: string; tone: BadgeTone }) {
  const cls =
    props.tone === "ok"
      ? "badge badgeOk"
      : props.tone === "warn"
        ? "badge badgeWarn"
        : props.tone === "broken"
          ? "badge badgeBad"
          : "badge";
  return <span className={cls}>{props.label}</span>;
}

