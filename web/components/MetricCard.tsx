import type { ReactNode } from "react";

export function MetricCard(props: {
  title: string;
  right?: ReactNode;
  children: ReactNode;
  span2?: boolean;
  className?: string;
  id?: string;
}) {
  return (
    <section
      id={props.id}
      className={["card", props.span2 ? "span2" : "", props.className || ""].filter(Boolean).join(" ")}
    >
      <div className="cardHeader">
        <div className="cardTitle">{props.title}</div>
        <div className="cardHeaderRight">{props.right}</div>
      </div>
      <div className="cardBody">{props.children}</div>
    </section>
  );
}

