import type { ReactNode } from "react";

export function MetricCard(props: {
  title: string;
  right?: ReactNode;
  children: ReactNode;
  span2?: boolean;
}) {
  return (
    <section className={`card ${props.span2 ? "span2" : ""}`}>
      <div className="cardHeader">
        <div className="cardTitle">{props.title}</div>
        <div className="cardHeaderRight">{props.right}</div>
      </div>
      <div className="cardBody">{props.children}</div>
    </section>
  );
}

