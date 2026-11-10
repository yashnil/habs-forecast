import { useId, type ReactNode } from "react";
import { cx, svgId } from "./util";
import s from "./Statement.module.css";

export type StatementSize = "xl" | "lg" | "md";
export type StatementTone = "ink" | "regulation" | "closure";

export interface StatementProps {
  /** The statement itself: one sentence a stranger can read from across the room. */
  title: ReactNode;
  size?: StatementSize;
  /** Regulation (magenta) is only for official delay and closure statements; closure is rust, official closures only. */
  tone?: StatementTone;
  as?: "h1" | "h2" | "h3" | "p";
  /** The line above: the port and fishery the statement is about. */
  scope?: ReactNode;
  /** A serif line under the title: the reason, in the agency's terms. */
  why?: ReactNode;
  /** The chart's regulation line under the title. Only with tone "regulation". */
  rule?: boolean;
  /** Status sentence(s) and the stamp. */
  children?: ReactNode;
  className?: string;
  id?: string;
}

/** The chart's T-dash regulation line (a restricted-area boundary), drawn in --reg. */
export function RegulationRule({ className }: { className?: string }) {
  const id = svgId(useId());
  return (
    <svg className={cx(s.rule, className)} height="8" width="100%" aria-hidden focusable="false">
      <defs>
        <pattern id={id} width="22" height="8" patternUnits="userSpaceOnUse">
          <rect x="0" y="0" width="16" height="2" />
          <rect x="7" y="2" width="2" height="5" />
        </pattern>
      </defs>
      <rect width="100%" height="8" fill={`url(#${id})`} />
    </svg>
  );
}

/** Page title plus status sentence, in Literata (spec section 9). */
export function Statement({ title, size = "xl", tone = "ink", as: As = "h1", scope, why, rule, children, className, id }: StatementProps) {
  return (
    <div className={cx(s.statement, s[size], className)} data-tone={tone}>
      {scope ? <p className={s.scope}>{scope}</p> : null}
      <As className={s.title} id={id}>
        {title}
      </As>
      {why ? <p className={s.why}>{why}</p> : null}
      {rule && tone === "regulation" ? <RegulationRule /> : null}
      {children ? <div className={s.body}>{children}</div> : null}
    </div>
  );
}
