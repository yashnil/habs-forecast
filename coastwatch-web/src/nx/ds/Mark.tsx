import { MARK_WORDS, NOT_VERIFIED, type MarkKind } from "./kinds";
import { cx } from "./util";
import s from "./Mark.module.css";

/** Diagonal hatch segments (x + y = c) clipped to a rectangle, so no pattern ids are needed. */
export function hatchSegments(x0: number, y0: number, x1: number, y1: number, step: number): string {
  const out: string[] = [];
  for (let c = x0 + y0 + step; c < x1 + y1; c += step) {
    const ax = Math.max(x0, c - y1), ay = c - ax;
    const bx = Math.min(x1, c - y0), by = c - bx;
    if (bx - ax > 0.2) out.push(`M${ax.toFixed(2)} ${ay.toFixed(2)}L${bx.toFixed(2)} ${by.toFixed(2)}`);
  }
  return out.join("");
}

const MODEL_HATCH = hatchSegments(2, 2, 12, 12, 3.2);
const NOVIEW_HATCH = hatchSegments(1, 1, 23, 13, 3);

export interface MarkProps {
  kind: MarkKind;
  /** Official marks carry their words; turn off only where the same line already says "not verified". */
  withWords?: boolean;
  className?: string;
}

/**
 * One of the five semantic marks (spec section 4). Shape carries the class: measured solid, model hatched,
 * official a ring with words, no view a hatched swatch, stale the word "(old)".
 */
export function Mark({ kind, withWords = true, className }: MarkProps) {
  if (kind === "stale") return <span className={cx(s.old, className)}>(old)</span>;
  if (kind === "official") {
    return (
      <span className={cx(s.mark, className)}>
        <svg className={s.glyph} width="14" height="14" viewBox="0 0 14 14" aria-hidden focusable="false">
          <circle cx="7" cy="7" r="5" className={s.ring} />
        </svg>
        {withWords ? <span className={s.words}>{NOT_VERIFIED}</span> : null}
      </span>
    );
  }
  if (kind === "noview") {
    return (
      <svg className={cx(s.glyph, s.swatch, className)} width="24" height="14" viewBox="0 0 24 14" role="img" aria-label="no view" focusable="false">
        <rect x="0.5" y="0.5" width="23" height="13" className={s.frame} />
        <path d={NOVIEW_HATCH} className={s.hatch} />
      </svg>
    );
  }
  return (
    <svg className={cx(s.glyph, className)} width="14" height="14" viewBox="0 0 14 14" role="img" aria-label={kind === "measured" ? "measured" : "model"} focusable="false">
      {kind === "measured" ? (
        <rect x="2" y="2" width="10" height="10" className={s.solid} />
      ) : (
        <>
          <path d={MODEL_HATCH} className={s.modelHatch} />
          <rect x="2.75" y="2.75" width="8.5" height="8.5" className={s.outline} />
        </>
      )}
    </svg>
  );
}

export interface LegendProps {
  /** The marks this view uses, in reading order. */
  kinds: MarkKind[];
  /** Words for this view, when they say more than the defaults ("C-HARM model, past run"). */
  words?: Partial<Record<MarkKind, string>>;
  /** Optional heading; the legend is still labelled for assistive tech without it. */
  title?: string;
  layout?: "row" | "column";
  className?: string;
}

/** One legend per view states the kinds once, instead of a caveat beside every figure. */
export function Legend({ kinds, words, title, layout = "row", className }: LegendProps) {
  return (
    <figure className={cx(s.legend, layout === "column" && s.column, className)} aria-label={title ?? "Legend"}>
      {title ? <figcaption className={s.title}>{title}</figcaption> : null}
      <ul className={s.items}>
        {kinds.map((k) => (
          <li key={k} className={s.item}>
            <Mark kind={k} withWords={false} />
            <span>{words?.[k] ?? MARK_WORDS[k]}</span>
          </li>
        ))}
      </ul>
    </figure>
  );
}
