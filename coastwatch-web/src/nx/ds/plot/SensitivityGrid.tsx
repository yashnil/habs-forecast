"use client";
import { useRef, type KeyboardEvent, type ReactNode } from "react";
import { usd } from "../format";
import { cx } from "../util";
import { breakEvenSteps } from "./model";
import s from "./plot.module.css";

export interface SensitivityGridProps {
  /** What the grid shows: "Net to boat as catch and price move, hypothetical". */
  caption: ReactNode;
  /** Down the side (catch): the label and its value. */
  rows: Array<{ label: string; value: number }>;
  /** Across the top (price). */
  cols: Array<{ label: string; value: number }>;
  /** Both axes named: "Catch, lb" down, "Price, $/lb" across. */
  rowAxis: string;
  colAxis: string;
  /** The value at a row and column, or null when it cannot be worked out. */
  value: (row: number, col: number) => number | null;
  format?: (v: number) => string;
  /** The chosen cell (the scenario the reader took). */
  selected?: { row: number; col: number } | null;
  onSelect?: (cell: { row: number; col: number }) => void;
  /** The cell under the pointer or the keyboard, or null when it leaves: a waterfall previews it. */
  onPreview?: (cell: { row: number; col: number } | null) => void;
  /** The reader's own numbers, marked "yours". */
  base?: { row: number; col: number };
  className?: string;
}

/**
 * A result across two inputs as a ruled grid of figures (no colour ramp: ramps live on the water). The
 * break-even line is drawn as a contour between the cells it separates; negative cells are lighter and carry a
 * true minus. A real grid for the keyboard: one tab stop, arrows move cell to cell, Enter or Space chooses.
 */
export function SensitivityGrid({ caption, rows, cols, rowAxis, colAxis, value, format = (v) => usd(Math.round(v)), selected, onSelect, onPreview, base, className }: SensitivityGridProps) {
  const cells = useRef(new Map<string, HTMLTableCellElement>());
  const grid = rows.map((_, r) => cols.map((__, c) => value(r, c)));
  const be = breakEvenSteps(grid);
  const start = selected ?? base ?? { row: 0, col: 0 };
  const go = (r: number, c: number) => cells.current.get(`${r}:${c}`)?.focus();
  const key = (r: number, c: number) => (e: KeyboardEvent<HTMLTableCellElement>) => {
    const d = { ArrowRight: [0, 1], ArrowLeft: [0, -1], ArrowDown: [1, 0], ArrowUp: [-1, 0] }[e.key];
    if (d) {
      e.preventDefault();
      go(Math.min(rows.length - 1, Math.max(0, r + d[0])), Math.min(cols.length - 1, Math.max(0, c + d[1])));
    } else if (e.key === "Home") {
      e.preventDefault();
      go(r, 0);
    } else if (e.key === "End") {
      e.preventDefault();
      go(r, cols.length - 1);
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onSelect?.({ row: r, col: c });
    }
  };
  return (
    <div className={cx(s.sensWrap, className)}>
    <table className={s.sens} role="grid" data-testid="sensitivity-grid" onPointerLeave={() => onPreview?.(null)}>
      <caption className={s.caption}>{caption}</caption>
      <thead>
        <tr>
          <th scope="col" className={s.sensCorner}>
            <span className={s.sensAxisRow}>{rowAxis}</span>
            <span className={s.sensAxisCol}>{colAxis}</span>
          </th>
          {cols.map((c) => (
            <th key={c.label} scope="col" className={s.sensHead}>
              {c.label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, r) => (
          <tr key={row.label}>
            <th scope="row" className={s.sensHead}>
              {row.label}
            </th>
            {cols.map((col, c) => {
              const v = grid[r][c];
              const on = selected?.row === r && selected?.col === c;
              const yours = base?.row === r && base?.col === c;
              // the contour: left of the first cell at or above zero in this row; above or below where the row's step moves
              const left = be[r] === c && c > 0;
              const above = r > 0 && v !== null && grid[r - 1][c] !== null && v >= 0 !== (grid[r - 1][c] as number) >= 0;
              return (
                <td
                  key={col.label}
                  ref={(el) => {
                    if (el) cells.current.set(`${r}:${c}`, el);
                    else cells.current.delete(`${r}:${c}`);
                  }}
                  role="gridcell"
                  tabIndex={start.row === r && start.col === c ? 0 : -1}
                  aria-selected={on}
                  className={s.sensCell}
                  data-negative={v !== null && v < 0 ? "" : undefined}
                  data-be-left={left || undefined}
                  data-be-top={above || undefined}
                  data-yours={yours || undefined}
                  onClick={() => onSelect?.({ row: r, col: c })}
                  onKeyDown={key(r, c)}
                  onFocus={() => onPreview?.({ row: r, col: c })}
                  onBlur={() => onPreview?.(null)}
                  onPointerEnter={() => onPreview?.({ row: r, col: c })}
                  data-testid={`sens-${r}-${c}`}
                >
                  {v === null ? <span className={s.sensNone}>n/a</span> : format(v)}
                  {yours ? <span className={s.sensYours}>yours</span> : null}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
      <tfoot>
        <tr>
          <td colSpan={cols.length + 1} className={s.sensFoot}>
            <span className={s.sensKey} aria-hidden="true" /> Break-even: net to boat is zero along this line.
          </td>
        </tr>
      </tfoot>
    </table>
    </div>
  );
}
