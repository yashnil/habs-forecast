"use client";
import type { ReactNode } from "react";
import { DataState, type DataStateKind } from "../DataState";
import { cx } from "../util";
import s from "./plot.module.css";

export interface PlotStateProps {
  /** Why the plot has nothing to draw. */
  kind: Extract<DataStateKind, "missing" | "not-collected" | "unavailable" | "outside-coverage" | "loading" | "offline" | "stale">;
  /** What the plot is of: "pDA at Monterey Wharf". */
  what: ReactNode;
  /** The counts and the reason in one sentence: "0 of 12 samples in this window measured pDA". */
  reason?: ReactNode;
  /** What to do: jump to the latest sample, widen the range (Buttons). */
  action?: ReactNode;
  /** The height the plot would have, so nothing moves when data arrives. */
  height?: number;
  className?: string;
}

/** Stands where a plot would be when it has nothing to draw: the reason, the counts and what to do. Never an empty axis. */
export function PlotState({ kind, what, reason, action, height = 280, className }: PlotStateProps) {
  return (
    <div className={cx(s.plotState, className)} style={{ minHeight: height }} data-testid="plot-state" data-state={kind}>
      <DataState kind={kind} what={what} reason={reason} action={action} variant="plot" />
    </div>
  );
}

export interface PlotTableColumn<R> {
  id: string;
  head: ReactNode;
  cell: (row: R) => ReactNode;
  /** Right-aligned, tabular (figures). */
  numeric?: boolean;
}
export interface PlotTableProps<R> {
  /** What the table holds: "pDA samples at Monterey Wharf, Jan 1 to Mar 31, 2025". */
  caption: ReactNode;
  columns: PlotTableColumn<R>[];
  rows: R[];
  rowKey: (row: R, i: number) => string;
  /** The row the plot has pinned, marked in the table. */
  selected?: string | null;
  onSelect?: (row: R) => void;
  /** Words when there are no rows. */
  empty?: ReactNode;
  /** Rows beyond this are behind "Show all" (default 200). */
  className?: string;
}

/** The plot's data as a real table: every sample with its exact value, state, source and date. */
export function PlotTable<R>({ caption, columns, rows, rowKey, selected, onSelect, empty = "No samples in this window.", className }: PlotTableProps<R>) {
  return (
    <div className={cx(s.tableWrap, className)}>
      <table className={s.table} data-testid="plot-table">
        <caption className={s.caption}>{caption}</caption>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.id} scope="col" data-numeric={c.numeric || undefined}>
                {c.head}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length ? (
            rows.map((r, i) => {
              const k = rowKey(r, i);
              return (
                <tr key={k} aria-selected={selected === k || undefined} data-selected={selected === k || undefined} onClick={onSelect ? () => onSelect(r) : undefined} data-clickable={onSelect ? "" : undefined}>
                  {columns.map((c) => (
                    <td key={c.id} data-numeric={c.numeric || undefined}>
                      {c.cell(r)}
                    </td>
                  ))}
                </tr>
              );
            })
          ) : (
            <tr>
              <td colSpan={columns.length} className={s.tableEmpty}>
                {empty}
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
