"use client";

import { useState } from "react";
import { day } from "@/nx/ds/format";
import { logRows, valueWords, type SortDir, type SortKey, type Strip } from "./model";
import s from "./research.module.css";

/**
 * The dated log of the focused quantity inside the window: one row per sample, tabular figures, sortable by
 * date or value. A reported 0 reads "reported 0" and sorts below every measured value; a rejected sample says
 * so and has no value; a row highlights when the crosshair is on its day, and choosing a row moves the crosshair.
 */
export function SampleLog({ strip, win, cursor, onCursor }: { strip: Strip; win: [number, number]; cursor: number | null; onCursor: (t: number) => void }) {
  const [key, setKey] = useState<SortKey>("date");
  const [dir, setDir] = useState<SortDir>("desc");
  const rows = logRows(strip.points, win, key, dir);
  const sortBy = (k: SortKey) => {
    if (k === key) setDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setKey(k);
      setDir("desc");
    }
  };
  const aria = (k: SortKey) => (k === key ? (dir === "asc" ? "ascending" : "descending") : "none");
  const arrow = (k: SortKey) => (k === key ? (dir === "asc" ? "↑" : "↓") : "");
  if (!rows.length) return <p className={s.quiet}>No {strip.label.toLowerCase()} sample in this window. A visit without this measurement is not a zero.</p>;
  return (
    <div className={s.logWrap}>
      <table className={s.log} data-testid="research-log">
        <caption className={s.vh}>
          {strip.label}, every sample from {day(win[0], { year: true })} to {day(win[1], { year: true })}, {rows.length} rows
        </caption>
        <thead>
          <tr>
            <th scope="col" aria-sort={aria("date")}>
              <button type="button" onClick={() => sortBy("date")} data-testid="log-sort-date">
                Sampled <span aria-hidden>{arrow("date")}</span>
              </button>
            </th>
            <th scope="col" aria-sort={aria("value")} className={s.num}>
              <button type="button" onClick={() => sortBy("value")} data-testid="log-sort-value">
                {strip.unitShort} <span aria-hidden>{arrow("value")}</span>
              </button>
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((p) => (
            <tr key={p.t} data-on={cursor != null && Math.abs(cursor - p.t) < 86_400_000 ? "1" : undefined} data-cls={p.cls}>
              <th scope="row">
                <button type="button" onClick={() => onCursor(p.t)} aria-label={`Read ${day(p.t, { year: true })} on every strip`}>
                  {day(p.t, { year: true })}
                </button>
              </th>
              <td className={s.num}>{valueWords(strip, p, false)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className={s.logFoot}>
        {rows.length} {rows.length === 1 ? "sample" : "samples"} in the window, as published. The export holds every quantity.
      </p>
    </div>
  );
}
