"use client";

import { useRef, type KeyboardEvent } from "react";
import { day } from "@/nx/ds/format";
import type { ObsVariableId } from "@/lib/observations";
import type { Availability, VisitTick } from "./availability";
import { DAY, LABEL, valueWords } from "./model";
import { ageWords, tickPaths } from "./plotData";
import s from "./v13.module.css";

export type RailRow = {
  a: Availability;
  /** what the unit is of: "ng/mL in seawater" */
  unit: string;
  unitShort: string;
  /** every visit of the station, read for this variable, over the whole record */
  ticks: VisitTick[];
};

/**
 * The page's primary selector, and its availability by variable and date in one: a radiogroup with one row per
 * variable the station has ever measured. Each row says the unit, when the variable was last measured and how
 * long ago, how many samples the window holds, and draws every visit of the record as a tick (tall: a value;
 * short at the foot: a reported 0; short at the head: rejected; a dot: visited, this variable not measured),
 * with the window framed. So "is there any pDA here, and when" is answered before a range is guessed.
 */
export function VariableRail({ rows, selected, onSelect, extent, win, now, stationName }: { rows: RailRow[]; selected: ObsVariableId; onSelect: (v: ObsVariableId) => void; extent: [number, number]; win: [number, number]; now: number; stationName: string }) {
  const refs = useRef<Array<HTMLButtonElement | null>>([]);
  const span = Math.max(1, extent[1] - extent[0]);
  const pct = (t: number) => Math.min(100, Math.max(0, ((t - extent[0]) / span) * 100));
  const key = (i: number) => (e: KeyboardEvent<HTMLButtonElement>) => {
    const d = e.key === "ArrowDown" || e.key === "ArrowRight" ? 1 : e.key === "ArrowUp" || e.key === "ArrowLeft" ? -1 : e.key === "Home" ? -i : e.key === "End" ? rows.length - 1 - i : 0;
    if (!d) return;
    e.preventDefault();
    const n = (i + d + rows.length) % rows.length;
    onSelect(rows[n].a.v);
    refs.current[n]?.focus();
  };
  return (
    <div className={s.rail} role="radiogroup" aria-label={`Variable measured at ${stationName}`} data-testid="research-rail">
      {rows.map((r, i) => {
        const on = r.a.v === selected;
        const last = r.a.last;
        const days = last ? Math.floor((now - last.t) / DAY) : null;
        const strip = { kind: "measured" as const, v: r.a.v, unitShort: r.unitShort };
        const paths = tickPaths(r.ticks, extent);
        const inWindow = `${r.a.withValue.toLocaleString("en-US")} in window`;
        return (
          <button
            key={r.a.v}
            ref={(el) => {
              refs.current[i] = el;
            }}
            type="button"
            role="radio"
            aria-checked={on}
            tabIndex={on ? 0 : -1}
            className={s.railRow}
            onClick={() => onSelect(r.a.v)}
            onKeyDown={key(i)}
            data-testid={`rail-${r.a.v}`}
            data-empty={r.a.withValue === 0 || undefined}
          >
            <span className={s.railName}>
              <span className={s.railLabel}>{LABEL[r.a.v]}</span>
              <span className={s.railUnit}>{r.unit}</span>
            </span>
            <svg className={s.railTicks} viewBox="0 0 1000 12" preserveAspectRatio="none" aria-hidden="true" focusable="false">
              <rect x={pct(win[0]) * 10} y={0} width={Math.max(4, (pct(win[1]) - pct(win[0])) * 10)} height={12} className={s.railWindow} />
              <path d={paths.absent} className={s.tickAbsent} />
              <path d={paths.quantified} className={s.tickValue} />
              <path d={paths.zero} className={s.tickZero} />
              <path d={paths.rejected} className={s.tickRejected} />
            </svg>
            <span className={s.railLast} data-testid={`rail-last-${r.a.v}`}>
              {last ? (
                <>
                  <span className={s.railLastDate}>Last {day(last.t, { year: true })}</span>
                  <span className={s.railAge}>
                    {ageWords(days as number)} · {valueWords(strip, last)}
                  </span>
                </>
              ) : (
                <span className={s.railLastDate}>Never measured</span>
              )}
            </span>
            <span className={s.railCount} data-testid={`rail-count-${r.a.v}`}>
              {inWindow}
            </span>
          </button>
        );
      })}
    </div>
  );
}
