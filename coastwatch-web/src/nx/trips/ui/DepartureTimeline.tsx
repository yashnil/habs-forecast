"use client";

import { useEffect, useId, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { motion, useReducedMotion } from "motion/react";
import { Disclosure, Stamp, Unavailable, WindArrow, type Provenance } from "@/nx/ds";
import { hatchSegments } from "@/nx/ds/Mark";
import { reveal, select as selectT, value as valueT } from "@/nx/ds/motion";
import { cx } from "@/nx/ds/util";
import { compassDeg, DERIVED_NOTE, NWS_TZ, periodLabel, seasRangeWords, tideMs, wavesWords, windStatedWords } from "../forecast";
import type { ForecastSnap, OperatingLimits } from "../types";
import { tideClockOf, windowCells, type AdvisoryIn, type TideEvent, type WindowCell } from "../windows";
import { WindowCompare } from "./WindowCompare";
import s from "./departure.module.css";

export type DepartureTimelineProps = {
  forecast: ForecastSnap | null;
  tides: TideEvent[] | null;
  alerts: AdvisoryIn[] | null;
  limits: OperatingLimits;
  /** the trip's departure date (YYYY-MM-DD, port's zone); its periods are marked and always shown */
  departDate: string | null;
  now: string;
  /** the selected period keys (at most two), whenever the selection changes */
  onSelect?: (keys: string[]) => void;
  /** the zone of the forecast's periods (defaults to Pacific) */
  timeZone?: string;
  className?: string;
};

// geometry (SVG user units; one unit = one CSS pixel, the chart scrolls sideways inside itself on a phone)
const COL = 92;
const PAD_L = 40;
const TOP = 40; // labels
const ADV_H = 10;
const WIND_T = TOP + ADV_H + 10;
const WIND_H = 96;
const SEAS_T = WIND_T + WIND_H + 34;
const SEAS_H = 56;
const TIDE_T = SEAS_T + SEAS_H + 16;
const TIDE_H = 20;
const H = TIDE_T + TIDE_H + 6;

const nice = (v: number, step: number) => Math.ceil(v / step) * step;

/**
 * The departure timeline: the next forecast periods on a time axis, the wind as bars from the lowest to the highest
 * speed NWS states (an arrow per stated direction, gust ticks), seas as hatched forecast bars, the reader's own limits
 * as lines, the part of a bar over a limit marked, NWS advisories as spans, the predicted tide's highs and lows as
 * ticks. A period without a forecast is hatched and says so; it is never drawn calm. The reader selects up to two
 * periods (pointer, or arrows + Space) to compare. Nothing here ranks periods or calls one good.
 */
export function DepartureTimeline({ forecast, tides, alerts, limits, departDate, now, onSelect, timeZone = NWS_TZ, className }: DepartureTimelineProps) {
  const cells = useMemo(
    () =>
      windowCells(forecast, limits, tides, alerts, now, {
        departDate,
        timeZone,
      }).filter((c) => c.date),
    [forecast, limits, tides, alerts, now, departDate, timeZone],
  );
  const reported = cells.some((c) => c.state === "reported");
  const firstOpen = Math.max(
    0,
    cells.findIndex((c) => c.departure && c.state !== "expired") >= 0 ? cells.findIndex((c) => c.departure && c.state !== "expired") : cells.findIndex((c) => c.state === "reported"),
  );
  const [focus, setFocus] = useState(firstOpen);
  const [picked, setPicked] = useState<string[]>([]);
  const reduce = useReducedMotion();
  const id = useId();
  const listRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  // open on the departure period (or the first with a forecast): scroll it into the chart's view once
  useEffect(() => {
    const el = scrollRef.current;
    if (el && firstOpen > 0) el.scrollLeft = Math.max(0, firstOpen * COL - el.clientWidth / 3);
  }, [firstOpen]);

  if (!forecast || !forecast.periods.length || !reported) {
    return (
      <div className={cx(s.root, className)} data-testid="departure-timeline" data-state="unavailable">
        <Unavailable
          what="No marine forecast to compare departures with"
          why={
            forecast?.zone
              ? `NWS zone ${forecast.zone} gave no current forecast periods to Neritic. That is not a calm forecast.`
              : "NWS did not answer Neritic, or no forecast zone covers this port. That is not a calm forecast."
          }
          next={
            <a href={forecast?.url ?? "https://www.weather.gov/marine/"} className={s.link}>
              NWS marine forecasts
            </a>
          }
        />
      </div>
    );
  }

  const W = cells.length * COL;
  const winds = cells.flatMap((c) => (c.period ? [c.period.windHi ?? 0, c.period.gustKt ?? 0] : []));
  const windMax = nice(Math.max(20, limits.windKt ?? 0, ...winds) * 1.1, 10);
  const seasMax = nice(Math.max(6, limits.seasFt ?? 0, ...cells.map((c) => c.period?.seasHi ?? 0)) * 1.1, 2);
  const wy = (kt: number) => WIND_T + WIND_H - (kt / windMax) * WIND_H;
  const sy = (ft: number) => SEAS_T + SEAS_H - (ft / seasMax) * SEAS_H;
  const x0 = (i: number) => i * COL;

  const toggle = (key: string) => {
    setPicked((p) => {
      const n = p.includes(key) ? p.filter((k) => k !== key) : [...p, key].slice(-2);
      onSelect?.(n);
      return n;
    });
  };
  const keyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : e.key === "Home" ? -Infinity : e.key === "End" ? Infinity : 0;
    if (step) {
      e.preventDefault();
      const n = Math.max(0, Math.min(cells.length - 1, focus + step));
      setFocus(n);
      const el = scrollRef.current;
      if (el) {
        const left = x0(n);
        if (left < el.scrollLeft || left + COL > el.scrollLeft + el.clientWidth)
          el.scrollTo({
            left: Math.max(0, left - COL / 2),
            behavior: reduce ? "auto" : "smooth",
          });
      }
    } else if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
      toggle(cells[focus].key);
    } else if (e.key === "Escape" && picked.length) {
      e.preventDefault();
      setPicked([]);
      onSelect?.([]);
    }
  };

  const focused = cells[focus] ?? cells[0];
  const pair = picked.map((k) => cells.find((c) => c.key === k)).filter((c): c is WindowCell => !!c);
  const nws: Provenance | null = forecast.issued
    ? {
        kind: "prediction",
        title: `NWS coastal waters forecast, ${forecast.zone}`,
        source: `NWS ${forecast.zone}`,
        sourceLong: `National Weather Service coastal waters forecast, zone ${forecast.zone}`,
        sourceUrl: forecast.url ?? undefined,
        date: forecast.issued,
        dateRole: "issued",
        caveats: [`${DERIVED_NOTE}: NWS's zone forecast gives each period a name, not a start and end time.`],
      }
    : null;
  const grow = (i: number) => (reduce ? { duration: 0 } : { ...valueT, delay: Math.min(0.4, i * 0.03) });

  return (
    <div className={cx(s.root, className)} data-testid="departure-timeline" data-state="ok">
      <div className={s.legend} aria-hidden>
        <span className={s.legWind}>Wind, kt</span>
        <span className={s.legSeas}>Seas, ft</span>
        {limits.windKt != null || limits.seasFt != null ? <span className={s.legLimit}>Your limits</span> : null}
        <span className={s.legNone}>No forecast</span>
      </div>
      <div className={s.frame}>
        {/* the scale stays put while the periods scroll beside it */}
        <svg className={s.axis} width={PAD_L} height={H} viewBox={`0 0 ${PAD_L} ${H}`} aria-hidden focusable="false">
          <text x={PAD_L - 6} y={wy(windMax) + 4} className={s.tick} textAnchor="end">
            {windMax}
          </text>
          <text x={PAD_L - 6} y={wy(0)} className={s.tick} textAnchor="end">
            0
          </text>
          <text x={PAD_L - 6} y={WIND_T - 16} className={s.tick} textAnchor="end">
            kt
          </text>
          <text x={PAD_L - 6} y={sy(seasMax) + 4} className={s.tick} textAnchor="end">
            {seasMax}
          </text>
          <text x={PAD_L - 6} y={sy(0)} className={s.tick} textAnchor="end">
            0
          </text>
          <text x={PAD_L - 6} y={SEAS_T - 8} className={s.tick} textAnchor="end">
            ft
          </text>
          <text x={PAD_L - 6} y={TIDE_T + 14} className={s.tick} textAnchor="end">
            tide
          </text>
          {limits.windKt != null ? (
            <text x={PAD_L - 6} y={wy(limits.windKt) + 4} className={s.limitTick} textAnchor="end">
              {limits.windKt}
            </text>
          ) : null}
          {limits.seasFt != null ? (
            <text x={PAD_L - 6} y={sy(limits.seasFt) + 4} className={s.limitTick} textAnchor="end">
              {limits.seasFt}
            </text>
          ) : null}
        </svg>
        <div className={s.scroll} ref={scrollRef}>
          <div
            ref={listRef}
            role="listbox"
            aria-label="Forecast periods. Arrow keys move; Space selects up to two to compare."
            aria-multiselectable="true"
            aria-activedescendant={`${id}-${focus}`}
            tabIndex={0}
            className={s.list}
            style={{ width: W }}
            onKeyDown={keyDown}
          >
            <svg className={s.svg} width={W} height={H} viewBox={`0 0 ${W} ${H}`} aria-hidden focusable="false">
              <defs>
                <pattern id={`${id}-hatch`} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                  <line x1="0" y1="0" x2="0" y2="6" className={s.hatchLine} />
                </pattern>
              </defs>
              {cells.map((c, i) => {
                const x = x0(i);
                const p = c.state === "reported" ? c.period : null;
                const on = picked.includes(c.key);
                const cx0 = x + COL / 2;
                return (
                  <g key={c.key} className={cx(s.col, c.night && s.night, c.state !== "reported" && s.off, c.state === "expired" && s.expired)}>
                    <rect x={x} y={0} width={COL} height={H} className={s.band} />
                    {c.departure ? <rect x={x} y={0} width={COL} height={3} className={s.depMark} /> : null}
                    <text x={x + 8} y={16} className={cx(s.colDay, c.departure && s.colDep)}>
                      {c.night ? "" : c.label.split(",")[0]}
                    </text>
                    <text x={x + 8} y={32} className={s.colHalf}>
                      {c.night ? `${c.label.split(",")[0]} night` : "day"}
                    </text>
                    {/* advisories */}
                    {c.advisories.length ? <rect x={x + 1} y={TOP} width={COL - 2} height={ADV_H} className={s.adv} /> : null}
                    {c.state !== "reported" ? (
                      <>
                        <rect x={x + 6} y={WIND_T} width={COL - 12} height={SEAS_T + SEAS_H - WIND_T} fill={`url(#${id}-hatch)`} className={s.none} />
                        <text x={cx0} y={WIND_T + WIND_H / 2} textAnchor="middle" className={s.noneText}>
                          {c.state === "expired" ? "over" : "no forecast"}
                        </text>
                      </>
                    ) : null}
                    {p ? <WindMarks cell={c} x={x} wy={wy} limit={limits.windKt} grow={grow(i)} reduce={!!reduce} /> : null}
                    {p && p.seasHi != null ? (
                      <motion.g initial={reduce ? false : { opacity: 0 }} animate={{ opacity: 1 }} transition={grow(i)}>
                        <rect
                          x={cx0 - 14}
                          y={sy(p.seasHi)}
                          width={28}
                          height={Math.max(1, sy(p.seasLo ?? p.seasHi) - sy(p.seasHi)) + (p.seasLo != null && p.seasLo !== p.seasHi ? 0 : 3)}
                          className={s.seasBar}
                        />
                        <path d={hatchSegments(cx0 - 14, sy(p.seasHi), cx0 + 14, sy(p.seasLo ?? p.seasHi) + (p.seasLo != null && p.seasLo !== p.seasHi ? 0 : 3), 4)} className={s.seasHatch} />
                        {limits.seasFt != null && p.seasHi > limits.seasFt ? (
                          <rect x={cx0 - 14} y={sy(p.seasHi)} width={28} height={Math.max(3, sy(Math.max(limits.seasFt, p.seasLo ?? 0)) - sy(p.seasHi))} className={cx(s.over, s.overSeas)} />
                        ) : null}
                        <text x={cx0 + 18} y={sy(p.seasHi) + 4} className={s.num}>
                          {seasRangeWords(p)?.replace(" ft", "")}
                        </text>
                      </motion.g>
                    ) : p ? (
                      <text x={cx0} y={SEAS_T + SEAS_H / 2} textAnchor="middle" className={s.noneText}>
                        seas not stated
                      </text>
                    ) : null}
                    {/* tide ticks, placed by time inside the period */}
                    <line x1={x} x2={x + COL} y1={TIDE_T + TIDE_H} y2={TIDE_T + TIDE_H} className={s.tideBase} />
                    {c.tides.map((t) => {
                      const f = (tideMs(t.at, timeZone) - Date.parse(c.start!)) / (Date.parse(c.end!) - Date.parse(c.start!));
                      const tx = x + Math.max(0, Math.min(1, f)) * COL;
                      return (
                        <g key={t.at}>
                          <line x1={tx} x2={tx} y1={TIDE_T + (t.type === "H" ? 2 : 10)} y2={TIDE_T + TIDE_H} className={s.tideTick} />
                          <text x={tx > x + COL - 12 ? tx - 3 : tx + 3} textAnchor={tx > x + COL - 12 ? "end" : "start"} y={TIDE_T + (t.type === "H" ? 9 : 17)} className={s.tideText}>
                            {t.type}
                          </text>
                        </g>
                      );
                    })}
                    {on ? (
                      <motion.rect
                        x={x + 1.5}
                        y={1.5}
                        width={COL - 3}
                        height={H - 3}
                        className={s.picked}
                        initial={reduce ? false : { opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={reduce ? { duration: 0 } : selectT}
                      />
                    ) : null}
                  </g>
                );
              })}
              {/* the reader's limits, across the whole axis */}
              {limits.windKt != null ? <LimitLine y={wy(limits.windKt)} x1={0} x2={W} /> : null}
              {limits.seasFt != null ? <LimitLine y={sy(limits.seasFt)} x1={0} x2={W} /> : null}
            </svg>
            <div
              className={s.hits}
              style={{
                gridTemplateColumns: `repeat(${cells.length}, ${COL}px)`,
              }}
            >
              {cells.map((c, i) => (
                <div
                  key={c.key}
                  id={`${id}-${i}`}
                  role="option"
                  aria-selected={picked.includes(c.key)}
                  aria-label={cellWords(c, timeZone)}
                  data-testid={`window-${c.key}`}
                  data-state={c.state}
                  data-crossings={c.crossings.length}
                  className={cx(s.hit, i === focus && s.hitFocus)}
                  onClick={() => {
                    setFocus(i);
                    toggle(c.key);
                    listRef.current?.focus({ preventScroll: true });
                  }}
                />
              ))}
            </div>
          </div>
        </div>
      </div>
      <p className={s.hint} aria-hidden>
        {picked.length === 0 ? "Select a period, then a second one to compare." : picked.length === 1 ? "Select a second period to compare." : null}
      </p>
      <motion.div
        key={focused.key}
        className={s.detail}
        initial={reduce ? false : { opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        transition={reduce ? { duration: 0 } : reveal}
        aria-live="polite"
      >
        <Detail cell={focused} timeZone={timeZone} />
      </motion.div>
      {pair.length === 2 ? <WindowCompare a={pair[0]} b={pair[1]} timeZone={timeZone} onClear={() => (setPicked([]), onSelect?.([]))} /> : null}
      <Disclosure summary="Source" value={forecast.zone ? `NWS ${forecast.zone}` : "NWS"} className={s.source} data-testid="departure-source">
        <p className={s.sourceText}>
          {nws ? <Stamp now={now} provenance={nws} /> : null}
          <span>{DERIVED_NOTE}: NWS gives each period a name, not a start and end; day is read as 6 AM–6 PM and night as 6 PM–6 AM, Pacific time.</span>
          <span>The limit lines are your own, set for this trip; Neritic does not set or certify limits.</span>
          {tides?.length ? <span>Tide highs and lows are NOAA CO-OPS astronomical predictions; wind and swell move the real water.</span> : <span>No tide predictions with this forecast.</span>}
        </p>
      </Disclosure>
    </div>
  );
}

function LimitLine({ y, x1, x2 }: { y: number; x1: number; x2: number }) {
  return (
    <g className={s.limit}>
      <line x1={x1} x2={x2} y1={y} y2={y} />
    </g>
  );
}

/** One period's wind: a bar from the lowest to the highest stated speed, the part over the limit marked, gusts as ticks. */
function WindMarks({ cell, x, wy, limit, grow, reduce }: { cell: WindowCell; x: number; wy: (kt: number) => number; limit: number | null; grow: object; reduce: boolean }) {
  const p = cell.period!;
  const cx0 = x + COL / 2;
  const dirs = cell.winds.filter((w) => w.dir).slice(0, 3);
  const arrows = dirs.map((w, j) => {
    const deg = compassDeg(w.dir);
    const ax = cx0 - ((dirs.length - 1) * 16) / 2 + j * 16 - 7;
    return deg == null ? null : (
      <foreignObject key={j} x={ax} y={WIND_T - 2} width={14} height={14}>
        <WindArrow fromDeg={deg} className={s.arrow} />
      </foreignObject>
    );
  });
  if (p.windHi == null) {
    return (
      <>
        {arrows}
        <text x={cx0} y={WIND_T + WIND_H / 2} textAnchor="middle" className={s.noneText}>
          {p.windDir === "light" ? "light" : "speed not stated"}
        </text>
      </>
    );
  }
  const lo = p.windLo ?? 0;
  const top = wy(p.windHi);
  const bottom = wy(lo) + (lo === p.windHi ? 3 : 0);
  return (
    <>
      {arrows}
      <motion.rect
        x={cx0 - 5}
        width={10}
        className={cx(s.windBar, p.windLo == null && s.windOpen)}
        initial={reduce ? false : { y: bottom, height: 0 }}
        animate={{ y: top, height: Math.max(2, bottom - top) }}
        transition={grow}
      />
      {limit != null && p.windHi > limit ? <rect x={cx0 - 5} y={top} width={10} height={Math.max(3, wy(Math.max(limit, lo)) - top)} className={s.over} /> : null}
      {p.gustKt != null ? (
        <g className={cx(s.gust, limit != null && p.gustKt > limit && s.gustOver)}>
          <line x1={cx0 - 10} x2={cx0 + 10} y1={wy(p.gustKt)} y2={wy(p.gustKt)} />
          <text x={cx0 + 13} y={wy(p.gustKt) + 4}>
            G{p.gustKt}
          </text>
        </g>
      ) : null}
      <text x={cx0 + 9} y={top + 12} className={s.num}>
        {p.windLo != null && p.windLo !== p.windHi ? `${p.windLo}–${p.windHi}` : p.windLo == null ? `≤${p.windHi}` : p.windHi}
      </text>
    </>
  );
}

/** The period in words for assistive tech: the same facts the marks draw. */
function cellWords(c: WindowCell, tz: string): string {
  if (c.state === "unavailable") return `${c.label}: no NWS forecast for this period.`;
  if (c.state === "expired") return `${c.label}: this period is over.`;
  const p = c.period!;
  return [
    periodLabel(p, tz),
    windStatedWords(p.text) ? `wind ${windStatedWords(p.text)}` : "no wind stated",
    p.seasHi != null ? `seas ${seasRangeWords(p)}` : "no seas stated",
    ...c.crossings.map((x) => x.words),
    ...c.advisories.map((a) => a.event),
    c.departure ? "your departure date" : null,
  ]
    .filter(Boolean)
    .join(". ");
}

/** The focused period: what NWS stated, which of the reader's limits a stated number exceeds, tide and advisories. */
function Detail({ cell: c, timeZone }: { cell: WindowCell; timeZone: string }) {
  if (c.state !== "reported" || !c.period) {
    return (
      <div className={s.detailHead}>
        <p className={s.detailName}>{c.label}</p>
        <p className={s.detailNone}>{c.state === "expired" ? "This period is over." : "No NWS forecast covers this period. That says nothing about the conditions."}</p>
      </div>
    );
  }
  const p = c.period;
  return (
    <>
      <div className={s.detailHead}>
        <p className={s.detailName}>{periodLabel(p, timeZone)}</p>
        <p className={s.detailWind}>{windStatedWords(p.text) ?? "No wind stated"}</p>
      </div>
      <dl className={s.facts}>
        <div>
          <dt>Seas</dt>
          <dd>{seasRangeWords(p) ?? "not stated"}</dd>
        </div>
        <div>
          <dt>Gusts</dt>
          <dd>{p.gustKt != null ? `${p.gustKt} kt` : "not stated"}</dd>
        </div>
        {p.waves.length ? (
          <div>
            <dt>Waves</dt>
            <dd>{wavesWords(p)}</dd>
          </div>
        ) : null}
        {c.tides.length ? (
          <div>
            <dt>Tide</dt>
            <dd>{c.tides.map((t) => `${t.type === "H" ? "High" : "Low"} ${t.ft.toFixed(1)} ft ${tideClockOf(t.at)}`).join(" · ")}</dd>
          </div>
        ) : null}
      </dl>
      {c.crossings.length || c.advisories.length ? (
        <ul className={s.flags}>
          {c.crossings.map((x) => (
            <li key={x.limit} className={s.flagOver}>
              {x.words}
            </li>
          ))}
          {c.advisories.map((a) => (
            <li key={a.event} className={s.flagAdv}>
              <a href={a.url} className={s.link}>
                {a.headline || a.event}
              </a>
            </li>
          ))}
        </ul>
      ) : null}
      <blockquote className={s.quote}>
        <span className={s.quoteName}>{p.name}.</span> {p.text}
      </blockquote>
    </>
  );
}
