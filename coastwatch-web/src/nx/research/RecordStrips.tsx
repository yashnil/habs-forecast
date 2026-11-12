"use client";

import { scaleLinear, scaleLog, scaleUtc } from "d3-scale";
import { motion, useReducedMotion } from "motion/react";
import { forwardRef, useImperativeHandle, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { day } from "@/nx/ds/format";
import { hatchSegments } from "@/nx/ds/Mark";
import { ink } from "@/nx/ds/motion";
import { useElementWidth } from "@/nx/ds/useMedia";
import { inWin, linearTicksFor, logTicksFor, readingAt, runsOf, tickLabel, valueWords, type RecPoint, type Strip } from "./model";
import s from "./research.module.css";

/** The plot's left gutter and right edge: the same as the ds Brush's, so the window frame lines up with every strip. */
export const LEFT = 84;
export const RIGHT = 12;
const HEAD = 26;
const LANE = 20;
const GAP = 10;
const AXIS = 26;
const r2 = (v: number) => Math.round(v * 100) / 100;

const utcYear = new Intl.DateTimeFormat("en-US", { timeZone: "UTC", year: "numeric" });
const utcMonth = new Intl.DateTimeFormat("en-US", { timeZone: "UTC", month: "short" });
const utcDay = new Intl.DateTimeFormat("en-US", { timeZone: "UTC", month: "short", day: "numeric" });
function xTickLabel(d: Date, spanDays: number): string {
  if (spanDays > 3 * 365) return utcYear.format(d);
  if (spanDays <= 120) return utcDay.format(d);
  return d.getUTCMonth() === 0 ? utcYear.format(d) : utcMonth.format(d);
}

export type StripLayout = { strip: Strip; top: number; head: number; plotTop: number; plotH: number; laneTop: number | null; bottom: number; folded: boolean };

/** One header line, px; each wrapped line of a threshold strip's unit adds another. */
export const HEAD_LINE = 18;

let ctx2d: CanvasRenderingContext2D | null | undefined;
/** Text width in px with the page's sans at `size`/`weight` (canvas); off the browser, about 0.5 em a character. */
function textWidth(text: string, size: number, weight = 400): number {
  if (typeof document === "undefined") return text.length * size * 0.5;
  if (ctx2d === undefined) ctx2d = document.createElement("canvas").getContext("2d");
  if (!ctx2d) return text.length * size * 0.5;
  // the system's sans, read where the tokens apply (the body itself is not set in it)
  const root = document.querySelector("[data-chart-mode]") ?? document.documentElement;
  const family = getComputedStyle(root).getPropertyValue("--font-sans").trim() || "sans-serif";
  ctx2d.font = `${weight} ${size}px ${family}`;
  return ctx2d.measureText(text).width;
}

/**
 * A strip's header height at a width: one line for most; a strip with a dashed threshold puts its unit and what
 * the dashes mean on lines of their own under the name, as many as the words need, so nothing is cut with an
 * ellipsis (critic 15). The name shares its line with the reading when both fit.
 */
export function stripHead(strip: Strip, width: number): number {
  if (!strip.threshold) return HEAD;
  const room = Math.max(160, width - 8);
  const name = textWidth(`${strip.kind === "model" ? "Model · " : ""}${strip.label}`, 16, 600) + 8 + textWidth("not measured that day", 16);
  const words = textWidth(`${strip.unit}; dashes: ${strip.threshold.label}`, 14);
  // a little over the bare width: wrapping at word breaks leaves the end of each line short
  const lines = Math.min(4, Math.max(1, Math.ceil((words * 1.06) / room)));
  return HEAD * (name > room ? 2 : 1) + lines * HEAD_LINE;
}

/**
 * Where each strip sits: header, plot, the censored lane when its record has any reported 0 or rejected sample.
 * A strip with nothing in the window folds to its header line, which says so.
 */
export function layoutStrips(strips: Strip[], plotH: number, folded: ReadonlySet<string> = new Set(), headOf: (strip: Strip) => number = () => HEAD): { rows: StripLayout[]; height: number } {
  let y = 0;
  const rows = strips.map((strip) => {
    if (folded.has(strip.id)) {
      const top = y;
      y = top + HEAD + GAP;
      return { strip, top, head: HEAD, plotTop: top + HEAD, plotH: 0, laneTop: null, bottom: top + HEAD, folded: true };
    }
    const h = strip.kind === "model" ? Math.max(36, Math.round(plotH * 0.66)) : plotH;
    const lane = strip.points.some((p) => p.cls !== "value");
    const top = y;
    const head = headOf(strip);
    // a wrapped unit line ends at the header's foot: the top tick label (centred on plotTop) needs the extra room
    const plotTop = top + head + (head > HEAD ? 10 : 4);
    const laneTop = lane ? plotTop + h : null;
    const bottom = plotTop + h + (lane ? LANE : 0);
    y = bottom + GAP;
    return { strip, top, head, plotTop, plotH: h, laneTop, bottom, folded: false };
  });
  return { rows, height: (rows.at(-1)?.bottom ?? 0) + AXIS };
}

export type RecordStripsHandle = { svg: () => SVGSVGElement | null };

export type RecordStripsProps = {
  strips: Strip[];
  win: [number, number];
  /** the crosshair's day (ms), shared by every strip */
  cursor: number | null;
  /** the day the reader is pointing at, or null when the pointer leaves (the page then shows its default) */
  onCursor: (t: number | null) => void;
  focus: string;
  onFocus: (id: string) => void;
  plotH: number;
  /** strips with nothing in the (committed) window: folded to their header line */
  folded: ReadonlySet<string>;
  /** a new key draws the record in again (a new station); window changes cut */
  inkKey: string;
  now: number;
};

/**
 * The record: every quantity a station measures, as strips on one time axis in one SVG, so a single
 * crosshair runs through all of them and the whole stack exports as one figure. Reading rules: a reported 0
 * sits in its strip's own lane, never at 0 (T7); a rejected sample is a cross in the lane, never a value; a
 * flagged value keeps its place and gains a ring; samples more than three weeks apart are not joined (T19);
 * the model is hollow and dashed, joined only day to day, and hatched where no history was kept (T2).
 */
export const RecordStrips = forwardRef<RecordStripsHandle, RecordStripsProps>(function RecordStrips({ strips, win, cursor, onCursor, focus, onFocus, plotH, folded, inkKey, now }, ref) {
  const [box, width] = useElementWidth<HTMLDivElement>(720);
  const svgRef = useRef<SVGSVGElement>(null);
  useImperativeHandle(ref, () => ({ svg: () => svgRef.current }), []);
  const reduce = useReducedMotion();
  const [said, setSaid] = useState("");
  const plotW = Math.max(120, width - LEFT - RIGHT);
  const x = useMemo(() => scaleLinear().domain(win).range([LEFT, LEFT + plotW]), [win, plotW]);
  const { rows, height } = useMemo(() => layoutStrips(strips, plotH, folded, (st) => stripHead(st, width)), [strips, plotH, folded, width]);
  const spanDays = (win[1] - win[0]) / 86_400_000;
  const xTicks = useMemo(() => {
    const u = scaleUtc().domain([new Date(win[0]), new Date(win[1])]);
    return u.ticks(spanDays > 3 * 365 ? Math.max(2, Math.min(Math.round(spanDays / 365), Math.floor(plotW / 64))) : Math.max(2, Math.round(plotW / 110)));
  }, [win, spanDays, plotW]);

  // every measured visit in the window: the crosshair steps through these
  const visits = useMemo(() => {
    const ts = new Set<number>();
    for (const st of strips) if (st.kind === "measured") for (const p of st.points) if (inWin(p.t, win)) ts.add(p.t);
    return [...ts].sort((a, b) => a - b);
  }, [strips, win]);
  const modelDays = useMemo(() => strips.find((st) => st.kind === "model")?.points.filter((p) => inWin(p.t, win)).map((p) => p.t) ?? [], [strips, win]);

  const nearest = (ts: number[], t: number) => {
    let best: number | null = null;
    for (const v of ts) if (best == null || Math.abs(v - t) < Math.abs(best - t)) best = v;
    return best;
  };

  const fromPointer = (e: PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - r.left;
    const py = e.clientY - r.top;
    if (px < LEFT - 8 || px > LEFT + plotW + 8) return;
    const t = x.invert(px);
    const over = rows.find((row) => py >= row.top && py <= row.bottom + GAP / 2);
    const snapped = over?.strip.kind === "model" ? nearest(modelDays, t) : nearest(visits, t);
    onCursor(snapped);
  };

  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (!visits.length) return;
    const i = cursor == null ? -1 : visits.findIndex((v) => v >= cursor - 1000);
    let n = i;
    if (e.key === "ArrowRight") n = i < 0 ? 0 : Math.min(visits.length - 1, visits[i] > (cursor ?? 0) + 1000 ? i : i + 1);
    else if (e.key === "ArrowLeft") n = i < 0 ? visits.length - 1 : Math.max(0, i - 1);
    else if (e.key === "Home") n = 0;
    else if (e.key === "End") n = visits.length - 1;
    else if (e.key === "Escape") return onCursor(null);
    else return;
    e.preventDefault();
    const t = visits[n];
    onCursor(t);
    setSaid(`${day(t, { year: true })}: ${strips.map((st) => `${st.label}, ${wordsAt(st, t)}`).join("; ")}.`);
  };

  const cx = cursor != null && inWin(cursor, win) ? r2(x(cursor)) : null;
  const pxPerDay = plotW / Math.max(1, spanDays);

  return (
    <div className={s.strips} data-testid="research-strips">
      {rows.map((row) => {
        const st = row.strip;
        const reading = cursor != null ? readingAt(st, cursor) : null;
        const focused = st.id === focus;
        return (
          <div key={st.id} className={s.stripHead} style={{ top: row.top, height: row.head }} data-strip={st.id} data-folded={row.folded || undefined} data-wrap={(!row.folded && row.head > HEAD) || undefined}>
            {st.kind === "measured" ? (
              <button type="button" className={s.stripName} aria-pressed={focused} onClick={() => onFocus(st.id)} data-testid={`strip-${st.id}`}>
                <StripLabel strip={st} />
              </button>
            ) : (
              <span className={s.stripName} data-testid="strip-model">
                <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden className={s.modelMark}>
                  <path d={hatchSegments(2, 2, 12, 12, 3.2)} />
                  <rect x="2.75" y="2.75" width="8.5" height="8.5" />
                </svg>
                <span>Model · {st.label}</span>
              </span>
            )}
            {/* a strip with a dashed threshold says what the dashes are on a line of its own, wrapped, never cut (critic 15) */}
            <span className={s.stripUnit} title={st.threshold ? `${st.unit}; dashed line: ${st.threshold.label}` : st.unit}>
              {st.unit}
              {st.threshold && !row.folded ? `; dashes: ${st.threshold.label}` : ""}
            </span>
            <span className={s.stripReading} data-testid={`reading-${st.id}`}>
              {row.folded ? "none in this window" : cursor == null ? "" : reading ? valueWords(st, reading) : st.kind === "model" ? "no run kept that day" : "not measured that day"}
            </span>
          </div>
        );
      })}
      <div
        ref={box}
        className={s.plotArea}
        tabIndex={0}
        role="group"
        aria-label={`The record, ${strips.length} strips on one time axis. Arrow keys step through the sampling visits in the window; Home and End read the first and the last.`}
        onPointerMove={fromPointer}
        onPointerDown={fromPointer}
        onPointerLeave={(e) => (e.pointerType === "mouse" ? onCursor(null) : undefined)}
        onKeyDown={onKey}
        style={{ height }}
      >
        <svg ref={svgRef} width={width} height={height} className={s.svg} aria-hidden focusable="false" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <clipPath id={`rx-clip-${inkKey.replace(/[^\w-]/g, "")}`}>
              <motion.rect key={inkKey} x={LEFT - 3} y={0} height={height} initial={{ width: 0 }} animate={{ width: plotW + 6 }} transition={reduce ? { duration: 0 } : ink(1.6)} />
            </clipPath>
          </defs>
          {rows.map((row) => row.folded ? null : (
            <StripPlot key={row.strip.id} row={row} x={x} plotW={plotW} xTicks={xTicks} win={win} clip={`rx-clip-${inkKey.replace(/[^\w-]/g, "")}`} pxPerDay={pxPerDay} now={now} cursor={cursor} />
          ))}
          {xTicks.map((d) => (
            <text key={`xl${d.getTime()}`} x={r2(x(d.getTime()))} y={height - 6} className={s.xTick}>
              {xTickLabel(d, spanDays)}
            </text>
          ))}
          {cx != null ? (
            <g data-testid="research-crosshair">
              {rows.filter((row) => !row.folded).map((row) => (
                <line key={row.strip.id} x1={cx} x2={cx} y1={row.plotTop} y2={row.bottom} className={s.cross} />
              ))}
              <AxisFlag x={cx} y={height - AXIS + 2} min={LEFT} max={LEFT + plotW} text={day(cursor as number, { year: true })} />
            </g>
          ) : null}
        </svg>
      </div>
      <p className={s.vh} aria-live="polite">
        {said}
      </p>
    </div>
  );
});

/** The crosshair's day, set on the time axis where the line meets it (ink, paper type), clamped to the plot. */
function AxisFlag({ x, y, min, max, text }: { x: number; y: number; min: number; max: number; text: string }) {
  const w = text.length * 8 + 16;
  const left = Math.min(max - w, Math.max(min - 40, x - w / 2));
  return (
    <g className={s.flag}>
      <rect x={r2(left)} y={y} width={w} height={22} />
      <text x={r2(left + w / 2)} y={y + 11} dy="0.34em">
        {text}
      </text>
    </g>
  );
}

function wordsAt(st: Strip, t: number) {
  const r = readingAt(st, t);
  return r ? valueWords(st, r) : st.kind === "model" ? "no run kept" : "not measured";
}

function StripLabel({ strip }: { strip: Strip }) {
  if (strip.v === "pn_seriata" || strip.v === "pn_delicatissima") {
    return (
      <span>
        <i>Pseudo-nitzschia</i>
        {strip.label.slice("Pseudo-nitzschia".length)}
      </span>
    );
  }
  return <span>{strip.label}</span>;
}

type PlotProps = {
  row: StripLayout;
  x: ReturnType<typeof scaleLinear<number, number>>;
  plotW: number;
  xTicks: Date[];
  win: [number, number];
  clip: string;
  pxPerDay: number;
  now: number;
  cursor: number | null;
};

function StripPlot({ row, x, plotW, xTicks, win, clip, pxPerDay, now, cursor }: PlotProps) {
  const st = row.strip;
  const { plotTop, plotH, laneTop } = row;
  const bottom = plotTop + plotH;
  const ys = st.scale === "log" ? scaleLog().domain(st.domain).range([bottom, plotTop]).clamp(true) : scaleLinear().domain(st.domain).range([bottom, plotTop]).clamp(true);
  const y = (v: number) => r2(ys(v));
  const ticks = st.scale === "log" ? logTicksFor(st.domain, plotH) : linearTicksFor(st.domain, plotH);
  const model = st.kind === "model";
  const runs = runsOf(st, win);
  const pad = 2 * 86_400_000;
  const shown = st.points.filter((p) => p.t >= win[0] - pad && p.t <= win[1] + pad);
  const laneY = laneTop != null ? laneTop + LANE / 2 : 0;
  const path = (run: RecPoint[]) => run.map((p, i) => `${i ? "L" : "M"}${r2(x(p.t))} ${y(p.v as number)}`).join("");
  const nowX = r2(x(now));
  const hs = st.historyStart;
  const hatchTo = hs != null ? Math.min(LEFT + plotW, Math.max(LEFT, x(hs))) : null;
  const reading = cursor != null ? readingAt(st, cursor) : null;
  const dense = model && pxPerDay < 5;
  // a long window draws smaller marks, so a decade of weekly samples stays a record and not a smear
  const inView = shown.filter((p) => p.cls === "value").length;
  const half = inView / Math.max(1, plotW) > 0.22 ? 1.5 : inView / Math.max(1, plotW) > 0.1 ? 2 : 2.75;
  return (
    <g data-plot={st.id}>
      {ticks.map((tk, i) => (
        <g key={tk.v}>
          <line x1={LEFT} x2={LEFT + plotW} y1={y(tk.v)} y2={y(tk.v)} className={s.grid} />
          {tk.label && !(laneTop != null && i === 0) ? (
            <text x={LEFT - 8} y={y(tk.v)} dy="0.34em" className={s.yTick}>
              {tickLabel(tk.v)}
            </text>
          ) : null}
        </g>
      ))}
      {xTicks.map((d) => (
        <line key={d.getTime()} x1={r2(x(d.getTime()))} x2={r2(x(d.getTime()))} y1={plotTop} y2={bottom} className={s.grid} />
      ))}
      <line x1={LEFT} x2={LEFT + plotW} y1={bottom} y2={bottom} className={s.baseline} />
      {model && hatchTo != null && hatchTo > LEFT + 1 ? (
        <g>
          <path d={hatchSegments(LEFT, plotTop, hatchTo, bottom, 6)} className={s.hatch} />
          {hatchTo - LEFT > 220 ? (
            <text x={LEFT + 8} y={plotTop + 18} className={s.hatchText}>
              No model history kept before {day(hs as number, { year: true })}
            </text>
          ) : null}
        </g>
      ) : null}
      {st.threshold ? (
        <g>
          <line x1={LEFT} x2={LEFT + plotW} y1={y(st.threshold.value)} y2={y(st.threshold.value)} className={s.threshold} />
        </g>
      ) : null}
      {nowX >= LEFT && nowX <= LEFT + plotW ? <line x1={nowX} x2={nowX} y1={plotTop} y2={laneTop != null ? laneTop + LANE : bottom} className={s.now} /> : null}
      <g clipPath={`url(#${clip})`}>
        {runs.map((run) => (run.length > 1 ? <path key={`${run[0].t}-${run.length}`} d={path(run)} className={model ? s.lineModel : s.line} style={dense ? { strokeDasharray: "3 2", strokeWidth: 1 } : half < 2 ? { strokeWidth: 0.75 } : undefined} /> : null))}
        {shown.map((p) => {
          if (p.cls !== "value") return null;
          const px = r2(x(p.t));
          const py = y(p.v as number);
          if (model) {
            if (st.partial?.has(p.t)) return <circle key={p.t} cx={px} cy={py} r={3.2} className={s.ptHollow} />;
            return dense ? null : <rect key={p.t} x={px - 2.75} y={py - 2.75} width={5.5} height={5.5} className={s.ptHollow} />;
          }
          return (
            <g key={p.t}>
              <rect x={px - half} y={py - half} width={2 * half} height={2 * half} className={s.pt} />
              {p.flagged ? <circle cx={px} cy={py} r={6.5} className={s.ring} /> : null}
            </g>
          );
        })}
        {laneTop != null
          ? shown.map((p) => {
              const px = r2(x(p.t));
              if (p.cls === "zero") return <rect key={`z${p.t}`} x={px - Math.max(2, half)} y={laneY - Math.max(2, half)} width={2 * Math.max(2, half)} height={2 * Math.max(2, half)} className={s.ptZero} />;
              if (p.cls === "rejected") return <path key={`r${p.t}`} d={`M${px - 3.5} ${laneY - 3.5}L${px + 3.5} ${laneY + 3.5}M${px + 3.5} ${laneY - 3.5}L${px - 3.5} ${laneY + 3.5}`} className={s.cross} />;
              return null;
            })
          : null}
      </g>
      {laneTop != null ? (
        <g>
          <text x={LEFT - 8} y={laneY} dy="0.34em" className={s.yTick}>
            Reported 0
          </text>
          <line x1={LEFT} x2={LEFT + plotW} y1={laneTop + LANE} y2={laneTop + LANE} className={s.grid} />
        </g>
      ) : null}
      {reading && inWin(reading.t, win) ? (
        reading.cls === "value" ? (
          <rect x={r2(x(reading.t)) - 5} y={y(reading.v as number) - 5} width={10} height={10} className={model ? s.crossHollow : s.crossMark} />
        ) : laneTop != null ? (
          <rect x={r2(x(reading.t)) - 5} y={laneY - 5} width={10} height={10} className={s.crossHollow} />
        ) : null
      ) : null}
      {shown.length === 0 ? (
        <text x={LEFT + plotW / 2} y={plotTop + plotH / 2} dy="0.34em" className={s.empty}>
          {model ? "No model day kept in this window" : "No sample in this window"}
        </text>
      ) : null}
    </g>
  );
}
