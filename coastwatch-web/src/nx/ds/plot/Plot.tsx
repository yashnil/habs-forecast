"use client";
import { scaleLinear, scaleLog } from "d3-scale";
import { line as d3line } from "d3-shape";
import { Children, createContext, isValidElement, useContext, useId, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";
import { day, monthYear, year } from "../format";
import { DATA_STATE_WORDS, type DataStateKind } from "../dataStates";
import { useElementWidth, usePhone } from "../useMedia";
import { cx } from "../util";
import { describePoint, linearTicks, plain, nearest, onAxis, runs, stepIndex, yDomainOf, type PlotPoint, type StepKey, type XSpec, type YSpec } from "./model";
import { logTickLabel, logTicks } from "../series/series";
import s from "./plot.module.css";

const r2 = (v: number) => Math.round(v * 100) / 100;
const DAY = 86_400_000;

/* ---------------------------------------------------------------- PlotSync: hover, pin and window shared by plots */

export interface PlotSyncValue {
  /** Where the crosshair is, on x (ms on a time axis); null when nothing is hovered. */
  hoverT: number | null;
  /** The pinned sample: which plot and where on x. */
  pinned: { plot: string; t: number } | null;
}
type SyncCtx = PlotSyncValue & { setHover: (t: number | null, p?: PlotPoint | null) => void; setPinned: (v: PlotSyncValue["pinned"], p?: PlotPoint | null) => void };
const Sync = createContext<SyncCtx | null>(null);

export interface PlotSyncProps {
  /** Controlled state (with `onChange`); uncontrolled when absent. */
  value?: PlotSyncValue;
  onChange?: (v: PlotSyncValue) => void;
  /** The hovered or pinned sample's place, for a linked chart (lon, lat), or null when it leaves. */
  onPlace?: (lonlat: [number, number] | null) => void;
  children: ReactNode;
}

/** Plots inside share one crosshair and one pinned sample: hover a date on one and every plot reads its sample there. */
export function PlotSync({ value, onChange, onPlace, children }: PlotSyncProps) {
  const [own, setOwn] = useState<PlotSyncValue>({ hoverT: null, pinned: null });
  const v = value ?? own;
  const ctx = useMemo<SyncCtx>(() => {
    const put = (next: PlotSyncValue) => {
      if (value === undefined) setOwn(next);
      onChange?.(next);
    };
    return {
      ...v,
      setHover: (t, p) => {
        if (v.hoverT === t) return;
        put({ ...v, hoverT: t });
        if (!v.pinned) onPlace?.(p?.place ?? null);
      },
      setPinned: (pin, p) => {
        put({ ...v, pinned: pin });
        onPlace?.(pin ? (p?.place ?? null) : null);
      },
    };
  }, [v, value, onChange, onPlace]);
  return <Sync.Provider value={ctx}>{children}</Sync.Provider>;
}

/* ---------------------------------------------------------------- Plot */

export interface PlotGeometry {
  id: string;
  width: number;
  /** The plot area, px. */
  left: number;
  right: number;
  top: number;
  bottom: number;
  x: (t: number) => number;
  y: (v: number) => number;
  xSpec: Required<XSpec>;
  ySpec: { type: "linear" | "log"; domain: [number, number]; unit?: string; format: (v: number) => string };
  /** The sample under the crosshair (hover or keyboard), and the pinned one. */
  focus: PlotPoint | null;
  pinned: PlotPoint | null;
}
const Geo = createContext<PlotGeometry | null>(null);
/** The plot's scales and state, for a custom mark. */
export function usePlot(): PlotGeometry {
  const g = useContext(Geo);
  if (!g) throw new Error("usePlot: a mark must be inside <Plot>");
  return g;
}

export interface PlotProps {
  /** Accessible name: "Particulate domoic acid at Monterey Wharf". */
  label: string;
  /** The inspectable samples, sorted by x: what the crosshair snaps to and the keyboard steps through. */
  data: PlotPoint[];
  x: XSpec;
  y?: YSpec;
  /** Plot height in px, or one for wide screens and one for phones. */
  height?: number | { desktop: number; phone: number };
  /** Room for the axes, px. */
  margin?: Partial<{ left: number; right: number; top: number; bottom: number }>;
  /** One sentence for assistive tech; defaults to the count and span. */
  describe?: (points: PlotPoint[]) => string;
  /** What a sample says in the live region and the default tooltip. */
  describePoint?: (p: PlotPoint) => string;
  /** A sample pinned or unpinned (click, tap, Enter; Escape clears). */
  onSelect?: (p: PlotPoint | null) => void;
  className?: string;
  children?: ReactNode;
  "data-testid"?: string;
}

/**
 * One interactive plot: d3 scales, SVG marks, an HTML tooltip. One tab stop. Mouse: the crosshair snaps to the
 * nearest sample within 24 px; click pins it. Touch: a tap pins the nearest sample; a long press (250 ms) then
 * a drag scrubs, so a vertical swipe still scrolls the page. Keyboard: Left/Right step sample by sample,
 * PageUp/PageDown a month, Home/End the first and last, Enter pins, Escape clears; each step is read out by a
 * polite live region. Data never tweens: a new window or a new variable is a cut.
 */
export function Plot({ label, data, x: xs, y: ys = {}, height = 280, margin, describe, describePoint: say, onSelect, className, children, ...rest }: PlotProps) {
  const own = useId();
  const id = `plot${own.replace(/[^a-zA-Z0-9]/g, "")}`;
  const phone = usePhone();
  const [setEl, width] = useElementWidth<HTMLDivElement>(640);
  const H = typeof height === "number" ? height : phone ? height.phone : height.desktop;
  const m = { left: 56, right: 12, top: 12, bottom: 28, ...margin };
  const left = m.left, right = Math.max(left + 40, width - m.right), top = m.top, bottom = top + H;
  const time = (xs.type ?? "time") === "time";
  const yType = ys.type ?? "linear";
  const yDomain = useMemo(() => ys.domain ?? yDomainOf(data, yType), [ys.domain, data, yType]);
  const geo = useMemo(() => {
    const sx = scaleLinear().domain(xs.domain).range([left, right]);
    const sy = yType === "log" ? scaleLog().domain(yDomain).range([bottom, top]).clamp(true) : scaleLinear().domain(yDomain).range([bottom, top]).clamp(true);
    return { x: (t: number) => r2(sx(t)), y: (v: number) => r2(sy(v)), invert: (px: number) => sx.invert(px) };
  }, [xs.domain, yDomain, yType, left, right, top, bottom]);
  const format = ys.format ?? plain;
  const sayPoint = say ?? ((p: PlotPoint) => describePoint(p, { unit: ys.unit, time, format }));

  // only the samples in view can be inspected
  const shown = useMemo(() => data.filter((p) => p.t >= xs.domain[0] && p.t <= xs.domain[1]), [data, xs.domain]);

  const outer = useContext(Sync);
  const [local, setLocal] = useState<PlotSyncValue>({ hoverT: null, pinned: null });
  const sync: SyncCtx = outer ?? {
    ...local,
    setHover: (t) => setLocal((v) => (v.hoverT === t ? v : { ...v, hoverT: t })),
    setPinned: (pin) => setLocal((v) => ({ ...v, pinned: pin })),
  };
  const near = (t: number | null, tolPx: number): PlotPoint | null => {
    if (t === null) return null;
    const i = nearest(shown, t);
    if (i < 0) return null;
    return Math.abs(geo.x(shown[i].t) - geo.x(t)) <= tolPx ? shown[i] : null;
  };
  const focus = near(sync.hoverT, 24);
  const pinned = sync.pinned && sync.pinned.plot === id ? near(sync.pinned.t, 0.5) : null;
  const [announce, setAnnounce] = useState("");

  const pin = (p: PlotPoint | null) => {
    sync.setPinned(p ? { plot: id, t: p.t } : null, p);
    onSelect?.(p);
    setAnnounce(p ? `Pinned. ${sayPoint(p)}` : "Cleared.");
  };
  const at = (e: { clientX: number; currentTarget: Element }) => geo.invert(e.clientX - e.currentTarget.getBoundingClientRect().left);

  // ------------------------------------------------------------ pointer: mouse hovers and clicks; touch taps, or long-presses to scrub
  const touch = useRef<{ x: number; y: number; timer: number; scrub: boolean } | null>(null);
  const onMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (e.pointerType === "touch") {
      const t = touch.current;
      if (!t) return;
      if (!t.scrub) {
        // moved before the press was long enough: the page is scrolling, not the plot
        if (Math.abs(e.clientX - t.x) > 8 || Math.abs(e.clientY - t.y) > 8) {
          window.clearTimeout(t.timer);
          touch.current = null;
        }
        return;
      }
    }
    const p = near(at(e), 24);
    sync.setHover(p ? p.t : null, p);
  };
  const onDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (e.pointerType !== "touch") return;
    const el = e.currentTarget;
    const pid = e.pointerId;
    const x0 = at(e);
    touch.current = {
      x: e.clientX,
      y: e.clientY,
      scrub: false,
      timer: window.setTimeout(() => {
        if (!touch.current) return;
        touch.current.scrub = true;
        try {
          el.setPointerCapture(pid);
        } catch {
          // the pointer already left
        }
        el.dataset.scrubbing = "";
        const p = near(x0, 48);
        sync.setHover(p ? p.t : null, p);
      }, 250),
    };
  };
  const onUp = (e: ReactPointerEvent<HTMLDivElement>) => {
    const p = near(at(e), e.pointerType === "touch" ? 48 : 24);
    if (e.pointerType === "touch") {
      const t = touch.current;
      touch.current = null;
      delete e.currentTarget.dataset.scrubbing;
      if (!t) return;
      window.clearTimeout(t.timer);
      // a scrub ends where it is (the crosshair stays); a tap pins the nearest sample
      if (t.scrub) return;
    }
    if (!p) return pin(null);
    pin(pinned && pinned.t === p.t ? null : p);
  };
  const onCancel = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (touch.current) window.clearTimeout(touch.current.timer);
    touch.current = null;
    delete e.currentTarget.dataset.scrubbing;
  };

  // ------------------------------------------------------------ keyboard
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const k: StepKey | null = e.key === "ArrowLeft" ? "prev" : e.key === "ArrowRight" ? "next" : e.key === "Home" ? "first" : e.key === "End" ? "last" : e.key === "PageUp" ? "pagePrev" : e.key === "PageDown" ? "pageNext" : null;
    if (k) {
      e.preventDefault();
      const from = focus ? shown.indexOf(focus) : pinned ? shown.indexOf(pinned) : -1;
      const i = stepIndex(shown, from, k, time);
      if (i < 0) return setAnnounce("No samples in this window.");
      sync.setHover(shown[i].t, shown[i]);
      setAnnounce(sayPoint(shown[i]));
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      if (focus) pin(pinned && pinned.t === focus.t ? null : focus);
    } else if (e.key === "Escape") {
      if (!pinned && sync.hoverT === null) return;
      e.preventDefault();
      sync.setHover(null, null);
      if (pinned) pin(null);
    }
  };

  const value: PlotGeometry = {
    id,
    width,
    left,
    right,
    top,
    bottom,
    x: geo.x,
    y: geo.y,
    xSpec: { type: time ? "time" : "linear", domain: xs.domain },
    ySpec: { type: yType, domain: yDomain, unit: ys.unit, format },
    focus,
    pinned,
  };
  const svgKids: ReactNode[] = [];
  const htmlKids: ReactNode[] = [];
  Children.forEach(children, (c) => {
    if (isValidElement(c) && (c.type as { plotLayer?: string }).plotLayer === "html") htmlKids.push(c);
    else svgKids.push(c);
  });
  const summary = describe ? describe(shown) : shown.length ? `${label}: ${shown.length} samples, ${describeX(shown[0].t, time)} to ${describeX(shown[shown.length - 1].t, time)}.` : `${label}: no samples in this window.`;
  return (
    <Geo.Provider value={value}>
      <div
        ref={setEl}
        className={cx(s.plot, className)}
        role="group"
        aria-roledescription="chart"
        aria-label={label}
        aria-describedby={`${id}-d`}
        tabIndex={0}
        onKeyDown={onKey}
        onPointerMove={onMove}
        onPointerDown={onDown}
        onPointerUp={onUp}
        onPointerCancel={onCancel}
        onPointerLeave={(e) => e.pointerType !== "touch" && sync.setHover(null, null)}
        onBlur={() => !pinned && sync.setHover(null, null)}
        data-plot={id}
        data-focus-t={focus ? focus.t : undefined}
        data-pinned-t={pinned ? pinned.t : undefined}
        data-testid={rest["data-testid"]}
      >
        <svg className={s.svg} width={width} height={bottom + m.bottom} viewBox={`0 0 ${width} ${bottom + m.bottom}`} aria-hidden="true" focusable="false">
          {svgKids}
        </svg>
        {htmlKids}
        <p id={`${id}-d`} className={s.vh}>
          {summary} Arrow keys step through the samples; Enter pins one; Escape clears.
        </p>
        <p className={s.vh} role="status" aria-live="polite" data-testid={rest["data-testid"] ? `${rest["data-testid"]}-live` : undefined}>
          {announce}
        </p>
      </div>
    </Geo.Provider>
  );
}
const describeX = (t: number, time: boolean) => (time ? day(t, { year: true }) : plain(t));

/* ---------------------------------------------------------------- axes and grid */

function timeTicks([a, b]: [number, number], n: number): Array<{ t: number; label: string }> {
  const span = b - a;
  const out: Array<{ t: number; label: string }> = [];
  const d0 = new Date(a);
  if (span > 2.5 * 365 * DAY) {
    const stepY = Math.max(1, Math.ceil(span / (365.25 * DAY) / n));
    for (let y = d0.getUTCFullYear() + 1; ; y += 1) {
      const t = Date.UTC(y, 0, 1);
      if (t > b) break;
      if (y % stepY === 0) out.push({ t, label: year(t + DAY) });
    }
    return out;
  }
  if (span > 75 * DAY) {
    const months = span / (30.44 * DAY);
    const stepM = [1, 2, 3, 6].find((k) => months / k <= n) ?? 12;
    for (let y = d0.getUTCFullYear(), mo = d0.getUTCMonth() + 1; ; mo += 1) {
      const t = Date.UTC(y, mo, 1);
      if (t > b) break;
      const d = new Date(t);
      if (d.getUTCMonth() % stepM === 0) out.push({ t, label: d.getUTCMonth() === 0 ? year(t + DAY) : monthYear(t + DAY).split(" ")[0] });
    }
    return out;
  }
  const stepD = [1, 2, 7, 14].find((k) => span / DAY / k <= n) ?? 30;
  for (let t = Math.ceil(a / DAY) * DAY, i = 0; t <= b; t += DAY, i += 1) if (i % stepD === 0) out.push({ t, label: day(t + DAY / 2) });
  return out;
}

export interface AxisXProps {
  /** A title under the axis ("km from the port"). */
  title?: string;
  /** About how many ticks (fewer are drawn when labels would collide). */
  ticks?: number;
  format?: (t: number) => string;
}
export function AxisX({ title, ticks, format }: AxisXProps) {
  const g = usePlot();
  const n = ticks ?? Math.max(2, Math.floor((g.right - g.left) / 96));
  const list = g.xSpec.type === "time" && !format ? timeTicks(g.xSpec.domain, n) : linearTicks(g.xSpec.domain, n).map((t) => ({ t, label: (format ?? plain)(t) }));
  // labels never collide with each other or run off the plot's right edge
  let lastEnd = -Infinity;
  return (
    <g className={s.axis} data-axis="x">
      <line x1={g.left} x2={g.right} y1={g.bottom} y2={g.bottom} className={s.axisLine} />
      {list.map(({ t, label }) => {
        const px = g.x(t);
        const w = label.length * 8 + 8;
        const show = px - w / 2 > lastEnd && px + w / 2 <= g.width;
        if (show) lastEnd = px + w / 2;
        return (
          <g key={t}>
            <line x1={px} x2={px} y1={g.bottom} y2={g.bottom + 5} className={s.tick} />
            {show ? (
              <text x={px} y={g.bottom + 20} textAnchor="middle" className={s.tickLabel}>
                {label}
              </text>
            ) : null}
          </g>
        );
      })}
      {title ? (
        <text x={g.right} y={g.bottom + 20} textAnchor="end" className={s.axisTitle} dy={18}>
          {title}
        </text>
      ) : null}
    </g>
  );
}

function yTickList(g: PlotGeometry, count?: number): number[] {
  const all = g.ySpec.type === "log" ? logTicks(g.ySpec.domain) : linearTicks(g.ySpec.domain, count ?? Math.max(2, Math.floor((g.bottom - g.top) / 56)));
  // at least 18 px between labels
  const every = Math.max(1, Math.ceil(18 / Math.max(1, (g.bottom - g.top) / Math.max(1, all.length - 1))));
  return all.filter((_, i) => (all.length - 1 - i) % every === 0);
}

export interface AxisYProps {
  /** Above the axis, left-aligned: "pDA, ng/mL". Never rotated. */
  title?: string;
  ticks?: number;
}
export function AxisY({ title, ticks }: AxisYProps) {
  const g = usePlot();
  const fmt = g.ySpec.type === "log" ? logTickLabel : g.ySpec.format;
  return (
    <g className={s.axis} data-axis="y">
      {yTickList(g, ticks).map((v) => (
        <text key={v} x={g.left - 8} y={g.y(v)} dy="0.32em" textAnchor="end" className={s.tickLabel}>
          {fmt(v)}
        </text>
      ))}
      {title ? (
        <text x={0} y={g.top - 2} className={s.axisTitle}>
          {title}
        </text>
      ) : null}
    </g>
  );
}

export function GridY({ ticks }: { ticks?: number }) {
  const g = usePlot();
  return (
    <g data-grid="y">
      {yTickList(g, ticks).map((v) => (
        <line key={v} x1={g.left} x2={g.right} y1={g.y(v)} y2={g.y(v)} className={s.grid} />
      ))}
    </g>
  );
}

/* ---------------------------------------------------------------- marks */

export interface LineProps {
  data: PlotPoint[];
  /** Samples further apart than this break the line (a gap is drawn as a gap). Default 45. */
  gapDays?: number;
  /** measured: solid ink; forecast or estimated: dashed (computed, never solid). */
  kind?: DataStateKind;
}
/** Straight segments between consecutive samples. Never smoothed; never joined across a gap, a zero or a censored value. */
export function Line({ data, gapDays = 45, kind = "measured" }: LineProps) {
  const g = usePlot();
  const gap = g.xSpec.type === "time" ? gapDays * DAY : Infinity;
  const path = d3line<PlotPoint>()
    .x((p) => g.x(p.t))
    .y((p) => g.y(p.v as number));
  return (
    <g clipPath={`url(#${g.id}-clip)`} data-line={kind}>
      <clipPath id={`${g.id}-clip`}>
        <rect x={g.left} y={g.top - 6} width={Math.max(0, g.right - g.left)} height={g.bottom - g.top + 12} />
      </clipPath>
      {runs(data, gap).map((run, i) => (run.length > 1 ? <path key={i} d={path(run) ?? ""} className={cx(s.line, kind !== "measured" && kind !== "historical" && s.lineModel)} /> : null))}
    </g>
  );
}

/** One sample's mark, by its state: shape first. Values without a place on the axis sit on the baseline strip. */
function PointMark({ p, x, y, base, r = 3.5 }: { p: PlotPoint; x: number; y: number; base: number; r?: number }) {
  const kind = p.kind ?? "measured";
  const on = onAxis(p);
  const cy = on ? y : base;
  switch (kind) {
    case "measured":
      return <circle cx={x} cy={cy} r={r} className={s.ptFill} />;
    case "historical":
      return <circle cx={x} cy={cy} r={r - 0.5} className={s.ptFill} />;
    case "reported-zero":
      return <rect x={x - r} y={base - r} width={r * 2} height={r * 2} className={s.ptHollow} />;
    case "below-detection":
      return <path d={`M${x - r - 0.5} ${cy - r}h${r * 2 + 1}L${x} ${cy + r}z`} className={s.ptHollow} />;
    case "estimated":
      return <path d={`M${x} ${cy - r - 1}L${x + r + 1} ${cy}L${x} ${cy + r + 1}L${x - r - 1} ${cy}z`} className={s.ptHollow} />;
    case "forecast":
    case "unverified":
      return <circle cx={x} cy={cy} r={r} className={s.ptHollow} />;
    case "stale":
      return <circle cx={x} cy={cy} r={r} className={s.ptDash} />;
    default:
      // missing, not collected, unavailable, outside coverage: a small cross on the baseline, never a value
      return <path d={`M${x - r} ${base - r}l${r * 2} ${r * 2}M${x + r} ${base - r}l${-r * 2} ${r * 2}`} className={s.ptCross} />;
  }
}

export interface PointsProps {
  data: PlotPoint[];
  /** Mark radius, px. */
  r?: number;
}
/** Every sample as its state's mark: measured solid, reported zero a hollow square on the baseline, below detection a hollow triangle, model hollow. */
export function Points({ data, r }: PointsProps) {
  const g = usePlot();
  const [a, b] = g.xSpec.domain;
  return (
    <g data-points="">
      {data
        .filter((p) => p.t >= a && p.t <= b)
        .map((p, i) => (
          <PointMark key={p.id ?? `${p.t}-${i}`} p={p} x={g.x(p.t)} y={onAxis(p) ? g.y(p.v as number) : g.bottom} base={g.bottom} r={r} />
        ))}
    </g>
  );
}

export interface NoDataProps {
  /** Spans with nothing collected, on x. */
  spans: Array<[number, number]>;
  /** The words written in the span when it is wide enough: "not collected". */
  words?: string;
}
/** A span where nothing was collected: hatched and said in words, never an empty stretch of axis. */
export function NoData({ spans, words = DATA_STATE_WORDS["not-collected"].word }: NoDataProps) {
  const g = usePlot();
  const [a, b] = g.xSpec.domain;
  return (
    <g data-nodata="">
      <defs>
        <pattern id={`${g.id}-hatch`} width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="8" className={s.hatch} />
        </pattern>
      </defs>
      {spans
        .filter(([f, t]) => t > a && f < b)
        .map(([f, t], i) => {
          const x0 = g.x(Math.max(a, f)), x1 = g.x(Math.min(b, t));
          return (
            <g key={i}>
              <rect x={x0} y={g.top} width={Math.max(0, x1 - x0)} height={g.bottom - g.top} fill={`url(#${g.id}-hatch)`} />
              {x1 - x0 > words.length * 8 + 16 ? (
                <text x={(x0 + x1) / 2} y={(g.top + g.bottom) / 2} textAnchor="middle" dy="0.32em" className={s.noDataWords}>
                  {words}
                </text>
              ) : null}
            </g>
          );
        })}
    </g>
  );
}

export interface ThresholdProps {
  value: number;
  /** Named, always: "Action level 30 ppm". Written above the line at the plot's left, clear of the data's latest end. */
  label: string;
}
/** A reference line with its name. Only on the scale it belongs to; never drawn without its name. */
export function Threshold({ value, label }: ThresholdProps) {
  const g = usePlot();
  const [lo, hi] = g.ySpec.domain;
  if (value < lo || value > hi) return null;
  const y = g.y(value);
  return (
    <g data-threshold="">
      <line x1={g.left} x2={g.right} y1={y} y2={y} className={s.threshold} />
      <text x={g.left + 6} y={y - 6} className={s.thresholdLabel}>
        {label}
      </text>
    </g>
  );
}

/** The present moment on a time axis: what lies right of it is a forecast, never a measurement. */
export function NowRule({ t, label = "Now" }: { t: number; label?: string }) {
  const g = usePlot();
  if (t < g.xSpec.domain[0] || t > g.xSpec.domain[1]) return null;
  const x = g.x(t);
  const flip = x > g.right - 40;
  return (
    <g data-now="">
      <line x1={x} x2={x} y1={g.top} y2={g.bottom} className={s.now} />
      <text x={flip ? x - 6 : x + 6} y={g.top + 12} textAnchor={flip ? "end" : "start"} className={s.nowLabel}>
        {label}
      </text>
    </g>
  );
}

export interface CrosshairProps {
  /** A horizontal rule at the sample's value too (off by default). */
  horizontal?: boolean;
}
/** The rule at the sample under the pointer or the keyboard, and a ring on the sample. It follows at once: no easing. */
export function Crosshair({ horizontal }: CrosshairProps) {
  const g = usePlot();
  const p = g.focus;
  if (!p) return null;
  const x = g.x(p.t);
  const y = onAxis(p) ? g.y(p.v as number) : g.bottom;
  return (
    <g data-crosshair="" pointerEvents="none">
      <line x1={x} x2={x} y1={g.top} y2={g.bottom} className={s.cross} />
      {horizontal && onAxis(p) ? <line x1={g.left} x2={g.right} y1={y} y2={y} className={s.cross} /> : null}
      <circle cx={x} cy={y} r={7} className={s.focusRing} />
    </g>
  );
}

/** The pinned sample's mark: a heavier ring that stays when the pointer leaves. */
export function Selection() {
  const g = usePlot();
  const p = g.pinned;
  if (!p) return null;
  const x = g.x(p.t);
  const y = onAxis(p) ? g.y(p.v as number) : g.bottom;
  return (
    <g data-selection="" pointerEvents="none">
      <line x1={x} x2={x} y1={g.top} y2={g.bottom} className={s.pinRule} />
      <rect x={x - 9} y={y - 9} width={18} height={18} className={s.pinRing} />
    </g>
  );
}

export interface TipProps {
  /** What the tooltip says about a sample; defaults to its date, value and state. */
  render?: (p: PlotPoint) => ReactNode;
}
/**
 * The tooltip for the sample under the crosshair (or the pinned one). It stands beside the sample, on the side
 * with room, and never covers it. Decorative for assistive tech: the same words go through the live region.
 */
export function Tip({ render }: TipProps) {
  const g = usePlot();
  const p = g.focus ?? g.pinned;
  if (!p) return null;
  const x = g.x(p.t);
  const y = onAxis(p) ? g.y(p.v as number) : g.bottom;
  const flip = x > (g.left + g.right) / 2;
  const kind = p.kind ?? "measured";
  return (
    <div className={s.tip} data-flip={flip || undefined} style={flip ? { right: g.width - x + 12, top: Math.max(0, y - 20) } : { left: x + 12, top: Math.max(0, y - 20) }} aria-hidden="true" data-testid="plot-tip">
      {render ? (
        render(p)
      ) : (
        <>
          <span className={s.tipValue}>{p.raw ?? (p.v !== null && onAxis(p) ? `${g.ySpec.format(p.v)}${g.ySpec.unit ? ` ${g.ySpec.unit}` : ""}` : DATA_STATE_WORDS[kind].word)}</span>
          <span className={s.tipMeta}>
            {g.xSpec.type === "time" ? day(p.t, { year: true }) : plain(p.t)} · {DATA_STATE_WORDS[kind].word.toLowerCase()}
          </span>
        </>
      )}
    </div>
  );
}
(Tip as unknown as { plotLayer: string }).plotLayer = "html";
