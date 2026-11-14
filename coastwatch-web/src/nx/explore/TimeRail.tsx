"use client";

import { useId, useState, type KeyboardEvent, type PointerEvent, type ReactNode } from "react";
import { Button, day, Micro, range, sig, time, useElementWidth } from "@/nx/ds";
import type { CurrentFrame, ExploreFrame, ExploreModel, ExploreProduct, ProductId, Track } from "./model";
import { axisLabels, axisLength, axisX, frameCount, hourX, nearestFrame, nowX, productSpan } from "./timeline";
import { isOld, LAYER_LABEL, layerKindWord, oldMark } from "./words";
import s from "./explore.module.css";

/**
 * The time rail (README §9 TimeRail; the availability ribbon grafted from the Tide concept): the active layer's
 * time tracks on a day axis, with the weight. Bars are each day's published coverage (how much of the coastal
 * ocean had a value); solid bars are measured, hatched boxes are model or outlook frames, a past model run is
 * drawn lighter with the words "past run"; the now line crosses every row. Choosing a frame is a cut (T26).
 * With currents on, the hours strip shows every published radar hour. Every other layer is one quiet line of
 * what it holds and when, each a way to switch to it, so the rail is short and its labels never collide.
 */
// the label column, as the stylesheet sets it (.railLabel is 96 px): the track, the axis and the now line share it
const LABEL_W = 96;
const ROW = 36;
/** a layer that is not the one on show: a thin row, so the three families stay compact (round 2, E) */
const ROW_QUIET = 20;
/** a model's step row: two lines of label room ("WCOFS / nowcast") */
const STEP_ROW = 32;
const ROW_ACTIVE = 56;

type Props = {
  model: ExploreModel;
  product: ExploreProduct;
  frameId: string;
  onFrame: (product: ExploreProduct, frame: ExploreFrame) => void;
  /** make another layer the active one (from the availability line) */
  onProduct: (id: ProductId) => void;
  currentsOn: boolean;
  currentId: string | null;
  onCurrent: (f: CurrentFrame) => void;
  playing: boolean;
  canPlay: boolean;
  onPlay: () => void;
  /** round 2: rows the page adds to a family (the live model and estimate layers) */
  extra?: Partial<Record<"obs" | "est" | "fc", ReactNode>>;
  /** what is on show when it is not one of the published frames (a model step, an estimate) */
  stamp?: string | null;
  /** the live model steps and the estimate's day, drawn at their true times on the shared axis (round 2) */
  ocean?: OceanRail | null;
  /** said under the axis when it is zoomed to one family's span */
  axisNote?: string | null;
  /** the selected family's own span, used as the axis when the shared one is too narrow for its 3-hour steps */
  zoomTo?: [string, string] | null;
};

export type OceanStepLite = { id: string; kind: "nowcast" | "forecast"; valid_time: string };
export type OceanRail = {
  nowcast: OceanStepLite[];
  forecast: OceanStepLite[];
  selected: string | null;
  failed: Set<string>;
  onStep: (id: string) => void;
  estimate: { date: string; words: string; on: boolean; onChoose: () => void } | null;
};

/**
 * A model's steps at their own valid times on the shared day axis: 3-hour ticks, nothing drawn between the first and
 * last step that is not a step, a step that could not be read drawn as an empty outline.
 */
function AxisStepRow({ model, label, steps, selected, failed, onStep, width, testid }: { model: ExploreModel; label: string; steps: OceanStepLite[]; selected: string | null; failed: Set<string>; onStep: (id: string) => void; width: number; testid: string }) {
  if (!steps.length) return null;
  const T = Math.max(40, width - LABEL_W);
  const px = (t: string) => hourX(model.axis, { time: t }) * T;
  // a 3-hour step's width on this axis (never under 2 px)
  const tick = Math.max(2, (T / (axisLength(model.axis) * 8)) - 1);
  const last = steps[steps.length - 1];
  const lx = px(last.valid_time) + tick / 2;
  // the "valid to" words stand right of the last step when there is room, else left of the first, else shorter, else
  // not at all (the time head says the step on show): never over the ticks, never past the row
  const fx = px(steps[0].valid_time) - tick / 2 - 4;
  const long = last.kind === "nowcast" ? `to ${day(last.valid_time)}, ${time(last.valid_time)}` : `valid to ${day(last.valid_time)}, ${time(last.valid_time)}`;
  const short = `to ${day(last.valid_time)}`;
  const w = (t: string) => t.length * 7.2;
  const sel = steps.find((x) => x.id === selected);
  const selectedWords = sel ? `${sel.kind === "nowcast" ? "Nowcast" : "Forecast"}, valid ${day(sel.valid_time)}, ${time(sel.valid_time)}` : "no step chosen";
  const placed = lx + 4 + w(long) < T ? { text: long, x: lx + 4, anchor: "start" as const } : fx - w(long) > 0 ? { text: long, x: fx, anchor: "end" as const } : lx + 4 + w(short) < T ? { text: short, x: lx + 4, anchor: "start" as const } : fx - w(short) > 0 ? { text: short, x: fx, anchor: "end" as const } : null;
  return (
    <div className={s.railRow} style={{ height: STEP_ROW }} data-testid={testid}>
      <span className={s.railLabel}>{label}</span>
      <div className={s.railTrack} style={{ overflow: "hidden" }}>
        <svg width={T} height={STEP_ROW} className={s.railSvg} aria-hidden="true">
          <line x1={0} x2={T} y1={STEP_ROW - 0.5} y2={STEP_ROW - 0.5} className={s.railBase} />
          <line x1={lx} x2={lx} y1={2} y2={STEP_ROW - 1} className={s.railLatest} />
          {placed ? (
            <text x={placed.x} y={14} textAnchor={placed.anchor} className={s.railLatestText}>
              {placed.text}
            </text>
          ) : null}
        </svg>
        {/* one control for the whole row (a slider over the steps): a tap picks the nearest step, the arrow keys step
            through the served ones; the 3-hour ticks are drawn only, so no target is ever narrower than its neighbour */}
        <svg width={T} height={STEP_ROW} className={s.railSvg} style={{ position: "absolute", left: 0, top: 0, pointerEvents: "none" }} aria-hidden="true">
          {steps.map((st) => (
            <rect key={st.id} x={px(st.valid_time) - tick / 2} y={4} width={tick} height={STEP_ROW - 9} rx={1} className={failed.has(st.id) ? s.axisTickFailed : st.id === selected ? s.axisTickOn : s.axisTick} data-testid={`nx-step-${st.id}`} data-on={st.id === selected ? "1" : "0"} data-failed={failed.has(st.id) ? "1" : undefined} />
          ))}
        </svg>
        <div
          role="slider"
          tabIndex={0}
          aria-label={`${label} steps`}
          aria-valuemin={0}
          aria-valuemax={steps.length - 1}
          aria-valuenow={Math.max(0, steps.findIndex((x) => x.id === selected))}
          aria-valuetext={selectedWords}
          className={s.axisSlider}
          data-testid={`${testid}-control`}
          onPointerDown={(e) => {
            const r = e.currentTarget.getBoundingClientRect();
            const x = e.clientX - r.left;
            let best = steps[0], d = Infinity;
            for (const st of steps) {
              const dd = Math.abs(px(st.valid_time) - x);
              if (dd < d) {
                d = dd;
                best = st;
              }
            }
            onStep(best.id);
          }}
          onKeyDown={(e) => {
            const ok = steps.filter((x) => !failed.has(x.id));
            if (!ok.length) return;
            const i = ok.findIndex((x) => x.id === selected);
            const to = e.key === "ArrowRight" || e.key === "ArrowUp" ? i + 1 : e.key === "ArrowLeft" || e.key === "ArrowDown" ? i - 1 : e.key === "Home" ? 0 : e.key === "End" ? ok.length - 1 : null;
            if (to === null) return;
            e.preventDefault();
            onStep(ok[Math.max(0, Math.min(ok.length - 1, to < 0 ? 0 : to))].id);
          }}
        />
      </div>
    </div>
  );
}

/** The estimate's one day, at its date on the axis, with its words beside it. */
function AxisDayRow({ model, label, date, words, on, onChoose, width, testid }: { model: ExploreModel; label: string; date: string; words: string; on: boolean; onChoose: () => void; width: number; testid: string }) {
  const T = Math.max(40, width - LABEL_W);
  const a = axisX(model.axis, date) * T, b = axisX(model.axis, date, 1) * T;
  const right = b + 260 < T;
  return (
    <div className={s.railRow} style={{ height: ROW_QUIET + 4 }} data-testid={testid}>
      <span className={s.railLabel}>{label}</span>
      <div className={s.railTrack} style={{ overflow: "hidden" }}>
        {/* design-lint-allow raw-button: the estimate's single day on the axis (a radio drawn as a bar) */}
        <button type="button" role="radio" aria-checked={on} aria-label={words} className={s.axisDay} style={{ left: a + 1, width: Math.max(4, b - a - 2) }} onClick={onChoose} data-testid={`${testid}-day`} />
        {/* a day before a zoomed axis: said so from the row's start, never cut mid-sentence */}
        <span className={s.axisDayWords} style={b < 0 ? { left: 0, right: 0 } : right ? { left: b + 6, right: 0 } : { left: 0, right: T - a + 6 }}>
          {b < 0 ? `Before this axis: ${words}` : words}
        </span>
      </div>
    </div>
  );
}

/**
 * Round 2 (E): the rail in three labelled families, each layer on its own row with its own latest valid point.
 * Observations are measured (satellite days, radar hours); Analysis / Estimates are blends and reconstructions; Forecasts
 * are model runs and outlooks. A family with nothing published says so; nothing is drawn after a layer's last frame.
 */
type FamilyId = "obs" | "est" | "fc";
const FAMILIES: Array<{ id: FamilyId; label: string; empty: string }> = [
  { id: "obs", label: "Observations", empty: "No observation is published in this data run." },
  { id: "est", label: "Analysis / Estimates", empty: "Not available yet: gap-filled chlorophyll and sea-temperature analysis are not published." },
  { id: "fc", label: "Forecasts", empty: "No forecast is published in this data run." },
];
export const familyOf = (p: Pick<ExploreProduct, "kind">): FamilyId => (p.kind === "measured" ? "obs" : p.kind === "analysis" ? "est" : "fc");

/** An expired run in one line: its last run, its valid interval, and why it is shown. */
export function expiredWords(p: Pick<ExploreProduct, "runIssued" | "frames">): string {
  const v = p.frames.map((f) => [f.start, f.end]).flat().sort();
  const valid = v.length ? range(v[0], v[v.length - 1]) : "unknown";
  // a run whose issue date is not published is dated by its valid days alone (round 2, C)
  if (!p.runIssued) return `Last run available: valid ${valid} · expired · no newer run has been published, so this is not a current forecast.`;
  return `Last run ${day(p.runIssued)}, valid ${valid}. Expired: no newer run has been published, so this is not a current forecast.`;
}

function Hatch({ id, dense }: { id: string; dense?: boolean }) {
  return (
    <pattern id={id} patternUnits="userSpaceOnUse" width={dense ? 4 : 5} height={dense ? 4 : 5} patternTransform="rotate(45)">
      <line x1="0" y1="0" x2="0" y2={dense ? 4 : 5} className={s.hatchLine} />
    </pattern>
  );
}

function frameWords(f: ExploreFrame): string {
  const kind = f.kind === "measured" ? "measured" : f.kind === "analysis" ? "analysis" : f.past ? `${f.kind}, past run` : f.kind;
  const cov = f.volume != null && f.kind === "measured" ? `, values over ${Math.round(f.volume * 100)} % of the coastal ocean` : "";
  return `${f.label === "Latest" ? `Latest view, ${f.stamp}` : f.stamp}, ${kind}${cov}`;
}

function TrackRow({ model, product, track, active, height, frameId, onFrame, width, hatchId, hatchPastId, plain }: { model: ExploreModel; product: ExploreProduct; track: Track; active: boolean; height: number; frameId: string; onFrame: Props["onFrame"]; width: number; hatchId: string; hatchPastId: string; plain?: boolean }) {
  const frames = product.frames.filter((f) => f.track === track.id);
  const T = Math.max(40, width - LABEL_W);
  const H = height;
  const x = (d: string, frac = 0) => axisX(model.axis, d, frac) * T;
  const maxVol = Math.max(0.0001, ...track.days.map((d) => d.volume), ...frames.map((f) => (f.start === f.end ? (f.volume ?? 0) : 0)));
  const pick = (e: PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const f = nearestFrame(model.axis, frames, (e.clientX - r.left) / r.width);
    if (f) onFrame(product, f);
  };
  const keys = (e: KeyboardEvent<HTMLDivElement>) => {
    const i = frames.findIndex((f) => f.id === frameId);
    const to = e.key === "ArrowRight" || e.key === "ArrowDown" ? i + 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? i - 1 : e.key === "Home" ? 0 : e.key === "End" ? frames.length - 1 : null;
    if (to === null) return;
    e.preventDefault();
    const f = frames[Math.max(0, Math.min(frames.length - 1, to < 0 ? 0 : to))];
    if (f) onFrame(product, f);
  };
  const measured = product.kind === "measured";
  // hit areas never overlap (every control is its own top element at any width): a latest view's bracket is the
  // row's top strip, the single days below it, each exactly its own day
  const bracketed = measured && frames.some((f) => f.start !== f.end);
  const strip = Math.min(12, Math.floor(H / 2));
  const hitStyle = (f: ExploreFrame, a: number, b: number) =>
    measured && f.start !== f.end ? { left: a, width: Math.max(2, b - a), top: 0, height: strip } : { left: a, width: Math.max(2, b - a), top: bracketed ? strip + 1 : 0, bottom: 0 };
  const last = frames.map((f) => f.end).filter(Boolean).sort().pop() ?? null;
  // with the sources folded away, the row is named for what it shows, not the satellite
  const name = plain ? LAYER_LABEL[product.id] : track.id === "viirs" ? "VIIRS" : track.id === "olci" ? "OLCI" : product.id === "chl" ? "Chlorophyll" : product.id === "sst" ? "Temperature" : product.id === "charm" ? "C-HARM" : "Outlook";
  const pastRun = product.pastRun && (track.id === "charm" || track.id === "outlook");
  return (
    <div className={s.railRow} data-active={active ? "1" : "0"} style={{ height: H }} data-testid={`nx-rail-${track.id}`}>
      <span className={s.railLabel}>
        {name}
        {pastRun && height > ROW_QUIET ? <small className={s.railNote}>past run</small> : null}
      </span>
      <div className={s.railTrack} onPointerDown={pick} onKeyDown={keys} role="radiogroup" aria-label={`${name} frames`}>
        <svg width={T} height={H} className={s.railSvg} aria-hidden="true">
          <line x1={0} x2={T} y1={H - 0.5} y2={H - 0.5} className={s.railBase} />
          {track.days.map((d) => {
            const h = d.volume > 0 ? Math.max(2, (d.volume / maxVol) * (H - 12)) : 0;
            return h ? <rect key={d.date} x={x(d.date) + 1} y={H - h} width={Math.max(1, x(d.date, 1) - x(d.date) - 2)} height={h} className={measured ? s.barMeasured : s.barModel} fill={measured ? undefined : `url(#${hatchId})`} /> : null;
          })}
          {frames.map((f) => {
            const a = x(f.start), b = x(f.end, 1);
            const sel = f.id === frameId && active;
            if (!measured) {
              return <rect key={f.id} x={a + 1} y={4} width={Math.max(2, b - a - 2)} height={H - 8} fill={`url(#${f.past ? hatchPastId : hatchId})`} className={sel ? s.frameSel : f.past ? s.framePast : s.frameModel} />;
            }
            if (f.start !== f.end) {
              // a latest view: a bracket over the days its pixels come from
              return (
                <g key={f.id}>
                  <path d={`M${a + 1},${10} V4 H${b - 1} V10`} className={sel ? s.bracketSel : s.bracket} />
                  {sel ? <rect x={a + 0.75} y={1} width={Math.max(2, b - a - 1.5)} height={H - 2} className={s.frameSel} fill="none" /> : null}
                </g>
              );
            }
            return sel ? <rect key={f.id} x={a + 0.75} y={1} width={Math.max(3, b - a - 1.5)} height={H - 2} className={s.frameSel} fill="none" /> : null;
          })}
          {last && x(last, 1) >= 0 && x(last, 1) <= T ? (
            // the row's own latest valid point: nothing after it is drawn, and it says its date
            <g data-testid={`nx-rail-latest-${track.id}`}>
              <line x1={x(last, 1)} x2={x(last, 1)} y1={2} y2={H - 1} className={s.railLatest} />
              <text x={x(last, 1) + (x(last, 1) + 104 < T ? 4 : -4)} y={12} textAnchor={x(last, 1) + 104 < T ? "start" : "end"} className={s.railLatestText}>
                {measured || product.kind === "analysis" ? `latest ${day(last)}` : `valid to ${day(last)}`}
              </text>
            </g>
          ) : null}
        </svg>
        {frames.map((f) => {
          const a = x(f.start), b = x(f.end, 1);
          return (
            // design-lint-allow raw-button: the rail's frames and hours are invisible hit areas laid over an SVG track, sized and placed by the time axis (a radio group drawn as bars); no ds control draws on a time axis
            <button
              key={f.id}
              type="button"
              role="radio"
              aria-checked={f.id === frameId && active}
              tabIndex={f.id === frameId && active ? 0 : frames[0].id === f.id && !active ? 0 : -1}
              className={s.railHit}
              style={hitStyle(f, a, b)}
              aria-label={frameWords(f)}
              data-testid={`nx-frame-${f.id}`}
              onClick={() => onFrame(product, f)}
            />
          );
        })}
      </div>
    </div>
  );
}

function HoursStrip({ model, currentId, onCurrent, width, on }: { model: ExploreModel; currentId: string | null; onCurrent: (f: CurrentFrame) => void; width: number; on: boolean; nowAt?: number }) {
  // the radar's hours at their true times on the shared day axis (round 2): narrow bars at their real hours, one
  // slider control over the row (taps pick the nearest hour, arrows step), the 24 h mean its own button by the label
  const hours = (model.currents?.frames ?? []).filter((f) => !f.mean).sort((a, b) => a.time.localeCompare(b.time));
  const mean = model.currents?.frames.find((f) => f.mean) ?? null;
  const T = Math.max(40, width - LABEL_W);
  const H = ROW_ACTIVE - 16;
  const px = (f: CurrentFrame) => hourX(model.axis, f) * T;
  const hourW = Math.max(1, T / (axisLength(model.axis) * 24));
  const maxVol = Math.max(0.0001, ...hours.map((h) => h.volume ?? 0));
  const old = hours.length > 0 && isOld(hours[hours.length - 1].fresh);
  const sel = hours.find((h) => h.id === currentId) ?? null;
  const last = hours[hours.length - 1];
  const lx = last ? px(last) + hourW : 0;
  return (
    <div className={s.railRow} data-active={on ? "1" : "0"} data-off={on ? undefined : "1"} data-old={old ? "1" : "0"} style={{ height: H }} data-testid="nx-rail-hours">
      <span className={s.railLabel}>
        Currents
        {mean ? (
          // design-lint-allow raw-button: the 24 h mean is one more frame of the hours, a radio beside its row's name
          <button type="button" role="radio" aria-checked={on && currentId === mean.id} className={s.meanHit} onClick={() => onCurrent(mean)} aria-label={mean.stamp} data-testid={`nx-hour-${mean.id}`}>
            24 h mean
          </button>
        ) : (
          <small className={s.railNote} data-testid="nx-hours-date">
            {old ? <b className={s.railOld}>old</b> : "by the hour"}
          </small>
        )}
      </span>
      <div className={s.railTrack} style={{ overflow: "hidden" }}>
        <svg width={T} height={H} className={s.railSvg} aria-hidden="true" style={{ pointerEvents: "none" }}>
          <line x1={0} x2={T} y1={H - 0.5} y2={H - 0.5} className={s.railBase} />
          {hours.map((h) => {
            const v = h.volume ?? 0;
            const ht = Math.max(v > 0 ? 2 : 0, (v / maxVol) * (H - 14));
            return (
              <rect key={h.id} x={px(h)} y={H - ht} width={hourW} height={ht} className={on && h.id === currentId ? s.barSelHour : s.barMeasured} data-testid={`nx-hour-${h.id}`} />
            );
          })}
          {last ? (
            <>
              <line x1={lx} x2={lx} y1={2} y2={H - 1} className={s.railLatest} />
              <text x={lx + (lx + 110 < T ? 4 : -4)} y={12} textAnchor={lx + 110 < T ? "start" : "end"} className={s.railLatestText}>
                {`latest ${time(last.time)}${old ? " (old)" : ""}`}
              </text>
            </>
          ) : null}
        </svg>
        {hours.length ? (
          <div
            role="slider"
            tabIndex={0}
            aria-label="Current hours"
            aria-valuemin={0}
            aria-valuemax={hours.length - 1}
            aria-valuenow={Math.max(0, hours.findIndex((h) => h.id === currentId))}
            aria-valuetext={sel ? `${sel.stamp}, measured${sel.volume != null ? `, ${Math.round(sel.volume * 100)} % of the radar's area` : ""}` : "no hour chosen"}
            className={s.axisSlider}
            data-testid="nx-rail-hours-control"
            onPointerDown={(e) => {
              const r = e.currentTarget.getBoundingClientRect();
              const x = e.clientX - r.left;
              let best = hours[0], d = Infinity;
              for (const h of hours) {
                const dd = Math.abs(px(h) + hourW / 2 - x);
                if (dd < d) {
                  d = dd;
                  best = h;
                }
              }
              onCurrent(best);
            }}
            onKeyDown={(e) => {
              const i = hours.findIndex((h) => h.id === currentId);
              const to = e.key === "ArrowRight" ? i + 1 : e.key === "ArrowLeft" ? i - 1 : e.key === "Home" ? 0 : e.key === "End" ? hours.length - 1 : null;
              if (to === null) return;
              e.preventDefault();
              onCurrent(hours[Math.max(0, Math.min(hours.length - 1, to))]);
            }}
          />
        ) : null}
      </div>
    </div>
  );
}


/** The other layers in one line: what each holds and when, each a button that makes it the active layer. */
function Availability({ model, product, currentsShown, onProduct }: { model: ExploreModel; product: ExploreProduct; currentsShown: boolean; onProduct: (id: ProductId) => void }) {
  const others = model.products.filter((p) => p.id !== product.id);
  const hours = (model.currents?.frames ?? []).filter((f) => !f.mean);
  const items = others.map((p) => {
    const sp = productSpan(p);
    const when = sp ? range(sp.first, sp.last) : "nothing published";
    const days = sp && (p.kind === "measured" || p.kind === "analysis") && sp.days > 1 ? `, ${sp.days} days with a view` : "";
    return { id: p.id, text: `${LAYER_LABEL[p.id]} ${when}`, sub: `${layerKindWord(p)}${days}` };
  });
  if (!items.length && (currentsShown || !hours.length)) return null;
  return (
    <p className={s.railAlso} data-testid="nx-rail-also">
      <span className={s.railAlsoHead}>Also</span>
      {items.map((it, i) => (
        <span key={it.id}>
          <button type="button" className={s.railAlsoItem} onClick={() => onProduct(it.id)} data-testid={`nx-rail-also-${it.id}`}>
            {it.text}
          </button>{" "}
          <span className={s.railAlsoSub}>({it.sub})</span>
          {i < items.length - 1 || (!currentsShown && hours.length) ? <span aria-hidden="true"> · </span> : null}
        </span>
      ))}
      {!currentsShown && hours.length ? (
        <span>
          Currents {range(hours[0].date, hours[hours.length - 1].date)} <span className={s.railAlsoSub}>(measured, {hours.length} hours{product.kind === "model" || product.kind === "outlook" ? "; not over a model" : ""})</span>
        </span>
      ) : null}
    </p>
  );
}

/**
 * What is on show and the Play control: the rail's head. On a phone Explore sets it at the top of the sheet,
 * beside the layer switch, so stepping time never needs the sheet raised over the chart (review F-01).
 */
export function TimeHead({ model, product, frameId, currentsOn, currentId, playing, canPlay, onPlay, stamp }: Pick<Props, "model" | "product" | "frameId" | "currentsOn" | "currentId" | "playing" | "canPlay" | "onPlay" | "stamp">) {
  const frame = product.frames.find((f) => f.id === frameId);
  const cur = model.currents?.frames.find((f) => f.id === currentId) ?? null;
  const count = frameCount(product, frameId, currentsOn ? (model.currents?.frames ?? []) : [], currentId);
  return (
    <div className={s.railHead} data-testid="nx-time-head">
      <h2 className={s.railTitle}>Time</h2>
      <p className={s.railNow} data-testid="nx-frame-stamp">
        {stamp ? stamp : frame ? (frame.label === "Latest" ? `Latest view: ${frame.stamp}` : frame.stamp.charAt(0).toUpperCase() + frame.stamp.slice(1)) : "Chart · no data layer"}
        {count ? (
          <span className={s.railCount} data-testid="nx-frame-count">
            {" "}
            · {count}
          </span>
        ) : null}
        {currentsOn && cur ? (
          <span className={s.railNowSub}>
            Currents: {cur.mean ? cur.label : cur.stamp}
            {oldMark(cur.fresh)}
          </span>
        ) : null}
      </p>
      {canPlay ? (
        <Button variant="text" onClick={onPlay} aria-pressed={playing} data-testid="nx-play">
          {playing ? "Pause" : "Play"}
        </Button>
      ) : null}
    </div>
  );
}

export function TimeRail({ model: model0, product, frameId, onFrame, onProduct, currentsOn, currentId, onCurrent, playing, canPlay, onPlay, head = true, extra, stamp, ocean, axisNote, zoomTo }: Props & { /** false where the page sets the head itself (a phone) */ head?: boolean }) {
  const [ref, width] = useElementWidth<HTMLDivElement>(320);
  // one shared, true axis; zoomed to the selected family's span only when its 3-hour steps would be under 3 px
  const dense = !!zoomTo && (Math.max(40, width - LABEL_W) / axisLength(model0.axis)) / 8 < 3;
  const model: ExploreModel = dense && zoomTo ? { ...model0, axis: zoomTo } : model0;
  const note = dense && zoomTo ? (axisNote ?? `Axis zoomed to ${range(zoomTo[0], zoomTo[1])}`) : null;
  const uid = useId().replace(/:/g, "");
  const hatchId = `nxh-${uid}`;
  const hatchPastId = `nxhp-${uid}`;
  const T = Math.max(40, width - LABEL_W);
  const nx = nowX(model.axis, model.now) * T;
  const labels = axisLabels(model.axis, T, nx);
  const frame = product.frames.find((f) => f.id === frameId);
  const [more, setMore] = useState(false);
  const moreId = `nx-rail-more-${uid}`;
  // a frame chosen on another sensor's row (from a link) keeps that row in view
  const all = more || (!!frame && frame.track !== product.tracks[0]?.id);
  const tracks = all ? product.tracks : product.tracks.slice(0, 1);
  return (
    <section className={s.rail} ref={ref} aria-label="Time" data-testid="nx-timerail">
      {head ? <TimeHead model={model} product={product} frameId={frameId} currentsOn={currentsOn} currentId={currentId} playing={playing} canPlay={canPlay} onPlay={onPlay} stamp={stamp} /> : null}
      <div className={s.railBody}>
        <svg width={width} height={0} className={s.defs} aria-hidden="true">
          <defs>
            <Hatch id={hatchId} />
            <Hatch id={hatchPastId} dense />
          </defs>
        </svg>
        <div className={s.railAxis} style={{ marginLeft: LABEL_W, width: T }} aria-hidden="true">
          {labels.map((l) => (
            <span key={l.date} className={s.railDay} style={{ left: l.x }}>
              {l.text}
            </span>
          ))}
          {/* a quiet reference for today; each row says its own latest valid point (round 2, E: no generic "Now") */}
          <span className={s.railTodayLabel} style={{ left: Math.min(Math.max(nx, 24), T - 24) }}>
            Today
          </span>
        </div>
        {note ? (
          <p className={s.railAxisNote} data-testid="nx-rail-axis-note">
            {note}
          </p>
        ) : null}
        {FAMILIES.map((fam) => {
          const ps = model.products.filter((p) => familyOf(p) === fam.id);
          const showHours = fam.id === "obs" && !!model.currents && currentsOn;
          return (
            <div key={fam.id} className={s.railFamily} data-testid={`nx-rail-family-${fam.id}`}>
              {/* a family with nothing published is one line: its name and why (round 2, E) */}
              {!ps.length && !showHours && !extra?.[fam.id] && !(ocean && fam.id !== "obs") ? (
                <p className={s.railEmptyLine} data-testid={`nx-rail-empty-${fam.id}`}>
                  <Micro as="span" className={s.railFamilyHead}>{fam.label}</Micro> {fam.empty}
                </p>
              ) : (
                <Micro as="p" className={s.railFamilyHead}>{fam.label}</Micro>
              )}
              {ps.map((p) => {
                // the active layer's rows at full height; every other layer one thin row
                const isActive = p.id === product.id && frameId !== "";
                const ts = isActive ? tracks : p.tracks.slice(0, 1);
                return (
                  <div key={p.id}>
                    <div className={s.railRows}>
                      {ts.map((t, i) => (
                        <TrackRow key={t.id} model={model} product={p} track={t} active={isActive} height={isActive ? (i === 0 ? ROW_ACTIVE : ROW) : ROW_QUIET} frameId={frameId} onFrame={onFrame} width={width} hatchId={hatchId} hatchPastId={hatchPastId} plain={!all || !isActive} />
                      ))}
                    </div>
                    {p.pastRun && (p.kind === "model" || p.kind === "outlook") ? (
                      <p className={s.railExpired} data-testid={`nx-rail-expired-${p.id}`}>
                        {expiredWords(p)}
                      </p>
                    ) : null}
                  </div>
                );
              })}
              {ocean && fam.id === "est" ? (
                <>
                  <AxisStepRow model={model} label="WCOFS nowcast" steps={ocean.nowcast} selected={ocean.selected} failed={ocean.failed} onStep={ocean.onStep} width={width} testid="nx-rail-wcofs-nowcast" />
                  {ocean.estimate ? <AxisDayRow model={model} label="Chl estimate" date={ocean.estimate.date} words={ocean.estimate.words} on={ocean.estimate.on} onChoose={ocean.estimate.onChoose} width={width} testid="nx-rail-chl-estimate" /> : null}
                </>
              ) : null}
              {ocean && fam.id === "fc" ? <AxisStepRow model={model} label="WCOFS forecast" steps={ocean.forecast} selected={ocean.selected} failed={ocean.failed} onStep={ocean.onStep} width={width} testid="nx-rail-wcofs-forecast" /> : null}
              {extra?.[fam.id] ?? null}
              {showHours ? (
                <div className={s.hoursBlock}>
                  <HoursStrip model={model} currentId={currentId} onCurrent={onCurrent} width={width} on={currentsOn} nowAt={nx} />
                </div>
              ) : null}

            </div>
          );
        })}
      </div>
      <div className={s.railMore}>
        <Button variant="text" aria-expanded={more} aria-controls={moreId} onClick={() => setMore((v) => !v)} data-testid="nx-rail-more">
          {more ? "Hide sources and coverage" : "Sources and coverage"}
        </Button>
        <div id={moreId} hidden={!more} className={s.railMoreBody}>
          <p className={s.railKey}>
            {product.kind === "measured" ? "Bars: share of the coastal ocean with a value that day." : "Hatched: a model or outlook frame, the days it is valid for."} {product.pastRun ? "Past run: every day it covers is before today." : ""}
            {frame && frame.volume != null && frame.kind === "measured" ? ` This frame: values over ${sig(frame.volume * 100, 2)} %.` : ""}
          </p>
          <Availability model={model} product={product} currentsShown={!!model.currents && currentsOn} onProduct={onProduct} />
        </div>
      </div>
    </section>
  );
}
