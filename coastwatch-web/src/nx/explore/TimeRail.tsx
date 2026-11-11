"use client";

import { useId, type KeyboardEvent, type PointerEvent } from "react";
import { day, sig, useElementWidth } from "@/nx/ds";
import type { CurrentFrame, ExploreFrame, ExploreModel, ExploreProduct, Track } from "./model";
import { axisDays, axisLength, axisX, nearestFrame, nearestHour, nowX } from "./timeline";
import s from "./explore.module.css";

/**
 * The time rail (README §9 TimeRail; the availability ribbon grafted from the Tide concept): one row per
 * product's time track on a shared day axis. Bars are each day's published coverage (how much of the coastal
 * ocean had a value); solid bars are measured, hatched boxes are model or outlook frames, a past model run is
 * drawn lighter with the words "past run"; the now line crosses every row. Choosing a frame is a cut (T26).
 * With currents on, the hours strip shows every published radar hour.
 */
const LABEL_W = 96;
const ROW = 30;
const ROW_ACTIVE = 48;

type Props = {
  model: ExploreModel;
  product: ExploreProduct;
  frameId: string;
  onFrame: (product: ExploreProduct, frame: ExploreFrame) => void;
  currentsOn: boolean;
  currentId: string | null;
  onCurrent: (f: CurrentFrame) => void;
  playing: boolean;
  canPlay: boolean;
  onPlay: () => void;
};

function Hatch({ id, dense }: { id: string; dense?: boolean }) {
  return (
    <pattern id={id} patternUnits="userSpaceOnUse" width={dense ? 4 : 5} height={dense ? 4 : 5} patternTransform="rotate(45)">
      <line x1="0" y1="0" x2="0" y2={dense ? 4 : 5} className={s.hatchLine} />
    </pattern>
  );
}

function frameWords(f: ExploreFrame): string {
  const kind = f.kind === "measured" ? "measured" : f.kind === "analysis" ? "analysis" : f.kind === "outlook" ? "outlook" : f.past ? "model, past run" : "model";
  const cov = f.volume != null && f.kind === "measured" ? `, values over ${Math.round(f.volume * 100)} % of the coastal ocean` : "";
  return `${f.label === "Latest" ? `Latest view, ${f.stamp}` : f.stamp}, ${kind}${cov}`;
}

function TrackRow({ model, product, track, active, frameId, onFrame, width, hatchId, hatchPastId }: { model: ExploreModel; product: ExploreProduct; track: Track; active: boolean; frameId: string; onFrame: Props["onFrame"]; width: number; hatchId: string; hatchPastId: string }) {
  const frames = product.frames.filter((f) => f.track === track.id);
  const T = Math.max(40, width - LABEL_W);
  const H = active ? ROW_ACTIVE : ROW;
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
  const name = track.id === "viirs" ? "VIIRS" : product.id === "chl" ? "Chlorophyll" : product.id === "sst" ? "Temperature" : product.id === "charm" ? "C-HARM" : "Outlook";
  const pastRun = product.pastRun && track.id === "charm";
  return (
    <div className={s.railRow} data-active={active ? "1" : "0"} style={{ height: H }} data-testid={`nx-rail-${track.id}`}>
      <span className={s.railLabel}>
        {name}
        {pastRun ? <small className={s.railNote}>past run</small> : null}
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
        </svg>
        {frames.map((f) => {
          const a = x(f.start), b = x(f.end, 1);
          return (
            <button
              key={f.id}
              type="button"
              role="radio"
              aria-checked={f.id === frameId && active}
              tabIndex={f.id === frameId && active ? 0 : frames[0].id === f.id && !active ? 0 : -1}
              className={s.railHit}
              style={{ left: a, width: Math.max(8, b - a) }}
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

function HoursStrip({ model, currentId, onCurrent, width, on }: { model: ExploreModel; currentId: string | null; onCurrent: (f: CurrentFrame) => void; width: number; on: boolean }) {
  const hours = (model.currents?.frames ?? []).filter((f) => !f.mean);
  const mean = model.currents?.frames.find((f) => f.mean) ?? null;
  const T = Math.max(40, width - LABEL_W);
  const meanW = mean ? 64 : 0;
  const HT = T - meanW;
  const step = hours.length > 1 ? HT / hours.length : HT;
  const maxVol = Math.max(0.0001, ...hours.map((h) => h.volume ?? 0));
  const pick = (e: PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - r.left;
    if (mean && px > HT) return onCurrent(mean);
    const f = nearestHour(hours, Math.max(0, Math.min(1, (px - step / 2) / Math.max(1, HT - step))));
    if (f) onCurrent(f);
  };
  const keys = (e: KeyboardEvent<HTMLDivElement>) => {
    const all = mean ? [...hours, mean] : hours;
    const i = all.findIndex((f) => f.id === currentId);
    const to = e.key === "ArrowRight" ? i + 1 : e.key === "ArrowLeft" ? i - 1 : e.key === "Home" ? 0 : e.key === "End" ? all.length - 1 : null;
    if (to === null) return;
    e.preventDefault();
    onCurrent(all[Math.max(0, Math.min(all.length - 1, to))]);
  };
  const ticks = hours.filter((h, i) => i === 0 || /^(12|6):00 (AM|PM)/.test(h.label));
  return (
    <div className={s.railRow} data-active={on ? "1" : "0"} data-off={on ? undefined : "1"} style={{ height: ROW_ACTIVE + 18 }} data-testid="nx-rail-hours">
      <span className={s.railLabel}>
        Currents
        <small className={s.railNote}>by the hour</small>
      </span>
      <div className={s.railTrack} onPointerDown={pick} onKeyDown={keys} role="radiogroup" aria-label="Current hours">
        <svg width={T} height={ROW_ACTIVE + 18} className={s.railSvg} aria-hidden="true">
          <line x1={0} x2={HT} y1={ROW_ACTIVE - 0.5} y2={ROW_ACTIVE - 0.5} className={s.railBase} />
          {hours.map((h, i) => {
            const v = h.volume ?? 0;
            const ht = Math.max(v > 0 ? 2 : 0, (v / maxVol) * (ROW_ACTIVE - 12));
            const sel = on && h.id === currentId;
            return (
              <g key={h.id}>
                {ht ? <rect x={i * step + 1} y={ROW_ACTIVE - ht} width={Math.max(1, step - 2)} height={ht} className={s.barMeasured} /> : null}
                {sel ? <rect x={i * step + 0.75} y={1} width={Math.max(3, step - 1.5)} height={ROW_ACTIVE - 2} className={s.frameSel} fill="none" /> : null}
              </g>
            );
          })}
          {ticks.map((h) => {
            const i = hours.indexOf(h);
            return (
              <text key={h.id} x={i * step + 1} y={ROW_ACTIVE + 15} className={s.railTick}>
                {h.label.replace(":00", "").replace(" PT", "")}
              </text>
            );
          })}
          {mean ? (
            <g>
              <rect x={HT + 6} y={8} width={meanW - 8} height={ROW_ACTIVE - 16} className={on && currentId === mean.id ? s.frameSel : s.meanBox} fill="none" />
              <text x={HT + 6 + (meanW - 8) / 2} y={ROW_ACTIVE / 2 + 5} textAnchor="middle" className={s.railTick}>
                24 h
              </text>
            </g>
          ) : null}
        </svg>
        {[...hours, ...(mean ? [mean] : [])].map((h, i) => (
          <button
            key={h.id}
            type="button"
            role="radio"
            aria-checked={h.id === currentId}
            tabIndex={h.id === currentId ? 0 : -1}
            className={s.railHit}
            style={h.mean ? { left: HT + 4, width: meanW - 4 } : { left: i * step, width: Math.max(6, step) }}
            aria-label={`${h.mean ? h.stamp : `${h.stamp}, measured`}${h.volume != null ? `, ${Math.round(h.volume * 100)} % of the radar's area` : ""}`}
            data-testid={`nx-hour-${h.id}`}
            onClick={() => onCurrent(h)}
          />
        ))}
      </div>
    </div>
  );
}

export function TimeRail({ model, product, frameId, onFrame, currentsOn, currentId, onCurrent, playing, canPlay, onPlay }: Props) {
  const [ref, width] = useElementWidth<HTMLDivElement>(320);
  const uid = useId().replace(/:/g, "");
  const hatchId = `nxh-${uid}`;
  const hatchPastId = `nxhp-${uid}`;
  const T = Math.max(40, width - LABEL_W);
  const days = axisDays(model.axis);
  const n = axisLength(model.axis);
  const every = Math.max(1, Math.ceil(n / Math.max(2, Math.floor(T / 38))));
  const nx = nowX(model.axis, model.now) * T;
  const rows = model.products.flatMap((p) => p.tracks.map((t) => ({ p, t })));
  const frame = product.frames.find((f) => f.id === frameId);
  const cur = model.currents?.frames.find((f) => f.id === currentId) ?? null;
  return (
    <section className={s.rail} ref={ref} aria-label="Time" data-testid="nx-timerail">
      <div className={s.railHead}>
        <h2 className={s.railTitle}>Time</h2>
        <p className={s.railNow} data-testid="nx-frame-stamp">
          {frame ? (frame.label === "Latest" ? `Latest view: ${frame.stamp}` : frame.stamp.charAt(0).toUpperCase() + frame.stamp.slice(1)) : ""}
          {currentsOn && cur ? <span className={s.railNowSub}>Currents: {cur.mean ? cur.label : cur.stamp}</span> : null}
        </p>
        {canPlay ? (
          <button type="button" className={s.textButton} onClick={onPlay} aria-pressed={playing} data-testid="nx-play">
            {playing ? "Pause" : "Play"}
          </button>
        ) : null}
      </div>
      <div className={s.railBody}>
        <svg width={width} height={0} className={s.defs} aria-hidden="true">
          <defs>
            <Hatch id={hatchId} />
            <Hatch id={hatchPastId} dense />
          </defs>
        </svg>
        <div className={s.railAxis} style={{ marginLeft: LABEL_W, width: T }} aria-hidden="true">
          {days.map((d, i) =>
            i % every === 0 && Math.abs(axisX(model.axis, d, 0) * T + 10 - nx) > 34 ? (
              <span key={d} className={s.railDay} style={{ left: axisX(model.axis, d, 0) * T }}>
                {i === 0 || d.endsWith("-01") ? day(d) : String(Number(d.slice(8, 10)))}
              </span>
            ) : null,
          )}
          <span className={s.railNowLabel} style={{ left: nx }}>
            Now
          </span>
        </div>
        <div className={s.railRows}>
          <span className={s.nowLine} style={{ left: LABEL_W + nx }} aria-hidden="true" />
          {rows.map(({ p, t }) => (
            <TrackRow key={t.id} model={model} product={p} track={t} active={p.id === product.id} frameId={frameId} onFrame={onFrame} width={width} hatchId={hatchId} hatchPastId={hatchPastId} />
          ))}
          {model.currents ? <HoursStrip model={model} currentId={currentId} onCurrent={onCurrent} width={width} on={currentsOn} /> : null}
        </div>
      </div>
      <p className={s.railKey}>
        Bars: share of the coastal ocean with a value that day. Hatched: model or outlook. {product.pastRun ? "Past run: every day it covers is before today." : ""}
        {frame && frame.volume != null && frame.kind === "measured" ? ` This frame: values over ${sig(frame.volume * 100, 2)} %.` : ""}
      </p>
    </section>
  );
}
