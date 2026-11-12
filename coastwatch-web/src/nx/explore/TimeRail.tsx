"use client";

import { useId, useState, type KeyboardEvent, type PointerEvent } from "react";
import { range, sig, useElementWidth } from "@/nx/ds";
import type { CurrentFrame, ExploreFrame, ExploreModel, ExploreProduct, ProductId, Track } from "./model";
import { axisLabels, axisX, nearestFrame, nearestHour, nowX, productSpan } from "./timeline";
import { LAYER_LABEL, layerKindWord, oldMark } from "./words";
import s from "./explore.module.css";

/**
 * The time rail (README §9 TimeRail; the availability ribbon grafted from the Tide concept): the active layer's
 * time tracks on a day axis, with the weight. Bars are each day's published coverage (how much of the coastal
 * ocean had a value); solid bars are measured, hatched boxes are model or outlook frames, a past model run is
 * drawn lighter with the words "past run"; the now line crosses every row. Choosing a frame is a cut (T26).
 * With currents on, the hours strip shows every published radar hour. Every other layer is one quiet line of
 * what it holds and when, each a way to switch to it, so the rail is short and its labels never collide.
 */
const LABEL_W = 88;
const ROW = 36;
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
};

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
  // with the sources folded away, the row is named for what it shows, not the satellite
  const name = plain ? LAYER_LABEL[product.id] : track.id === "viirs" ? "VIIRS" : track.id === "olci" ? "OLCI" : product.id === "chl" ? "Chlorophyll" : product.id === "sst" ? "Temperature" : product.id === "charm" ? "C-HARM" : "Outlook";
  const pastRun = product.pastRun && (track.id === "charm" || track.id === "outlook");
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
  // hour labels at 12 and 6 o'clock, thinned so each keeps its own room (about 7.5 px a character at 14 px)
  const tickText = (h: CurrentFrame) => h.label.replace(":00", "").replace(" PT", "");
  const ticks: CurrentFrame[] = [];
  let right = -Infinity;
  hours.forEach((h, i) => {
    if (i !== 0 && !/^(12|6):00 (AM|PM)/.test(h.label)) return;
    const x = i * step + 1;
    if (x < right + 10) return;
    const w = tickText(h).length * 7.5;
    if (x + w > HT) return;
    ticks.push(h);
    right = x + w;
  });
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
                {tickText(h)}
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

export function TimeRail({ model, product, frameId, onFrame, onProduct, currentsOn, currentId, onCurrent, playing, canPlay, onPlay }: Props) {
  const [ref, width] = useElementWidth<HTMLDivElement>(320);
  const uid = useId().replace(/:/g, "");
  const hatchId = `nxh-${uid}`;
  const hatchPastId = `nxhp-${uid}`;
  const T = Math.max(40, width - LABEL_W);
  const nx = nowX(model.axis, model.now) * T;
  const labels = axisLabels(model.axis, T, nx);
  const frame = product.frames.find((f) => f.id === frameId);
  const cur = model.currents?.frames.find((f) => f.id === currentId) ?? null;
  const [more, setMore] = useState(false);
  const moreId = `nx-rail-more-${uid}`;
  // a frame chosen on another sensor's row (from a link) keeps that row in view
  const all = more || (!!frame && frame.track !== product.tracks[0]?.id);
  const tracks = all ? product.tracks : product.tracks.slice(0, 1);
  return (
    <section className={s.rail} ref={ref} aria-label="Time" data-testid="nx-timerail">
      <div className={s.railHead}>
        <h2 className={s.railTitle}>Time</h2>
        <p className={s.railNow} data-testid="nx-frame-stamp">
          {frame ? (frame.label === "Latest" ? `Latest view: ${frame.stamp}` : frame.stamp.charAt(0).toUpperCase() + frame.stamp.slice(1)) : ""}
          {currentsOn && cur ? (
            <span className={s.railNowSub}>
              Currents: {cur.mean ? cur.label : cur.stamp}
              {oldMark(cur.fresh)}
            </span>
          ) : null}
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
          {labels.map((l) => (
            <span key={l.date} className={s.railDay} style={{ left: l.x }}>
              {l.text}
            </span>
          ))}
          <span className={s.railNowLabel} style={{ left: Math.min(Math.max(nx, 20), T - 20) }}>
            Now
          </span>
        </div>
        <div className={s.railRows}>
          <span className={s.nowLine} style={{ left: LABEL_W + nx }} aria-hidden="true" />
          {tracks.map((t, i) => (
            <TrackRow key={t.id} model={model} product={product} track={t} active height={i === 0 ? ROW_ACTIVE : ROW} frameId={frameId} onFrame={onFrame} width={width} hatchId={hatchId} hatchPastId={hatchPastId} plain={!all} />
          ))}
          {model.currents && currentsOn ? <HoursStrip model={model} currentId={currentId} onCurrent={onCurrent} width={width} on /> : null}
        </div>
      </div>
      <div className={s.railMore}>
        <button type="button" className={s.railMoreButton} aria-expanded={more} aria-controls={moreId} onClick={() => setMore((v) => !v)} data-testid="nx-rail-more">
          {more ? "Hide sources and coverage" : "Sources and coverage"}
        </button>
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
