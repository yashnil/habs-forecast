import type { Map as MLMap } from "maplibre-gl";
import { useCallback, useState } from "react";
import { dms, fmtKm } from "../lib/geo";
import { fmtDay } from "../lib/time";
import { DirMap } from "./DirMap";
import { facts, noticeLabel, StatusLine } from "./facts";
import { type DirState, POINT } from "./shared";

/**
 * Direction A — Marine Observatory. A precise instrument: dark, full-bleed sea; a thin inspector
 * rail with hairlines; values large and tabular; class lines instead of a filled field; a real
 * graticule at the map edge.
 */
interface Tick { pos: number; label: string }

export function Observatory({ st, mobile }: { st: DirState; mobile: boolean }) {
  const [ticks, setTicks] = useState<{ x: Tick[]; y: Tick[] }>({ x: [], y: [] });
  const onMove = useCallback((m: MLMap) => setTicks(graticule(m)), []);
  const f = facts(st);
  return (
    <div className="obs cw-dark" data-mobile={mobile || undefined}>
      <header className="obs-bar">
        <span className="obs-brand">CoastWatch</span>
        <span className="obs-bar-sep" aria-hidden="true" />
        <span className="obs-bar-k">Water point</span>
        <span className="t-coord obs-bar-coord">{dms(POINT[1], POINT[0])}</span>
        <span className="obs-bar-sp" />
        <span className="obs-bar-k">Forecast for</span><span className="obs-bar-v">{fmtDay(f.validFor, { weekday: true })}</span>
        <span className="obs-bar-k">Issued</span><span className="obs-bar-v">{fmtDay(f.issued)} · <span className="obs-warn">{f.freshness}</span></span>
      </header>
      <StatusLine className="obs-status" />
      <div className="obs-stage">
        <DirMap v="a" st={st} field={{ mode: "edges" }} center={[-122.0, 36.82]} zoom={mobile ? 8.4 : 9.1} onMove={onMove}
          padding={mobile ? { top: 20, bottom: 320, left: 10, right: 10 } : { top: 30, bottom: 30, left: 360, right: 60 }} />
        {!mobile && (
          <div className="obs-grat" aria-hidden="true">
            {ticks.x.map((t) => <span key={"x" + t.label} className="obs-tx" style={{ left: t.pos }}>{t.label}</span>)}
            {ticks.y.map((t) => <span key={"y" + t.label} className="obs-ty" style={{ top: t.pos }}>{t.label}</span>)}
          </div>
        )}
        <aside className="obs-rail" aria-label="Inspector">
          <h1 className="obs-place">{f.place}</h1>
          <section className="obs-sec">
            <p className="obs-k"><span className="mark mark-field" aria-hidden="true" />Modelled · C-HARM particulate domoic acid</p>
            {f.cls ? (
              <p className="obs-big"><span className="t-value">{f.cls.lo}–{f.cls.hi}</span><span className="obs-unit">% chance</span></p>
            ) : <p className="obs-big obs-none">No value in this cell</p>}
            <p className="obs-sub">{f.threshold}. Cell value <span className="num">{f.value?.toFixed(4)}</span>; one 3 km cell, exact edges outlined.</p>
          </section>
          <section className="obs-sec">
            <p className="obs-k"><span className="mark mark-sample" aria-hidden="true" />Measured · shore sample</p>
            <p className="obs-big"><span className="t-value">{f.pDA?.toFixed(2)}</span><span className="obs-unit">ng/mL pDA</span></p>
            <p className="obs-sub">{f.stationText} · sampled {f.sampled}. Describes the pier on that day, not this water.</p>
          </section>
          <section className="obs-sec">
            <p className="obs-k"><span className="mark mark-rule" aria-hidden="true" />Official · {f.nCover} covering</p>
            <ul className="obs-list">
              {f.covering.map((n) => <li key={n.record.id}><span className="obs-li-a">{noticeLabel(n)}</span> {n.record.title}</li>)}
            </ul>
            {f.limitText && <p className="obs-sub">Nearest limit {f.limitText}.</p>}
            {f.disputed && <p className="obs-sub obs-disp">{f.disputed.by.agency} {f.disputed.by.release} appears to lift “{f.disputed.record.title}” — not shown as in effect, awaiting review.</p>}
          </section>
          <section className="obs-sec">
            <p className="obs-k"><span className="mark mark-gap" aria-hidden="true" />Not known</p>
            <ul className="obs-gaps">{f.gaps.map((g) => <li key={g}>{g}</li>)}</ul>
          </section>
          <p className="obs-foot t-coord">nearest limit {f.limit ? fmtKm(f.limit.km) : "—"} · station {fmtKm(f.station.km)}</p>
        </aside>
        <div className="obs-legend" aria-hidden="true">
          <span>class lines</span>
          {Array.from({ length: 10 }, (_, i) => <i key={i} style={{ background: `var(--cw-p${i})` }} />)}
          <span>0–100 %</span>
        </div>
      </div>
    </div>
  );
}

/** Graticule labels at round intervals inside the current bounds (real coordinates). */
function graticule(m: MLMap) {
  const b = m.getBounds();
  const span = b.getEast() - b.getWest();
  const step = span > 3 ? 1 : span > 1.2 ? 0.5 : span > 0.5 ? 0.25 : 0.1;
  const x: Tick[] = [], y: Tick[] = [];
  for (let lon = Math.ceil(b.getWest() / step) * step; lon < b.getEast(); lon += step) {
    const p = m.project([lon, b.getSouth()]);
    x.push({ pos: p.x, label: `${Math.abs(lon).toFixed(step < 0.25 ? 1 : 2)}°W` });
  }
  for (let lat = Math.ceil(b.getSouth() / step) * step; lat < b.getNorth(); lat += step) {
    const p = m.project([b.getWest(), lat]);
    y.push({ pos: p.y, label: `${lat.toFixed(step < 0.25 ? 1 : 2)}°N` });
  }
  return { x, y };
}
