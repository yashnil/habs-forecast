"use client";

import { useEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import type { ObsStation } from "@/generated/observations";
import type { Sample } from "@/lib/observations";
import { modelOnAxis, modelSegments, type ModelPoint } from "@/lib/bloom";
import type { CharmVariable } from "@/lib/layers";
import { distanceKm } from "@/lib/grid";
import { Icon } from "@/v4/ui/Icon";
import { dayLabel, kmLabel } from "@/v4/ui/time";
import { SampleValue, monthTicks, tickLabel } from "@/v4/charts/SeriesChart";
import { useWidth } from "@/v4/charts/useWidth";
import { preciseProbability } from "@/v4/explore/Inspect";
import { VARIABLE_WORDS } from "@/v4/explore/layers";
import type { TransectCell } from "./transect";

const dayKey = (t: number) => new Date(t).toISOString().slice(0, 10);
const pct = (v: number) => `${Math.round(v * 100)}%`;

/** A quiet text action (an export, a redraw): one visual language for every secondary button. */
export const QUIET = "nx-press nx-focus inline-flex min-h-11 items-center gap-2 rounded-lg border border-n-line-strong px-3 text-[14px] font-semibold text-n-ink";

export function ExportButton({ onClick, testid, children }: { onClick: () => void; testid: string; children: ReactNode }) {
  return (
    <button type="button" onClick={onClick} className={`${QUIET} self-start`} data-testid={testid}>
      <Icon name="download" className="h-4 w-4" /> {children}
    </button>
  );
}

/**
 * One segmented control for every few-way choice on the page (quantity, range, tool, source, day):
 * the v7 Segmented look, with a test id per option. It scrolls sideways on a narrow screen rather
 * than wrapping into rows of chips.
 */
export function Seg<T extends string | number>({ label, value, options, onChange, testId, scroll = false }: { label: string; value: T; options: { id: T; label: ReactNode; testId?: string; title?: string }[]; onChange: (v: T) => void; testId?: string; scroll?: boolean }) {
  return (
    <div className={scroll ? "-mx-4 min-w-0 overflow-x-auto px-4 [scrollbar-width:none] sm:mx-0 sm:px-0" : "min-w-0"}>
      <div role="radiogroup" aria-label={label} className={`inline-flex rounded-xl bg-n-surface-2 p-1 ${scroll ? "" : "flex-wrap"}`} data-testid={testId}>
        {options.map((o) => (
          <button
            key={String(o.id)}
            type="button"
            role="radio"
            aria-checked={o.id === value}
            title={o.title}
            onClick={() => onChange(o.id)}
            className={`nx-focus min-h-10 shrink-0 whitespace-nowrap rounded-lg px-3 text-[14px] font-semibold ${o.id === value ? "bg-n-surface text-n-ink shadow-[0_1px_2px_rgba(11,24,32,0.12)]" : "text-n-ink-2 hover:text-n-ink"}`}
            data-testid={o.testId}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

/** The section tabs: an underline tablist; every panel stays mounted (hidden), so its wording is in the page. */
export function Tabs<T extends string>({ tabs, value, onChange, idBase }: { tabs: { id: T; label: string }[]; value: T; onChange: (v: T) => void; idBase: string }) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const onKey = (e: KeyboardEvent, i: number) => {
    const n = tabs.length;
    const j = e.key === "ArrowRight" ? (i + 1) % n : e.key === "ArrowLeft" ? (i - 1 + n) % n : e.key === "Home" ? 0 : e.key === "End" ? n - 1 : -1;
    if (j < 0) return;
    e.preventDefault();
    onChange(tabs[j].id);
    refs.current[j]?.focus();
  };
  return (
    <div className="-mx-4 overflow-x-auto px-4 [scrollbar-width:none] sm:mx-0 sm:px-0">
      <div role="tablist" aria-label="Station record sections" className="flex min-w-max gap-5 border-b border-n-line sm:gap-7">
        {tabs.map((t, i) => {
          const on = t.id === value;
          return (
            <button
              key={t.id}
              ref={(el) => {
                refs.current[i] = el;
              }}
              type="button"
              role="tab"
              id={`${idBase}-tab-${t.id}`}
              aria-selected={on}
              aria-controls={`${idBase}-panel-${t.id}`}
              tabIndex={on ? 0 : -1}
              onClick={() => onChange(t.id)}
              onKeyDown={(e) => onKey(e, i)}
              className={`nx-focus -mb-px min-h-12 whitespace-nowrap border-b-2 text-[14.5px] font-semibold ${on ? "border-n-accent text-n-ink" : "border-transparent text-n-ink-2 hover:text-n-ink"}`}
              data-testid={`research-tab-${t.id}`}
            >
              {t.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function TabPanel({ id, idBase, active, children, className = "" }: { id: string; idBase: string; active: boolean; children: ReactNode; className?: string }) {
  return (
    <div role="tabpanel" id={`${idBase}-panel-${id}`} aria-labelledby={`${idBase}-tab-${id}`} hidden={!active} className={`pt-6 ${className}`} data-testid={`research-panel-${id}`}>
      {children}
    </div>
  );
}

/** The header's Export menu: every station-level export, each with its provenance header. */
export function ExportMenu({ children }: { children: (close: () => void) => ReactNode }) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!open) return;
    const down = (e: PointerEvent) => !box.current?.contains(e.target as Node) && setOpen(false);
    const key = (e: globalThis.KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("pointerdown", down);
    document.addEventListener("keydown", key);
    return () => {
      document.removeEventListener("pointerdown", down);
      document.removeEventListener("keydown", key);
    };
  }, [open]);
  return (
    <div ref={box} className="relative">
      <button type="button" aria-expanded={open} aria-controls="r-export-menu" onClick={() => setOpen((o) => !o)} className={`${QUIET} max-sm:border-transparent max-sm:px-2.5`} data-testid="research-export-menu">
        <Icon name="download" className="h-4 w-4" /> Export <Icon name="chevronDown" className={`h-4 w-4 text-n-ink-3 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      <div id="r-export-menu" hidden={!open} className="nx-deck absolute top-[calc(100%+6px)] z-30 flex w-[min(320px,calc(100vw-32px))] flex-col p-1.5 max-sm:left-0 sm:right-0">
        {children(() => setOpen(false))}
      </div>
    </div>
  );
}

/** A row of the Export menu. */
export const MENU_ITEM = "nx-focus flex min-h-12 w-full flex-col items-start justify-center rounded-lg px-3 py-1.5 text-left hover:bg-n-surface-2";

export function ModelPanel({ st, variable, from, to, cursorT, onCursor }: { st: ObsStation; variable: CharmVariable; from: number; to: number; cursorT: number | null; onCursor: (t: number) => void }) {
  const hist = st.charm?.history?.[variable];
  const { points, historyStart } = modelOnAxis(hist, from, to);
  const segs = modelSegments(points);
  const [box, W] = useWidth<HTMLElement>(720);
  const svg = useRef<SVGSVGElement | null>(null);
  const H = 116;
  const pad = { l: 50, r: 10, t: 8, b: 22 };
  const x = (t: number) => pad.l + ((t - from) / Math.max(1, to - from)) * (W - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - v) * (H - pad.t - pad.b);
  const months = monthTicks(from, to, W);
  if (!st.charm?.history?.[variable]?.length)
    return (
      <div className="border-t border-n-line pt-3 text-[14px] text-n-ink-2" data-testid="model-panel-none">
        <p>No C-HARM nowcast history is published for this station.</p>
        {st.charm?.error && (
          <details className="mt-1 text-[12.5px] text-n-ink-3">
            <summary className="nx-focus min-h-11 cursor-pointer py-2 font-semibold text-n-accent">What the pipeline reported</summary>
            <p className="break-all">{st.charm.error}</p>
          </details>
        )}
      </div>
    );
  const near = (clientX: number) => {
    const r = svg.current?.getBoundingClientRect();
    if (!r || !points.length) return;
    const t = from + (((clientX - r.left) / r.width) * W - pad.l) / (W - pad.l - pad.r) * (to - from);
    let best = points[0];
    for (const p of points) if (Math.abs(p.t - t) < Math.abs(best.t - t)) best = p;
    onCursor(best.t);
  };
  const curX = cursorT != null && cursorT >= from && cursorT <= to ? x(cursorT) : null;
  const onDay = cursorT != null ? points.find((p) => p.date === dayKey(cursorT)) : undefined;
  const hatched = historyStart != null && historyStart > from;
  return (
    <figure ref={box} className="flex flex-col gap-1 border-t border-n-line pt-3" data-testid="model-panel">
      <p className="flex items-center gap-2 text-[12.5px] font-semibold text-n-ink-2">
        <span aria-hidden className="inline-block h-2 w-2 rounded-full bg-n-model" />
        NOAA model probability near the station · C-HARM nowcast, {VARIABLE_WORDS[variable].short}
      </p>
      <svg
        ref={svg}
        width={W}
        height={H}
        className="block touch-pan-y select-none"
        role="img"
        aria-label={`C-HARM nowcast probability near ${st.name}, ${points.length} days in view`}
        onPointerMove={(e) => near(e.clientX)}
        onPointerDown={(e) => near(e.clientX)}
        data-testid="model-svg"
      >
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line x1={pad.l} x2={W - pad.r} y1={y(v)} y2={y(v)} stroke="var(--nx-line)" />
            <text x={pad.l - 6} y={y(v) + 4} textAnchor="end" fontSize="11" fill="var(--nx-ink-3)">
              {Math.round(v * 100)}%
            </text>
          </g>
        ))}
        {months.filter((t) => x(t) <= W - pad.r - 18).map((t) => (
          <text key={t} x={x(t)} y={H - 5} fontSize="11.5" fill="var(--nx-ink-3)" textAnchor="middle">
            {tickLabel(t, from, to)}
          </text>
        ))}
        {hatched && (
          <g>
            <rect x={pad.l} y={pad.t} width={Math.max(0, x(historyStart) - pad.l)} height={H - pad.t - pad.b} fill="url(#nx-hatch)" />
            {x(historyStart) - pad.l > 230 && (
              <text x={pad.l + 6} y={pad.t + 14} fontSize="11" fill="var(--nx-ink-3)">
                no model history kept before {dayLabel(new Date(historyStart).toISOString().slice(0, 10))}
              </text>
            )}
          </g>
        )}
        <defs>
          <pattern id="nx-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="6" stroke="var(--nx-line)" strokeWidth="1.5" />
          </pattern>
        </defs>
        {segs.map((sg, i) => (
          <polyline key={i} points={sg.map((p) => `${x(p.t)},${y(p.v)}`).join(" ")} fill="none" stroke="var(--nx-model)" strokeWidth={1.6} />
        ))}
        {points.map((p) => (
          <circle key={p.t} cx={x(p.t)} cy={y(p.v)} r={p.partial ? 2.8 : 1.6} fill={p.partial ? "var(--nx-surface)" : "var(--nx-model)"} stroke="var(--nx-model)" strokeWidth={p.partial ? 1.2 : 0} />
        ))}
        {onDay && <circle cx={x(onDay.t)} cy={y(onDay.v)} r={4.5} fill="var(--nx-surface)" stroke="var(--nx-model)" strokeWidth={2} />}
        {curX != null && <line x1={curX} x2={curX} y1={pad.t} y2={H - pad.b} stroke="var(--nx-ink-3)" strokeDasharray="2 3" data-testid="model-cursor" />}
      </svg>
      <figcaption className="nx-note-t">
        Median of the valued cells within {st.charm!.radius_km} km of the station (nearest cell {st.charm!.nearest_cell_km != null ? kmLabel(st.charm!.nearest_cell_km) : "—"}). Shown beside the measurements on the same dates, never compared with them as numbers. Open circles: days when fewer than 60% of the usual cells had a value (not joined); days without a value are left out.
        {hatched ? ` Hatched: before ${dayLabel(new Date(historyStart).toISOString().slice(0, 10))} ${new Date(historyStart).getUTCFullYear()}, no model history is kept.` : ""}
      </figcaption>
    </figure>
  );
}

/**
 * The shared cursor's readout: what was measured at the pier that day and what C-HARM published for
 * the same day, as two separate soundings, each with its kind. Two different quantities; nothing is
 * computed between them.
 */
export function CursorReadout({ t, sample, unit, fmt, quantity, model, variable, radiusKm, historyStart }: { t: number | null; sample: Sample | null; unit: string; fmt: (v: number) => string; quantity: string; model: ModelPoint | null | undefined; variable: CharmVariable | null; radiusKm: number; historyStart: number | null }) {
  if (t == null) return <p className="min-h-[64px] text-[14px] text-n-ink-2">Point at the chart to read a day.</p>;
  const day = dayKey(t);
  return (
    <div className="grid min-h-[64px] grid-cols-2 gap-x-4 sm:gap-x-8" aria-live="polite" data-testid="cursor-readout">
      <div className="flex min-w-0 gap-3">
        <span aria-hidden className="nx-tick" data-family="measured" />
        <div className="min-w-0" data-testid="cursor-measured">
          <p className="text-[12.5px] font-semibold leading-4 text-n-ink-2 sm:text-[13px]">
            Measured<span className="max-sm:hidden"> at the pier,</span> <span className="whitespace-nowrap">{dayLabel(day, true)}</span>
          </p>
          <p className="mt-0.5 leading-[30px]">{sample ? <SampleValue s={sample} unit={unit} fmt={fmt} size="lg" /> : <span className="text-[15px] text-n-ink-2">no sample that day</span>}</p>
          <p className="nx-note-t">
            <span className="max-sm:hidden">{quantity} · </span>measured
          </p>
        </div>
      </div>
      {variable && (
        <div className="flex min-w-0 gap-3">
          <span aria-hidden className="nx-tick" data-family="model" />
          <div className="min-w-0" data-testid="cursor-model">
            <p className="text-[12.5px] font-semibold leading-4 text-n-ink-2 sm:text-[13px]">
              <span className="max-sm:hidden">NOAA </span>C-HARM nowcast, same day
            </p>
            {model ? (
              <>
                <p className="mt-0.5 leading-[30px]">
                  <span className="nx-sounding-v">{pct(model.v)}</span>
                </p>
                <p className="nx-note-t">
                  model probability <span className="max-sm:hidden">({preciseProbability(model.v, null)}) of {VARIABLE_WORDS[variable].short}, median of {model.n} cells within {radiusKm} km{model.partial ? "; fewer cells than usual" : ""}</span>
                </p>
              </>
            ) : (
              <p className="mt-0.5 text-[15px] leading-[30px] text-n-ink-2">{historyStart != null && t < historyStart ? "before the model history Neritic keeps" : "no model value kept for that day"}</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export function TransectChart({ cells, a, b }: { cells: TransectCell[]; a: { lat: number; lon: number }; b: { lat: number; lon: number } }) {
  const [box, W] = useWidth<HTMLElement>(640);
  // the whole line, so the last cell keeps its own width
  const end = Math.max(0.1, distanceKm(a.lat, a.lon, b.lat, b.lon));
  const H = 150;
  const pad = { l: 40, r: 8, t: 8, b: 26 };
  const x = (km: number) => pad.l + (km / end) * (W - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - v) * (H - pad.t - pad.b);
  const valued = cells.filter((c) => c.value != null).length;
  const ll = (p: { lat: number; lon: number }) => `${p.lat.toFixed(3)}°N ${Math.abs(p.lon).toFixed(3)}°W`;
  return (
    <figure ref={box} className="flex flex-col gap-1">
      <svg width={W} height={H} className="block" role="img" aria-label={`Transect of ${cells.length} cells; ${valued} with a published value`}>
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line x1={pad.l} x2={W - pad.r} y1={y(v)} y2={y(v)} stroke="var(--nx-line)" />
            <text x={pad.l - 5} y={y(v) + 4} textAnchor="end" fontSize="11" fill="var(--nx-ink-3)">
              {Math.round(v * 100)}%
            </text>
          </g>
        ))}
        {cells.map((c, i) => {
          const x0 = x(c.km);
          const x1 = x(cells[i + 1]?.km ?? end);
          return c.value == null ? (
            <rect key={i} x={x0} y={pad.t} width={Math.max(1, x1 - x0)} height={H - pad.t - pad.b} fill="url(#nx-hatch-t)" />
          ) : (
            <line key={i} x1={x0} x2={x1} y1={y(c.value)} y2={y(c.value)} stroke="var(--nx-model)" strokeWidth={2.2} />
          );
        })}
        <defs>
          <pattern id="nx-hatch-t" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="6" stroke="var(--nx-line-strong)" strokeWidth="1.5" />
          </pattern>
        </defs>
        <text x={pad.l} y={H - 6} fontSize="11" fill="var(--nx-ink-3)">
          0 km
        </text>
        <text x={W - pad.r} y={H - 6} fontSize="11" fill="var(--nx-ink-3)" textAnchor="end">
          {end.toFixed(1)} km
        </text>
      </svg>
      <figcaption className="text-[12.5px] text-n-ink-3">
        {cells.length} cells crossed, {valued} with a published value; each value is held flat across its own cell (no interpolation). Hatched: no value (land, nearshore or masked). From {ll(a)} to {ll(b)}.
      </figcaption>
    </figure>
  );
}
