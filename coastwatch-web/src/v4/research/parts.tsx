"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import type { Sample } from "@/lib/observations";
import { distanceKm } from "@/lib/grid";
import { Icon } from "@/v4/ui/Icon";
import { readingAt, type LinkedModel } from "@/v4/charts/LinkedSeries";
import { useWidth } from "@/v4/charts/useWidth";
import type { TransectCell } from "./transect";

const pct = (v: number) => `${Math.round(v * 100)}%`;

/** A quiet text action (an export, a redraw): one visual language for every secondary button. */
export const QUIET = "nx-press nx-focus inline-flex min-h-11 items-center gap-2 rounded-[4px] border border-n-line-strong px-3 text-[14px] font-medium text-n-ink";

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
    <div className={scroll ? "-mx-4 min-w-0 overflow-x-auto px-4 [scrollbar-width:none] sm:mx-0 sm:px-0 max-sm:[mask-image:linear-gradient(to_right,#000_calc(100%-40px),transparent)] max-sm:pr-10" : "min-w-0"}>
      <div role="radiogroup" aria-label={label} className={`inline-flex rounded-[12px] bg-n-surface-2 p-[3px] ${scroll ? "" : "flex-wrap"}`} data-testid={testId}>
        {options.map((o) => (
          <button
            key={String(o.id)}
            type="button"
            role="radio"
            aria-checked={o.id === value}
            title={o.title}
            onClick={() => onChange(o.id)}
            className={`nx-focus min-h-11 shrink-0 whitespace-nowrap rounded-[9px] px-3 text-[13.5px] font-medium ${o.id === value ? "bg-n-surface-3 text-n-ink" : "text-n-ink-2 hover:text-n-ink"}`}
            data-testid={o.testId}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export function TabPanel({ id, idBase, active, label, children, className = "" }: { id: string; idBase: string; active: boolean; label: string; children: ReactNode; className?: string }) {
  return (
    <div role="tabpanel" id={`${idBase}-panel-${id}`} aria-label={label} hidden={!active} className={`pt-8 ${className}`} data-testid={`research-panel-${id}`}>
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
      <button type="button" aria-expanded={open} aria-controls="r-export-menu" onClick={() => setOpen((o) => !o)} className="nx-press nx-focus inline-flex min-h-11 items-center gap-1.5 rounded-[4px] px-2.5 text-[13.5px] font-medium text-n-ink-2 hover:text-n-ink aria-expanded:text-n-ink" data-testid="research-export-menu">
        <Icon name="download" className="h-4 w-4" /> <span className="max-sm:sr-only">Export</span> <Icon name="chevronDown" className={`h-4 w-4 text-n-ink-3 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      <div id="r-export-menu" hidden={!open} className="nx-deck absolute top-[calc(100%+6px)] z-30 flex w-[min(320px,calc(100vw-32px))] flex-col p-1.5 max-sm:left-0 sm:right-0">
        {children(() => setOpen(false))}
      </div>
    </div>
  );
}

/** A row of the Export menu. */
export const MENU_ITEM = "nx-focus flex min-h-12 w-full flex-col items-start justify-center rounded-lg px-3 py-1.5 text-left hover:bg-n-surface-2";

/** The window's samples as a table (newest first): the same values, QC words and the model's same day. */
export function SamplesTable({ samples, unit, model }: { samples: Sample[]; unit: string; model: LinkedModel | null }) {
  const rows = [...samples].reverse();
  return (
    <div className="nx-plate max-h-[340px] overflow-auto">
      <table className="w-full text-left text-[13px]">
        <thead className="sticky top-0 bg-n-surface">
          <tr className="border-b border-n-line">
            <th className="nx-caps px-3 py-2 font-medium">Sample day (UTC)</th>
            <th className="nx-caps px-3 py-2 text-right font-medium">{unit}</th>
            <th className="nx-caps px-3 py-2 font-medium">QC</th>
            {model && <th className="nx-caps px-3 py-2 text-right font-medium text-n-model">C-HARM same day</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((s) => {
            const r = readingAt(s.t, samples, model);
            return (
              <tr key={s.index} className="border-t border-n-line">
                <td className="px-3 py-1.5 font-data text-[12px]">{s.time.slice(0, 10)}</td>
                <td className="px-3 py-1.5 text-right font-data text-[12px]">{s.q === "rejected_negative" ? "—" : s.value == null ? "—" : s.q === "reported_zero" ? "0 (reported)" : s.value}</td>
                <td className="px-3 py-1.5 text-n-ink-2">{s.q === "rejected_negative" ? "rejected (QC)" : s.q === "flag_high" ? "flagged high" : s.q === "reported_zero" ? "reported zero" : s.value == null ? "not measured" : ""}</td>
                {model && <td className="px-3 py-1.5 text-right font-data text-[12px] text-n-ink-2">{r.model ? pct(r.model.v) : "—"}</td>}
              </tr>
            );
          })}
        </tbody>
      </table>
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
