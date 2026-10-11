"use client";

import { useMemo } from "react";
import type { FisheriesDataset, SpeciesGroup } from "@/generated/fisheries";
import { FISHERIES_EVENTS } from "@/content/copy";
import { formatTick, formatValue, groupColor, groupShort, groupValue, niceTicks, pct, selectedTotal, shareOfState, stateTotal, yearFraction, type Unit } from "@/lib/fisheries";
import { useWidth } from "@/lib/useWidth";

/**
 * The primary figure (M7): one stacked bar per calendar year for the selected species groups, in one
 * unit at a time. Choosing a group shows it alone on its own scale: next to Dungeness crab most groups
 * would be slivers, so a stack cannot compare them. Totals sit on the bars, the share of NOAA's statewide total under the
 * years, and documented regulatory events are bracketed above, positioned by date within the year
 * slots. The bracket is context: nothing here attributes a change in landings to an event.
 *
 * The years are HTML buttons laid under the bars, so the chart is read and driven by keyboard and
 * screen reader as well as by pointer.
 */
export function LandingsChart({
  ds,
  groups,
  unit,
  focus,
  year,
  hover,
  onHover,
  onYear,
  label,
}: {
  ds: FisheriesDataset;
  groups: SpeciesGroup[];
  unit: Unit;
  focus: string | null;
  year: number;
  hover: number | null;
  onHover: (y: number | null) => void;
  onYear: (y: number) => void;
  label: string;
}) {
  const [ref, W] = useWidth<HTMLDivElement>(900, 300);
  const narrow = W < 560;
  const years = ds.years;
  const ML = narrow ? 50 : 60;
  const MR = 4;
  const shownIds = new Set((focus && groups.some((g) => g.id === focus) ? [focus] : groups.map((g) => g.id)) as string[]);
  const events = FISHERIES_EVENTS.filter((e) => e.groups.some((g) => shownIds.has(g)) && yearFraction(e.end) >= years[0] && yearFraction(e.start) <= years[years.length - 1] + 1);
  const ROW = 19;
  const TOP = 26 + events.length * ROW;
  const PH = narrow ? 190 : 250;
  const H = TOP + PH + 8;
  const band = (W - ML - MR) / years.length;
  const bw = Math.min(64, band * (narrow ? 0.7 : 0.6));
  const xL = (i: number) => ML + i * band;
  const xT = (fy: number) => ML + (fy - years[0]) * band; // a date within the year slots

  const order = useMemo(() => (focus && groups.some((g) => g.id === focus) ? groups.filter((g) => g.id === focus) : groups), [groups, focus]);
  const totals = useMemo(() => years.map((y) => selectedTotal(order, y, unit)), [years, order, unit]);
  const ticks = useMemo(() => niceTicks(Math.max(0, ...totals.map((t) => t ?? 0))), [totals]);
  const top = ticks[ticks.length - 1];
  const yS = (v: number) => TOP + PH - (v / top) * PH;
  const shares = years.map((y, i) => shareOfState(totals[i], stateTotal(ds, y, unit)));
  const active = hover ?? year;

  const fromPointer = (clientX: number, el: Element) => {
    const r = el.getBoundingClientRect();
    const i = Math.floor((clientX - r.left - ML) / band);
    return i >= 0 && i < years.length ? years[i] : null;
  };

  return (
    <div ref={ref} className="relative select-none" data-testid="landings-chart" data-unit={unit} data-focus={focus ?? ""} data-groups={groups.map((g) => g.id).join(",")}>
      {/* sized to the container from the first paint: before the width is measured the drawing
          scales, so nothing overflows a phone */}
      <svg
        width="100%"
        height={H}
        viewBox={`0 0 ${W} ${H}`}
        className="block max-w-full touch-pan-y"
        role="img"
        aria-label={`${label}. ${years.map((y, i) => `${y}: ${formatValue(totals[i], unit)}`).join("; ")}.`}
        onPointerMove={(e) => onHover(fromPointer(e.clientX, e.currentTarget))}
        onPointerDown={(e) => onHover(fromPointer(e.clientX, e.currentTarget))}
        onPointerLeave={(e) => e.pointerType === "mouse" && onHover(null)}
        onClick={(e) => {
          const y = fromPointer(e.clientX, e.currentTarget);
          if (y != null) onYear(y);
        }}
      >
        {/* hovered or selected year */}
        {years.map((y, i) =>
          y === active || y === year ? <rect key={`bg${y}`} x={xL(i) + 2} y={TOP - 6} width={band - 4} height={PH + 6} rx={4} fill={y === year ? "var(--cw-surface-3)" : "var(--cw-surface-2)"} /> : null,
        )}
        {ticks.map((t) => (
          <g key={t}>
            <line x1={ML} x2={W - MR} y1={yS(t)} y2={yS(t)} stroke={t === 0 ? "var(--cw-ink-3)" : "var(--cw-hairline)"} strokeWidth={t === 0 ? 1 : 1} />
            <text x={ML - 8} y={yS(t) + 4} textAnchor="end" className={`fill-[var(--cw-ink-3)] tabular ${narrow ? "text-[11px]" : "text-[12px]"}`}>
              {formatTick(t, unit)}
            </text>
          </g>
        ))}

        {/* stacked bars: focused group on the axis */}
        {years.map((y, i) => {
          let acc = 0;
          const cx = xL(i) + band / 2;
          return (
            <g key={y} data-year={y}>
              {order.map((g) => {
                const v = groupValue(g, y, unit);
                if (v == null || v <= 0) return null;
                const y0 = yS(acc);
                acc += v;
                const y1 = yS(acc);
                return (
                  <rect
                    key={g.id}
                    x={cx - bw / 2}
                    y={y1}
                    width={bw}
                    height={Math.max(0.5, y0 - y1)}
                    fill={groupColor(g.id)}
                    stroke="var(--cw-surface)"
                    strokeWidth={y0 - y1 > 2 ? 1 : 0}
                    data-group={g.id}
                  />
                );
              })}
              {totals[i] == null && (
                <text x={cx} y={TOP + PH - 6} textAnchor="middle" className="fill-[var(--cw-ink-3)] text-[11px]">
                  no value
                </text>
              )}
              {totals[i] != null && (!narrow || y === active) && (
                <text
                  x={Math.min(cx, W - MR - 26)}
                  // on a phone the label is wider than a bar: lift it clear of the neighbouring bars
                  y={yS(narrow ? Math.max(totals[i]!, totals[i - 1] ?? 0, totals[i + 1] ?? 0) : totals[i]!) - 6}
                  textAnchor="middle" className={`tabular ${y === active ? "fill-[var(--cw-ink)] font-semibold" : "fill-[var(--cw-ink-2)]"} text-[12px]`} data-testid={`bar-total-${y}`}>
                  {formatValue(totals[i], unit)}
                </text>
              )}
            </g>
          );
        })}

        {/* documented events: a dated bracket above the bars, context only */}
        {events.map((e, k) => {
          const a = Math.max(ML, xT(yearFraction(e.start)));
          const b = Math.min(W - MR, xT(yearFraction(e.end)));
          const ly = 13 + k * ROW;
          return (
            <g key={e.id} data-testid={`event-${e.id}`} data-x0={Math.round(a)} data-x1={Math.round(b)}>
              <path d={`M${a},${ly + 6} v-4 H${b}${e.continues ? "" : " v4"}`} fill="none" stroke="var(--cw-ink-2)" strokeWidth={1.25} />
              {e.continues && <path d={`M${b},${ly + 2} h${Math.min(18, band * 0.25)}`} fill="none" stroke="var(--cw-ink-2)" strokeWidth={1.25} strokeDasharray="2 2.5" />}
              <text x={b + (e.continues ? Math.min(18, band * 0.25) : 0) + 6} y={ly + 6} className="fill-[var(--cw-ink)] text-[12.5px] font-medium">
                {narrow ? e.short : `${e.period}: ${e.label.charAt(0).toUpperCase()}${e.label.slice(1)}`}
              </text>
            </g>
          );
        })}
      </svg>

      {/* years (and the share of the state total) as buttons under the bars */}
      <div className="relative" style={{ height: narrow ? 30 : 44 }}>
        {years.map((y, i) => {
          const sel = y === year;
          return (
            <button
              key={y}
              type="button"
              aria-pressed={sel}
              aria-label={`${y}: ${formatValue(totals[i], unit)}, ${pct(shares[i])} of all California commercial landings`}
              onClick={() => onYear(y)}
              onFocus={() => onHover(y)}
              onBlur={() => onHover(null)}
              onKeyDown={(e) => {
                const j = e.key === "ArrowRight" ? i + 1 : e.key === "ArrowLeft" ? i - 1 : e.key === "Home" ? 0 : e.key === "End" ? years.length - 1 : null;
                if (j == null || j < 0 || j >= years.length) return;
                e.preventDefault();
                onYear(years[j]);
                (e.currentTarget.parentElement?.children[j] as HTMLElement | undefined)?.focus();
              }}
              className={`absolute top-0 flex flex-col items-center rounded-md pb-0.5 pt-1 leading-tight ${sel ? "text-ink" : "text-ink-2 hover:text-ink"}`}
              style={{ left: `${(xL(i) / W) * 100}%`, width: `${(band / W) * 100}%` }}
              data-testid={`year-${y}`}
            >
              <span className={`tabular text-[13px] ${sel ? "font-semibold underline decoration-2 underline-offset-[5px]" : ""}`}>{narrow ? `’${String(y).slice(2)}` : y}</span>
              {!narrow && <span className="tabular text-[12px] text-ink-3" data-testid={`share-${y}`}>{pct(shares[i])}</span>}
            </button>
          );
        })}
      </div>
      {!narrow && (
        <p className="text-[12px] text-ink-3" style={{ paddingLeft: `${(ML / W) * 100}%` }}>
          % under each year: share of all California commercial landings that year (NOAA&apos;s state total), in the same unit.
        </p>
      )}
    </div>
  );
}

export function Legend({ groups, focus, onFocus, unit }: { groups: SpeciesGroup[]; focus: string | null; onFocus: (id: string | null) => void; unit: Unit }) {
  return (
    <div className="flex flex-wrap items-center gap-x-1 gap-y-1" role="group" aria-label="Species groups: choose one to see it alone on its own scale" data-testid="chart-legend">
      <button
        type="button"
        aria-pressed={focus == null}
        onClick={() => onFocus(null)}
        className={`rounded-full px-2.5 py-1 text-[13px] transition-colors duration-150 ${focus == null ? "bg-surface font-semibold text-ink ring-1 ring-hairline-strong" : "text-ink-2 hover:text-ink"}`}
        data-testid="legend-all"
      >
        All, stacked
      </button>
      {groups.map((g) => {
        const on = focus === g.id;
        return (
          <button
            key={g.id}
            type="button"
            aria-pressed={on}
            onClick={() => onFocus(on ? null : g.id)}
            className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[13px] transition-colors duration-150 ${on ? "bg-surface font-semibold text-ink ring-1 ring-hairline-strong" : "text-ink-2 hover:text-ink"}`}
            data-testid={`legend-${g.id}`}
            title={unit === "pounds" && g.id === "bivalves" ? "Bivalve pounds are meat weight (without shell)" : undefined}
          >
            <span className="h-2.5 w-2.5 shrink-0 rounded-[2px]" style={{ background: groupColor(g.id) }} aria-hidden />
            {groupShort(g)}
            {unit === "pounds" && g.id === "bivalves" ? <span className="text-ink-3">(meat weight)</span> : null}
          </button>
        );
      })}
    </div>
  );
}
