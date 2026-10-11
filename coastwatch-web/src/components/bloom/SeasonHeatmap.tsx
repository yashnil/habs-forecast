"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ObsVariable } from "@/generated/observations";
import { fmtDate } from "@/components/charts/StationChart";
import { heatmap, periodOf, type Axis, type HeatCell } from "@/lib/bloom";
import { formatNumber, formatObs, formatTick, unitLabel, type Sample } from "@/lib/observations";

/** Light to dark teal: measured data (spec §5.3). Display bins, never risk levels. */
const RAMP = ["#cfe7e1", "#9fd0c5", "#62b1a2", "#2c8778", "#0b5a51"];
/** Zero-only periods: a quiet warm tone with a small dot. Not sampled: blank (no fill at all). */
const ZERO_BG = "#e9e5dc";
const ROW = 14;
const HEAD = 20;
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function binLabels(axis: Axis, unit: string): string[] {
  const e = axis.edges;
  const f = (v: number) => (axis.scale === "log" ? formatTick(v) : `${v}`);
  return [`< ${f(e[0])}`, ...e.slice(0, -1).map((v, i) => `${f(v)}–${f(e[i + 1])}`), `≥ ${f(e[e.length - 1])}`].map((s, i) => (i === e.length ? `${s} ${unit}` : s));
}

/**
 * Every week (or month, on a phone) since the station's first year: the highest value of the
 * selected quantity, binned on its display edges (spec §3.2 item 8). Weeks with only reported zeros
 * and weeks without a sample of this quantity have their own swatches, never a colour bin. It is
 * also the page's navigation through time (M6 §0.2 item 4): choosing a year sets the chart to it.
 */
export function SeasonHeatmap({
  variable,
  samples,
  axis,
  selectedYear,
  onYear,
  now,
}: {
  variable: ObsVariable;
  samples: Sample[];
  axis: Axis;
  selectedYear: number | null;
  onYear: (y: number | null) => void;
  now: Date | null;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(900);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.round(e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const period: "week" | "month" = width < 640 ? "month" : "week";
  const hm = useMemo(() => heatmap(samples, axis, period), [samples, axis, period]);
  const [hover, setHover] = useState<{ year: number; i: number; c: HeatCell | undefined } | null>(null);
  const unit = unitLabel(variable.units);
  const labels = binLabels(axis, unit);
  const nowP = now ? periodOf(now.getTime(), period) : null;
  const cols = hm.cols;
  const monthStarts = period === "week" ? MONTHS.map((_, m) => Math.floor((Date.UTC(2021, m, 1) - Date.UTC(2021, 0, 1)) / (7 * 86_400_000))) : MONTHS.map((_, m) => m);
  const describe = (year: number, i: number, c: HeatCell | undefined) => {
    const when =
      period === "week" ? `Week of ${fmtDate(Date.UTC(year, 0, 1, 12) + i * 7 * 86_400_000, { month: "short", day: "numeric", year: "numeric" })}` : `${MONTHS[i]} ${year}`;
    if (!c) return `${when}: not sampled for this quantity`;
    if (c.kind === "zero") return `${when}: reported 0 only (${c.n} sample${c.n > 1 ? "s" : ""}; not quantified, not absent)`;
    if (c.kind === "bin") return `${when}: highest ${formatObs(variable, c.max, null)} on ${fmtDate(c.at, { month: "short", day: "numeric" })} (${c.n} sample${c.n > 1 ? "s" : ""})`;
    return `${when}: not sampled`;
  };

  const gridW = Math.max(240, width - 44);
  const cw = gridW / cols;
  const drawn = useMemo(() => {
    const bins: string[] = RAMP.map(() => "");
    let zeros = "";
    let dots = "";
    let zeroCount = 0;
    let binCount = 0;
    hm.years.forEach((year, row) => {
      for (let i = 0; i < cols; i++) {
        const c = hm.cells.get(`${year}:${i}`);
        if (!c) continue; // not sampled: blank
        const x = i * cw + 0.5;
        const y = HEAD + row * ROW + 0.5;
        const rect = `M${x.toFixed(1)},${y}h${(cw - 1).toFixed(1)}v${ROW - 1}h${(-(cw - 1)).toFixed(1)}z`;
        if (c.kind === "bin") {
          bins[c.bin] += rect;
          binCount++;
        } else {
          zeros += rect;
          zeroCount++;
          const cx = x + (cw - 1) / 2;
          const cy = y + (ROW - 1) / 2;
          dots += `M${(cx - 1.5).toFixed(1)},${cy.toFixed(1)}a1.5,1.5 0 1,0 3,0a1.5,1.5 0 1,0 -3,0`;
        }
      }
    });
    return { bins, zeros, dots, zeroCount, binCount };
  }, [hm, cols, cw]);

  if (!hm.years.length) {
    return (
      <p className="text-[13.5px] text-ink-2" data-testid="heatmap-empty">
        This quantity has not been measured at this station, so there is no seasonal record to show.
      </p>
    );
  }

  return (
    <div ref={ref} className="min-w-0 space-y-3" data-testid="season-heatmap" data-period={period}>
      {/* One SVG: a path per colour, one pointer handler that finds the cell under the pointer. About
          690 cells as separate elements with their own handlers cost ~140 ms of blocking time on a
          throttled phone; this draws the same encoding in a handful of nodes. Years stay real buttons. */}
      <div className="flex min-w-0" data-testid="heat-grid">
        <div className="w-[44px] shrink-0" style={{ paddingTop: HEAD }}>
          {hm.years.map((year) => {
            const isSel = selectedYear === year;
            const isNow = nowP?.year === year;
            return (
              <button
                key={year}
                type="button"
                onClick={() => onYear(isSel ? null : year)}
                aria-pressed={isSel}
                aria-label={`Show ${year} in the chart`}
                className={`block w-full pr-1.5 text-left text-[13px] tabular hover:text-accent focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent ${isSel ? "font-semibold text-accent" : isNow ? "font-semibold text-ink" : "text-ink-3"}`}
                style={{ height: ROW, lineHeight: `${ROW}px` }}
                data-testid={`heat-year-${year}`}
              >
                {year}
              </button>
            );
          })}
        </div>
        <svg
          width={gridW}
          height={HEAD + hm.years.length * ROW}
          className="block shrink-0"
          aria-hidden
          onPointerMove={(e) => {
            const r = e.currentTarget.getBoundingClientRect();
            const i = Math.floor((e.clientX - r.left) / cw);
            const row = Math.floor((e.clientY - r.top - HEAD) / ROW);
            const year = hm.years[row];
            if (year == null || i < 0 || i >= cols) return setHover(null);
            setHover({ year, i, c: hm.cells.get(`${year}:${i}`) });
          }}
          onPointerLeave={() => setHover(null)}
          data-zero-cells={drawn.zeroCount}
          data-bin-cells={drawn.binCount}
        >
          {MONTHS.map((m, k) => (
            <text key={m} x={monthStarts[k] * cw + 1} y={HEAD - 6} className="fill-[var(--cw-ink-3)] text-[12px]">
              {period === "month" ? m[0] : m}
            </text>
          ))}
          {drawn.bins.map((d, b) => (d ? <path key={b} d={d} fill={RAMP[b]} /> : null))}
          {drawn.zeros && <path d={drawn.zeros} fill={ZERO_BG} />}
          {drawn.dots && <path d={drawn.dots} fill="var(--cw-ink-3)" fillOpacity={0.6} />}
          {selectedYear != null && hm.years.includes(selectedYear) && (
            <rect x={0.5} y={HEAD + hm.years.indexOf(selectedYear) * ROW + 0.5} width={gridW - 1} height={ROW - 1} fill="none" stroke="var(--cw-accent)" strokeOpacity={0.6} />
          )}
          {nowP && hm.years.includes(nowP.year) && (
            <rect x={nowP.i * cw + 1} y={HEAD + hm.years.indexOf(nowP.year) * ROW + 1} width={cw - 2} height={ROW - 2} fill="none" stroke="var(--cw-ink)" strokeWidth={1.5} />
          )}
          {hover && hm.years.includes(hover.year) && (
            <rect x={hover.i * cw + 0.5} y={HEAD + hm.years.indexOf(hover.year) * ROW + 0.5} width={cw - 1} height={ROW - 1} fill="none" stroke="var(--cw-accent)" strokeWidth={1.5} />
          )}
        </svg>
      </div>

      {/* screen-reader alternative; a table cannot shrink to the 1 px sr-only box itself, so it is wrapped */}
      <div className="sr-only">
        <table data-testid="heat-table">
          <caption>
            {variable.label}: highest value each year at this station, and how many {period}s were sampled
          </caption>
          <thead>
            <tr>
              <th scope="col">Year</th>
              <th scope="col">Highest value</th>
              <th scope="col">{period === "week" ? "Weeks" : "Months"} with a value</th>
              <th scope="col">{period === "week" ? "Weeks" : "Months"} with reported 0 only</th>
            </tr>
          </thead>
          <tbody>
            {hm.years.map((year) => {
              const cs = [...hm.cells.entries()].filter(([k]) => k.startsWith(`${year}:`)).map(([, c]) => c);
              const bins = cs.filter((c): c is Extract<HeatCell, { kind: "bin" }> => c.kind === "bin");
              const top = bins.reduce<Extract<HeatCell, { kind: "bin" }> | null>((a, c) => (!a || c.max > a.max ? c : a), null);
              return (
                <tr key={year}>
                  <th scope="row">{year}</th>
                  <td>{top ? `${formatObs(variable, top.max, null)} on ${fmtDate(top.at)}` : "none (not sampled, or reported 0 only)"}</td>
                  <td>{bins.length}</td>
                  <td>{cs.length - bins.length}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="min-h-[20px] text-[13px] text-ink-2" aria-live="polite" data-testid="heat-detail">
        {hover ? describe(hover.year, hover.i, hover.c) : "Point at a cell for its highest value and date; choose a year to chart it."}
      </p>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[13px] text-ink-2" data-testid="heat-legend">
        {labels.map((l, i) => (
          <span key={l} className="inline-flex items-center gap-1.5">
            <span className="inline-block h-3 w-3" style={{ background: RAMP[i] }} />
            {l}
          </span>
        ))}
        <span className="inline-flex items-center gap-1.5">
          <span className="relative inline-block h-3 w-3" style={{ background: ZERO_BG }}>
            <span className="absolute left-1/2 top-1/2 h-[3px] w-[3px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-ink-3/60" />
          </span>
          reported 0 only
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 border border-dashed border-hairline-strong" />
          not sampled (blank)
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-3 w-3" style={{ boxShadow: "inset 0 0 0 1.5px var(--cw-ink)" }} />
          this {period}
        </span>
      </div>
      <p className="text-[13px] text-ink-3">
        {axis.scale === "log" ? "Decade bins" : `${formatNumber(variable, axis.edges[1] - axis.edges[0])} ${unit} bins`} for reading the pattern, <b className="font-semibold">not risk levels</b>. No
        regulatory action level applies to seawater; action levels apply to seafood tissue.
      </p>
    </div>
  );
}
