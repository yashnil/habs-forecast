"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ObsVariable } from "@/generated/observations";
import { fmtDate } from "@/components/charts/StationChart";
import { heatmap, periodOf, type Axis, type HeatCell } from "@/lib/bloom";
import { formatNumber, formatObs, unitLabel, type Sample } from "@/lib/observations";

/** Light to dark teal: measured data (spec §5.3). Display bins, never risk levels. */
const RAMP = ["#e1f0ec", "#b3dbd2", "#78bfb1", "#3a978a", "#0d655b"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function binLabels(axis: Axis, unit: string): string[] {
  const e = axis.edges;
  const f = (v: number) => (axis.scale === "log" ? String(v) : `${v}`);
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

  if (!hm.years.length) {
    return (
      <p className="text-[13.5px] text-ink-2" data-testid="heatmap-empty">
        This quantity has not been measured at this station, so there is no seasonal record to show.
      </p>
    );
  }

  return (
    <div ref={ref} className="min-w-0 space-y-3" data-testid="season-heatmap" data-period={period}>
      <div className="overflow-x-auto">
        <div className="min-w-[320px]">
          {/* month labels */}
          <div className="grid items-end" style={{ gridTemplateColumns: `44px repeat(${cols}, minmax(0, 1fr))` }}>
            <span />
            {Array.from({ length: cols }, (_, i) => {
              const m = monthStarts.indexOf(i);
              return (
                <span key={i} className="whitespace-nowrap text-[13px] leading-5 text-ink-3">
                  {m >= 0 ? (period === "month" ? MONTHS[m][0] : MONTHS[m]) : ""}
                </span>
              );
            })}
          </div>
          {hm.years.map((year) => {
            const isSel = selectedYear === year;
            const isNow = nowP?.year === year;
            return (
              <div key={year} className="grid items-center" style={{ gridTemplateColumns: `44px repeat(${cols}, minmax(0, 1fr))` }} data-year={year}>
                <button
                  type="button"
                  onClick={() => onYear(isSel ? null : year)}
                  aria-pressed={isSel}
                  aria-label={`Show ${year} in the chart`}
                  className={`h-[14px] pr-1.5 text-left text-[13px] leading-[14px] tabular hover:text-accent focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent ${isSel ? "font-semibold text-accent" : isNow ? "font-semibold text-ink" : "text-ink-3"}`}
                  data-testid={`heat-year-${year}`}
                >
                  {year}
                </button>
                {Array.from({ length: cols }, (_, i) => {
                  const c = hm.cells.get(`${year}:${i}`);
                  const future = nowP && (year > nowP.year || (year === nowP.year && i > nowP.i));
                  const isNowCell = nowP && year === nowP.year && i === nowP.i;
                  return (
                    <span
                      key={i}
                      onPointerEnter={() => setHover({ year, i, c })}
                      onPointerLeave={() => setHover(null)}
                      className={`relative block h-[14px] border-[0.5px] border-surface ${isSel ? "outline outline-1 outline-accent/40" : ""}`}
                      style={{
                        background: future ? "transparent" : !c ? "var(--cw-surface-2)" : c.kind === "bin" ? RAMP[c.bin] : "var(--cw-surface-3)",
                        boxShadow: isNowCell ? "inset 0 0 0 1.5px var(--cw-ink)" : undefined,
                      }}
                      data-kind={c?.kind ?? (future ? "future" : "unsampled")}
                    >
                      {c?.kind === "zero" && <span className="absolute left-1/2 top-1/2 h-[4px] w-[4px] -translate-x-1/2 -translate-y-1/2 rounded-full border border-ink-3" />}
                    </span>
                  );
                })}
              </div>
            );
          })}
        </div>
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
          <span className="relative inline-block h-3 w-3 bg-surface-3">
            <span className="absolute left-1/2 top-1/2 h-[4px] w-[4px] -translate-x-1/2 -translate-y-1/2 rounded-full border border-ink-3" />
          </span>
          reported 0 only
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 border border-hairline bg-surface-2" />
          not sampled
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
