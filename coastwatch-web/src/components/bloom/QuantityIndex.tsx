"use client";

import type { ObsStation, ObsVariable } from "@/generated/observations";
import type { FreshnessPolicy } from "@/generated/schema";
import { fmtDate } from "@/components/charts/StationChart";
import { AXES, SELECTOR_VARS, headline, previewPoints, type SelectorVar } from "@/lib/bloom";
import { formatNumber, monthYear, samples, summaryOf, unitLabel } from "@/lib/observations";
import { FreshDot } from "./FreshDot";

/** What each quantity is, in a few words, under its name. */
export const QUANTITY_NOTE: Record<SelectorVar, string> = {
  pDA: "Toxin in seawater",
  pn_seriata: "Larger cells",
  pn_delicatissima: "Smaller cells",
  chl_extracted: "Algae biomass, not toxin",
  temp: "At the sampling point",
};
export const QUANTITY_SHORT: Record<SelectorVar, string> = {
  pDA: "Particulate domoic acid",
  pn_seriata: "Pseudo-nitzschia, seriata group",
  pn_delicatissima: "Pseudo-nitzschia, delicatissima group",
  chl_extracted: "Chlorophyll-a",
  temp: "Water temperature",
};

/**
 * The measurement selector as an index, not dashboard cards (M6 §0.2 item 2): one hairline strip,
 * a column per quantity with its latest value, date and freshness, and a preview of the chart's
 * period on the chart's fixed axis. A value older than the stale limit is never shown at headline
 * size (rule 8). Selecting a column makes it the primary chart.
 */
export function QuantityIndex({
  station,
  vars,
  selected,
  onSelect,
  policy,
  now,
  x0,
  x1,
}: {
  station: ObsStation;
  vars: Record<string, ObsVariable>;
  selected: SelectorVar;
  onSelect: (v: SelectorVar) => void;
  policy: FreshnessPolicy | null;
  now: Date | null;
  x0: number;
  x1: number;
}) {
  return (
    <section aria-labelledby="index-h" className="space-y-2">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4">
        <h2 id="index-h" className="text-[14px] font-semibold text-ink">
          Latest measurement of each quantity
        </h2>
        <p className="text-[13px] text-ink-3">Choose one to chart it. Previews show the chart&apos;s period on its axis.</p>
      </div>
      <div
        role="tablist"
        aria-label="Quantity to chart"
        className="-mx-4 flex snap-x snap-mandatory overflow-x-auto border-y border-hairline px-4 [scrollbar-width:none] sm:mx-0 sm:grid sm:grid-cols-5 sm:overflow-visible sm:px-0"
        data-testid="quantity-index"
      >
        {SELECTOR_VARS.map((id) => (
          <Column key={id} id={id} v={vars[id]} station={station} active={id === selected} onSelect={onSelect} policy={policy} now={now} x0={x0} x1={x1} />
        ))}
      </div>
      <p className="text-[13px] text-ink-3">A Pseudo-nitzschia count is a size group of cells; not every Pseudo-nitzschia produces toxin. Chlorophyll measures algae biomass, not toxin.</p>
    </section>
  );
}

function Column({
  id,
  v,
  station,
  active,
  onSelect,
  policy,
  now,
  x0,
  x1,
}: {
  id: SelectorVar;
  v: ObsVariable;
  station: ObsStation;
  active: boolean;
  onSelect: (v: SelectorVar) => void;
  policy: FreshnessPolicy | null;
  now: Date | null;
  x0: number;
  x1: number;
}) {
  const sm = summaryOf(station, id);
  const h = policy && now ? headline(sm, policy, now) : null;
  const unit = unitLabel(v.units);
  const value = (val: number | null, q: string | null | undefined) => (q === "reported_zero" ? "0*" : val == null ? "—" : formatNumber(v, val));
  const pts = previewPoints(samples(station, id), AXES[id], x0, x1);
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      onClick={() => onSelect(id)}
      data-testid={`quantity-${id}`}
      data-headline={h?.kind}
      className={`group relative w-[72vw] max-w-[260px] shrink-0 snap-start border-r border-hairline px-3 pb-3 pt-2.5 text-left transition-colors last:border-r-0 hover:bg-surface-2 focus-visible:outline focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-accent sm:w-auto sm:max-w-none ${active ? "bg-measured-bg/60" : ""}`}
    >
      <span className={`absolute inset-x-0 bottom-0 h-[3px] ${active ? "bg-measured" : "bg-transparent"}`} aria-hidden />
      <span className="block text-[13px] font-medium leading-tight text-ink">{QUANTITY_SHORT[id]}</span>
      <span className="block text-[13px] leading-tight text-ink-3">{QUANTITY_NOTE[id]}</span>
      <span className="mt-2 block min-h-[54px]">
        {!h || h.kind === "never" ? (
          <span className="block text-[14px] font-medium leading-snug text-ink-2">Never measured here</span>
        ) : h.kind === "old" ? (
          <>
            <span className="block text-[14px] font-medium leading-snug text-ink-2">Not measured since {monthYear(h.lastDate)}</span>
            <span className="block text-[13px] text-ink-3">
              last: {value(h.value, h.q)} {h.q === "reported_zero" ? "(reported 0)" : unit}
            </span>
          </>
        ) : (
          <>
            <span className="block whitespace-nowrap text-[24px] font-semibold leading-tight text-ink tabular">
              {value(h.value, h.q)}
              <span className="ml-1 text-[13px] font-normal text-ink-3">{h.q === "reported_zero" ? "reported 0" : unit}</span>
            </span>
            <span className="mt-0.5 flex items-center gap-1.5 text-[13px] text-ink-3">
              <FreshDot f={h.freshness} showWord={false} />
              {fmtDate(Date.parse(`${h.lastDate}T12:00:00Z`), { month: "short", day: "numeric" })} · {sm?.n_last_365d ?? 0} in 12 mo
            </span>
          </>
        )}
      </span>
      <Preview pts={pts} axisLog={AXES[id].scale === "log"} domain={AXES[id].domain} x0={x0} x1={x1} />
      <span className="block text-[13px] text-ink-3">{pts.length ? `${pts.length} in view` : "none in view"}</span>
    </button>
  );
}

function Preview({ pts, axisLog, domain, x0, x1 }: { pts: ReturnType<typeof previewPoints>; axisLog: boolean; domain: [number, number]; x0: number; x1: number }) {
  const W = 200;
  const H = 36;
  const x = (t: number) => 2 + ((t - x0) / Math.max(1, x1 - x0)) * (W - 4);
  const y = (v: number) => {
    const f = axisLog ? (Math.log10(v) - Math.log10(domain[0])) / (Math.log10(domain[1]) - Math.log10(domain[0])) : (v - domain[0]) / (domain[1] - domain[0]);
    return 2 + (1 - f) * (H - 9);
  };
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" className="mt-1.5 block" aria-hidden data-testid="preview">
      <line x1={0} x2={W} y1={H - 2} y2={H - 2} stroke="var(--cw-hairline)" />
      {pts.map((p, i) =>
        p.zero ? (
          <circle key={i} cx={x(p.t)} cy={H - 2.5} r={1.4} fill="var(--cw-surface)" stroke="var(--cw-ink-3)" strokeWidth={0.7} />
        ) : (
          <circle key={i} cx={x(p.t)} cy={y(p.v)} r={1.6} fill="var(--cw-chl)" />
        ),
      )}
    </svg>
  );
}
