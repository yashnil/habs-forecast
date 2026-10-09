import type { ProductClass } from "@/generated/schema";
import { FRESHNESS_LABEL, type Freshness, type FreshnessState } from "@/lib/freshness";
import { formatAge } from "@/lib/time";

const STATE_STYLE: Record<FreshnessState, { color: string; icon: React.ReactNode }> = {
  current: {
    color: "var(--cw-good)",
    icon: <circle cx="6" cy="6" r="4.5" fill="currentColor" />,
  },
  stale: {
    color: "var(--cw-warning)",
    icon: (
      <>
        <circle cx="6" cy="6" r="4.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
        <path d="M6 1.5 A4.5 4.5 0 0 1 6 10.5 Z" fill="currentColor" />
      </>
    ),
  },
  // Historical: an open ring in neutral grey. Old data is not an alarm, it is simply not current.
  historical: {
    color: "var(--cw-neutral)",
    icon: <circle cx="6" cy="6" r="4.25" fill="none" stroke="currentColor" strokeWidth="1.6" />,
  },
  unavailable: {
    color: "var(--cw-neutral)",
    icon: <path d="M2.5 2.5l7 7M9.5 2.5l-7 7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />,
  },
};

const BASIS_WORD: Record<string, string> = { issued_date: "issued", observed_date: "observed", valid_date: "updated" };

export function FreshnessBadge({ f, compact = false, basis }: { f: Freshness | null; compact?: boolean; basis?: string }) {
  const state: FreshnessState = f?.state ?? "unavailable";
  const s = STATE_STYLE[state];
  return (
    <span
      data-testid="freshness"
      data-state={state}
      className="inline-flex items-center gap-1.5 rounded-full border border-hairline-strong bg-surface-2 px-2 py-0.5 text-[11px] font-medium text-ink"
    >
      <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden style={{ color: s.color }}>
        {s.icon}
      </svg>
      {f ? FRESHNESS_LABEL[state] : "Checking…"}
      {!compact && f?.ageDays != null && (
        <span className="text-ink-3">
          · {basis ? `${BASIS_WORD[basis] ?? ""} ` : ""}
          {formatAge(f.ageDays)}
        </span>
      )}
    </span>
  );
}

const CLASS_LABEL: Record<ProductClass, string> = {
  official_regulatory: "Official · regulatory",
  official_forecast: "Agency forecast",
  observation: "Observation",
  experimental_model: "Experimental",
  historical_context: "Historical",
  reference: "Reference",
  derived_summary: "Derived summary",
};

// Semantic product colours (design reset §2.2): violet = model, teal = measured,
// amber = official, slate = historical. Never used decoratively.
const CLASS_STYLE: Record<ProductClass, string> = {
  official_regulatory: "border-official-line bg-official-bg text-official-ink",
  official_forecast: "border-model-line bg-model-bg text-model-ink",
  observation: "border-measured-line bg-measured-bg text-measured",
  experimental_model: "border-dashed border-ink-3 text-ink-2",
  historical_context: "border-history-line bg-history-bg text-history",
  reference: "border-hairline-strong text-ink-2",
  derived_summary: "border-hairline-strong text-ink-2",
};

export function ProductClassBadge({ pc }: { pc: ProductClass }) {
  return (
    <span
      data-testid="product-class"
      className={`inline-flex items-center rounded-full border px-2 py-px text-[10.5px] font-semibold uppercase tracking-wider ${CLASS_STYLE[pc]}`}
    >
      {CLASS_LABEL[pc]}
    </span>
  );
}

/** Issuing agency of an official notice (CDFW, CDPH, …), always in the official amber family. */
export function AgencyChip({ agency }: { agency: string }) {
  return (
    <span className="mt-px inline-flex shrink-0 items-center rounded border border-official-line bg-surface px-1 text-[10.5px] font-semibold tracking-wide text-official-ink">
      {agency}
    </span>
  );
}
