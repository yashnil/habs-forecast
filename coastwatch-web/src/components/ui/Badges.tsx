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
  historical: {
    color: "var(--cw-serious)",
    icon: (
      <>
        <circle cx="6" cy="6" r="4.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
        <path d="M6 3.3v3l2 1.2" stroke="currentColor" strokeWidth="1.4" fill="none" strokeLinecap="round" />
      </>
    ),
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
};

export function ProductClassBadge({ pc }: { pc: ProductClass }) {
  const style =
    pc === "official_forecast" || pc === "official_regulatory"
      ? "bg-ink text-page border-ink"
      : pc === "experimental_model"
        ? "border-dashed border-ink-3 text-ink-2"
        : "border-hairline-strong text-ink-2";
  return (
    <span
      data-testid="product-class"
      className={`inline-flex items-center rounded-md border px-1.5 py-px text-[10px] font-semibold uppercase tracking-wider ${style}`}
    >
      {CLASS_LABEL[pc]}
    </span>
  );
}
