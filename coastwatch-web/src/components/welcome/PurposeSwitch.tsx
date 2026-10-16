"use client";

import { useId } from "react";
import { PURPOSES, purposeLabel, usePurpose, type PurposeChoice } from "@/lib/purpose";
import { TOUR_SEEN_KEY } from "@/components/map/Tour";

const OPTIONS: { value: string; label: string; choice: PurposeChoice | null }[] = [
  { value: "none", label: "No preference", choice: null },
  ...PURPOSES.flatMap((p) =>
    p.id === "recreational"
      ? [
          { value: "recreational:fishing", label: "Recreational fishing", choice: { purpose: "recreational", harvest: "fishing" } as PurposeChoice },
          { value: "recreational:shellfish", label: "Shellfish harvest", choice: { purpose: "recreational", harvest: "shellfish" } as PurposeChoice },
        ]
      : [{ value: p.id, label: p.label, choice: { purpose: p.id } as PurposeChoice }],
  ),
];
const valueOf = (c: PurposeChoice | null) => (!c ? "none" : c.purpose === "recreational" ? `recreational:${c.harvest ?? "fishing"}` : c.purpose);

/**
 * Change (or clear) what Neritic is set up for. Kept on this device only; it changes defaults
 * and order, never what the data say. "Show the welcome again" brings back the first-visit card.
 */
export function PurposeSwitch({ showReset = false, compact = false }: { showReset?: boolean; compact?: boolean }) {
  const id = useId();
  const [p, setP, ready] = usePurpose();
  if (!ready) return null;
  return (
    <div className={`flex flex-wrap items-center gap-x-2 gap-y-1 text-[14px] ${compact ? "" : "rounded-xl border border-hairline bg-surface px-4 py-3"}`} data-testid="purpose-switch">
      <label htmlFor={id} className="text-ink-2">
        Set up for
      </label>
      <select
        id={id}
        value={valueOf(p)}
        onChange={(e) => setP(OPTIONS.find((o) => o.value === e.target.value)?.choice ?? null)}
        className="min-h-[44px] rounded-md border border-hairline-strong bg-surface px-2 text-[14px] text-ink"
        data-testid="purpose-select"
      >
        {OPTIONS.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
      <span className="sr-only" aria-live="polite">
        Now set up for {purposeLabel(p)}
      </span>
      {showReset && (
        <button
          type="button"
          onClick={() => {
            setP(null);
            try {
              window.localStorage.removeItem(TOUR_SEEN_KEY);
              window.localStorage.removeItem("cw-my-waters");
            } catch {
              // nothing stored
            }
            window.location.assign("/");
          }}
          className="min-h-[36px] font-medium text-accent hover:underline"
          data-testid="purpose-reset"
        >
          Show the welcome again
        </button>
      )}
    </div>
  );
}
