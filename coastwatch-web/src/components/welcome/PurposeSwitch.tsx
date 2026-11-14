"use client";

import { useId } from "react";
import { PURPOSES, purposeLabel, usePurpose, type PurposeChoice } from "@/lib/purpose";
import { TOUR_SEEN_KEY } from "@/components/map/Tour";
import s from "./purpose.module.css";

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
 * v15.1: drawn in the v14 design system's tokens (About is a page of the shell now).
 */
export function PurposeSwitch({ showReset = false, compact = false }: { showReset?: boolean; compact?: boolean }) {
  const id = useId();
  const [p, setP, ready] = usePurpose();
  if (!ready) return null;
  return (
    <div className={compact ? `${s.row} ${s.compact}` : s.row} data-testid="purpose-switch">
      <label htmlFor={id} className={s.label}>
        Set up for
      </label>
      <select
        id={id}
        value={valueOf(p)}
        onChange={(e) => setP(OPTIONS.find((o) => o.value === e.target.value)?.choice ?? null)}
        className={s.select}
        data-testid="purpose-select"
      >
        {OPTIONS.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
      <span className={s.sr} aria-live="polite">
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
            // v4 has no welcome card: start again from My Coast's choice of place
            window.location.assign("/coast");
          }}
          className={s.reset}
          data-testid="purpose-reset"
        >
          Forget this device&rsquo;s setup and start again
        </button>
      )}
    </div>
  );
}
