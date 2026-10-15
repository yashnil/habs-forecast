"use client";

import type { Palette } from "@/generated/schema";
import { colorAt } from "@/components/ui/ProbabilityLegend";
import { formatDate, relativeDay } from "@/lib/time";

export type LeadItem = { lead: number; date: string | null; value: number | null; issued: boolean };

/**
 * The C-HARM days at one place (M5 inspector): nowcast to +3 days, each with its valid date
 * and value, as a row of buttons that also set the map's day. A day the run did not issue,
 * or a cell without a value, shows a dash; nothing is filled in between days.
 */
export function LeadStrip({ items, lead, onLead, palette, label, testidPrefix, valueNote, now }: { items: LeadItem[]; lead: number; onLead: (l: number) => void; palette: Palette | null | undefined; label: string; testidPrefix: string; valueNote?: string; now?: Date | null }) {
  return (
    <div role="radiogroup" aria-label={label} className="grid grid-cols-4 gap-1">
      {items.map((it) => {
        const on = it.lead === lead;
        const v = it.value;
        // the chip names the day's distance from today ("today", "2 days ago"); "nowcast" is the
        // run's word for lead 0 and stays in the accessible name, where it cannot clash with the date
        const when = it.date && now ? relativeDay(it.date, now) : it.lead === 0 ? "nowcast" : `+${it.lead}d`;
        return (
          <button
            key={it.lead}
            type="button"
            role="radio"
            aria-checked={on}
            disabled={!it.issued}
            data-testid={`${testidPrefix}-${it.lead}`}
            aria-label={`${it.lead === 0 ? "Nowcast" : `Forecast +${it.lead} day${it.lead > 1 ? "s" : ""}`}${it.date ? `, ${formatDate(it.date)}` : ""}: ${v != null ? `${Math.round(v * 100)}%${valueNote ? ` ${valueNote}` : ""}` : it.issued ? "no value" : "not issued"}`}
            onClick={() => onLead(it.lead)}
            className={`flex flex-col items-stretch gap-1 rounded-lg border px-1.5 pb-1.5 pt-1 text-left transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-40 ${on ? "border-navy-900 ring-1 ring-navy-900" : "border-hairline hover:border-hairline-strong"}`}
          >
            <span className="flex items-baseline justify-between gap-1">
              <span className="truncate text-[13px] font-medium text-ink-3">{when}</span>
              <span className="text-[13px] text-ink-3 tabular">{it.date ? formatDate(it.date).replace(/^\w+, /, "") : "—"}</span>
            </span>
            <span className="text-[17px] font-semibold leading-none text-ink tabular">{v != null ? `${Math.round(v * 100)}%` : "—"}</span>
            <span className="h-1.5 overflow-hidden rounded-full bg-surface-3" aria-hidden>
              {v != null && palette && <span className="block h-full rounded-full" style={{ width: `${Math.max(3, v * 100)}%`, background: colorAt(palette, v) }} />}
            </span>
          </button>
        );
      })}
    </div>
  );
}
