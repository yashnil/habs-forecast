"use client";

import type { FisheriesDataset, SpeciesGroup } from "@/generated/fisheries";
import { FISHERIES_EVENTS } from "@/content/copy";
import { formatValue, groupColor, groupShort, groupValue, pct, rowsWithoutValue, selectedTotal, shareOfState, stateTotal, withheldValue, yearFraction, type Unit } from "@/lib/fisheries";

/**
 * The year beside the chart: what each selected group landed, the total, NOAA's statewide total and
 * the withheld amount, all in the chosen unit. A documented event that overlaps the year is quoted as
 * context, with the reminder that annual totals are calendar years and seasons span two of them.
 */
export function YearPanel({ ds, groups, unit, year, focus, previewing }: { ds: FisheriesDataset; groups: SpeciesGroup[]; unit: Unit; year: number; focus: string | null; previewing: boolean }) {
  const total = selectedTotal(groups, year, unit);
  const state = stateTotal(ds, year, unit);
  const withheld = withheldValue(ds, year, unit);
  const missing = rowsWithoutValue(groups, year);
  const ids = new Set(groups.map((g) => g.id) as string[]);
  const events = FISHERIES_EVENTS.filter((e) => e.groups.some((g) => ids.has(g)) && yearFraction(e.start) < year + 1 && yearFraction(e.end) >= year);
  const rows = [...groups].sort((a, b) => (groupValue(b, year, unit) ?? -1) - (groupValue(a, year, unit) ?? -1));
  return (
    <div className="space-y-3" data-testid="year-panel" data-year={year} aria-live="polite">
      <div className="flex items-baseline justify-between gap-3 border-b border-hairline pb-2">
        <p className="font-display text-[30px] font-medium leading-none text-ink">{year}</p>
        <p className="text-[12px] text-ink-3">{previewing ? "Pointing at" : "Selected year"}</p>
      </div>
      <table className="w-full text-[13.5px] tabular">
        <caption className="sr-only">
          {year}, {unit === "pounds" ? "pounds landed" : "landed value"} by species group
        </caption>
        <tbody>
          {rows.map((g) => {
            const v = groupValue(g, year, unit);
            return (
              <tr key={g.id} className={focus && focus !== g.id ? "text-ink-3" : "text-ink"} data-testid={`year-row-${g.id}`}>
                <th scope="row" className="py-[3px] pr-2 text-left font-normal">
                  <span className="mr-2 inline-block h-2.5 w-2.5 rounded-[2px] align-baseline" style={{ background: groupColor(g.id) }} aria-hidden />
                  {groupShort(g)}
                </th>
                <td className="py-[3px] text-right">{formatValue(v, unit)}</td>
                <td className="w-[3.2rem] py-[3px] text-right text-ink-3">{pct(total ? (v ?? 0) / total : null)}</td>
              </tr>
            );
          })}
          <tr className="border-t border-hairline-strong font-semibold text-ink" data-testid="year-total">
            <th scope="row" className="pt-1.5 text-left">Selected groups</th>
            <td className="pt-1.5 text-right">{formatValue(total, unit)}</td>
            <td />
          </tr>
          <tr className="text-ink-2">
            <th scope="row" className="py-[3px] text-left font-normal">All California commercial</th>
            <td className="py-[3px] text-right">{formatValue(state, unit)}</td>
            <td className="py-[3px] text-right text-ink-3" data-testid="year-share">
              {pct(shareOfState(total, state))}
            </td>
          </tr>
        </tbody>
      </table>
      <p className="text-[13px] leading-snug text-ink-3">
        The state total includes {formatValue(withheld, unit)} that NOAA withheld for confidentiality and generic categories such as unspecified crabs. Neither belongs to a group, so group values are lower bounds.
        {missing > 0 && ` ${missing} source row${missing > 1 ? "s" : ""} in these groups had no published value this year (left out, not counted as zero).`}
      </p>
      {events.map((e) => (
        <div key={e.id} className="border-l-2 border-ink-2 pl-3 text-[13px] leading-snug text-ink-2" data-testid="year-event">
          <p className="font-semibold text-ink">
            {e.period}: {e.label}
          </p>
          <p className="mt-0.5">{e.detail}</p>
          <p className="mt-1">
            Historical context, not a measured effect: these are calendar-year totals, and a crab season runs across two calendar years.{" "}
            <a href={e.source.url} target="_blank" rel="noopener noreferrer" className="font-medium text-accent underline-offset-2 hover:underline">
              Timeline ↗
            </a>
          </p>
        </div>
      ))}
    </div>
  );
}
