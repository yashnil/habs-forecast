"use client";

import { useState } from "react";
import type { FisheriesDataset } from "@/generated/fisheries";
import { FISHERIES_COPY, FISHERIES_EVENTS } from "@/content/copy";
import { SourceLink } from "@/components/ui/Primitives";
import { formatValue, stateTotal, unitLabel, withheldValue, type Unit } from "@/lib/fisheries";
import { formatDateTimePT } from "@/lib/time";

function Note({ title, compact, children }: { title: string; compact: boolean; children: React.ReactNode }) {
  if (compact)
    return (
      <details className="py-2.5">
        <summary className="cursor-pointer font-semibold text-ink">{title}</summary>
        <div className="mt-1">{children}</div>
      </details>
    );
  return (
    <div>
      <h3 className="mb-1 text-[14px] font-semibold text-ink">{title}</h3>
      {children}
    </div>
  );
}

/**
 * How these numbers are made (M7 §4.7): four short columns in plain language, then disclosures for
 * every caveat, the full values table (with the withheld row), the inflation index, duplicates, the
 * processing steps, the sources and their licences. Same pattern as Bloom's "About these data".
 */
export function FisheriesMethods({ ds, unit, compact }: { ds: FisheriesDataset; unit: Unit; compact: boolean }) {
  const base = ds.deflator.base_year;
  const foss = ds.provenance.find((p) => p.source_id === "foss_landings") ?? ds.provenance[0];
  // the 80-cell values table is built when first opened, not on load
  const [valuesOpen, setValuesOpen] = useState(false);
  return (
    <section aria-labelledby="methods-h" className="space-y-4" data-testid="fisheries-methods">
      <h2 id="methods-h" className="font-display text-[24px] font-medium leading-tight text-ink">
        How these numbers are made
      </h2>
      <div className={compact ? "divide-y divide-hairline border-y border-hairline text-[14.5px] leading-relaxed text-ink-2" : "grid gap-x-8 gap-y-4 text-[14.5px] leading-relaxed text-ink-2 md:grid-cols-2 xl:grid-cols-4"} data-testid="fisheries-notes">
        <Note title="Source" compact={compact}>
          <p>
            NOAA Fisheries&apos; commercial landings for California (compiled from PacFIN), one calendar year at a time, January to December. NOAA revises recent years and calls annual summaries preliminary. Courtesy: National Oceanic and Atmospheric Administration.
          </p>
        </Note>
        <Note title="Dollars" compact={compact}>
          <p data-testid="dollars-note">
            Ex-vessel revenue: what buyers paid at the dock. “Nominal” is as reported each year; “{base} dollars” rescales it with the consumer price index (CPI-U). That index tracks general consumer prices, not seafood prices or fishing costs, and revenue is not profit.
          </p>
        </Note>
        <Note title="Confidentiality" compact={compact}>
          <p>
            NOAA withholds landings it treats as confidential and publishes them as one statewide amount per year ({formatValue(withheldValue(ds, ds.years[ds.years.length - 1], unit), unit)} in {ds.years[ds.years.length - 1]}). It is in the state total but belongs to no group, so group values are lower bounds. Nothing is divided among ports.
          </p>
        </Note>
        <Note title="Species groups" compact={compact}>
          <p>
            NOAA reports market categories; each is assigned to at most one group by exact name, and generic ones (such as unspecified crabs) to none. {FISHERIES_COPY.reviewPending} Bivalve pounds are meat weight and include aquaculture.
          </p>
        </Note>
      </div>

      <div className="divide-y divide-hairline border-y border-hairline text-[13.5px]">
        <details className="py-2.5" data-testid="values-disclosure" onToggle={(e) => e.currentTarget.open && setValuesOpen(true)}>
          <summary className="cursor-pointer font-medium text-ink">Every value, {unitLabel(unit, base)}</summary>
          {valuesOpen && (
          <div className="mt-2 overflow-x-auto" tabIndex={0} role="region" aria-label="Every value, scrolls sideways">
            <table className="w-full min-w-[860px] text-right text-[13px] tabular" data-testid="fisheries-table">
              <caption className="sr-only">Landings by species group and year, {unitLabel(unit, base)}</caption>
              <thead className="text-ink-3">
                <tr>
                  <th scope="col" className="py-1 pr-2 text-left font-medium">
                    Group
                  </th>
                  {ds.years.map((y) => (
                    <th scope="col" key={y} className="px-1.5 py-1 font-medium">
                      {y}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {ds.groups.map((g) => (
                  <tr key={g.id} className="border-t border-hairline">
                    <th scope="row" className="py-1 pr-2 text-left font-normal text-ink">
                      {g.label} <span className="text-ink-3">· Tier {g.tier}</span>
                    </th>
                    {ds.years.map((y) => {
                      const a = g.annual.find((x) => x.year === y);
                      const v = a ? (unit === "real" ? a.dollars_real : unit === "nominal" ? a.dollars_nominal : a.pounds) : null;
                      return (
                        <td key={y} className="px-1.5 py-1 text-ink-2">
                          {formatValue(v ?? null, unit)}
                          {a?.n_rows_without_value ? <sup title={`${a.n_rows_without_value} source row(s) without a published value`}>*</sup> : null}
                        </td>
                      );
                    })}
                  </tr>
                ))}
                <tr className="border-t-2 border-hairline-strong">
                  <th scope="row" className="py-1 pr-2 text-left font-medium text-ink">
                    All California commercial landings (NOAA state total)
                  </th>
                  {ds.years.map((y) => (
                    <td key={y} className="px-1.5 py-1 text-ink">
                      {formatValue(stateTotal(ds, y, unit), unit)}
                    </td>
                  ))}
                </tr>
                <tr className="border-t border-hairline" data-testid="withheld-row">
                  <th scope="row" className="py-1 pr-2 text-left font-normal text-ink-2">
                    of which withheld for confidentiality (not attributed)
                  </th>
                  {ds.years.map((y) => (
                    <td key={y} className="px-1.5 py-1 text-ink-3">
                      {formatValue(withheldValue(ds, y, unit), unit)}
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
          )}
          <p className="mt-1.5 text-[12.5px] text-ink-3">* A source row in this group and year had no published value; it is left out, never counted as zero.</p>
        </details>
        <details className="py-2.5" data-testid="deflator">
          <summary className="cursor-pointer font-medium text-ink">Inflation adjustment ({ds.deflator.series_id})</summary>
          <div className="mt-2 space-y-1.5 text-ink-2">
            <p>
              {ds.deflator.title}. {ds.deflator.method}
            </p>
            {(ds.deflator.notes ?? []).map((n) => (
              <p key={n}>{n}</p>
            ))}
            <p className="tabular text-[12.5px] text-ink-3">
              Annual index:{" "}
              {Object.entries(ds.deflator.annual_index)
                .map(([y, v]) => `${y} ${v}`)
                .join(" · ")}
            </p>
            <SourceLink href={ds.deflator.source_url}>BLS series {ds.deflator.series_id} ↗</SourceLink>
          </div>
        </details>
        {ds.excluded_rows.length > 0 && (
          <details className="py-2.5" data-testid="excluded-rows">
            <summary className="cursor-pointer font-medium text-ink">{ds.excluded_rows.length} duplicate source rows counted once</summary>
            <ul className="mt-2 space-y-0.5 text-ink-2">
              {ds.excluded_rows.map((e) => (
                <li key={`${e.year}-${e.source_name}`}>
                  {e.year}: “{e.source_name}” duplicates “{e.duplicate_of}” ({formatValue(e.dollars_nominal, "nominal")} nominal). {e.reason}
                </li>
              ))}
            </ul>
          </details>
        )}
        <details className="py-2.5">
          <summary className="cursor-pointer font-medium text-ink">All caveats</summary>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-ink-2" data-testid="fisheries-caveats">
            {ds.caveats.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
        </details>
        <details className="py-2.5">
          <summary className="cursor-pointer font-medium text-ink">How CoastWatch processes these data</summary>
          <dl className="mt-2 grid gap-x-4 gap-y-1.5 sm:grid-cols-[auto_1fr]">
            {Object.entries(ds.method).map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="capitalize text-ink-3">{k.replace("_", " ")}</dt>
                <dd className="text-ink-2">{v}</dd>
              </div>
            ))}
          </dl>
        </details>
      </div>

      <div className="space-y-1 text-[13px] leading-relaxed text-ink-3" data-testid="fisheries-provenance">
        {ds.provenance.map((p) => (
          <p key={p.source_id}>
            <SourceLink href={p.source_url}>{p.source_name} ↗</SourceLink> · retrieved {formatDateTimePT(p.retrieved_at)}
          </p>
        ))}
        <p>
          2015–16 crab closures: <SourceLink href={FISHERIES_EVENTS[0].source.url}>{FISHERIES_EVENTS[0].source.label} ↗</SourceLink>
        </p>
      </div>
      <details className="text-[13px] text-ink-3">
        <summary className="cursor-pointer">Citation and licence</summary>
        <p className="mt-1 leading-relaxed">
          {foss?.citation} {foss?.license}
        </p>
      </details>
    </section>
  );
}
