"use client";

import Link from "next/link";
import { useState } from "react";
import type { FisheriesDataset } from "@/generated/fisheries";
import type { OfficialDataset } from "@/generated/official";
import type { SourceStatus } from "@/generated/schema";
import { FISHERIES_COPY, OFFICIAL_STATUS } from "@/content/copy";
import { BarChart, money } from "@/components/charts/BarChart";
import { FreshnessBadge, ProductClassBadge } from "@/components/ui/Badges";
import { Notice, Segmented, SourceLink } from "@/components/ui/Primitives";
import { classifySource } from "@/lib/freshness";
import { formatDateTimePT } from "@/lib/time";
import { useNow } from "@/lib/useNow";
import { fisheriesValue, selectedTotal, type Dollars } from "@/lib/fisheries";

type TierSel = "1" | "12";
const BAR = "#8fb3e8";

export function FisheriesExposure({ ds, status, official }: { ds: FisheriesDataset; status: SourceStatus | null; official: OfficialDataset | null }) {
  const now = useNow();
  const [dollars, setDollars] = useState<Dollars>("real");
  const [tier, setTier] = useState<TierSel>("1");
  const base = ds.deflator.base_year;
  const unit = dollars === "real" ? `${base} dollars (CPI-U adjusted)` : "nominal dollars (as reported)";
  const groups = ds.groups.filter((g) => (tier === "1" ? g.tier === 1 : true));
  const latest = ds.years[ds.years.length - 1];
  const latestTotal = selectedTotal(groups, latest, dollars);
  const avg = ds.years.map((y) => selectedTotal(groups, y, dollars)).filter((v): v is number => v != null);
  const mean = avg.length ? avg.reduce((a, b) => a + b, 0) / avg.length : null;
  const state = ds.statewide_total.find((t) => t.year === latest);
  const stateV = state ? fisheriesValue(state, dollars) : null;
  const withheld = ds.withheld.find((w) => w.year === latest);
  const records = new Map((official?.registry.records ?? []).map((r) => [r.id, r]));
  const fresh = status && now ? classifySource(status, now) : null;

  return (
    <div className="mx-auto w-full max-w-[1200px] space-y-4 px-4 py-5" data-testid="fisheries-page">
      <header className="max-w-3xl space-y-1.5">
        <div className="flex flex-wrap items-center gap-2">
          <ProductClassBadge pc="historical_context" />
          <FreshnessBadge f={fresh} compact />
          <span className="text-[11.5px] text-ink-3" data-testid="data-through">
            Annual data through {latest}
          </span>
          <span className="rounded-md border border-dashed border-hairline-strong px-1.5 py-px text-[10px] font-semibold uppercase tracking-wider text-ink-3" data-testid="science-review-pending">
            Scientific review pending
          </span>
        </div>
        <h1 className="text-[22px] font-semibold tracking-tight text-ink">{FISHERIES_COPY.heading}</h1>
        <p className="text-[13.5px] leading-relaxed text-ink-2">{FISHERIES_COPY.lede}</p>
      </header>

      <section className="rounded-lg border border-accent/30 bg-accent/[0.05] px-4 py-3" data-testid="exposure-definition">
        <p className="text-[13px] leading-relaxed text-ink">
          <span className="font-semibold">What “exposure” means here. </span>
          {ds.terminology}
        </p>
      </section>

      <Notice tone="warning" title={FISHERIES_COPY.portUnavailable} testid="port-level-unavailable">
        <ul className="mt-1 list-disc space-y-0.5 pl-4">
          {ds.port_level.reasons.map((r) => (
            <li key={r}>{r}</li>
          ))}
        </ul>
        <details className="mt-1.5 text-[11.5px] text-ink-2">
          <summary className="cursor-pointer">Port-level sources considered</summary>
          <ul className="mt-1 space-y-0.5">
            {ds.port_level.adapters.map((a) => (
              <li key={a.id}>
                <span className="font-mono text-ink-3">{a.id}</span> — <span className="font-medium">{a.status}</span>: {a.note}
              </li>
            ))}
          </ul>
        </details>
      </Notice>

      <div className="flex flex-wrap items-end gap-3" data-testid="fisheries-filters">
        <div className="w-[280px]">
          <p className="mb-1 text-[11px] font-medium uppercase tracking-wider text-ink-3">Species</p>
          <Segmented
            label="Species tiers"
            value={tier}
            onChange={setTier}
            testidPrefix="tier"
            options={[
              { value: "1", label: "Tier 1", sub: "toxin closures" },
              { value: "12", label: "Tiers 1 + 2", sub: "+ advisories" },
            ]}
          />
        </div>
        <div className="w-[280px]">
          <p className="mb-1 text-[11px] font-medium uppercase tracking-wider text-ink-3">Dollars</p>
          <Segmented
            label="Dollar basis"
            value={dollars}
            onChange={setDollars}
            testidPrefix="dollars"
            options={[
              { value: "real", label: `Real (${base} $)`, sub: "inflation-adjusted" },
              { value: "nominal", label: "Nominal", sub: "as reported" },
            ]}
          />
        </div>
        <p className="pb-1 text-[11.5px] text-ink-3">
          Statewide, {ds.years[0]}–{latest}. {(ds.years_requested_unavailable ?? []).length > 0 && `${(ds.years_requested_unavailable ?? []).join(", ")} not yet published by NOAA.`}
        </p>
      </div>

      <section className="grid gap-3 sm:grid-cols-3" aria-label="Summary" data-testid="fisheries-tiles">
        <Tile label={`Historical exposure, ${latest}`} value={latestTotal != null ? money(latestTotal) : "no value"} sub={`${groups.length} species groups · ${unit}`} testid="tile-latest" />
        <Tile label={`Average per year, ${ds.years[0]}–${latest}`} value={mean != null ? money(mean) : "no value"} sub={unit} testid="tile-mean" />
        <Tile
          label={`Share of all California commercial landings, ${latest}`}
          value={latestTotal != null && stateV ? `${Math.round((latestTotal / stateV) * 100)}%` : "—"}
          sub={`of ${stateV != null ? money(stateV) : "—"} statewide (NOAA's state total, which includes ${withheld?.dollars_nominal != null ? money(withheld.dollars_nominal) : "the"} withheld as confidential)`}
          testid="tile-share"
        />
      </section>

      <section className="space-y-2 rounded-lg border border-hairline bg-surface p-4" aria-labelledby="total-h">
        <h2 id="total-h" className="text-[13.5px] font-semibold text-ink">
          Landed value of {tier === "1" ? "Tier 1" : "Tier 1 and 2"} species, by year
        </h2>
        <p className="text-[11.5px] text-ink-3">{unit}. Past landings only — not a forecast.</p>
        <BarChart
          testid="chart-total"
          label={`Landed value of selected species by year, ${unit}`}
          color={BAR}
          format={money}
          height={190}
          bars={ds.years.map((y) => ({ key: String(y), label: String(y), value: selectedTotal(groups, y, dollars) }))}
        />
      </section>

      <section className="space-y-3" aria-labelledby="groups-h">
        <h2 id="groups-h" className="text-[13.5px] font-semibold text-ink">
          By species group <span className="font-normal text-ink-3">(each chart has its own scale)</span>
        </h2>
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {groups.map((g) => (
            <article key={g.id} className="min-w-0 space-y-2 rounded-lg border border-hairline bg-surface p-3.5" data-testid={`group-${g.id}`}>
              <header>
                <div className="flex items-center justify-between gap-2">
                  <h3 className="text-[13px] font-semibold text-ink">{g.label}</h3>
                  <span className="rounded border border-hairline-strong px-1.5 text-[10px] font-semibold uppercase tracking-wider text-ink-2">Tier {g.tier}</span>
                </div>
                <p className="mt-1 text-[11.5px] leading-snug text-ink-2">{g.tier_basis}</p>
              </header>
              <BarChart
                label={`${g.label}, ${unit}`}
                color={BAR}
                format={money}
                height={120}
                bars={g.annual.map((a) => ({
                  key: String(a.year),
                  label: String(a.year),
                  value: fisheriesValue(a, dollars),
                  note: a.n_rows_without_value ? `${a.n_rows_without_value} source row(s) without a published value` : undefined,
                }))}
              />
              {(g.official_record_ids ?? []).length > 0 && (
                <ul className="space-y-0.5 text-[11.5px]">
                  {(g.official_record_ids ?? []).map((id) => {
                    const r = records.get(id);
                    return (
                      <li key={id} className="text-[#ffcf85]">
                        Official: {r ? `${r.agency} — ${r.title}` : id}
                        {r?.status && r.status !== "active" ? ` (${r.status})` : ""}
                      </li>
                    );
                  })}
                </ul>
              )}
              <p className="text-[10.5px] text-ink-3">NOAA categories: {g.source_names.join(", ") || "none in these years"}</p>
            </article>
          ))}
        </div>
        <p className="text-[11.5px] text-ink-3">
          {FISHERIES_COPY.reviewPending} Official notices take precedence and are listed on the{" "}
          <Link href="/" className="text-accent hover:underline">
            Live Ocean Map
          </Link>
          . {OFFICIAL_STATUS.missingNotOpen}
        </p>
      </section>

      <section className="space-y-2 rounded-lg border border-hairline bg-surface p-4" aria-labelledby="table-h">
        <h2 id="table-h" className="text-[13.5px] font-semibold text-ink">
          Values table <span className="font-normal text-ink-3">({unit})</span>
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[820px] text-right text-[11.5px] tabular" data-testid="fisheries-table">
            <thead className="text-ink-3">
              <tr>
                <th className="py-1 pr-2 text-left font-medium">Group</th>
                {ds.years.map((y) => (
                  <th key={y} className="px-1.5 py-1 font-medium">
                    {y}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ds.groups.map((g) => (
                <tr key={g.id} className="border-t border-hairline">
                  <td className="py-1 pr-2 text-left text-ink">
                    {g.label} <span className="text-ink-3">· T{g.tier}</span>
                  </td>
                  {ds.years.map((y) => {
                    const a = g.annual.find((x) => x.year === y);
                    const v = a ? fisheriesValue(a, dollars) : null;
                    return (
                      <td key={y} className="px-1.5 py-1 text-ink-2">
                        {v == null ? "no value" : money(v)}
                      </td>
                    );
                  })}
                </tr>
              ))}
              <tr className="border-t-2 border-hairline-strong">
                <td className="py-1 pr-2 text-left font-medium text-ink">All CA commercial landings (NOAA state total)</td>
                {ds.years.map((y) => {
                  const t = ds.statewide_total.find((x) => x.year === y);
                  const v = t ? fisheriesValue(t, dollars) : null;
                  return (
                    <td key={y} className="px-1.5 py-1 text-ink">
                      {v == null ? "no value" : money(v)}
                    </td>
                  );
                })}
              </tr>
              <tr className="border-t border-hairline" data-testid="withheld-row">
                <td className="py-1 pr-2 text-left text-ink-2">Withheld for confidentiality (not attributed)</td>
                {ds.years.map((y) => {
                  const w = ds.withheld.find((x) => x.year === y);
                  const v = w ? (dollars === "real" ? w.dollars_real : w.dollars_nominal) : null;
                  return (
                    <td key={y} className="px-1.5 py-1 text-ink-3">
                      {v == null ? "—" : money(v)}
                    </td>
                  );
                })}
              </tr>
            </tbody>
          </table>
        </div>
        <p className="text-[11px] text-ink-3">
          Confidential landings are aggregated by NOAA into one withheld value per year. They are part of the statewide total, as in NOAA&apos;s published state totals, but CoastWatch never assigns them to a species or group. Rows without a published value are shown as “no value”, never as $0. Species groups leave out generic categories (for example unspecified crabs), so they are lower bounds.
        </p>
        {ds.excluded_rows.length > 0 && (
          <details className="text-[11.5px] text-ink-2" data-testid="excluded-rows">
            <summary className="cursor-pointer">
              {ds.excluded_rows.length} duplicate source rows counted once
            </summary>
            <ul className="mt-1 space-y-0.5">
              {ds.excluded_rows.map((e) => (
                <li key={`${e.year}-${e.source_name}`}>
                  {e.year}: “{e.source_name}” duplicates “{e.duplicate_of}” ({e.dollars_nominal != null ? money(e.dollars_nominal) : "no value"} nominal). {e.reason}
                </li>
              ))}
            </ul>
          </details>
        )}
      </section>

      <section className="grid gap-3 md:grid-cols-2">
        <div className="space-y-1.5 rounded-lg border border-hairline bg-surface p-4 text-[12px] text-ink-2" data-testid="deflator">
          <h2 className="text-[13.5px] font-semibold text-ink">Inflation adjustment</h2>
          <p>{ds.deflator.title}.</p>
          <p>{ds.deflator.method}</p>
          {(ds.deflator.notes ?? []).map((n) => (
            <p key={n} className="text-warning">
              {n}
            </p>
          ))}
          <p className="tabular text-[11px] text-ink-3">
            Annual index: {Object.entries(ds.deflator.annual_index).map(([y, v]) => `${y} ${v}`).join(" · ")}
          </p>
          <SourceLink href={ds.deflator.source_url}>BLS series {ds.deflator.series_id} ↗</SourceLink>
        </div>
        <div className="space-y-1.5 rounded-lg border border-hairline bg-surface p-4 text-[12px] text-ink-2">
          <h2 className="text-[13.5px] font-semibold text-ink">Methods, limits and sources</h2>
          <ul className="list-disc space-y-1 pl-4" data-testid="fisheries-caveats">
            {ds.caveats.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
          <details>
            <summary className="cursor-pointer font-medium text-ink">Processing steps</summary>
            <dl className="mt-1 space-y-1 text-[11.5px]">
              {Object.entries(ds.method).map(([k, v]) => (
                <div key={k}>
                  <dt className="inline capitalize text-ink-3">{k}: </dt>
                  <dd className="inline">{v}</dd>
                </div>
              ))}
            </dl>
          </details>
          {ds.provenance.map((p) => (
            <p key={p.source_id} className="text-[11px] text-ink-3">
              <SourceLink href={p.source_url}>{p.source_name} ↗</SourceLink> · retrieved {formatDateTimePT(p.retrieved_at)}
              {p.citation ? ` · ${p.citation}` : ""}
            </p>
          ))}
        </div>
      </section>
    </div>
  );
}

function Tile({ label, value, sub, testid }: { label: string; value: string; sub: string; testid: string }) {
  return (
    <div className="rounded-lg border border-hairline bg-surface p-3.5" data-testid={testid}>
      <p className="text-[11.5px] text-ink-3">{label}</p>
      <p className="mt-0.5 text-[24px] font-semibold tracking-tight text-ink tabular">{value}</p>
      <p className="text-[11px] text-ink-3">{sub}</p>
    </div>
  );
}
