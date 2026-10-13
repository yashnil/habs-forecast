"use client";

import { Fragment, useMemo, useState } from "react";
import type { FisheriesDataset, SpeciesGroup } from "@/generated/fisheries";
import type { OfficialDataset } from "@/generated/official";
import { useOfficialDrawer } from "@/components/shell/OfficialShell";
import { ACTION_LABEL, disputeOf } from "@/lib/official";
import { formatValue, groupColor, groupShort, tierInk, groupValue, isDollars, pct, shareOfState, span, stateTotal, unitLabel, type Unit } from "@/lib/fisheries";

type SortKey = "name" | "year" | "mean";

/**
 * One table by species group (M7 §4.5): the selected year, its share of the statewide total, the
 * 10-year average and a 10-year bar strip on the group's own scale. Groups outside the selected tiers
 * are dimmed, not hidden. Each row opens its detail in place: why the group is here, the official
 * notices it references, revenue and weight side by side on separate scales, and NOAA's categories.
 */
export function SpeciesTable({
  ds,
  unit,
  year,
  inSet,
  open,
  onOpen,
  official,
}: {
  ds: FisheriesDataset;
  unit: Unit;
  year: number;
  inSet: Set<string>;
  open: string | null;
  onOpen: (id: string | null) => void;
  official: OfficialDataset | null;
}) {
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: "year", dir: -1 });
  const stats = useMemo(() => new Map(ds.groups.map((g) => [g.id, span(ds.years, (y) => groupValue(g, y, unit))])), [ds, unit]);
  const state = stateTotal(ds, year, unit);
  const rows = useMemo(() => {
    const val = (g: SpeciesGroup) => (sort.key === "year" ? (groupValue(g, year, unit) ?? -1) : (stats.get(g.id)?.mean ?? -1));
    return [...ds.groups].sort((a, b) => (sort.key === "name" ? sort.dir * groupShort(a).localeCompare(groupShort(b)) : sort.dir * (val(a) - val(b))));
  }, [ds.groups, sort, year, unit, stats]);

  const head = (key: SortKey, label: string, cls = "") => {
    const on = sort.key === key;
    return (
      <th scope="col" aria-sort={on ? (sort.dir === 1 ? "ascending" : "descending") : "none"} className={`py-2 font-medium ${cls}`}>
        <button
          type="button"
          onClick={() => setSort(on ? { key, dir: sort.dir === 1 ? -1 : 1 } : { key, dir: key === "name" ? 1 : -1 })}
          className={`inline-flex items-center gap-1 hover:text-ink ${on ? "text-ink" : ""}`}
          data-testid={`sort-${key}`}
        >
          {label}
          <span aria-hidden className="w-2 text-[13px]">
            {on ? (sort.dir === 1 ? "▲" : "▼") : ""}
          </span>
        </button>
      </th>
    );
  };

  return (
    <table className="w-full border-collapse text-[14px] tabular" data-testid="species-table" data-sort={`${sort.key}-${sort.dir}`}>
      <caption className="sr-only">
        Species groups: landed {unit === "pounds" ? "weight" : "value"} in {year}, share of all California commercial landings, and the 10-year record, in {unitLabel(unit, ds.deflator.base_year)}. Choose a group to see its detail.
      </caption>
      <thead className="text-left text-[13px] text-ink-3">
        <tr className="border-b border-hairline-strong">
          {head("name", "Species group", "pr-3")}
          {head("year", String(year), "pr-3 text-right [&>button]:flex-row-reverse")}
          <th scope="col" className="py-2 pr-3 text-right font-medium">
            Share<span className="hidden sm:inline"> of state</span>
          </th>
          {head("mean", "10-yr avg", "hidden pr-3 text-right md:table-cell [&>button]:flex-row-reverse")}
          <th scope="col" className="hidden py-2 font-medium sm:table-cell">
            {ds.years[0]}–{ds.years[ds.years.length - 1]} <span className="font-normal">(own scale)</span>
          </th>
        </tr>
      </thead>
      <tbody>
        {rows.map((g) => {
          const v = groupValue(g, year, unit);
          const isOpen = open === g.id;
          const dim = !inSet.has(g.id);
          const st = stats.get(g.id)!;
          return (
            <Fragment key={g.id}>
              <tr className={`border-b border-hairline ${isOpen ? "bg-surface-2" : ""} ${dim ? "text-ink-3" : "text-ink"}`} data-testid={`species-row-${g.id}`} data-dim={dim}>
                <th scope="row" className="py-0 pr-3 text-left font-normal">
                  <button
                    type="button"
                    aria-expanded={isOpen}
                    aria-controls={`detail-${g.id}`}
                    onClick={() => onOpen(isOpen ? null : g.id)}
                    className="flex w-full items-center gap-2.5 py-2.5 text-left"
                    data-testid={`species-open-${g.id}`}
                  >
                    <span aria-hidden className={`w-2 shrink-0 text-[15px] leading-none text-ink-3 transition-transform duration-150 ${isOpen ? "rotate-90" : ""}`}>
                      ›
                    </span>
                    <span className="h-3 w-3 shrink-0 rounded-[3px]" style={{ background: groupColor(g.id), opacity: dim ? 0.5 : 1 }} aria-hidden />
                    <span className="min-w-0">
                      <span className={`block font-medium ${dim ? "" : "text-ink"}`}>{groupShort(g)}</span>
                      <span className="block text-[13px] text-ink-3">
                        Tier {g.tier} · {g.tier === 1 ? "toxin closures" : "advisories"}
                        {dim ? " · not in chart" : ""}
                      </span>
                    </span>
                  </button>
                </th>
                <td className="py-2.5 pr-3 text-right" data-testid={`species-value-${g.id}`}>
                  {formatValue(v, unit)}
                </td>
                <td className="py-2.5 pr-3 text-right text-ink-2">{pct(shareOfState(v, state))}</td>
                <td className="hidden py-2.5 pr-3 text-right text-ink-2 md:table-cell">{formatValue(st.mean, unit)}</td>
                <td className="hidden py-2.5 sm:table-cell">
                  <Strip g={g} years={ds.years} unit={unit} year={year} dim={dim} />
                </td>
              </tr>
              {isOpen && (
                <tr className="border-b border-hairline bg-surface-2" id={`detail-${g.id}`}>
                  <td colSpan={5} className="px-1 pb-5 pt-1 sm:px-3">
                    <GroupDetail ds={ds} g={g} unit={unit} year={year} official={official} />
                  </td>
                </tr>
              )}
            </Fragment>
          );
        })}
      </tbody>
    </table>
  );
}

/** Ten small bars on the group's own scale, the selected year darker; values are in the row label. */
function Strip({ g, years, unit, year, dim }: { g: SpeciesGroup; years: number[]; unit: Unit; year: number; dim: boolean }) {
  const vals = years.map((y) => groupValue(g, y, unit));
  const max = Math.max(0, ...vals.map((v) => v ?? 0));
  const W = 150;
  const H = 26;
  const bw = W / years.length;
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} aria-hidden className="block" data-testid={`strip-${g.id}`}>
      <line x1={0} x2={W} y1={H - 0.5} y2={H - 0.5} stroke="var(--cw-hairline-strong)" />
      {vals.map((v, i) =>
        v == null ? (
          <text key={i} x={i * bw + bw / 2} y={H - 3} textAnchor="middle" className="fill-[var(--cw-ink-3)] text-[9px]">
            ·
          </text>
        ) : (
          <rect key={i} x={i * bw + 1.5} width={bw - 3} y={H - 1 - (max ? (v / max) * (H - 3) : 0)} height={max ? (v / max) * (H - 3) : 0} fill={tierInk(g.tier)} opacity={dim ? 0.3 : years[i] === year ? 1 : 0.5} />
        ),
      )}
    </svg>
  );
}

function GroupDetail({ ds, g, unit, year, official }: { ds: FisheriesDataset; g: SpeciesGroup; unit: Unit; year: number; official: OfficialDataset | null }) {
  const { openDrawer, verification } = useOfficialDrawer();
  const records = new Map((official?.registry.records ?? []).map((r) => [r.id, r]));
  const notices = (g.official_record_ids ?? []).map((id) => records.get(id)).filter((r) => r != null);
  const dollars = isDollars(unit) ? unit : "real";
  const missingYears = g.annual.filter((a) => (a.n_rows_without_value ?? 0) > 0);
  return (
    <div className="grid gap-x-10 gap-y-5 pt-2 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]" data-testid={`species-detail-${g.id}`}>
      <div className="space-y-3 text-[14px] leading-relaxed text-ink-2">
        <p className="font-display text-[20px] leading-snug text-ink">{g.label}</p>
        <p>{g.tier_basis}</p>
        {notices.length > 0 ? (
          <div className="rounded-md border border-official-line bg-official-bg px-3 py-2 text-[13.5px]" data-testid={`species-notices-${g.id}`}>
            <p className="font-semibold text-official-ink">
              Official notices referenced ({verification?.state === "verified" ? "verified" : "not verified by a person"})
            </p>
            <ul className="mt-1 space-y-0.5 text-ink">
              {notices.map((r) => (
                <li key={r.id}>
                  <span className="mr-1.5 font-mono text-[12px] font-semibold text-official">{r.agency.toUpperCase()}</span>
                  {ACTION_LABEL[r.action]}: {r.title}
                  {disputeOf(official, r.id) && <span className="ml-1 font-medium text-warning"> (reported lifted by the agency; awaiting review)</span>}
                </li>
              ))}
            </ul>
            <button type="button" onClick={(e) => openDrawer(e.currentTarget)} className="mt-1 font-medium text-official-ink underline-offset-2 hover:underline">
              Read the notices ›
            </button>
          </div>
        ) : null}
        <p className="text-[13px] text-ink-3">
          NOAA market categories: {g.source_names.join(", ") || "none in these years"}.
          {missingYears.length > 0 && ` Rows without a published value (left out, not zero): ${missingYears.map((a) => `${a.year} (${a.n_rows_without_value})`).join(", ")}.`}
        </p>
      </div>
      <div className="space-y-3">
        <MiniSeries g={g} years={ds.years} unit={dollars} year={year} title={dollars === "real" ? `Revenue, ${ds.deflator.base_year} dollars` : "Revenue, nominal dollars"} testid={`detail-revenue-${g.id}`} />
        <MiniSeries g={g} years={ds.years} unit="pounds" year={year} title={g.id === "bivalves" ? "Weight landed, pounds of meat (without shell)" : "Weight landed, pounds"} testid={`detail-pounds-${g.id}`} />
        <p className="text-[13px] leading-snug text-ink-3">Revenue and weight are separate measures on separate scales. What a pound earns changes with species, size, market and year, so the two need not move together.</p>
      </div>
    </div>
  );
}

function MiniSeries({ g, years, unit, year, title, testid }: { g: SpeciesGroup; years: number[]; unit: Unit; year: number; title: string; testid: string }) {
  const vals = years.map((y) => groupValue(g, y, unit));
  const s = span(years, (y) => groupValue(g, y, unit));
  const max = s.high?.v ?? 0;
  const H = 46;
  return (
    <figure data-testid={testid}>
      <figcaption className="flex flex-wrap items-baseline justify-between gap-x-3 text-[13px]">
        <span className="font-medium text-ink">{title}</span>
        <span className="text-ink-3">
          {year}: <b className="font-semibold text-ink">{formatValue(groupValue(g, year, unit), unit)}</b>
          {s.high ? ` · peak ${formatValue(s.high.v, unit)} (${s.high.year})` : ""}
        </span>
      </figcaption>
      <div className="mt-1 grid gap-[3px]" style={{ gridTemplateColumns: `repeat(${years.length}, minmax(0, 1fr))` }}>
        {vals.map((v, i) => (
          <div key={years[i]} className="flex flex-col items-stretch" title={`${years[i]}: ${formatValue(v, unit)}`}>
            <div className="relative" style={{ height: H }}>
              {v == null ? (
                <span className="absolute inset-x-0 bottom-0 text-center text-[13px] text-ink-3">n/a</span>
              ) : (
                <div className="absolute inset-x-0 bottom-0 rounded-t-[2px]" style={{ height: max ? Math.max(1, (v / max) * H) : 0, background: tierInk(g.tier), opacity: years[i] === year ? 1 : 0.45 }} />
              )}
            </div>
            <span className={`mt-0.5 text-center text-[13px] tabular ${years[i] === year ? "font-semibold text-ink" : "text-ink-3"}`}>’{String(years[i]).slice(2)}</span>
          </div>
        ))}
      </div>
      <div className="sr-only">
        <table>
        <caption>{title}</caption>
        <tbody>
          {vals.map((v, i) => (
            <tr key={years[i]}>
              <th scope="row">{years[i]}</th>
              <td>{formatValue(v, unit)}</td>
            </tr>
          ))}
        </tbody>
        </table>
      </div>
    </figure>
  );
}
