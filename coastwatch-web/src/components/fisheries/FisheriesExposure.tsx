"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { FisheriesDataset } from "@/generated/fisheries";
import type { OfficialDataset } from "@/generated/official";
import type { SourceStatus } from "@/generated/schema";
import { FISHERIES_COPY } from "@/content/copy";
import { OfficialStrip } from "@/components/bloom/OfficialStrip";
import { FreshDot } from "@/components/bloom/FreshDot";
import { classifySource } from "@/lib/freshness";
import { useNow } from "@/lib/useNow";
import { formatValue, groupShort, groupsInTier, pct, selectedTotal, shareOfState, span, stateTotal, unitLabel, viewQuery, type FisheriesView, type TierSet, type Unit } from "@/lib/fisheries";
import { FisheriesMethods } from "./FisheriesMethods";
import { Legend, LandingsChart } from "./LandingsChart";
import dynamic from "next/dynamic";
import { SpeciesTable } from "./SpeciesTable";
import { YearPanel } from "./YearPanel";

// the reference map (and its 14 kB of CDFW points) loads only when its section nears the screen
const PortAreaMap = dynamic(() => import("./PortAreaMap").then((m) => m.PortAreaMap), { ssr: false, loading: () => <MapSpace /> });
function MapSpace() {
  return <div className="aspect-[329/251] w-full max-w-[329px] rounded-md bg-surface-3/60" aria-hidden />;
}
function NearScreen({ children }: { children: React.ReactNode }) {
  const [near, setNear] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el || near) return;
    const io = new IntersectionObserver(([e]) => e.isIntersecting && setNear(true), { rootMargin: "600px 0px" });
    io.observe(el);
    return () => io.disconnect();
  }, [near]);
  return <div ref={ref}>{near ? children : <MapSpace />}</div>;
}

function Seg<T extends string>({ label, value, options, onChange, testid }: { label: string; value: T; options: { value: T; label: string }[]; onChange: (v: T) => void; testid: string }) {
  return (
    <div role="radiogroup" aria-label={label} className="flex rounded-[10px] bg-surface-3 p-[3px]">
      {options.map((o) => {
        const on = o.value === value;
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={on}
            onClick={() => onChange(o.value)}
            data-testid={`${testid}-${o.value}`}
            className={`flex-1 whitespace-nowrap rounded-[8px] px-3 py-1 text-[13.5px] sm:py-1.5 font-medium transition-colors duration-150 ${on ? "bg-surface text-ink shadow-[0_1px_2px_rgba(13,27,42,0.12)] ring-1 ring-hairline-strong" : "text-ink-2 hover:text-ink"}`}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

/**
 * Fisheries & Economic Exposure (M7): what California's toxin-affected fisheries have landed, told as
 * one data story. Official notices first; the definition as the lede; three figures for the selected
 * year; one stacked annual chart with the year beside it; one table by species group that opens in
 * place; what the figures can and cannot establish; where landings are reported; how the numbers are
 * made. Everything is statewide NOAA data: history, never a forecast or a loss estimate.
 */
export function FisheriesExposure({ ds, status, official, initial }: { ds: FisheriesDataset; status: SourceStatus | null; official: OfficialDataset | null; initial: FisheriesView }) {
  const now = useNow();
  const latest = ds.years[ds.years.length - 1];
  const base = ds.deflator.base_year;
  const [unit, setUnit] = useState<Unit>(initial.unit);
  const [tiers, setTiers] = useState<TierSet>(initial.tiers);
  const [focus, setFocus] = useState<string | null>(initial.group);
  const [year, setYear] = useState<number>(initial.year);
  const [hover, setHover] = useState<number | null>(null);
  const [narrow, setNarrow] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(max-width: 720px)");
    const on = () => setNarrow(mq.matches);
    on();
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);

  // the view lives in the link
  useEffect(() => {
    const q = viewQuery({ unit, tiers, group: focus, year }, latest);
    const next = `${window.location.pathname}${q ? `?${q}` : ""}`;
    if (next !== `${window.location.pathname}${window.location.search}`) window.history.replaceState(null, "", next);
  }, [unit, tiers, focus, year, latest]);

  const groups = useMemo(() => groupsInTier(ds.groups, tiers), [ds.groups, tiers]);
  const inSet = useMemo(() => new Set(groups.map((g) => g.id)), [groups]);
  const focusGroup = (id: string | null) => {
    // a Tier 2 group can only be focused while Tier 2 is in the chart
    if (id && !inSet.has(id)) setTiers("12");
    setFocus(id);
  };
  const changeTiers = (t: TierSet) => {
    setTiers(t);
    if (t === "1" && focus && ds.groups.find((g) => g.id === focus)?.tier === 2) setFocus(null);
  };

  const total = selectedTotal(groups, year, unit);
  const state = stateTotal(ds, year, unit);
  const sp = span(ds.years, (y) => selectedTotal(groups, y, unit));
  const fresh = status && now ? classifySource(status, now) : null;
  const setName = tiers === "1" ? "Tier 1" : "Tier 1 and 2";
  const unitShort = unit === "real" ? `${base} dollars` : unit === "nominal" ? "nominal dollars" : "pounds";
  const recordIds = ds.groups.flatMap((g) => g.official_record_ids ?? []);
  const unavailable = ds.years_requested_unavailable ?? [];
  const meanText = `a year on average, ${ds.years[0]}–${latest}; lowest ${sp.low?.year} (${formatValue(sp.low?.v ?? null, unit)}), highest ${sp.high?.year} (${formatValue(sp.high?.v ?? null, unit)})`;
  const shareText = `of all California commercial landings in ${year} (${formatValue(state, unit)}); a lower bound, since confidential landings and generic categories belong to no group`;
  const focusG = focus ? ds.groups.find((g) => g.id === focus) : undefined;
  const focusLabel = focusG?.label ?? null;
  const focusShort = focusG ? groupShort(focusG) : null;

  return (
    <div className="mx-auto w-full max-w-[1280px] space-y-7 px-4 pb-16 pt-4 sm:space-y-9 sm:px-8" data-testid="fisheries-page">
      <OfficialStrip official={official} error={null} recordIds={recordIds} heading={narrow ? "{n} notices name these species" : "Official notices that name these species"} emptyText="No active notice in CoastWatch’s list names these species." compact testid="fisheries-official" />

      {/* the story's opening, in the published terminology, beside the three figures (no cards) */}
      <div className="grid gap-x-12 gap-y-6 xl:grid-cols-[minmax(0,1fr)_400px]">
        <header className="space-y-2.5 sm:space-y-3">
          <p className="text-[12px] font-semibold uppercase tracking-[0.08em] text-history">
            Historical context · Statewide<span className="hidden sm:inline"> · NOAA Fisheries landings</span>
          </p>
          <h1 className="text-balance font-display text-[29px] font-medium leading-[1.08] text-ink sm:text-[44px]">{FISHERIES_COPY.title}</h1>
          <p className="max-w-[70ch] text-[15.5px] leading-normal text-ink-2 sm:text-[16px] sm:leading-relaxed" data-testid="exposure-definition">
            {ds.terminology} {FISHERIES_COPY.atStake.split(". ")[0]}.
          </p>
          <p className="flex flex-wrap items-center gap-x-4 gap-y-0.5 text-[13px] text-ink-3 sm:text-[13px]">
            {fresh && <FreshDot f={fresh} />}
            <span data-testid="data-through">
              Annual data {ds.years[0]}–{latest}
              <span className="max-sm:hidden">{unavailable.length ? `; ${unavailable.join(", ")} not yet published by NOAA` : ""}</span>
            </span>
            <span data-testid="science-review-pending">
              Species tiers await independent scientific review
            </span>
          </p>
        </header>

        <section
          aria-label={`${setName} species in ${year}`}
          className="grid grid-cols-3 gap-x-4 gap-y-4 border-y border-hairline py-3 sm:gap-x-8 sm:py-4 xl:grid-cols-1 xl:gap-y-0 xl:divide-y xl:divide-hairline xl:border-y-0 xl:border-l xl:py-0 xl:pl-8"
          data-testid="fisheries-figures"
        >
          <Figure value={formatValue(total, unit)} short={`in ${year}`} testid="figure-year">
            landed by {setName} species in {year}, in {unitShort}
          </Figure>
          <Figure value={formatValue(sp.mean, unit)} short="a year on average" testid="figure-mean">
            {meanText}
          </Figure>
          <Figure value={pct(shareOfState(total, state))} short="of the state total" testid="figure-share">
            {shareText}
          </Figure>
          {/* phones: the short captions above; the full sentences one tap away */}
          <details className="col-span-3 text-[13px] text-ink-2 sm:hidden" data-testid="figures-detail">
            <summary className="cursor-pointer font-medium text-ink">
              {setName} species, {unitShort}: what these figures include
            </summary>
            <ul className="mt-1.5 space-y-1">
              <li>
                {formatValue(total, unit)} landed in {year}.
              </li>
              <li>{formatValue(sp.mean, unit)} {meanText}.</li>
              <li>{pct(shareOfState(total, state))} {shareText}.</li>
            </ul>
          </details>
        </section>
      </div>

      {/* phones and tablets: the page's parts, one tap away */}
      <nav aria-label="On this page" className="-mx-4 -my-3 overflow-x-auto px-4 sm:-my-4 lg:hidden" data-testid="fisheries-jump">
        <ul className="flex gap-2 whitespace-nowrap text-[14px]">
          {[
            ["chart-h", "By year"],
            ["species-h", "By species"],
            ["limits-h", "Can and cannot"],
            ["geo-h", "Ports"],
            ["methods-h", "Methods"],
          ].map(([id, label]) => (
            <li key={id}>
              <a href={`#${id}`} className="inline-flex min-h-[36px] items-center rounded-full border border-hairline-strong bg-surface px-3.5 font-medium text-ink hover:border-accent">
                {label}
              </a>
            </li>
          ))}
        </ul>
      </nav>

      {/* the primary figure */}
      <section aria-labelledby="chart-h" className="scroll-mt-24 space-y-4">
        <div className="flex flex-wrap items-end justify-between gap-x-8 gap-y-3">
          <div>
            <h2 id="chart-h" className="font-display text-[23px] sm:text-[26px] font-medium leading-tight text-ink">
              {focusLabel ? `${focusShort}: ${unit === "pounds" ? "pounds landed" : "landed value"} by year` : `${unit === "pounds" ? "Pounds landed" : "Landed value"}, by year and species group`}
            </h2>
            <p className="text-[13.5px] text-ink-2" data-testid="chart-unit">
              {focusLabel ? `${focusLabel} alone, on its own scale` : `${setName} species`}, statewide, {unitLabel(unit, base)}. Past landings<span className="max-sm:hidden"> only</span>, not a forecast.
            </p>
          </div>
          <div className="grid w-full gap-2 sm:flex sm:w-auto sm:flex-wrap sm:items-center" data-testid="fisheries-filters">
            <Seg label="Species tiers" value={tiers} onChange={changeTiers} testid="tier" options={[{ value: "1", label: "Tier 1" }, { value: "12", label: "Tiers 1 + 2" }]} />
            <Seg
              label="Measure"
              value={unit}
              onChange={setUnit}
              testid="unit"
              options={[
                { value: "real", label: `${base} $` },
                { value: "nominal", label: "Nominal $" },
                { value: "pounds", label: "Pounds" },
              ]}
            />
          </div>
        </div>
        <Legend groups={groups} focus={focus} onFocus={focusGroup} unit={unit} />
        <div className="grid grid-cols-[minmax(0,1fr)] gap-x-10 gap-y-6 lg:grid-cols-[minmax(0,1fr)_300px]">
          <div className="min-w-0">
            <LandingsChart
              ds={ds}
              groups={groups}
              unit={unit}
              focus={focus}
              year={year}
              hover={hover}
              onHover={setHover}
              onYear={setYear}
              label={`${unit === "pounds" ? "Pounds landed" : "Landed value"} of ${focusLabel ?? `${setName} species`} by year, ${unitLabel(unit, base)}`}
            />
          </div>
          <div className="lg:border-l lg:border-hairline lg:pl-8">
            <YearPanel ds={ds} groups={groups} unit={unit} year={hover ?? year} focus={focus} previewing={hover != null && hover !== year} />
          </div>
        </div>
        <ChartNotes narrow={narrow}>
          <p>
            <b className="font-semibold text-ink">Reading this chart.</b> Each bar is one calendar year, January to December. Crab seasons run from late autumn into the next year, so one season&apos;s landings fall into two bars. The brackets mark when the 2015–16 domoic acid closures happened (select 2015 or 2016 for the record); they do not say how much of any change between years the closures explain.
          </p>
          {unit === "pounds" ? (
            <p>Pounds are weight, not value or effort: a pound of anchovy and a pound of crab earn very different amounts. Bivalve pounds are meat weight (without shell), unlike the other groups.</p>
          ) : unit === "real" ? (
            <p>
              {base} dollars rescale each year&apos;s revenue by consumer prices (CPI-U), so years can be compared in purchasing power; they are not adjusted for seafood prices or costs. In {base} itself the two are equal.
            </p>
          ) : (
            <p>Nominal dollars are as reported each year, without adjusting for inflation, so earlier years look smaller than they were in purchasing power.</p>
          )}
        </ChartNotes>
      </section>

      {/* the inspector */}
      <section aria-labelledby="species-h" className="scroll-mt-24 grid grid-cols-[minmax(0,1fr)] gap-x-10 gap-y-4 lg:grid-cols-[minmax(0,1fr)_300px]">
        <div className="min-w-0 space-y-3">
          <h2 id="species-h" className="font-display text-[26px] font-medium leading-tight text-ink">
            By species group <span className="text-[17px] text-ink-3">{year}, {unitShort}</span>
          </h2>
          <SpeciesTable ds={ds} unit={unit} year={year} inSet={inSet} open={focus} onOpen={focusGroup} official={official} />
        </div>
        <div className="space-y-3 text-[13.5px] leading-relaxed text-ink-2 lg:border-l lg:border-hairline lg:pl-8 lg:pt-12" data-testid="tier-definitions">
          <p>
            <b className="font-semibold text-ink">Tier 1</b>: fisheries with commercial closures or take restrictions for marine toxins, now or in the past.
          </p>
          <p>
            <b className="font-semibold text-ink">Tier 2</b>: species under consumption advisories, sport-harvest quarantines or toxin monitoring.
            {tiers === "1" ? " Dimmed while the chart shows Tier 1; choosing one adds Tier 2." : ""}
          </p>
          <p className="text-ink-3">Tiers are CoastWatch&apos;s grouping and await independent scientific review. Choose a group to see its revenue and weight side by side, and the notices it references.</p>
        </div>
      </section>

      {/* what the figures can and cannot establish */}
      <section aria-labelledby="limits-h" className="scroll-mt-24 grid gap-x-10 gap-y-4 border-t border-hairline pt-8 lg:grid-cols-[minmax(0,0.8fr)_minmax(0,1fr)_minmax(0,1.2fr)]" data-testid="fisheries-limits">
        <h2 id="limits-h" className="font-display text-[26px] font-medium leading-tight text-ink">
          What these figures can and cannot tell you
        </h2>
        {narrow ? (
          <details className="group order-last border-t border-hairline pt-3" data-testid="limits-can">
            <summary className="cursor-pointer text-[13px] font-semibold uppercase tracking-[0.06em] text-ink-3">They show</summary>
            <ul className="mt-1.5 space-y-2 text-[14.5px] leading-relaxed text-ink-2">
              {FISHERIES_COPY.can.map((t) => (
                <li key={t}>{t}</li>
              ))}
            </ul>
          </details>
        ) : (
          <div data-testid="limits-can">
            <h3 className="mb-1.5 text-[13px] font-semibold uppercase tracking-[0.06em] text-ink-3">They show</h3>
            <ul className="space-y-2 text-[14.5px] leading-relaxed text-ink-2">
              {FISHERIES_COPY.can.map((t) => (
                <li key={t}>{t}</li>
              ))}
            </ul>
          </div>
        )}
        <div>
          <h3 className="mb-1.5 text-[13px] font-semibold uppercase tracking-[0.06em] text-ink-3">They do not show</h3>
          <ul className="space-y-2 text-[14.5px] leading-relaxed text-ink-2">
            {FISHERIES_COPY.cannot.map((t) => (
              <li key={t}>{t}</li>
            ))}
          </ul>
        </div>
      </section>

      {/* geography, only as far as the data allow */}
      <section aria-labelledby="geo-h" className="scroll-mt-24 grid gap-x-12 gap-y-6 border-t border-hairline pt-8 md:grid-cols-[minmax(0,1fr)_minmax(0,420px)]">
        <div className="max-w-[60ch] space-y-3 text-[14.5px] leading-relaxed text-ink-2">
          <h2 id="geo-h" className="font-display text-[26px] font-medium leading-tight text-ink">
            Where landings are reported
          </h2>
          <p>Commercial catches are reported to CDFW at more than a hundred ports, which CDFW groups into port areas from Eureka to San Diego. The figures on this page add all of them together.</p>
          <p data-testid="port-level-unavailable">
            <b className="font-semibold text-ink">{FISHERIES_COPY.portUnavailable}</b>, so CoastWatch does not split the statewide totals among ports or areas: any split would be an estimate, and confidential values cannot be redistributed.
          </p>
          <details className="text-[13.5px]" data-testid="port-level-why">
            <summary className="cursor-pointer font-medium text-ink">Why port-level values are missing</summary>
            <ul className="mt-2 list-disc space-y-1 pl-5">
              {ds.port_level.reasons.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
            <p className="mt-2 text-ink-3">Sources considered: {ds.port_level.adapters.map((a) => `${a.id} (${a.status})`).join(", ")}.</p>
          </details>
        </div>
        <NearScreen>
          <PortAreaMap />
        </NearScreen>
      </section>

      <div className="scroll-mt-24 border-t border-hairline pt-8" id="methods">
        <FisheriesMethods ds={ds} unit={unit} compact={narrow} />
      </div>
    </div>
  );
}

/** How to read the chart: open beside the chart on wide screens, one tap away on phones. */
function ChartNotes({ narrow, children }: { narrow: boolean; children: React.ReactNode }) {
  const body = <div className="max-w-[92ch] space-y-1.5 text-[13.5px] leading-relaxed text-ink-2">{children}</div>;
  return narrow ? (
    <details className="border-y border-hairline py-2.5" data-testid="chart-notes">
      <summary className="cursor-pointer text-[14px] font-medium text-ink">How to read this chart</summary>
      <div className="mt-2">{body}</div>
    </details>
  ) : (
    <div data-testid="chart-notes">{body}</div>
  );
}

function Figure({ value, short, children, testid }: { value: string; short: string; children: React.ReactNode; testid: string }) {
  return (
    <div className="min-w-0 items-baseline gap-4 xl:flex xl:py-3" data-testid={testid}>
      <p className="shrink-0 font-display text-[25px] font-medium leading-none tracking-tight text-ink tabular sm:text-[34px] xl:w-[8rem] xl:text-[36px]">{value}</p>
      <p className="mt-1 text-[13px] leading-snug text-ink-2 sm:hidden">{short}</p>
      <p className="mt-2 hidden text-[13.5px] leading-snug text-ink-2 sm:block xl:mt-0">{children}</p>
    </div>
  );
}
