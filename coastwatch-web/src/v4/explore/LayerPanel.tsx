"use client";

import type { LayerArtifact, Manifest, Palette } from "@/generated/schema";
import { CHARM_VARIABLES, charmRun, satelliteLatest, sourceStatus, type CharmVariable } from "@/lib/layers";
import { KNOTS_PER_MS, currentsHourly } from "@/lib/currents";
import { AGE_COLOURS, SENSOR_COLOURS, displayPalette, gradientCss } from "@/lib/palette";
import { SPEED_CLASSES, ARROW_LENGTHS } from "@/lib/basemap";
import { stampLines } from "@/lib/stamp";
import { COUNTIES } from "@/content/copy";
import type { CountyPrefs } from "@/components/map/Counties";
import { Icon } from "@/v4/ui/Icon";
import { SAT_WORDS, VARIABLE_WORDS, coverageWhen, type LayerChoice, type Resolved } from "./layers";
import { clockLabel, dayLabel, relTime } from "@/v4/ui/time";
import { classifySource, classifyTime } from "@/lib/freshness";

type Family = "none" | "forecast" | "satellite" | "currents";

const FAMILY: Record<Family, { name: string; kind: string; icon: string; tone: string }> = {
  none: { name: "Seafloor and coast", kind: "Reference", icon: "map", tone: "text-n-ink-2" },
  forecast: { name: "NOAA C-HARM forecast", kind: "Model", icon: "cell", tone: "text-n-model" },
  satellite: { name: "Satellite chlorophyll", kind: "Observation", icon: "layers", tone: "text-n-measured" },
  currents: { name: "Surface currents", kind: "Observation", icon: "compass", tone: "text-n-conditions" },
};

function freshnessLine(m: Manifest, f: Family, now: Date | null): string | null {
  if (f === "forecast") {
    const run = charmRun(m);
    return run ? `Run of ${dayLabel(run.issued_date)}${now ? ` · ${relTime(run.issued_date, now)}` : ""}` : "No run published";
  }
  if (f === "satellite") {
    const l = satelliteLatest(m, "olci300") ?? satelliteLatest(m, "viirs750");
    const d = l?.composite?.newest_observed_date ?? l?.time.observed_date;
    return d ? `Newest pixels ${now ? relTime(d, now) : d}` : "No recent clear view";
  }
  if (f === "currents") {
    const h = currentsHourly(m);
    const t = h[h.length - 1]?.time.valid_time;
    return t ? `Newest hour ${now ? relTime(t, now) : t}` : "No radar hour published";
  }
  return null;
}

function familyOf(c: LayerChoice): Family {
  return c.kind;
}

/** Choosing a layer family: its default choice, and which families the dataset can show. */
function useFamilies(manifest: Manifest, choice: LayerChoice, onChoice: (c: LayerChoice) => void, defaults: { lead: number }) {
  const active = familyOf(choice);
  const pick = (f: Family) => {
    if (f === active) return;
    if (f === "none") onChoice({ kind: "none" });
    if (f === "forecast") onChoice({ kind: "forecast", variable: "particulate_domoic", lead: defaults.lead });
    if (f === "satellite") onChoice({ kind: "satellite", product: satelliteLatest(manifest, "olci300") ? "olci300" : "viirs750", day: null });
    if (f === "currents") onChoice({ kind: "currents", hour: null, mean: false });
  };
  const available: Record<Family, boolean> = {
    none: true,
    forecast: !!charmRun(manifest),
    satellite: !!(satelliteLatest(manifest, "olci300") || satelliteLatest(manifest, "viirs750")),
    currents: currentsHourly(manifest).length > 0,
  };
  return { active, pick, available };
}

const CHIP_NAME: Record<Family, string> = { none: "Seafloor", forecast: "Forecast", satellite: "Satellite", currents: "Currents" };

/** On a phone, the layer families as one row of chips at the sheet's peek: one tap from any state. */
export function LayerChips({ manifest, choice, onChoice, defaults }: { manifest: Manifest; choice: LayerChoice; onChoice: (c: LayerChoice) => void; defaults: { lead: number } }) {
  const { active, pick, available } = useFamilies(manifest, choice, onChoice, defaults);
  return (
    <div role="radiogroup" aria-label="Map layer" className="-mx-1 flex gap-1.5 overflow-x-auto px-1 [scrollbar-width:none]" data-testid="layer-chips">
      {(Object.keys(FAMILY) as Family[]).map((f) => (
        <button
          key={f}
          type="button"
          role="radio"
          aria-checked={f === active}
          disabled={!available[f]}
          onClick={() => pick(f)}
          data-testid={`chip-${f}`}
          className={`nx-press nx-focus flex min-h-14 shrink-0 items-center gap-2 rounded-xl border px-3 text-[14.5px] font-semibold disabled:opacity-40 ${f === active ? "border-n-accent bg-n-accent-bg text-n-ink" : "border-n-line-strong text-n-ink-2"}`}
        >
          <Icon name={FAMILY[f].icon} className={`h-[18px] w-[18px] ${FAMILY[f].tone}`} />
          {CHIP_NAME[f]}
        </button>
      ))}
    </div>
  );
}

export function LayerList({ manifest, choice, onChoice, now, defaults }: { manifest: Manifest; choice: LayerChoice; onChoice: (c: LayerChoice) => void; now: Date | null; defaults: { lead: number } }) {
  const { active, pick, available } = useFamilies(manifest, choice, onChoice, defaults);
  return (
    <div
      role="radiogroup"
      aria-label="What the map shows"
      className="flex flex-col gap-1"
      data-testid="layer-list"
      // a radio group moves with the arrow keys (one tab stop, the checked layer)
      onKeyDown={(e) => {
        const order = (Object.keys(FAMILY) as Family[]).filter((f) => available[f]);
        const i = order.indexOf(active);
        const step = e.key === "ArrowDown" || e.key === "ArrowRight" ? 1 : e.key === "ArrowUp" || e.key === "ArrowLeft" ? -1 : 0;
        if (!step || i < 0) return;
        e.preventDefault();
        const next = order[(i + step + order.length) % order.length];
        pick(next);
        (e.currentTarget.querySelector(`[data-testid="layer-${next}"]`) as HTMLElement | null)?.focus();
      }}
    >
      {(Object.keys(FAMILY) as Family[]).map((f) => {
        const on = f === active;
        const meta = FAMILY[f];
        const line = freshnessLine(manifest, f, now);
        return (
          <button
            key={f}
            type="button"
            role="radio"
            aria-checked={on}
            tabIndex={on ? 0 : -1}
            disabled={!available[f]}
            onClick={() => pick(f)}
            data-testid={`layer-${f}`}
            className={`nx-press nx-focus flex min-h-[56px] w-full items-center gap-3 rounded-xl px-3 text-left disabled:opacity-45 ${on ? "bg-n-accent-bg ring-1 ring-n-accent" : "hover:bg-n-surface-2"}`}
          >
            <span className={`grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-n-surface-2 ${meta.tone}`}>
              <Icon name={meta.icon} className="h-5 w-5" />
            </span>
            <span className="min-w-0 flex-1">
              <span className="block text-[15px] font-semibold leading-tight text-n-ink">{meta.name}</span>
              <span className={`block text-[13px] leading-snug ${on ? "text-n-ink-2" : "text-n-ink-3"}`}>
                <span className={meta.tone}>{meta.kind}</span>
                {line ? ` · ${line}` : ""}
                {!available[f] ? " · not available" : ""}
              </span>
            </span>
            <span aria-hidden className={`grid h-5 w-5 shrink-0 place-items-center rounded-full border-2 ${on ? "border-n-accent" : "border-n-line-strong"}`}>
              {on && <span className="h-2.5 w-2.5 rounded-full bg-n-accent" />}
            </span>
          </button>
        );
      })}
    </div>
  );
}

/** Options of the active layer: the C-HARM variable, the satellite sensor. Time lives in the timebar. */
export function LayerOptions({ manifest, choice, onChoice }: { manifest: Manifest; choice: LayerChoice; onChoice: (c: LayerChoice) => void }) {
  if (choice.kind === "forecast")
    return (
      <div className="flex flex-col gap-1.5">
        <label htmlFor="nx-variable" className="nx-eyebrow">
          Variable
        </label>
        <select
          id="nx-variable"
          value={choice.variable}
          onChange={(e) => onChoice({ ...choice, variable: e.target.value as CharmVariable })}
          className="nx-focus min-h-11 rounded-lg border border-n-line-strong bg-n-surface px-3 text-[15px] text-n-ink"
          data-testid="variable-select"
        >
          {CHARM_VARIABLES.map((v) => (
            <option key={v} value={v}>
              {VARIABLE_WORDS[v].name}
            </option>
          ))}
        </select>
      </div>
    );
  if (choice.kind === "satellite") {
    const prods = (["olci300", "viirs750", "multi"] as const).filter((p) => satelliteLatest(manifest, p));
    return (
      <div className="flex flex-col gap-1.5">
        <span className="nx-eyebrow">Sensor</span>
        <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label="Sensor">
          {prods.map((p) => (
            <button
              key={p}
              type="button"
              role="radio"
              aria-checked={choice.product === p}
              onClick={() => onChoice({ kind: "satellite", product: p, day: null })}
              className={`nx-press nx-focus min-h-11 rounded-lg px-3 text-[14px] font-semibold ${choice.product === p ? "bg-n-ink text-n-bg" : "border border-n-line-strong text-n-ink-2 hover:text-n-ink"}`}
              data-testid={`sensor-${p}`}
            >
              {SAT_WORDS[p].name} <span className="font-normal opacity-75">{SAT_WORDS[p].native}</span>
            </button>
          ))}
        </div>
        {!choice.day && (
          <>
            <span className="nx-eyebrow mt-2">Show</span>
            <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label="What the colours show">
              {(["chl", "age", ...(choice.product === "multi" ? (["sensor"] as const) : [])] as const).map((v) => (
                <button
                  key={v}
                  type="button"
                  role="radio"
                  aria-checked={(choice.view ?? "chl") === v}
                  onClick={() => onChoice({ ...choice, view: v })}
                  className={`nx-press nx-focus min-h-11 rounded-lg px-3 text-[14px] font-semibold ${(choice.view ?? "chl") === v ? "bg-n-ink text-n-bg" : "border border-n-line-strong text-n-ink-2 hover:text-n-ink"}`}
                  data-testid={`satview-${v}`}
                >
                  {v === "chl" ? "Chlorophyll" : v === "age" ? "Age of each pixel" : "Which sensor"}
                </button>
              ))}
            </div>
          </>
        )}
      </div>
    );
  }
  if (choice.kind === "currents" && (satelliteLatest(manifest, "olci300") || satelliteLatest(manifest, "viirs750")))
    return (
      <label className="nx-focus flex min-h-11 items-start gap-2.5 text-[14px] text-n-ink" data-testid="currents-chl">
        <input type="checkbox" className="mt-[3px] h-[18px] w-[18px] accent-[var(--nx-accent)]" checked={!!choice.chl} onChange={(e) => onChoice({ ...choice, chl: e.target.checked })} data-testid="currents-chl-toggle" />
        <span>
          <span className="font-semibold">Compare with chlorophyll</span>
          <span className="block text-[12.5px] leading-snug text-n-ink-3">The newest clear satellite view underneath. Two observations from different times, shown together, not combined into one value.</span>
        </span>
      </label>
    );
  return null;
}

function Ramp({ palette, labels }: { palette: Palette; labels: [string, string, string] }) {
  return (
    <div>
      <div className="h-3 w-full rounded-sm" style={{ background: gradientCss(displayPalette(palette)) }} aria-hidden />
      <div className="nx-data mt-1 flex justify-between text-[12px] text-n-ink-3">
        <span>{labels[0]}</span>
        <span>{labels[1]}</span>
        <span>{labels[2]}</span>
      </div>
    </div>
  );
}

function NoData() {
  return (
    <span className="flex items-center gap-2 text-[13px] text-n-ink-2">
      <span aria-hidden className="h-3 w-5 rounded-sm" style={{ background: "repeating-linear-gradient(135deg, rgba(160,180,195,0.55) 0 1.5px, transparent 1.5px 4px)", outline: "1px solid rgba(160,180,195,0.4)" }} />
      No value (not modelled, cloud or no observation): never zero
    </span>
  );
}

/** The legend of the layer on the map: units, scale, the no-data mark. */
export function Legend({ choice, resolved }: { choice: LayerChoice; resolved: Resolved }) {
  if (choice.kind === "forecast" && resolved.forecastLayer?.palette)
    return (
      <figure className="flex flex-col gap-2" data-testid="legend-forecast" aria-label={`Legend: ${VARIABLE_WORDS[choice.variable].what}`}>
        <figcaption className="text-[13.5px] leading-snug text-n-ink-2">{VARIABLE_WORDS[choice.variable].what}</figcaption>
        <Ramp palette={resolved.forecastLayer.palette} labels={["0%", "50%", "100%"]} />
        <p className="text-[12.5px] leading-snug text-n-ink-3" data-testid="forecast-caveat">
          A probability for each 3 km model cell, not a toxin measurement and not a closure decision. A low probability does not mean an area is safe. Classes are display steps, not risk levels.
        </p>
        <NoData />
      </figure>
    );
  if (choice.kind === "satellite" && choice.view === "age")
    return (
      <figure className="flex flex-col gap-2" data-testid="legend-age" aria-label="Legend: days since each pixel was observed">
        <figcaption className="text-[13.5px] leading-snug text-n-ink-2">Days since each pixel was last seen clear</figcaption>
        <div className="flex">
          {AGE_COLOURS.map((c, i) => (
            <span key={c} className="flex flex-1 flex-col items-center gap-1">
              <span className="h-3 w-full" style={{ background: c }} aria-hidden />
              <span className="nx-data text-[11.5px] text-n-ink-3">{i}</span>
            </span>
          ))}
        </div>
        <p className="text-[12.5px] leading-snug text-n-ink-3">Each colour is an observation date, not a value. Older than 7 days is left out.</p>
      </figure>
    );
  if (choice.kind === "satellite" && choice.view === "sensor")
    return (
      <figure className="flex flex-col gap-2" data-testid="legend-sensor" aria-label="Legend: which sensor each pixel comes from">
        <figcaption className="text-[13.5px] leading-snug text-n-ink-2">Which sensor each pixel comes from</figcaption>
        <div className="flex gap-4 text-[13px]">
          <span className="flex items-center gap-2"><span className="h-3 w-5" style={{ background: SENSOR_COLOURS[0] }} aria-hidden />Sentinel-3 OLCI 300 m</span>
          <span className="flex items-center gap-2"><span className="h-3 w-5" style={{ background: SENSOR_COLOURS[1] }} aria-hidden />VIIRS 750 m</span>
        </div>
      </figure>
    );
  if (choice.kind === "satellite" && resolved.satLayer?.palette)
    return (
      <figure className="flex flex-col gap-2" data-testid="legend-satellite" aria-label="Legend: chlorophyll-a, milligrams per cubic metre, log scale">
        <figcaption className="text-[13.5px] leading-snug text-n-ink-2">Chlorophyll-a, mg/m³ (log scale)</figcaption>
        <Ramp palette={resolved.satLayer.palette} labels={["0.05", "1.6", "50"]} />
        <p className="text-[12.5px] leading-snug text-n-ink-3">Algae of every kind, not toxin. Cloud and coastal glare leave gaps; a gap is not clear water.</p>
        <NoData />
      </figure>
    );
  if (choice.kind === "currents" && choice.chl && resolved.satLayer?.palette)
    return (
      <div className="flex flex-col gap-4" data-testid="legend-compare">
        <Legend choice={{ kind: "satellite", product: "olci300", day: null }} resolved={resolved} />
        <Legend choice={{ ...choice, chl: false }} resolved={resolved} />
      </div>
    );
  if (choice.kind === "currents")
    return (
      <figure className="flex flex-col gap-2" data-testid="legend-currents" aria-label="Legend: current speed classes, metres per second">
        <figcaption className="text-[13.5px] leading-snug text-n-ink-2">Surface current speed, m/s (arrow length)</figcaption>
        <div className="flex items-end gap-3">
          {SPEED_CLASSES.map((v, i) => (
            <span key={v} className="flex flex-col items-center gap-1">
              <svg width="30" height="14" aria-hidden>
                <line x1={15 - ARROW_LENGTHS[i] / 2} y1={7} x2={15 + ARROW_LENGTHS[i] / 2} y2={7} stroke="currentColor" strokeWidth={1.7} strokeLinecap="round" className="text-n-ink" />
              </svg>
              <span className="nx-data text-[11.5px] text-n-ink-3">{i === SPEED_CLASSES.length - 1 ? `≥${v}` : v}</span>
            </span>
          ))}
        </div>
        <p className="text-[12.5px] leading-snug text-n-ink-3" data-testid="currents-note">
          One arrow per observed 2 km radar cell. No arrow means no observation, not calm water. Observed currents, not a forecast. 1 m/s ≈ {KNOTS_PER_MS.toFixed(1)} knots.{" "}
          {choice.mean ? "The 24-hour mean smooths out most daily and tidal back-and-forth." : "Each hour is a separate snapshot; nothing is shown between hours."}
        </p>
      </figure>
    );
  if (choice.kind === "none")
    return <p className="text-[13.5px] leading-snug text-n-ink-2">NOAA ETOPO 2022 and Coastal Relief Model seafloor. Harbours, shore stations and official limits stay on every layer.</p>;
  return null;
}

/** Where the field comes from and when it is valid: one line per source, its own date; how old it is
 *  against the source's published freshness policy (stale, or historical: not current conditions);
 *  a failed update; a day with nothing observed; regions the radar did not see. */
export function Provenance({ manifest, choice, resolved, now, regions = [] }: { manifest: Manifest; choice: LayerChoice; resolved: Resolved; now: Date | null; regions?: { id: string; label: string }[] }) {
  const stamps = stampLines({
    group: choice.kind === "none" ? null : choice.kind,
    forecast: resolved.forecastLayer,
    run: charmRun(manifest),
    satellite: resolved.satLayer,
    currents: resolved.curLayer,
    // currents compared with chlorophyll: both sources, each with its own time, and how far apart they are
    combined: choice.kind === "currents" && !!choice.chl,
    now,
  });
  // the radar's coverage, region by region: the regions it did not see are named, in one line
  if (choice.kind === "currents" && resolved.curLayer?.coverage) {
    const cl = resolved.curLayer;
    const unseen = regions.filter((r) => stampLines({ group: "currents", currents: cl, region: r }).some((x) => x.testid === "stamp-currents-gap"));
    if (unseen.length)
      stamps.push({
        kind: "gap",
        testid: "stamp-currents-gap",
        text: `No radar observations ${cl.layer_id.endsWith("mean24h") ? "for the 24 h mean" : "this hour"} in ${unseen.map((r) => r.label).join(", ")}: no data, not calm water`,
      });
  }
  const sat = choice.kind === "satellite" ? resolved.satLayer : null;
  // a daily overpass with nothing clear: said, never drawn as an empty sea
  const clouded = !!sat && !sat.grid && !sat.tiles && !!sat.time.observed_date;
  if (!stamps.length && !clouded) return null;
  const status = choice.kind === "forecast" ? sourceStatus(manifest, "charm") : choice.kind === "satellite" ? sourceStatus(manifest, "satellite_chl") : choice.kind === "currents" ? sourceStatus(manifest, "hf_radar") : null;
  const layer = choice.kind === "forecast" ? resolved.forecastLayer : choice.kind === "satellite" ? resolved.satLayer : choice.kind === "currents" ? resolved.curLayer : null;
  // forecast freshness follows the run's issue date; observations follow the newest observation shown
  const fresh = status && now ? (choice.kind === "forecast" ? classifySource(status, now) : layer ? classifyTime(layer.freshness, layer.composite ? { ...layer.time, observed_date: layer.composite.newest_observed_date } : layer.time, now) : null) : null;
  return (
    <div className="flex flex-col gap-1" data-testid="provenance">
      {stamps.map((s, i) => (
        <p key={`${s.testid}-${i}`} data-testid={s.testid} className={`flex items-start gap-2 text-[13px] leading-snug ${s.kind === "gap" ? "text-n-stale" : "text-n-ink-2"}`}>
          <span className={`mt-[5px] h-2 w-2 shrink-0 rounded-full ${s.kind === "model" ? "bg-n-model" : s.kind === "gap" ? "bg-n-stale" : "bg-n-measured"}`} aria-hidden />
          {s.text}
        </p>
      ))}
      {sat && sat.coverage && !clouded && <CoverageLine layer={sat} regions={regions} />}
      {choice.kind === "currents" && choice.chl && resolved.satLayer?.coverage && <CoverageLine layer={resolved.satLayer} regions={regions} />}
      {clouded && (
        <p className="flex items-start gap-2 text-[13px] font-semibold leading-snug text-n-stale" data-testid="sat-clouded">
          <Icon name="alert" className="mt-0.5 h-4 w-4 shrink-0" />
          No clear observation on {dayLabel(sat!.time.observed_date!)}: clouds, fog or no overpass. Nothing is shown for that day; it is not clear water.
        </p>
      )}
      {choice.kind === "forecast" && charmRun(manifest)?.issued_date_derived && (
        <p className="text-[12.5px] leading-snug text-n-ink-3" data-testid="issued-inferred">
          Issue date inferred: NOAA does not publish it; Neritic infers it from the forecast&rsquo;s own valid dates.
        </p>
      )}
      {fresh?.state === "stale" && (
        <p className="flex items-start gap-2 text-[13px] leading-snug text-n-stale" data-testid="stale-note">
          <Icon name="clock" className="mt-0.5 h-4 w-4 shrink-0" />
          {choice.kind === "forecast"
            ? `No newer C-HARM run has been published since ${dayLabel(fresh.basisDate!)} (${fresh.ageDays} days). This is the newest available forecast; its dates are shown as issued.`
            : `Older than usual: newest observation ${fresh.ageDays} days ago.`}
        </p>
      )}
      {fresh?.state === "historical" && (
        <p className="flex items-start gap-2 text-[13px] font-semibold leading-snug text-n-down" data-testid="historical-note">
          <Icon name="alert" className="mt-0.5 h-4 w-4 shrink-0" />
          {choice.kind === "forecast" ? "This forecast is out of date" : "These observations are out of date"} ({fresh.ageDays} days). Shown for reference only; it does not describe current conditions.
        </p>
      )}
      {status && status.outcome !== "updated" && (
        <p className="flex items-start gap-2 text-[13px] text-n-stale" data-testid="provenance-status">
          <Icon name="clock" className="mt-0.5 h-4 w-4 shrink-0" />
          The last update {status.outcome === "failed" ? "failed" : "did not complete"}
          {status.last_attempt_at ? ` (${clockLabel(status.last_attempt_at)})` : ""}; the newest published data is shown.
        </p>
      )}
    </div>
  );
}

/** Reference lines on land: county borders and names, with what they do not mean. */
export function ReferenceLines({ prefs, onChange }: { prefs: CountyPrefs; onChange: (p: CountyPrefs) => void }) {
  const row = "nx-focus flex min-h-11 items-center gap-2.5 text-[14px] text-n-ink";
  return (
    <fieldset className="flex flex-col gap-0.5" data-testid="reference-lines">
      <legend className="nx-eyebrow mb-1">Reference on land</legend>
      <label className={row}>
        <input type="checkbox" className="h-[18px] w-[18px] accent-[var(--nx-accent)]" checked={prefs.lines} onChange={(e) => onChange({ ...prefs, lines: e.target.checked })} data-testid="county-lines" />
        County lines
      </label>
      <label className={row}>
        <input type="checkbox" className="h-[18px] w-[18px] accent-[var(--nx-accent)]" checked={prefs.labels} onChange={(e) => onChange({ ...prefs, labels: e.target.checked })} data-testid="county-labels" />
        County names
      </label>
      <p className="text-[12.5px] leading-snug text-n-ink-3" data-testid="county-note">
        {COUNTIES.note}
      </p>
    </fieldset>
  );
}

/** How much of the sea a satellite view actually saw: the published coverage, with what it counts
 *  (C-HARM's 3 km ocean cells with any clear pixel), when (the view's own dates) and where (by region). */
function CoverageLine({ layer, regions }: { layer: LayerArtifact; regions: { id: string; label: string }[] }) {
  const cov = layer.coverage!;
  const when = coverageWhen(layer);
  const label = (id: string) => regions.find((r) => r.id === id)?.label ?? id.replace(/_/g, " ");
  const pct = (f: number) => `${Math.round(f * 100)}%`;
  return (
    <div className="flex flex-col gap-0.5 text-[13px] leading-snug text-n-ink-2" data-testid="sat-coverage">
      <p>
        <span className="font-semibold text-n-ink">Seen clearly: {pct(cov.domain_observed_fraction)}</span> of the {cov.domain_reference_cells != null ? `${cov.domain_reference_cells.toLocaleString("en-US")} ` : ""}ocean cells in this view&rsquo;s area{when ? ` (${when})` : ""}. The rest is cloud, glare or no overpass, not low chlorophyll.
      </p>
      {cov.regions.length > 0 && (
        <p data-testid="sat-coverage-regions">
          By region: {cov.regions.map((r) => `${label(r.region_id)} ${pct(r.observed_fraction)}`).join(" · ")}
        </p>
      )}
      <details className="text-[12.5px] text-n-ink-3">
        <summary className="nx-focus inline-flex min-h-11 cursor-pointer items-center font-semibold">What is counted</summary>
        {cov.reference}
      </details>
    </div>
  );
}
