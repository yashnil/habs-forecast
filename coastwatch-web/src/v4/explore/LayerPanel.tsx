"use client";

import { useEffect, useRef } from "react";

import type { LayerArtifact, Manifest, Palette } from "@/generated/schema";
import { CHARM_VARIABLES, charmRun, coveragePct, satelliteLatest, sourceStatus, type CharmVariable } from "@/lib/layers";
import { KNOTS_PER_MS, currentsHourly, defaultHour } from "@/lib/currents";
import { AGE_COLOURS, SENSOR_COLOURS, displayPalette, gradientCss } from "@/lib/palette";
import { SPEED_CLASSES, ARROW_LENGTHS } from "@/lib/basemap";
import { stampLines } from "@/lib/stamp";
import { COUNTIES } from "@/content/copy";
import { OUTSIDE_TINT_RGB } from "@/lib/gridTiles";
import { outsideLabel } from "@/lib/gridTilesContext";
import { BAY_TURBID } from "@/lib/bays";
import { ESTIMATE_WORDS, classWords } from "@/components/map/HoverReadout";
import { Icon } from "@/v4/ui/Icon";
import { CHL_PALETTE, SST_PALETTE, cToF, derivedSst, sstIsAnalysis, type SstLayer, bandLabel, checkSentence, derivedOutlook, lowerIsBetter, type DerivedValidationEntry, skillRows, type DerivedIndex, type EstimateLayer, type OutlookLayer } from "@/lib/derived";
import { SAT_WORDS, VARIABLE_WORDS, coverageWhen, type LayerChoice, type Resolved } from "./layers";
import { clockLabel, dayLabel, relTime } from "@/v4/ui/time";
import { classifySource, classifyTime } from "@/lib/freshness";

type Family = "none" | "forecast" | "satellite" | "outlook" | "sst" | "currents";

const FAMILY: Record<Family, { name: string; kind: string; icon: string; tone: string }> = {
  none: { name: "Seafloor and coast", kind: "Reference", icon: "map", tone: "text-n-ink-2" },
  forecast: { name: "NOAA C-HARM forecast", kind: "Model", icon: "cell", tone: "text-n-model" },
  satellite: { name: "Satellite chlorophyll", kind: "Observation", icon: "layers", tone: "text-n-measured" },
  outlook: { name: "Outlook (8 days)", kind: "Outlook, not observed", icon: "clock", tone: "text-n-model" },
  // MUR: a gap-free analysis blended from several sensors, not one observation
  sst: { name: "Sea temperature", kind: "Analysis", icon: "chart", tone: "text-n-conditions" },
  currents: { name: "Surface currents", kind: "Observation", icon: "compass", tone: "text-n-conditions" },
};

/** "Oct 21–28" (or across months, "Oct 30–Nov 6"): an outlook's window. */
export function windowLabel(from: string, to: string): string {
  const a = dayLabel(from);
  const b = dayLabel(to);
  return a.split(" ")[0] === b.split(" ")[0] && !a.includes(",") ? `${a}–${b.split(" ")[1]}` : `${a}–${b}`;
}

function freshnessLine(m: Manifest, f: Family, now: Date | null, derived: DerivedIndex | null = null): string | null {
  if (f === "sst") {
    const t = derivedSst(derived);
    const d = t?.observed_dates[t.observed_dates.length - 1];
    return t && d ? `${sstIsAnalysis(t) ? "Analysis of" : "Observed"} ${dayLabel(d)}${now ? ` · ${relTime(d, now)}` : ""}` : null;
  }
  if (f === "outlook") {
    const o = derivedOutlook(derived);
    return o ? `Issued ${dayLabel(o.issued)} · mean ${windowLabel(o.valid_from, o.valid_to)}` : null;
  }
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
    // the hour the map opens on: the newest with usable coverage (the same rule as the timebar)
    const t = defaultHour(m)?.time.valid_time;
    return t ? `Newest hour ${now ? relTime(t, now) : t}` : "No radar hour published";
  }
  return null;
}

function familyOf(c: LayerChoice): Family {
  return c.kind;
}

/** Choosing a layer family: its default choice, and which families the dataset can show. */
function useFamilies(manifest: Manifest, choice: LayerChoice, onChoice: (c: LayerChoice) => void, defaults: { lead: number }, derived: DerivedIndex | null) {
  const active = familyOf(choice);
  const pick = (f: Family) => {
    if (f === active) return;
    if (f === "none") onChoice({ kind: "none" });
    if (f === "outlook") onChoice({ kind: "outlook" });
    if (f === "sst") onChoice({ kind: "sst" });
    if (f === "forecast") onChoice({ kind: "forecast", variable: "particulate_domoic", lead: defaults.lead });
    if (f === "satellite") onChoice({ kind: "satellite", product: satelliteLatest(manifest, "olci300") ? "olci300" : "viirs750", day: null });
    if (f === "currents") onChoice({ kind: "currents", hour: null, mean: false });
  };
  const available: Record<Family, boolean> = {
    none: true,
    forecast: !!charmRun(manifest),
    satellite: !!(satelliteLatest(manifest, "olci300") || satelliteLatest(manifest, "viirs750")),
    outlook: !!derivedOutlook(derived),
    sst: !!derivedSst(derived),
    currents: currentsHourly(manifest).length > 0,
  };
  // a family the dataset does not offer at all (the outlook without a derived index) is not listed
  const listed = (Object.keys(FAMILY) as Family[]).filter((f) => (f !== "outlook" || available.outlook) && (f !== "sst" || available.sst));
  return { active, pick, available, listed };
}

const CHIP_NAME: Record<Family, string> = { none: "Seafloor", forecast: "Forecast", satellite: "Satellite", outlook: "Outlook", sst: "Sea temp", currents: "Currents" };

/** On a phone, the layer families as one row of chips at the sheet's peek: one tap from any state. */
export function LayerChips({ manifest, choice, onChoice, defaults, derived = null }: { manifest: Manifest; choice: LayerChoice; onChoice: (c: LayerChoice) => void; defaults: { lead: number }; derived?: DerivedIndex | null }) {
  const { active, pick, available, listed } = useFamilies(manifest, choice, onChoice, defaults, derived);
  const row = useRef<HTMLDivElement | null>(null);
  // the chosen chip is always in view (five chips scroll on a phone)
  useEffect(() => {
    const r = row.current;
    const el = r?.querySelector<HTMLElement>('[aria-checked="true"]');
    if (r && el && (el.offsetLeft < r.scrollLeft || el.offsetLeft + el.offsetWidth > r.scrollLeft + r.clientWidth)) r.scrollLeft = el.offsetLeft - 8;
  }, [active]);
  return (
    <div ref={row} role="radiogroup" aria-label="Map layer" className="-mx-1 flex gap-1.5 overflow-x-auto px-1 [scrollbar-width:none]" data-testid="layer-chips">
      {listed.map((f) => (
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

export function LayerList({ manifest, choice, onChoice, now, defaults, derived = null }: { manifest: Manifest; choice: LayerChoice; onChoice: (c: LayerChoice) => void; now: Date | null; defaults: { lead: number }; derived?: DerivedIndex | null }) {
  const { active, pick, available, listed } = useFamilies(manifest, choice, onChoice, defaults, derived);
  return (
    <div
      role="radiogroup"
      aria-label="What the map shows"
      className="flex flex-col gap-1"
      data-testid="layer-list"
      // a radio group moves with the arrow keys (one tab stop, the checked layer)
      onKeyDown={(e) => {
        const order = listed.filter((f) => available[f]);
        const i = order.indexOf(active);
        const step = e.key === "ArrowDown" || e.key === "ArrowRight" ? 1 : e.key === "ArrowUp" || e.key === "ArrowLeft" ? -1 : 0;
        if (!step || i < 0) return;
        e.preventDefault();
        const next = order[(i + step + order.length) % order.length];
        pick(next);
        (e.currentTarget.querySelector(`[data-testid="layer-${next}"]`) as HTMLElement | null)?.focus();
      }}
    >
      {listed.map((f) => {
        const on = f === active;
        const meta = FAMILY[f];
        const line = freshnessLine(manifest, f, now, derived);
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
export function LayerOptions({ manifest, choice, onChoice, estimate = null }: { manifest: Manifest; choice: LayerChoice; onChoice: (c: LayerChoice) => void; estimate?: EstimateLayer | null }) {
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
          className="nx-select nx-focus min-h-11 rounded-lg border border-n-line-strong bg-n-surface px-3 text-[15px] text-n-ink"
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
        {/* the gap-filling estimate: under the observed pixels, never over them; on unless turned off */}
        {estimate && !choice.day && (choice.view ?? "chl") === "chl" && (
          <label className="nx-focus mt-2 flex min-h-11 items-start gap-2.5 text-[14px] text-n-ink" data-testid="sat-estimate">
            <input type="checkbox" className="mt-[3px] h-[18px] w-[18px] accent-[var(--nx-accent)]" checked={choice.estimate !== false} onChange={(e) => onChoice({ ...choice, estimate: e.target.checked })} data-testid="sat-estimate-toggle" />
            <span>
              <span className="font-semibold">Fill the gaps with estimates</span>
              <span className="block text-[12.5px] leading-snug text-n-ink-3">Where clouds or the coast hid the water, a stippled estimate; next to the shore it may be extrapolated, not filled. Estimated, not observed; observed pixels are never changed.</span>
            </span>
          </label>
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

function NoData({ children = "No value (not modelled, cloud or no observation): never zero", testid }: { children?: React.ReactNode; testid?: string }) {
  return (
    <span className="flex items-center gap-2 text-[13px] text-n-ink-2" data-testid={testid}>
      <span aria-hidden className="h-3 w-5 shrink-0 rounded-sm" style={{ background: "repeating-linear-gradient(135deg, rgba(160,180,195,0.55) 0 1.5px, transparent 1.5px 4px)", outline: "1px solid rgba(160,180,195,0.4)" }} />
      {children}
    </span>
  );
}

/** A legend swatch in the map's own texture: `stipple` (context, estimate), `strip` (the estimate's shoreline strip: paler, denser dots) or `hatch` (outlook), over `color`. */
export function Swatch({ color, texture }: { color: string; texture: "stipple" | "strip" | "hatch" }) {
  const bg =
    texture === "stipple"
      ? `radial-gradient(circle, rgba(11,29,51,0.85) 0.7px, transparent 0.9px) 0 0 / 4px 4px, radial-gradient(circle, rgba(11,29,51,0.85) 0.7px, transparent 0.9px) 2px 2px / 4px 4px, ${color}`
      : texture === "strip"
        ? `radial-gradient(circle, rgba(11,29,51,0.7) 0.7px, transparent 0.9px) 0 0 / 3px 3px, ${color}`
        : `repeating-linear-gradient(135deg, transparent 0 5px, rgba(11,29,51,0.6) 5px 6px), ${color}`;
  return <span aria-hidden className="h-3 w-5 shrink-0 rounded-sm" style={{ background: bg, outline: "1px solid rgba(160,180,195,0.35)" }} />;
}

/** The legend of the layer on the map: units, scale, the no-data mark. */
export function Legend({ choice, resolved, check = null }: { choice: LayerChoice; resolved: Resolved; check?: DerivedValidationEntry | null }) {
  if (choice.kind === "forecast" && resolved.forecastLayer?.palette)
    return (
      <figure className="flex flex-col gap-2" data-testid="legend-forecast" aria-label={`Legend: ${VARIABLE_WORDS[choice.variable].what}`}>
        <figcaption className="text-[13.5px] leading-snug text-n-ink-2">{VARIABLE_WORDS[choice.variable].what}</figcaption>
        <Ramp palette={resolved.forecastLayer.palette} labels={["0%", "50%", "100%"]} />
        <p className="text-[12.5px] leading-snug text-n-ink-3" data-testid="forecast-caveat">
          A probability for each 3 km model cell, not a toxin measurement and not a closure decision. A low probability does not mean an area is safe. Classes are display steps, not risk levels.
        </p>
        <span className="flex items-start gap-2 text-[13px] leading-snug text-n-ink-2" data-testid="legend-outside">
          {/* the tint as drawn: OUTSIDE_TINT at the forecast's opacity over the sea */}
          <span aria-hidden className="mt-[3px] h-3 w-5 shrink-0 rounded-sm" style={{ background: `rgb(${OUTSIDE_TINT_RGB.join(",")})`, opacity: 0.8, outline: "1px solid rgba(160,180,195,0.35)" }} />
          <span>
            <span className="font-semibold text-n-ink">{outsideLabel(choice.variable)}.</span> Most of the nearshore band, the bays and the harbours: no probability is shown there, and it is not zero.{choice.variable === "pseudo_nitzschia" ? "" : " Pseudo-nitzschia may still have a value where domoic acid has none."} Tap it for the nearest NOAA value (as text), the pier samples nearby and the filled chlorophyll.
            {check && (
              <span className="mt-0.5 block text-[12.5px] text-n-ink-3" data-testid="legend-context-check">
                {checkSentence(check)}
              </span>
            )}
          </span>
        </span>
        <NoData>Outside the model&rsquo;s area: no value, never zero</NoData>
      </figure>
    );
  if (choice.kind === "sst" && resolved.sstLayer) {
    const [lo, hi] = SST_PALETTE.domain;
    const mid = (lo + hi) / 2;
    return (
      <figure className="flex flex-col gap-2" data-testid="legend-sst" aria-label="Legend: sea-surface temperature, degrees Fahrenheit">
        <figcaption className="text-[13.5px] leading-snug text-n-ink-2">Sea-surface temperature, °F</figcaption>
        <Ramp palette={SST_PALETTE} labels={[`≤${Math.round(cToF(lo))}°F`, `${Math.round(cToF(mid))}°F`, `≥${Math.round(cToF(hi))}°F`]} />
        <p className="text-[12.5px] leading-snug text-n-ink-3" data-testid="sst-caveat">
          {sstIsAnalysis(resolved.sstLayer)
            ? "An analysis: several satellite sensors and in-situ observations blended, with gaps filled. The foundation temperature (just below the skin), not the temperature at depth. "
            : "Observed by satellite: the skin of the sea, not the temperature at depth. "}
          A sharp change in colour may mark a front where satellites saw clear sky. Tap a point for °C and °F.
        </p>
        <NoData>{sstIsAnalysis(resolved.sstLayer) ? "No value (land, or outside the coastal scope): never cold water" : "No value (cloud, land or no observation): never cold water"}</NoData>
      </figure>
    );
  }
  if (choice.kind === "outlook" && resolved.outlookLayer)
    return (
      <figure className="flex flex-col gap-2" data-testid="legend-outlook" aria-label="Legend: chlorophyll outlook, milligrams per cubic metre, log scale">
        <figcaption className="text-[13.5px] leading-snug text-n-ink-2">Chlorophyll-a expected on average over {windowLabel(resolved.outlookLayer.valid_from, resolved.outlookLayer.valid_to)}, mg/m³ (log scale)</figcaption>
        <Ramp palette={resolved.outlook?.palette ?? CHL_PALETTE} labels={["0.05", "1.6", "50"]} />
        <span className="flex items-center gap-2 text-[13px] text-n-ink-2">
          <Swatch color="#5bb267" texture="hatch" />
          <span>
            <span className="font-semibold text-n-ink">Outlook, not observed.</span> Fine diagonal lines mark every outlook pixel.
          </span>
        </span>
        <p className="text-[12.5px] leading-snug text-n-ink-3" data-testid="outlook-coverage">Algae of every kind, not toxin, and not a day-by-day forecast. A 4 km grid of the open coast, about 11–19 km offshore. No outlook in bays (turbid water; MODIS chlorophyll unreliable): never drawn or read there. Its skill against plain baselines is below.</p>
        <NoData>No outlook (outside the area it covers): never zero</NoData>
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
        <p className="text-[12.5px] leading-snug text-n-ink-3" data-testid="legend-turbid">
          {BAY_TURBID}: San Francisco, San Pablo, Suisun, Tomales, Humboldt, Morro and San Diego Bays and Elkhorn Slough.
        </p>
        {resolved.estimate &&
          (resolved.estimateLayer?.class_grid ? (
            <>
              <span className="flex items-start gap-2 text-[13px] leading-snug text-n-ink-2" data-testid="legend-estimate">
                <span className="mt-[3px] flex">
                  <Swatch color="#8cc3a4" texture="stipple" />
                </span>
                <span>
                  <span className="font-semibold text-n-ink">Estimated, not observed.</span> {classWords(resolved.estimateLayer, 1)}. In bays: {classWords(resolved.estimateLayer, 3)}. Lightened and stippled; observed pixels are drawn over it, unchanged.
                </span>
              </span>
              <span className="flex items-start gap-2 text-[13px] leading-snug text-n-ink-2" data-testid="legend-strip">
                <span className="mt-[3px] flex">
                  <Swatch color="#b4d4c2" texture="strip" />
                </span>
                <span>
                  <span className="font-semibold text-n-ink">Extrapolated toward shore, never observed.</span> {classWords(resolved.estimateLayer, 2)}. Paler, densely stippled.
                </span>
              </span>
            </>
          ) : (
            <span className="flex items-start gap-2 text-[13px] leading-snug text-n-ink-2" data-testid="legend-estimate">
              <span className="mt-[3px] flex">
                <Swatch color="#8cc3a4" texture="stipple" />
              </span>
              <span>
                <span className="font-semibold text-n-ink">Estimated, not observed.</span> Gaps estimated from recent and nearby views; near shore it may be extrapolated. Lightened and stippled; observed pixels are drawn over it, unchanged.
              </span>
            </span>
          ))}
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
  if (choice.kind === "outlook") return resolved.outlookLayer ? <OutlookCard layer={resolved.outlookLayer} now={now} /> : null;
  if (choice.kind === "sst") return resolved.sstLayer ? <SstCard layer={resolved.sstLayer} now={now} /> : null;
  return (
    <>
      <ObservedProvenance manifest={manifest} choice={choice} resolved={resolved} now={now} regions={regions} />
      {resolved.estimate && resolved.estimateLayer && <EstimateCard layer={resolved.estimateLayer} />}
    </>
  );
}

function ObservedProvenance({ manifest, choice, resolved, now, regions = [] }: { manifest: Manifest; choice: LayerChoice; resolved: Resolved; now: Date | null; regions?: { id: string; label: string }[] }) {
  if (choice.kind === "outlook" || choice.kind === "sst") return null;
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
            ? // a failed update cannot say NOAA published nothing newer: only that Neritic's copy is this run
              status?.outcome === "failed"
              ? `Neritic's newest copy is the run of ${dayLabel(fresh.basisDate!)} (${fresh.ageDays} days old). It could not reach NOAA${status.last_attempt_at ? ` at ${clockLabel(status.last_attempt_at)}` : ""}, so a newer run may exist. Dates are shown as issued.`
              : `No newer C-HARM run has been published since ${dayLabel(fresh.basisDate!)} (${fresh.ageDays} days). This is the newest available forecast; its dates are shown as issued.`
            : `Older than usual: newest observation ${fresh.ageDays} days ago.`}
        </p>
      )}
      {fresh?.state === "historical" && (
        <p className="flex items-start gap-2 text-[13px] font-semibold leading-snug text-n-down" data-testid="historical-note">
          <Icon name="alert" className="mt-0.5 h-4 w-4 shrink-0" />
          {choice.kind === "forecast" ? "This forecast is out of date" : "These observations are out of date"} ({fresh.ageDays} days). Shown for reference only; it does not describe current conditions.
        </p>
      )}
      {/* a failed forecast update is already said in the stale line above, with its time */}
      {status && status.outcome !== "updated" && !(choice.kind === "forecast" && status.outcome === "failed" && fresh?.state === "stale") && (
        <p className="flex items-start gap-2 text-[13px] text-n-stale" data-testid="provenance-status">
          <Icon name="clock" className="mt-0.5 h-4 w-4 shrink-0" />
          The last update {status.outcome === "failed" ? "failed" : "did not complete"}
          {status.last_attempt_at ? ` (${clockLabel(status.last_attempt_at)})` : ""}; the newest published data is shown.
        </p>
      )}
    </div>
  );
}

/** County borders and names are part of the basemap (always drawn): what they are, and what they do not mean. */
export function ReferenceLines() {
  return (
    <div className="flex flex-col gap-1" data-testid="reference-lines">
      <p className="nx-eyebrow">Reference on land</p>
      <p className="flex items-center gap-2 text-[13px] text-n-ink-2">
        <svg width="28" height="8" aria-hidden className="shrink-0 text-n-ink-2">
          <line x1="1" y1="4" x2="27" y2="4" stroke="currentColor" strokeWidth="1.2" strokeDasharray="5 2 1 2" />
        </svg>
        County borders and names
      </p>
      <p className="text-[12.5px] leading-snug text-n-ink-3" data-testid="county-note">
        {COUNTIES.note}
      </p>
    </div>
  );
}

/** One baseline row against the model: the number as published, with the metric's direction. */
function SkillTable({ rows, metric, testid }: { rows: { key: string; label: string; value: number }[]; metric: string; testid: string }) {
  const model = rows.find((r) => r.key === "model")?.value;
  return (
    <table className="w-full text-left text-[13px]" data-testid={testid}>
      <caption className="pb-1 text-left text-[12px] text-n-ink-3">
        {metric}
        {lowerIsBetter(metric) ? ", lower is better" : ""}
      </caption>
      <tbody className="nx-data">
        {rows.map((r) => (
          <tr key={r.key} className="border-t border-n-line">
            <td className={`py-1 font-sans ${r.key === "model" ? "font-semibold text-n-ink" : "text-n-ink-2"}`}>{r.label}</td>
            <td className="py-1 text-right">{r.value.toFixed(3)}</td>
            <td className="py-1 pl-2 text-right font-sans text-[12px] text-n-ink-3">{r.key !== "model" && model != null ? (lowerIsBetter(metric) ? (model < r.value ? "outlook better" : model > r.value ? "outlook worse" : "same") : "") : ""}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Caveats({ items }: { items: string[] }) {
  if (!items.length) return null;
  return (
    <ul className="list-disc space-y-0.5 pl-4 text-[12.5px] leading-snug text-n-ink-3" data-testid="derived-caveats">
      {items.map((c) => (
        <li key={c}>{c}</li>
      ))}
    </ul>
  );
}

/** The outlook's provenance: what it is, when it was issued and what it covers, how it was made, its skill and its limits. */
export function OutlookCard({ layer, now }: { layer: OutlookLayer; now: Date | null }) {
  const s = layer.skill;
  return (
    <div className="flex flex-col gap-2 text-[13px] leading-snug text-n-ink-2" data-testid="provenance">
      <p className="flex items-start gap-2" data-testid="stamp-outlook">
        <span className="mt-[5px] h-2 w-2 shrink-0 rounded-full bg-n-model" aria-hidden />
        <span>
          <span className="font-semibold text-n-ink">Outlook, not observed.</span> {layer.label}: issued {dayLabel(layer.issued)}
          {now ? ` (${relTime(layer.issued, now)})` : ""}, the mean over {windowLabel(layer.valid_from, layer.valid_to)}.
        </span>
      </p>
      <p data-testid="outlook-method">
        <span className="font-semibold text-n-ink">Method.</span> {layer.method}
      </p>
      <p data-testid="outlook-basis">
        <span className="font-semibold text-n-ink">Made from.</span> {layer.basis}
      </p>
      <div className="rounded-lg border border-n-line px-2.5 py-2">
        <p className="font-semibold text-n-ink">
          Skill on held-out data{s.test_period ? ` (${s.test_period}` : ""}
          {s.n != null ? `${s.test_period ? ", " : " ("}n = ${s.n.toLocaleString("en-US")})` : s.test_period ? ")" : ""}
        </p>
        <p className="text-[12.5px] text-n-ink-2" data-testid="outlook-is-baseline">This outlook is the climatology + AR(1) baseline (no learned model).</p>
        <SkillTable rows={skillRows(s)} metric={s.metric} testid="outlook-skill" />
        {s.note && <p className="mt-1 text-[12.5px] text-n-ink-3">{s.note}</p>}
      </div>
      {layer.uncertainty_grid && <p className="text-[12.5px] text-n-ink-3">Each pixel&rsquo;s typical range comes from the outlook&rsquo;s own error on held-out data; tap a point to read it.</p>}
      <Caveats items={layer.caveats} />
    </div>
  );
}

/** Sea-surface temperature: observed or analysed, by whom, when, at what resolution, and its limits. */
export function SstCard({ layer, now }: { layer: SstLayer; now: Date | null }) {
  const d = layer.observed_dates;
  const last = d[d.length - 1];
  const analysis = sstIsAnalysis(layer);
  return (
    <div className="flex flex-col gap-2 text-[13px] leading-snug text-n-ink-2" data-testid="provenance">
      <p className="flex items-start gap-2" data-testid="stamp-sst">
        <span className="mt-[5px] h-2 w-2 shrink-0 rounded-full bg-n-conditions" aria-hidden />
        <span>
          <span className="font-semibold text-n-ink">{analysis ? "Analysis, from observations (foundation temperature)." : "Observation."}</span> {layer.label}, {d.length > 1 ? `${dayLabel(d[0])}–${dayLabel(last)}` : dayLabel(last)}
          {now ? ` (${relTime(last, now)})` : ""}. {layer.source}
          {layer.resolution_km ? `, ${layer.resolution_km < 1 ? `${Math.round(layer.resolution_km * 1000)} m` : `${layer.resolution_km} km`} cells` : ""}.
        </span>
      </p>
      {layer.method && <p>{layer.method}</p>}
      <Caveats items={layer.caveats} />
    </div>
  );
}

/** The estimate's provenance: what it fills, from which days, how, how well it did by distance from shore, and its limits. */
export function EstimateCard({ layer }: { layer: EstimateLayer }) {
  const v = layer.validation;
  const days = [...layer.based_on_days].sort();
  return (
    <details className="rounded-lg border border-n-line px-2.5 py-1.5 text-[13px] leading-snug text-n-ink-2" data-testid="estimate-provenance">
      <summary className="nx-focus flex min-h-11 cursor-pointer items-center gap-2 font-semibold text-n-ink">
        <span className="h-2 w-2 shrink-0 rounded-full bg-n-model" aria-hidden />
        Estimated, not observed: how the gaps are filled
      </summary>
      <div className="flex flex-col gap-2 pb-2">
        <p>
          {layer.label}. Issued {dayLabel(layer.issued)}
          {days.length ? `, from the clear views of ${days.length === 1 ? dayLabel(days[0]) : `${dayLabel(days[0])}–${dayLabel(days[days.length - 1])}`}` : ""}.
        </p>
        <p data-testid="estimate-method">
          <span className="font-semibold text-n-ink">Method.</span> {layer.method}
        </p>
        <div>
          <p className="font-semibold text-n-ink">
            Checked by hiding clear pixels{v.n != null ? ` (n = ${v.n.toLocaleString("en-US")})` : ""}
          </p>
          <table className="w-full text-left text-[13px]" data-testid="estimate-validation">
            <caption className="pb-1 text-left text-[12px] text-n-ink-3">
              {v.metric}
              {lowerIsBetter(v.metric) ? ", lower is better" : ""}, by distance from shore
            </caption>
            <tbody className="nx-data">
              {Object.entries(v.by_band).map(([b, x]) => (
                <tr key={b} className="border-t border-n-line">
                  <td className="py-1 font-sans text-n-ink-2">{bandLabel(b)}</td>
                  <td className="py-1 text-right">{x.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {v.baseline && <p className="mt-1 text-[12.5px] text-n-ink-3">Baseline: {v.baseline}</p>}
        </div>
        {layer.class_validation && (
          <div data-testid="estimate-class-validation">
            <p className="font-semibold text-n-ink">By kind of pixel</p>
            <table className="w-full text-left text-[13px]">
              <caption className="pb-1 text-left text-[12px] text-n-ink-3">RMSE log10(mg m-3), lower is better, as published</caption>
              <tbody className="nx-data">
                {([1, 2, 3] as const).flatMap((k) =>
                  Object.entries(layer.class_validation?.[k]?.rmse_log10 ?? {}).map(([b, x]) => (
                    <tr key={`${k}-${b}`} className="border-t border-n-line">
                      <td className="py-1 font-sans text-n-ink-2">
                        {layer.class_legend?.[k] ?? ESTIMATE_WORDS[k]} · {bandLabel(b)}
                      </td>
                      <td className="py-1 text-right">{x.toFixed(2)}</td>
                    </tr>
                  )),
                )}
              </tbody>
            </table>
          </div>
        )}
        {layer.class_note && <p className="text-[12.5px] text-n-ink-3" data-testid="estimate-class-note">{layer.class_note}</p>}
        <Caveats items={layer.caveats} />
      </div>
    </details>
  );
}

/** How much of the sea a satellite view actually saw: the published coverage, with what it counts
 *  (C-HARM's 3 km ocean cells with any clear pixel), when (the view's own dates) and where (by region). */
function CoverageLine({ layer, regions }: { layer: LayerArtifact; regions: { id: string; label: string }[] }) {
  const cov = layer.coverage!;
  const when = coverageWhen(layer);
  const label = (id: string) => regions.find((r) => r.id === id)?.label ?? id.replace(/_/g, " ");
  const pct = coveragePct;
  return (
    <div className="flex flex-col gap-0.5 text-[13px] leading-snug text-n-ink-2" data-testid="sat-coverage">
      <p>
        <span className="font-semibold text-n-ink">Seen clearly: {pct(cov.domain_observed_fraction)}</span> of the {cov.domain_reference_cells != null ? `${cov.domain_reference_cells.toLocaleString("en-US")} ` : ""}ocean cells across the California domain{when ? ` (${when})` : ""}. The rest is cloud, glare or no overpass, not low chlorophyll.
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
