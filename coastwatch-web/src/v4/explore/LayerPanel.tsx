"use client";

import { useEffect, useId, useRef, useState } from "react";

import type { LayerArtifact, Manifest, Palette } from "@/generated/schema";
import { CHARM_VARIABLES, charmRun, coveragePct, satelliteLatest, sourceStatus, type CharmVariable } from "@/lib/layers";
import { KNOTS_PER_MS, currentsHourly, defaultHour } from "@/lib/currents";
import { AGE_COLOURS, SENSOR_COLOURS, displayPalette, gradientCss } from "@/lib/palette";
import { SPEED_CLASSES, ARROW_LENGTHS } from "@/lib/basemap";
import { FLOW_SKY, SPEED_FULL, SPEED_RAMP, skyColour, speedShare } from "@/lib/flowSim";
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

/**
 * v10 layer index (direction A): each family by an honest name, what kind of thing it is (model,
 * observed, outlook, analysis, reference) in its family colour, and its source. Listed in the
 * concept's order: the model, the observations, the derived products, the reference.
 */
const FAMILY: Record<Family, { name: string; kind: string; source: string; icon: string; tone: string; sw: string; hatch?: boolean }> = {
  forecast: { name: "Bloom & toxin probability", kind: "Model", source: "NOAA C-HARM", icon: "cell", tone: "text-n-model", sw: "var(--nx-model)" },
  satellite: { name: "Chlorophyll", kind: "Observed", source: "satellite", icon: "layers", tone: "text-n-measured", sw: "var(--nx-measured)" },
  outlook: { name: "Chlorophyll outlook", kind: "Outlook, not observed", source: "8-day mean", icon: "clock", tone: "text-n-model", sw: "var(--nx-model)", hatch: true },
  // MUR: a gap-free analysis blended from several sensors, not one observation
  sst: { name: "Sea temperature", kind: "Analysis", source: "JPL MUR", icon: "chart", tone: "text-n-conditions", sw: "var(--nx-conditions)" },
  currents: { name: "Surface currents", kind: "Observed", source: "HF radar", icon: "compass", tone: "text-n-conditions", sw: "var(--nx-conditions)" },
  none: { name: "Seafloor only", kind: "Reference", source: "NOAA relief", icon: "map", tone: "text-n-ink-2", sw: "#55636c" },
};

/** "Oct 21–28" (or across months, "Oct 30–Nov 6"): an outlook's window. */
export function windowLabel(from: string, to: string): string {
  const a = dayLabel(from);
  const b = dayLabel(to);
  return a.split(" ")[0] === b.split(" ")[0] && !a.includes(",") ? `${a}–${b.split(" ")[1]}` : `${a}–${b}`;
}

/** The index's short date line for a family: the run, the newest pixels, the window, the hour. */
function indexLine(m: Manifest, f: Family, derived: DerivedIndex | null, now: Date | null): string | null {
  const age = (d: string) => (now ? ` · ${relTime(d, now)}` : "");
  if (f === "forecast") {
    const run = charmRun(m);
    return run ? `run ${dayLabel(run.issued_date)}${age(run.issued_date)}` : "no run published";
  }
  if (f === "satellite") {
    const l = satelliteLatest(m, "olci300") ?? satelliteLatest(m, "viirs750");
    const d = l?.composite?.newest_observed_date ?? l?.time.observed_date;
    return d ? `newest ${dayLabel(d)}${age(d)}` : "no recent clear view";
  }
  if (f === "outlook") {
    const o = derivedOutlook(derived);
    return o ? windowLabel(o.valid_from, o.valid_to) : null;
  }
  if (f === "sst") {
    const t = derivedSst(derived);
    const d = t?.observed_dates[t.observed_dates.length - 1];
    return d ? `${dayLabel(d)}${age(d)}` : null;
  }
  if (f === "currents") {
    const t = defaultHour(m)?.time.valid_time;
    return t ? `newest ${clockLabel(t)}` : "no radar hour published";
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

export function LayerList({ manifest, choice, onChoice, now, defaults, derived = null }: { manifest: Manifest; choice: LayerChoice; onChoice: (c: LayerChoice) => void; now: Date | null; defaults: { lead: number }; derived?: DerivedIndex | null }) {
  const { active, pick, available, listed } = useFamilies(manifest, choice, onChoice, defaults, derived);
  return (
    <div
      role="radiogroup"
      aria-label="What the map shows"
      className="flex flex-col"
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
        const line = indexLine(manifest, f, derived, now);
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
            className={`nx-index-row nx-focus grid w-full grid-cols-[14px_minmax(0,1fr)] gap-x-2.5 border-t border-n-line py-2 text-left first:border-t-0 disabled:opacity-45 ${on ? "text-n-ink" : "text-n-ink-2 hover:text-n-ink"}`}
            style={{ ["--tone" as string]: meta.sw }}
          >
            <span aria-hidden className={`nx-index-sw mt-[6px] h-2 w-2 rounded-[1px] ${meta.hatch ? "nx-index-sw-hatch" : ""}`} />
            <span className="min-w-0">
              <span className="block text-[14px] leading-snug">{meta.name}</span>
              <span className="block text-[11.5px] leading-snug text-n-ink-3">
                <span className={on ? meta.tone : ""}>{meta.kind}</span> · {meta.source}
                {line ? `, ${line}` : ""}
                {!available[f] ? " · not available" : ""}
              </span>
            </span>
          </button>
        );
      })}
    </div>
  );
}

/**
 * Options of the active layer. `primary`: what the layer shows, beside it in the index (the C-HARM
 * variable, the chlorophyll compared under the currents). `adjust`: how it is drawn, one level in
 * (the satellite sensor and view, the estimate under the gaps, the field's opacity, and for the
 * currents the flow or arrows and the speed colours, passed in as `flow`). Time lives in the rail.
 */
export function LayerOptions({ manifest, choice, onChoice, estimate = null, part = "all", opacity, onOpacity, flow = null }: { manifest: Manifest; choice: LayerChoice; onChoice: (c: LayerChoice) => void; estimate?: EstimateLayer | null; part?: "all" | "primary" | "adjust"; opacity?: number; onOpacity?: (v: number) => void; flow?: React.ReactNode }) {
  const primary = part !== "adjust";
  const adjust = part !== "primary";
  const seg = (on: boolean) => `nx-press nx-focus min-h-11 rounded-[3px] px-2.5 text-[13px] ${on ? "bg-n-ink text-n-bg" : "border border-n-line-strong text-n-ink-2 hover:text-n-ink"}`;
  const out: React.ReactNode[] = [];
  if (primary && choice.kind === "forecast")
    out.push(
      <div key="var" className="flex flex-col gap-1">
        <label htmlFor="nx-variable" className="nx-caps">
          Variable
        </label>
        <select
          id="nx-variable"
          value={choice.variable}
          onChange={(e) => onChoice({ ...choice, variable: e.target.value as CharmVariable })}
          className="nx-select nx-focus min-h-11 rounded-[3px] border border-n-line-strong bg-n-surface px-2.5 text-[14px] text-n-ink"
          data-testid="variable-select"
        >
          {CHARM_VARIABLES.map((v) => (
            <option key={v} value={v}>
              {VARIABLE_WORDS[v].name}
            </option>
          ))}
        </select>
      </div>,
    );
  if (primary && choice.kind === "currents" && (satelliteLatest(manifest, "olci300") || satelliteLatest(manifest, "viirs750")))
    out.push(
      <label key="chl" className="nx-focus flex min-h-11 items-start gap-2.5 text-[13.5px] text-n-ink" data-testid="currents-chl">
        <input type="checkbox" className="mt-[3px] h-[18px] w-[18px] accent-[var(--nx-accent)]" checked={!!choice.chl} onChange={(e) => onChoice({ ...choice, chl: e.target.checked })} data-testid="currents-chl-toggle" />
        <span>
          <span>Compare with chlorophyll</span>
          <span className="nx-margin block text-[12.5px] leading-snug">The newest clear satellite view underneath: two observations from different times, shown together, not combined into one value.</span>
        </span>
      </label>,
    );
  if (adjust && choice.kind === "satellite") {
    const prods = (["olci300", "viirs750", "multi"] as const).filter((p) => satelliteLatest(manifest, p));
    out.push(
      <div key="sensor" className="flex flex-col gap-1">
        <span className="nx-caps">Sensor</span>
        <div className="flex flex-wrap gap-1" role="radiogroup" aria-label="Sensor">
          {prods.map((p) => (
            <button key={p} type="button" role="radio" aria-checked={choice.product === p} onClick={() => onChoice({ kind: "satellite", product: p, day: null })} className={seg(choice.product === p)} data-testid={`sensor-${p}`}>
              {SAT_WORDS[p].name} <span className="nx-data opacity-75">{SAT_WORDS[p].native}</span>
            </button>
          ))}
        </div>
      </div>,
    );
    if (!choice.day)
      out.push(
        <div key="view" className="flex flex-col gap-1">
          <span className="nx-caps">Colours show</span>
          <div className="flex flex-wrap gap-1" role="radiogroup" aria-label="What the colours show">
            {(["chl", "age", ...(choice.product === "multi" ? (["sensor"] as const) : [])] as const).map((v) => (
              <button key={v} type="button" role="radio" aria-checked={(choice.view ?? "chl") === v} onClick={() => onChoice({ ...choice, view: v })} className={seg((choice.view ?? "chl") === v)} data-testid={`satview-${v}`}>
                {v === "chl" ? "Chlorophyll" : v === "age" ? "Age of each pixel" : "Which sensor"}
              </button>
            ))}
          </div>
        </div>,
      );
    // the gap-filling estimate: under the observed pixels, never over them; on unless turned off
    if (estimate && !choice.day && (choice.view ?? "chl") === "chl")
      out.push(
        <label key="est" className="nx-focus flex min-h-11 items-start gap-2.5 text-[13.5px] text-n-ink" data-testid="sat-estimate">
          <input type="checkbox" className="mt-[3px] h-[18px] w-[18px] accent-[var(--nx-accent)]" checked={choice.estimate !== false} onChange={(e) => onChoice({ ...choice, estimate: e.target.checked })} data-testid="sat-estimate-toggle" />
          <span>
            <span>Fill the gaps with estimates</span>
            <span className="nx-margin block text-[12.5px] leading-snug">Stippled where clouds or the coast hid the water; near shore it may be extrapolated, not filled. Estimated, not observed; observed pixels are never changed.</span>
          </span>
        </label>,
      );
  }
  if (adjust && flow) out.push(<div key="flow">{flow}</div>);
  // the field's opacity (display only: the colours keep their legend values; nothing is blended between families)
  if (adjust && onOpacity && opacity != null && choice.kind !== "none" && !(choice.kind === "currents" && !choice.chl))
    out.push(
      <label key="opacity" className="flex flex-col gap-0.5" data-testid="layer-opacity">
        <span className="flex items-baseline justify-between">
          <span className="nx-caps">Field opacity</span>
          <span className="nx-data text-[11.5px] text-n-ink-2">{Math.round(opacity * 100)}%</span>
        </span>
        <input type="range" min={30} max={100} step={5} value={Math.round(opacity * 100)} onChange={(e) => onOpacity(Number(e.target.value) / 100)} className="nx-range nx-focus w-full" aria-label="Field opacity" data-testid="layer-opacity-range" />
      </label>,
    );
  if (!out.length) return null;
  return <div className="flex flex-col gap-3">{out}</div>;
}

function Ramp({ palette, labels, inline = false }: { palette: Palette; labels: string[]; inline?: boolean }) {
  if (inline)
    return (
      <div className="flex items-center gap-2">
        <span className="nx-data shrink-0 text-[10.5px] text-n-ink-2">{labels[0]}</span>
        <div className="h-1.5 min-w-0 flex-1 rounded-[1px]" style={{ background: gradientCss(displayPalette(palette)) }} aria-hidden />
        <span className="nx-data shrink-0 text-[10.5px] text-n-ink-2">{labels[labels.length - 1]}</span>
      </div>
    );
  return (
    <div>
      <div className="h-1.5 w-full rounded-[1px]" style={{ background: gradientCss(displayPalette(palette)) }} aria-hidden />
      <div className="nx-data mt-1 flex justify-between text-[10.5px] text-n-ink-2">
        {labels.map((l, i) => (
          <span key={i}>{l}</span>
        ))}
      </div>
    </div>
  );
}

function NoData({ children = "No value (not modelled, cloud or no observation): never zero", testid }: { children?: React.ReactNode; testid?: string }) {
  return (
    <span className="flex items-start gap-2 text-[12px] leading-snug text-n-ink-2" data-testid={testid}>
      <span aria-hidden className="nx-sw-gap mt-[3px] h-[9px] w-[18px] shrink-0" />
      <span>{children}</span>
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

/** The legend's short caption: what the colours are, with units, in one line. */
const CAPTION: Record<CharmVariable, string> = {
  particulate_domoic: "Chance domoic acid exceeds 500 ng/L",
  pseudo_nitzschia: "Chance Pseudo-nitzschia exceeds 10,000 cells/L",
  cellular_domoic: "Chance cellular domoic acid exceeds 10 pg/cell",
};

/**
 * The ⓘ beside a legend: the longer explanation (every caveat, the no-data marks, the bay list) stays
 * in the page, mounted and hidden, and opens above the dock (or in place on a phone). One tap, Escape
 * or a tap elsewhere closes it.
 */
function MoreAbout({ label, children, inline }: { label: string; children: React.ReactNode; inline: boolean }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const box = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!open) return;
    const k = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      setOpen(false);
    };
    const c = (e: PointerEvent) => !inline && box.current && !box.current.contains(e.target as Node) && setOpen(false);
    window.addEventListener("keydown", k, true);
    window.addEventListener("pointerdown", c);
    return () => {
      window.removeEventListener("keydown", k, true);
      window.removeEventListener("pointerdown", c);
    };
  }, [open, inline]);
  return (
    // static: the explanation opens above the whole dock, aligned with the legend (its static position)
    <div ref={box} className="static">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((o) => !o)}
        className={`nx-focus nx-tint -my-2 -mr-2 grid h-11 w-11 shrink-0 place-items-center rounded-full hover:text-n-ink ${open ? "text-n-accent" : "text-n-ink-3"}`}
        data-testid="legend-more"
      >
        <Icon name="info" className="h-[18px] w-[18px]" />
        <span className="sr-only">{open ? "Hide" : "More about"} {label}</span>
      </button>
      <div
        id={id}
        hidden={!open}
        className={
          inline
            ? "mt-2 flex flex-col gap-2 border-t border-n-line pt-2 text-[13px] leading-snug text-n-ink-2"
            : "nx-deck absolute bottom-[calc(100%+12px)] z-30 flex max-h-[min(60vh,520px)] w-[min(420px,calc(100vw-32px))] flex-col gap-2.5 overflow-y-auto p-4 text-[13px] leading-snug text-n-ink-2 [scrollbar-width:thin]"
        }
        data-testid="legend-detail"
      >
        {children}
      </div>
    </div>
  );
}

/**
 * The map's one legend (v7 dock): what the colours mean with units, the ramp, and one line of what it
 * is not; everything longer (the full caveats, the no-data marks, the bays) behind the ⓘ, in the page.
 * `inline`: on a phone the explanation opens in place in the sheet; `noRamp`: the sheet's peek already
 * draws the ramp.
 */
/** `flow`: the currents drawn as moving particles (null: static arrows), and whether they are coloured by speed. */
export function Legend({ choice, resolved, check = null, inline = false, noRamp = false, flow = null }: { choice: LayerChoice; resolved: Resolved; check?: DerivedValidationEntry | null; inline?: boolean; noRamp?: boolean; flow?: { colour: boolean } | null }) {
  const frame = (testid: string, aria: string, caption: React.ReactNode, ramp: React.ReactNode, visible: React.ReactNode, detail: React.ReactNode, label: string) => (
    <figure className="flex min-w-0 flex-col gap-2" data-testid={testid} aria-label={aria}>
      <div className="flex items-start justify-between gap-2">
        <figcaption className="nx-caps min-w-0 pt-0.5">{caption}</figcaption>
        {detail && (
          <MoreAbout label={label} inline={inline}>
            {detail}
          </MoreAbout>
        )}
      </div>
      {!noRamp && ramp}
      {visible}
    </figure>
  );
  const short = (t: React.ReactNode) => <p className="text-[12px] leading-snug text-n-ink-2">{t}</p>;

  if (choice.kind === "forecast" && resolved.forecastLayer?.palette)
    return frame(
      "legend-forecast",
      `Legend: ${VARIABLE_WORDS[choice.variable].what}`,
      CAPTION[choice.variable],
      <Ramp palette={resolved.forecastLayer.palette} labels={["0%", "25%", "50%", "75%", "100%"]} />,
      <>
        {short("A model probability: not toxin, not a closure; low is not safe.")}
        <span className="flex items-start gap-2 text-[12.5px] leading-snug text-n-ink-2">
          <span aria-hidden className="mt-[2px] h-3 w-5 shrink-0 rounded-sm" style={{ background: `rgb(${OUTSIDE_TINT_RGB.join(",")})`, opacity: 0.8, outline: "1px solid rgba(160,180,195,0.35)" }} />
          <span>Shaded near shore and in bays: no NOAA value, not zero</span>
        </span>
      </>,
      <>
        <p className="font-semibold text-n-ink">{VARIABLE_WORDS[choice.variable].what}</p>
        <p data-testid="forecast-caveat">A probability for each 3 km model cell, not a toxin measurement and not a closure decision. A low probability does not mean an area is safe. Classes are display steps, not risk levels.</p>
        <span className="flex items-start gap-2" data-testid="legend-outside">
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
      </>,
      "the forecast's colours",
    );
  if (choice.kind === "sst" && resolved.sstLayer) {
    const [lo, hi] = SST_PALETTE.domain;
    const mid = (lo + hi) / 2;
    const analysis = sstIsAnalysis(resolved.sstLayer);
    return frame(
      "legend-sst",
      "Legend: sea-surface temperature, degrees Fahrenheit",
      "Sea-surface temperature, °F",
      <Ramp palette={SST_PALETTE} labels={[`≤${Math.round(cToF(lo))}°F`, `${Math.round(cToF(mid))}°F`, `≥${Math.round(cToF(hi))}°F`]} />,
      short(analysis ? "An analysis (blended, gaps filled): the temperature just below the skin, not at depth." : "Observed by satellite: the skin of the sea, not at depth."),
      <>
        <p data-testid="sst-caveat">
          {analysis
            ? "An analysis: several satellite sensors and in-situ observations blended, with gaps filled. The foundation temperature (just below the skin), not the temperature at depth. "
            : "Observed by satellite: the skin of the sea, not the temperature at depth. "}
          A sharp change in colour may mark a front where satellites saw clear sky. Tap a point for °C and °F.
        </p>
        <NoData>{analysis ? "No value (land, or outside the coastal scope): never cold water" : "No value (cloud, land or no observation): never cold water"}</NoData>
      </>,
      "sea temperature",
    );
  }
  if (choice.kind === "outlook" && resolved.outlookLayer)
    return frame(
      "legend-outlook",
      "Legend: chlorophyll outlook, milligrams per cubic metre, log scale",
      <>Chlorophyll-a, mean {windowLabel(resolved.outlookLayer.valid_from, resolved.outlookLayer.valid_to)}, mg/m³</>,
      <Ramp palette={resolved.outlook?.palette ?? CHL_PALETTE} labels={["0.05", "0.3", "1.6", "9", "50"]} />,
      <span className="flex items-start gap-2 text-[12.5px] leading-snug text-n-ink-2">
        <span className="mt-[2px] flex">
          <Swatch color="#5bb267" texture="hatch" />
        </span>
        <span>
          <span className="font-semibold text-n-ink">Outlook, not observed.</span> Algae, not toxin; not a daily forecast.
        </span>
      </span>,
      <>
        <p>Chlorophyll-a expected on average over {windowLabel(resolved.outlookLayer.valid_from, resolved.outlookLayer.valid_to)}, mg/m³ (log scale). Fine diagonal lines mark every outlook pixel.</p>
        <p data-testid="outlook-coverage">Algae of every kind, not toxin, and not a day-by-day forecast. A 4 km grid of the open coast, about 11–19 km offshore. No outlook in bays (turbid water; MODIS chlorophyll unreliable): never drawn or read there. Its skill against plain baselines is in About this layer.</p>
        <NoData>No outlook (outside the area it covers): never zero</NoData>
      </>,
      "the outlook",
    );
  if (choice.kind === "satellite" && choice.view === "age")
    return frame(
      "legend-age",
      "Legend: days since each pixel was observed",
      "Days since each pixel was last seen clear",
      <div className="flex">
        {AGE_COLOURS.map((c, i) => (
          <span key={c} className="flex flex-1 flex-col items-center gap-0.5">
            <span className="h-2 w-full" style={{ background: c }} aria-hidden />
            <span className="nx-data text-[11px] text-n-ink-3">{i}</span>
          </span>
        ))}
      </div>,
      short("Each colour is an observation date, not a value. Older than 7 days is left out."),
      null,
      "pixel ages",
    );
  if (choice.kind === "satellite" && choice.view === "sensor")
    return frame(
      "legend-sensor",
      "Legend: which sensor each pixel comes from",
      "Which sensor each pixel comes from",
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[12.5px]">
        <span className="flex items-center gap-2"><span className="h-3 w-5" style={{ background: SENSOR_COLOURS[0] }} aria-hidden />Sentinel-3 OLCI 300 m</span>
        <span className="flex items-center gap-2"><span className="h-3 w-5" style={{ background: SENSOR_COLOURS[1] }} aria-hidden />VIIRS 750 m</span>
      </div>,
      null,
      null,
      "sensors",
    );
  if (choice.kind === "satellite" && resolved.satLayer?.palette) {
    const est = resolved.estimate ? resolved.estimateLayer : null;
    return frame(
      "legend-satellite",
      "Legend: chlorophyll-a, milligrams per cubic metre, log scale",
      "Chlorophyll-a, mg/m³ (log scale)",
      <Ramp palette={resolved.satLayer.palette} labels={["0.05", "0.3", "1.6", "9", "50"]} />,
      <>
        <NoData testid="legend-gap">No observation (cloud, fog or no pass): a gap is not clear water, and not low chlorophyll.</NoData>
        {resolved.estimate && (
          // the swatch words stay in view; the pipeline's class words are in the page, behind the ⓘ
          <span className="flex flex-col gap-1 text-[12px] leading-snug text-n-ink-2">
            <span className="flex items-center gap-2" data-testid="legend-estimate">
              <Swatch color="#8cc3a4" texture="stipple" />
              <span className="font-semibold text-n-ink">Estimated, not observed.</span>
              <span hidden>{est?.class_grid ? `${classWords(est, 1)}. In bays: ${classWords(est, 3)}.` : "Gaps estimated from recent and nearby views; near shore it may be extrapolated."} Lightened and stippled; observed pixels are drawn over it, unchanged.</span>
            </span>
            {est?.class_grid && (
              <span className="flex items-center gap-2" data-testid="legend-strip">
                <Swatch color="#b4d4c2" texture="strip" />
                <span className="font-semibold text-n-ink">Extrapolated toward shore, never observed.</span>
                <span hidden>{classWords(est, 2)}. Paler, densely stippled.</span>
              </span>
            )}
          </span>
        )}
        <p className="text-[12px] leading-snug text-n-ink">Algae of every kind, not toxin.</p>
      </>,
      <>
        <p data-testid="legend-turbid">{BAY_TURBID}: San Francisco, San Pablo, Suisun, Tomales, Humboldt, Morro and San Diego Bays and Elkhorn Slough.</p>
        {resolved.estimate &&
          (est?.class_grid ? (
            <>
              <span className="flex items-start gap-2">
                <span className="mt-[3px] flex">
                  <Swatch color="#8cc3a4" texture="stipple" />
                </span>
                <span>
                  <span className="font-semibold text-n-ink">Estimated, not observed.</span> {classWords(est, 1)}. In bays: {classWords(est, 3)}. Lightened and stippled; observed pixels are drawn over it, unchanged.
                </span>
              </span>
              <span className="flex items-start gap-2">
                <span className="mt-[3px] flex">
                  <Swatch color="#b4d4c2" texture="strip" />
                </span>
                <span>
                  <span className="font-semibold text-n-ink">Extrapolated toward shore, never observed.</span> {classWords(est, 2)}. Paler, densely stippled.
                </span>
              </span>
            </>
          ) : (
            <span className="flex items-start gap-2">
              <span className="mt-[3px] flex">
                <Swatch color="#8cc3a4" texture="stipple" />
              </span>
              <span>
                <span className="font-semibold text-n-ink">Estimated, not observed.</span> Gaps estimated from recent and nearby views; near shore it may be extrapolated. Lightened and stippled; observed pixels are drawn over it, unchanged.
              </span>
            </span>
          ))}
      </>,
      "chlorophyll",
    );
  }
  if (choice.kind === "currents" && choice.chl && resolved.satLayer?.palette)
    return (
      <div className="grid min-w-0 gap-3 sm:grid-cols-2" data-testid="legend-compare">
        <Legend choice={{ kind: "satellite", product: "olci300", day: null }} resolved={resolved} inline={inline} noRamp={noRamp} />
        <Legend choice={{ ...choice, chl: false }} resolved={resolved} inline={inline} noRamp={noRamp} flow={flow} />
      </div>
    );
  if (choice.kind === "currents" && flow)
    return frame(
      "legend-currents",
      "Legend: moving streaks show the direction and relative speed of observed surface currents",
      "Surface currents, observed",
      flow.colour ? <SpeedBuckets /> : <FlowScale />,
      <>
        {short("Observed, not a forecast. No streaks means no observation, not calm water.")}
        <p className="text-[12.5px] leading-snug text-n-ink-2" data-testid="flow-scale-note">
          Longer, brighter streaks are faster currents (the cell&rsquo;s observed speed, m/s). Screen speed is a scaled symbol: not the water&rsquo;s real speed, not a path.
        </p>
      </>,
      <p data-testid="currents-note">
        Streaks move through one observed {choice.mean ? "24-hour mean" : "hour"}, each in its own 2 km radar cell&rsquo;s vector (nothing is interpolated between cells); they stop at the edge of coverage, and the faint squares are the cells the radar observed. On screen they move at a speed proportional to the current, the same at every zoom: a symbol, not a drift you could follow. 1 m/s ≈ {KNOTS_PER_MS.toFixed(1)} knots.{" "}
        {choice.mean ? "The 24-hour mean smooths out most daily and tidal back-and-forth." : "Each hour is a separate snapshot; nothing is shown between hours, and a new hour is a cut."}
      </p>,
      "currents",
    );
  if (choice.kind === "currents")
    return frame(
      "legend-currents",
      "Legend: current speed classes, metres per second",
      "Surface current speed, m/s",
      <CurrentArrows />,
      short("Observed, not a forecast. No arrow means no observation, not calm."),
      <p data-testid="currents-note">
        One arrow per observed 2 km radar cell. No arrow means no observation, not calm water. Observed currents, not a forecast. 1 m/s ≈ {KNOTS_PER_MS.toFixed(1)} knots.{" "}
        {choice.mean ? "The 24-hour mean smooths out most daily and tidal back-and-forth." : "Each hour is a separate snapshot; nothing is shown between hours."}
      </p>,
      "currents",
    );
  if (choice.kind === "none")
    return (
      <figure className="flex min-w-0 flex-col gap-1.5" data-testid="legend-none" aria-label="Legend: seafloor depth, metres">
        <figcaption className="flex items-baseline justify-between gap-2">
          <span className="nx-caps">Seafloor depth · m</span>
          <span className="text-[11px] text-n-ink-3">reference, not data</span>
        </figcaption>
        <DepthRamp />
        <p className="text-[12px] leading-snug text-n-ink-2">NOAA ETOPO 2022 and Coastal Relief Model relief, with isobaths. The shelf break (200 m) is the brighter line. Harbours, shore stations and official limits stay on every layer.</p>
      </figure>
    );
  return null;
}

/** The seafloor's depth tint (pipeline/scripts/build_relief.py TINT), shore to abyss, with the isobaths' depths. */
const DEPTH_STOPS: [number, string][] = [[0, "#3e8fb0"], [60, "#2f7a9e"], [200, "#225e84"], [600, "#1a486c"], [1500, "#133555"], [3200, "#0e2540"]];
function DepthRamp() {
  const ticks = [0, 200, 500, 1000, 2000, 3000];
  // a square-root depth axis: the shelf, where the fishing is, gets room
  const at = (d: number) => (Math.sqrt(d) / Math.sqrt(3200)) * 100;
  return (
    <div aria-hidden className="w-full max-w-[240px]">
      <div className="h-2 w-full rounded-[1px]" style={{ background: `linear-gradient(90deg, ${DEPTH_STOPS.map(([d, c]) => `${c} ${at(d).toFixed(1)}%`).join(", ")})` }} />
      <div className="relative mt-0.5 h-3.5">
        {ticks.map((d) => (
          <span key={d} className="nx-data absolute top-0 -translate-x-1/2 text-[10.5px] leading-[14px] text-n-ink-3 first:translate-x-0" style={{ left: `${at(d)}%` }}>
            {d.toLocaleString("en-US")}
          </span>
        ))}
      </div>
    </div>
  );
}

/**
 * The flow's speed scale (v10 rework): the sky ramp the streaks are drawn in, with real speeds in m/s, and
 * three sample streaks (a faster current is longer, brighter and lighter). The screen speed is a symbol.
 */
function FlowScale() {
  const ticks = [0, 0.2, 0.4, SPEED_FULL];
  const at = (v: number) => (v / SPEED_FULL) * 100;
  return (
    <div className="flex w-full max-w-[240px] flex-col gap-1" data-testid="legend-flow-scale">
      <svg width="100%" height="12" viewBox="0 0 240 12" preserveAspectRatio="none" aria-hidden>
        {[0.08, 0.3, 0.6].map((v, i) => {
          const [r, g, b] = skyColour(v);
          const x = 10 + (v / SPEED_FULL) * 200;
          const len = 8 + (v / SPEED_FULL) * 34;
          return <line key={i} x1={x - len / 2} y1={6} x2={x + len / 2} y2={6} stroke={`rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`} strokeWidth={2} strokeLinecap="round" opacity={0.22 + 0.73 * Math.sqrt(speedShare(v))} />;
        })}
      </svg>
      <div className="h-1.5 w-full rounded-[1px]" style={{ background: `linear-gradient(90deg, ${FLOW_SKY.map(([t, c]) => `${c} ${t * 100}%`).join(", ")})` }} />
      <div className="relative h-3.5">
        {ticks.map((v, i) => (
          <span key={v} className={`nx-data absolute top-0 whitespace-nowrap text-[10.5px] leading-[14px] text-n-ink-3 ${i === 0 ? "" : i === ticks.length - 1 ? "-translate-x-full" : "-translate-x-1/2"}`} style={{ left: `${at(v)}%` }} data-testid="flow-scale-tick">
            {i === ticks.length - 1 ? `≥${v} m/s` : v}
          </span>
        ))}
      </div>
    </div>
  );
}

/** The optional speed colours: one sequential colour per speed class (the same classes as the arrows), m/s. */
function SpeedBuckets() {
  return (
    <div className="flex items-end gap-1.5" data-testid="legend-speed-buckets">
      {SPEED_CLASSES.map((v, i) => (
        <span key={v} className="flex flex-col items-center gap-0.5">
          <span aria-hidden className="h-2 w-7 rounded-sm" style={{ background: SPEED_RAMP[i] }} />
          <span className="nx-data text-[11px] text-n-ink-3">{i === SPEED_CLASSES.length - 1 ? `≥${v}` : v}</span>
        </span>
      ))}
      <span className="pb-px text-[11px] text-n-ink-3">m/s</span>
    </div>
  );
}

function CurrentArrows() {
  return (
    <div className="flex items-end gap-3">
      {SPEED_CLASSES.map((v, i) => (
        <span key={v} className="flex flex-col items-center gap-0.5">
          <svg width="30" height="12" aria-hidden>
            <line x1={15 - ARROW_LENGTHS[i] / 2} y1={6} x2={15 + ARROW_LENGTHS[i] / 2} y2={6} stroke="currentColor" strokeWidth={1.7} strokeLinecap="round" className="text-n-ink" />
          </svg>
          <span className="nx-data text-[11px] text-n-ink-3">{i === SPEED_CLASSES.length - 1 ? `≥${v}` : v}</span>
        </span>
      ))}
    </div>
  );
}

/**
 * The ramp alone, in one line (the phone sheet's peek: the colours and their range under the layer
 * button). Decorative: the legend itself, with its words, follows in the sheet.
 */
export function LegendLine({ choice, resolved, flow = null }: { choice: LayerChoice; resolved: Resolved; flow?: { colour: boolean } | null }) {
  const pal = choice.kind === "forecast" ? resolved.forecastLayer?.palette : choice.kind === "satellite" && (choice.view ?? "chl") === "chl" ? resolved.satLayer?.palette : choice.kind === "outlook" ? (resolved.outlook?.palette ?? CHL_PALETTE) : choice.kind === "sst" ? SST_PALETTE : choice.kind === "currents" && choice.chl ? resolved.satLayer?.palette : null;
  const labels: [string, string, string] | null =
    choice.kind === "forecast" ? ["0%", "50%", "100%"] : choice.kind === "sst" ? [`≤${Math.round(cToF(SST_PALETTE.domain[0]))}°F`, "", `≥${Math.round(cToF(SST_PALETTE.domain[1]))}°F`] : pal ? ["0.05", "1.6", "50 mg/m³"] : null;
  if (pal && labels) return <div aria-hidden><Ramp palette={pal} labels={labels} inline /></div>;
  if (choice.kind === "currents" && flow) return <div aria-hidden>{flow.colour ? <SpeedBuckets /> : <FlowScale />}</div>;
  if (choice.kind === "currents")
    return (
      <div aria-hidden className="flex items-center gap-2 text-[11.5px] text-n-ink-3">
        <CurrentArrows />
        <span>m/s</span>
      </div>
    );
  return null;
}

/** Where the field comes from and when it is valid: one line per source, its own date; how old it is
 *  against the source's published freshness policy (stale, or historical: not current conditions);
 *  a failed update; a day with nothing observed; regions the radar did not see.
 *  `part` (v7): "dock" is what stays in view under the legend (the layer's own date line and every
 *  problem: stale, out of date, failed, clouded, unseen regions); "about" is the rest (further stamps,
 *  coverage, the inferred issue date, the outlook's and estimate's method and skill), in About this layer. */
export function Provenance({ manifest, choice, resolved, now, regions = [], part = "all" }: { manifest: Manifest; choice: LayerChoice; resolved: Resolved; now: Date | null; regions?: { id: string; label: string }[]; part?: "all" | "dock" | "about" }) {
  if (choice.kind === "outlook") {
    const o = resolved.outlookLayer;
    if (!o) return null;
    if (part === "dock") return <p className="text-[12.5px] leading-snug text-n-ink-3">Issued {dayLabel(o.issued)}{now ? ` (${relTime(o.issued, now)})` : ""} · one mean over {windowLabel(o.valid_from, o.valid_to)}</p>;
    return <OutlookCard layer={o} now={now} />;
  }
  if (choice.kind === "sst") {
    const t = resolved.sstLayer;
    if (!t) return null;
    const last = t.observed_dates[t.observed_dates.length - 1];
    if (part === "dock") return last ? <p className="text-[12.5px] leading-snug text-n-ink-3">{sstIsAnalysis(t) ? "Analysis of" : "Observed"} {dayLabel(last)}{now ? ` (${relTime(last, now)})` : ""} · {t.source}</p> : null;
    return <SstCard layer={t} now={now} />;
  }
  return (
    <>
      <ObservedProvenance manifest={manifest} choice={choice} resolved={resolved} now={now} regions={regions} part={part} />
      {part !== "dock" && resolved.estimate && resolved.estimateLayer && <EstimateCard layer={resolved.estimateLayer} />}
    </>
  );
}

function ObservedProvenance({ manifest, choice, resolved, now, regions = [], part = "all" }: { manifest: Manifest; choice: LayerChoice; resolved: Resolved; now: Date | null; regions?: { id: string; label: string }[]; part?: "all" | "dock" | "about" }) {
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
  // the dock keeps the layer's own line (both sources' lines when two are compared) and every gap;
  // About this layer keeps any further line
  // (the forecast's own line repeats the timebar's run and day, so it is in About this layer)
  const docked = (x: (typeof stamps)[number], i: number) => x.kind === "gap" || (i === 0 && choice.kind !== "forecast") || (choice.kind === "currents" && !!choice.chl);
  const shown = stamps.filter((x, i) => part === "all" || (part === "dock" ? docked(x, i) : !docked(x, i)));
  const dock = part !== "about";
  const about = part !== "dock";
  const status = choice.kind === "forecast" ? sourceStatus(manifest, "charm") : choice.kind === "satellite" ? sourceStatus(manifest, "satellite_chl") : choice.kind === "currents" ? sourceStatus(manifest, "hf_radar") : null;
  const layer = choice.kind === "forecast" ? resolved.forecastLayer : choice.kind === "satellite" ? resolved.satLayer : choice.kind === "currents" ? resolved.curLayer : null;
  // forecast freshness follows the run's issue date; observations follow the newest observation shown
  const fresh = status && now ? (choice.kind === "forecast" ? classifySource(status, now) : layer ? classifyTime(layer.freshness, layer.composite ? { ...layer.time, observed_date: layer.composite.newest_observed_date } : layer.time, now) : null) : null;
  const problem = dock && (clouded || fresh?.state === "stale" || fresh?.state === "historical" || (!!status && status.outcome !== "updated"));
  if (!shown.length && !problem && !about) return null;
  return (
    <div className="flex flex-col gap-1" data-testid="provenance">
      {shown.map((s, i) => (
        <p key={`${s.testid}-${i}`} data-testid={s.testid} className={`flex items-start gap-2 text-[13px] leading-snug ${s.kind === "gap" ? "text-n-stale" : "text-n-ink-2"}`}>
          <span className={`mt-[5px] h-2 w-2 shrink-0 rounded-full ${s.kind === "model" ? "bg-n-model" : s.kind === "gap" ? "bg-n-stale" : "bg-n-measured"}`} aria-hidden />
          {s.text}
        </p>
      ))}
      {about && sat && sat.coverage && !clouded && <CoverageLine layer={sat} regions={regions} />}
      {about && choice.kind === "currents" && choice.chl && resolved.satLayer?.coverage && <CoverageLine layer={resolved.satLayer} regions={regions} />}
      {dock && clouded && (
        <p className="flex items-start gap-2 text-[13px] font-semibold leading-snug text-n-stale" data-testid="sat-clouded">
          <Icon name="alert" className="mt-0.5 h-4 w-4 shrink-0" />
          No clear observation on {dayLabel(sat!.time.observed_date!)}: clouds, fog or no overpass. Nothing is shown for that day; it is not clear water.
        </p>
      )}
      {about && choice.kind === "forecast" && charmRun(manifest)?.issued_date_derived && (
        <p className="text-[12.5px] leading-snug text-n-ink-3" data-testid="issued-inferred">
          Issue date inferred: NOAA does not publish it; Neritic infers it from the forecast&rsquo;s own valid dates.
        </p>
      )}
      {dock && fresh?.state === "stale" && (
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
      {dock && fresh?.state === "historical" && (
        <p className="flex items-start gap-2 text-[13px] font-semibold leading-snug text-n-down" data-testid="historical-note">
          <Icon name="alert" className="mt-0.5 h-4 w-4 shrink-0" />
          {choice.kind === "forecast" ? "This forecast is out of date" : "These observations are out of date"} ({fresh.ageDays} days). Shown for reference only; it does not describe current conditions.
        </p>
      )}
      {/* a failed forecast update is already said in the stale line above, with its time */}
      {dock && status && status.outcome !== "updated" && !(choice.kind === "forecast" && status.outcome === "failed" && fresh?.state === "stale") && (
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

/** The active layer in words, for the layer button: its name, what it shows, and its kind (a safety distinction: model, observed, outlook, analysis). */
export function layerWords(c: LayerChoice): { title: string; detail: string | null; kind: string; tone: string; icon: string; sw: string; hatch: boolean } {
  const f = FAMILY[familyOf(c)];
  const detail =
    c.kind === "forecast"
      ? VARIABLE_WORDS[c.variable].short
      : c.kind === "satellite"
        ? `${SAT_WORDS[c.product]?.name ?? ""}${c.day ? ` · ${dayLabel(c.day)}` : ""}`
        : c.kind === "currents"
          ? ["HF radar", c.mean ? "24 h mean" : null, c.chl ? "with chlorophyll" : null].filter(Boolean).join(", ")
          : f.source;
  return { title: f.name, detail, kind: f.kind, tone: f.tone, icon: f.icon, sw: f.sw, hatch: !!f.hatch };
}
