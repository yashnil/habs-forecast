"use client";

import { useEffect, useRef, useState } from "react";
import type { LayerArtifact, Manifest } from "@/generated/schema";
import type { CurrentField } from "@/lib/currents";
import { currentsAt, forecastAt, NO_FORECAST, pnSibling, satelliteAt, type Readout } from "@/lib/readout";
import { contextWords, forecastContext, outsideLabel, outsideWords } from "@/lib/gridTilesContext";
import { checkSentence, cToF, derivedAt, estimateClassAt, fmtMg, sstIsAnalysis, type DerivedValidationEntry, type EstimateLayer, type OutlookLayer, type SstLayer } from "@/lib/derived";
import { BAY_TURBID, bayAt, NO_OUTLOOK_IN_BAYS } from "@/lib/bays";
import { dayLabel } from "@/v4/ui/time";

export type HoverSource =
  /** context: the map draws nearshore context for cells without a value (lib/gridTilesContext) */
  /** check: the pier-sample check of the nearest NOAA value near shore (lib/derived) */
  | { kind: "forecast"; layer: LayerArtifact; context?: boolean; check?: DerivedValidationEntry | null }
  /** estimate: the gap-filling estimate drawn under the observed pixels (lib/derived) */
  | { kind: "satellite"; layer: LayerArtifact; estimate?: EstimateLayer | null }
  | { kind: "outlook"; layer: OutlookLayer }
  | { kind: "sst"; layer: SstLayer }
  | { kind: "currents"; layer: LayerArtifact; field: CurrentField }
  | null;

/** A readout, with what kind of thing its value is when it is not a published value at that water. */
export type Read = Readout & { label?: string; tone?: "context" | "outside" | "estimate" | "strip" | "outlook" | "bay" };

const rangeWords = (r: [number, number] | null) => (r ? ` · typically ${fmtMg(r[0])}–${fmtMg(r[1])} mg/m³` : "");

/** The outlook at a point: never an observation, always with its window and its typical range; never in a bay. */
export async function outlookRead(baseUrl: string, l: OutlookLayer, lat: number, lon: number): Promise<Read> {
  const when = `Neritic outlook, issued ${dayLabel(l.issued)} · mean ${dayLabel(l.valid_from)}–${dayLabel(l.valid_to)}`;
  // v6: the map never draws the outlook over bay water, so it is never read there either
  if (bayAt(lat, lon)) return { kind: "model", value: null, unit: "", what: "chlorophyll outlook", when, none: NO_OUTLOOK_IN_BAYS, tone: "bay" };
  const r = await derivedAt(baseUrl, l, lat, lon);
  if (r.value == null) return { kind: "model", value: null, unit: "", what: "chlorophyll outlook", when, none: "No outlook for this water: outside the area the outlook covers. Not low chlorophyll." };
  return { kind: "model", value: fmtMg(r.value), unit: "mg/m³", what: `chlorophyll expected on average over the 8 days${rangeWords(r.range)}`, when, label: "Outlook, not observed", tone: "outlook" };
}

/**
 * Sea-surface temperature at a point: °C as published, with °F beside it. MUR is an analysis: a
 * gap-free blend, so a missing value is land or water outside its coastal scope, never cloud.
 */
export async function sstRead(baseUrl: string, l: SstLayer, lat: number, lon: number): Promise<Read> {
  const r = await derivedAt(baseUrl, l, lat, lon);
  const last = l.observed_dates[l.observed_dates.length - 1];
  const analysis = sstIsAnalysis(l);
  const when = `${l.source} · ${analysis ? "analysis for" : "observed"} ${l.observed_dates.length > 1 ? `${dayLabel(l.observed_dates[0])}–${dayLabel(last)}` : dayLabel(last)}`;
  if (r.value == null)
    return { kind: "observation", value: null, unit: "", what: "sea-surface temperature", when, none: analysis ? "No temperature here: land, or water outside the analysis's coastal scope. Not cold water." : "No temperature here: cloud, land or no observation. Not cold water." };
  return { kind: "observation", value: r.value.toFixed(1), unit: "°C", what: `≈ ${cToF(r.value).toFixed(1)} °F · ${analysis ? "foundation temperature (below the skin), an analysis" : "sea-surface temperature"}`, when, label: analysis ? "Analysis" : "Observation" };
}

/**
 * The estimate's words for each class when the index publishes no legend for it (v6). The pipeline's
 * own `class_legend` is quoted verbatim when present; no multiplier or error is ever hard-coded.
 */
export const ESTIMATE_WORDS = {
  1: "Filled from nearby clear water",
  2: "Extrapolated toward shore from water farther out; never seen by the satellite here; not checkable",
  3: "Bay estimate from a borrowed background; turbid water",
  unknown: "Estimated; near shore it may be extrapolated",
} as const;

/** A class's words: the published legend verbatim; for the shoreline strip, said to be uncheckable when the legend does not say so. */
export function classWords(l: Pick<EstimateLayer, "class_legend">, k: 1 | 2 | 3 | null): string {
  if (!k) return ESTIMATE_WORDS.unknown;
  const pub = l.class_legend?.[k];
  if (!pub) return ESTIMATE_WORDS[k];
  if (k === 2) {
    const add = [!/never (seen|observed)/i.test(pub) ? "never seen by the satellite here" : null, !/check/i.test(pub) ? "not checkable" : null].filter(Boolean);
    return add.length ? `${pub.replace(/[.\s]+$/, "")}; ${add.join("; ")}` : pub;
  }
  if (k === 3 && !/turbid/i.test(pub)) return `${pub.replace(/[.\s]+$/, "")}; turbid water`;
  return pub;
}

/** The estimate at a point, read only where the observation has no value. */
export async function estimateRead(baseUrl: string, l: EstimateLayer, lat: number, lon: number): Promise<Read | null> {
  const [r, cls] = await Promise.all([derivedAt(baseUrl, l, lat, lon), estimateClassAt(baseUrl, l, lat, lon).catch(() => null)]);
  if (r.value == null) return null;
  const days = [...l.based_on_days].sort();
  const span = days.length ? (days[0] === days[days.length - 1] ? dayLabel(days[0]) : `${dayLabel(days[0])}–${dayLabel(days[days.length - 1])}`) : dayLabel(l.issued);
  // without a class (no class grid, or none for this pixel) no estimate is said to be a fill; a bay is still said to be turbid (lib/bays)
  const k = cls;
  const bay = k !== 3 && bayAt(lat, lon) ? BAY_TURBID : undefined;
  return { kind: "model", value: fmtMg(r.value), unit: "mg/m³", what: `${classWords(l, k)}${rangeWords(r.range)}`, when: `Neritic estimate from views of ${span}`, label: k === 2 ? "Extrapolated, not observed" : "Estimated, not observed", tone: k === 2 ? "strip" : "estimate", flag: bay };
}

/** Observed satellite chlorophyll in a bay is approximate (turbid water): said beside the value. */
export function withBayCaveat(r: Readout, lat: number, lon: number): Readout {
  if (r.value == null || !bayAt(lat, lon)) return r;
  return { ...r, flag: r.flag ? `${r.flag} · ${BAY_TURBID}` : BAY_TURBID };
}

/**
 * What the map shows at a point for the C-HARM layer (v5): the published value; or, where the cell
 * has none and the map tints the water ("NOAA publishes no toxin probability here", by variable),
 * the nearest NOAA cell within 6 km as text only, and the pipeline's pier-sample check of that
 * nearest value. `water` says whether the point is water (null: unknown, read as water); over
 * land the plain published message stays.
 */
export async function forecastRead(manifest: Manifest, baseUrl: string, l: LayerArtifact, lat: number, lon: number, context: boolean, water: boolean | null, check: DerivedValidationEntry | null = null): Promise<Read> {
  const r = await forecastAt(baseUrl, l, lat, lon, pnSibling(manifest, l));
  if (r.value != null || !context || water === false) return r;
  const c = await forecastContext(baseUrl, l, lat, lon);
  if (c.kind === "value") return r;
  // the toxin mask keeps its reason (the same run has a Pseudo-nitzschia value in this cell)
  const why = r.none === NO_FORECAST.toxinMask ? ` ${NO_FORECAST.toxinMask}` : "";
  const checked = c.kind === "context" && check ? ` ${checkSentence(check)}` : "";
  return { ...r, none: (c.kind === "context" ? contextWords(c, l.variable) : outsideWords(l.variable)) + why + checked, tone: c.kind === "context" ? "context" : "outside", label: outsideLabel(l.variable) };
}

/**
 * Desktop hover readout (M5): the exact published value of the cell under the pointer, with
 * its date and source, in a small label beside the cursor. The latest pointer position wins;
 * grids are fetched once and cached. Click opens the full inspector, which is also the
 * keyboard and screen-reader route to the same values, so this label is aria-hidden.
 */
export function HoverReadout({ manifest, baseUrl, source, point, screen, bounds, waterAt }: { manifest: Manifest; baseUrl: string; source: HoverSource; point: { lat: number; lon: number } | null; screen: { x: number; y: number } | null; bounds: { w: number; h: number }; waterAt?: (screen: { x: number; y: number }) => boolean | null }) {
  const [r, setR] = useState<Read | null>(null);
  const seq = useRef(0);
  useEffect(() => {
    const n = ++seq.current;
    if (!source || !point) {
      setR(null);
      return;
    }
    const run = async (): Promise<Read> => {
      if (source.kind === "forecast") return forecastRead(manifest, baseUrl, source.layer, point.lat, point.lon, !!source.context, source.context && screen && waterAt ? waterAt(screen) : null, source.check);
      if (source.kind === "outlook") return outlookRead(baseUrl, source.layer, point.lat, point.lon);
      if (source.kind === "sst") return sstRead(baseUrl, source.layer, point.lat, point.lon);
      if (source.kind === "satellite") {
        const obs = withBayCaveat(await satelliteAt(manifest, baseUrl, source.layer, point.lat, point.lon), point.lat, point.lon);
        // an observed pixel is always what is read; the estimate only where nothing was observed
        if (obs.value != null || !source.estimate || (screen && waterAt?.(screen) === false)) return obs;
        return (await estimateRead(baseUrl, source.estimate, point.lat, point.lon)) ?? obs;
      }
      return currentsAt(source.field, source.layer, point.lat, point.lon);
    };
    run()
      .then((x) => n === seq.current && setR(x))
      .catch(() => n === seq.current && setR(null));
    // the screen position follows the point (they arrive together)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [manifest, baseUrl, source, point]);

  if (!r || !screen || !point) return null;
  const W = 264;
  const left = screen.x + 16 + W > bounds.w ? screen.x - 16 - W : screen.x + 16;
  const top = Math.min(Math.max(8, screen.y + 16), bounds.h - 120);
  return (
    <div
      aria-hidden
      data-testid="hover-readout"
      data-tone={r.tone}
      className="pointer-events-none absolute z-30 rounded-lg bg-navy-950/[0.93] px-2.5 py-1.5 text-[13px] leading-snug text-on-navy-2 shadow-[0_6px_18px_rgba(4,11,23,0.4)] ring-1 ring-white/10 backdrop-blur-sm"
      style={{ left, top, width: W }}
    >
      {r.label && (
        <p className={`text-[11.5px] font-semibold uppercase tracking-[0.06em] ${r.tone === "outside" || r.tone === "context" ? "text-white/75" : "text-[#cbbcf7]"}`} data-testid="hover-label">
          {r.label}
        </p>
      )}
      {r.value != null ? (
        <>
          <p className="flex items-baseline gap-1.5 text-white">
            <span className="text-[16px] font-semibold tabular" data-testid="hover-value">
              {r.value}
            </span>
            {r.unit && <span className="text-[13px] text-on-navy-2">{r.unit}</span>}
          </p>
          <p className="text-[13px] leading-snug text-white/85">{r.what}</p>
          {r.flag && (
            <p className="text-[13px] leading-snug text-[#f6c87a]" data-testid="hover-flag">
              {r.flag}
            </p>
          )}
        </>
      ) : (
        <p className="text-[13px] text-white" data-testid={r.tone === "context" ? "hover-context" : r.tone === "outside" ? "hover-outside" : r.tone === "bay" ? "hover-bay" : "hover-none"}>
          {r.none}
          {r.tone === "context" || r.tone === "outside" ? " Click for pier samples nearby and the filled chlorophyll." : ""}
        </p>
      )}
      <p className="text-[13px] leading-snug" data-testid="hover-when">
        <span aria-hidden className={`mr-1 inline-block h-1.5 w-1.5 -translate-y-px rounded-full ${r.kind === "model" ? "bg-[#b9a6f5]" : "bg-[#7fd4c1]"}`} />
        {r.when}
      </p>
      <p className="text-[13px] text-on-navy-2/70 tabular">
        {point.lat.toFixed(3)}°N {Math.abs(point.lon).toFixed(3)}°W · click for details
      </p>
    </div>
  );
}
