import type { LayerArtifact, Manifest } from "@/generated/schema";
import { compass, type CurrentField, sampleField } from "@/lib/currents";
import { CHL_NEAR_COPY } from "@/content/copy";
import { pixelStatus } from "@/lib/chlNear";
import { artifactUrl, nativeLabel } from "@/lib/layers";
import { cellAt, loadGrid, sampleChunked, valueOf } from "@/lib/grid";
import { multiSensorMembers, multiSensorPick, observedDate, type SensorObs } from "@/lib/multisensor";
import { sensorLabel } from "@/lib/stamp";
import { formatDate, formatDateTimePT } from "@/lib/time";

/**
 * Exact value under a point (M5 hover and click readout): the published cell that contains
 * the point, with its own date and source. Never the nearest cell, never interpolated; where
 * the cell has no value the readout says so and why it may be missing.
 */
export type Readout = {
  kind: "model" | "observation";
  value: string | null;
  unit: string;
  what: string;
  /** the product, its native resolution and the observation or valid time, e.g. "C-HARM forecast, native 3 km · valid Oct 9" */
  when: string;
  none?: string;
  flag?: string;
};

const short = (d: string) => formatDate(d).replace(/^\w+, /, "");
export const fmtChl = (v: number) => (v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(2));

/** Why a C-HARM cell has no value, in the words of the published caveats. */
export const NO_FORECAST = {
  outside: "Outside the C-HARM model area",
  toxinMask:
    "No toxin probability in this cell: C-HARM does not provide toxin probabilities for many cells within about 3–6 km of shore. The same run has a Pseudo-nitzschia value here.",
  notModelled: "No forecast value: land, or water C-HARM does not model (such as enclosed bays and some cells next to the shore)",
} as const;

/** `sibling`: the same run's Pseudo-nitzschia layer, used only to say why a toxin cell is empty. */
export async function forecastAt(baseUrl: string, l: LayerArtifact, lat: number, lon: number, sibling: LayerArtifact | null = null): Promise<Readout> {
  const what = (l.threshold_text ?? l.short_title).replace(/^Probability (that|of) /i, "chance ");
  // the product and its native cell size, then the day the value is for (never the day it was read)
  const res = nativeLabel(l);
  const when = `C-HARM ${l.time.lead_days ? "forecast" : "nowcast"}${res ? `, native ${res}` : ""} · valid ${short(l.time.valid_date ?? "")}`;
  if (!l.grid) return { kind: "model", value: null, unit: "", what, when, none: "No forecast grid published" };
  const codes = await loadGrid(artifactUrl(baseUrl, l.grid.url));
  const c = cellAt(l.grid, lat, lon);
  if (!c) return { kind: "model", value: null, unit: "", what, when, none: NO_FORECAST.outside };
  const v = valueOf(l.grid, codes, c);
  if (v == null) {
    if (sibling?.grid && sibling.layer_id !== l.layer_id) {
      const sc = cellAt(sibling.grid, lat, lon);
      const sv = sc ? valueOf(sibling.grid, await loadGrid(artifactUrl(baseUrl, sibling.grid.url)), sc) : null;
      if (sv != null) return { kind: "model", value: null, unit: "", what, when, none: NO_FORECAST.toxinMask };
    }
    return { kind: "model", value: null, unit: "", what, when, none: NO_FORECAST.notModelled };
  }
  return { kind: "model", value: `${Math.round(v * 100)}%`, unit: "", what, when };
}

/** The Pseudo-nitzschia layer of the same C-HARM run and lead as `l`, if published. */
export function pnSibling(manifest: Manifest | null, l: LayerArtifact): LayerArtifact | null {
  if (!manifest || l.variable === "pseudo_nitzschia") return null;
  return manifest.layers.find((x) => x.variable === "pseudo_nitzschia" && x.group_id === l.group_id && x.time.lead_days === l.time.lead_days && x.time.issued_date === l.time.issued_date) ?? null;
}

async function member(baseUrl: string, l: LayerArtifact | null, lat: number, lon: number): Promise<SensorObs> {
  if (!l?.grid?.chunks || !l.composite) return null;
  const s = await sampleChunked(baseUrl, l.grid, lat, lon, l.composite.age_grid, 0);
  return s.kind === "value" && s.ageDays != null ? { value: s.value, date: observedDate(l.composite.reference_date, s.ageDays) } : null;
}

const NO_OBS = "No observation: cloud, fog, land or no overpass. Not low chlorophyll.";

/** Satellite chlorophyll: a latest-clear-view composite, a single day, or the multi-sensor display. */
export async function satelliteAt(m: Manifest, baseUrl: string, l: LayerArtifact, lat: number, lon: number): Promise<Readout> {
  const what = "chlorophyll-a";
  if (l.multisensor) {
    const [p, s] = multiSensorMembers(m, l);
    const [a, b] = await Promise.all([member(baseUrl, p, lat, lon), member(baseUrl, s, lat, lon)]);
    const pick = multiSensorPick(a, b, l.multisensor.prefer_primary_within_days);
    const obs = pick === 1 ? a : pick === 2 ? b : null;
    const src = pick === 1 ? p : s;
    if (!obs || !src) return { kind: "observation", value: null, unit: "", what, when: "Multi-sensor, last 7 days", none: NO_OBS };
    return { kind: "observation", value: fmtChl(obs.value), unit: "mg/m³", what, when: `${sensorLabel(src)} · observed ${short(obs.date)}`, flag: await flagOf(baseUrl, src, lat, lon) };
  }
  if (!l.grid?.chunks) return { kind: "observation", value: null, unit: "", what, when: `${sensorLabel(l)} · ${short(l.time.observed_date ?? "")}`, none: NO_OBS };
  const s = await sampleChunked(baseUrl, l.grid, lat, lon, l.composite?.age_grid ?? null, 0);
  const date = l.composite ? (s.kind === "value" && s.ageDays != null ? observedDate(l.composite.reference_date, s.ageDays) : null) : (l.time.observed_date ?? null);
  const when = `${sensorLabel(l)}${date ? ` · observed ${short(date)}` : l.composite ? `, last ${l.composite.window_days} days` : ""}`;
  if (s.kind !== "value") return { kind: "observation", value: null, unit: "", what, when, none: NO_OBS };
  return { kind: "observation", value: fmtChl(s.value), unit: "mg/m³", what, when, flag: await flagOf(baseUrl, l, lat, lon) };
}

/**
 * The layer whose pixel a satellite readout at this point comes from: the layer itself, or for the
 * multi-sensor display the member whose observation it shows (the primary when neither observed it).
 * Its grid gives the pixel's native footprint (lib/footprint).
 */
export async function satelliteSource(m: Manifest, baseUrl: string, l: LayerArtifact, lat: number, lon: number): Promise<LayerArtifact | null> {
  if (!l.multisensor) return l;
  const [p, s] = multiSensorMembers(m, l);
  const [a, b] = await Promise.all([member(baseUrl, p, lat, lon), member(baseUrl, s, lat, lon)]);
  const pick = multiSensorPick(a, b, l.multisensor.prefer_primary_within_days);
  return pick === 2 ? s : (p ?? s);
}

/** The place rules (lib/chlNear) applied to this one pixel: why a place would not use it. */
async function flagOf(baseUrl: string, l: LayerArtifact, lat: number, lon: number): Promise<string | undefined> {
  try {
    const why = await pixelStatus(baseUrl, l, lat, lon);
    return why ? CHL_NEAR_COPY.pixel[why] : undefined;
  } catch {
    return undefined;
  }
}

export function currentsAt(f: CurrentField, l: LayerArtifact, lat: number, lon: number): Readout {
  const mean = l.layer_id.endsWith("mean24h");
  const t = mean ? (l.time.observed_times ?? []).at(-1) : l.time.valid_time;
  const res = nativeLabel(l);
  const when = `HF radar${res ? `, native ${res}` : ""} · ${mean ? "24-hour mean to " : "observed "}${t ? formatDateTimePT(t) : "—"}`;
  const s = sampleField(f, lat, lon, 0);
  if (!s) return { kind: "observation", value: null, unit: "", what: "surface current", when, none: "No radar observation here. No data, not calm water." };
  return { kind: "observation", value: s.speed.toFixed(2), unit: "m/s", what: `toward ${compass(s.dir)} (${Math.round(s.dir)}°)`, when };
}
