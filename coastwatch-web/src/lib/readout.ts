import type { LayerArtifact, Manifest } from "@/generated/schema";
import { compass, type CurrentField, sampleField } from "@/lib/currents";
import { artifactUrl } from "@/lib/layers";
import { cellAt, loadGrid, sampleChunked, valueOf } from "@/lib/grid";
import { multiSensorMembers, multiSensorPick, observedDate, type SensorObs } from "@/lib/multisensor";
import { sensorLabel } from "@/lib/stamp";
import { formatDate, formatDateTimePT } from "@/lib/time";

/**
 * Exact value under a point (M5 hover and click readout): the published cell that contains
 * the point, with its own date and source. Never the nearest cell, never interpolated; where
 * the cell has no value the readout says so and why it may be missing.
 */
export type Readout = { kind: "model" | "observation"; value: string | null; unit: string; what: string; when: string; none?: string };

const short = (d: string) => formatDate(d).replace(/^\w+, /, "");
export const fmtChl = (v: number) => (v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(2));

export async function forecastAt(baseUrl: string, l: LayerArtifact, lat: number, lon: number): Promise<Readout> {
  const what = (l.threshold_text ?? l.short_title).replace(/^Probability (that|of) /i, "chance ");
  const when = `C-HARM ${l.time.lead_days ? "forecast" : "nowcast"} · valid ${short(l.time.valid_date ?? "")}`;
  if (!l.grid) return { kind: "model", value: null, unit: "", what, when, none: "No forecast grid published" };
  const codes = await loadGrid(artifactUrl(baseUrl, l.grid.url));
  const c = cellAt(l.grid, lat, lon);
  const v = c ? valueOf(l.grid, codes, c) : null;
  if (v == null) return { kind: "model", value: null, unit: "", what, when, none: "No forecast value: land, or a nearshore cell C-HARM does not cover" };
  return { kind: "model", value: `${Math.round(v * 100)}%`, unit: "", what, when };
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
    return { kind: "observation", value: fmtChl(obs.value), unit: "mg/m³", what, when: `${sensorLabel(src)} · observed ${short(obs.date)}` };
  }
  if (!l.grid?.chunks) return { kind: "observation", value: null, unit: "", what, when: `${sensorLabel(l)} · ${short(l.time.observed_date ?? "")}`, none: NO_OBS };
  const s = await sampleChunked(baseUrl, l.grid, lat, lon, l.composite?.age_grid ?? null, 0);
  const date = l.composite ? (s.kind === "value" && s.ageDays != null ? observedDate(l.composite.reference_date, s.ageDays) : null) : (l.time.observed_date ?? null);
  const when = `${sensorLabel(l)}${date ? ` · observed ${short(date)}` : l.composite ? `, last ${l.composite.window_days} days` : ""}`;
  if (s.kind !== "value") return { kind: "observation", value: null, unit: "", what, when, none: NO_OBS };
  return { kind: "observation", value: fmtChl(s.value), unit: "mg/m³", what, when };
}

export function currentsAt(f: CurrentField, l: LayerArtifact, lat: number, lon: number): Readout {
  const mean = l.layer_id.endsWith("mean24h");
  const t = mean ? (l.time.observed_times ?? []).at(-1) : l.time.valid_time;
  const when = `HF radar · ${mean ? "24-hour mean to " : "observed "}${t ? formatDateTimePT(t) : "—"}`;
  const s = sampleField(f, lat, lon, 0);
  if (!s) return { kind: "observation", value: null, unit: "", what: "surface current", when, none: "No radar observation here. No data, not calm water." };
  return { kind: "observation", value: s.speed.toFixed(2), unit: "m/s", what: `toward ${compass(s.dir)} (${Math.round(s.dir)}°)`, when };
}
