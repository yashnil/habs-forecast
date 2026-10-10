import type { ForecastRun, LayerArtifact } from "@/generated/schema";
import { hoursAgo } from "@/lib/currents";
import { formatDate, formatDateTimePT } from "@/lib/time";

export type Stamp = { kind: "observation" | "model" | "gap"; text: string; testid: string };
const short = (d: string) => formatDate(d).replace(/^\w+, /, ""); // "Oct 8"

/** "Sentinel-3 300 m" or "VIIRS 750 m": the sensor and native resolution of a satellite layer. */
export function sensorLabel(l: LayerArtifact): string {
  const name = l.platforms?.some((p) => p.startsWith("Sentinel-3")) ? "Sentinel-3" : l.platforms?.includes("VIIRS") ? "VIIRS" : null;
  const res = l.native_resolution_m ? `${Math.round(l.native_resolution_m)} m` : null;
  return [name, res].filter(Boolean).join(" ") || l.short_title.replace(/^Chlorophyll · /, "");
}

/** Multi-sensor view: Sentinel-3 300 m where it has a recent pixel, VIIRS 750 m elsewhere. */
function multiSensorLabel(l: LayerArtifact): string {
  const p = l.platforms ?? [];
  const parts = [p.some((x) => x.startsWith("Sentinel-3")) && "Sentinel-3 300 m", p.includes("VIIRS") && "VIIRS 750 m"].filter(Boolean);
  return parts.length ? parts.join(" + ") : "multi-sensor";
}

/** One line per layer on the map: what it is and when it was observed or is valid for. */
export function stampLines(p: {
  group: "forecast" | "satellite" | "currents";
  forecast?: LayerArtifact | null;
  run?: ForecastRun | null;
  satellite?: LayerArtifact | null;
  imagery?: LayerArtifact | null;
  currents?: LayerArtifact | null;
  combined?: boolean;
  now?: Date | null;
  region?: { id: string; label: string } | null;
}): Stamp[] {
  const out: Stamp[] = [];
  if (p.group === "forecast" && p.forecast?.time.valid_date) {
    const lead = p.forecast.time.lead_days ?? 0;
    out.push({
      kind: "model",
      testid: "stamp-forecast",
      text: `C-HARM ${lead === 0 ? "nowcast" : "forecast"} for ${formatDate(p.forecast.time.valid_date)}${p.run ? ` · issued ${short(p.run.issued_date)}` : ""}`,
    });
  }
  const sat = p.satellite;
  if ((p.group === "satellite" || p.combined) && sat) {
    const ms = sat.multisensor;
    const comp = sat.composite;
    const text = ms
      ? `Chlorophyll, ${multiSensorLabel(sat)} · newest pixel ${short(sat.time.observed_date ?? "")}; each pixel has its own date`
      : comp
        ? `Chlorophyll, ${sensorLabel(sat)} · pixels observed ${short(comp.oldest_observed_date)}–${short(comp.newest_observed_date)}`
        : sat.time.observed_date
          ? `Chlorophyll, ${sensorLabel(sat)} · overpass ${formatDate(sat.time.observed_date)}`
          : "Chlorophyll";
    out.push({ kind: "observation", testid: "stamp-satellite", text });
  }
  if (p.group === "satellite" && p.imagery?.time.observed_date && !sat) {
    out.push({ kind: "observation", testid: "stamp-imagery", text: `NASA GIBS imagery · ${formatDate(p.imagery.time.observed_date)}` });
  }
  const c = p.currents;
  if (p.combined && sat && c) {
    const gap = combinedGapDays(sat, c);
    if (gap != null && gap > 0) out.push({ kind: "gap", testid: "stamp-combined-gap", text: `Different times: chlorophyll a median ${gap} day${gap === 1 ? "" : "s"} older than the currents` });
  }
  if (p.group === "currents" && c) {
    const mean = c.layer_id.endsWith("mean24h");
    const t = mean ? (c.time.observed_times ?? []).at(-1) : c.time.valid_time;
    out.push({
      kind: "observation",
      testid: "stamp-currents",
      text: t
        ? `Currents, HF radar · ${mean ? `24 h mean to ${formatDateTimePT(t)}` : `observed ${formatDateTimePT(t)}`}${p.now ? ` (${hoursAgo(t, p.now)} h ago)` : ""}`
        : "Currents, HF radar",
    });
    // missing coverage is said on the map, not only in the dock
    // a region absent from the layer's coverage lies outside its domain: no observations either
    const rc = p.region ? c.coverage?.regions.find((r) => r.region_id === p.region!.id) : undefined;
    if (p.region && c.coverage && (!rc || rc.observed_fraction === 0)) {
      out.push({ kind: "gap", testid: "stamp-currents-gap", text: `No radar observations in ${p.region!.label} ${mean ? "for the 24 h mean" : "this hour"}: no data, not calm water` });
    }
  }
  return out;
}

/** Median observation date of a latest-clear-view composite (from its age histogram). */
export function medianObserved(l: LayerArtifact | null | undefined): string | null {
  const c = l?.composite;
  if (!c || !c.age_histogram.length) return null;
  let acc = 0;
  const bin = [...c.age_histogram].sort((a, b) => a.age_days - b.age_days).find((b) => (acc += b.fraction) >= 0.5) ?? c.age_histogram[c.age_histogram.length - 1];
  return new Date(Date.parse(`${c.reference_date}T00:00:00Z`) - bin.age_days * 86_400_000).toISOString().slice(0, 10);
}

/** Whole days from the chlorophyll's median observation date to the currents' date. */
export function combinedGapDays(chl: LayerArtifact | null | undefined, currents: LayerArtifact | null | undefined): number | null {
  const t = currents?.layer_id.endsWith("mean24h") ? (currents.time.observed_times ?? []).at(-1) : currents?.time.valid_time;
  const med = medianObserved(chl);
  if (!t || !med) return null;
  return Math.round((Date.parse(`${t.slice(0, 10)}T00:00:00Z`) - Date.parse(`${med}T00:00:00Z`)) / 86_400_000);
}
