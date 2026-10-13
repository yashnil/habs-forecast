import type { Official, OfficialRecord, Station } from "../data/types";
import { NOW } from "../lib/time";
import { fmtValue } from "./scale";

export type QuantityId = "pDA" | "pn" | "chl" | "temp";
export interface Quantity {
  id: QuantityId;
  variable: string;
  label: string;          // switcher label
  tiny: string;           // switcher label on narrow screens
  short: string;          // readout label
  units: string;          // symbol
  unitsWords: string;     // axis title
  scale: "log" | "linear";
  model?: { key: "particulate_domoic" | "pseudo_nitzschia"; label: string };
}
export const QUANTITIES: Quantity[] = [
  { id: "pDA", variable: "pDA", label: "Toxin in seawater", tiny: "Toxin", short: "Particulate domoic acid", units: "ng/mL", unitsWords: "nanograms per millilitre (log scale)", scale: "log",
    model: { key: "particulate_domoic", label: "C-HARM chance particulate domoic acid exceeds 500 ng/L" } },
  { id: "pn", variable: "pn_seriata", label: "Pseudo-nitzschia", tiny: "Algae", short: "Pseudo-nitzschia, larger cells", units: "cells/L", unitsWords: "cells per litre (log scale)", scale: "log",
    model: { key: "pseudo_nitzschia", label: "C-HARM chance Pseudo-nitzschia exceeds 10,000 cells/L" } },
  { id: "chl", variable: "chl_extracted", label: "Chlorophyll", tiny: "Chlorophyll", short: "Chlorophyll-a (extracted)", units: "mg/m³", unitsWords: "milligrams per cubic metre", scale: "linear" },
  { id: "temp", variable: "temp", label: "Temperature", tiny: "Temp.", short: "Water temperature", units: "°C", unitsWords: "degrees Celsius", scale: "linear" },
];

export type WindowId = "3m" | "12m" | "5y" | "all";
export const WINDOWS: { id: WindowId; label: string; days: number | null }[] = [
  { id: "3m", label: "3 months", days: 92 },
  { id: "12m", label: "12 months", days: 365 },
  { id: "5y", label: "5 years", days: 1826 },
  { id: "all", label: "All", days: null },
];

export const DAY = 86_400_000;
export const GAP_DAYS = 21;

export interface Sample {
  key: string;          // stable across quantity and window changes
  t: number;
  date: string;
  v: Record<QuantityId, number | null>;
  zero: Record<QuantityId, boolean>;
}
export interface StationWithModel extends Station {
  charm?: { radius_km: number; history: Record<string, { date: string; value: number; n: number }[]> };
}

export function samplesOf(st: Station): Sample[] {
  const ser = (variable: string) => st.series.find((s) => s.variable === variable);
  const out: Sample[] = [];
  st.sample_times.forEach((iso, i) => {
    const v = {} as Sample["v"], zero = {} as Sample["zero"];
    for (const q of QUANTITIES) {
      const s = ser(q.variable);
      const x = s?.values[i] ?? null;
      v[q.id] = x;
      zero[q.id] = x === 0; // published as "reported_zero": at or below the lab's limit, not absent
    }
    out.push({ key: `${iso}#${i}`, t: Date.parse(iso), date: iso.slice(0, 10), v, zero });
  });
  return out;
}

export function windowDomain(samples: Sample[], w: WindowId): [number, number] {
  const end = NOW.getTime();
  const days = WINDOWS.find((x) => x.id === w)!.days;
  return [days ? end - days * DAY : samples[0].t - 7 * DAY, end];
}

/** Notices whose official geography includes the station (statewide, or a latitude band around it). */
export function stationNotices(off: Official | undefined, st: Station): OfficialRecord[] {
  if (!off) return [];
  return off.registry.records.filter((r) => {
    if (r.status !== "active" || !r.effective_date) return false;
    if (r.area.type === "statewide") return true;
    if (r.area.type === "lat_band" && r.area.lat_north != null && r.area.lat_south != null) return st.lat <= r.area.lat_north && st.lat >= r.area.lat_south;
    return false;
  });
}
export function shortNotice(r: OfficialRecord) {
  const sp = r.species[0] ?? "";
  const what = r.action === "take_restriction" ? "take restriction" : r.action === "quarantine" ? "quarantine" : r.action === "consumption_advisory" ? "advisory" : r.action.replace(/_/g, " ");
  const species = sp.replace(/\s*\(.*\)\s*/g, "").replace(/^(northern|all) /, "").replace(/s$/, "");
  return `${r.agency} ${species} ${what}`.replace(/\s+/g, " ");
}

/* ------------------------------------------------------------------ story */
export interface Step {
  id: string;
  title: string;
  body: string;
  quantity: QuantityId;
  window: WindowId;
  highlight: { samples?: string[]; rules?: string[]; tailGap?: boolean; model?: boolean };
}
const fmtDate = (iso: string, year = true) =>
  new Date(iso + "T12:00:00Z").toLocaleDateString("en-US", { month: "short", day: "numeric", year: year ? "numeric" : undefined, timeZone: "UTC" });

export function storySteps(samples: Sample[], notices: OfficialRecord[]): Step[] {
  const end = NOW.getTime();
  const yearAgo = end - 365 * DAY;
  const steps: Step[] = [];
  const lastYear = samples.filter((s) => s.t >= yearAgo);
  const peak = lastYear.filter((s) => s.v.pDA != null).sort((a, b) => b.v.pDA! - a.v.pDA!)[0];
  if (peak) {
    const before = samples.filter((s) => s.t >= peak.t - 60 * DAY && s.t <= peak.t + 14 * DAY && s.v.pn != null).sort((a, b) => b.v.pn! - a.v.pn!)[0];
    steps.push({
      id: "peak",
      title: `The year's highest toxin: ${fmtDate(peak.date)}`,
      body: `Particulate domoic acid in the pier's water reached ${fmtValue(peak.v.pDA!)} ng/mL, the highest of ${lastYear.filter((s) => s.v.pDA != null).length} samples in the last 12 months.`,
      quantity: "pDA", window: "12m", highlight: { samples: [peak.key] },
    });
    if (before && before.v.pn! > 0) {
      const dd = Math.round((peak.t - before.t) / DAY);
      steps.push({
        id: "cells",
        title: "The algae that can make it",
        body: `Pseudo-nitzschia (larger cells) peaked at ${fmtValue(before.v.pn!)} cells/L on ${fmtDate(before.date, false)}, ${dd === 0 ? "in the same sample" : dd > 0 ? `${dd} days before the toxin peak` : `${-dd} days after it`}. More cells do not always mean more toxin.`,
        quantity: "pn", window: "12m", highlight: { samples: [before.key, peak.key] },
      });
    }
  }
  const recent = notices.filter((r) => r.effective_date && Date.parse(r.effective_date) >= end - 92 * DAY).sort((a, b) => a.effective_date!.localeCompare(b.effective_date!));
  if (recent.length) {
    const first = recent[0];
    const near = samples.filter((s) => s.v.pDA != null && s.t <= Date.parse(first.effective_date!) + DAY).at(-1);
    steps.push({
      id: "notices",
      title: recent.length > 1 ? "Two agencies act" : "An agency acts",
      body: `${recent.map((r) => `${r.agency} on ${fmtDate(r.effective_date!, false)}: ${r.title.charAt(0).toLowerCase()}${r.title.slice(1)}`).join(". ")}.${near ? ` The pier sample just before read ${fmtValue(near.v.pDA!)} ng/mL (${fmtDate(near.date, false)}).` : ""} Agencies decide from seafood testing, not from these pier samples.`,
      quantity: "pDA", window: "3m", highlight: { rules: recent.map((r) => r.id), samples: near ? [near.key] : [] },
    });
  }
  const last = samples.filter((s) => s.v.pDA != null).at(-1);
  if (last) {
    const age = Math.round((end - last.t) / DAY);
    steps.push({
      id: "latest",
      title: `Latest sample: ${fmtDate(last.date, false)}`,
      body: `${fmtValue(last.v.pDA!)} ng/mL particulate domoic acid, ${age} days ago.${last.v.pn != null ? ` Pseudo-nitzschia ${fmtValue(last.v.pn)} cells/L.` : ""}`,
      quantity: "pDA", window: "3m", highlight: { samples: [last.key] },
    });
    steps.push({
      id: "unknown",
      title: "What isn't known",
      body: `Nothing has been measured here for ${age} days. A pier sample describes the water at this pier on that day. It is not a measurement of seafood, and it says nothing about offshore waters.`,
      quantity: "pDA", window: "3m", highlight: { tailGap: true },
    });
  }
  return steps;
}
