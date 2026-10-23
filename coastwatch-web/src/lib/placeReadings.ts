import { stationName, type Conditions } from "@/lib/conditions";

/**
 * What is measured at a place now, for My Coast's front door (v8): water temperature, waves and wind,
 * each with its own instrument and its own observation time. Two instruments can feed one card (the
 * harbour's tide gauge for water, the nearest buoy for waves and wind) and a buoy's fields report at
 * different times, so no reading borrows another's source or age. Where there is no value the reading
 * says why, in a few words, instead of a bare dash. Observations only: never a forecast.
 */

/** One instrument: `short` sits under a value ("Gauge", "Buoy"); `full` names it once in the card's source line. */
export type ReadingSource = { kind: "gauge" | "buoy"; short: string; full: string };
export type Reading<T> = ({ ok: true; at: string; source: ReadingSource } & T) | { ok: false; why: string; source: ReadingSource | null };
export type PlaceReadings = {
  water: Reading<{ c: number }>;
  waves: Reading<{ ft: number; s: number | null }>;
  wind: Reading<{ kt: number; dir: number | null }>;
  /** each instrument used, once, in the order the readings name them */
  sources: ReadingSource[];
};

const km = (k: number) => (k < 10 ? k.toFixed(1) : String(Math.round(k)));

export function placeReadings(c: Conditions): PlaceReadings {
  const buoyPart = c.buoy;
  const buoy = buoyPart.status === "ok" ? buoyPart : null;
  const buoySrc: ReadingSource | null = buoy
    ? { kind: "buoy", short: "Buoy", full: buoy.station ? `${stationName(buoy.station.name)} buoy (NDBC ${buoy.station.id}), ${km(buoy.station.km)}\u00a0km` : "NDBC buoy" }
    : null;
  // why a buoy reading is missing: no buoy within reach, its feed not answering, or this buoy not reporting it
  const noBuoy = buoyPart.status === "none" ? "No buoy nearby" : "Buoy not reached";
  const missing = (src: ReadingSource | null) => ({ ok: false as const, why: src ? "Not reported" : noBuoy, source: src });

  const h = c.harbor?.status === "ok" ? c.harbor : null;
  const gaugeSrc: ReadingSource | null = h
    ? { kind: "gauge", short: "Tide gauge", full: h.station ? `${stationName(h.station.name)} tide gauge (NOAA CO-OPS), ${km(h.station.km)}\u00a0km` : "NOAA CO-OPS tide gauge" }
    : null;

  // water: the harbour's own gauge while it is current, else the buoy, else the gauge's last reading with its true age
  let water: PlaceReadings["water"];
  if (h && gaugeSrc && !h.data.stale) water = { ok: true, c: h.data.waterC.value, at: h.data.waterC.at, source: gaugeSrc };
  else if (buoy?.data.waterC && buoySrc) water = { ok: true, c: buoy.data.waterC.value, at: buoy.data.waterC.at, source: buoySrc };
  else if (h && gaugeSrc) water = { ok: true, c: h.data.waterC.value, at: h.data.waterC.at, source: gaugeSrc };
  else water = buoySrc ? missing(buoySrc) : { ok: false, why: buoyPart.status === "none" ? "Not measured nearby" : "Buoy not reached", source: null };

  const w = buoy?.data.waveFt;
  const waves: PlaceReadings["waves"] = w && buoySrc ? { ok: true, ft: w.value, s: buoy!.data.periodS?.value ?? null, at: w.at, source: buoySrc } : missing(buoySrc);
  const k = buoy?.data.windKt;
  const wind: PlaceReadings["wind"] = k && buoySrc ? { ok: true, kt: k.value, dir: buoy!.data.windDirDeg?.value ?? null, at: k.at, source: buoySrc } : missing(buoySrc);

  const sources: ReadingSource[] = [];
  for (const r of [water, waves, wind]) if (r.ok && !sources.includes(r.source)) sources.push(r.source);
  return { water, waves, wind, sources };
}
