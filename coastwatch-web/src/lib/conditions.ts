/**
 * Ocean conditions for a place, from three official NOAA services (fetched by
 * app/api/conditions/route.ts; everything here is pure and tested):
 *
 * - NWS coastal marine zone forecast and active marine alerts (api.weather.gov): a forecast,
 *   issued by forecasters for a whole zone, shown in their words.
 * - The nearest NDBC station's latest observations (ndbc.noaa.gov realtime2): what one buoy or
 *   shore station measured, each value with its own time.
 * - CO-OPS tide predictions (high and low water) at the nearest prediction station:
 *   astronomical predictions, not observed water levels.
 *
 * Ocean conditions are context for planning a trip. They are not a toxin forecast and say
 * nothing about where fish are.
 */
import NDBC from "@/content/stations/ndbc-ca.json";
import COOPS from "@/content/stations/coops-tide-ca.json";

export type Station = { id: string; name: string; lat: number; lon: number; type?: string | null };

const KM_PER_DEG = 111.32;
export function distanceKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const dy = (lat2 - lat1) * KM_PER_DEG;
  const dx = (lon2 - lon1) * KM_PER_DEG * Math.cos(((lat1 + lat2) / 2) * (Math.PI / 180));
  return Math.hypot(dx, dy);
}

export function nearest<T extends Station>(list: T[], lat: number, lon: number, n = 3, maxKm = 80): (T & { km: number })[] {
  return list
    .map((s) => ({ ...s, km: distanceKm(lat, lon, s.lat, s.lon) }))
    .filter((s) => s.km <= maxKm)
    .sort((a, b) => a.km - b.km)
    .slice(0, n);
}

export const BUOYS = NDBC as Station[];
export const TIDE_STATIONS = COOPS as Station[];

// ---------------------------------------------------------------- NDBC realtime2 standard met

export type Obs = { value: number; at: string };
export type BuoyObs = {
  windDirDeg?: Obs;
  windKt?: Obs;
  gustKt?: Obs;
  waveFt?: Obs;
  periodS?: Obs;
  waterC?: Obs;
  airC?: Obs;
};

const MS_TO_KT = 1.943844;
const M_TO_FT = 3.28084;

/**
 * Latest valid value of each field (rows are newest first; "MM" is missing). Many stations
 * alternate rows with and without waves, so each field takes its own latest row, with its own
 * time, and nothing older than `maxAgeH` hours before `now`.
 */
export function parseNdbc(text: string, now: Date, maxAgeH = 6): BuoyObs {
  const lines = text.split("\n").filter((l) => l.trim() && !l.startsWith("#"));
  const header = text.split("\n")[0]?.replace(/^#/, "").trim().split(/\s+/) ?? [];
  const col = (name: string) => header.indexOf(name);
  const fields: [keyof BuoyObs, string, (v: number) => number][] = [
    ["windDirDeg", "WDIR", (v) => v],
    ["windKt", "WSPD", (v) => v * MS_TO_KT],
    ["gustKt", "GST", (v) => v * MS_TO_KT],
    ["waveFt", "WVHT", (v) => v * M_TO_FT],
    ["periodS", "DPD", (v) => v],
    ["waterC", "WTMP", (v) => v],
    ["airC", "ATMP", (v) => v],
  ];
  const out: BuoyObs = {};
  for (const line of lines) {
    const c = line.trim().split(/\s+/);
    if (c.length < 5) continue;
    const at = new Date(Date.UTC(+c[0], +c[1] - 1, +c[2], +c[3], +c[4]));
    if (Number.isNaN(at.getTime()) || now.getTime() - at.getTime() > maxAgeH * 3600_000 || at.getTime() - now.getTime() > 3600_000) continue;
    for (const [key, name, conv] of fields) {
      if (out[key]) continue;
      const i = col(name);
      const raw = i >= 0 ? c[i] : undefined;
      if (!raw || raw === "MM") continue;
      const v = Number(raw);
      if (!Number.isFinite(v)) continue;
      out[key] = { value: conv(v), at: at.toISOString() };
    }
  }
  return out;
}

export function compass(deg: number): string {
  const n = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];
  return n[Math.round((((deg % 360) + 360) % 360) / 22.5) % 16];
}

// ---------------------------------------------------------------- NWS marine zone

export type MarinePeriod = { name: string; text: string };
export type MarineForecast = { zone: string; zoneName: string; updated: string | null; periods: MarinePeriod[] };
export type MarineAlert = { event: string; headline: string; onset: string | null; ends: string | null; url: string };

export function parseZoneForecast(zone: string, zoneName: string, json: unknown): MarineForecast | null {
  const p = (json as { properties?: { updated?: string; periods?: { name: string; detailedForecast: string }[] } })?.properties;
  if (!p?.periods?.length) return null;
  return { zone, zoneName, updated: p.updated ?? null, periods: p.periods.slice(0, 4).map((x) => ({ name: x.name, text: x.detailedForecast })) };
}

export function parseAlerts(json: unknown): MarineAlert[] {
  const f = (json as { features?: { id: string; properties: { event: string; headline?: string; onset?: string; ends?: string; expires?: string } }[] })?.features ?? [];
  return f.map((x) => ({ event: x.properties.event, headline: x.properties.headline ?? x.properties.event, onset: x.properties.onset ?? null, ends: x.properties.ends ?? x.properties.expires ?? null, url: x.id }));
}

/** First sentence of a period ("W wind 10 to 15 kt, … Seas 3 to 5 ft.") shortened for a summary line; the full text is always shown nearby. */
export function marineHeadline(p: MarinePeriod): string {
  const wind = p.text.match(/^[^.]*?\bwind\b[^.,]*/i)?.[0];
  const seas = p.text.match(/\bSeas?\b[^.]*/)?.[0];
  return [wind, seas].filter(Boolean).join(" · ") || p.text.split(". ")[0];
}

// ---------------------------------------------------------------- CO-OPS tide predictions

export type Tide = { at: string; ft: number; type: "H" | "L" };

/** CO-OPS "hilo" predictions in local standard/daylight time ("2026-10-12 18:42"). */
export function parseTides(json: unknown): Tide[] {
  const p = (json as { predictions?: { t: string; v: string; type: string }[] })?.predictions ?? [];
  return p.filter((x) => x.type === "H" || x.type === "L").map((x) => ({ at: x.t, ft: Number(x.v), type: x.type as "H" | "L" }));
}

// ---------------------------------------------------------------- the response

export type Part<T> =
  | { status: "ok"; data: T; source: string; url: string; fetchedAt: string; station?: { id: string; name: string; km: number } }
  | { status: "none"; reason: string; source: string }
  | { status: "unavailable"; reason: string; source: string; url?: string };

export type Conditions = {
  lat: number;
  lon: number;
  marine: Part<MarineForecast>;
  alerts: Part<MarineAlert[]>;
  buoy: Part<BuoyObs>;
  tides: Part<Tide[]>;
};
