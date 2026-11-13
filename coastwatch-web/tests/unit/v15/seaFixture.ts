import { readFileSync } from "node:fs";
import path from "node:path";
import type { CoastSea, SeaPeriod } from "@/nx/coast/brief";
import { datePeriods } from "@/lib/nwsPeriods";

/** The recorded NWS zone forecasts (tests/conditions-fixtures.json), by zone id. Fixed clock 2026-10-13T06:00:00Z. */
export const NOW = "2026-10-13T06:00:00Z";
const raw = JSON.parse(readFileSync(path.resolve(__dirname, "../../conditions-fixtures.json"), "utf8")) as Record<string, unknown>;

type Zf = { properties: { zone: string; updated: string; periods: { name: string; detailedForecast: string }[] } };
export function recordedZone(zone: string): { updated: string; periods: { name: string; text: string }[] } {
  for (const v of Object.values(raw)) {
    const s = typeof v === "string" ? v : JSON.stringify(v);
    if (!s.includes("detailedForecast") || !s.includes(`/${zone}"`)) continue;
    const o = JSON.parse(s) as Zf;
    return { updated: o.properties.updated, periods: o.properties.periods.map((p) => ({ name: p.name, text: p.detailedForecast })) };
  }
  throw new Error(`no recorded forecast for ${zone}`);
}

/** A CoastSea as /coast/data?part=sea would answer, from periods (dated by lib/nwsPeriods, as the route does). */
export function seaOf(zone: string, updated: string, periods: { name: string; text: string }[], over: Partial<CoastSea["sea"]> = {}): CoastSea {
  const lanes: SeaPeriod[] = datePeriods(periods, updated).map((p) => ({ name: p.name, date: p.date, night: p.night, text: p.text, wind: null, seasLo: null, seasHi: null }));
  return {
    ok: true,
    now: NOW,
    sea: { state: "ok", lanes, zone, issued: updated, marineUrl: `https://api.weather.gov/zones/forecast/${zone}/forecast`, marineReason: null, alerts: [], buoy: null, buoyReason: null, windBuoy: null, tides: [], tideStation: null, tideReason: null, water: null, ...over },
  };
}

export const recordedSea = (zone: string) => {
  const z = recordedZone(zone);
  return seaOf(zone, z.updated, z.periods);
};
