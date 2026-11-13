import type { CoastGates, CoastSea } from "@/nx/coast/brief";
import { takeSnapshot, type Answer } from "@/nx/trips/snapshot";
import type { Snapshot, Trip } from "@/nx/trips/types";

/** Agent D's fixtures: a Moss Landing commercial Dungeness trip and the two My Coast answers, shaped like the endpoints'. */
export const PLAN_AT = "2026-10-13T06:00:00Z";
export const LATER = "2026-10-13T12:00:00Z";
export const CDFW_URL = "https://wildlife.ca.gov/Conservation/Marine/Invertebrates/Crabs/Dungeness";
export const CDPH_URL = "https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Domoic-Acid.aspx";
export const ALERT_URL = "https://api.weather.gov/alerts/urn:oid:2.49.0.1.840.0.abc.003.1";

export const clone = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T;

export const trip: Trip = {
  schema: 1,
  id: "t1",
  name: "Moss Landing, Dungeness",
  autoName: true,
  activity: "commercial",
  fishery: "dungeness_commercial",
  placeKey: "port-592",
  departKey: null,
  depart: { date: "2026-10-15", time: null, timeZone: "America/Los_Angeles" },
  area: null,
  vesselId: null,
  limits: { windKt: 20, seasFt: 8 },
  economics: null,
  refs: [],
  history: [],
  notes: "",
  state: "planned",
  createdAt: PLAN_AT,
  updatedAt: PLAN_AT,
  snapshots: { planning: null, rechecks: [], superseded: [] },
  outcome: null,
};

export const gates = (): CoastGates => ({
  ok: true,
  now: PLAN_AT,
  fixture: true,
  fishery: "dungeness_commercial",
  season: {
    text: "Commercial season delayed to at least Jan 1, 2027.",
    tone: "regulation",
    level: "delayed",
    conflict: false,
    seasonal: false,
    source: "CDFW",
    url: CDFW_URL,
    checked: "2026-10-12T18:00:00Z",
    verified: false,
    stale: false,
    differ: null,
  },
  whale: { zone: "3", area: "Pigeon Pt to Lopez Pt", measures: "Fleet Advisory. No depth limit or gear reduction on record.", next: "Dec 15–18, subject to change", assessed: "2026-10-01", url: "https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries", verified: false },
  test: { site: "Monterey Bay", species: "Dungeness crab", date: "2026-10-02", tissue: "viscera", raw: ["<2.5", "9.8"], values: [null, 9.8], max: 9.8, avg: 9.8, over: 0, action: 30, ageDays: 11, stale: false, source: "CDPH", url: CDPH_URL, lat: null, lon: null },
  statusError: false,
  testsError: false,
});

export const sea = (): CoastSea => ({
  ok: true,
  now: PLAN_AT,
  sea: {
    state: "ok",
    lanes: [
      { name: "Tuesday", date: "2026-10-13", night: false, text: "NW winds 10 to 15 kt. Seas 5 ft.", wind: "NW 10–15 kt", seasLo: 5, seasHi: 5 },
      { name: "Wednesday", date: "2026-10-14", night: false, text: "NW winds 10 to 15 kt. Seas 5 to 6 ft.", wind: "NW 10–15 kt", seasLo: 5, seasHi: 6 },
    ],
    zone: "PZZ535",
    issued: "2026-10-13T02:31:00-07:00",
    marineUrl: "https://forecast.weather.gov/shmrn.php?mz=pzz535",
    marineReason: null,
    alerts: [{ event: "Small Craft Advisory", headline: "Small craft advisory until Tuesday 3 PM", url: ALERT_URL, ends: "2026-10-13T22:00:00Z" }],
    buoy: { name: "Monterey Bay buoy", id: "46042", km: 30, url: "https://www.ndbc.noaa.gov/station_page.php?station=46042", windKt: 12, windDir: "NW", gustKt: 15, waveFt: 5.2, periodS: 9, waterC: 14, at: "2026-10-13T05:40:00Z", lat: 36.8, lon: -122.4 },
    buoyReason: null,
    windBuoy: null,
    tides: [{ at: "2026-10-13 05:12", ft: 4.9, type: "H" }],
    tideStation: { id: "9413450", name: "Monterey", url: "https://tidesandcurrents.noaa.gov/stationhome.html?id=9413450" },
    tideReason: null,
    water: null,
  },
});

export const snap = (o: { gates?: Answer<CoastGates>; sea?: Answer<CoastSea>; now?: string; kind?: Snapshot["kind"]; id?: string } = {}): Snapshot =>
  takeSnapshot({ trip, gates: o.gates === undefined ? gates() : o.gates, sea: o.sea === undefined ? sea() : o.sea, forecast: null, now: o.now ?? PLAN_AT, mode: "live", savedFrom: null, kind: o.kind ?? "planning", id: o.id ?? "plan-1" });
