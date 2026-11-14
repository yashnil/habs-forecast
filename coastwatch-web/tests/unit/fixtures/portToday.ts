import { cpSync, mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import type { BiotoxinIndex, FisheriesStatusIndex, PortToday } from "@/pilot/contracts";
import { buildPortToday, FISHERY_OPTIONS, type CharmInput, type PortTodayInputs } from "@/pilot/facts";
import { parseBiotoxin, parseFisheriesStatus } from "@/pilot/data";

/**
 * Port facts in the REAL shape, for unit tests: built by `buildPortToday` (src/pilot/facts.ts) from the
 * data track's fixture sidecars, every fishery selected, exactly as alerts and Ask Neritic receive them
 * from src/pilot/today/portToday.ts. No hand-written stand-in facts.
 */
const ROOT = path.join(__dirname, "..", "..");
const fx = (rel: string) => JSON.parse(readFileSync(path.join(ROOT, "fixture-data", "v1", rel), "utf8"));

export const REAL_NOW = "2026-10-13T13:00:00Z"; // 6 AM PDT, the fixture's morning

const PLACES: Record<string, { label: string; lat: number; lon: number }> = {
  "port-592": { label: "Moss Landing", lat: 36.8044, lon: -121.78691 },
  "port-452": { label: "Princeton / Half Moon Bay", lat: 37.50222, lon: -122.48222 },
};

const charm = (p: number): CharmInput => ({
  issued: "2026-10-12",
  state: "current",
  ageDays: 1,
  url: "https://coastalscience.noaa.gov/science-areas/habs/hab-forecasts/california/",
  days: [0, 1, 2, 3].map((lead) => ({ lead, date: `2026-10-${12 + lead}`, pda: { here: null, median: p, max: p, n: 20, nearestKm: 3 }, pn: { here: null, median: p / 2, max: p, n: 20, nearestKm: 3 } })),
});

/** Moss Landing (or Half Moon Bay) on the fixture's morning: status, tests, whale rules, a quiet C-HARM run, data age. No sea. */
export function realPortToday(placeId: keyof typeof PLACES | string = "port-592", over: Partial<PortTodayInputs> = {}): PortToday {
  const place = PLACES[placeId];
  if (!place) throw new Error(`no test place ${placeId}`);
  return buildPortToday({
    place: { place_id: placeId, ...place },
    now: REAL_NOW,
    dataGeneratedAt: "2026-10-13T06:00:00Z",
    selected: FISHERY_OPTIONS.map((o) => o.id),
    fisheries: parseFisheriesStatus(fx("fisheries-status/index.json")) as FisheriesStatusIndex,
    biotoxin: parseBiotoxin(fx("biotoxin/index.json")) as BiotoxinIndex,
    notices: [],
    charm: charm(0.1),
    outlook: null,
    conditions: null,
    ...over,
  });
}

/**
 * The environment the e2e servers run with (playwright.config.ts), for unit tests that load facts the
 * way the app does: the committed fixture data plus the derived fixture in one folder, recorded NOAA
 * answers, the fixture's clock. Returns the env to set (and restore).
 */
export function fixtureServerEnv(): Record<string, string> {
  const dir = mkdtempSync(path.join(tmpdir(), "neritic-fixture-"));
  cpSync(path.join(ROOT, "fixture-data", "v1"), dir, { recursive: true });
  cpSync(path.join(ROOT, "fixture-derived", "v1"), dir, { recursive: true });
  return { CW_DATA_DIR: dir, CW_CONDITIONS_FIXTURE: path.join(ROOT, "conditions-fixtures.json"), CW_NWS_API: "1", CW_FIXED_NOW: "2026-10-13T06:00:00Z" };
}
