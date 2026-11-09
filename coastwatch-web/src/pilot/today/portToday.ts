/**
 * ONE fact source for the pilot (server only). Today, alerts and Ask Neritic all read a port's facts
 * from here, so a status row, an alert and a cited answer are the same statement with the same id.
 *
 *   loadPortTodayFacts(placeId) = loadToday(placeId)            the published data for the port
 *                               + loadConditions(lat, lon)      NWS, NDBC, CO-OPS, fetched here (no request to self)
 *                               + buildPortToday(...)           with EVERY fishery selected
 *
 * The Today page draws the same model in the browser from the same inputs (its sea arrives from
 * /api/conditions, which calls the same `loadConditions`). A failed sea fetch is `conditions: "error"`
 * (the facts then say the sea could not be loaded); a missing port or manifest is `null`. `fixture` is
 * true when the manifest is the test fixture: such facts are labelled "sample" and are never pushed.
 */
import type { FisheryId, PortToday } from "@/pilot/contracts";
import type { Conditions } from "@/lib/conditions";
import { loadConditions } from "@/lib/conditionsServer";
import { FISHERY_OPTIONS, buildPortToday, type PortTodayInputs } from "@/pilot/facts";
import { loadToday } from "./load";

export const ALL_FISHERIES: FisheryId[] = FISHERY_OPTIONS.map((o) => o.id);
/** NWS forecast periods Today asks for (about five days); the page's /api/conditions request uses the same number. */
export const TODAY_PERIODS = 12;

export type PortTodayFacts = { today: PortToday; fixture: boolean; inputs: PortTodayInputs };

export type PortFactsOptions = {
  /** false: skip the sea (alerts never read sea, tide or buoy facts; see alerts/diff.ts KIND_TRIGGER) */
  sea?: boolean;
};

/** The point Today asks conditions for (4 decimals, as in the page's request, so both read one cached answer). */
export const conditionsPoint = (lat: number, lon: number) => ({ lat: Number(lat.toFixed(4)), lon: Number(lon.toFixed(4)) });

export async function loadPortTodayFacts(placeId: string, opts: PortFactsOptions = {}): Promise<PortTodayFacts | null> {
  const load = await loadToday(placeId).catch(() => null);
  if (!load || !load.ok || !load.place || !load.inputs) return null;
  let conditions: Conditions | null | "error" = null;
  if (opts.sea !== false) {
    const p = conditionsPoint(load.inputs.place.lat, load.inputs.place.lon);
    conditions = await loadConditions(p.lat, p.lon, TODAY_PERIODS).catch(() => "error" as const);
  }
  const inputs: PortTodayInputs = { ...load.inputs, selected: ALL_FISHERIES, conditions };
  return { today: buildPortToday(inputs), fixture: load.fixture, inputs };
}

/** Which dataset the server is reading, from the manifest alone: the test fixture, published data, or none. */
export async function datasetKind(): Promise<"fixture" | "published" | "unavailable"> {
  const load = await loadToday(null).catch(() => null);
  if (!load || !load.ok) return "unavailable";
  return load.fixture ? "fixture" : "published";
}
