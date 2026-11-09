import type { PortToday } from "../contracts";
import { fixturePortToday } from "./fixture";
import type { Dataset } from "./types";

/**
 * Where Ask Neritic gets its PortFacts. The ONLY retrieval it does: one port's Today model.
 *
 * INTEGRATION POINT: when the Today track's `src/pilot/facts.ts` lands, replace the body of
 * `loadPortToday` with its `buildPortToday(...)` (published data) and return dataset "published".
 * Until then the typed fixture is used and every answer is labelled "Sample data".
 */
export type PortTodayLoader = (placeId: string, now: Date) => Promise<{ today: PortToday; dataset: Dataset } | null>;

export const loadPortToday: PortTodayLoader = async (placeId) => {
  const today = fixturePortToday(placeId);
  return today ? { today, dataset: "fixture" } : null;
};
