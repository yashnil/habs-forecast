import type { PortToday } from "../contracts";
import { loadPortTodayFacts } from "../today/portToday";
import type { Dataset } from "./types";

/**
 * Where Ask Neritic gets its PortFacts. The ONLY retrieval it does: one port's Today facts, built by
 * the same function Today and alerts use (src/pilot/today/portToday.ts), with every fishery and the
 * sea. `dataset` is "fixture" only when the server is reading the test fixture; answers then say
 * "Sample data". A port whose data cannot be loaded is null (Ask says the data is unavailable).
 */
export type PortTodayLoader = (placeId: string, now: Date) => Promise<{ today: PortToday; dataset: Dataset } | null>;

export const loadPortToday: PortTodayLoader = async (placeId) => {
  const loaded = await loadPortTodayFacts(placeId);
  return loaded ? { today: loaded.today, dataset: loaded.fixture ? "fixture" : "published" } : null;
};
