import { FISHERY_OPTIONS } from "@/pilot/facts";
import type { FisheryId } from "@/pilot/contracts";
import { readPortRecord, updatePortRecord } from "@/lib/portRecord";
import { CHART_MODE_KEY, OPENING_SEEN_KEY, type ChartMode } from "@/nx/shell/entry";
import { homePortKey, setHomePort } from "@/nx/shell/port";
import { emit, readRaw, remove } from "./store";

/**
 * The reader's preferences, read and changed in one place for the account page: the home port, the fisheries
 * in the reader's order, the vessel length, the chart mode, and the two first-visit pieces that can be played
 * again. Each lives in the key the rest of the app already reads (keys.ts); nothing new is stored here.
 */
export const VESSELS = [
  { id: "none", label: "No vessel / shore" },
  { id: "under_25ft", label: "Under 25 ft" },
  { id: "25_40ft", label: "25–40 ft" },
  { id: "40_60ft", label: "40–60 ft" },
  { id: "over_60ft", label: "Over 60 ft" },
] as const;
export type Vessel = (typeof VESSELS)[number]["id"];

const TOUR_KEY = "nx-tour-v1";
const FISHERY_IDS = new Set<string>(FISHERY_OPTIONS.map((f) => f.id));
const VESSEL_IDS = new Set<string>(VESSELS.map((v) => v.id));

export type Prefs = {
  homePort: string | null;
  lastPort: string | null;
  /** the reader's fisheries, first one leads Today */
  fisheries: FisheryId[];
  vessel: Vessel | null;
  chartMode: ChartMode | null;
  tourDismissed: boolean;
  openingSeen: boolean;
};

export function readPrefs(): Prefs {
  const r = readPortRecord();
  const mode = readRaw(CHART_MODE_KEY);
  return {
    homePort: homePortKey(),
    lastPort: r.lastPort,
    fisheries: r.fisheries.filter((f) => FISHERY_IDS.has(f)) as FisheryId[],
    vessel: r.vessel && VESSEL_IDS.has(r.vessel) ? (r.vessel as Vessel) : null,
    chartMode: mode === "day" || mode === "dusk" || mode === "night" ? mode : null,
    tourDismissed: readRaw(TOUR_KEY) === "dismissed",
    openingSeen: readRaw(OPENING_SEEN_KEY) === "1",
  };
}

export { setHomePort };

/** The fisheries in the reader's order (unknown ids dropped, no repeats). Ids of other pages' fisheries are kept after them. */
export function setFisheries(ids: FisheryId[]): boolean {
  const mine = [...new Set(ids.filter((f) => FISHERY_IDS.has(f)))];
  const kept = updatePortRecord((r) => ({ ...r, fisheries: [...mine, ...r.fisheries.filter((f) => !FISHERY_IDS.has(f))] }));
  emit();
  return kept;
}

/** Move one fishery up or down the order; the first leads Today. */
export function moveFishery(id: FisheryId, by: -1 | 1): boolean {
  const list = readPrefs().fisheries;
  const i = list.indexOf(id);
  const j = i + by;
  if (i < 0 || j < 0 || j >= list.length) return false;
  const next = [...list];
  [next[i], next[j]] = [next[j], next[i]];
  return setFisheries(next);
}

export function setVessel(v: Vessel | null): boolean {
  const kept = updatePortRecord((r) => ({ ...r, vessel: v }));
  emit();
  return kept;
}

/** "Show me around" is offered again on Today. */
export function replayTour() {
  remove(TOUR_KEY);
}

/** The opening plays again on the next visit to the front door. */
export function replayOpening() {
  remove(OPENING_SEEN_KEY);
}
