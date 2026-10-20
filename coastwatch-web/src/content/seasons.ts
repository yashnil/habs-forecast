import REGIONS_JSON from "./seasons-cdfw.json";
import type { SpeciesId } from "./species";

/**
 * Seasons, as CDFW states them: Neritic's dated copy of CDFW's "Ocean Sport Fishing" regional
 * summaries (https://wildlife.ca.gov/Fishing/Ocean/Regulations/Fishing-Map/<region>), one per
 * groundfish management area.
 *
 * Each region page lists species under CDFW's own headings "Open Fishing Seasons" and "Closed
 * Fishing Seasons", each with CDFW's marker (green: open; yellow: "Partially closed fishing season
 * or alert"; red: closed), and a section per species. `seasons-cdfw.json` holds, verbatim, each list
 * entry, its marker and the first paragraph of its section. Neritic never decides that a season is
 * open or closed: it says which list CDFW put the species in, on the date CDFW gives, and quotes CDFW.
 *
 * Read 2026-10-20 from the five region pages (HTTP 200; HTML comments removed first, because the
 * pages keep old seasons inside comments). CDFW's own "updated on" date is kept per region.
 * These pages cover sport (recreational) fishing only. Commercial seasons, in-season actions
 * (RAMP gear and depth limits, domoic acid delays) and quotas move faster than any copy: the
 * briefing links CDFW's in-season page and says so.
 *
 * Regenerate: docs/coastwatch/v5/fishermen-seasons-extract.py (fetch the five pages, run it, update SEASONS_READ).
 */
export type SeasonMark = "open" | "alert" | "closed";
export type SeasonItem = { id: string; name: string; list: "open" | "closed"; mark: SeasonMark; quote: string };
export type SeasonRegion = { id: string; label: string; bounds: string; south: number; north: number; url: string; updated: string; items: SeasonItem[] };

export const SEASON_REGIONS = REGIONS_JSON as SeasonRegion[];
/** When Neritic read the CDFW pages (Pacific date). */
export const SEASONS_READ = "2026-10-20";
/** After this many days the copy is called old in the briefing, and the reader is sent to CDFW first. */
export const SEASONS_STALE_DAYS = 14;

export const CDFW_INSEASON = { label: "CDFW in-season ocean fishing changes", url: "https://wildlife.ca.gov/Fishing/Ocean/Regulations/inseason" };
export const CDFW_COMMERCIAL = { label: "CDFW commercial fishing regulations", url: "https://wildlife.ca.gov/Fishing/Commercial" };
export const CDFW_GROUNDFISH = { label: "CDFW recreational groundfish summary", url: "https://wildlife.ca.gov/Fishing/Ocean/Regulations/Groundfish-Summary" };
/** Phone lines as CDFW prints them on its Ocean Salmon page (read 2026-10-20). */
export const SALMON_LINES = [
  { label: "CDFW ocean salmon regulations hotline", phone: "(707) 576-3429", tel: "+17075763429" },
  { label: "NOAA Fisheries ocean salmon hotline", phone: "(800) 662-9825", tel: "+18006629825" },
];

/**
 * CDFW's recreational groundfish table for 2026 (Groundfish-Summary, "Updated June 23, 2026"; read
 * 2026-10-20), quoted per management area and period. The region pages send rockfish, cabezon,
 * greenlings and lingcod readers to this table. `from`/`to` are month-day, inclusive.
 */
export type GroundfishRow = { from: string; to: string; nearshore: string; shelf: string };
export const GROUNDFISH_UPDATED = "June 23, 2026";
const ALL_DEPTHS: GroundfishRow[] = [
  { from: "01-01", to: "03-31", nearshore: "Closed - unlawful to possess in all waters", shelf: "Closed - unlawful to possess in all waters" },
  { from: "04-01", to: "12-31", nearshore: "Open all depths", shelf: "Open all depths" },
];
export const GROUNDFISH_2026: Record<string, GroundfishRow[]> = {
  northern: ALL_DEPTHS,
  mendocino: ALL_DEPTHS,
  "san-francisco": ALL_DEPTHS,
  central: ALL_DEPTHS,
  southern: [
    { from: "01-01", to: "03-31", nearshore: "Closed - unlawful to possess in all waters", shelf: "Closed - unlawful to possess in all waters" },
    { from: "04-01", to: "06-30", nearshore: "Open All Depths", shelf: "Open All Depths" },
    { from: "07-01", to: "09-30", nearshore: "50 Fathom - Inshore Only", shelf: "50 Fathom - Inshore Only" },
    { from: "10-01", to: "12-31", nearshore: "Closed - unlawful to possess in all waters", shelf: "50 Fathom - Offshore Only" },
  ],
};

/** Which entries of CDFW's region lists are about each of Neritic's species (CDFW's anchors on the page). */
export const SPECIES_SEASON_IDS: Partial<Record<SpeciesId, string[]>> = {
  dungeness: ["dungeness", "dungenessclosed"],
  "rock-crab": ["rock"],
  "razor-clam": ["rclam", "clams"],
  mussel: ["mussels"],
  bivalve: ["clams"],
  salmon: ["salmon"],
  groundfish: ["rockfish", "lingcod", "cabezon", "greenlings"],
  halibut: ["halibut", "pachalibut"],
  lobster: ["lobster"],
  tunas: ["tunas"],
};

/** The entries shown first for "any species": what most boats and shore anglers ask about. */
export const HEADLINE_SEASON_IDS = ["rockfish", "lingcod", "salmon", "dungeness", "dungenessclosed", "halibut", "pachalibut", "tunas", "rock", "lobster", "whiteseabass", "yellowtail", "kelpbass"];
