import { GROUNDFISH_2026, HEADLINE_SEASON_IDS, SEASON_REGIONS, SEASONS_READ, SPECIES_SEASON_IDS, type GroundfishRow, type SeasonItem, type SeasonRegion } from "@/content/seasons";
import { LANDINGS, SPECIES_FISHERIES } from "@/content/landings";
import type { SpeciesId } from "@/content/species";

/**
 * The fishing side of a briefing: which CDFW sport-fishing region (groundfish management area) a
 * place is in, what CDFW's dated summary for that region lists, CDFW's groundfish table for the trip
 * day, and what the catch was worth (CDFW By The Numbers). Pure and tested
 * (tests/unit/speciesSeasons.test.ts). Nothing here decides that a season is open: it says which of
 * CDFW's lists an entry is in, as of CDFW's date.
 */

/** The CDFW region by latitude (CDFW's boundaries are lines of latitude; each region owns its south edge). */
export function regionFor(lat: number): SeasonRegion | null {
  return SEASON_REGIONS.find((r) => lat >= r.south && lat < r.north) ?? null;
}

export type RegionSeasons = {
  region: SeasonRegion;
  /** CDFW's entries for the reader's species (empty for "any species" or a species CDFW's page does not list) */
  mine: SeasonItem[];
  /** the commonly fished entries, in CDFW's list order, for a quick read */
  headline: SeasonItem[];
  /** every entry, CDFW's order: open (green), alert (yellow), closed (red) */
  all: SeasonItem[];
  counts: { open: number; alert: number; closed: number };
};

export function regionSeasons(region: SeasonRegion, species: SpeciesId): RegionSeasons {
  const ids = SPECIES_SEASON_IDS[species] ?? [];
  const mine = region.items.filter((i) => ids.includes(i.id));
  const headline = region.items.filter((i) => HEADLINE_SEASON_IDS.includes(i.id) && !mine.includes(i));
  const counts = { open: 0, alert: 0, closed: 0 };
  for (const i of region.items) counts[i.mark]++;
  return { region, mine, headline, all: region.items, counts };
}

/** CDFW's own words for each marker. */
export const MARK_WORDS: Record<SeasonItem["mark"], string> = {
  open: "On CDFW's open list",
  alert: "On CDFW's open list, marked “partially closed fishing season or alert”",
  closed: "On CDFW's closed list",
};
export const MARK_SHORT: Record<SeasonItem["mark"], string> = { open: "Open list", alert: "Open list · alert", closed: "Closed list" };

/** CDFW's groundfish row for a Pacific date (YYYY-MM-DD) in a region; null outside 2026, which the table covers. */
export function groundfishOn(regionId: string, date: string): (GroundfishRow & { year: number }) | null {
  if (!date.startsWith("2026-")) return null;
  const md = date.slice(5);
  const row = GROUNDFISH_2026[regionId]?.find((r) => md >= r.from && md <= r.to);
  return row ? { ...row, year: 2026 } : null;
}

const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
/** "04-01" → "Apr 1" */
export const monthDay = (md: string) => `${MON[Number(md.slice(0, 2)) - 1]} ${Number(md.slice(3))}`;

/** Days between Neritic's reading of the CDFW pages and a Pacific date. */
export function seasonsAgeDays(today: string): number {
  return Math.round((Date.parse(`${today}T12:00:00Z`) - Date.parse(`${SEASONS_READ}T12:00:00Z`)) / 86400000);
}

// ---------------------------------------------------------------- what it was worth

/** The CDFW port area of a port's `port_area` field ("SAN FRANCISCO" → its rows in each report). */
export function portAreaValues(portArea: string | null | undefined) {
  if (!portArea) return [];
  const key = portArea.toLowerCase();
  return LANDINGS.map((r) => {
    const i = r.portAreas.findIndex((a) => a.area.toLowerCase() === key);
    if (i < 0) return null;
    // rank by value, as CDFW's table is not always sorted by value
    const rank = [...r.portAreas].sort((a, b) => b.dollars - a.dollars).findIndex((a) => a.area.toLowerCase() === key) + 1;
    return { year: r.year, report: r, ...r.portAreas[i], rank, of: r.portAreas.length, share: r.portAreas[i].dollars / r.portTotal.dollars };
  }).filter((x): x is NonNullable<typeof x> => x !== null);
}

export type FisheryValue = { year: number; fishery: string; pounds: number; dollars: number; perLb: number; rank: number; report: (typeof LANDINGS)[number] };

/** The reader's species in CDFW's "Top Fisheries by Ex-Vessel Value", newest year first; empty when it is not in a top ten. */
export function speciesValues(species: SpeciesId): FisheryValue[] {
  const names = SPECIES_FISHERIES[species] ?? [];
  return LANDINGS.flatMap((r) =>
    r.topByValue
      .map((f, i) => ({ f, i }))
      .filter(({ f }) => names.includes(f.fishery))
      .map(({ f, i }) => ({ year: r.year, fishery: f.fishery, pounds: f.pounds, dollars: f.dollars, perLb: f.dollars / f.pounds, rank: i + 1, report: r })),
  );
}

/** "$64.98M", "$628k", "$1,234" */
export function dollars(v: number): string {
  if (v >= 1e6) return `$${(v / 1e6).toFixed(v >= 1e8 ? 0 : 1)}M`;
  if (v >= 1e4) return `$${Math.round(v / 1e3)}k`;
  return `$${Math.round(v).toLocaleString("en-US")}`;
}
export function pounds(v: number): string {
  if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M lb`;
  if (v >= 1e4) return `${Math.round(v / 1e3)}k lb`;
  return `${Math.round(v).toLocaleString("en-US")} lb`;
}
export const perPound = (v: number) => `$${v < 10 ? v.toFixed(2) : v.toFixed(0)}/lb`;
