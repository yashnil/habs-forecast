import type { Audience, CoastPlace } from "@/lib/coast";
import { groundfishOn, regionFor, regionSeasons } from "@/lib/speciesSeasons";
import type { SeasonItem } from "@/content/seasons";
import type { Species } from "@/content/species";

/** Species CDFW sets for commercial fishing only (no sport-list entry of their own). */
export const COMMERCIAL_ONLY = new Set(["squid", "urchin"]);

export type SeasonStatus = {
  /** CDFW's own sport-list entry for the species in this region, if it has one */
  item: SeasonItem | null;
  regionLabel: string | null;
  commercial: boolean;
  commercialOnly: boolean;
  /** CDFW's sport-list sentence "scheduled to reopen on …" (never for a commercial reader: v6 P0 2) */
  reopen: string | null;
  /** the species is on CDFW's closed sport list here */
  closed: boolean;
  groundfish: ReturnType<typeof groundfishOn>;
};

/**
 * The season facts the page quotes, from Neritic's copy of CDFW's regional sport summary. Never a
 * statement that a fishery is open: the commercial season is CDFW's, which Neritic does not hold.
 */
export function seasonStatus(place: CoastPlace, species: Species, audience: Audience, day: string | null): SeasonStatus {
  const region = regionFor(place.lat);
  const rs = region ? regionSeasons(region, species.id) : null;
  const item = rs?.mine[0] ?? null;
  const commercial = audience === "commercial";
  return {
    item,
    regionLabel: region?.label ?? null,
    commercial,
    commercialOnly: COMMERCIAL_ONLY.has(species.id),
    reopen: !commercial ? (item?.quote.match(/scheduled to reopen on ([A-Z][a-z]+ \d{1,2}, \d{4})/)?.[1] ?? null) : null,
    closed: item?.mark === "closed",
    groundfish: species.id === "groundfish" && region && day ? groundfishOn(region.id, day) : null,
  };
}
