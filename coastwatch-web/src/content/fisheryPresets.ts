import type { SpeciesId } from "./species";
import type { CrewKind } from "@/lib/tripEconomics";

/**
 * Starting values for the trip worksheet, by fishery. Every default is editable and carries its
 * source; anything Neritic has no source for is blank and entered by the user. No preset ever holds
 * a catch or a price: those are the fisherman's own (references sit beside the fields instead).
 *
 * Sources
 * - NWFSC-121: Lian, C.E. (2012), "West Coast Limited Entry Groundfish Cost Earnings Survey: Protocol
 *   and Results for 2008", NOAA Tech. Memo. NMFS-NWFSC-121 (Dec 2012). Table 30 (trawl fleet fuel use),
 *   Table 31 (fixed-gear fleet fuel use), Table 32/33 (shares of revenue after deductions). A 2008
 *   survey of limited-entry groundfish vessels: a fleet average then, not a norm now.
 * - FR 2024-28144: NMFS, "Fishing Capacity Reduction Program for the Pacific Coast Groundfish
 *   Fishery", 89 FR (Dec 2, 2024): fee 2.25 % of ex-vessel value on landings from Jan 1, 2025.
 * - FGC §8276.5: California Fish and Game Code, Dungeness crab trap limit tiers (500 … 175 tags).
 */

export type SourceId = "nwfsc121-t30" | "nwfsc121-t31" | "nwfsc121-t32" | "nwfsc121-t33" | "fr-2024-28144" | "fgc-8276.5";

export const SOURCES: Record<SourceId, { short: string; long: string; url: string; note?: string }> = {
  "nwfsc121-t30": {
    short: "NOAA NWFSC-121, Table 30 (2008 survey)",
    long: "NOAA Tech. Memo. NMFS-NWFSC-121, Table 30: trawl fleet fuel use, all trawlers, 2008",
    url: "https://repository.library.noaa.gov/gsearch?terms=NMFS-NWFSC-121",
    note: "Fleet average then, not a norm",
  },
  "nwfsc121-t31": {
    short: "NOAA NWFSC-121, Table 31 (2008 survey)",
    long: "NOAA Tech. Memo. NMFS-NWFSC-121, Table 31: fixed-gear fleet fuel use, all fixed gear, 2008",
    url: "https://repository.library.noaa.gov/gsearch?terms=NMFS-NWFSC-121",
    note: "Fleet average then, not a norm",
  },
  "nwfsc121-t32": {
    short: "NOAA NWFSC-121, Table 32 (2008 survey)",
    long: "NOAA Tech. Memo. NMFS-NWFSC-121, Table 32: trawl fleet crew share of revenue after deductions, 2008",
    url: "https://repository.library.noaa.gov/gsearch?terms=NMFS-NWFSC-121",
    note: "Fleet average then, not a norm",
  },
  "nwfsc121-t33": {
    short: "NOAA NWFSC-121, Table 33 (2008 survey)",
    long: "NOAA Tech. Memo. NMFS-NWFSC-121, Table 33: fixed-gear fleet crew share of revenue after deductions, 2008",
    url: "https://repository.library.noaa.gov/gsearch?terms=NMFS-NWFSC-121",
    note: "Fleet average then, not a norm",
  },
  "fr-2024-28144": {
    short: "NMFS buyback fee notice, FR 2024-28144",
    long: "NMFS, Fishing Capacity Reduction Program for the Pacific Coast Groundfish Fishery (Federal Register 2024-28144, Dec 2, 2024): 2.25 % from Jan 1, 2025",
    url: "https://www.federalregister.gov/documents/2024/12/02/2024-28144/fishing-capacity-reduction-program-for-the-pacific-coast-groundfish-fishery",
  },
  "fgc-8276.5": {
    short: "Fish and Game Code §8276.5",
    long: "California Fish and Game Code §8276.5: Dungeness crab trap limit tiers",
    url: "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=FGC&sectionNum=8276.5",
    note: "Your permit's tier and any in-season reduction are on your CDFW allocation",
  },
};

export type Sourced = { value: number; source: SourceId; label: string };

export type FisheryPresetId = "dungeness" | "salmon-troll" | "groundfish-hl" | "groundfish-trawl" | "squid" | "albacore" | "urchin";

export type CatchTemplate = {
  /** key into referenceValues PRICE_REFS when a species-exact reference exists; else a label-only key */
  key: string;
  label: string;
};

export type FisheryPreset = {
  id: FisheryPresetId;
  label: string;
  species: SpeciesId;
  /** squid is landed and sold by the short ton */
  unit: "lb" | "ton";
  /** what one scenario covers */
  effort: "trip" | "day fished" | "dive day";
  fishingGph: Sourced | null;
  steamGph: Sourced | null;
  crewKind: CrewKind;
  crewPct: Sourced | null;
  /** another sourced share the user may prefer */
  crewPctAlt: Sourced | null;
  feesPct: Sourced | null;
  /** fee or quota lease per lb: a labelled blank field for the user */
  perLbLabel: string;
  potTiers: { tiers: number[]; source: SourceId } | null;
  catch: CatchTemplate[];
  note?: string;
};

const T31 = "nwfsc121-t31" as const;
const STEAM_FIXED: Sourced = { value: 7.8, source: T31, label: "steaming full, all fixed gear" };

export const FISHERY_PRESETS: FisheryPreset[] = [
  {
    id: "dungeness",
    label: "Dungeness crab (pots)",
    species: "dungeness",
    unit: "lb",
    effort: "trip",
    fishingGph: { value: 4.8, source: T31, label: "crabbing, all fixed gear" },
    steamGph: STEAM_FIXED,
    crewKind: "afterExpenses",
    crewPct: { value: 28.1, source: "nwfsc121-t33", label: "crabber vessels, owner as captain" },
    crewPctAlt: { value: 30.7, source: "nwfsc121-t33", label: "all fixed gear, owner as captain" },
    feesPct: null,
    perLbLabel: "Fees, $ per lb",
    potTiers: { tiers: [500, 450, 400, 350, 300, 250, 175], source: "fgc-8276.5" },
    catch: [{ key: "dungeness", label: "Dungeness crab" }],
  },
  {
    id: "salmon-troll",
    label: "Salmon troll",
    species: "salmon",
    unit: "lb",
    effort: "trip",
    fishingGph: { value: 3.6, source: T31, label: "salmon trolling, all fixed gear" },
    steamGph: STEAM_FIXED,
    crewKind: "afterExpenses",
    crewPct: null,
    crewPctAlt: null,
    feesPct: null,
    perLbLabel: "Fees, $ per lb",
    potTiers: null,
    catch: [{ key: "salmon", label: "Salmon" }],
    note: "2008 was a year of very little salmon trolling (NWFSC-121).",
  },
  {
    id: "groundfish-hl",
    label: "Groundfish, hook-and-line",
    species: "groundfish",
    unit: "lb",
    effort: "trip",
    fishingGph: { value: 5.9, source: T31, label: "longlining, all fixed gear" },
    steamGph: STEAM_FIXED,
    crewKind: "afterExpenses",
    crewPct: { value: 30.7, source: "nwfsc121-t33", label: "all fixed gear, owner as captain" },
    crewPctAlt: { value: 28.7, source: "nwfsc121-t33", label: "all fixed gear, hired captain" },
    feesPct: null,
    perLbLabel: "Fees, $ per lb",
    potTiers: null,
    catch: [
      { key: "groundfish-live", label: "Groundfish, live" },
      { key: "groundfish-dead", label: "Groundfish, dead" },
    ],
  },
  {
    id: "groundfish-trawl",
    label: "Groundfish, trawl",
    species: "groundfish",
    unit: "lb",
    effort: "trip",
    fishingGph: { value: 17.1, source: "nwfsc121-t30", label: "groundfish trawling, all trawlers" },
    steamGph: { value: 14.6, source: "nwfsc121-t30", label: "steaming full, all trawlers" },
    crewKind: "afterExpenses",
    crewPct: { value: 25.7, source: "nwfsc121-t32", label: "all trawl, owner as captain" },
    crewPctAlt: { value: 22.5, source: "nwfsc121-t32", label: "all trawl, hired captain" },
    feesPct: { value: 2.25, source: "fr-2024-28144", label: "buyback loan fee, from Jan 1, 2025" },
    perLbLabel: "Quota lease and fees, $ per lb",
    potTiers: null,
    catch: [
      { key: "sablefish", label: "Sablefish" },
      { key: "petrale", label: "Petrale sole" },
    ],
  },
  {
    id: "squid",
    label: "Market squid",
    species: "squid",
    unit: "ton",
    effort: "trip",
    fishingGph: null,
    steamGph: null,
    crewKind: "afterExpenses",
    crewPct: null,
    crewPctAlt: null,
    feesPct: null,
    perLbLabel: "Fees, $ per lb",
    potTiers: null,
    catch: [{ key: "squid", label: "Market squid" }],
  },
  {
    id: "albacore",
    label: "Albacore",
    species: "tunas",
    unit: "lb",
    effort: "day fished",
    fishingGph: null,
    steamGph: null,
    crewKind: "afterExpenses",
    crewPct: null,
    crewPctAlt: null,
    feesPct: null,
    perLbLabel: "Fees, $ per lb",
    potTiers: null,
    catch: [{ key: "albacore", label: "Albacore" }],
  },
  {
    id: "urchin",
    label: "Red sea urchin (dive)",
    species: "urchin",
    unit: "lb",
    effort: "dive day",
    fishingGph: null,
    steamGph: null,
    crewKind: "afterExpenses",
    crewPct: null,
    crewPctAlt: null,
    feesPct: null,
    perLbLabel: "Fees, $ per lb",
    potTiers: null,
    catch: [
      { key: "urchin", label: "Red urchin, grade 1" },
      { key: "urchin", label: "Red urchin, grade 2" },
    ],
    note: "Buyers pay by grade: enter each grade's pounds and price.",
  },
];

/** The preset for a My Coast species (null when Neritic has none: the worksheet then starts blank). */
export function presetForSpecies(species: SpeciesId | string | null | undefined): FisheryPreset | null {
  return FISHERY_PRESETS.find((p) => p.species === species) ?? null;
}

export function presetById(id: string | null | undefined): FisheryPreset | null {
  return FISHERY_PRESETS.find((p) => p.id === id) ?? null;
}
