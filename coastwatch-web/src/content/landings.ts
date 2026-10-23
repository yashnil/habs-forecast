import type { SpeciesId } from "./species";

/**
 * What commercial catch was worth at the dock: CDFW Marine Region "By The Numbers" reports,
 * transcribed from the published PDFs (read 2026-10-20). Calendar-year totals, as CDFW notes:
 * "landings and value reported over the calendar year (January 1 – December 31) This may differ from
 * seasonal landings for specific fisheries reported elsewhere." Ex-vessel value is, in CDFW's words,
 * "the amount paid to the fishermen at the dock".
 *
 * History only: never this season, never a price today, never a forecast. Neritic's one piece of
 * arithmetic is dollars ÷ pounds, shown as "about $X a pound on average over the year" and labelled.
 * The pipeline's statewide NOAA series (Fisheries page) covers six toxin-affected groups back to 2015.
 */
export type LandingsReport = {
  year: number;
  label: string;
  url: string;
  asOf: string;
  /** "Weight and Ex-Vessel Value of Commercial Landings by Port Area" */
  portAreas: { area: string; pounds: number; dollars: number }[];
  portTotal: { pounds: number; dollars: number };
  /** "Top Fisheries by Ex-Vessel Value" (top ten as CDFW lists them) */
  topByValue: { fishery: string; pounds: number; dollars: number }[];
};

export const LANDINGS: LandingsReport[] = [
  {
    year: 2024,
    label: "CDFW Marine Region, 2024 By The Numbers",
    url: "https://nrm.dfg.ca.gov/FileHandler.ashx?DocumentID=229396&inline=",
    asOf: "data as of 02/14/2025",
    portAreas: [
      { area: "Santa Barbara", pounds: 81_517_823, dollars: 64_983_615 },
      { area: "Los Angeles", pounds: 50_566_187, dollars: 37_451_989 },
      { area: "San Francisco", pounds: 7_722_825, dollars: 26_515_641 },
      { area: "Eureka", pounds: 7_359_938, dollars: 20_018_255 },
      { area: "San Diego", pounds: 4_556_133, dollars: 12_110_114 },
      { area: "Bodega Bay", pounds: 3_276_381, dollars: 10_939_611 },
      { area: "Monterey", pounds: 16_123_299, dollars: 7_930_496 },
      { area: "Fort Bragg", pounds: 4_953_436, dollars: 5_423_659 },
      { area: "Morro Bay", pounds: 1_608_502, dollars: 3_678_279 },
    ],
    portTotal: { pounds: 117_684_528, dollars: 189_051_663 },
    topByValue: [
      { fishery: "Market Squid", pounds: 126_430_905, dollars: 67_854_320 },
      { fishery: "Dungeness Crab", pounds: 14_173_811, dollars: 49_744_974 },
      { fishery: "California Spiny Lobster", pounds: 1_080_004, dollars: 20_893_435 },
      { fishery: "Red Sea Urchin", pounds: 2_130_789, dollars: 7_476_765 },
      { fishery: "California Halibut", pounds: 1_014_939, dollars: 5_756_950 },
      { fishery: "Sablefish", pounds: 2_879_288, dollars: 4_734_080 },
      { fishery: "Spot Prawn", pounds: 176_968, dollars: 3_942_641 },
      { fishery: "Yellow Rock Crab", pounds: 765_516, dollars: 1_801_066 },
      { fishery: "Petrale Sole", pounds: 1_492_852, dollars: 1_788_001 },
      { fishery: "Bluefin Tuna", pounds: 288_453, dollars: 1_712_489 },
    ],
  },
  {
    year: 2023,
    label: "CDFW Marine Region, 2023 By The Numbers",
    url: "https://nrm.dfg.ca.gov/FileHandler.ashx?DocumentID=220034&inline=",
    asOf: "data as of 2/9/2024",
    portAreas: [
      { area: "Eureka", pounds: 21_485_167, dollars: 44_004_979 },
      { area: "Santa Barbara", pounds: 23_359_142, dollars: 35_131_162 },
      { area: "Los Angeles", pounds: 37_027_607, dollars: 32_722_082 },
      { area: "San Francisco", pounds: 4_250_247, dollars: 13_000_690 },
      { area: "San Diego", pounds: 4_757_370, dollars: 14_131_925 },
      { area: "Fort Bragg", pounds: 5_960_578, dollars: 7_696_037 },
      { area: "Monterey", pounds: 10_373_498, dollars: 5_767_286 },
      { area: "Bodega Bay", pounds: 2_163_851, dollars: 5_176_363 },
      { area: "Morro Bay", pounds: 1_361_409, dollars: 3_198_890 },
    ],
    portTotal: { pounds: 110_738_872, dollars: 160_829_414 },
    topByValue: [
      { fishery: "Dungeness Crab", pounds: 21_019_733, dollars: 54_078_572 },
      { fishery: "Market Squid", pounds: 52_270_346, dollars: 30_397_059 },
      { fishery: "California Spiny Lobster", pounds: 1_095_802, dollars: 21_506_406 },
      { fishery: "Red Sea Urchin", pounds: 2_417_778, dollars: 8_454_398 },
      { fishery: "Sablefish", pounds: 3_340_506, dollars: 5_724_189 },
      { fishery: "California Halibut", pounds: 982_099, dollars: 5_343_100 },
      { fishery: "Spot Prawn", pounds: 216_125, dollars: 4_570_767 },
      { fishery: "Bigeye Tuna", pounds: 788_264, dollars: 3_422_285 },
      { fishery: "Petrale Sole", pounds: 1_959_754, dollars: 2_298_081 },
      { fishery: "Bluefin Tuna", pounds: 409_482, dollars: 2_261_460 },
    ],
  },
];

/** The CDFW fishery names in "Top Fisheries by Ex-Vessel Value" that are each of Neritic's species. */
export const SPECIES_FISHERIES: Partial<Record<SpeciesId, string[]>> = {
  dungeness: ["Dungeness Crab"],
  "rock-crab": ["Yellow Rock Crab"],
  lobster: ["California Spiny Lobster"],
  halibut: ["California Halibut"],
  squid: ["Market Squid"],
  urchin: ["Red Sea Urchin"],
  groundfish: ["Sablefish", "Petrale Sole"],
  tunas: ["Bluefin Tuna", "Bigeye Tuna"],
};

/** CDFW's tool for port-level and recent landings, which Neritic does not copy (its terms ask users to consult it first). */
export const CDFW_MFDE = { label: "CDFW Marine Fisheries Data Explorer", url: "https://wildlife.ca.gov/Conservation/Marine/Data-Management-Research/MFDE" };
