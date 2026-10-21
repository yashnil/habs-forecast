/**
 * Reference values shown BESIDE the trip worksheet's inputs, never pre-filled into them.
 *
 * Prices: CDFW Marine Region "By The Numbers" (the same reports as content/landings.ts), statewide,
 * all ports and gears, calendar year. Neritic computes dollars ÷ pounds; CDFW publishes no $/lb.
 * Only a species-exact fishery line counts: albacore and salmon are not in CDFW's top-ten list, so
 * they have no species-level reference here (bluefin is not albacore; sablefish is not "groundfish").
 *
 * Diesel: EIA weekly California No. 2 diesel retail price (series EMD_EPD2D_PTE_SCA_DPG), a dated
 * static value, read 2026-10-20. Not a feed: it goes stale and says so after 14 days.
 */

export type PriceRef = {
  key: string;
  /** the CDFW fishery line, exactly as CDFW names it */
  fishery: string;
  dollars: number;
  pounds: number;
  /** dollars ÷ pounds, computed by Neritic */
  usdPerLb: number;
  prior: { dollars: number; pounds: number; usdPerLb: number; year: number };
  unit: "$/lb";
  period: string;
  source: string;
  url: string;
  asOf: string;
  /** e.g. Dungeness: a calendar year mixes the end of one season and the start of the next */
  caveat?: string;
};

const CDFW_2024 = { label: "CDFW Marine Region, 2024 By The Numbers", url: "https://nrm.dfg.ca.gov/FileHandler.ashx?DocumentID=229396&inline=", asOf: "data as of 02/14/2025" };
const per = (d: number, p: number) => d / p;

function ref(key: string, fishery: string, y24: [number, number], y23: [number, number], caveat?: string): PriceRef {
  return {
    key,
    fishery,
    dollars: y24[0],
    pounds: y24[1],
    usdPerLb: per(y24[0], y24[1]),
    prior: { dollars: y23[0], pounds: y23[1], usdPerLb: per(y23[0], y23[1]), year: 2023 },
    unit: "$/lb",
    period: "calendar year 2024 (January 1 – December 31)",
    source: CDFW_2024.label,
    url: CDFW_2024.url,
    asOf: CDFW_2024.asOf,
    caveat,
  };
}

/** [dollars, pounds] from "Top Fisheries by Ex-Vessel Value", 2024 and 2023 reports. */
export const PRICE_REFS: Record<string, PriceRef> = {
  dungeness: ref("dungeness", "Dungeness Crab", [49_744_974, 14_173_811], [54_078_572, 21_019_733], "mixes two seasons"),
  squid: ref("squid", "Market Squid", [67_854_320, 126_430_905], [30_397_059, 52_270_346]),
  urchin: ref("urchin", "Red Sea Urchin", [7_476_765, 2_130_789], [8_454_398, 2_417_778]),
  halibut: ref("halibut", "California Halibut", [5_756_950, 1_014_939], [5_343_100, 982_099]),
  sablefish: ref("sablefish", "Sablefish", [4_734_080, 2_879_288], [5_724_189, 3_340_506]),
  petrale: ref("petrale", "Petrale Sole", [1_788_001, 1_492_852], [2_298_081, 1_959_754]),
  "spot-prawn": ref("spot-prawn", "Spot Prawn", [3_942_641, 176_968], [4_570_767, 216_125]),
  lobster: ref("lobster", "California Spiny Lobster", [20_893_435, 1_080_004], [21_506_406, 1_095_802]),
};

export const PRICE_REF_LABEL =
  "Past average, not today’s dock price: CDFW 2024, statewide, all ports and gears, calendar year, $ ÷ lb computed by Neritic. Ask your buyer.";

export const NO_PRICE_REF = "No species-level reference";

/** The reference for a catch line's species key, or null (never a neighbouring species). */
export function priceRef(key: string | null | undefined): PriceRef | null {
  return (key && PRICE_REFS[key]) || null;
}

export type DieselRef = {
  usdPerGal: number;
  week: string;
  prior: { usdPerGal: number; week: string };
  unit: "$/gal";
  series: string;
  source: string;
  url: string;
  /** when Neritic read the value */
  fetchedAt: string;
  label: string;
};

export const DIESEL_REF: DieselRef = {
  usdPerGal: 8.242,
  week: "2026-10-19",
  prior: { usdPerGal: 4.898, week: "2025-10-20" },
  unit: "$/gal",
  series: "Weekly California No 2 Diesel Retail Prices (Dollars per Gallon)",
  source: "U.S. Energy Information Administration",
  url: "https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=EMD_EPD2D_PTE_SCA_DPG&f=W",
  fetchedAt: "2026-10-20",
  label: "On-road retail incl. road taxes; your fuel dock will differ",
};

export const DIESEL_STALE_DAYS = 14;

/** Days since the diesel reference's week began (null without a clock), and whether it is stale. */
export function dieselAge(now: Date | null): { days: number | null; stale: boolean } {
  if (!now) return { days: null, stale: false };
  const days = Math.floor((now.getTime() - Date.parse(`${DIESEL_REF.week}T00:00:00Z`)) / 86_400_000);
  return { days, stale: days > DIESEL_STALE_DAYS };
}

/** What a saved scenario keeps about a reference it used (so later edits to this file never rewrite history). */
export type RefSnapshot = { key: string; value: number; unit: string; source: string; period: string; fetchedAt: string };

export function priceSnapshot(r: PriceRef): RefSnapshot {
  return { key: `cdfw:${r.key}`, value: r.usdPerLb, unit: r.unit, source: `${r.source} (${r.url})`, period: r.period, fetchedAt: r.asOf };
}

export function dieselSnapshot(): RefSnapshot {
  return { key: "eia:diesel-ca", value: DIESEL_REF.usdPerGal, unit: DIESEL_REF.unit, source: `${DIESEL_REF.source}, ${DIESEL_REF.series} (${DIESEL_REF.url})`, period: `week of ${DIESEL_REF.week}`, fetchedAt: DIESEL_REF.fetchedAt };
}
