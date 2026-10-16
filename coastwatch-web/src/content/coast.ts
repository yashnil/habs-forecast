/**
 * My Coast reference content: named harvest locations, CDFW's recreational regulation regions,
 * agency links, and the reviewable part of the notice-applicability model (docs/coastwatch/28).
 * Every entry names its source. Nothing here is a regulatory fact of its own: the agencies'
 * pages are the authority.
 */

export type HarvestSite = {
  id: string;
  name: string;
  county: string;
  lat: number;
  lon: number;
  /** what the agency samples there (not a statement that harvest is allowed) */
  sampled: string;
  source: { label: string; url: string; date: string };
};

/**
 * Named harvest locations: CDPH marine biotoxin monitoring sites, with CDPH's own coordinates.
 * A site is a sampling point on the shore, not the extent of a beach or bay.
 */
export const HARVEST_SITES: HarvestSite[] = [
  {
    id: "crescent-beach",
    name: "Crescent Beach, Crescent City",
    county: "Del Norte",
    lat: 41.71,
    lon: -124.143,
    sampled: "razor clams",
    source: { label: "CDPH razor clam biotoxin results (PDF)", url: "https://cdphdata.maps.arcgis.com/sharing/rest/content/items/f349952bae944a17abb685669ff33be1/data", date: "2026-10-08" },
  },
  {
    id: "clam-beach",
    name: "Clam Beach, Humboldt",
    county: "Humboldt",
    lat: 41.008,
    lon: -124.115,
    sampled: "razor clams",
    source: { label: "CDPH razor clam biotoxin results (PDF)", url: "https://cdphdata.maps.arcgis.com/sharing/rest/content/items/f349952bae944a17abb685669ff33be1/data", date: "2026-10-08" },
  },
  {
    id: "tomales-bay",
    name: "Tomales Bay",
    county: "Marin",
    lat: 38.205,
    lon: -122.931,
    sampled: "mussels and oysters (shellfish growing leases)",
    source: { label: "CDPH bivalve biotoxin results, past 30 days (PDF)", url: "https://cdphdata.maps.arcgis.com/sharing/rest/content/items/1f96ab557a014e2d8275b651a6447b1d/data", date: "2026-10-08" },
  },
  {
    id: "goleta-pier",
    name: "Goleta Pier",
    county: "Santa Barbara",
    lat: 34.413,
    lon: -119.828,
    sampled: "wild sea mussels",
    source: { label: "CDPH bivalve biotoxin results, past 30 days (PDF)", url: "https://cdphdata.maps.arcgis.com/sharing/rest/content/items/1f96ab557a014e2d8275b651a6447b1d/data", date: "2026-10-08" },
  },
];

/** CDFW's ocean recreational fishing regulation regions, north to south (limits as CDFW states them). */
export const CDFW_REGIONS = [
  { id: "northern", name: "Northern", north: 42.0, south: 40 + 10 / 60, limits: "42°00′ N (Oregon border) to 40°10′ N", url: "https://wildlife.ca.gov/Fishing/Ocean/Regulations/Fishing-Map/Northern" },
  { id: "mendocino", name: "Mendocino", north: 40 + 10 / 60, south: 38 + 57.5 / 60, limits: "40°10′ N (near Cape Mendocino) to 38°57.5′ N", url: "https://wildlife.ca.gov/Fishing/Ocean/Regulations/Fishing-Map/Mendocino" },
  { id: "san-francisco", name: "San Francisco", north: 38 + 57.5 / 60, south: 37 + 11 / 60, limits: "38°57.5′ N (Point Arena) to 37°11′ N", url: "https://wildlife.ca.gov/Fishing/Ocean/Regulations/Fishing-Map/San-Francisco" },
  { id: "central", name: "Central", north: 37 + 11 / 60, south: 34 + 27 / 60, limits: "37°11′ N (Pigeon Point) to 34°27′ N", url: "https://wildlife.ca.gov/Fishing/Ocean/Regulations/Fishing-Map/Central" },
  { id: "southern", name: "Southern", north: 34 + 27 / 60, south: 32.5, limits: "34°27′ N (Point Conception) to the Mexico border", url: "https://wildlife.ca.gov/Fishing/Ocean/Regulations/Fishing-Map/Southern" },
] as const;

/** Agency pages (checked 2026-10-12; OEHHA's site refuses automated checks and was not machine-verified). */
export const AGENCY_LINKS = {
  cdfwAdvisories: { label: "CDFW health advisories and fishery closures", url: "https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories" },
  cdfwCommercial: { label: "CDFW commercial fishing", url: "https://wildlife.ca.gov/Fishing/Commercial" },
  cdfwMpa: { label: "CDFW Marine Protected Areas: maps and rules", url: "https://wildlife.ca.gov/Conservation/Marine/MPAs" },
  cdfwClosureLine: { label: "CDFW domoic acid fishery closure line", phone: "(831) 649-2883", tel: "+18316492883" },
  cdphAdvisories: { label: "CDPH shellfish and seafood advisories", url: "https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx" },
  cdphMap: { label: "CDPH shellfish advisory map", url: "https://experience.arcgis.com/experience/394836318cfe4f7494e1c09097a43559" },
  cdphResults: { label: "CDPH shellfish toxin test results, past 30 days (PDF)", url: "https://cdphdata.maps.arcgis.com/sharing/rest/content/items/1f96ab557a014e2d8275b651a6447b1d/data" },
  cdphRazorResults: { label: "CDPH razor clam toxin test results (PDF)", url: "https://cdphdata.maps.arcgis.com/sharing/rest/content/items/f349952bae944a17abb685669ff33be1/data" },
  cdphLine: { label: "CDPH shellfish and biotoxin information line", phone: "1 (800) 553-4133", tel: "+18005534133" },
  oehha: { label: "OEHHA: domoic acid in fish and shellfish", url: "https://oehha.ca.gov/fish/general-info/marine-biotoxin-domoic-acid-fish-and-shellfish" },
} as const;

/**
 * The reviewable part of the applicability model (docs/coastwatch/28 §3).
 *
 * `noticeParts`: geography a notice states that its registry record does not encode. Each part
 * borrows a published outline and is therefore approximate: associations from a part are never
 * shown as having explicit geographic evidence.
 *
 * `reviews`: a person's decision on one location–notice association. Empty until someone reviews
 * them; never filled in automatically and never with anyone's name by default.
 */
export type NoticePart = { noticeId: string; part: string; outlineFrom: string; quote: string; status: "candidate" };
export type AssociationReview = { location: string; noticeId: string; decision: "confirmed" | "rejected"; reviewer: string; reviewedAt: string; note: string };

export const APPLICABILITY: { noticeParts: NoticePart[]; reviews: AssociationReview[] } = {
  noticeParts: [
    {
      noticeId: "cdph-rock-crab-viscera-north-coast-nci",
      part: "Northern Channel Islands",
      outlineFrom: "cdph-nci-bivalve-special-advisory",
      quote: "and from the waters off Santa Rosa Island and the Northern Channel Islands",
      status: "candidate",
    },
  ],
  reviews: [],
};

/** Distances that decide how an association is described (docs/coastwatch/28 §4). */
export const APPLICABILITY_KM = {
  /** a shore location this close to an area outline counts as on that area's shore (outlines are generalized) */
  shore: 1,
  /** a shore location this close to an outline's edge (beyond the shore tolerance) is unresolved, not outside */
  edge: 5,
  /** a separate area or a latitude limit within this distance is listed as nearby */
  nearby: 30,
  /** a latitude limit within this distance of a location inside the band is called out */
  nearLimit: 10,
} as const;

export const COAST_COPY = {
  title: "My Coast",
  lede: "Choose the waters you care about. Neritic shows the official notices that may apply, what was measured nearby and when, where the forecast has values, and what is not known.",
  notAuthority: "Neritic is not the authority. Regulations and advisories change; the agencies' own pages decide.",
  pierNotHere: "A pier sample describes the water at that pier on that day. It may not represent other waters, even nearby, or seafood.",
  modelNotHere: "C-HARM gives probabilities for 3 km ocean cells. A value in a cell offshore is not a forecast for this location.",
  chlNotToxin: "Chlorophyll measures algae, not toxin.",
  currentsNotForecast: "Observed surface currents show recent water movement, not where a bloom will go.",
  unreviewed: "Matched by location from each notice's official geography; not reviewed by a person.",
  noPsp: "Shellfish toxins from paralytic shellfish poisoning (PSP) are tested by CDPH and are not shown in Neritic.",
} as const;
