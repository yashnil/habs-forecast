/**
 * User-facing copy that carries safety meaning lives here so it can be tested
 * (tests/unit/safety.test.ts). Official URLs were verified on 2026-10-08
 * (docs/coastwatch/evidence/sources-regulatory.md).
 */

export const OFFICIAL_STATUS = {
  heading: "Official closures and advisories",
  notTracked:
    "CoastWatch lists notices transcribed from CDFW and CDPH pages. The list has not been checked by a person and may be incomplete. The absence of a notice here does not mean an area is open or that seafood is safe.",
  missingNotOpen:
    "No notice listed here does not mean an area is open or that seafood is safe. Always confirm with CDFW and CDPH.",
  instruction: "Before fishing or harvesting, check the official sources:",
  links: [
    {
      label: "CDFW — Health advisories and fishery closures",
      href: "https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories",
    },
    {
      label: "CDPH — Shellfish and seafood advisories",
      href: "https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx",
    },
    {
      label: "CDPH — Marine Biotoxin Monitoring Program",
      href: "https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Marine-Biotoxin-Monitoring-Program.aspx",
    },
    {
      label: "CDFW — Whale-safe fisheries / Dungeness crab status",
      href: "https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries",
    },
  ],
  hotlines: [
    { label: "CDPH shellfish & biotoxin information", phone: "(800) 553-4133", tel: "+18005534133" },
    { label: "CDFW domoic acid fishery closure line", phone: "(831) 649-2883", tel: "+18316492883" },
  ],
  verification: {
    verified: "Verified",
    aging: "Review ageing",
    unverified: "Not verified",
    unavailable: "Unavailable",
  },
  notVerifiedLead: "Treat these records as unconfirmed. The agency pages are the authority.",
  statementsHeading: "Official statements (quoted)",
  statementsNote: "Quoted for context. A statement that no toxin closure exists is not a statement that a fishery is open; seasons and other rules still apply.",
  areaNote: "Map outlines and latitude lines are drawn from the agency's wording; the wording controls. Latitude-defined notices state no offshore limit.",
} as const;

export const FORECAST_COPY = {
  heading: "Bloom and domoic acid forecast",
  productLine: "C-HARM v3.1 · NOAA CoastWatch West Coast",
  whatItIs:
    "Probability forecasts of a Pseudo-nitzschia bloom and of domoic acid in the water, from a NOAA model.",
  notA: "A forecast probability, not a measurement of toxin in seafood and not a closure decision.",
  lowNotSafe: "A low probability does not mean an area is safe.",
  issuedInferred: "Issue date inferred — C-HARM publishes valid days only.",
  noValue: "No forecast value here (land, outside the model, or a nearshore cell the model does not cover).",
} as const;

export const PORT_COPY = {
  spatial:
    "Summaries cover forecast or satellite cells within 15 km of the CDFW port location. They do not describe conditions at the dock or at a particular fishing ground.",
  nearshore: "C-HARM does not provide domoic acid probabilities for many cells within about 3–6 km of shore, so fewer cells contribute to those values.",
  history: "One value per C-HARM nowcast (median of the same cells). Days without a published run are left blank.",
  chlorophyll:
    "NOAA VIIRS 8-day composites (dated at the centre of each 8-day window; neighbouring values overlap). Median of clear pixels within 15 km. Chlorophyll is algae biomass, not toxin.",
} as const;

export const CHLOROPHYLL_COPY = {
  heading: "Satellite chlorophyll",
  biomass: "Chlorophyll measures algae biomass. It does not measure toxins and does not predict where fish are.",
  gaps: "Gaps are clouds or missing passes, not low chlorophyll.",
} as const;

export const DISCLAIMER =
  "CoastWatch brings together public data from state and federal agencies. It is not an official source. Closures, advisories and seasons from CDFW, CDPH and OEHHA always take precedence. Forecasts are probabilities, not guarantees.";

/** Primary navigation. Only built experiences appear: unbuilt features are not navigation. */
export const EXPERIENCES = [
  { key: "map", label: "Ocean Map", short: "Map", href: "/" },
  { key: "bloom", label: "Bloom Intelligence", short: "Blooms", href: "/bloom" },
  { key: "fisheries", label: "Fisheries", short: "Fisheries", href: "/fisheries" },
] as const;

export const REGION_LABEL: Record<string, string> = {
  north_coast: "North Coast",
  mendocino_sonoma: "Mendocino–Sonoma",
  sf_bay_farallones: "San Francisco & Farallones",
  monterey_bay: "Monterey Bay",
  central_coast: "Central Coast",
  southern_california: "Southern California",
  other: "Other",
};

export const BLOOM_COPY = {
  heading: "Bloom Intelligence",
  lede: "Measured harmful-algal-bloom data from CalHABMAP shore stations: domoic acid and Pseudo-nitzschia in water samples, with the C-HARM model shown separately on the same timeline.",
  measuredVsModel:
    "Measurements (top) are laboratory values from one water sample at one pier. The model (bottom) is a probability for nearby ocean cells. They are different quantities and are never compared numerically.",
  notSeafood: "Toxin in seawater is not toxin in seafood. Only official agency testing decides whether seafood can be harvested or eaten.",
  absence: "A blank means not measured. No measurement is not the same as no toxin, and a reported 0 means not quantified, not absent.",
  pointNotArea: "A station describes the water sampled at that pier on that day, not nearby beaches or fishing grounds.",
  chlNotToxin: "Chlorophyll measures algae biomass. High chlorophyll is not a toxic bloom; low chlorophyll does not rule one out.",
  reviewPending: "These pages have not yet been reviewed by an independent HAB scientist.",
  unavailable: "Measured observations are unavailable in this dataset.",
} as const;

export const FISHERIES_COPY = {
  heading: "Fisheries & Economic Exposure",
  lede: "How much California's commercial fisheries for toxin-affected species have landed in past years, from NOAA Fisheries landings data.",
  definition:
    "Historical fisheries exposure is the reported value of past commercial landings of species that marine toxins can affect. It is not a prediction of losses, not an estimate of harm, and says nothing about any current or future season.",
  portUnavailable: "Port-level values are not available",
  reviewPending: "Species tiers are CoastWatch's editorial grouping and await review by an independent HAB scientist.",
  unavailable: "Fisheries data are unavailable in this dataset.",
} as const;

/** Portfolio-preview introduction (src/components/DemoIntro.tsx). */
export const DEMO_COPY = {
  eyebrow: "Research preview",
  heading: "Harmful algal blooms on the California coast, on one map",
  lede:
    "Pseudo-nitzschia blooms produce domoic acid, a toxin that builds up in shellfish and fish. It has delayed Dungeness crab seasons and closed razor clam harvests on this coast, which matters to fishermen, shellfish harvesters, and tribal and coastal communities. The forecasts, satellite data and notices that describe a bloom are published in different places. CoastWatch puts them on one map, each labelled with its source and date.",
  layers: [
    { kind: "model", badge: "Agency forecast", name: "C-HARM", text: "NOAA's modeled probability of a bloom and of domoic acid on a 3 km grid: a nowcast and days 1–3, dated by issue." },
    { kind: "observation", badge: "Observation", name: "Satellite chlorophyll", text: "Sentinel-3 (300 m) and VIIRS (750 m), dated by the day each pixel was seen. Chlorophyll shows algae, not toxin." },
    { kind: "official", badge: "Not verified", name: "Official notices", text: "CDPH and CDFW advisories and closures, transcribed by CoastWatch with links to the agencies. Not checked by a person." },
  ],
  caveat:
    "An independent research project, not an advisory service. Not affiliated with or endorsed by NOAA, CDPH or CDFW, and not yet reviewed by an independent HAB scientist. CoastWatch does not decide whether seafood can be harvested or eaten:",
  cta: "Explore Monterey Bay",
} as const;
