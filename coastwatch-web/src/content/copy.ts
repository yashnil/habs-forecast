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

const km = (v: number) => (v < 10 ? v.toFixed(1) : v.toFixed(0)).replace(/\.0$/, "");
const mg = (v: number) => (v >= 10 ? v.toFixed(0) : v.toPrecision(2));

/**
 * Chlorophyll near a place (lib/chlNear): how a headline was made, and pixels set aside. Pixels on
 * land or right against the shore, and single pixels far above the water around them, are never a
 * place's headline; they stay on the map. No fixed ceiling: real blooms here have exceeded 300 mg/m³.
 */
export const CHL_NEAR_COPY = {
  /** how a neighbourhood headline was made: "Median of 33 clear water pixels within 1.5 km · range 0.91–9.2 · nearest 0.5 km away" */
  basis: (b: { n: number; radiusKm: number; min: number; max: number; nearestKm: number }) =>
    [`Median of ${b.n} clear water pixel${b.n === 1 ? "" : "s"} within ${km(b.radiusKm)} km`, mg(b.min) === mg(b.max) ? null : `range ${mg(b.min)}–${mg(b.max)}`, `nearest ${km(b.nearestKm)} km away`].filter(Boolean).join(" · "),
  /** the highest pixel set aside, when it is far above the headline (one line, about that pixel) */
  aside: (a: { value: number; reason: "shore" | "spike" }) =>
    a.reason === "shore"
      ? `One pixel right against the shore read ${mg(a.value)} mg/m³. Pixels this close to land are often wrong (glare from land, rocks and surf); it is not used here.`
      : `One pixel nearby read ${mg(a.value)} mg/m³, more than ten times the water around it. A lone pixel that far above its neighbours is usually a measurement error (such as glare or a cloud edge), not a bloom; it is not used here.`,
  flagged: (place: string, radiusKm: number) =>
    `No reading from open water within ${km(radiusKm)} km of ${place}. The only clear pixels there are on land, right against the shore or far out of line with the water around them, and they are not treated as a measurement.`,
  /** My Coast's "what is not known" list */
  unknownFlagged:
    "No satellite reading from open water near these waters in the last week: the only clear pixels nearby are on land, right against the shore or far out of line with the water around them.",
  noCoastline: "The coastline could not be loaded, so pixels on land or right against the shore could not be set aside.",
} as const;

export const DISCLAIMER =
  "CoastWatch brings together public data from state and federal agencies. It is not an official source. Closures, advisories and seasons from CDFW, CDPH and OEHHA always take precedence. Forecasts are probabilities, not guarantees.";

/** Primary navigation. Only built experiences appear: unbuilt features are not navigation. */
export const EXPERIENCES = [
  { key: "map", label: "Ocean Map", short: "Map", href: "/" },
  { key: "coast", label: "My Coast", short: "My Coast", href: "/coast" },
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
  /** the source CSV link (original upstream data, not CoastWatch's chart) */
  csvLabel: "Download source data (CSV)",
  csvWhat:
    "The original CalHABMAP measurements for this station since 2014, exactly as SCCOOS ERDDAP serves them: not a copy of the chart above. Units are in the second row, NaN means not measured, and a 0 is a reported 0 (not quantified).",
  csvOutage: "If SCCOOS is unavailable the file will not load; try again later, or open the station's dataset page.",
  csvStale: "SCCOOS did not respond when CoastWatch last checked this station, so the download may fail too.",
} as const;

export const FISHERIES_COPY = {
  heading: "Fisheries & Economic Exposure",
  title: "What California’s toxin‑affected fisheries have landed",
  lede: "How much California's commercial fisheries for toxin-affected species have landed in past years, from NOAA Fisheries landings data.",
  definition:
    "Historical fisheries exposure is the reported value of past commercial landings of species that marine toxins can affect. It is not a prediction of losses, not an estimate of harm, and says nothing about any current or future season.",
  atStake: "Past landings show what was at stake in earlier seasons. They are not losses, not a forecast of losses, and say nothing about the current season.",
  portUnavailable: "Port-level values are not available",
  reviewPending: "Species tiers are CoastWatch's editorial grouping and await review by an independent HAB scientist.",
  unavailable: "Fisheries data are unavailable in this dataset.",
  /** what the figures can and cannot establish (M7 §5) */
  can: [
    "How much revenue and weight each species group landed statewide in each calendar year since 2015, as NOAA reports it.",
    "How large these fisheries are compared with each other and with all California commercial landings.",
    "When documented regulatory events, such as the 2015–16 domoic acid closures, fell relative to the annual totals.",
  ],
  cannot: [
    "Harm from algal blooms. A change between years is not a loss, and an overlap in time does not show a cause: markets, regulations, ocean conditions, weather and fishing effort all move landings.",
    "Profit or income. Ex-vessel revenue is what buyers paid at the dock, before costs, and it excludes processing, retail and tourism.",
    "Fishing effort. Pounds landed say nothing about trips, traps or time spent fishing.",
    "Anything about a port, county or the current season. The figures are statewide annual totals.",
  ],
} as const;

/**
 * Documented regulatory events marked on the Fisheries chart as historical context only. Each needs
 * an official source; the chart never attributes a change in landings to one.
 */
const CRAB_TIMELINE = {
  label: "California Legislature, Joint Committee on Fisheries and Aquaculture: 2015–16 Commercial Crab Fishing Timeline (July 2016)",
  url: "https://fisheries.legislature.ca.gov/sites/fisheries.legislature.ca.gov/files/u8/2015-16%20Commercial%20Crab%20Fishing%20Timeline%20.pdf",
};

export const FISHERIES_EVENTS = [
  {
    id: "dungeness-2015-16",
    start: "2015-11-06",
    end: "2016-05-26",
    continues: false,
    label: "Dungeness crab season delayed for domoic acid",
    short: "Dungeness delayed",
    period: "Nov 2015 – May 2016",
    detail:
      "On 6 November 2015 CDFW delayed the commercial Dungeness crab season, scheduled for 15 November, after domoic acid was found in crab. It reopened south of the Sonoma–Mendocino county line on 26 March 2016 and along the whole coast on 26 May 2016.",
    source: CRAB_TIMELINE,
    groups: ["dungeness_crab"],
  },
  {
    id: "rock-crab-2015-16",
    start: "2015-11-06",
    end: "2016-07-21",
    continues: true,
    label: "commercial rock crab closed for domoic acid",
    short: "Rock crab closed",
    period: "From Nov 2015",
    detail:
      "On 6 November 2015 CDFW closed the commercial rock crab fishery, which usually runs all year, after domoic acid was found in crab. It reopened in stages from 31 December 2015; on 21 July 2016 it was still closed north of Pigeon Point (San Mateo County).",
    source: CRAB_TIMELINE,
    groups: ["rock_crab"],
  },
] as const;

/** The published research behind the project, and how it relates to what the app shows. */
export const RESEARCH = {
  title: "Physics-Guided Neural Forecasts of Nearshore Harmful Algal Blooms in the California Current System",
  author: "Yashnil Mohanty",
  venue: "Proceedings of the 6th World Conference on Climate Change and Global Warming, Vol. 2 No. 2 (2025)",
  published: "Published January 29, 2026",
  doi: "https://doi.org/10.33422/ccgconf.v2i2.1619",
  relation:
    "The paper develops physics-informed machine-learning forecasts of nearshore chlorophyll-a. CoastWatch does not run those research models: its forecast layer is NOAA's C-HARM, and its observations come from NOAA CoastWatch, HF-radar networks, CalHABMAP, NOAA Fisheries and the CDFW and CDPH notice pages. The publication covers the research, not this application, which has not been reviewed by an independent HAB scientist.",
};

/** The About page (M8): what CoastWatch is, the kinds of information, its limits, the research, providers. */
export const ABOUT = {
  title: "Harmful algal blooms on the California coast, in one place",
  lede:
    "CoastWatch is an independent research project, published as a research preview: no independent scientist or agency has reviewed it, and it is not a regulatory or seafood-safety service. It brings together the public information about harmful algal blooms along California’s coast: official notices, NOAA’s bloom forecast, satellite and shore-station measurements, and past fisheries landings. Each is labelled for what it is.",
  kindsIntro:
    "Every page uses the same four colours, and they never stand for anything else. A forecast is never shown as a measurement, chlorophyll is never shown as toxin, and history is never shown as a prediction.",
  kinds: [
    {
      kind: "official",
      name: "Notices from CDFW and CDPH",
      text: "Closures, advisories and quarantines, quoted from the agencies’ pages. They come first on every page. They are transcribed and not yet checked by a person, so the agency pages are always the authority.",
      where: "See the notices on the map",
      href: "/",
    },
    {
      kind: "model",
      name: "NOAA’s C-HARM forecast",
      text: "The modelled probability of Pseudo-nitzschia and of domoic acid, the toxin it can produce, at 3 km, for today and the next three days. A probability, not a measurement and not a closure decision.",
      where: "Ocean Map",
      href: "/",
    },
    {
      kind: "measured",
      name: "Satellite and shore-station measurements",
      text: "Chlorophyll seen from space (algae biomass, not toxin), surface currents from HF radar, and water samples from CalHABMAP piers tested for domoic acid and Pseudo-nitzschia, each with its own date.",
      where: "Bloom Intelligence",
      href: "/bloom",
    },
    {
      kind: "historical",
      name: "Past fisheries landings",
      text: "What California’s toxin-affected fisheries landed each year since 2015, from NOAA Fisheries, beside the 2015–16 domoic acid closures. History, not a forecast of losses.",
      where: "Fisheries",
      href: "/fisheries",
    },
  ],
  cannot: [
    "It does not decide whether seafood can be eaten or an area is open to fishing or harvest: only CDFW, CDPH and OEHHA do, and a missing notice here does not mean an area is open or seafood is safe.",
    "How much toxin is in seafood. Shore stations measure toxin in seawater; satellites measure chlorophyll, which is algae, not toxin.",
    "What will happen. C-HARM forecasts are probabilities for the next three days, and past landings say nothing about any current or future season.",
    "Anything finer than its sources: forecast cells are 3 km, water next to the shore is often not modelled, and port-level landings are not published.",
  ],
  research: {
    what:
      "The paper studies forecasting nearshore chlorophyll-a, a proxy for algal blooms, eight days ahead along the California coast with neural networks, including one constrained by the physics of advection and diffusion.",
    result:
      "Evaluated with the same diagnostics, the physics-informed model matches, but does not beat, a standard ConvLSTM baseline; it modestly improves spatial fidelity and training stability.",
    relation:
      "CoastWatch does not run those research models. Its forecast is NOAA’s operational C-HARM, and its observations come from the agencies below. The publication covers the research, not this application: CoastWatch has not been peer reviewed or reviewed by an independent HAB scientist.",
  },
  providersIntro:
    "CoastWatch reads public data from these providers every six hours, checks it against the source, and shows the last good data with its own dates when a source fails.",
  providers: [
    { name: "NOAA C-HARM v3.1", what: "Bloom and domoic acid forecast (NOAA CoastWatch West Coast ERDDAP)" },
    { name: "NOAA CoastWatch, NASA GIBS", what: "Satellite chlorophyll: Sentinel-3 OLCI 300 m, VIIRS 750 m" },
    { name: "HFRNet (IOOS)", what: "HF-radar surface currents, 2 km, hourly" },
    { name: "CalHABMAP (SCCOOS ERDDAP)", what: "Shore-station domoic acid, Pseudo-nitzschia, chlorophyll and temperature" },
    { name: "NOAA Fisheries FOSS, BLS CPI-U", what: "Commercial landings since 2015, and the inflation index for constant dollars" },
    { name: "CDFW, CDPH", what: "Official closures, advisories and landing ports" },
    { name: "OpenFreeMap, OpenMapTiles, OpenStreetMap; NOAA NCEI", what: "Basemap, coastline and seafloor relief" },
  ],
  repo: "https://github.com/yashnil/habs-forecast",
  review: [
    "The official notices are transcribed from the agency pages and have not been checked by a person. They may be incomplete.",
    "The species tiers on the Fisheries page are CoastWatch’s grouping and await independent scientific review.",
    "The application has not been reviewed by an independent HAB scientist.",
  ],
};

/** The guided tour of the Ocean Map (one short step per control; never a safety statement). */
export const TOUR = {
  steps: {
    what: {
      title: "What CoastWatch shows",
      body: [
        "Harmful algal bloom information for the California coast in one place: NOAA's C-HARM model, satellite and shore-station measurements, and official notices. Each is labelled for what it is.",
        "CoastWatch is not an official source. Closures and advisories come from CDFW and CDPH.",
      ],
    },
    place: {
      title: "Choose a place",
      body: ["Search for a region, port or shore station here. The shield counts the official notices that may apply to the place you choose."],
    },
    forecast: {
      title: "The forecast is a model",
      body: [
        "C-HARM gives the probability that Pseudo-nitzschia or domoic acid exceeds a threshold, today and up to three days ahead, on a 3 km grid. Choose the quantity and the day here.",
        "It is a probability, not a measurement of toxin in water or seafood, and a low probability does not mean an area is safe.",
      ],
    },
    satellite: {
      title: "What satellites saw",
      body: [
        "Satellite chlorophyll shows where algae were seen on a given day. Chlorophyll is not toxin, and clouds or fog often leave the coast without data.",
      ],
    },
    gaps: {
      title: "Empty water is missing data",
      body: [
        "Hatched or empty water has no value. C-HARM has no toxin values for many cells within about 3–6 km of shore, or for bays such as San Francisco Bay; satellites miss cloudy days. CoastWatch never fills these in.",
        "Click or tap any spot on the water to see why it is empty.",
      ],
    },
    port: {
      title: "A port and what was measured nearby",
      body: [
        "A port shows its official notices first, then the forecast near it and the nearest shore-station measurements, each with its date. Measurements describe the water at one pier on one day.",
      ],
    },
    official: {
      title: "Official restrictions",
      body: [
        "Closures and advisories come only from CDFW and CDPH. CoastWatch lists them with links to the agencies; it has not verified them, and the list may be incomplete. This button opens the list from any page.",
      ],
    },
    pages: {
      title: "Bloom Intelligence and Fisheries",
      body: [
        "Bloom Intelligence follows one shore station's measurements over time. Fisheries shows past landings of species that marine toxins can affect.",
      ],
    },
    limits: {
      title: "Not a safety clearance",
      body: [
        "CoastWatch does not say that seafood is safe or that an area is open, and a missing notice is not an all-clear. Before you fish or harvest, check CDFW and CDPH, or call the CDPH shellfish line at 1 (800) 553-4133.",
      ],
    },
  },
} as const;

/** County reference layer (CA Department of Technology boundaries, public domain). */
export const COUNTIES = {
  attribution: "County boundaries: CA Dept. of Technology / CDTFA",
  note: "Reference only, drawn on land. County lines do not define fishery closures, crab fishing zones, state waters, or how far a health advisory reaches offshore.",
  source: "California Department of Technology, California County Boundaries and Identifiers (boundaries from CDTFA; public domain), simplified to about 100 m.",
} as const;
