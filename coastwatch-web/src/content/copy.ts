/**
 * User-facing copy that carries safety meaning lives here so it can be tested
 * (tests/unit/safety.test.ts). Official URLs were verified on 2026-10-08
 * (docs/coastwatch/evidence/sources-regulatory.md).
 */

export const OFFICIAL_STATUS = {
  heading: "Official closures and health advisories",
  notTracked:
    "CoastWatch does not track closures, quarantines or health advisories yet. They are not shown on this map, and nothing here means an area is open or clear.",
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

export const CHLOROPHYLL_COPY = {
  heading: "Satellite chlorophyll",
  biomass: "Chlorophyll measures algae biomass. It does not measure toxins and does not predict where fish are.",
  gaps: "Gaps are clouds or missing passes, not low chlorophyll.",
} as const;

export const DISCLAIMER =
  "CoastWatch brings together public data from state and federal agencies. It is not an official source. Closures, advisories and seasons from CDFW, CDPH and OEHHA always take precedence. Forecasts are probabilities, not guarantees.";

export const EXPERIENCES = [
  { key: "map", label: "Live Ocean Map", href: "/", available: true },
  { key: "bloom", label: "Bloom Intelligence", href: null, available: false },
  { key: "fisheries", label: "Fisheries & Economic Exposure", href: null, available: false },
  { key: "coast", label: "My Coast", href: null, available: false },
] as const;
