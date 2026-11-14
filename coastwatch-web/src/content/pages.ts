/**
 * Words of the informational pages (About, Sources, the legal pages, Fisheries, kept briefings, error pages) that
 * carry a disclosure, a caveat or an attribution. Moved here word for word from the v4 frames they lived in
 * (src/components/AppShell.tsx, src/pilot/shell/PilotShell.tsx, src/pilot/account/LegalPage.tsx,
 * src/components/coast/KeptView.tsx) when those pages moved onto the v14 design system (src/nx/pages): kept as data
 * so they stay testable and are never reworded by a layout change.
 */

/** The reference pages' footer (was AppShell's). */
export const SITE_FOOTER = {
  brand: "Neritic · research preview",
  disclaimer: "Not an official source and not independently validated. It does not replace CDFW or CDPH: closures, advisories and seafood safety come from them.",
} as const;

/** The legal pages' footer (was PilotShell's PilotFooter). */
export const LEGAL_FOOTER =
  "Neritic is an independent research preview, not an official source and not a seafood-safety service. Closures, seasons and advisories come from CDFW and CDPH; their notices always take precedence. Chlorophyll does not measure toxin. Forecasts are probabilities, not guarantees.";

/** Said in place of a page's data when the published dataset could not be read (was AppShell's). */
export const DATA_UNAVAILABLE = "Neritic’s data could not be loaded. Official closures and advisories are always on the CDFW and CDPH sites.";

/** The legal pages' head and foot (was LegalPage's). */
export const LEGAL_FRAME = {
  kicker: "Neritic research preview · legal",
  draftLead: "Draft — pending owner/legal review.",
  draftRest: "This text has not yet been reviewed by a lawyer and is not legal advice.",
  attributionLead: "Structure and some wording adapted from",
  attributionPolicies: "Basecamp’s open-source policies",
  attributionPoliciesUrl: "https://github.com/basecamp/policies",
  attributionLicensed: ", licensed",
  attributionLicence: "CC BY 4.0",
  attributionLicenceUrl: "https://creativecommons.org/licenses/by/4.0/",
  attributionRest: "; changed substantially for Neritic.",
} as const;

/** The registry line every informational page carries (the v4 trust pill's words). */
export const REGISTRY_LINE = {
  fixture: "Test data",
  fixtureSr: ": recorded fixture data, not a live feed",
  preview: "Research preview",
  previewStatement: ", not an official source and not a seafood-safety service. The list of notices may be incomplete.",
  checkAgencies: "Before harvesting or eating seafood, check CDFW and CDPH.",
  noData: "Neritic’s copy of the notices could not be loaded. Check CDFW and CDPH directly.",
  allNotices: "Every notice, with its wording, dates and sources",
} as const;

/** The kept briefing's words (was KeptView's). */
export const KEPT_COPY = {
  empty: "No briefing is kept on this device. On My Coast, choose “Keep a dated copy”.",
  kicker: "Neritic briefing · kept copy",
  offline: " · read offline",
  print: "Print",
  file: "Save as a file (opens without a connection)",
  delete: "Delete this copy",
  current: "Current information on My Coast",
  currentOffline: "My Coast (needs a connection)",
  agencies: "Official notices · CDFW and CDPH",
  agenciesNote: "No notice listed does not mean open or safe. The agencies are the authority; call before you go.",
  builtFrom: "Built from Neritic data generated",
  notAgency: "Neritic is an independent research project, not a government agency.",
  others: "Other kept copies on this device",
} as const;

/** The error and not-found pages. */
export const ERROR_COPY = {
  notFoundKicker: "Not found",
  notFoundTitle: "There is no page at this address",
  notFoundLede: "The link may be old or mistyped. Neritic’s pages are below; official notices always come from CDFW and CDPH.",
  errorKicker: "Something went wrong",
  errorTitle: "This page could not be drawn",
  errorLede: "Nothing on it is shown rather than a partial or out-of-date page. Try again, or open another page. Official closures and advisories are always on the CDFW and CDPH sites.",
  retry: "Try again",
} as const;

/** Sources read live when a page needs them (not part of the published dataset), for Sources. WCOFS and DINEOF
 *  take their names, links and licences from src/lib/ocean (the code that reads them). */
export const LIVE_SOURCES_COPY = {
  heading: "Live sources",
  intro: "Read by Neritic's server when a page needs them, not from the published dataset. Each is shown with its own time and source; when one cannot be reached, the page says so.",
  wcofsWhat: "Ocean model guidance on Explore: surface currents and temperature, 3-hourly nowcast and 72-hour forecast. NOAA labels it developmental; a model, not a measurement, and not a bloom or toxin prediction.",
  wcofsVia: "NOAA CO-OPS OPeNDAP (THREDDS)",
  dineofWhat: "Gap-filled chlorophyll on Explore: reconstructed where satellites saw nothing, shown as an estimate, never as an observation. Chlorophyll is algae, not toxin.",
  dineofCite: "Liu, X. & Wang, M. (2023), IEEE Transactions on Geoscience and Remote Sensing",
  rows: [
    { id: "ndbc", name: "NOAA National Data Buoy Center (NDBC)", what: "Buoy wind, waves and water temperature, as measured, with each reading's time.", url: "https://www.ndbc.noaa.gov/", licence: "U.S. Government work (NOAA), public domain" },
    { id: "nws", name: "National Weather Service marine text", what: "Coastal waters forecasts and advisories in NWS's own words, from its plain-text products.", url: "https://tgftp.nws.noaa.gov/data/forecasts/marine/coastal/pz/", licence: "U.S. Government work (NOAA), public domain" },
    { id: "coops", name: "NOAA CO-OPS Tides and Currents", what: "Tide predictions and water temperature at tide stations.", url: "https://tidesandcurrents.noaa.gov/", licence: "U.S. Government work (NOAA), public domain" },
  ],
} as const;
