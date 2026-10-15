/**
 * The product's name and its standing statements, in one place.
 *
 * The project was called CoastWatch until 2026-10. NOAA runs a long-established program of that name
 * (NOAA CoastWatch, coastwatch.noaa.gov), whose datasets this project uses; to avoid any confusion
 * with it, the product is now Neritic Signal. "Neritic": the shallow waters over the continental
 * shelf, where these blooms, fisheries and harvests are.
 *
 * Only the product's own name changed. Every reference to NOAA CoastWatch as a data provider, its
 * hosts and product names, published citations and historical records keep their original words.
 * Stable technical identifiers (the repository, the coastwatch-web package and pipeline module,
 * storage keys, schema ids, the data branches and the public URL) keep their legacy names; see
 * docs/coastwatch/brand/neritic-signal.md for the list and the reasons.
 */
export const BRAND = {
  name: "Neritic Signal",
  tagline: "Coastal intelligence, grounded in evidence.",
  /** said where the sources are attributed and on About; never louder than that */
  independence: "Neritic Signal is an independent research project and is not affiliated with or endorsed by NOAA, CDFW, or CDPH.",
  /** for readers who knew the earlier name (About, saved briefings) */
  formerly: "Neritic Signal was called CoastWatch until October 2026. It is not NOAA CoastWatch, the NOAA program whose satellite data it uses.",
} as const;
