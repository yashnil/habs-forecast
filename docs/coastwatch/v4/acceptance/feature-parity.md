# v2 → v4 feature parity (final pass, 2026-10-16)

Every capability the acceptance audit found v4 had removed or weakened, and where it stands now.
"Test" names the spec that holds it. Rebuilt in v4's architecture, not by reusing v2's components.

| # | v2 capability | v4 now | Provenance kept | Test |
|---|---|---|---|---|
| 1 | Home port: set, open, clear (device only) | Briefing place bar: "Make my home port", "Home: …", "Forget"; landing lists it first; never a point | "kept on this device only" | parity |
| 2 | Choose waters on My Coast's map | Locator: tap a harbour or the water → that place's briefing, trip kept | point named by its coordinates, "Point chosen on the map" | parity |
| – | Change harbour within My Coast | Place picker: harbours by region, CDPH sites; keeps the trip | – | parity |
| 3 | Combined currents + chlorophyll (`chl=1`) | "Compare with chlorophyll" on the currents layer; same URL | each source with its own time; the gap between them; "not combined into one value" | parity |
| 4 | Regional satellite coverage % | Layer panel: share of C-HARM ocean cells seen clearly, with dates and by region; what is counted | the published coverage definition, quoted | parity |
| 5 | Multi-sensor agreement lines | Research (Satellite · multi-sensor): which sensor supplied each pixel and its date, at a point, along a transect and in an area; an inter-sensor consistency check using the pipeline's published method (each VIIRS cell against ≥5 same-day Sentinel-3 pixels, log10), with the pipeline's own agreement figures quoted | consistency check, not validation; when no same-day pairs exist it says "cannot be computed" and why (the case in the fixture) | research |
| 6 | Per-harbour notice count | Notices control: "N here"; panel leads with the place's notices, then "may apply: not settled", then elsewhere | official geography; not verified; disputed never counted | parity, discrepancy, navigation |
| 7 | Per-variable port medians | Harbour panel "Advanced": per variable median, range, cells / pixels, size, dates | neighbourhood statistic, never at the harbour; not toxin | parity |
| 8 | Checklist ticks | "Before you go": the reader's own ticks per trip | "not a check by Neritic or by any agency" | parity |
| – | Notices drawer completeness | Notices panel: reasons, ages, expected-end warning, agency words, phone lines | unverified throughout | trust |
| – | Freshness (stale / historical / failed) | Layer panel and status centre | issue and observation dates | trust |
| – | Clouded day, radar gaps, nothing current | Said explicitly | "not clear water", "not calm water" | trust |
| – | Values by click/keyboard; v2 `inspect=` | Point view | product, resolution, time | trust |
| – | Knots, tidal note | Currents legend and point | – | trust |
| – | Satellite / radar sampling | Research point, transect and area for satellite (OLCI 300 m, VIIRS 750 m, multi) and HF radar (2 km, hourly and 24 h mean) | each pixel's own date or the radar hour, the pixel/cell footprint drawn, observed vs crossed counts and coverage, QC flags, empty = "no observation", native resolution, CSV with provenance | research |
| – | Offline completeness | Pack saves every resource Explore asks for inside its box (all satellite tile sets, every radar hour); 0 external requests offline; storage-full, damaged and older packs said truthfully | saved times kept; "nothing is live" | offline-recovery, offline proof |

Intentionally not restored: the guided tour and welcome card (replaced by the start card,
readiness and featured places), the Coastal/Research mode switch (replaced by workspaces).
