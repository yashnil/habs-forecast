# B. How California fishers find ocean, HAB and regulatory information today: a product critique

**Date of observation:** 2026-10-13. All pages were fetched on this date, between about 21:00 and 23:30 UTC.
**Scope:** This is desk research only. No one was contacted, and no accounts, subscriptions or forms were used.
**Builds on:** [`stakeholders-and-interviews.md`](../wt-ff/docs/coastwatch/research/stakeholders-and-interviews.md) §2.2–2.3 and [`ocean-sources.md`](../wt-ff/docs/coastwatch/research/ocean-sources.md). This document does not repeat their data-source evaluations; it looks at the *user-facing products*.

**Evidence tags:**
- **[O]** I observed this directly on the page or endpoint today; the URL is given.
- **[D]** The provider documents it, for example in an app-store listing or on its own help page, but I did not observe it working.
- **[R]** It comes from research, literature, or the earlier CoastWatch research docs.
- **[H]** It is a hypothesis or an inference.

> **Correction to the earlier doc §2.3 (fetched 2026-10-12):**
> - The Monterey County bivalve warning SN26-018 is **no longer active**. CDPH lifted it on **2026-10-12** [O: [CDPH Shellfish Advisories](https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx), "October 12, 2026 — … has lifted the shellfish safety notification today related to sport-harvested bivalve shellfish from Monterey County"].
> - The Del Norte **PSP** bivalve advisory was lifted **2026-07-14** [O: same page].
> - Del Norte razor clams went through three changes: closed 7/31, reopened 8/31 and closed again 10/9 [O: [CDFW In-Season page](https://wildlife.ca.gov/Fishing/Ocean/Regulations/Inseason)].

---

## 1. Matrix: each information need against the best current source

Ratings mean how well a California fisher is served *today* by the best available product:
- **Solved well**: CoastWatch should link to it.
- **Partial**
- **Poorly served**

| # | Information need | Best current source (URL) | Format | Timestamp clarity | Geographic clarity | Phone usability | Staleness / contradiction risk | Rating |
|---|---|---|---|---|---|---|---|---|
| 1 | **Official restrictions** (what is closed right now, any reason) | CDFW [Health Advisories](https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories) plus CDFW [In-Season Changes](https://wildlife.ca.gov/Fishing/Ocean/Regulations/Inseason) | HTML prose grouped by fishery. No map, no table [O] | **Poor.** The page has no "last updated" stamp. The two rock crab items carry no date [O] | Text boundaries: county lines, landmarks and latitudes [O] | Readable HTML of about 40 KB [O]. A fisher must read the whole page | **High.** The page omits the 10/12 Monterey lift and the Northern Channel Islands bivalve advisory (§5) [O] | **Partial** |
| 2 | **Fishery openings and closures (in-season changes)** | CDFW [In-Season Changes](https://wildlife.ca.gov/Fishing/Ocean/Regulations/Inseason) | A reverse-chronological HTML list, with summaries that include latitudes [O] | **Good.** Every item is dated [O] | Latitudes plus landmarks [O] | Good, but very long [O] | **Medium.** The hub page's own link points to a different URL that loops back to the hub (§5). Salmon also needs NOAA/PFMC [O] | **Partial (best single CA page)** |
| 3 | **Domoic acid and PSP advisories** | CDPH [Shellfish Advisories](https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx) and the CDPH [Bivalve Advisory Map](https://experience.arcgis.com/experience/394836318cfe4f7494e1c09097a43559). Phone lines (800) 553-4133 and CDFW (831) 649-2883 | A dated news list, plus an ArcGIS Experience map with county polygons [O] | The list is dated. Both pages show "Page Last Updated :" **blank** [O]. Popups are mixed: razor clam says "Updated October 8, 2026", the "Lifted" popup has no date [O] | **County-level only** [O] | The list is about 440 KB of SharePoint chrome, including a federal-shutdown banner [O]. The map needs ArcGIS JS [O] | **High.** Two agencies, two lists, different coverage (§5) | **Partial** |
| 4 | **Local sampling results** (shellfish, crab, phytoplankton) | CDPH [30-day bivalve PDF](https://cdphdata.maps.arcgis.com/sharing/rest/content/items/1f96ab557a014e2d8275b651a6447b1d/data), [razor clam PDF](https://cdphdata.maps.arcgis.com/sharing/rest/content/items/f349952bae944a17abb685669ff33be1/data), [crab DA PDF](https://www.cdph.ca.gov/Programs/CEH/DFDCS/Pages/FDBPrograms/FoodSafetyProgram/DomoicAcid.aspx), [phytoplankton map](https://experience.arcgis.com/experience/7edb5ccdfa2c4ca6b2852138847e0b32), [CalHABMAP pier data](https://calhabmap.org/datasites) | PDF tables with site lat/lon. The PDFs are dated "10/8/2026" [O] | Collection dates are per row [O]. Crab PDF title range conflicts with its link label (§5) | **Good.** Site name plus lat/lon per row [O] | Poor. Landscape PDF tables. One table repeats rows across a page break [O] | Medium. Monthly program reports end at **April 2026** [O] | **Partial (data good, delivery poor)** |
| 5 | **HAB forecasts** | NOAA CoastWatch C-HARM v3.1 via [ERDDAP](https://coastwatch.pfeg.noaa.gov/erddap/griddap/wvcharmV3_0day.html). CA HAB Bulletin ([SCCOOS](https://sccoos.org/california-hab-bulletin/)) | A science grid with no consumer viewer. A bulletin issued every 2–3 months [O] | Grid is current (2026-10-12 nowcast) [O]. Latest Bulletin issue is "2026: May & June" [O] | Pixel grid of about 3 km. No port or fishery framing [O/R] | **Poor.** No mobile viewer. The CalHABMAP "HAB Forecast" page links to a **dead v1 dataset ending 2022-03-17** [O] | **High.** Stale links, and the Bulletin lags by about 4 months [O] | **Poorly served** |
| 6 | **Marine weather warnings** | NWS Coastal Waters Forecast (e.g. [MTR CWF via api.weather.gov](https://api.weather.gov/products/types/CWF/locations/MTR/latest)), [marine point forecast](https://marine.weather.gov/MapClick.php?lat=36.7&lon=-122.0), NOAA Weather Radio / VHF WX | Plain text in caps and zone codes. Point forecast HTML is about 52 KB [O] | **Excellent.** "141 PM PDT Tue Oct 13 2026" plus expiry codes [O] | Zones (e.g. PZZ535 Monterey Bay) [O] | Text is light, but the zone codes are cryptic. NWS has no official marine app [R] | Low [O] | **Solved well** |
| 7 | **Tides and currents** | NOAA [Tides & Currents](https://tidesandcurrents.noaa.gov/) and its API | HTML, charts and API. Station page is about 39 KB [O] | Predictions are explicit [O] | Station-based (e.g. 9418767 Humboldt Bay N Spit) [O] | OK on phone. Many free apps re-skin it [R] | Low [O] | **Solved well** |
| 8 | **Buoy conditions** | NDBC [mobile](https://www.ndbc.noaa.gov/mobile/station.php?station=46042), CDIP | HTML. Mobile page is **2.6 KB**; desktop page is 95 KB [O] | Good ("4:00 pm PDT 13-Oct-2026") [O] | Station name plus distance ("27NM WNW of Monterey") [O] | **Excellent** [O] | Low. The mobile view shows only the latest row, so wave height was missing on a non-wave cycle [O] | **Solved well** |
| 9 | **Species-specific regulations** | CDFW [Sport Fishing hub](https://wildlife.ca.gov/Fishing/Ocean/Regulations/Sport-Fishing), [Groundfish Summary](https://wildlife.ca.gov/Fishing/Ocean/Regulations/Groundfish-Summary), [Ocean Salmon](https://wildlife.ca.gov/Fishing/Ocean/Regulations/Salmon), 2026 booklet PDF, Fish Rules app (third party) | HTML tables, Westlaw links, PDFs [O] | Groundfish: "Updated June 23, 2026". Salmon PDF: "Updated August 19" [O]. The hub has no date [O] | Management areas are defined by latitude text [O] | Mixed. About 90 species, most linking to Westlaw legal text [O] | **Medium.** The hub label says razor clam is "closed in Humboldt County" and omits Del Norte (§5) [O] | **Partial** |
| 10 | **Whale-entanglement restrictions (RAMP)** | CDFW [Whale Safe Fisheries](https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries) | HTML status plus stacks of PDFs per assessment, and a JPG zone map [O] | Each assessment is dated. The page itself is undated, and its labels mix the 2025-26 and 2026-27 seasons (§5) [O] | Zones 1–6, defined in a JPG and in latitude text [O] | Status is short ("Season is closed"). Detail is in PDFs [O] | Medium [O] | **Partial** |
| 11 | **Marine protected areas** | CDFW [MPA page](https://wildlife.ca.gov/Conservation/Marine/MPAs) and the [Ocean Sport Fishing map](https://wildlife.ca.gov/OceanSportfishMap) (ArcGIS Experience, item modified 2026-10-01) [O] | Interactive map. Regional PDFs [O] | MPA layer metadata says "as of January 1, 2019" [R: ocean-sources §1e] | Polygons are good, but "it is ultimately up to the user to determine if they are in an MPA" [O] | The map is marketed "from your cell phone" [O]. Needs JS and a connection [H] | **High in Oct 2026.** The FGC was to vote on 15 MPA petitions on Oct 15–16 [R: [BHA](https://www.backcountryhunters.org/news/details/california-mpa-review-advances-as-anglers-rally-for-science-based-management), [OPC](https://opc.ca.gov/2026/07/opc-releases-recommendations-on-mpa-petitions/)] | **Partial (good map, pending change)** |
| 12 | **Historical fisheries information** (past closures, toxin history, landings, disasters) | CDFW [Whale Safe archive](https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries), CDPH [monthly reports](https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Marine-Biotoxin-Monitoring-Reports.aspx), CDFW [Disaster Info](https://wildlife.ca.gov/Fishing/Ocean/Disaster-Info), HABMAP ERDDAP | Scattered PDFs. "Past monitoring reports are available upon request" [O] | Per document [O] | Varies [O] | Poor (PDF archives) [O] | Medium. CDPH monthly reports end at April 2026 [O] | **Poorly served** |

---

## 2. Workflows that are already solved well (CoastWatch should link to them, not rebuild them)

1. **Marine weather and warnings: NWS.**
   - Zone forecasts are timestamped to the minute, carry Small Craft Advisory headlines, and are machine-readable [O: api.weather.gov CWF MTR issued 2026-10-13T20:41Z; six CA Small Craft Advisories active].
   - CoastWatch may *display* the official text and alert verbatim, with its issuance time. It should not make its own wind or sea forecasts. (This is consistent with ocean-sources Rank 1.)
2. **Buoy observations: NDBC mobile.** The 2.6 KB page is a model of low-bandwidth design [O]. CoastWatch should deep-link to it per station.
3. **Tide predictions: NOAA CO-OPS** [O]. These are commodity data, and every app re-skins them [R]. Show the next low and high tide, and link to CO-OPS.
4. **The in-season change log: CDFW [Inseason page](https://wildlife.ca.gov/Fishing/Ocean/Regulations/Inseason).** It is dated, has summaries, and quotes latitudes [O]. It is the closest thing California has to a changelog. Link to it as "the official record".
5. **Email and text alerts: CDFW GovDelivery topics.**
   - Topics include "Domoic Acid, PSP Toxin Alerts", "Whale Safe Fisheries", Northern and Southern California news, and MPAs [O: [Marine Subscribe](https://wildlife.ca.gov/Regions/Marine/News/Subscribe)].
   - CoastWatch should *point people to subscribe* there, not run a parallel official-alert channel.
6. **Fish identification: CDFW fish ID pages and existing apps** [D]. These are out of scope.
7. **Licensing: the CDFW License App** (2024) [R: [CDFW news archive](https://wildlife.ca.gov/News/Archive/cdfw-introduces-license-application-for-mobile-devices)]. Out of scope.
8. **Offshore SST and chlorophyll chart services** (FishTrack, Terrafin, RipCharts, Hilton's) for pelagic targeting [D]. These are mature, paid products, and CoastWatch should not compete with them.

---

## 3. Tasks that need several sources: the hops for four concrete tasks

### Task A: "Will the Dungeness crab opener happen for Crescent City (Zone 1)?" (asked 2026-10-13)

| Hop | Where | What you get / the friction |
|---|---|---|
| 1 | CDFW [Whale Safe Fisheries](https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries) | "Commercial Fishery: Season is closed". This is pre-season and true but uninformative. No date is given for the next risk assessment. The last-season status is mislabelled (§5) [O] |
| 2 | CDFW [Crabs page](https://wildlife.ca.gov/Conservation/Marine/Invertebrates/Crabs) | Season framework. The newest dated item is May 15, 2026 [O] |
| 3 | CDFW [In-Season Changes](https://wildlife.ca.gov/Fishing/Ocean/Regulations/Inseason) | Nothing for 2026-27 yet. The last crab item is May 15, 2026 [O] |
| 4 | CDFW [Health Advisories](https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories) | "No closures" for Dungeness [O]. That is not the same as "tested clean" [H] |
| 5 | CDPH FDB [crab DA results PDF](https://www.cdph.ca.gov/Programs/CEH/DFDCS/Pages/FDBPrograms/FoodSafetyProgram/DomoicAcid.aspx) | The 2026-27 PDF so far contains **only Monterey (BMA G) 9/28/2026**. There are **no Crescent City / BMA A samples yet** [O]. The fisher must keep re-downloading it |
| 6 | Pre-season meat-fill (quality) testing | Linked from CDFW to PSMFC [O]. I could not find 2026 results [O: search]. Oregon's rule documents carry the Tri-State context [R: ODFW DFW_108-2025] |
| 7 | Oregon ODA [closures](https://www.oregon.gov/oda/food-safety/shellfish/pages/shellfish-closures.aspx) (the bloom moves across the border) | Oregon razor clams "CLOSED from the Washington border to the California border", while crab is "OPEN" [O, page updated 10/2/2026] |
| 8 | Phone (831) 649-2883, the port, the association (PCFFA [crab resources](https://www.pcffa.org/crab-resources)) | Trusted sources, but unstructured [R] |

**Result:** at least eight hops, and none of them answers "what gates remain, and when is the next decision?" [H]. Three independent gates apply: domoic acid testing (two clean sets at least 7 days apart [R]), meat fill, and RAMP. Each gate is published by a different body, in a different format.

### Task B: "Can I dig razor clams at Clam Beach (Humboldt) this weekend?"

| Hop | Where | Result |
|---|---|---|
| 1 | CDFW [Health Advisories](https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories) | "Humboldt: closed since May 2, 2024 … remains closed" [O] |
| 2 | CDPH [advisory map](https://experience.arcgis.com/experience/394836318cfe4f7494e1c09097a43559) | County-level popup: "advising consumers not to eat … razor clams from Del Norte or Humboldt … Updated October 8, 2026" [O] |
| 3 | CDPH [razor clam PDF](https://cdphdata.maps.arcgis.com/sharing/rest/content/items/f349952bae944a17abb685669ff33be1/data) (reached from the map's "Results" link) | Clam Beach on 8/28/2026 measured **86, 96 and 52 ppm** DA against an alert level of ≥20 ppm. There are no newer samples [O] |
| 4 | Regulations: 14 CCR §29.45 via Westlaw (odd/even-year beach alternation, limit 20, first 20 kept) | Legal text [R: stakeholders doc §2.1] |
| 5 | Tides: CO-OPS 9418767 | This week's minus tides fall **after dark**: −0.29 ft at 20:29 on 10/13 and −0.03 ft at 21:11 on 10/14 [O: CO-OPS API] |
| 6 | CDFW Sport Fishing hub | Label: "Clam, razor (fishery currently closed in Humboldt County)" [O] |

**Result:** the answer is "No (health closure)". It is findable in 1–2 hops, but *why* and *what would reopen it* (two clean samples) are spread across 3–4 places [O/H]. In Washington, by contrast, one map combines the DOH safety status with the WDFW season (§7).

### Task C: "Tomorrow's (Wed 10/14) marine weather off Monterey"

| Hop | Where | Result |
|---|---|---|
| 1 | NWS CWF MTR, zone **PZZ535 Monterey Bay** | "SMALL CRAFT ADVISORY IN EFFECT UNTIL 9 PM PDT THIS EVENING … WED…NE wind around 5 kt, backing to W in the afternoon. Seas 3 to 4 ft" [O] |
| 2 | The user must know which zone applies: PZZ535 (the bay) versus PZZ576 (Pt Pinos–Pt Piedras Blancas, 10–60 NM) versus PZZ530/565 | Zone codes are not self-explanatory [O/H] |
| 3 | NDBC 46042 / CDIP 46236 for current sea state | 46042: WNW 16 kt, 63 °F at 4:00 pm PDT [O] |
| 4 | Tides: CO-OPS 9413450 | [O] |
| 5 | Optional: Windy, PredictWind or Surfline for model graphics | Third-party [D] |

**Result:** this need is mostly solved. The friction is translating zones and reading text in all caps [H].

### Task D: "Groundfish depth limits for Half Moon Bay"

| Hop | Where | Result |
|---|---|---|
| 1 | CDFW [Groundfish Summary](https://wildlife.ca.gov/Fishing/Ocean/Regulations/Groundfish-Summary) | The user must map Half Moon Bay into the **San Francisco Management Area**, "38°57.5' N. latitude (Point Arena) to 37°11' N. latitude (Pigeon Point)". The town is not named [O] |
| 2 | Read the table | Jan 1–Mar 31 "Closed". Apr 1–Dec 31 "**Open all depths**" (nearshore and shelf/slope/lingcod). The page is "Updated June 23, 2026" and "subject to in-season change" [O] |
| 3 | Sub-limits and prohibited species | Separate PDFs and sheets for copper, vermilion/sunset and canary rockfish, and the no-retention species (quillback, yelloweye, cowcod, bronzespotted) [O] |
| 4 | Check the in-season page for groundfish changes | None in 2026 [O] |
| 5 | MPAs near Half Moon Bay (e.g. Montara SMR, Pillar Point SMCA) | The ArcGIS map [O: redirect]. Boundaries may be affected by the Oct 15–16 FGC vote [R] |
| 6 | Federal waters / Cordell Bank / RCA lines | NOAA pages, and eCFR for 50-fm waypoints [O: links on CDFW page] |

**Result:** the answer is simple today ("all depths, Apr–Dec"). The user still has to resolve the latitude, check sub-limits, check MPAs and check for in-season changes: about 4–6 hops [O/H].

---

## 4. Decisions that require interpreting complicated documents

1. **Latitude-boundary text.** Closures are defined as "from the Mendocino/Humboldt county line (40° 00.00' N. lat.) to Cape Mendocino (40° 30.00' N. lat.)" [O: Health Advisories]. A fisher has to convert this to their own grounds using a chart plotter. The CDFW pages also disagree on the latitude of the same landmark (§5).
2. **RAMP actions.** Each assessment has four PDFs: the Director's Declaration, the Final Assessment, the Working Group Recommendation, and the Preliminary Assessment and Available Data [O]. Combined actions read like "15% Gear Reduction and 30-fathom Depth Constraint in Zones 1–2 … Crab Trap Prohibition in Zones 3–5" [O]. Zone boundaries come from a JPG [O].
3. **CDPH result PDFs.** These are multi-page landscape tables.
   - Alert levels are stated inconsistently: crab "Viscera: >30 ppm" in the crab PDF header [O] versus "at or above action level (≥30ppm)" on the evisceration page [O].
   - Non-detects are coded "n", "ND", "<2.5" and "MM"-like blanks, with footnotes [O].
   - The 30-day bivalve PDF **repeats Santa Barbara, Ventura and San Diego rows** across the page break [O].
4. **Salmon.** The fall season is split by subarea ("38º02' N. lat. (near Point Reyes) to Pigeon Point Subarea, September 1–October 31") [O]. Federal in-season actions are relayed through CDFW and two hotlines: NMFS (800) 662-9825 and CDFW (707) 576-3429 [O].
5. **Groundfish.** Management areas by latitude, depth by fathom line (Rockfish Conservation Area waypoints in eCFR), and sub-bag limits in separate PDFs [O].
6. **Commercial regulations.** The [2026 Commercial Fishing Digest (PDF)](https://wildlife.ca.gov/Fishing/Commercial), "effective April 1, 2026 – March 31, 2027, unless otherwise noted", plus a seasons-calendar PDF [O].
7. **Reopening criteria.** These are rarely stated on the status pages. Oregon ODA states them inline ("two samples in a row … at least one week apart") [O]. The California status pages do not [O].

---

## 5. Stale or contradictory information observed (2026-10-13)

| # | Observation | URL | Tag |
|---|---|---|---|
| S1 | CDPH **lifted** the Monterey County bivalve warning on 10/12. At the agent's first fetch on 2026-10-13, the CDFW Health Advisories "Recent News" list still showed the 9/14 warning as the latest Monterey item, and the CDFW Marine Region home page did the same. **Rechecked by the lead researcher at 23:24 UTC on 2026-10-13:** the CDFW list now shows the 10/12 lift above the 9/14 warning. The finding is therefore a **lag of up to about a day between agencies**, not a lasting contradiction | [CDPH list](https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx); [CDFW HA](https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories); [Marine Region](https://wildlife.ca.gov/Regions/Marine/Fishery-Disasters) (this URL redirects to the Marine Region home) | [O] |
| S2 | The CDPH map shows the Monterey "Shellfish Safety Notification Lifted" layer as visible, and its webmap was modified 2026-10-13 16:39 UTC. The *lifted* popup carries **no date** | [webmap item](https://www.arcgis.com/sharing/rest/content/items/ad52187023f24b09b073d287f656df14/data?f=json) | [O] |
| S3 | The CalHABMAP "HAB Forecast" page links to `charmForecast0day`, which is **C-HARM v1, time_coverage_end 2022-03-17**. The current product is `wvcharmV3_0day` (to 2026-10-12). The ERDDAP metadata of the old dataset says "sensor: MODIS". The SCCOOS Bulletin page still describes C-HARM as merged with "MODIS Aqua" | [calhabmap.org/hab-forecast](https://calhabmap.org/hab-forecast); [ERDDAP info](https://coastwatch.pfeg.noaa.gov/erddap/info/charmForecast0day/index.html) | [O] |
| S4 | The CA HAB Bulletin page says "monthly", but the latest issue is "**2026: May & June**". CalHABMAP news is current to September 2026. The PNW HAB Bulletin's latest issue is **4 Oct 2026** | [SCCOOS](https://sccoos.org/california-hab-bulletin/); [NANOOS](https://www.nanoos.org/products/habs/forecasts/bulletins.php) | [O] |
| S5 | CDPH "Marine Biotoxin Monitoring Monthly Report": the newest is **April 2026**, a lag of about 5.5 months | [CDPH reports](https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Marine-Biotoxin-Monitoring-Reports.aspx) | [O] |
| S6 | The crab DA link is labelled "**September 29, 2026–October 8, 2026**", but the PDF header reads "**September 28, 2026 – September 28, 2026**" and "As of 10/8/2026" | [CDPH FDB](https://www.cdph.ca.gov/Programs/CEH/DFDCS/Pages/FDBPrograms/FoodSafetyProgram/DomoicAcid.aspx) | [O] |
| S7 | The CDPH FDB "News and Updates" newest entry is **May 7, 2025**. It omits the Jan 2026 Dungeness viscera advisories that are listed on the OPA page. Lobster DA results stop at Oct 2023 | same | [O] |
| S8 | The same landmark has different latitudes. Cape Mendocino is "40° 30.00' N" (rock crab closure, Health Advisories) but "40° 10.0' N" (In-Season page, crab). The 40°10' line is the federal "near Cape Mendocino" line | [HA](https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories); [Inseason](https://wildlife.ca.gov/Fishing/Ocean/Regulations/Inseason) | [O] |
| S9 | The commercial rock crab closure (Mendocino/Humboldt line to Cape Mendocino) and the recreational viscera advisory are **undated**. CDPH FDB lists "April 29, 2025 — CDPH Removes Previous Warning for Commercially Harvested Rock Crab from Portions of Humboldt and Del Norte". Whether the remaining CDFW closure is still current cannot be determined from the pages | HA; FDB | [O] observation / [H] possible conflict |
| S10 | The Sport Fishing hub says "Clam, razor (fishery currently closed in Humboldt County)" and omits the Del Norte closure of 10/9 | [Sport-Fishing hub](https://wildlife.ca.gov/Fishing/Ocean/Regulations/Sport-Fishing) | [O] |
| S11 | `.../Sport-Fishing/In-Season-Changes` 301-redirects to `/Fishing/Ocean/In-Season-Changes`, which renders the **hub page** again. The real changelog is at `/Fishing/Ocean/Regulations/Inseason`. This is an easy dead end | curl headers | [O] |
| S12 | The Whale Safe page's in-page nav says "Dungeness Crab **2025-26** Fishery Status", but the heading reads "2026-27 Fishery Status". The news line says all documents are "under the 2026-27 Season section", which holds the June 2026 (2025-26 season) assessments. The "Fishery Status Maps" block appears commented out | [Whale Safe](https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries) | [O] |
| S13 | The mussel-quarantine popup on the CDPH map links to the sub-page slug `…/page/2025-Statewide-Mussel-Quarantine` while describing the 2026 quarantine. The other popup links to `2026-…` | webmap JSON | [O] |
| S14 | Both CDPH pages show "Page Last Updated :" followed by nothing | CDPH OPA and DRSEM pages | [O] |
| S15 | The underlying CDPH county "advisory symbol" feature layers were last edited **2020-08-25**. Status is toggled by webmap filters, so the data carry no issue date, reason or SN number | [Health_Advisory_Symbols_(Red)](https://services2.arcgis.com/wi1yEacfYjH5viqb/arcgis/rest/services/Health_Advisory_Symbols_(Red)/FeatureServer/0) | [O] |
| S16 | The CDFW "Current Fishery Disasters" section lists Red Sea Urchin 2018–2019 as current. NOAA's determinations list includes "163. California Red Sea Urchin Fishery, 2023/2024", which is absent from the CDFW page | [CDFW Disaster Info](https://wildlife.ca.gov/Fishing/Ocean/Disaster-Info); [NOAA](https://www.fisheries.noaa.gov/national/funding-financial-services/fishery-resource-disaster-determinations) | [O] |
| S17 | The OEHHA advisories page sits behind an Incapsula bot challenge (950-byte interstitial) and could not be read automatically. The SCCOOS and Surfline sites return HTTP 403 to non-browser clients. This suggests fragile access for low-end or embedded browsers | curl | [O] / [H] |

---

## 6. Missing location or fishery specificity, and missing source timestamps

- **County-only granularity for bivalves.** The CDPH map has one polygon per county. Its symbol layers are county centroids, including inland Contra Costa, Napa and Solano [O]. A Santa Cruz wharf picker cannot tell whether a beach 2 km away was sampled.
- **No reason or date per active status on CDFW.** Rock crab items are undated. Health Advisories has no page-level "as of" date [O].
- **No "next decision" date anywhere in California**:
  - The RAMP status gives no next-assessment date pre-season [O].
  - There is no CDPH re-test schedule [O].
  - Contrast this with FWC's "The next status report will be issued on Friday, October 16" [O: [FWC](https://myfwc.com/research/redtide/statewide/)] and WDFW's tentative digs "dependent on final approval of marine toxin testing … about a week or less before" [O: [WDFW](https://wdfw.wa.gov/fishing/shellfishing-regulations/razor-clams)].
- **"No closure listed" is ambiguous.** A missing listing is not evidence that a species was tested and is safe, for example market squid and Dungeness pre-season [O/H]. No California page distinguishes "tested, below action level" from "not tested".
- **Fishery naming does not match fisher language.** Pages use "Biotoxin Management Area G", "Fishing Zone 3" and "San Francisco Management Area", while users think in ports ("Half Moon Bay", "Crescent City") [O/H].
- **Commercial versus recreational differences are buried.** For example, the anchovy rules differ: commercial dead bait is allowed if labelled, and recreational take is limited to bait [O: Inseason page].
- **HAB forecast has no fishery translation.** C-HARM gives the probability that particulate DA is ≥500 ng/L [O: calhabmap], with no link to which fisheries or ports are exposed [O/H].

---

## 7. Good patterns from analogues, and anti-patterns

### Patterns worth borrowing

| Pattern | Example | Tag |
|---|---|---|
| **Page-level "last updated" stamp** | ODA "This webpage last updated: 10/2/2026" | [O] [ODA](https://www.oregon.gov/oda/food-safety/shellfish/pages/shellfish-closures.aspx) |
| **Status by species, with a plain-language extent** | ODA: razor clams "CLOSED from the Washington border to the California border for high levels of … domoic acid". Mussels and crab "OPEN…" | [O] ODA |
| **Reopening rule stated inline** | ODA: "requires two samples in a row … at least one week apart" | [O] ODA |
| **Multilingual status** | ODA: English, Español, русский, 中国人, Tiếng Việt, 한국인 | [O] ODA |
| **One map that combines safety and harvest season, by beach and species** | WA [Shellfish Safety Map](https://fortress.wa.gov/doh/biotoxin/biotoxin.html): "Washington shellfish resources are managed by both the DOH (safety & water quality) and WDFW (harvest regulations)". Legend includes "Closed for Razor Clams only", "WDFW Harvest Seasons Closed", "Conditionally Open"; lightweight at 24 KB | [O] |
| **Scheduled cadence plus "next update" date** | FWC: "Status Update for October 9, 2026 … next status report … Friday, October 16". The daily sample map "updated daily at 5 p.m." and covers the last 8 days | [O] FWC |
| **Tentative schedule, explicitly conditional on toxin tests** | WDFW razor clam digs (40 tentative days Oct 9–Dec 27), with a domoic acid caveat and history | [O] WDFW |
| **Timely, event-driven bulletin** | PNW HAB Bulletin issues on 24 Sep and 4 Oct 2026, during the bloom | [O] NANOOS |
| **Ultra-light mobile view** | NDBC mobile, 2.6 KB | [O] |
| **Topic-granular email/text alerts** | CDFW GovDelivery topics; FWC email subscription; Maine DMR "Subscribe to Email/Text Notifications" | [O] |
| **Per-row site and lat/lon in result tables** | CDPH result PDFs (good data despite PDF delivery) | [O] |

### Anti-patterns to avoid

- Status pages with no "as of" date, or a blank "Page Last Updated" field (CDFW HA, CDPH) [O].
- Status that lives only in map-layer visibility toggles, with no dated record behind it (CDPH map, S15) [O].
- Map-only answers that need heavy JavaScript (ArcGIS Experience Builder for both the CDPH advisories and the CDFW regulations map) [O]. At sea on weak signal, this will likely fail [H].
- Cumulative PDFs as the sole delivery of results, with no notifications (CDPH crab and razor) [O/R].
- Two agencies keeping parallel lists that drift apart (S1, S7, S9) [O].
- Dead links to retired datasets (S3), and redirect loops (S11) [O].
- Undated legal-latitude text with inconsistent landmark coordinates (S8) [O].
- "Monthly" products that lag by months (S4, S5) [O].

---

## 8. Third-party and consumer apps

Prices are as documented by the vendor or a store tracker and are not verified in checkout. None of these apps shows California HAB or biotoxin status.

| App / service | What it solves well | Price | HAB / biotoxin / regulatory status? | Offline |
|---|---|---|---|---|
| **Windy** | Model wind and wave visualisation, free tier | Free + Premium (price not confirmed) [D] | No | Limited [H] |
| **PredictWind** | Multi-model wind and routing, departure planner | Basic about US$29/yr, Standard about $249, Pro about $499 [D: [help](https://help.predictwind.com/en/articles/14132215-do-you-have-monthly-payment-plans-for-the-pro-or-standard-subscriptions), [yacht.de](https://www.yacht.de/en/sailing-knowledge/navigation/the-20-most-important-sailing-apps-2026-everything-about-weather-navigation-and-trip-planning/)] | No | Offshore/satellite modes [D] |
| **Buoyweather / FishWeather / Surfline** | Point marine forecasts; surf cams (Surfline) | Freemium subscriptions [D, not verified] | No | No [H] |
| **FishTrack** | SST, chlorophyll and true-colour charts for offshore targeting | Premium about $14.99/mo or $95.99/yr [D: App Store] | Shows chlorophyll, which **users may misread as toxin risk** [H] | Partial [D] |
| **Terrafin / RipCharts / Hilton's** | Pro SST, chlorophyll and altimetry charts | Terrafin about $99–119/yr [D, conflicting]. RipCharts and Hilton's by subscription [D] | No | Downloadable charts [D] |
| **Navionics (Garmin)** | Charts, depth contours, community edits | About $49.99/yr US+Canada [D: third-party guide] | No. Some MPA overlays [H] | Yes, downloaded charts [D] |
| **Fishbrain** | Social catch log, spots | In-app purchases $5.99–$79.99 [D: [apppricinglab](https://www.apppricinglab.com/iap/apple/477967747)] | No | Partial [H] |
| **Fishing Points / logbook apps** | Waypoints, tides, solunar, logs | Freemium [D] | No | Yes [D] |
| **Fish Rules** | Location-aware bag/size/season rules. **Covers California** recreational saltwater [D: [App Store](https://apps.apple.com/us/app/fish-rules-local-fishing-laws/id597875361)] | Free; "ProStaff $29.99" IAP [D] | Seasons and limits. **No health advisories listed**. Disclaimer: "do not have legal force or effect" [D] | "Works offline" [D] |
| **Fish Rules Commercial** | Commercial rules | — | Regions: South Atlantic and Gulf only. **Not California** [D] | — |
| **FishVerify** | Photo ID plus GPS-based limits. California claimed in an undated press item [D] | Free + subscription [D] | Size and bag limits only [D] | Unknown |
| **CDFW License App** | Digital license; links to regulation booklets [R] | Free | Booklets only [R] | License display [R] |
| **NWS** | — | — | **No official NWS marine app.** Users rely on the web, VHF WX channels and NOAA Weather Radio [R] | VHF works without cell coverage [R] |

**Gap [H]:** no app, official or commercial, joins *where I fish* to *what is currently closed for health or whale reasons, since when, and why*. Fish Rules comes closest on regulations, but it has no biotoxin layer.

---

## 9. Official disaster-assistance and adaptation resources (direct sources only)

| Resource | URL | What it offers | Tag |
|---|---|---|---|
| NOAA Fisheries: Fishery Resource Disaster Assistance | https://www.fisheries.noaa.gov/national/funding-financial-services/fishery-resource-disaster-assistance | Process overview: Governor request, Secretary of Commerce determination, appropriation, spend plan | [O] page live |
| NOAA Fisheries: Disaster Determinations (list) | https://www.fisheries.noaa.gov/national/funding-financial-services/fishery-resource-disaster-determinations | Numbered list, including #163 CA Red Sea Urchin 2023/2024 and #144 and #130 CA salmon | [O] |
| CDFW: Federal Fishery Disaster Relief | https://wildlife.ca.gov/Fishing/Ocean/Disaster-Info | CA request letters, determinations and spend plans (salmon 2023/2024/2025, urchin, sardine, Dungeness and rock crab 2015-16). Contact: salmondisaster@wildlife.ca.gov | [O] |
| PSMFC Federal Fishery Disaster Relief | https://relief.psmfc.org/ | Administers payments. CA contact CAFishDisaster@psmfc.org, (888) 517-7266. Notice dated May 19, 2026: "payments … are considered taxable income". The 2023 salmon application closed June 1, 2025 [O: CDFW page] | [O] |
| SBA disaster assistance / EIDL | https://www.sba.gov/funding-programs/disaster-assistance (redirects to https://www.sba.gov/disaster/) | EIDL working-capital loans. Small fishing businesses can be eligible when SBA issues a declaration tied to a Commerce fishery disaster | [O] page live; eligibility [D] |
| California Sea Grant (extension) | https://caseagrant.ucsd.edu/ | Extension advisors and fisheries outreach. Among the highest-trust sources for fishers [R: stakeholders doc S3] | [O] home page live (the guessed `/extension` path 404s) |
| CDFW Whale Safe: Trap Gear Retrieval Program, Alternative Gear authorizations | https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries | Adaptation: pop-up gear authorizations (first authorised March 2026) and retrieval program | [O] |

USDA: none found for wild-capture ocean fishers. I did not verify this further [H].

---

## 10. Opportunity statement

**Where CoastWatch can be a trusted synthesis layer [H, grounded in the [O] findings above]:**

1. **A port- and fishery-centric status card that cites its sources.** For example: "Crescent City · Dungeness (commercial)". It would list the three gates (DA testing, meat fill, RAMP), each with:
   - the official source link,
   - that source's own date,
   - "last checked by CoastWatch" time,
   - plain-language extent.

   Agencies publish these gates separately today (Task A, 8+ hops).
2. **Source freshness and disagreement shown openly.** When CDFW and CDPH disagree or lag (S1, S7, S9), show both, dated, with, for example, "CDPH lifted on 10/12; CDFW page not yet updated as of <time checked>" while the lag lasts. Never resolve the conflict silently.
3. **Translate boundaries into the user's frame.** Convert the latitude text into a drawn line on the map and the name of the nearest port, labelled "approximate, see official text". This addresses S8 and Task D.
4. **Turn sampling results into dated, sited facts.** For example: "Clam Beach razor clams 8/28: 52–96 ppm (alert ≥20). No newer sample." Build this from the CDPH PDFs and lat/lon, with a link to the PDF. Present it as **observations, not forecasts**.
5. **Make the HAB forecast usable**, as the current C-HARM v3.1 with its correct date. That replaces the dead v1 link (S3). Frame it as *bloom/particulate DA probability, not seafood toxicity or a closure prediction*. The PINN matches, and does not beat, the ConvLSTM. Chlorophyll is not toxin.
6. **A "what would change this" and "when is the next decision" line**, but only when an official source states it, for example the two-clean-samples rule and RAMP assessment cadence. Borrow the ODA and FWC patterns.
7. **Low-bandwidth, dated, text-first status**, in the style of NDBC mobile or the WA map, with deep links to the official map-heavy pages.

**What CoastWatch must NOT do or duplicate:**

- **Do not issue or imply official status.**
  - Never mark anything open, closed, lifted or human-verified on its own. That status changes only through `cwp review-official` by a person.
  - Never say "safe to eat" or "you may fish here".
  - "No listing" must render as "no official notice found", not "open/safe".
- **Do not rebuild NWS forecasts, buoy pages, tide predictions, or licensing and fish-ID tools.** Link to them, or display them verbatim with their timestamps.
- **Do not restate bag limits, seasons or depth tables as rules.** There is no machine-readable source [R: ocean-sources §1e]. Link to CDFW and Fish Rules-class tools.
- **Do not compute MPA legality** (the boundary layer is dated 2019, petitions are pending) [R/O].
- **Do not run a competing official alert channel.** Point users to the CDFW GovDelivery topics and the agency hotlines. Any CoastWatch notification must be clearly labelled "unofficial digest; check source".
- **Do not compete with paid offshore SST and chlorophyll chart services** for targeting.
- **Do not present observations as forecasts, or chlorophyll as toxin.**

---

### Verification items before using externally

- Re-check the S1 lag on the CDFW Health Advisories page. CDFW may update it within days of 10/12.
- Confirm the outcome of the FGC MPA petition vote (Oct 15–16, 2026) on fgc.ca.gov.
- Confirm third-party app prices in-store (all tagged [D]).
- S9 (rock crab commercial closure currency) needs a call to the CDFW line (831) 649-2883. That call must be made by the user; this research made no calls.
