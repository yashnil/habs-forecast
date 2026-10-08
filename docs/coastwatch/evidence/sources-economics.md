> **Evidence log.** Raw verification notes from live requests on 2026-10-08 (UTC times inside). Summarized in [`../02-data-sources.md`](../02-data-sources.md). Re-verify before relying on any value; sources change.

# California commercial landings and HAB economics: verified data sources

Checked on 2026-10-08, between 16:27 and 16:40 UTC. Every request was made live with curl or WebFetch.
Labels used below:
- **VERIFIED** means I loaded the page or got a response from the endpoint and saw the content.
- **UNVERIFIED** means I could not confirm it. The reason is given each time.

---

## TL;DR

| Source | Best use for MVP | Port resolution | Years (verified) | Value $? | Machine access | License / terms |
|---|---|---|---|---|---|---|
| **CDFW MFDE** | **Primary source**: CDFW is the official CA data owner, and the data reaches 2025 | 9 CDFW port areas, plus individual ports | 1980–2025 (2026 not yet loaded) | Yes, annual and monthly | Undocumented JSON backend (`mfde-api.wildlife.ca.gov`); Power BI visuals with Excel export | CA site "public domain unless otherwise indicated"; data confidential under FGC §8022, so suppressed cells show `-1` |
| CDFW legacy CCL PDFs | Historical cross-check | Port areas and ports | ≤2019 (last edition published) | Yes | PDF only | Same as above |
| CALFISH (Dryad) / wcfish | Long history before 1980 | Ports and port complexes | 1941–2019 | Yes | XLSX on Dryad; `.rda` in the R package | **Dryad: CC0-1.0**. **wcfish GitHub: no license** |
| **PacFIN APEX (public)** | Second source, and port groups that match the CA–OR–WA convention | 10 CA port groups (Crescent City is split out from Eureka) | 1979–2026 YTD | Yes (revenue $, mt, price) | APEX web reports with CSV/Excel download; no documented API | Public site applies a "≥3 vessels and ≥3 dealers" rule; no explicit license found |
| NOAA FOSS | Statewide sanity check only | **State only** (per species); "Top US Ports" gives totals only | Commercial CA through 2024 | Yes (nominal) | **Documented ORDS JSON API** | Non-confidential only; terms page not loadable |

---

## 1. CDFW Marine Fisheries Data Explorer (MFDE): VERIFIED

**URLs (all returned HTTP 200):**
- Landing page: https://wildlife.ca.gov/Conservation/Marine/Data-Management-Research/MFDE
- User guide: https://wildlife.ca.gov/Conservation/Marine/Data-Management-Research/MFDE/User-Guide
- App, embedded via iframe: https://mfde.wildlife.ca.gov/ — subpages include:
  - `/summarize/CaliforniaCommercialLandings/`
  - `/visualize/LandingsSummary`
  - `/summarize/PercentLandingsByMajorPortOrPorth` (the spelling is CDFW's)
  - `/CustomQueries`
- The old URL https://wildlife.ca.gov/Fishing/Commercial/Landings now redirects to the MFDE landing page.

**Technology (found by inspecting the page):**
- The MFDE is a React single-page app.
- It embeds **Microsoft Power BI (Government cloud, `app.powerbigov.us`)** reports for these visuals:
  - Landings by Value/Participation
  - Block Summary
  - Top Species
  - Average Price
  - Percent Landings by Port
- The user guide says those visuals have a "More Options … Data Export feature."
- The CCL tables and Custom Queries are drawn from a JSON backend. The host is `mfde-api.wildlife.ca.gov`, read from the app's JS bundle `main.f9bb2738.chunk.js`.
- The DataTables export button produces Excel. I found no CSV button in the bundle.

**Backend endpoints found in the JS bundle (all POST with a JSON body):**
- `/api/ccl/byportarea`
- `/api/ccl/byportarea_A`
- `/api/ccl/byport`
- `/api/ccl/monthlyreports`
- `/api/ccl/originwaters`
- `/api/cpfv/statewide`
- `/api/cpfv/portside`
- `/api/landing/CustomQuery`
- `/api/Token/getToken` (Power BI embed token)

I tested three of them live:

| Endpoint | Request body | Response | Fields |
|---|---|---|---|
| `/api/ccl/byportarea` | `{"year":"2025"}` | HTTP 200, 1,130 rows | `ccL_Category, speciesName, year, portArea, totalPounds, totalValue` |
| `/api/ccl/byport` | `{"year":"2024","portAreaId":"1"}` | HTTP 200, 140 rows | `majorPort, port, speciesName, year, totalPounds, totalValue` |
| `/api/landing/CustomQuery` | columns year, month, speciesName, portArea; filters `startDate:"012024"`, `endDate:"122024"`, `portArea:[3]` (MMyyyy dates) | HTTP 200 | `y, m, sn, mp, v, l` (value and pounds) |

- `byportarea` also returns per-species state "Total" rows and a "Grand Total" row for each port area.
- Year coverage of `byportarea`: 1980 and 2025 return data. 1979 and 2026 return **HTTP 204 (empty)**.
- `CustomQuery` gives **monthly value and pounds by port area and species**.
- `/api/ccl/monthlyreports` returns monthly **pounds only** (no value).
- Swagger is not exposed (`/swagger/index.html` and `/swagger/v1/swagger.json` both return 404).
- **Caveat:** this backend is undocumented and internal to the app. CDFW has made no public commitment to it, so the request shape could change. For production, cache annual extracts and/or confirm with CDFW (contact on the user guide page).

**Coverage and granularity (from the user guide plus live checks):**
- Data runs from 1980 "to the most recent complete calendar year." The latest available year is **2025**, and 2026 returns nothing (verified).
- Dimensions available:
  - year and month
  - species, species group, and species management group
  - **port area**: Eureka, Fort Bragg, Bodega Bay, San Francisco, Monterey, Morro Bay, Santa Barbara, Los Angeles, San Diego, plus Sacramento Delta, Inland Waters, and Unknown
  - port
  - gear, condition, use
  - fishing block
- **CDFW's Eureka port area includes Crescent City.** The MFDE Port Reference Table (2023) lists port 201 Crescent City under port area EUREKA. Source: https://nrm.dfg.ca.gov/FileHandler.ashx?DocumentID=216411&inline (PDF, verified).
- Other reference tables (PDFs, verified HTTP 200):
  - Species: DocumentID=216412
  - Gear: 225459
  - Condition: 216409
- The user guide says the "California Commercial Landings (CCL) tables" now live in the MFDE and can be produced from 1980 on. It adds: "Due to data confidentiality regulations, the CCL tables in the MFDE may not match those previously provided online."
- **Caveat from the user guide:** for species whose seasons span calendar years (e.g., Dungeness crab), interpret the final year with caution.

**Confidentiality (VERIFIED from the user guide):**
- "Pursuant to California Fish and Game Code Section 8022, commercial landings data is considered confidential … summarized and presented so as not to disclose data from an individual or business." Cells marked "Confidential" had "insufficient data."
- In the API, suppressed cells are returned as **`-1`** in both pounds and value.
  - About 44% of rows in the 2024 and 2025 port-area tables are suppressed (475/1096 and 482/1130).
  - The state "Total" row for a species can itself be `-1`. Example: Dungeness 2023 and 2025.
  - Port-area "Grand Total" rows appear to **include** confidential landings. In 2024, the Eureka port-area grand total ($20,018,255) equals the byport GRAND TOTAL.
- **"Rule of three" for CDFW: UNVERIFIED as an official CDFW rule.**
  - The user guide gives no numeric threshold, except for the CPFV tables: "confidential if fewer than three CPFVs are represented."
  - FGC §8022 (text read via the third-party mirror law.onecle.com) only requires summaries "so as not to disclose the individual record or business of any person."
  - leginfo.legislature.ca.gov returned 403 to automated requests.

**Terms:**
- The CDFW Conditions of Use (https://wildlife.ca.gov/Conditions-of-Use, verified) say: "information presented on this web site, unless otherwise indicated, is considered in the public domain."
- The MFDE disclaimer says data are "provided as-is." It says MFDE "is not intended to be used for management purposes," and that CDFW "recommends users consult with CDFW prior to data use."
- Update cadence: annual, after scientists review the data. E-tickets have been mandatory since 2019-07-01, but the public MFDE lags to the last complete year.

**Sample values (2024 Dungeness crab, MFDE byportarea, nominal $):**
- San Francisco: $20.97M
- Eureka: $16.89M
- Bodega Bay: $9.63M
- Monterey: $1.15M
- Fort Bragg: $1.09M
- Morro Bay: $8.3k
- State total: $49,747,568

The state total matches NOAA FOSS (PacFIN source) for 2024 CA Dungeness: $49,745,602.

---

## 2. CDFW legacy "Final California Commercial Landings" (CCL) tables: VERIFIED (historical only)

- **Status:** discontinued as stand-alone publications. The last edition is **2019**.
  - The archived legacy page reads: "For landings information after 2019, call the Marine Fisheries Statistical Unit at (562) 342-7130."
  - Wayback snapshot: http://web.archive.org/web/20220119033955/https://wildlife.ca.gov/Fishing/Commercial/Landings (verified).
- The MFDE user guide confirms the CCL tables were "published online between 2000 and 2019" and that "MFDE will now be the home."
- The PDFs themselves are **still live on nrm.dfg.ca.gov**. 2019 edition (HTTP 200, `application/pdf`; filename from the Content-Disposition header):

| Table | Content | URL |
|---|---|---|
| Table 15 | pounds and value by port area | https://nrm.dfg.ca.gov/FileHandler.ashx?DocumentID=178022&inline (`Table15_2019_ADA.pdf`) |
| Table 15a | — | `DocumentID=178023` (`Table15a_2019_ADA.pdf`) |
| Table 16PUB | by port, Eureka area | `DocumentID=178025` (`Table16PUB_2019_ADA.pdf`) |
| Table 17PUB | by port, San Francisco area | `DocumentID=178026` (`Table17PUB_2019_ADA_corrected.pdf`) |
| Table 18PUB | by port, Monterey area | `DocumentID=178027` (`Table18PUB_2019_ADA.pdf`) |
| Tables 14MB, 14SD | — | `178020`, `178021` |
| Appendix B | introduction and source of data | `DocumentID=178008` |

  - The full set of Document IDs for 2019 (178003–178036, plus 184471) and for 2018 (171049–171086…) can be read from the Wayback snapshot.
- **Format:** PDF only. Example from Table 16PUB 2019: Crescent City Dungeness crab, 5,623,572 lb, $19,199,222.
- For any year from 1980 to 2025, **use the MFDE** in preference to these PDFs. The MFDE re-applies current confidentiality rules, so numbers may differ.

---

## 3. CALFISH database (Dryad) and the `wcfish` R package

### CALFISH, doi:10.25349/D9M907: VERIFIED via the Dryad API v2
- Landing page: https://datadryad.org/dataset/doi:10.25349/D9M907
- Title: "The CALFISH database: A century of California's non-confidential fisheries landings and participation data."
- **License: CC0-1.0** (`https://spdx.org/licenses/CC0-1.0.html`).
- Published 2022-02-22, version 4, 7.2 MB.
- Paper: Free, C.M., Vargas Poulsen, C., Bellquist, L.F., Wassermann, S.N., Oken, K.L. (2022). *Ecological Informatics* 69:101599. https://doi.org/10.1016/j.ecoinf.2022.101599 (DOI resolved).
- Files, all XLSX:
  - **`CDFW_1941_2019_landings_by_port_species.xlsx`** (4.3 MB): annual pounds and value by port and species, 1941–2019
  - `CDFW_1936_2019_landings_by_waters_species.xlsx`
  - `CDFW_1934_2020_n_comm_vessels.xlsx`
  - `CDFW_1934_1956_n_comm_vessels_by_port_complex.xlsx`
  - `CDFW_1916_2020_n_fishers_by_area_of_residence.xlsx`
  - CPFV files
  - kelp files
  - `README.txt`
- Content: data extracted from 58 CDFW landings reports (1928–2020). Port names and species names are harmonized. The data are non-confidential, i.e., as published.
- **Not updated after 2019.** No dataset modifications since 2022-02-22.

### wcfish (https://github.com/cfree14/wcfish): VERIFIED via the GitHub API
- **No license.** The GitHub license field is null, and the DESCRIPTION file literally reads `License: What license is it under?`.
  - Without an explicit license, code and bundled data in the repo cannot be safely redistributed.
  - **Pull the CC0 Dryad files instead.**
- Last push: 2025-07-29.
- Tables documented in `R/data.R`:

| Table | Content |
|---|---|
| `cdfw_ports` | landings in lb and USD by port and port complex (1987–2019 typology) and species, 1941–2019 |
| `cdfw_waters` | — |
| `cdfw_n_comm_vessels_port` | — |
| `noaa` | FOSS state-level data, 1950–2019 |
| `pacfin_all1` | 1980–2024 |
| `pacfin_all2` | 1980–2022 |
| `pacfin_all5` | port complex × species, 1980–2020; documentation says "More coming soon" |
| `pacfin_all6` | — |
| `pacfin_crab2` | monthly Dungeness by port complex, 1980–2020, with a confidentiality flag |
| `pacfin_ports` | — |
| `ports` | — |
| `blocks` | — |

---

## 4. PacFIN APEX public reports: VERIFIED

- Dashboard: https://reports.psmfc.org/pacfin/f?p=501:1000:::::: (HTTP 200, "Public" session, with an optional Login).
- Reports and code lists open without login. Page IDs were read from the dashboard links:

| Report | Description | URL |
|---|---|---|
| **ALL001** | WOC All Species | https://reports.psmfc.org/pacfin/f?p=501:1:0:INITIAL:::: |
| **ALL005** | Monthly Commercial Landed Catch by Port Group: mt, Revenue, Price/lb | https://reports.psmfc.org/pacfin/f?p=501:5:0:INITIAL:::: |
| ALL006 | by Month | `f?p=501:6` |
| CRAB001 | Monthly Dungeness Crab by State | `f?p=501:401` |
| **CRAB002** | Monthly Dungeness Crab by Port Group | https://reports.psmfc.org/pacfin/f?p=501:402:0:INITIAL:::: |
| **CODE015** | Port code list | https://reports.psmfc.org/pacfin/f?p=501:815:0:INITIAL:::: |

- Other reports exist: CPS001/002 (coastal pelagic species), GMT005 (groundfish by port group), GSAFE005 (IOPAC port engagement/dependence), and SAFE portals.
- **Years:** the year selector offers **1979–2026**. ALL005 and CRAB002 default to 2026 year-to-date, with data present.
  - CRAB002, crab year 2026 (Nov 2025–Oct 2026), all CA Dungeness: 4,009.5 mt, **$44,713,941** (CRAB001 CA row).
- **Download:** the APEX "Content Selector" offers CSV and Excel. I found **no documented REST API**. Automation would mean scraping APEX sessions, which is fragile, so prefer manual or periodic CSV downloads.
- **California port group codes**, each confirmed by filtering CODE015 (all have agency CDFW):

| Code | Name |
|---|---|
| CCA | CRESCENT CITY AREA PORTS |
| ERA | EUREKA AREA PORTS |
| **BGA** | **FORT BRAGG AREA PORTS** (not Bodega) |
| **BDA** | **BODEGA BAY AREA PORTS** |
| SFA | SAN FRANCISCO AREA PORTS |
| MNA | MONTEREY AREA PORTS |
| MRA | MORRO BAY AREA PORTS |
| SBA | SANTA BARBARA AREA PORTS |
| LAA | LOS ANGELES AREA PORTS |
| SDA | SAN DIEGO AREA PORTS |
| CA2 | OTHER OR UNKNOWN CALIFORNIA PORTS |
| OCA | OTHER OR UNKNOWN CALIFORNIA PORTS |
| ACA | ALL CALIFORNIA PORTS |

  - The guessed code "BGA = Bodega Bay" in the task brief is **wrong**: BGA is Fort Bragg and BDA is Bodega Bay.
  - PacFIN has **10** CA port groups (Crescent City is split out), while CDFW has **9** marine port areas (Crescent City sits inside Eureka).
  - ALL005 column headers confirm the 10: Crescent City, Eureka, Fort Bragg, Bodega Bay, San Francisco, Monterey, Morro Bay, Santa Barbara, Los Angeles, San Diego, plus "UNKNOWN CA PORT."
- **Confidentiality (VERIFIED):**
  - ALL005 description: "Data that involve fewer than three vessels or dealers have been withheld to preserve confidentiality."
  - The APEX User Manual (https://pacfin.psmfc.org/wp-content/uploads/2019/03/PacFIN_APEX_Reports_User_Manual_v20.pdf) says that on the public site "non-confidential catch summaries will include a minimum of 3 vessels and 3 dealers in the aggregation."
  - Suppressed cells are shown as `-` with a `*` flag. The CRAB002 footer also describes secondary suppression of subtotals.
  - **This is the verified "rule of three."**
- **Units:** metric tons (landed or round weight); revenue in nominal $; price per lb. Crab years run November–October.
- **Terms/license: UNVERIFIED.** I found no explicit license or terms of use on pacfin.psmfc.org, its About page, or the APEX pages. Cite "PacFIN APEX, PSMFC, retrieval date" and treat the data as public summaries.
- Operational note: the PacFIN homepage logged outages in April 2025 (PSMFC office move), and service has since been restored.

---

## 5. NOAA Fisheries FOSS: VERIFIED (API); main site page blocked

- `https://www.fisheries.noaa.gov/foss` returned **403 Access Denied** (Akamai bot block) to curl.
- The app itself works: https://apps-st.fisheries.noaa.gov/foss/f?p=215:200 (HTTP 200).
- **Documented API** (the "API for Developers" page is `f?p=215:35`): `https://apps-st.fisheries.noaa.gov/ods/foss/landings/` (Oracle ORDS JSON).
  - Filters go in `q=` as JSON. Paging uses `offset` and `limit`, with 1,000 rows per request by default.
  - Fields: `tsn, ts_afs_name, ts_scientific_name, region_name, state_name, year, pounds, dollars, tot_count, source, collection`.
  - Verified query: `?q={"year":2024,"state_name":"CALIFORNIA","collection":"Commercial","ts_afs_name":{"$like":"%DUNGENESS%"}}` returned `CRAB, DUNGENESS`, 14,173,706 lb, **$49,745,602**, source `PACFIN`.
  - Year 2025 returned 0 rows, so **CA commercial data currently runs through 2024**.
- **Port resolution:**
  - The landings API is **state-level only**. There is no port field, and `/ods/foss/ports`, `/port_landings`, and `/landings_by_port` all returned 404.
  - The "Top US Ports" page (`f?p=215:11`) lists ports such as Crescent City, Eureka, Fort Bragg, Bodega Bay, Monterey, Moss Landing, Morro Bay, Avila Beach, Los Angeles, and Dana Point for 1981–2024. These are **total** values only, not by species.
  - **FOSS is not suitable for port × species exposure.**
- **Caveats (metadata page `f?p=215:240`, verified):**
  - Results are non-confidential only; confidential data are grouped as "WITHHELD FOR CONFIDENTIALITY."
  - Values are **nominal ex-vessel** dollars.
  - The data are "updated weekly."
  - Mollusks are reported in **meat weight**, so they differ from state data.
- License/terms: UNVERIFIED (the www.fisheries.noaa.gov page was blocked). As US federal government data it is generally public domain, but I did not confirm a statement on the page.

---

## 6. HAB economic-impact references

All DOIs below were resolved via Crossref, OpenAlex, or Semantic Scholar on 2026-10-08.

**Federal disaster determination, VERIFIED** (via WebFetch of https://www.fisheries.noaa.gov/national/funding-and-financial-services/fishery-disaster-determinations):
- Entry 67, **"California Dungeness Crab and Rock Crab, 2015-2016"**:
  - Determination: Approved, 01/18/2017
  - Cause: "Natural Causes (harmful algal bloom)"
  - Allocated: **$25,797,268**
- The request letter's estimated loss figure: **UNVERIFIED** (not shown on the page).
- I saw no other CA Dungeness or domoic-acid entries in the portions I read. California sardine and red sea urchin disasters exist but are not HAB-attributed on that page.

**Recent domoic-acid delays and closures, VERIFIED** (CDFW Dungeness crab page, https://wildlife.ca.gov/Conservation/Marine/Invertebrates/Crabs/Dungeness-Crab, and the Health Advisories page):
- **10/24/2025:** "Director Declaration of Season Delay in Commercial Dungeness Crab Fishery…"
- **12/12/2025:** recreational Dungeness closure in northern CA "Due to … Elevated Levels of Domoic Acid."
- **1/8/2026 and 1/23/2026:** northern CA commercial Dungeness delay (Reading Rock MPAs to Cape Mendocino), declared and then lifted.
- **11/2024:** "Public Health Hazard Will Delay November 2 Recreational Dungeness Crab Opener in Far Northern California."
- I did not enumerate a full 2016–2024 history of delays: **UNVERIFIED as a complete list**.
  - CDFW posts the Director Declarations on the Dungeness page; scrape those PDFs to build a closure calendar.
  - Some delays are for meat quality or whale entanglement, not HABs, so classify each by its cause text.

**Peer-reviewed studies (DOIs verified):**

| Study | Citation | DOI | Note |
|---|---|---|---|
| Holland & Leonard (2020) | "Is a delay a disaster? Economic impacts of the delay of the California Dungeness crab fishery due to a harmful algal bloom." *Harmful Algae* 98:101904 | https://doi.org/10.1016/j.hal.2020.101904 | Abstract: the 2015/16 CA opening was delayed "almost five months." Crab revenue losses were "less than was initially estimated when a request for disaster assistance was submitted," but fishers lost "revenue from other fisheries equal in magnitude." |
| Moore et al. (2019) | "An index of fisheries closures due to harmful algal blooms and a framework for identifying vulnerable fishing communities on the U.S. West Coast." *Marine Policy* 110:103543 | https://doi.org/10.1016/j.marpol.2019.103543 | A HAB closure index for 17 communities, 2005–2016. Crescent City, Fort Bragg, and Moss Landing had the highest social vulnerability. |
| Moore et al. (2020) | "Harmful algal blooms and coastal communities: Socioeconomic impacts and actions taken to cope with the 2015 U.S. West Coast domoic acid event." *Harmful Algae* 96:101799 | https://doi.org/10.1016/j.hal.2020.101799 | Survey of 16 communities; 84% negatively impacted. |
| Fisher et al. (2021) | "Climate shock effects and mediation in fisheries." *PNAS* 118(2):e2014379117 | https://doi.org/10.1073/pnas.2014379117 | Fishery participation shifted during the closures and then rebounded. |
| Ritzman et al. (2018) | "Economic and sociocultural impacts of fisheries closures in two fishing-dependent communities following the massive 2015 U.S. West Coast harmful algal bloom." *Harmful Algae* 80:35–45 | https://doi.org/10.1016/j.hal.2018.09.002 | — |
| McCabe et al. (2016) | "An unprecedented coastwide toxic algal bloom linked to anomalous ocean conditions." *GRL* 43 | https://doi.org/10.1002/2016GL070023 | Oceanographic context. |

**MFDE context numbers** (nominal $, calendar year, CA Dungeness total; values with `-1` were suppressed, so the per-area sum is used where needed):

| Year | Value |
|---|---|
| 2013 | at least $70.6M (sum of non-suppressed port areas) |
| 2014 | at least $66.9M (sum of non-suppressed port areas) |
| **2015** | **$17.1M** |
| 2016 | $83.2M |
| 2017 | $47.1M |

These figures alone are not a loss estimate. **Calendar-year totals do not line up with the crab season**: the delayed 2015–16 season's landings fell in calendar 2016.

---

## 7. HAB / domoic-acid-sensitive California commercial species (CDFW / CDPH sources)

| Species / group | Evidence (verified pages) | MFDE species name(s) |
|---|---|---|
| **Dungeness crab** | CDFW Health Advisories page: OEHHA, in consultation with CDPH, recommends closures and delays under **FGC §5523** when domoic acid (DA) ≥ the federal action level, and CDFW implements them. CDPH FAQ: the action level is **crab viscera > 30 ppm, crab meat > 20 ppm**. CDPH does "pre-season testing of Dungeness crab." Repeated 2024–2026 delays are listed (see §6). | `Crab, Dungeness` |
| **Rock crab** | Current (Oct 2026) **commercial rock crab DA closure**, Mendocino/Humboldt line (40°00′N) to Cape Mendocino (40°30′N). CDPH advises against eating rock crab viscera from the CA/OR border to the Mendocino/Sonoma line and around the Northern Channel Islands. Rock crab is included in the 2015–16 federal disaster. | `Crab, red rock`, `Crab, yellow rock`, `Crab, brown rock`, `Crab, rock unspecified` |
| **Spiny lobster** | The CDFW Health Advisories page has a "Spiny Lobster Fisheries: Open and Closed Ocean Waters" section (no closures today). The CDPH FAQ notes CDPH "may increase sampling of … crab, lobsters" and that cooking does not remove DA "in the viscera" of crab or lobsters. The specific historical lobster closure dates are **UNVERIFIED** here. | `Lobster, California spiny` |
| **Razor clam** | Recreational razor clam fishery closed in Humboldt County since 2024-05-02. Del Norte reopened 2026-08-31 after DA closure. Razor clam is **recreational only** in CA, so it carries **no commercial landings exposure**. | n/a |
| **Bivalve shellfish / aquaculture** (mussels, oysters, clams) | CDPH marine biotoxin monitoring (PSP + DA); the annual sport-harvested mussel quarantine runs May 1–Oct 31. Commercial shellfish harvesters must submit weekly samples, and "commercially harvested mussels from certified companies are not included in the quarantine." 9/14/2026 CDPH warning on sport bivalves in Monterey County. | `Mussel`, `Oyster, giant Pacific`, `Clam, purple` (often suppressed). **Most aquaculture production may not appear in CDFW landing receipts**, which is UNVERIFIED and should be confirmed with CDFW Aquaculture. |
| **Northern anchovy / Pacific sardine** | **9/11/2026: commercial and recreational take restriction on northern anchovy in Monterey Bay** (Pigeon Point to Point Lobos) because of DA. Commercially taken fish may be sold only as dead bait. CDPH 9/14/2026 advisory: do not eat sardines or anchovies from the area. | `Anchovy, northern`, `Sardine, Pacific` |

- OEHHA's own domoic-acid page (https://oehha.ca.gov/fish/general-info/marine-biotoxin-domoic-acid-fish-and-shellfish) **could not be read**: Incapsula returned "Request unsuccessful." OEHHA's role is verified only through the CDFW page.
- Not on the list, but worth a judgment call: market squid, the CA fishery's largest by value at $67.9M statewide in 2024. It is **not** a documented DA-closure species in the sources I checked, so exclude it from "HAB-sensitive" unless the team finds evidence.

---

## 8. Recommended MVP metric: "Historical HAB-sensitive landings exposure by port area"

**Definition (descriptive, not a loss estimate):**

For each port area *p* and each recent year *y* (default: the last N = 5 complete years, 2021–2025):

```
V_sens(p,y)  = Σ value of landings of HAB-sensitive species s ∈ S at p in y
V_tot(p,y)   = port-area Grand Total value (includes confidential landings)
Share(p,y)   = V_sens / V_tot
Exposure$(p) = mean_y [ V_sens(p,y) × CPI(2025)/CPI(y) ]    # real 2025 USD
ExposureShare(p) = Σ_y V_sens / Σ_y V_tot                   # value-weighted share
```

**Settings:**
- **Species set S, tier 1 (with documented commercial closures):**
  - Dungeness crab
  - rock crabs (red, yellow, brown, unspecified)
  - northern anchovy
- **Tier 2 (advisories, monitoring, or historical):**
  - spiny lobster
  - Pacific sardine
  - commercial mussels, oysters, clams
- Show the tiers separately, with a toggle.
- **Primary data:** MFDE `byportarea` (annual) and `CustomQuery` (monthly, so a seasonal profile can be drawn, such as Dungeness value concentrated in Nov–Mar). Use MFDE `byport` for a port drill-down.
- **Cross-check:** PacFIN ALL005/CRAB002. Show PacFIN when the user wants Crescent City separated from Eureka.
- **Inflation:** BLS CPI-U, series `CUUR0000SA0` (BLS public API v2 verified, latest value Aug 2026 = 334.980), or the FRED `CPIAUCSL` CSV (verified). Use annual averages. Label the results as "real 2025 USD." Ex-vessel values are nominal in every source.
- **Why N = 5:**
  - It balances the high volatility in Dungeness prices and volume (2023 Eureka $39.4M vs 2024 $16.9M).
  - It keeps the period after electronic tickets started.
  - Also show the median and min–max, and allow N = 10.

**Caveats to display in the UI:**
1. **Confidential suppression (`-1`).** Suppressed cells must be counted as "unknown," never as 0.
   - Report V_sens as a **lower bound** and flag "includes suppressed cells."
   - Small ports and species are disproportionately hidden.
   - The Grand Total includes confidential landings, so Share is biased low when sensitive cells are suppressed.
2. **This measures exposure (value at risk), not impact.** It is the value of landings that *could* be affected.
   - Historical closures did not erase this value. Delays shift landings in time, and prices respond.
   - Holland & Leonard (2020) found that crab losses were smaller than the disaster request, but that spillover losses in other fisheries were comparable in size.
3. **Calendar year vs season.** Dungeness seasons span Nov–Jul. Annual totals mix two seasons.
4. **Port attribution is the landing port, not the fishing location.** Vessels move between ports during closures.
5. **CDFW and PacFIN groupings differ**: Crescent City is inside Eureka in CDFW and its own group in PacFIN. Do not mix the two sources in one chart without a crosswalk.
6. **Nominal-to-real conversion** uses national CPI-U. It is not a seafood price index.
7. **No aquaculture, processing, or tourism impacts.** Ex-vessel value understates community economic dependence. Multipliers from input-output models are out of scope for the MVP.
8. **Licensing:**
   - MFDE and the CDFW site are public domain "unless otherwise indicated," and need attribution.
   - CALFISH Dryad is CC0.
   - **Do not redistribute wcfish repo data** (no license).
   - PacFIN terms are unverified, so attribute it and link back.
9. **Data latency:** MFDE currently runs to 2025. PacFIN covers 2026 YTD (preliminary). FOSS runs to 2024.

**Hard separation from modeled loss:**
- Present this metric under a heading such as **"Historical exposure (observed landings)."** Keep it in a separate panel, with separate colors and its own data-provenance footer, from any **"Modeled expected revenue loss."**
- A modeled loss would combine forecast HAB probability, the probability of a closure given a HAB, the duration of the closure, and recovery or substitution behavior.
- **Never multiply exposure by a forecast probability and label the product "loss" without a validated closure-to-revenue model.**
- The modeled number must carry its own uncertainty band and its own methodology link.

---

## 9. Verification log (UTC, 2026-10-08)

| Time (approx) | Request | Result |
|---|---|---|
| 16:27 | GET wildlife.ca.gov MFDE pages and user guide | 200 |
| 16:28 | GET mfde.wildlife.ca.gov and its JS bundle | 200; Power BI Gov and the API host found |
| 16:29 | POST mfde-api …/api/ccl/byportarea for years 1979/1980/2023/2024/2025/2026 | 204/200/200/200/200/204 |
| 16:30 | POST …/api/ccl/byport, …/monthlyreports | 200 |
| 16:31 | GET nrm.dfg.ca.gov CCL 2019 PDFs (HEAD/GET) | 200, PDF |
| 16:31 | Wayback snapshot of the legacy landings page | 200 |
| 16:32 | Dryad API v2 dataset and files; doi.org CALFISH paper | 200 |
| 16:32 | GitHub API cfree14/wcfish | license null |
| 16:33 | PacFIN APEX dashboard, ALL005, CRAB001/002, CODE015 filters | 200 |
| 16:34 | FOSS www page | **403**; apps-st app and ORDS API 200 |
| 16:35 | Crossref, OpenAlex, Semantic Scholar DOI checks | all 6 resolved |
| 16:35 | NOAA disaster determinations (WebFetch) | entry 67 confirmed |
| 16:36 | CDFW Health Advisories, Dungeness page; CDPH DA FAQ and mussel quarantine | 200 |
| 16:36 | OEHHA DA page | **blocked (Incapsula)** |
| 16:36 | leginfo FGC §8022 | **403** (read via law.onecle.com mirror instead) |
| 16:37 | BLS CPI API; FRED CPIAUCSL CSV | 200 |
| 16:38 | MFDE CustomQuery (monthly value) for 2024 / 2026 | 200 / 204 |
