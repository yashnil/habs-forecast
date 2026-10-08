> **Evidence log.** Raw verification notes from live requests on 2026-10-08 (UTC times inside). Summarized in [`../02-data-sources.md`](../02-data-sources.md). Re-verify before relying on any value; sources change.

# Regulatory / Advisory / Protected-Area Data Sources: California HAB Decision Support

**Verification window:** 2026-10-08, 16:27–16:35 UTC. Checks used `curl` with a desktop Chrome User-Agent. WebFetch was tried on OEHHA. Each item below is marked:
- **VERIFIED**: confirmed by a live request in this window.
- **UNVERIFIED**: could not be confirmed, with the reason.

**Ground rule:** every URL, layer ID and value below was returned by a live request. Nothing was inferred. Where a value came only from a search-engine snippet, it is labelled that way.

**Context observed:** every CDPH page shows a banner saying "The federal government has shut down…". This is a federal-funding notice. The state sites still worked. NOAA pages returned 200 and showed content dated as recently as Sept 2026.

---

## 0. Current status as of 2026-10-08 (HAB-related)

| Item | Status | Area (as officially worded) | Since | Source (verified) |
|---|---|---|---|---|
| Annual sport-harvested mussel quarantine (PSP + domoic acid) | **IN EFFECT** | Entire CA coast, OR border to MX border, incl. bays, inlets, harbors | May 1, 2026; runs "through at least October 31" | CDPH SN26-008, SN26-018, ArcGIS web map |
| Bivalve shellfish (mussels, clams incl. razor, scallops, oysters) sport-harvest advisory, **domoic acid** | **IN EFFECT** | Monterey County | Sept 14, 2026 | CDPH SN26-018; web-map layer visible with `County_Name = 'Monterey County'` |
| Northern anchovy consumption advisory, domoic acid (CDPH) | **IN EFFECT** | Pigeon Point (37°11.00′N) to Point Lobos (36°31.46′N) | Sept 14, 2026 | CDPH SN26-019 |
| Northern anchovy take restriction: bait use only, commercial + recreational (CDFW, on OEHHA/CDPH recommendation) | **IN EFFECT** | South of Pigeon Point (37°11.000′N) to a line due west of Point Lobos (36°31.461′N) | Sept 11, 2026 | CDFW Health Advisories page + news release; declaration PDF `nrm.dfg.ca.gov/FileHandler.ashx?DocumentID=247331` |
| Recreational razor clam fishery closure, domoic acid | **IN EFFECT** | Humboldt County | May 2, 2024 | CDFW Health Advisories page; CDPH web-map razor clam layer visible for Humboldt |
| Razor clams, Del Norte County | Lifted Aug 31, 2026 (fishery reopened) | Del Norte County | — | CDPH SN26-017; CDFW news 8/31/2026 |
| Bivalve shellfish "Special Advisory" | **IN EFFECT** (map popup: "will remain in place… additional samples are required to lift") | Northern Channel Islands (Anacapa, Santa Cruz, Santa Rosa, San Miguel) | Feature `dataLastEditDate` 2025-07-10; issue date not stated | CDPH ArcGIS web map |
| Commercial rock crab fishery, domoic acid closure | **IN EFFECT** per CDFW page | Mendocino/Humboldt county line (40°00.00′N) to Cape Mendocino (40°30.00′N) | Start date not shown on page (**UNVERIFIED**) | CDFW Health Advisories page |
| Rock crab viscera: do-not-eat advisory (CDPH) | Listed as current on CDFW page | CA/OR border (42°00′N) to Sonoma/Mendocino line (38°46.125′N); also Santa Rosa Island / Northern Channel Islands (per CDFW page) | Dec 12, 2025 | CDPH SN25-030; CDFW page |
| Dungeness crab, domoic acid closures | **None** (commercial and recreational) | — | — | CDFW Health Advisories page |
| Dungeness crab season (2026-27) | **Commercial: closed. Recreational: closed.** This is the normal off-season before the November openers. No 2026-27 RAMP assessment has been posted yet. | Statewide | — | CDFW Whale Safe Fisheries page |
| Spiny lobster, domoic acid closures | **None** | — | — | CDFW Health Advisories page |
| Del Norte, Sonoma, Marin and San Mateo bivalve PSP advisories (spring 2026) | Lifted (Jul 14, Jun 11, May 21, May 26, 2026) | County | — | CDPH OPA list |

**Hotlines (verified on official pages):**
- CDPH Shellfish/Biotoxin Information Line: **(800) 553-4133** toll-free, also (510) 412-4643.
- CDFW Domoic Acid Fishery Closure Information Line: **(831) 649-2883**.
- Whale entanglement reporting: 1-877-SOS-WHALE or VHF Ch 16.

---

## 1. CDPH: Marine Biotoxin Monitoring Program

### 1a. The legacy URL in the existing app is dead
- `https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/MarineBiotech.aspx` returned **HTTP 404** with no redirect. VERIFIED 16:27 UTC.
- **Replace it with** `https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Marine-Biotoxin-Monitoring-Program.aspx` (HTTP 200).
- Two guessed URLs also returned 404: `.../Shellfish/Shellfish-Advisories.aspx` and `.../Shellfish/Marine-Biotoxin-Data.aspx`. Do not use them.

### 1b. Live CDPH pages (all HTTP 200, VERIFIED)
| Page | URL | Content / format |
|---|---|---|
| Program home | `https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Marine-Biotoxin-Monitoring-Program.aspx` | HTML overview and link hub |
| Quarantines & Health Advisories | `https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Marine-Biotoxin-Quarantines-and-Health-Advisories.aspx` | Pointer page only. Links to the ArcGIS map, the hotline and the news-release list. No advisory text of its own. |
| Annual Mussel Quarantine FAQ | `https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Annual-Mussel-Quarantine.aspx` | HTML FAQ. "normally in effect from May 1 through October 31"; CDPH "may begin the quarantine early, or extend it". Sport-harvested mussels only. Commercial product from certified dealers is exempt. |
| Shellfish & Seafood Advisories (news list) | `https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx` | **Best authoritative list.** HTML, reverse-chronological. Each item links to `/Programs/OPA/Pages/SN26-0NN.aspx`. 2026 runs SN26-001 (Jan 2) to SN26-019 (Sept 14). |
| Domoic Acid FAQ | `https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Domoic-Acid.aspx` | HTML |
| Monitoring Reports | `https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Marine-Biotoxin-Monitoring-Reports.aspx` | Links to **ArcGIS StoryMaps**, not PDFs. Monthly reports exist from Jan 2023. The **latest listed is April 2026**, so publication runs about 5–6 months behind. Annual reports 2020–2024. |
| FDB Domoic Acid (crab/lobster/finfish) | `https://www.cdph.ca.gov/Programs/CEH/DFDCS/Pages/FDBPrograms/FoodSafetyProgram/DomoicAcid.aspx` | HTML. **PDF** test-result tables, e.g. `.../DomoicAcid/CrabDAResultsSeptember12025toJanuary232026.pdf` and `.../DomoicAcid/SeafoodDAResults082826.pdf` (seafood results Aug 28, 2026). Also crab evisceration program information (SB 80; viscera ≥30 ppm, meat <20 ppm). |
| Phytoplankton Monitoring | `https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Phytoplankton-Monitoring-Program.aspx` | Linked from the pages above. Not separately fetched. |

**Advisory press-release pattern:**
- URLs follow `https://www.cdph.ca.gov/Programs/OPA/Pages/SN{YY}-{NNN}.aspx` (for example SN26-018 and SN26-019, both checked).
- Areas are given by **county** for bivalves, or by **named headland plus latitude** for crab and finfish. Examples: "Pigeon Point, San Mateo County (37⁰ 11.00' N. lat.)" and "Reading Rock State Marine Reserve (41° 17.6' N.)".
- **Watch for:** the text contains the degree sign written in several ways (`°`, `⁰`), zero-width spaces (U+200B) scattered through words and numbers, and broken line wraps. Normalise text before parsing.

**"Page Last Updated" stamp:** the HTML has an empty `<span id="spnModifiedDate">`, which is filled by client-side JS. Without a JS engine the update date cannot be read. **Change detection must hash the content** inside `#DeltaPlaceHolderMain`.

**RSS:** no RSS/Atom feed was found on the CDPH pages checked. **UNVERIFIED** that one exists.

### 1c. CDPH ArcGIS: Recreational Bivalve Shellfish Advisory Map (most machine-readable CDPH source)

**Web Experience:** `https://experience.arcgis.com/experience/394836318cfe4f7494e1c09097a43559`
- Title: "CDPH RECREATIONAL BIVALVE SHELLFISH ADVISORIES".
- Owner: `Steve.Etter@cdph.ca.gov_CDPHDATA`. Access: public. licenseInfo: none (null).
- Item modified **2026-09-11 17:10 UTC**.

**Underlying Web Map item:** `ad52187023f24b09b073d287f656df14` ("CDPH COASTAL SHELLFISH ADVISORIES MAP"), modified **2026-09-10 23:54 UTC**.
- Definition: `https://www.arcgis.com/sharing/rest/content/items/ad52187023f24b09b073d287f656df14/data?f=json`

**Critical finding:**
- **The advisory status is not stored as feature attributes.** The feature services hold static geometry: 21 county polygons, 21 red and 21 yellow county symbol points, all last edited in 2020.
- CDPH turns advisories on and off by **editing the web map's `layerDefinition.definitionExpression` and `visibility` per layer.** The status can therefore be read only by parsing the web-map JSON.
- **This is fragile.** Layer titles, filters and popups are hand-edited. Popup text can be stale; for example, the razor-clam popup still says "Del Norte or Humboldt" although Del Norte was lifted Aug 31.
- **Use it as a secondary signal only, never the sole source.**

**Layers and filters observed at 16:31 UTC:**

| Layer title | Service URL | definitionExpression | visible |
|---|---|---|---|
| Statewide Mussel Quarantine | `https://services2.arcgis.com/wi1yEacfYjH5viqb/arcgis/rest/services/Export_Output/FeatureServer/0` (1 coastal buffer polygon) | none | **true** |
| All Bivalve Shellfish Health Advisory - Special Advisory | `.../All_Bivalve_Shellfish_Health_Advisory_Special_Advisory/FeatureServer/3` (1 polygon, "Special Advisory Area - Northern Channel Islands") | none | **true** |
| Razor Clam Health Advisory Boundaries | `.../California_Coastal_Counties/FeatureServer/3` | `TITLE = 'Humboldt County'` | **true** |
| Razor Clam Health Advisory | `.../Limited_Advisory_Symbols_(Yellow)/FeatureServer/0` | `County_Name = 'Humboldt County'` | **true** |
| Razor Clam Advisory Lifted | `.../Health_Advisory_Symbols_(Red)/FeatureServer/0` | `County_Name = 'Del Norte County'` | **true** |
| All Bivalve Shellfish Health Advisory Boundaries | `.../California_Coastal_Counties/FeatureServer/3` | `TITLE = 'Monterey County'` | **true** |
| All Bivalve … - Domoic Acid | `.../Health_Advisory_Symbols_(Red)/FeatureServer/0` | `County_Name = 'Monterey County'` | **true** |
| All Bivalve … - PSP Toxins | same | `County_Name = 'Del Norte County'` | false |
| All Bivalve … - PSP and Domoic Acid | same | `County_Name = 'Sonoma County'` | false |
| Shellfish Safety Notification Lifted | same | `County_Name = 'Del Norte County'` | false |

- **County polygon service** `California_Coastal_Counties/FeatureServer/3`: 21 features (19 coastal and bay counties plus Northern and Southern Channel Islands). Fields: `TITLE`, `VISIBLE`, …. It is reusable as the **county-granularity advisory geometry**.
- **Spatial granularity:** county (Channel Islands treated as separate units) plus one statewide coastal buffer.

### 1d. CDPH ArcGIS: Phytoplankton Map (bonus, machine-readable)
- **Experience:** `https://experience.arcgis.com/experience/7edb5ccdfa2c4ca6b2852138847e0b32`, modified 2026-10-01.
- **Web map:** `cc82cd3a55f04e669c5cd813bb232b1d`.
- **Point services** (82 records each, `dataLastEditDate` **2026-10-01 19:42 UTC**):
  - `https://services2.arcgis.com/wi1yEacfYjH5viqb/arcgis/rest/services/Pseudo_nitzschia/FeatureServer/6`
  - `https://services2.arcgis.com/wi1yEacfYjH5viqb/arcgis/rest/services/Alexandrium/FeatureServer/7`
- **Fields:** `Date_Sampled, Sample_Site, Latitude, Longitude, PN_Percent_Comp, PN_Density, PN_RA_Rank, PN_RA_Index, AL_*`.
- **Window:** `Date_Sampled` ranges from 2026-08-17 to 2026-09-28. It is a **rolling ~6-week window**, so archive snapshots yourself.
- **Cadence:** about weekly.
- **Caveat:** CDPH states that no advisories are issued based on plankton.

**Recommendation for CDPH:**
1. **Primary:** scrape the OPA "Shellfish-Advisories.aspx" list on a schedule (every 6 h), with change detection on the item list. New SN releases alert a human, who updates a **curated `advisories.json`** with a "last verified by / at" stamp.
2. **Secondary automated check:** poll the web-map JSON (`items/ad52187…/data`) for changes to `definitionExpression` or `visibility`, and to the item `modified` time. Use it only as a "something changed" alarm.
3. Fetch the phytoplankton FeatureServers automatically, daily.
4. Mussel quarantine: encode as a calendar rule (May 1–Oct 31), with a manual override because CDPH can start early or extend.
5. Always display the hotline (800) 553-4133.

---

## 2. CDFW: HAB/domoic acid fishery closures, Dungeness crab, RAMP, rock crab, lobster, razor clam

### 2a. Pages (all HTTP 200, VERIFIED 16:29–16:33 UTC)
| Page | URL | Notes |
|---|---|---|
| **Health Advisories and Closures for CA Finfish, Shellfish and Crustaceans** | `https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories` | **Single best HTML status page** for HAB closures. Has anchor sections `#news`, `#finfish`, `#razor-clam`, `#spiny-lobster`, `#crab`. **Contains HTML-commented-out stale sections**, so a scraper must strip `<!-- -->` first. Without that, an old anchovy/sardine advisory leaks through, as I found. |
| **Whale Safe Fisheries (RAMP)** | `https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries` | Shows "Dungeness Crab 2026-27 Fishery Status" with commercial and recreational status, plus the full chronological list of RAMP assessments and Director's Declarations (PDF links). Anchor `#dungenesscrabstatus`. |
| Crabs page | `https://wildlife.ca.gov/Conservation/Marine/Invertebrates/Crabs` | Press releases, and **Declarations/Memos/Notices** (Director's Declarations for domoic acid delays and openings, as PDFs), FAQ. |
| Groundfish summary | `https://wildlife.ca.gov/Fishing/Ocean/Regulations/Groundfish-Summary` | "Updated June 23, 2026". Season and depth tables by Groundfish Management Area. |
| Invertebrate sport regs | `https://wildlife.ca.gov/Fishing/Ocean/Regulations/Sport-Fishing/Invertebrate-Fishing-Regs` | 200; content not parsed |
| Commercial | `https://wildlife.ca.gov/Fishing/Commercial` | 200; content not parsed |
| MPAs | `https://wildlife.ca.gov/Conservation/Marine/MPAs` | Links to `/Conservation/Marine/GIS/MarineBIOS` |
| News item (anchovy) | `https://wildlife.ca.gov/News/Archive/commercial-and-recreational-take-restrictions-in-place-for-northern-anchovy-in-monterey-bay-due-to-public-health-hazard` | Sept 11, 2026 |

**Guessed URLs that look alive but are not real content pages:**
- `https://wildlife.ca.gov/Conservation/Marine/Dungeness-Crab` redirects to `/Regions/Marine/Dungeness-Crab`.
- `https://wildlife.ca.gov/Conservation/Marine/HABs` redirects to `/Regions/Marine/HABs`.
- Both return a generic Marine Region shell (about 36 KB). Do not use them.

### 2b. Director's Declarations / OAL
- **Hosting:** declarations, RAMP preliminary and final assessments, and Working Group recommendations are PDFs at `https://nrm.dfg.ca.gov/FileHandler.ashx?DocumentID=<id>&inline`.
  - Tested: DocumentID 245569 (June 18, 2026 RAMP Final Assessment) returned 200 `application/pdf`, 2 pages, and `pdftotext` extracts clean text. VERIFIED.
- **Indexing:** the IDs are opaque and not sequential per fishery. The only index is the links on the Whale-Safe-Fisheries and Crabs pages.
- **Some links on the CDFW Crabs page are malformed**, e.g. `http://chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://nrm.dfg.ca.gov/...` for DocumentID 241092. The scraper must extract the inner `nrm.dfg.ca.gov` URL.
- **Regulation text:** RAMP is **Title 14 CCR §132.8**, zones in §132.8(a)(7); confirmed in the PDF and in the GIS service metadata. The amended-regulation link on the CDFW page points to Westlaw (`govt.westlaw.com/calregs/Document/I407D0770B58C11F0910AF954C72AC998…`), which returned **HTTP 403 Cloudflare "Just a moment…" challenge**. **BLOCKED / UNVERIFIED** via curl.
- **OAL:** I did not find or verify a separate OAL page for Director's Declarations. **UNVERIFIED.** Declarations appear to be published only as nrm.dfg PDFs linked from CDFW pages.

### 2c. Dungeness crab status (VERIFIED from the Whale Safe Fisheries page)
- **"Dungeness Crab 2026-27 Fishery Status": Commercial "Season is closed"; Recreational "Season is closed".**
  - The page's own table of contents still says "2025-26", and the 2026-27 risk-assessment tab has no entries yet.
- **Health Advisories page:** "no closures in the recreational [or commercial] Dungeness crab fishery due to naturally occurring marine toxins."
- **End of 2025-26 season** (June 18, 2026 final assessment PDF):
  - Commercial: Zones 1–2 had a 15% Gear Reduction and a **30-fathom Depth Constraint**; Zones 3–5 were closed with Alternative (pop-up) Gear authorized.
  - Recreational: Crab Trap Prohibition in Zones 3–5; Fleet Advisory for all zones.
  - The statutory season ended July 15, 2026 (northern zones, per the PDF).
- **2025-26 domoic acid history** (from Director's Declarations listed on the Crabs page):
  - Recreational closure CA/OR border to Sonoma/Mendocino line from 10/24/2025, lifted in stages 12/12/2025 and 12/30/2025.
  - Commercial delay from Reading Rock MPA south boundary to Cape Mendocino on 1/8/2026, lifted 1/23/2026.
- **Normal opener dates** (recreational first Saturday in November; commercial Nov 15 central / Dec 1 north) were **not confirmed** on a page fetched today. **UNVERIFIED.**

### 2d. RAMP Fishing Zones: official boundary latitudes (VERIFIED from GIS attributes and the official map JPG)
From CDFW BIOS **ds3120** attributes, matching the official map `https://wildlife.ca.gov/Portals/0/Images/marine/WSF/RAMP-Fishing-Zones.jpg` (Last-Modified 2025-11-03):

| Zone | Northern boundary | Southern boundary |
|---|---|---|
| Zone 1 | CA/OR border **42° 00′ N** | Cape Mendocino **40° 10′ N** |
| Zone 2 | Cape Mendocino 40° 10′ N | Sonoma/Mendocino county line **38° 46.125′ N** |
| Zone 3 | Sonoma/Mendocino line 38° 46.125′ N | Pigeon Point **37° 11′ N** |
| Zone 4 | Pigeon Point 37° 11′ N | Lopez Point **36° 00′ N** |
| Zone 5 | Lopez Point 36° 00′ N | Point Conception **34° 27′ N** |

- Service metadata: **"As of October 2025, the RAMP zones were updated to only include zones 1-5."**
- Pre-Oct-2025 documents reference **Zone 6**. Its boundary is **not in current GIS** and was **not verified**.
- **Do not hard-code six zones.**

**GIS endpoints (VERIFIED):**
- **Official BIOS:** `https://services2.arcgis.com/Uq9r85Potqm3MfRV/arcgis/rest/services/biosds3120_fpu/FeatureServer/0`
  - "Risk Assessment and Mitigation Program (RAMP) Fishing Zones - R7 - CDFW [ds3120]", AGOL item `a2cf5b219ab34c5e9adc9ae24e574f67`, owner BIOS_Admin.
  - 5 polygons. Fields `FishingZone`, `Description`. `dataLastEditDate` 2026-04-21.
  - **`f=geojson` query returned HTTP 200** (288 KB for `resultRecordCount=2`).
- **Whale-safe team copy:** `https://services2.arcgis.com/Uq9r85Potqm3MfRV/arcgis/rest/services/RAMP_Fishing_Zones/FeatureServer/0`
  - 5 polygons. Adds an `EFPs` field. Last edit 2025-11-04.
- **What is NOT in GIS:**
  - The **30-fathom depth constraint** line is not published here as a RAMP layer.
  - **Zone status** (open/closed/gear reduction) is **not** an attribute. It exists only in HTML and PDF.

### 2e. Razor clam, rock crab, lobster, anchovy
These are listed in §0 and all come from the Health Advisories page.
- **Area definitions:** county-level (razor clam), or headland-plus-latitude (rock crab 40°00′–40°30′N; anchovy 37°11′–36°31.461′N).
- **No GIS** is published for these closure areas.

### 2f. Machine-readable feeds (VERIFIED)
- **CDFW Marine Region blog RSS:** `https://cdfwmarine.wordpress.com/feed/`. Returns `application/rss+xml`; latest item 2026-10-01. Covers general marine news, not every closure.
- **CDFW News RSS (LiveArticles module):** `https://wildlife.ca.gov/DesktopModules/LiveArticles/API/Syndication/GetRssFeeds?category=rock%20crab&cid=171&mid=56678&PortalId=0&tid=4232&ArticlePage=1&itemcount=5`. Returns `text/xml` RSS 2.0.
  - **Caveat:** the results appear scoped by the module/tag (`mid`/`tid`), not by `category`. It returned crab news up to 2026-05-15 but **not** the 2026-09-11 anchovy release.
  - The scope is unclear. Treat it as a supplementary signal only.

**Recommendation for CDFW:**
- **RAMP zones:** automated GIS fetch from ds3120, weekly, with a hash check.
- **Zone status, domoic acid closures and declarations:** scheduled scrape (every 6 h in season, daily otherwise) of `/Fishing/Ocean/Health-Advisories` (strip HTML comments) and `/Conservation/Marine/Whale-Safe-Fisheries`, with change detection on the `#crab`, `#razor-clam`, `#spiny-lobster`, `#finfish` and `#dungenesscrabstatus` sections. Also fetch new nrm.dfg PDF links.
- **Every change goes to a human**, who updates a curated `closures.json` with zone, latitude bounds, status, effective datetime, source PDF URL and "last verified by". The app must **never** show auto-parsed open/closed status without review.

---

## 3. OEHHA: domoic acid recommendations and consumption advisories

**HTML pages: BLOCKED / UNVERIFIED.**
- curl (both plain and with a Chrome User-Agent) and WebFetch received an **Imperva/Incapsula JS challenge**: a 212-byte page loading `/_Incapsula_Resource`. Content could not be read.
- Affected URLs (each returned 200 with challenge HTML only):
  - `https://oehha.ca.gov/habs/domoic-acid-marine-biotoxin-fish-and-shellfish` (also linked from CDFW as `…#download`)
  - `https://oehha.ca.gov/fish/general-info/marine-biotoxin-domoic-acid-fish-and-shellfish`
  - `https://oehha.ca.gov/fish/advisories`
  - `https://oehha.ca.gov/habs`

**Static PDFs are reachable** (VERIFIED 200 `application/pdf`):
- `https://oehha.ca.gov/sites/default/files/media/2024-10/razorclamrecreationalfisherymemo050224.pdf` (May 2, 2024 Humboldt razor clam recommendation memo)
- `https://oehha.ca.gov/sites/default/files/media/2025-03/Domoic%20Acid%20FAQ.pdf`

**Search-index snippets only (NOT live-verified):** the OEHHA domoic acid page reportedly lists:
- recommendation memos dated Sept 11, 2026 (northern anchovy, Monterey Bay), Aug 31, 2026 (Del Norte razor clam reopen) and Jan 23, 2026 (Humboldt commercial Dungeness open);
- action levels of ≥20 ppm in edible tissue and >30 ppm in Dungeness crab viscera.

These are consistent with the CDFW and CDPH pages verified above.

**Role** (from the CDFW page, verified): OEHHA, with CDPH, *recommends* closures, delays and reopenings under Fish & Game Code §5523, and CDFW implements them. Reopening requires two successive samples below the action level, at least 7 days apart.

**Format:** HTML index plus per-decision **PDF memos**. Spatial granularity follows the fishery: county, or headland with latitude.

**Recommendation for OEHHA:** do **not** scrape (bot protection). Treat OEHHA as upstream provenance. When CDFW posts a closure, a human records the OEHHA memo PDF URL in curated JSON. The OEHHA mercury/chemical advisories are a separate topic; link to them, don't ingest them.

---

## 4. CDFW Marine Protected Areas GIS (VERIFIED)

**Dataset:** **"California Marine Protected Areas [ds582]"**, CDFW Marine Region GIS (r7mrgis@wildlife.ca.gov).

**Endpoints:**
- **ArcGIS REST:** `https://services2.arcgis.com/Uq9r85Potqm3MfRV/arcgis/rest/services/biosds582_fpu/FeatureServer/0`
  - AGOL item `117a99c8745a48c6a48bac70005b1b11`, owner BIOS_Admin.
  - `maxRecordCount` 2000. Formats: JSON, geoJSON, PBF.
- **Test query:** `.../biosds582_fpu/FeatureServer/0/query?where=1=1&outFields=*&resultRecordCount=2&f=geojson`
  - Returned HTTP 200 FeatureCollection, 2 Polygon features, first is "Pyramid Point SMCA", WGS84 lon/lat.
- **Hub downloads (CDFW Open Data, all HTTP 200):**
  - GeoJSON: `https://data-cdfw.opendata.arcgis.com/api/download/v1/items/117a99c8745a48c6a48bac70005b1b11/geojson?layers=0` (4.17 MB)
  - Shapefile: `https://data-cdfw.opendata.arcgis.com/api/download/v1/items/117a99c8745a48c6a48bac70005b1b11/shapefile?layers=0` (0.93 MB)
  - CSV and KML: same pattern, `/csv?layers=0` and `/kml?layers=0` (listed on data.ca.gov; not separately fetched)
  - Hub page: `https://data-cdfw.opendata.arcgis.com/datasets/CDFW::california-marine-protected-areas-ds582`
- **BIOS file library zip:** `https://filelib.wildlife.ca.gov/Public/BDB/GIS/BIOS/Public_Datasets/500_599/ds582.zip` (200, 490 KB, Last-Modified 2023-03-08)
- **data.ca.gov package:** `california-marine-protected-areas-ds582`, found via the CKAN API `https://data.ca.gov/api/3/action/package_search?q=…`. Metadata modified 2026-09-30, which reflects harvest time, not data time.

**Features and attributes:**
- **Feature count: 155.** Types: SMR 49, SMCA 61, SMCA (No-Take) 10, SMP 7, SMRMA 5, Special Closure 14, FMR 8, FMCA 1. The federal MPAs are the northern Channel Islands complements.
- **Attributes:** `NAME`, `FULLNAME`, `SHORTNAME`, `Type`, `CCR`, `CCR_Int`, `Study_Regi` (NCSR etc.), `Area_sq_mi`, `Acres`, `Hectares`.
  - `CCR` holds the regulation citation, e.g. "Section 632 (b) (1)".
  - **There is no per-MPA regulations URL field.** The app must link to Title 14 §632 or to the CDFW MPA regional pages (e.g. `https://wildlife.ca.gov/Conservation/Marine/MPAs/Network/Northern-California`, 200).
- **Last updated:** service `dataLastEditDate` **2024-01-09**. Item description says "all of California's MPAs as January 1, 2019".
  - The tribally-led 2023 MPA petitions are still under evaluation (CDFW blog, Jul 31, 2026).
  - **Check the dataset after any Fish & Game Commission MPA rulemaking.**
- **License:** **Creative Commons Attribution 4.0 International** (item licenseInfo). Cite per `https://www.wildlife.ca.gov/Data/BIOS/Citing-BIOS`.
- **Disclaimer:** "not intended for navigational use or defining legal boundaries."

**Related BIOS layers:**
- ds3207 "Marine Protected Areas Coordinates" (points): `.../biosds3207_fmu/FeatureServer/0`, CC-BY 4.0.
- ds3158 "Three Nautical Mile State Maritime Limit": `.../biosds3158_fnu/FeatureServer`, found via search only.
- **Do not use** third-party copies found on AGOL, such as "Marine_Protected_Areas_ds582_2026" by coastalquest or an Audubon copy. They are not official.

**NOAA MPA Inventory (fallback):**
- **Service:** `https://services2.arcgis.com/C8EMgrsFcRFL6LrL/arcgis/rest/services/NOAA_MPA_Inventory_2023/FeatureServer/0`
  - AGOL item `eb2b36aecb004f14ac29cf0260624291` "NOAA Marine Protected Areas Inventory 2024", owner MPA_noaa. Last edit 2026-04-14.
  - `State='CA'` returns **197** features; this includes non-CDFW sites such as sanctuaries.
  - Useful fields: `Site_Name, Gov_Level, Prot_Lvl, Mgmt_Agen, Fish_Rstr, URL, IUCNcat`.
- **Viewer:** `https://marineprotectedareas.noaa.gov/dataanalysis/mpainventory/` (200).
- **License:** not checked (US federal, generally public domain). **UNVERIFIED.**

**Recommendation for MPAs:**
- **Automated GIS fetch** of ds582 GeoJSON from the REST endpoint or hub URL, weekly, comparing feature count, `dataLastEditDate` and a geometry hash.
- Bundle a static snapshot with the app; refresh only after human sign-off.

---

## 5. Groundfish RCAs, NOAA West Coast closures, ocean sport-fishing season pages (brief)

**CDFW BIOS (all CC-BY 4.0, Marine Region GIS, VERIFIED via REST):**
- **Rockfish Conservation Area lines [ds3144]:** `https://services2.arcgis.com/Uq9r85Potqm3MfRV/arcgis/rest/services/biosds3144_fnu/FeatureServer/0`
  - **Polylines**, 71 features. Fields `area_name` (e.g. "150-fm (274-m) Contour - Coastwide"), `START_Fath`, `Region`. Data last edit 2025-01-04.
  - Waypoints are in ds3145 (`biosds3145_fmu`).
- **Groundfish Recreational Boundaries [ds3143]:** `.../biosds3143_fpu/FeatureServer/0`
  - 6 polygons (Northern, Mendocino, San Francisco, Central-North 36°, Central-South 36°, Southern). Field `Detail` holds the latitude text, e.g. "38°57.5 N (Point Arena)". Last edit 2025-11-12.
- **Cowcod Conservation Areas [ds3165]:** `.../biosds3165_fpu/FeatureServer/0`. 2 polygons (Western and Eastern CCA).

**CDFW Groundfish Summary page** (`/Fishing/Ocean/Regulations/Groundfish-Summary`):
- HTML season and depth table per management area. "Updated June 23, 2026."
- Example: in one southern area, Oct 1–Dec 31 is "Closed" or "50 Fathom - Offshore Only" depending on the column. The exact column mapping was not parsed.
- **Scheduled scrape plus manual curation.**

**NOAA Fisheries West Coast Groundfish Closed Areas:** `https://www.fisheries.noaa.gov/west-coast/sustainable-fisheries/west-coast-groundfish-closed-areas` (200).
- RCA boundary coordinate CSVs in a zip, **"current as of August 2026"**: `https://www.fisheries.noaa.gov/s3/2026-09/rca-lines-latlongs-august-2026.zip` (200, `application/zip`, 96 KB).
- Groundfish Exclusion Areas CSV: `/s3/2026-06/geas-updated-06-23-2026.csv`.
- YRCA CSV: `/s3/2023-11/YRCAs-01012024.csv`.
- Legal authority: 50 CFR 660 (eCFR).
- NOAA inseason bulletins list: `https://www.fisheries.noaa.gov/rules-and-announcements/bulletins?field_region_vocab_target_id=1000001126` (200, not parsed).
- **Ingest:** automated download of the RCA zip with change detection on the S3 filename/date. Which RCA line is "active" is set by regulation, so curate that manually.

---

## 6. Summary table

| # | Source | Endpoint (verified) | Covers | Spatial granularity | Cadence | Format | License / terms | Access constraints | Last-updated observed | Status / time (UTC) | Ingest recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | CDPH OPA Shellfish & Seafood Advisories list | cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx | All CDPH biotoxin advisories and lifts (bivalves, crab, finfish) | County; headland + latitude | Event-driven (19 releases in 2026) | HTML list, links to HTML releases | State website Use Policy (not reviewed) | SharePoint; zero-width chars; date stamp JS-only; no RSS found | Newest item 2026-09-14 (SN26-019) | VERIFIED 16:28 | Scrape every 6 h with change detection, then human-curated JSON |
| 2 | CDPH Bivalve Advisory web map | arcgis item ad52187023f24b09b073d287f656df14 (data?f=json) + services2.arcgis.com/wi1yEacfYjH5viqb/... | Mussel quarantine, county bivalve advisories, razor clam | County polygons + statewide buffer | Event-driven | ArcGIS web-map JSON (status in layer filters, not attributes) | No licenseInfo | Hand-edited; stale popups; schema can change without notice | Web map modified 2026-09-10 | VERIFIED 16:31 | Automated poll as change alarm only |
| 3 | CDPH Mussel Quarantine FAQ | …/Shellfish/Annual-Mussel-Quarantine.aspx | Quarantine rules | Statewide | Annual (May 1–Oct 31) | HTML | — | — | — | VERIFIED 16:27 | Calendar rule + manual override |
| 4 | CDPH Monitoring Reports | …/Shellfish/Marine-Biotoxin-Monitoring-Reports.aspx | Monthly toxin data summaries | Site/county | Monthly, about 5–6 months behind | ArcGIS StoryMaps | — | Not structured data | Latest = April 2026 | VERIFIED 16:28 | Link only |
| 5 | CDPH FDB Domoic Acid | …/DFDCS/…/DomoicAcid.aspx | Crab, lobster, finfish DA results; evisceration | Port/area | Event/season | HTML + PDF tables | — | PDF parsing | Seafood PDF 2026-08-28 | VERIFIED 16:28 | Link; optional PDF extraction with human QA |
| 6 | CDPH Phytoplankton FeatureServers | services2.arcgis.com/wi1yEacfYjH5viqb/.../Pseudo_nitzschia/FeatureServer/6 and Alexandrium/FeatureServer/7 | PN and Alexandrium relative abundance | Point (site) | About weekly, rolling ~6 weeks | ArcGIS REST / GeoJSON | No license stated | Rolling window; archive yourself | dataLastEdit 2026-10-01 | VERIFIED 16:34 | Automated daily fetch |
| 7 | CDFW Health Advisories & Closures | wildlife.ca.gov/Fishing/Ocean/Health-Advisories | DA closures: crab, rock crab, lobster, razor clam, finfish | County; headland + latitude | Event-driven | HTML | CDFW Conditions of Use (not reviewed) | Commented-out stale HTML; no API | Newest item 2026-09-14 | VERIFIED 16:29 | Scrape every 6 h with change detection, then human-curated JSON |
| 8 | CDFW Whale Safe Fisheries (RAMP) | wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries | Dungeness season status, RAMP actions, depth constraints | RAMP Zone 1–5 (latitude bands) | About biweekly in season | HTML + PDFs | — | Opaque nrm.dfg PDF IDs | Last action 2026-06-18; 2026-27 status "closed" | VERIFIED 16:30 | Scrape with change detection, then curated JSON |
| 9 | CDFW Director's Declarations | nrm.dfg.ca.gov/FileHandler.ashx?DocumentID=…&inline | Legal closures/openings | Zone / latitude | Event-driven | PDF (text-extractable) | — | Indexed only via HTML pages; some malformed links | e.g. 247331 (anchovy, Sept 2026) | VERIFIED (245569) 16:32 | Human review; store PDF URL as provenance |
| 10 | CDFW RAMP Zones GIS ds3120 | services2.arcgis.com/Uq9r85Potqm3MfRV/.../biosds3120_fpu/FeatureServer/0 | Zone 1–5 polygons | Latitude-band polygons | Rare (regulation changes) | ArcGIS REST / GeoJSON | BIOS (CC-BY 4.0 typical; ds3120 item license not checked) | — | dataLastEdit 2026-04-21 | VERIFIED 16:33 | Automated GIS fetch, weekly |
| 11 | Title 14 CCR §132.8 (Westlaw) | govt.westlaw.com/calregs/… | RAMP legal text | — | Rulemaking | HTML | — | **403 Cloudflare challenge** | — | BLOCKED | Manual |
| 12 | CDFW news RSS / Marine blog RSS | wildlife.ca.gov/DesktopModules/LiveArticles/API/Syndication/GetRssFeeds?…; cdfwmarine.wordpress.com/feed/ | News | — | Event-driven | RSS 2.0 | — | News RSS scope unclear (missed anchovy item) | Blog 2026-10-01; news RSS 2026-05-15 | VERIFIED 16:33 | Supplementary alert trigger |
| 13 | OEHHA domoic acid page + memos | oehha.ca.gov/habs/domoic-acid-marine-biotoxin-fish-and-shellfish; /sites/default/files/media/...pdf | Closure/reopen recommendations | County / latitude | Event-driven | HTML + PDF | — | **Incapsula bot-block on HTML**; PDFs OK | Not readable live | HTML BLOCKED; PDFs VERIFIED 16:34 | Manual provenance only |
| 14 | CDFW MPAs ds582 | services2.arcgis.com/Uq9r85Potqm3MfRV/.../biosds582_fpu/FeatureServer/0; data-cdfw hub geojson/shapefile; filelib ds582.zip | 155 MPAs / special closures | Polygon | Rare | REST, GeoJSON, SHP, CSV, KML | **CC-BY 4.0** | maxRecordCount 2000 (155 rows, so fine) | dataLastEdit 2024-01-09 | VERIFIED 16:33 | Automated GIS fetch, weekly; human sign-off |
| 15 | NOAA MPA Inventory | services2.arcgis.com/C8EMgrsFcRFL6LrL/.../NOAA_MPA_Inventory_2023/FeatureServer/0 | National MPAs (197 in CA) | Polygon | Annual-ish | REST | License not checked | — | lastEdit 2026-04-14 | VERIFIED 16:34 | Fallback only |
| 16 | CDFW RCA / Groundfish areas / CCA (ds3144/3143/3165) | services2.arcgis.com/Uq9r85Potqm3MfRV/.../biosds3144_fnu, biosds3143_fpu, biosds3165_fpu | Depth-contour lines, management areas | Lines / polygons | Rare | REST | CC-BY 4.0 (BIOS) | Active depth set by regulation, not GIS | 2025-01-04 / 2025-11-12 / 2024-04-04 | VERIFIED 16:33 | Automated GIS fetch + curated active-depth JSON |
| 17 | CDFW Groundfish Summary | wildlife.ca.gov/Fishing/Ocean/Regulations/Groundfish-Summary | Season/depth by area | Management area | Few times a year | HTML table | — | Table parsing | "Updated June 23, 2026" | VERIFIED 16:34 | Scrape with change detection, then curated |
| 18 | NOAA West Coast groundfish closed areas | fisheries.noaa.gov/west-coast/sustainable-fisheries/west-coast-groundfish-closed-areas; RCA zip s3/2026-09/… | Federal RCA/EFH/YRCA coordinates | Lines / polygons | Inseason | HTML + CSV zip | US federal | — | "current as of August 2026" | VERIFIED 16:34 | Automated download + manual activation |

---

## 7. Implementation notes for the app
1. **Status must be human-curated.** No official source publishes HAB closure or advisory *status* as structured data. The CDPH web map comes closest, but it stores status in hand-edited map filters.
   - Recommended record: `{id, agency, species[], action (advisory|closure|delay|quarantine|evisceration_order|trap_prohibition|depth_constraint), area: {type: county|lat_band|statewide|polygon_ref, counties[], lat_north, lat_south, ref}, toxin (DA|PSP), effective_at, lifted_at, source_urls[], last_verified_by, last_verified_at}`.
2. **Geometry can be automated:**
   - RAMP zones (ds3120);
   - CDPH county advisory polygons (`California_Coastal_Counties/FeatureServer/3`);
   - MPAs (ds582);
   - RCA lines and groundfish areas (ds3144/ds3143).
   - Latitude-band closures (rock crab 40°00′–40°30′N; anchovy 37°11′–36°31.461′N) can be generated by clipping the coastline or state waters with the stated latitudes. Label them "approximate; legal description controls."
3. **Change-detection scrapers:**
   - Strip `<script>`, `<style>` and **HTML comments**.
   - Remove U+200B and normalise `⁰` to `°`.
   - Hash only the main-content region: CDPH `#DeltaPlaceHolderMain`; CDFW the breadcrumb through "Additional Information".
   - Alert on diff. Use a browser User-Agent; plain curl with a UA worked for CDPH, CDFW and NOAA.
4. **Always show the hotlines and a "check official source" link.** Show "last verified" timestamps prominently. If curated data is older than about 72 h during an active event, show a stale-data warning.
5. **Update the app:** replace the dead `MarineBiotech.aspx` link.
