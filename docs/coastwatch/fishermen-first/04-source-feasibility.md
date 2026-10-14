# CoastWatch source audit C: real-time vs. irregular sources, and a respectful change monitor

This is a read-only audit. No accounts were created, nobody was contacted, and no repository was written to.

- **Live checks:** 2026-10-13, **23:12–23:22 UTC** (16:12–16:22 PDT). Every request used `curl` with the User-Agent `CoastWatch-research/0.1 (contact: project maintainer)`. There were about 150 requests across 25 hosts, at most one per resource except for retries.
- **Raw evidence:** the one-line request log is committed as [`evidence/source-audit-log.txt`](evidence/source-audit-log.txt). The raw headers and bodies (8.7 MB) were kept out of the repo, in the research session's scratchpad. The condensed log is in §6.
- **This extends earlier work and does not repeat it:**
  - `docs/coastwatch/02-data-sources.md` (verified 10-08) describes the source register.
  - `docs/coastwatch/research/ocean-sources.md` (05:09–05:35Z today) covers the WCOFS, NWS, NDBC, CO-OPS, SST, HF-radar and MPA evaluations and the staging designs.
  - `docs/coastwatch/research/data-health.md` (05:50Z today) covers pipeline freshness and the "Unconfirmed" rule.
  - Where those documents already hold the answer, this one cites them and records only what is new or changed today.

**Vocabulary used below:**
- **Real-time** means automated sensing, published within about 1 h.
- **Near-real-time** means automated, published within 1–3 days.
- **Irregular / lab-paced** means discrete samples with lab turnaround, published days to months later.
- **Event-driven** means agency notices issued when the agency acts.
- Under each source, *cadence of new data* (when something new is measured or decided) is reported separately from *page refresh* (when a web page or header changes).

---

## 0. Headline findings (new today)

1. **CDPH lifted the Monterey County sport-bivalve warning on 2026-10-12 (SN26-021).** CoastWatch's registry still carries `cdph-2026-sn26-018-monterey-bivalves` as `active`. The existing watcher (`official.py: reported_lifts`) is designed to flag exactly this case as a published *discrepancy*. Only a person running `cwp review-official` may change the record.
   - SharePoint timestamps show the release page was created 15:47Z, the list page was updated 16:01Z, and the release page was **edited again at 22:58Z**, seven hours after publication.
   - The CDPH ArcGIS **web map was not switched to "Shellfish Safety Notification Lifted – Monterey" until 2026-10-13 16:39Z, about 25 h after the release.** The map lags the press release, so it must never be the trigger.
2. **CDPH SharePoint exposes a cheap, exact change signal.** The page's own "Last updated" script calls `/_api/web/getfilebyserverrelativeurl('<page>')?$select=TimeLastModified`, which returns `TimeLastModified` and `TimeCreated` in about 1 KB. `/_api/` is **not** disallowed by cdph.ca.gov's robots.txt, which disallows only `/_layouts/`, `/_vti_bin/` and `/_catalogs/`.
   - CDPH HTML pages send no `ETag` or `Last-Modified`. CDPH **PDFs do**, and a conditional GET returned **304**.
3. **CDPH's ArcGIS layers are not a status feed.** `editingInfo.lastEditDate` on the advisory symbol and county layers is 2020-07 or 2020-08, and 2025-07-10 on the NCI polygon. Advisory status lives **only** in the web map item's layer `visibility` and `definitionExpression`. The web map item `ad52187023f24b09b073d287f656df14` reported `modified = 2026-10-13T16:39:53Z`.
   - The same org also hosts `CDFW_Fisheries_Closure*` (last edited 2021) and `SampleDetails` (2016). Both are dead; do not use them.
4. **CDPH crab and seafood domoic-acid PDFs are near-weekly during events, with about 10–14 days from catch to publication.**
   - `CrabDAResults.pdf`: collection 9/28/2026, "As of 10/8/2026", file modified 10-12 17:23Z (ETag version `,7`).
   - The link text on the page ("September 29 – October 8") does not match the PDF content ("September 28").
   - Monthly biotoxin "reports" are StoryMaps. The newest is **April 2026**, about 5.5 months behind.
5. **CalHABMAP toxin and count data are not real-time.**
   - Counts are weekly at 8 southern and central piers, with a median gap of 7 days. Every northern station (Trinidad, Humboldt ×2, Bodega ×2, Tomales ×3) has **no sample since March–April 2026**.
   - **pDA lags counts by about 2 months at most piers:** the last pDA is 08-03 or 08-10 at Cal Poly, Stearns, Santa Monica, Newport and Scripps, while counts run to 10-04 or 10-05. Santa Cruz is the exception, with same-week pDA (09-30). Monterey Wharf has temperature only.
6. **The NOAA-20 and NOAA-21 750 m VIIRS sector datasets (the "Primary" in 02-data-sources §1.2) are stuck at 2026-10-05**, 8 days stale.
   - `erdVHNchla1day` has recovered to 10-10.
   - The non-sector `noaacwN20VIIRSchlaDaily` is at 10-11. That makes it the best S-NPP replacement candidate (data-health §5 #3).
7. **C-HARM ran today.** `history` begins "2026-10-13: DINEOF…", the nowcast is valid 10-12 and lead 3 is valid 10-15.
   - Valid days 10-05 and 10-08 are **missing** from the 0-day series.
   - The first request returned **HTTP 503**; it succeeded 30 s later.
   - `date_created/modified/issued` are static (2022-12-07), so `history` is the only issue-date signal.
8. **CDFW news RSS exists and does carry HAB notices.** The URL is `wildlife.ca.gov/DesktopModules/LiveArticles/API/Syndication/GetRssFeeds?mid=56678&PortalId=0&tid=4232&ArticlePage=1&itemcount=20`. With `itemcount=20` it includes the 9/11 anchovy and 10/9 Del Norte razor-clam releases. 02-data-sources says it "missed" the anchovy release; that was probably the default `itemcount=5`.
   - It sends no ETag or Last-Modified, but each item has a `pubDate`.
   - CDFW's GovDelivery account is **`CNRA`**, not `CANRA`. Topic **42866 is "Domoic Acid, PSP Toxin Alerts"** and 42890 is "Whale Safe Fisheries". No public bulletin archive or RSS was found (`content.govdelivery.com/accounts/CNRA/bulletins` → 404), but individual bulletins are public and carry "sent at" timestamps.
9. **The Fish and Game Commission votes on the MPA petitions on Oct 15–16, 2026** (news sources). The CDFW `ds582` MPA layer was last edited 2024-01-09, so the boundary layer may soon disagree with adopted regulation. This needs a watch and a banner.
10. **Robots and compliance findings that need a user decision (§5).**
    - `api.weather.gov/robots.txt` is `Disallow: /`.
    - ERDDAP hosts disallow `/*?` and `*.csv*`: on coastwatch.pfeg that is `/*?`, `*.csv*` and so on; on coastwatch.noaa.gov it is `/erddap/` plus `Crawl-delay: 10`; on cencoos it is `/*?` and `*.csv*`.
    - `cdip.ucsd.edu` is `Disallow: /`, but `thredds.cdip.ucsd.edu` has no robots.txt.
    - `nowcoast.noaa.gov` disallows `/geoserver/`.
    - `data.chhs.ca.gov` and `data.ca.gov` disallow `/api/`.
    - **I made one CKAN `/api/` search on each of the two portals before reading their robots files.** Both found nothing relevant (CHHS: 0 results; data.ca.gov: 1 unrelated result). I made no further `/api/` requests.
    - CoastWatch's own watcher sends a **browser-impersonating User-Agent** (`official.py: BROWSER_UA`, "Mozilla/5.0 … Chrome/141.0 … CoastWatch-watcher"). That should be replaced; see §4.

---

## 1. Existing CoastWatch coverage (code read at `scratchpad/wt-ff`, HEAD `d414dd63`)

| Area | Where | What it does today |
|---|---|---|
| Official notices | `pipeline/coastwatch_pipeline/sources/official.py`, `data/curated/official_notices.json`, CLI `watch-official` / `review-official` (`cli.py:241-251`) | A **human-curated registry** of 11 records: CDPH quarantine and advisories, CDFW closures and take restrictions. The review status is `pending_human_review`, last transcribed by Claude on 2026-10-08 and **not yet reviewed by a person**. **The watcher** fetches 3 pages every pipeline run: the CDPH Shellfish-Advisories list, CDFW Health-Advisories and CDFW Whale-Safe. It extracts main text (strips HTML comments, zero-width characters and navigation), takes a SHA-256 fingerprint, and compares it with the fingerprint stored at the last human review. For CDPH it parses the `SN##-###` release list and reports IDs that were not reviewed. `reported_lifts()` flags unreviewed CDPH releases titled "Lifts…" that match an active county record, and publishes them as a *discrepancy*. It never edits records. `record_review()` (only via `cwp review-official --reviewer … --confirm`) stores the fingerprints and reviewed release IDs. Geometry comes from the CDPH `California_Coastal_Counties/FeatureServer/3` and NCI `…/FeatureServer/3` layers, plus latitude lines derived from the C-HARM ocean mask. **What is missing:** conditional GETs; archived snapshots; per-item `first_seen` timestamps; diffs; a review *queue* (beyond a GitHub issue, `notify-review` #12); RSS, GovDelivery and SharePoint `_api` signals; PDF watches; the CDPH web-map JSON watch (proposed in 04-architecture §`watch-regulatory`, not implemented). |
| NWS, NDBC, CO-OPS | `coastwatch-web/src/app/api/conditions/route.ts`, `src/lib/conditions.ts`, `src/lib/noaaStations.ts` | **Live server-side fetch per place**: NWS coastal zone by point (ring search), `zones/coastal/{id}/forecast` and `alerts/active?zone=`; nearest NDBC `realtime2/{id}.txt` (ranged read of the first 8 KB, latest non-`MM` value per field, 6 h limit); nearest CO-OPS hilo predictions. Uses an honest UA string with a project URL, a 6 s timeout, `revalidate: 600` and CDN `s-maxage=600`. Each part fails independently with a reason; nothing is carried forward. **No CO-OPS observed water level**, no CDIP and no shore stations. |
| C-HARM | `sources/charm.py` | `wvcharmV3_{0..3}day`, pfeg ERDDAP, `product_version == "3.1"` gate. Issue date is derived as nowcast + 1. |
| Satellite chlorophyll | `sources/satellite.py`, `multisensor.py`, `gibs.py`, `port_intel.py` | OLCI S3A/S3B 300 m (coastwatch.noaa.gov); VIIRS `erdVHNchla1day` (S-NPP, pfeg); GIBS browse tiles; `erdVHNchla8day` for port intel. **No SST layer.** |
| HF radar | `sources/currents.py` | `ucsdHfrW2` (pfeg). No NDBC mirror yet. |
| CalHABMAP | `sources/observations.py` | All 17 `HABs-*` datasets on erddap.sccoos.org, with per-station status and carry-forward. |
| Ports and fisheries | `ports.py` (CDFW ds3081), `fisheries.py` (FOSS, BLS) | Annual and reference data. |
| Not covered at all | — | CDPH crab and seafood DA PDFs, CDPH Pseudo-nitzschia/Alexandrium FeatureServers, the CA HAB Bulletin, CDFW news RSS, GovDelivery, RAMP declarations, MPA and RCA geometry, the salmon in-season pages, CDIP, shore stations, SST. |

---

## 2. Per-source audit

Columns appear in each block in this order:
- **Site:** the official website.
- **Access:** the API, feed or access method.
- **Owner/terms:** owner and licence or terms.
- **Cadence:** measured.
- **Coverage / res.:** geographic coverage and spatial resolution.
- **Latest:** the latest observation or issue seen today.
- **Lag:** time from observation or decision to availability.
- **Failure:** behaviour when down or stale.
- **Authority:** regulatory or scientific.
- **Decisions:** relevant user decisions.
- **CW today:** existing CoastWatch coverage.
- **Value:** what integration would add.
- **Mode:** recommended integration mode.

### 2.1 CDPH Marine Biotoxin Monitoring Program

**(a) Shellfish advisory releases (list plus SN pages)**
- **Site:** [Shellfish-Advisories.aspx](https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx). Releases are at `/Programs/OPA/Pages/SN{YY}-{NNN}.aspx`.
- **Access:**
  - HTML only, from SharePoint 2016 behind Volterra. The response headers carry `cache-control: private` and **no ETag or Last-Modified**.
  - Change signal: `GET /Programs/OPA/_api/web/getfilebyserverrelativeurl('/Programs/OPA/Pages/Shellfish-Advisories.aspx')?$select=TimeLastModified` returns Atom XML of about 950 bytes. It is the same call the page makes for its own "Last updated" line.
- **Owner/terms:** CDPH Office of Communications.
  - The cdph.ca.gov conditions-of-use page was not re-fetched in this session (UNVERIFIED). California government web content is generally public information.
  - CoastWatch quotes it verbatim with a link. Do not republish full releases; link to them and quote short passages.
- **Cadence:** event-driven. 2026 has had 21 releases, about 2 a month.
  - The latest is **SN26-021, 2026-10-12**: "CDPH Lifts Warnings about Sport-Harvested Bivalve Shellfish from Monterey County".
  - SN26-020 (2026-10-09) is the Del Norte razor-clam warning.
- **Coverage / res.:** statewide; area by county, or headland plus latitude.
- **Latest:**
  - List `TimeLastModified` 2026-10-12T16:01:27Z.
  - SN26-021 `TimeCreated` 15:47:21Z and `TimeLastModified` **22:58:14Z** (edited after release).
- **Lag:** the decision is announced on the issue date. The list updates about 15 minutes after a release page is created.
- **Failure:**
  - The list page is large (about 458 KB).
  - The title markup contains zero-width spaces.
  - The "last updated" date is filled in by JavaScript and is absent from the static HTML.
  - No outage was observed today.
- **Authority:** **regulatory and health authority** (CDPH consumption advisories, quarantine).
- **Decisions:** whether to harvest or eat sport bivalves, razor clams, anchovy, sardine or crab viscera.
- **CW today:** watched every run (release-ID diff plus lift-discrepancy); human-curated records.
- **Value:** use `_api` TimeLastModified as a cheap pre-check; snapshot the archive; record `first_seen` per release; diff release pages after publication, because edits do happen.
- **Mode:** **Change monitor with human review**, as now, but upgraded (§4). Never auto-publish status.

**(b) Annual mussel quarantine**
- **Site:** [Annual-Mussel-Quarantine.aspx](https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Annual-Mussel-Quarantine.aspx).
- **Cadence:** yearly. It runs May 1 to Oct 31 by default and can be extended.
- **Latest:** the 2026 quarantine is in effect (SN26-008). SN26-021 says it "remains in effect".
- **Value:** the Oct 31 end date is a predictable "expected end". The record must not be auto-expired; a person must confirm a release or extension.
- **Mode:** change monitor and human review. The `expected_end_note` already exists in the schema.

**(c) Monthly biotoxin reports**
- **Site:** [Marine-Biotoxin-Monitoring-Reports.aspx](https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Marine-Biotoxin-Monitoring-Reports.aspx). Reports are ArcGIS StoryMaps.
- **Latest:** the monthly report for **April 2026**. Annual reports cover 2020–2024.
- **Lag:** about 5–6 months.
- **Mode:** **link only**, as historical context. Not real-time.

**(d) Crab and seafood domoic-acid results (FDB)**
- **Site:** [DomoicAcid.aspx](https://www.cdph.ca.gov/Programs/CEH/DFDCS/Pages/FDBPrograms/FoodSafetyProgram/DomoicAcid.aspx).
- **Access:** the PDFs `…/DomoicAcid/CrabDAResults.pdf` and `…/SeafoodDAResults.pdf` are overwritten in place at stable URLs. Both send **ETag** (`{GUID},version`) and **Last-Modified**; a conditional GET returned **304**. They are text-extractable.
- **Cadence:**
  - New samples arrive weekly to irregularly while a fishery is being cleared or an event is under way.
  - The crab PDF ETag reached version 7. Seafood results cover anchovy samples from 8/28 and 9/29.
- **Coverage / res.:** port and BMA (Biotoxin Management Area, e.g. "BMA G – Monterey Bay"); species and tissue; 6-crab sets.
- **Latest:**
  - Crab: collected 9/28/2026, "As of 10/8/2026", LM 2026-10-12 17:23Z. Dungeness viscera averaged 5.52 ppm; rock crab viscera had a maximum of 20 ppm; 0 % of samples exceeded the action level.
  - Seafood: anchovy viscera 1200 ppm on 8/28 and 14 ppm on 9/29; LM 10-12 15:10Z.
- **Lag:** 10–14 days from catch to PDF.
- **Failure:** the page link label disagreed with the PDF date range today, and the page was not updated in step with the file.
- **Authority:** laboratory results under the regulatory program. The results are informational; status comes from OEHHA, CDPH and CDFW actions.
- **Decisions:** context only, for crab fishers and processors.
- **CW today:** none.
- **Value:** shows *why* a closure is or isn't lifted, with real toxin measurements (not chlorophyll).
- **Mode:**
  - **Change monitor with human review.** Use a conditional GET daily, archive each version, and extract the table into a draft for a person to check.
  - Never publish parsed numbers without review, because the layout is free-form.
  - Label them "lab results, sampled X, posted Y".

**(e) ArcGIS behind the advisory map**
- **Access:**
  - Experience `394836318cfe4f7494e1c09097a43559` uses web map `ad52187023f24b09b073d287f656df14` (owner `Steve.Etter@cdph.ca.gov_CDPHDATA`, public).
  - Layers are on `services2.arcgis.com/wi1yEacfYjH5viqb`: `Export_Output/0` (statewide quarantine), `Health_Advisory_Symbols_(Red)/0`, `Limited_Advisory_Symbols_(Yellow)/0`, `California_Coastal_Counties/3` and `All_Bivalve_…_Special_Advisory/3`.
- **Latest:**
  - Feature layer edit dates are 2020-07-16, 2020-08-25, 2020-08-25, 2020-08-26 and 2025-07-10, so they say nothing about status.
  - The web map item `modified` is **2026-10-13 16:39:53Z**. The "Lifted – Monterey" layer is visible, and the "Razor Clam Advisory – Humboldt OR Del Norte" layer is visible.
- **Lag:** about 1 day behind the releases.
- **Failure:** the status is encoded in hand-edited `visibility` and `definitionExpression` settings, and stale layers persist with visibility off.
- **CW today:** geometry only (counties, NCI).
- **Value:** a second, independent signal that the agency changed something.
- **Mode:** **change monitor** of the web map item's `modified` field and a hash of the `operationalLayers` (visibility plus definition expressions), at `…/sharing/rest/content/items/<id>?f=json` and `/data?f=json`. It feeds the review queue and is never treated as status.

**(f) CDPH phytoplankton FeatureServers**
- **Access:** `Pseudo_nitzschia/FeatureServer/6` and `Alexandrium/FeatureServer/7`.
- **Latest:** PN `dataLastEditDate` 2026-10-01 19:42Z. 82 features; `Date_Sampled` runs from 2026-08-17 to **2026-09-28**, which is a rolling window of about 6 weeks.
- **Cadence:** weekly volunteer and agency net tows.
- **Lag:** about 3 days from sample to layer. Today the newest sample is 15 days old.
- **Authority:** scientific screening. CDPH issues no advisories from plankton (02-data-sources §1.5). Relative-abundance index; **not toxin**.
- **CW today:** none.
- **Mode:** **API ingest (P2)** that archives every week, because the window rolls off. Label it "plankton screening, not a toxin test".

**(g) Open-data portals**
- `data.chhs.ca.gov` and `data.ca.gov` CKAN searches for shellfish, biotoxin, domoic and mussel returned **0 relevant datasets**: CHHS 0 results; data.ca.gov 1 unrelated result, "Infectious Diseases by Disease…".
- Both portals' robots.txt disallow `/api/`. See §5.
- **Mode:** none available.

### 2.2 CalHABMAP (SCCOOS ERDDAP `HABs-*`)

- **Site:** [calhabmap.org](https://calhabmap.org), [habs.sccoos.org](http://habs.sccoos.org/).
- **Access:** `https://erddap.sccoos.org/erddap/tabledap/HABs-<Station>.csv?…`. There are 17 datasets. The CeNCOOS ERDDAP search found no `HABs` or Pseudo-nitzschia datasets (404), so SCCOOS is the home.
- **Owner/terms:** CalHABMAP and SCCOOS. ERDDAP `license`: "may be used and redistributed for free but is not intended for legal use".
- **Cadence:**
  - Weekly grab samples, with a median gap of 7 days at all 8 active piers.
  - **Northern stations have been inactive for about 6 months:** Trinidad last sampled 2026-04-17, Humboldt 04-07, BML 04-13, Inner Tomales 03-03. Queries from May onward return 404 "outside actual_range".
- **Coverage / res.:** pier points.
- **Latest sample date, per field (today):**

| Station | Counts (PN seriata) | pDA | Temp |
|---|---|---|---|
| Santa Cruz Wharf | 09-30 | **09-30** | 09-30 |
| Monterey Wharf | — | — (last 2022-08) | 10-07 |
| Morro Bay front/back | 09-01 | — | 09-01 |
| Cal Poly Pier | 10-04 | 08-03 | 10-04 |
| Stearns Wharf | 09-28 | 08-03 | 09-28 |
| Santa Monica Pier | 10-05 | 08-03 | 10-05 |
| Newport Beach Pier | 06-22 | 08-10 | 08-18 |
| Scripps Pier | 09-14 | 08-10 | 10-05 |

- **Lag:**
  - Counts are usually posted within 0–3 days. Santa Monica 10-05 was already present on 10-08 (02-data-sources).
  - **pDA arrives in batches about 8–10 weeks later**, except at Santa Cruz.
  - There is no `date_created`, `date_modified` or `history` in the metadata. ERDDAP `Last-Modified` equals the request time, and `time_coverage_end` lags the data. Lag can only be measured by diffing successive pulls, so `first_seen` per row is needed.
- **Failure:** ERDDAP returns 404 for empty query results and 404/502 while reloading (`observations.py` already retries). Stations go silent with no announcement.
- **Authority:** scientific monitoring. Not regulatory.
- **Decisions:** context ("is a bloom building near me").
- **CW today:** full ingest of all 17 stations.
- **Value of a change:** a **per-variable** last-sample date (pDA vs counts) in the UI. Today a 10-05 station date implies a fresh toxin value when the pDA is from 08-03. Record a `first_seen_at` per row so lab lag can be measured.
- **Mode:** **API ingest (existing)**. Label it "weekly samples", never "real-time".

### 2.3 California HAB Bulletin (SCCOOS)

- **Site:** [sccoos.org/california-hab-bulletin](https://sccoos.org/california-hab-bulletin/). Issues are hosted on the SCCOOS site and on Emma (`t.e2ma.net/webview/…`).
- **Access:** HTML or e-mail only, with a listserv. The page sends `Last-Modified` equal to the fetch time (dynamic WordPress), so it is useless as a change signal. SCCOOS robots.txt sets `Crawl-delay: 10`.
- **Cadence:** about every two months, retrospective.
- **Latest:** the **"May–June 2026"** issue (Emma webview). Before that came Mar–Apr 2026 and Jan–Feb 2026. No Jul–Aug 2026 issue has been posted.
- **Lag:** about 2–4 months after the period it covers.
- **Authority:** expert scientific synthesis (Kenitz and Anderson, SCCOOS), drawing on C-HARM, HABMAP, IFCB, CDPH and marine-mammal strandings.
- **CW today:** none.
- **Value:** a context and education link.
- **Mode:** **link only**, plus a weekly change monitor of the issue list, with no review needed because nothing regulatory is implied.

### 2.4 NOAA C-HARM v3.1

- **Access:** `coastwatch.pfeg.noaa.gov/erddap/griddap/wvcharmV3_{0..3}day` (CoastWatch already ingests it).
- **Latest:**
  - `history` begins "**2026-10-13**: DINEOF gap filling…", so the issue is 10-13.
  - Nowcast valid 2026-10-12T12Z; `_3day` valid 2026-10-15T12Z.
  - In the 0-day series from 09-25, **10-05 and 10-08 are missing**.
- **Lag:** valid day + 1.
- **Failure:** **503 on the first request at 23:17:03Z**; 200 at 23:17:35Z. `date_created/issued/modified` are frozen at 2022-12-07.
- **Authority:** scientific model forecast, labelled experimental; not regulatory.
- **Mode:** keep the existing API ingest. Parse the `history` date as the issue date (data-health §5 #9).

### 2.5 Satellite chlorophyll and SST (latest `time` today)

| Dataset | Host | Latest | Lag vs today | Note |
|---|---|---|---|---|
| `erdVHNchla1day` (S-NPP 750 m) | pfeg | 2026-10-10 | 3.5 d | Recovered from 10-06 this morning. S-NPP ends 2026-11-02 (data-health) |
| `noaacwN20VIIRSchlaDaily` | coastwatch.noaa.gov | 2026-10-11 | 2.5 d | Best NOAA-20 replacement candidate |
| `noaacwN20/N21VIIRSchlaSectorUSDaily` | coastwatch.noaa.gov | **2026-10-05** | **8 d** | Stalled; 02-data-sources lists these as "Primary" |
| `noaacwS3AOLCIchlaSectorCIDaily` | coastwatch.noaa.gov | 2026-10-12T18:46Z | 1.2 d | Healthy |
| `noaacwS3BOLCIchlaSectorCIDaily` | coastwatch.noaa.gov | **2026-10-01** | 12 d | Still stalled (as in the earlier report) |
| `noaacwBLENDEDsstDNDaily` (Geo-Polar 5 km) | coastwatch.noaa.gov | 2026-10-11 | 2.5 d | Was also 10-11 at 05:35Z, so no new day was added during the UTC day |
| `noaacwLEOACSPOSSTL3SnrtCDaily` (2 km) | coastwatch.noaa.gov | 2026-10-12 | 1.5 d | |
| `jplMURSST41` | pfeg | 2026-10-12T09Z | 1.6 d | |

- **Owner/terms:** NOAA, free redistribution, "not for legal use". OLCI requires "Contains modified Copernicus Sentinel data".
- **Authority:** scientific observation (L3 has cloud gaps; L4 is interpolated).
- **Mode:** API ingest. Freshness rules are in data-health. **Add a per-dataset stall alarm**: no new `time` for more than 3 days while the server answers.
- **Robots:** coastwatch.noaa.gov disallows `/erddap/` for crawlers and sets `Crawl-delay: 10`. See §5.

### 2.6 NWS (api.weather.gov)

- **Access:** `/alerts/active?zone=PZZxxx`, `/zones/coastal/PZZxxx/forecast` and `/products/types/CWF/locations/{MTR,EKA,LOX,SGX}`.
- **Owner/terms:** NOAA NWS, public domain. The [API docs](https://www.weather.gov/documentation/services-web-api) ask for an identifying User-Agent.
- **Live:**
  - **PZZ535 (Monterey Bay):** Small Craft Advisory sent 13:42 PDT, ends 21:00 PDT. Forecast `updated` 13:41 PDT, 11 periods.
  - **PZZ455:** no alerts. Forecast `updated` 15:05 PDT.
  - **PZZ356 (Cape Blanco–Pt St George, MFR):** SCA sent 12:24 PDT, until 10-14 23:00 PDT. Forecast updated 12:25 PDT.
- **Headers:**
  - Alerts: `cache-control: max-age=5`, weak `ETag`.
  - Zone forecast: `max-age=3600, s-maxage=120`, no ETag.
  - CORS is `*`.
  - The alerts `updated` field changes every minute, so it is not a change signal. Use the alert `id` and `sent` instead.
- **Cadence (measured):** MTR CWF issued at 03:58, 10:10, 16:08, 21:39 (10-12), and 03:10, 09:53, 15:30, 20:41Z (10-13): **4 routine issuances a day**, plus amendments. EKA has the same pattern.
- **Lag:** minutes.
- **Failure:** occasional 5xx and rate limits (ocean-sources). CoastWatch's route shows "unavailable" with a reason.
- **Robots:** `Disallow: /`. See §5.
- **Authority:** official forecast and warnings. Not fishery regulation.
- **CW today:** live per place.
- **Value:** "what changed" items for new or upgraded alerts.
- **Mode:** API ingest (existing). Optionally add an hourly alert snapshot for change items.

### 2.7 NDBC realtime2

- **Access:** `https://www.ndbc.noaa.gov/data/realtime2/{id}.txt`. Sends `ETag` and `Last-Modified` with `max-age=600`. A **conditional GET returned 304**.
- **Latest (UTC):**
  - 46042: 22:50, WVHT 2.6 m, WTMP 17.2 °C. LM 23:10.
  - 46026: 22:50, 1.7 m. LM 23:05.
  - 46022: 22:50, 1.8 m. LM 23:10.
  - 46244 (CDIP relay): 22:26, 1.6 m. LM 22:50.
- **Lag:** about 15–25 min from observation to file, and about 30 min to the check.
- **Cadence:** 10-min meteorological data; waves every 30–60 min. Rows alternate wave and no-wave (already handled).
- **Owner/terms:** public domain.
- **Robots:** only bots 008 and Semrush are blocked.
- **Authority:** observation.
- **CW today:** live (route.ts).
- **Value of a change:** send `If-None-Match` from the server cache. The ranged read is already polite.
- **Mode:** API ingest (existing). **Genuinely real-time.**

### 2.8 CO-OPS Tides & Currents

- **Access:** `api.tidesandcurrents.noaa.gov/api/prod/datagetter`. It sends `no-store` and no validators.
- **Latest water level (MLLW, m, quality `p` = preliminary):**
  - Monterey 9413450: 23:12Z, 0.974 m.
  - Crescent City 9419750: 23:06Z, 1.511 m.
  - North Spit 9418767: 23:12Z, 1.723 m.
- **Lag:** 6–12 min.
- **Predictions (hilo, 48 h):** returned for all three.
- **Owner/terms:** public domain. Pass the `application=` parameter.
- **Robots:** the api host returns 403 for robots.txt; the main site returns 404.
- **Authority:** observation (6-min) and astronomical prediction. Predictions exclude surge.
- **CW today:** predictions only.
- **Value:** observed minus predicted, to show surge or anomaly, for harvesters at low tide.
- **Mode:** API ingest; the extension is cheap.

### 2.9 NOAA nowCOAST

- **Site:** [nowcoast.noaa.gov](https://nowcoast.noaa.gov/). It is a JavaScript single-page app; the home page `Last-Modified` is 2026-09-24. Map services run on GeoServer: WMS, WFS and time-enabled layers (e.g. `…/geoserver/observations/weather_radar/wms`).
- **Robots:** `Disallow: /geoserver/`.
- **Content:** it aggregates NWS, NDBC, CO-OPS, OFS and satellite data that CoastWatch can get from the primary sources.
- **Value:** none beyond what the primary sources give.
- **Mode:** **link only**. Do not integrate: it duplicates primary sources, and robots disallows the service path.

### 2.10 SCCOOS/CeNCOOS automated shore stations

- **Access:** `erddap.cencoos.org/erddap/tabledap/<id>.csv`.
- **Latest `time`:**
  - Humboldt Bay pier: **23:00Z** (21 min old).
  - Scripps Pier: **23:12Z** (9 min).
  - Monterey Wharf: **02:40Z**, 20.7 h stale. It was 02:40Z at the 05:15Z check too, so it has been frozen all day.
  - Tiburon: **2026-09-24**, 19 days stale.
- **Cadence:** 4–15 min. QARTOD flags are present.
- **Owner/terms:** IOOS-style "free, not for legal use".
- **Robots:** `/*?` and `*.csv*` are disallowed. See §5.
- **Authority:** observation. Fluorescence is not toxin.
- **CW today:** none.
- **Mode:** API ingest (staging), per ocean-sources rank 4, with a per-station offline badge.

### 2.11 CDFW fisheries regulations and in-season changes; feeds; robots

- **Pages:**
  - [Health-Advisories](https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories) is 39 KB. It sends **no ETag or Last-Modified**, `cache-control: no-cache, public, max-age=31536000, immutable` (contradictory) and `Expires: -1`. It has **45 HTML comments** (stale sections).
  - Dated items on that page today: 10/9, 10/8 (CDPH release, i.e. SN26-020), 9/14 ×2 and 9/11. **There is no entry for the 10/12 CDPH lift (SN26-021) yet**, so CDFW's page trails CDPH.
- **Feeds:**
  - **CDFW news RSS** (`GetRssFeeds?mid=56678&PortalId=0&tid=4232&itemcount=20`) has 20 items with `pubDate`. HAB items: "Del Norte County Recreational Razor Clam Fishery Closes…" (2026-10-09 15:56Z) and "Commercial and Recreational Take Restrictions… Northern Anchovy…" (09-11). It is all-agency news, so it is mostly hunting and grants and needs a filter.
  - **Marine blog RSS** `cdfwmarine.wordpress.com/feed/`: `lastBuildDate` 10-12. Feature articles, low value.
  - **GovDelivery** account `CNRA`. Signup IDs:
    - 35123: all Marine news
    - 42866: **Domoic Acid / PSP alerts**
    - 42890: Whale Safe Fisheries
    - 42892: MPAs
    - 42888 / 42889: Northern / Southern California news
    - 37436: press releases
  - Bulletins are public at `content.govdelivery.com/accounts/CNRA/bulletins/<hex>` and state "This bulletin was sent at 08/31/2026 02:46 PM PDT". No archive index or topic RSS was found.
  - E-mail subscription would require submitting an address. Not done (no sign-ups).
- **Regulation pages:**
  - Groundfish summary: "Updated June 23, 2026 … subject to in-season change".
  - Salmon page: recreational section "Updated August 19". "The 2026 commercial ocean salmon fishery has concluded". Hotlines **NMFS (800) 662-9825** and **CDFW (707) 576-3429** (verified on the page).
- **robots.txt:** wildlife.ca.gov returns **404**, so there are no restrictions. nrm.dfg.ca.gov has a sitemap only. cdph.ca.gov disallows `/_layouts/`, `/_vti_bin/` and `/_catalogs/`.
- **Authority:** **regulatory.** CDFW implements closures on OEHHA's recommendation (FGC §5523), using Director's Declarations.
- **Cadence:**
  - HAB closures are event-driven (2026: 9/11, 10/9, …).
  - Groundfish and salmon in-season changes are a few per year.
  - Page refreshes are unknown because there are no headers; hash them.
- **CW today:** two pages watched by full-text hash.
- **Value:**
  - RSS gives a **pubDate-stamped** event stream for releases.
  - A hidden-comment-stripped *section* hash (anchors `#crab`, `#razor-clam`, `#spiny-lobster`, `#finfish`) reduces false alarms.
  - GovDelivery is the fastest channel, but it is e-mail.
- **Mode:**
  - **Change monitor with human review** for the pages and RSS.
  - GovDelivery: let a project maintainer decide whether to subscribe a dedicated inbox. That needs a person's consent, and those e-mails would feed the same queue.
  - Regulation tables: **link only**; never restate bag or season rules (ocean-sources).

### 2.12 CDFW RAMP (Whale Safe Fisheries)

- **Site:** [Whale-Safe-Fisheries](https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries) (112 KB, no validators).
- **How actions are published:**
  - A dated paragraph on the page ("The Director of CDFW conducted a risk assessment on June 18, 2026 …").
  - The Director's Declaration and Final Assessment PDFs at `nrm.dfg.ca.gov/FileHandler.ashx?DocumentID=…&inline`.
  - News releases and GovDelivery topic 42890.
  - Example: DocumentID 245569 = "CDFW Final Assessment_2025-26 Season_20260618_ADA.pdf". It sends **LM 2026-06-18 22:32Z** and an ETag, and its text is extractable.
- **Cadence (from the page history):**
  - Risk assessments run about every 2–5 weeks from late October to June (2025-26: 10/24, 12/19, 3/13, 4/3, 4/17, 5/15, 6/18).
  - **The next assessment is expected around late October 2026** (prior years: 10/24, 10/25, 10/27, 10/28). This is inferred from history, not announced.
- **Zones GIS:** `biosds3120_fpu/FeatureServer/0` (CDFW BIOS ds3120). `dataLastEditDate` is **2026-04-21**, which postdates 02-data-sources' note that Zone 6 was removed in Oct 2025, so re-hash the geometry. Status is not an attribute.
- **Coverage:** 5 latitude zones; depth constraints (30 fathom).
- **Authority:** regulatory (14 CCR §132.8).
- **Decisions:** commercial and recreational Dungeness trap use, and the season opener.
- **CW today:** page hash.
- **Value:** zone polygons plus human-curated per-zone status.
- **Mode:**
  - Zones: **API ingest with review** (weekly geometry hash).
  - Status: **change monitor with human review** (page section plus new `DocumentID` links plus GovDelivery).

### 2.13 MPAs and groundfish lines

**CDFW ds582 MPAs**
- **Access:** `biosds582_fpu/FeatureServer/0`. `dataLastEditDate` 2024-01-09; copyright "CDFW Marine Region GIS Lab"; CC BY 4.0 per BIOS metadata (ocean-sources).
- **Pending change:** **the FGC hearing on MPA petitions is Oct 15–16, 2026** ([Independent, 2026-08-21](https://www.independent.com/2026/08/21/california-divided-over-ocean-conservation-proposals-including-creation-of-marine-protected-area-off-carpinteria-coast/); [OPC recommendations](https://opc.ca.gov/wp-content/uploads/2026/07/OPC-Recommendations-on-MPA-Petitions-July-2026-508.pdf)). Adopted changes take effect only after rulemaking and OAL approval. Until the GIS is updated, show "boundaries as of [layer date]; regulation changes may be pending".
- **Mode:** **API ingest with review** as a reference overlay, plus a **change monitor** of the FGC meeting outcomes page and the layer's `dataLastEditDate`.

**Groundfish recreational boundaries ds3143**
- `dataLastEdit` 2025-11-12.
- The RCA lines (ds3144) and the 20-fathom line (ds3160) are *not* at `biosds3144_fpu` or `biosds3160_fpu` (400 "Invalid URL"). Their service names differ and need a lookup in the BIOS catalog.

**NOAA RCA waypoints (50 CFR 660)**
- NOAA WCR GIS "approximate representation, not an official record" (ocean-sources).
- **Mode:** LATER, link plus a reference overlay with a disclaimer. Which depth is active is regulatory, so it must be curated.

### 2.14 Other sources

| Source | Status today | Mode |
|---|---|---|
| **NOAA Fisheries salmon in-season actions** (`fisheries.noaa.gov/bulletin/inseason-action-…`, GovDelivery `USNOAAFISHERIES`) | 2026 actions #1–#27 (to Aug 31) are all **north of Cape Falcon (WA/OR)** per search. The CA commercial season has "concluded" (CDFW). NMFS hotline (800) 662-9825 | **Link only** plus the CDFW salmon page in the change monitor (in-season changes are regulatory). Low priority until the 2027 season (PFMC, March–April) |
| **CDIP** (THREDDS) | `156p1_rt.nc` (Monterey Canyon Outer): `time_coverage_end` 23:00Z, `date_modified` 23:06Z. License: "may be redistributed and used without restriction". `cdip.ucsd.edu` robots is `Disallow: /`, so use THREDDS only | NDBC relays (462xx) already cover real-time waves. **LATER**: MOP nearshore wave forecasts (ocean-sources) |
| **HF radar** | Covered (`ucsdHfrW2`) plus the NDBC mirror proposal (data-health) | — |
| **Marine mammal strandings** (TMMC, NOAA stranding network) | Not real-time, and not regulatory. Summarized in the HAB Bulletin | **Link only** (via the Bulletin) |
| **CA IFCB Network** (Imaging FlowCytobot, near-real-time PN cell imagery at several piers; cited in the Bulletin) | Not probed today | **Investigate**: possibly the only automated (hours) *Pseudo-nitzschia* signal. Still not toxin |
| **OEHHA crab openings page** | Bot-blocked (02-data-sources) | Link; store memo PDF URLs as provenance |

---

## 3. Summary matrix

The cadence column records how often **new data** arrive, not how often the page refreshes.

| # | Source | New-data cadence (measured or derived) | Latest today | Pub. lag | Class | Authority | CW now | Mode |
|---|---|---|---|---|---|---|---|---|
| 1a | CDPH advisory releases | Event, about 2/month (21 in 2026) | SN26-021, 10-12 | Same day | Event-driven | **Regulatory/health** | Watched + curated | Change monitor + human review |
| 1b | CDPH crab/seafood DA PDFs | About weekly during events | Crab 9/28 (as of 10/8); seafood 9/29 | 10–14 d | Irregular lab | Lab results (program) | — | Change monitor + review (conditional GET) |
| 1c | CDPH monthly reports | Monthly | April 2026 | About 5.5 mo | Retrospective | Program report | — | Link only |
| 1d | CDPH web map | Hand-edited | Item modified 10-13 16:39Z | About 1 d after release | Event-driven | Display only | Geometry | Change monitor (signal only) |
| 1e | CDPH PN/Alexandrium layers | Weekly | Sample 9/28, edit 10/01 | About 3 d | Irregular | Scientific screening | — | API ingest (P2, archive) |
| 2 | CalHABMAP | Weekly (counts); pDA in batches | Counts 10-05; pDA 08-03/08-10 (SC 09-30) | Counts 0–3 d; pDA about 8–10 wk | Irregular lab | Scientific | Ingest | API ingest; per-variable dates |
| 3 | CA HAB Bulletin | About every 2 months | May–June 2026 | About 2–4 mo | Retrospective synthesis | Scientific | — | Link only |
| 4 | C-HARM | Daily, with gaps | Issue 10-13; nowcast 10-12 | About 1 d | Model forecast | Scientific (experimental) | Ingest | API ingest |
| 5 | Sat. chl/SST | Daily | 10-05 (stalled) … 10-12 | 1.2–3.5 d (8–12 d when stalled) | Near-real-time | Scientific | Partial | API ingest + stall alarm |
| 6 | NWS marine | 4×/day plus amendments; alerts as needed | SCA PZZ535/356 | Minutes | Real-time | **Official forecast/warning** | Live | API ingest |
| 7 | NDBC | 10–60 min | 22:50Z | About 20–30 min | Real-time | Observation | Live | API ingest |
| 8 | CO-OPS | 6 min | 23:12Z | 6–12 min | Real-time | Observation / prediction | Predictions | API ingest (+ water level) |
| 9 | nowCOAST | Mirrors others | — | — | Aggregator | — | — | Link only |
| 10 | Shore stations | 4–15 min | 23:12Z … stale 19 d | Minutes | Real-time (per station, when up) | Observation | — | API ingest (staging) |
| 11 | CDFW advisories, RSS, GovDelivery | Event | RSS 10-09 razor clam | Same day; the CDFW page trails CDPH | Event-driven | **Regulatory** | Page hash | Change monitor + human review |
| 12 | CDFW RAMP | About 2–5 wk, Oct–Jun | 2026-06-18 assessment | Same day | Event-driven | **Regulatory** | Page hash | Zones: ingest + review; status: monitor + review |
| 13 | MPAs ds582 / groundfish lines | Rulemaking | Edited 2024-01-09 / 2025-11-12 | Months after rule | Reference | **Regulatory (approx. GIS)** | — | Ingest + review; FGC watch |
| 14 | Salmon in-season, CDIP, strandings, IFCB | Varies | See §2.14 | — | — | — | — | Link / later |

---

## 4. Design: a compliant, respectful source monitor for page- and PDF-only sources

### 4.1 Principles

1. **Monitor, never infer status.** The monitor detects that an official source changed and proposes a review item. Only a person running `cwp review-official` (after reading the sources) may create, edit, verify, lift or expire a record. This is unchanged from the current design and from CLAUDE.md.
2. **Be polite by construction.**
   - Use an honest User-Agent: `CoastWatch-monitor/1.0 (+https://github.com/yashnil/habs-forecast; contact: <maintainer address>)`. **Replace `official.py: BROWSER_UA`**, which impersonates Chrome.
   - If a site blocks the honest UA, record the block and fall back to manual review. Do not spoof.
   - Make one request at a time per host, at least 10 s apart (SCCOOS and coastwatch.noaa.gov declare `Crawl-delay: 10`).
   - Use per-host daily request budgets.
   - Use exponential backoff and stop for the run on any 429 or 403.
3. **robots.txt first.** Fetch and cache robots.txt per host each day, and refuse any URL it disallows for `*` or for the CoastWatch UA. Today that excludes:
   - data.chhs.ca.gov and data.ca.gov `/api/`
   - cdph `/_layouts/`, `/_vti_bin/` and `/_catalogs/`
   - nowcoast `/geoserver/`
   - cdip.ucsd.edu (all paths)
   
   The data APIs whose robots files disallow crawling (ERDDAP, api.weather.gov) are discussed in §5.
4. **Lowest frequency that meets the review SLO.** Data-health sets an SLO of human review within 3 days, so a 6-hourly check is more than enough:

| Source | Cheap pre-check | Frequency |
|---|---|---|
| CDPH list and release pages | SharePoint `_api … ?$select=TimeLastModified` (about 1 KB). Fetch full HTML only when the value changes, plus once a week regardless | 6 h. Daily outside active events |
| CDPH PDFs (crab, seafood) | `If-None-Match` / `If-Modified-Since` → 304 | Daily |
| CDPH web map | `items/<id>?f=json` `modified` | 6 h |
| CDFW Health Advisories and Whale-Safe pages | No validators: full GET, section hash | 6 h Oct–Jun (crab season), daily otherwise |
| CDFW news RSS | Full GET (no validators); diff by `guid`/`link` + `pubDate` | 6 h |
| nrm.dfg.ca.gov declaration PDFs | New `DocumentID`s found on watched pages; then a conditional GET per known PDF | On discovery; weekly re-check |
| GIS layers (ds3120, ds582, ds3143) | `?f=json` `editingInfo.dataLastEditDate` + `ETag` | Weekly |
| FGC meeting and outcome pages, CDFW salmon and groundfish pages | Section hash | Weekly; daily in the week of an FGC meeting |
| HAB Bulletin | Issue list | Weekly |

### 4.2 Pipeline per watched source

1. **Fetch** with validators taken from the last snapshot (`If-None-Match`, `If-Modified-Since`). A 304 means no change: record `checked_at`, `http_status=304` and stop.
2. **Normalize the relevant section only**, using a per-source extractor (as `main_text` and `cdph_release_items` do today):
   - Strip `<!-- -->`, scripts, zero-width spaces and degree-sign variants.
   - Cut to the anchor section (`#crab`, `#razor-clam`, `#dungenesscrabstatus`, `#DeltaPlaceHolderMain`).
   - For PDFs, extract text with `pdftotext -layout` and normalize whitespace.
   - For the web map, hash the sorted `[(title, url, visibility, definitionExpression)]`.
3. **Hash** the normalized text (SHA-256, as `fingerprint()` already does) and also the raw bytes. If the section hash is unchanged, record "unchanged (cosmetic)".
4. **Archive immutably** on change, or on first sight. Store the raw bytes in content-addressed storage (`sha256/<hash>`), never overwritten, together with a sidecar holding:
   - `url`, `final_url`, `retrieved_at` (UTC)
   - `http_status` and the response headers (`ETag`, `Last-Modified`, `Content-Type`, `Date`)
   - `raw_sha256`, `normalized_sha256` and `extractor_version`
   - `robots_checked_at` and `user_agent`
   
   This needs a store that is **not** the public data branch, since we must not republish full agency pages. Options: a private bucket, or a git-LFS repo with access limited to reviewers.
5. **Diff** the normalized text against the previous snapshot as a unified diff. Then extract **items**:
   - new `SN##-###` IDs
   - new RSS `guid`s
   - new `DocumentID`s
   - new dated paragraphs ("risk assessment on <date>")
   - changed PDF ETag versions
   - web map layer visibility flips
   - GIS `dataLastEditDate` changes
6. **Queue for human review.** Open or append a review item: today a GitHub issue (`notify-review`); later a `review_queue.json`. Each item carries the diff, links to the live URL and the archived snapshot, the extracted candidate items, and suggested (not applied) registry edits. `cwp review-official` should list the open queue items and require the reviewer to resolve each one (accepted → registry edit; dismissed as cosmetic → reason) before it records `human_verified`.
7. **Publish only signals, never status.** The app already shows "not verified" when there is a pending change. Discrepancies such as `reported_lifts` remain published conflict notes.

### 4.3 Item types for "What changed since your last visit"

Every item must keep **agency time** and **CoastWatch time** separate. The UI sorts and filters by `first_seen_by_coastwatch`, but it *describes* the item by the agency date. An old notice seen for the first time must therefore read "CoastWatch first saw on Oct 13 a notice CDPH issued on May 2, 2024", never "new notice".

| Item type | Trigger | Required metadata |
|---|---|---|
| `official_notice_issued` | New CDPH SN release, CDFW news or declaration, or GovDelivery bulletin, confirmed by a reviewer | `agency`, `source_url`, `agency_issued_at` (date on the document; null if not stated, with `issued_date_note`), `agency_effective_at`, `agency_expected_end` (+ note), `first_seen_by_coastwatch` (earliest snapshot `retrieved_at` containing it), `reviewed_by`, `reviewed_at`, `snapshot_sha256` |
| `official_notice_lifted` / `_expired` / `_extended` | A reviewer changes a record's status after reading the agency text | All of the above, plus `record_id`, `previous_status`, `agency_lifted_at` (from the document), `status_changed_in_coastwatch_at` |
| `official_source_changed_unreviewed` | The monitor saw a change; nobody has reviewed it yet. Shown as "Agency page changed; not yet reviewed" | `source_id`, `first_seen_by_coastwatch`, `last_checked_at`, `diff_summary`, **no** interpretation |
| `official_discrepancy` | E.g. `reported_lifts`, CDPH vs CDFW disagreement, or web map vs release | `sources[]`, `first_seen_by_coastwatch`, `explanation` |
| `lab_results_posted` | New CDPH crab or seafood PDF version (after review), or new CalHABMAP pDA rows | `sample_collected_dates[]`, `agency_as_of` (PDF "As of"), `file_last_modified`, `first_seen_by_coastwatch`, `lab`/`program` |
| `monitoring_sample_posted` | New CalHABMAP counts or CDPH plankton rows | `sampled_at`, `first_seen_by_coastwatch`, `station`, `variables_updated[]` (so pDA vs counts is explicit) |
| `forecast_issued` | New C-HARM run | `issued_date` (+ basis: `history` or derived), `valid_dates[]`, `first_seen_by_coastwatch` |
| `marine_alert_issued` / `_cancelled` | New or ended NWS alert for a followed zone | `nws_id`, `sent`, `onset`, `ends`, `first_seen_by_coastwatch` |
| `management_action` (RAMP) | New risk-assessment paragraph or declaration PDF (after review) | `assessment_date`, `effective_at`, `zones[]`, `document_url`, `first_seen_by_coastwatch` |
| `reference_layer_updated` | MPA, RAMP or groundfish GIS `dataLastEditDate` change; FGC adoption | `layer_edit_date`, `regulation_effective_date` (if known), `first_seen_by_coastwatch` |
| `source_status_changed` | A station goes offline, a dataset stalls, or a source recovers | `last_observation_at`, `detected_at` |

**Rules:**
- An item is "new since your last visit" if `first_seen_by_coastwatch > last_visit`. Its headline uses the agency date. Add the "(older notice, newly found)" qualifier when `agency_issued_at` is more than 7 days before `first_seen_by_coastwatch`.
- `first_seen_by_coastwatch` is set once and never rewritten. Backfilled items (e.g. from the first run of a new watcher) get `backfilled=true` and are excluded from "since your last visit".
- Post-publication edits to the same notice (e.g. SN26-021 modified 7 h after creation) produce an `official_source_changed_unreviewed` item with `agency_page_modified_at`, not a new notice.
- Per the manifest-compatibility memory, these fields belong in a **new, separately versioned artifact** (e.g. `changes-v1.json`). Do not add them to the existing manifest or `official-*.json` schemas.

---

## 5. Compliance notes and decisions for the user

1. **Data APIs whose robots.txt disallows crawling:**
   - `api.weather.gov` (`Disallow: /`)
   - ERDDAP hosts (`/*?`, `*.csv*`, or `/erddap/`)
   
   These files exist to stop search-engine crawling of infinite query URLs. The providers document programmatic API use (the NWS API docs; ERDDAP's RESTful services). CoastWatch already depends on both. I treated targeted, documented API queries as *not* scraping and kept them to the minimum (about 45 ERDDAP and 11 NWS requests).
   
   **Decision needed:** accept this interpretation, record it in `docs/coastwatch/02-data-sources.md`, and send identifying UAs. Alternatively, ask NOAA CoastWatch, SCCOOS and CeNCOOS to confirm. For coastwatch.noaa.gov, honour `Crawl-delay: 10` between requests regardless.
2. **My robots lapse:** one CKAN `/api/3/action/package_search` request each to data.chhs.ca.gov and data.ca.gov was sent in the same command as the robots.txt check, before I had read the result. Both portals disallow `/api/`. No further requests were made, and nothing should be automated against those paths.
3. **The watcher's UA impersonates a browser.** `official.py: BROWSER_UA` is "Mozilla/5.0 … Chrome/141.0 … CoastWatch-watcher" and is used for the CDFW and CDPH pages. Today an honest UA got HTTP 200 from cdph.ca.gov, wildlife.ca.gov and nrm.dfg.ca.gov, so the impersonation is unnecessary. **Recommendation:** switch to an honest UA with a contact address, and test it in staging.
4. **GovDelivery e-mail subscription** (CNRA topics 42866 and 42890) would be the fastest CDFW signal, but it requires submitting an e-mail address. That is a maintainer decision; I did not subscribe.
5. **CDPH `_api` use.** It is not disallowed and is the call the page itself makes. Keep it to the pre-check (`$select=TimeLastModified`) and do not enumerate libraries.

---

## 6. Evidence log (UTC, 2026-10-13; full lines in [`evidence/source-audit-log.txt`](evidence/source-audit-log.txt))

| Time | Request | Status | Key headers / payload |
|---|---|---|---|
| 23:12:04 | cdph.ca.gov/robots.txt | 200 | Disallow `/_layouts/`, `/_vti_bin/`, `/_catalogs/` |
| 23:12:05 | wildlife.ca.gov/robots.txt | 404 | none |
| 23:12:05 | nrm.dfg.ca.gov/robots.txt | 200 | sitemap only |
| 23:12:06 | sccoos.org/robots.txt | 200 | `Crawl-delay: 10` |
| 23:12:12 | CDPH Shellfish-Advisories.aspx | 200 | 457,896 B; `cache-control: private`; **no ETag/LM**; SN26-021 and SN26-020 at top |
| 23:12:16 | CDPH Marine-Biotoxin-Monitoring-Program.aspx | 200 | no validators |
| 23:12:34 | SN26-021.aspx | 200 | "CDPH Lifts Warnings … Monterey County, October 12, 2026"; mussel quarantine "remains in effect" |
| 23:12:38 | SN26-020.aspx | 200 | "… Razor Clams from Del Norte County, October 9, 2026" |
| 23:12:53 | `_api … Shellfish-Advisories.aspx ?$select=TimeLastModified` | 200 | TimeLastModified **2026-10-12T16:01:27Z** |
| 23:12:53 | `_api … SN26-021.aspx` | 200 | TimeCreated 15:47:21Z; TimeLastModified **22:58:14Z** |
| 23:12:54 | `_api … Marine-Biotoxin-Monitoring-Program.aspx` | 200 | TimeLastModified 2026-03-03T23:43:41Z |
| 23:13:10 | Marine-Biotoxin-Monitoring-Reports.aspx | 200 | newest monthly StoryMap: April 2026 |
| 23:13:13–16 | Quarantines-and-Health-Advisories, Annual-Mussel-Quarantine | 200 | link to experience 394836… |
| 23:13:18 | FDB DomoicAcid.aspx | 200 | links CrabDAResults.pdf ("Sept 29–Oct 8"), SeafoodDAResults.pdf |
| 23:13:29 | CrabDAResults.pdf | 200 | LM Mon 12 Oct 2026 17:23:06 GMT; ETag `{9D219E7D…},7`; collected 9/28; "As of 10/8/2026" |
| 23:13:30 | SeafoodDAResults.pdf | 200 | LM 12 Oct 15:10:26 GMT; ETag `{A6A0CA12…},4`; anchovy 8/28, 9/29 |
| 23:13:38 | CrabDAResults.pdf with If-None-Match / If-Modified-Since | **304 / 304** | conditional GET works |
| 23:13:52 | arcgis.com items/ad52187023f24b09b073d287f656df14 | 200 | modified **2026-10-13 16:39:53Z**; access public |
| 23:13:53 | …/data?f=json | 200 | ETag "1a136227fa8"; "Shellfish Safety Notification Lifted" def `County_Name='Monterey County'` visible |
| 23:14:04–05 | CDPH FS Export_Output, Symbols Red/Yellow, Coastal_Counties, Special_Advisory | 200 | lastEditDate 2020-07-16 / 2020-08-25 / 2020-08-25 / 2020-08-26 / 2025-07-10 |
| 23:14:20–21 | Pseudo_nitzschia/FeatureServer/6 (+ stats) | 200 | dataLastEdit 2026-10-01 19:42Z; Date_Sampled 08-17…**09-28**; n=82 |
| 23:14:29–30 | CDFW_Fisheries_Closure(_Boundaries), SampleDetails (CDPH org) | 200 | LM 2021-10-04 / 2021-10-05 / 2016-01-11 (dead) |
| 23:14:38 | data.chhs.ca.gov/robots.txt | 200 | **Disallow /api/** |
| 23:14:39 | data.chhs.ca.gov CKAN package_search | 200 | count 0 (**sent before robots was read; see §5**) |
| 23:14:40 | data.ca.gov CKAN package_search | 200 | 1 unrelated (**same lapse**) |
| 23:14:46 | data.ca.gov/robots.txt | 200 | Disallow /api/ |
| 23:14:56 | erddap.sccoos.org info HABs-SantaCruzWharf | 200 | no date_created/modified; time_coverage_end 2026-09-30T15:30Z |
| 23:15:11–48 | 17× `HABs-<stn>.csv?time,pDA,PN_seriata,Temp&time>=2026-05-01` | 8×404 (no rows), 9×200 | table in §2.2 |
| 23:16:02 | orderByMax(time) Trinidad/Humboldt/BML/InnerTomales | 200 | 04-17 / 04-07 / 04-13 / 03-03 |
| 23:16:09–18 | erddap.cencoos.org search "HABs domoic", "pseudo-nitzschia" | 404 | no matching datasets |
| 23:16:29 | sccoos.org/california-hab-bulletin/ | 200 | LM = fetch time (dynamic); latest listed "2026: May & June" |
| 23:16:42 | t.e2ma.net webview (May–June 2026 bulletin) | 200 | Kenitz and Anderson summary |
| 23:17:03 | wvcharmV3_0day time[(last)] | **503** | "Service Unavailable … (temporary?)" |
| 23:17:05 | wvcharmV3_3day time[(last)] | 200 | 2026-10-15T12Z |
| 23:17:05–08 | erdVHNchla1day / jplMURSST41 / N20 chl / N20 sector / N21 sector / OLCI S3A / S3B / Blended / ACSPO | 200 | 10-10 / 10-12T09 / 10-11 / **10-05** / **10-05** / 10-12T18:46 / **10-01** / 10-11 / 10-12 |
| 23:17:35 | wvcharmV3_0day time[(2026-09-25):(last)] | 200 | 16 steps, missing 10-05, 10-08; last 10-12 |
| 23:17:35 | info/wvcharmV3_0day | 200 | history "2026-10-13: DINEOF…"; date_* 2022-12-07; product_version 3.1 |
| 23:17:44–45 | NWS alerts PZZ535 / PZZ455 / PZZ356 | 200 | weak ETag; max-age=5; SCA / none / SCA |
| 23:17:45–46 | NWS zones/coastal/{PZZ535,455,356}/forecast | 200 | updated 13:41 / 15:05 / 12:25 PDT; max-age=3600 s-maxage=120 |
| 23:17:53–54 | NWS products/types/CWF/locations/MTR, EKA | 200 | MTR issuances 20:41, 15:30, 09:53, 03:10Z … |
| 23:17:54 | api.weather.gov/robots.txt | 200 | **Disallow: /** |
| 23:18:03–08 | robots: coastwatch.pfeg (`/*?`, `*.csv*`), coastwatch.noaa.gov (`/erddap/`, crawl-delay 10), ndbc (2 bots only), api.tidesandcurrents (403), tidesandcurrents (404), erddap.cencoos (`/*?`, `*.csv*`), services2.arcgis.com (403), arcgis.com (none), nowcoast (`/geoserver/`), fisheries.noaa.gov (core paths), public.govdelivery (502), content.govdelivery (404), cdip.ucsd.edu (**Disallow: /**), thredds.cdip (404), storymaps (`/themes`) | — | — |
| 23:18:25 | NDBC realtime2 46042 / 46026 / 46022 / 46244 | 200 | LM 23:10:46 / 23:05:41 / 23:10:46 / 22:50:53; newest rows 22:50/22:50/22:50/22:26Z |
| 23:18:35 | 46244.txt with If-None-Match | **304** | conditional GET works |
| 23:18:35–37 | CO-OPS water_level latest 9413450 / 9419750 / 9418767 | 200 | 23:12 / 23:06 / 23:12Z; q=p; `no-store` |
| 23:18:35–37 | CO-OPS hilo predictions (same stations) | 200 | predictions returned |
| 23:18:44 | nowcoast.noaa.gov | 200 | SPA; LM 2026-09-24 |
| 23:19:03 | wildlife.ca.gov/Marine-Subscribe | 200 | GovDelivery CNRA signup IDs 35123, 42866, 42888–42892, 42867, 43142, 37436 |
| 23:19:03 | CDFW Health-Advisories | 200 | no validators; 45 HTML comments; latest dated item 10/9/2026 |
| 23:19:04 | CDFW Whale-Safe-Fisheries | 200 | latest assessment June 18, 2026 |
| 23:19:19 | content.govdelivery.com/accounts/CNRA/bulletins | 404 | no public archive |
| 23:19:20 | public.govdelivery.com signup/42866 | 200 | title "CDFW Marine Region News – Domoic Acid and other public health alerts" |
| 23:19:45 | content.govdelivery.com/accounts/CNRA/bulletins/427c3f0 | 200 | "This bulletin was sent at 08/31/2026 02:46 PM PDT" |
| 23:19:55 | cdfwmarine.wordpress.com/feed/ | 200 | lastBuildDate 12 Oct 2026 18:58Z |
| 23:20:24 | CDFW GetRssFeeds?mid=56678&tid=4232&itemcount=20 | 200 | 20 items; razor clam 10-09 15:56Z; anchovy 09-11 |
| 23:20:47 | nrm FileHandler DocumentID=245569 | 200 | LM 18 Jun 2026 22:32:34; ETag; "Final Assessment … 20260618" |
| 23:20:56–57 | BIOS ds3120 / ds582 / ds3143 FeatureServer/0 | 200 | dataLastEdit 2026-04-21 / 2024-01-09 / 2025-11-12 |
| 23:20:57 | biosds3144_fpu, biosds3160_fpu | 200 (body error 400) | "Invalid URL": the service names differ |
| 23:21:15–22 | CeNCOOS orderByMax(time): Humboldt / Monterey wharf / Scripps / Tiburon | 200 | 23:00Z / **02:40Z** / 23:12Z / **2026-09-24** |
| 23:21:33 | thredds.cdip.ucsd.edu 156p1_rt.nc.das | 200 | time_coverage_end 23:00Z; date_modified 23:06:34Z; licence "without restriction" |
| 23:21:41 | CDFW Groundfish-Summary | 200 | "Updated June 23, 2026" |
| 23:21:42 | CDFW Salmon regs | 200 | recreational "Updated August 19"; commercial "concluded"; hotlines (800) 662-9825, (707) 576-3429 |

**Web searches** (results were used as pointers only, not as evidence of current status):
- [nowCOAST GeoServer WMS (Esri sample)](https://developers.arcgis.com/net/maui/sample-code/wms-layer-url/)
- [CDFW Marine Subscribe](https://wildlife.ca.gov/marine-subscribe)
- [GovDelivery CNRA bulletin 427c3f0](https://content.govdelivery.com/accounts/CNRA/bulletins/427c3f0)
- [Independent 2026-08-21 (FGC Oct 15–16)](https://www.independent.com/2026/08/21/california-divided-over-ocean-conservation-proposals-including-creation-of-marine-protected-area-off-carpinteria-coast/)
- [OPC MPA recommendations July 2026](https://opc.ca.gov/wp-content/uploads/2026/07/OPC-Recommendations-on-MPA-Petitions-July-2026-508.pdf)
- [NOAA in-season action #27](https://www.fisheries.noaa.gov/bulletin/inseason-action-27-ocean-salmon-fisheries-north-cape-falcon-recreational)

**Not verified in this session:**
- the cdph.ca.gov and wildlife.ca.gov terms-of-use pages
- the topic RSS for GovDelivery
- the BIOS service names for ds3144 and ds3160
- CA IFCB Network endpoints
- the FGC Oct 15–16 outcome (it has not happened yet)
