# CoastWatch: Suomi-NPP end-of-delivery audit (read-only)

Auditor role: satellite oceanographer. Date of checks: 2026-10-13, 07:02–07:11 UTC.
Code: worktree `wt-redesign` @ c638f274 (read only; nothing edited, nothing committed, nobody contacted).
Raw evidence (ERDDAP info JSON, subsets, GIBS capabilities, live manifest): `research3/live/` next to this file.
"UNVERIFIED" marks anything I did not confirm against a primary source.

---------------------------------------------------------------------------------------------------
## 0. The deadline (what NOAA actually says)

- NOAA NESDIS notice, issued 2026-08-03 16:00 UTC: "All S-NPP data products" stop; VIIRS "data will no longer be
  available after November 2, 2026, at 13:00 UTC"; outage "Permanent"; "All users should transition to using
  NOAA-21 (Primary) and NOAA-20 (Secondary) science products" (via PDA).
  https://www.nesdis.noaa.gov/news/cessation-of-suomi-national-polar-orbiting-partnership-s-npp-data-users-onafter-november-01-2026 (fetched 2026-10-13)
- CoastWatch news: 2026-08-10 item said **November 1, 2026** 13:00 UTC; "Update #1" of **2026-10-01** moved it to
  **November 2, 2026 13:00 UTC**. CoastWatch suggests the help desk for "finding replacement data (likely NOAA-20 VIIRS)".
  https://coastwatch.noaa.gov/cwn/news/2026-10-01/cessation-suomi-national-polar-orbiting-partnership-s-npp-data-update-1.html
  https://coastwatch.noaa.gov/cwn/news/2026-08-10/cessation-suomi-national-polar-orbiting-partnership-s-npp-data.html
- NASA Earthdata alert (published 2026-08-07, updated 2026-10-05): 13:00 UTC 2026-11-02 (2026/306); "the cessation
  date ... is subject to change". https://www.earthdata.nasa.gov/data/alerts-outages/suomi-npp-data-product-delivery-cease-november-1-2026
- None of these notices names any ocean-colour, ERDDAP, DINEOF or C-HARM product. Expect the last S-NPP passes on
  2026-11-02 morning UTC; with 5–7 d latency of the West Coast 750 m composites, the last S-NPP pixels may still be
  *appearing* until roughly 2026-11-07..09, then nothing newer.

---------------------------------------------------------------------------------------------------
## 1. Inventory: every CoastWatch dependency on S-NPP

Paths are relative to `wt-redesign/`. Line numbers at c638f274.

### 1a. Direct S-NPP datasets (CoastWatch fetches them)

| # | Dataset (server) | Upstream metadata proving S-NPP | Code | What it feeds |
|---|---|---|---|---|
| D1 | `erdVHNchla1day` (https://coastwatch.pfeg.noaa.gov/erddap) var `chla` | `satellite = Suomi-NPP`; summary "VIIRS ... aboard the Suomi-NPP satellite ... NOAA-MSL12 NRT processing with LAND+CLDICE+HIGLINT+HISATZEN+HISOLZEN masking"; history "from daily merge files ... star.nesdis.noaa.gov/thredds/catalog/chloraNPPVIIRSSectorDaily"; source_data `VRSVCW_B2026279_C20/C49/C43 ... UZ00/UY00/VY00_edgemask_chlora.nc` | `pipeline/coastwatch_pipeline/sources/satellite.py:9-10` (doc), `:76` LICENSE_VIIRS, `:83` freshness note, `:154-168` `VIIRS = Product(key="viirs750", server=pfeg, platforms=(("VIIRS", (("erdVHNchla1day", ...),)),), variable="chla", step=0.0075, native_m=750, source_name="VIIRS chlorophyll-a, 750 m daily composite (NOAA CoastWatch West Coast)")`; run at `:674-678`; `pipeline.py:208-209` (source title "Sentinel-3 OLCI 300 m (VIIRS 750 m fallback)") | Map layer `viirs750_chl_latest` (VIIRS product in the satellite selector); **multi-sensor** `multisensor_chl_latest` secondary member (`sources/multisensor.py:176-253`, title at `:229`, member label `:242`, provenance dataset_id `"<olci ids>,erdVHNchla1day"` at `:250-251`); `SatelliteNear` readout fallback when OLCI absent (`coastwatch-web/src/components/map/SatelliteNear.tsx:26-27,58`); first-paint fallback (`components/LiveOceanMap.tsx:125`, `LayerDock.tsx:905,924`, `lib/opening.ts:23`); health `satellite_stale` (`health.py:38` VIIRS_MAX_AGE_DAYS=8, `:78`) |
| D2 | `erdVHNchla8day` (pfeg) var `chla` | same S-NPP attributes; source_data `VHN2026279_2026279_chla.nc ...`; time_coverage_end 2026-10-03T00:00Z | `sources/port_intel.py:8-9` (doc), `:48-49` `CHL_DATASET="erdVHNchla8day"`, `CHL_SERVER=pfeg`; request `:190`; parse `:196-200` (fill, `<=0` and `>200` dropped); method text `:326` ("NOAA VIIRS S-NPP 750 m pixels (8-day composites ...)"); provenance `:341-349` (`source_id="viirs_chl_8day"`, source_name "NOAA VIIRS S-NPP chlorophyll-a, 750 m, 8-day composite ...") | Port/My Coast panel chlorophyll (latest + 60-day history, 15 km median): `components/port/PortPanel.tsx:47,183,224`, copy `content/copy.ts:65` |

### 1b. Indirect S-NPP dependency (NOAA-produced input; CoastWatch cannot change it)

| # | Dataset | Evidence | Code | Feeds |
|---|---|---|---|---|
| I1 | C-HARM v3.1 `wvcharmV3_0day/1day/2day/3day` (pfeg) | see §2 — every lead says S-NPP VIIRS DINEOF inputs | `sources/charm.py` (SERVER `:42`, ids `:61`, EXPECTED_VERSION "3.1" `:44`, version check `:228-229`, provenance `:325-345`); `port_intel.py:141,325,331-340`; `verify.py:54,106` | The whole forecast product: C-HARM map layers (PN, pDA, cDA, leads 0–3), Bloom page, port C-HARM stats/history, verify-charm |

### 1c. Things that are NOT S-NPP (checked)

- NASA GIBS layer used by CoastWatch is `VIIRS_NOAA20_Chlorophyll_a` (`sources/gibs.py:58-64`) plus `OCI_PACE_Chlorophyll_a` (`:65-71`). **No S-NPP GIBS layer is used.** GIBS capabilities at 07:07:36Z: `VIIRS_NOAA20_Chlorophyll_a` default 2026-10-12, range 2018-02-24/2026-10-12; `VIIRS_NOAA21_Chlorophyll_a` default 2026-10-12, range 2025-10-25/2026-10-12 (available as a GIBS fallback); no `VIIRS_SNPP_Chlorophyll_a` (only `VIIRS_SNPP_L2_Chlorophyll_A`, unused). No true-colour (`CorrectedReflectance`) layer is referenced anywhere in pipeline or web.
- OLCI 300 m (`noaacwS3[AB]OLCIchlaSector{CI,DI}Daily`, coastwatch.noaa.gov) — Sentinel-3, unaffected.
- The "multi-sensor" layer is CoastWatch's own OLCI-over-VIIRS display (`multisensor.py`), **not** a NOAA DINEOF/gap-filled product. CoastWatch uses no DINEOF product directly (only indirectly through C-HARM).
- No S-NPP use in the research/PINN code outside `docs/` (grep of the repo root, excluding pipeline/web/docs).

### 1d. Tests, fixtures and scripts that hard-code S-NPP ids

- `pipeline/coastwatch_pipeline/fixtures.py:85` URL regex `...|erdVHNchla1day)`; `:24` note.
- `pipeline/tests/fixtures/satellite/index.json` (`erdVHNchla1day` scenes, files `*/viirs750_1day_2026-10-0*.nc`).
- `pipeline/tests/fixtures/recorded/index.json:54,58` (`erdVHNchla8day.nc?...` Monterey port requests).
- `pipeline/tests/test_satellite.py:199,214,316`, `pipeline/tests/test_multisensor.py:75` (override regex `erdVHNchla1day`).
- `pipeline/scripts/record_p1_fixtures.py:27` (`"viirs750_1day": "erdVHNchla1day"`), `pipeline/scripts/record_m2_fixtures.py:32` (`erdVHNchla8day`).
- Compat / web fixtures with recorded provenance (`pipeline/tests/fixtures/compat/m1|m2/*`, `coastwatch-web/public/data/fixture*/v1/*`, `compat-m2/v1/*`) — these are historical artifacts; leave them.

### 1e. Web code coupled to the S-NPP product's labels

- `coastwatch-web/src/lib/layers.ts:41` `SAT_PRODUCTS = ["multi","olci300","viirs750"]`; `:68` `PRODUCT_ERROR` regexes `/viirs|erdVHN/` decide "update failed" from the status error text.
- `coastwatch-web/src/lib/stamp.ts:10,18` — sensor stamp recognises a VIIRS layer only by `platforms.includes("VIIRS")` (exact string). A new platform string such as "NOAA-20" would silently drop the stamp.
- `LayerDock.tsx:356-373,496,527,659-661` (labels "VIIRS · NOAA", "VIIRS 750 m", "published about 5 days after observation"), `SatelliteNear.tsx:58`, `lib/multisensor.ts:39` (`ratioPhrase(..., secondary="VIIRS")`), `lib/palette.ts:72` + `pipeline/.../process/palette.py:58` (sensor colours), `content/copy.ts:65,242`, `verify_satellite.py:237` (sensor names).
- None of these say "S-NPP"; the only user-visible "S-NPP" text is in port-intel `method`/`provenance` (pipeline-produced).

### 1f. How provenance is recorded today (live production, fetched 07:02:11Z)

Manifest https://yashnil.github.io/habs-forecast/v1/manifest.json (HTTP 200, 428,183 B; generated_at 2026-10-13T00:42:33Z, run 38328551760):

- `viirs750_chl_latest`: `provenance.source_id="satellite_chl"` (shared with OLCI and multi-sensor!), `dataset_id="erdVHNchla1day"`, `source_url=.../griddap/erdVHNchla1day.html`, `platforms=["VIIRS"]` (no "S-NPP"/"Suomi-NPP" anywhere), `source_name="VIIRS chlorophyll-a, 750 m daily composite (NOAA CoastWatch West Coast)"`, observed_date 2026-10-06; QC checks named `erdVHNchla1day:coordinates_on_lattice|plausible_values|required_variables|time_matches_listing|within_domain`.
- `multisensor_chl_latest`: `dataset_id="noaacwS3AOLCIchlaSectorCIDaily,...,erdVHNchla1day"`, `platforms=["Sentinel-3A","VIIRS"]`.
- Source status `satellite_chl`: outcome `partial`, error "VIIRS erdVHNchla1day: time listing failed ... (HTTP Error 403: )" (pfeg 403 to the GitHub runner this run), latest_valid_date 2026-10-10 (OLCI).
- `port-intel-3f0dd056e8.json` (HTTP 200, 244,434 B): provenance `[{charm, wvcharmV3_0day..3day}, {viirs_chl_8day, erdVHNchla8day, "NOAA VIIRS S-NPP chlorophyll-a, 750 m, 8-day composite ..."}]`; every port's chlorophyll `error` = 403 / circuit open on `erdVHNchla8day` this run.
- C-HARM layers: `upstream_metadata` keeps `title, product_version, date_created, creator_*, source, history_first_line` (`charm.py:326-330`). Live value of `history_first_line`: "2026-10-11: DINEOF gap filling was applied to daily S-NPP NOAA VIIRS chlorophyll, 551nm reflectance, and 486nm reflectance fields ...". `satellite`, `platform`, `product_name`, `summary` are not kept.

Gap: today only the dataset id identifies S-NPP; platforms say "VIIRS" generically. Old artifacts are already distinguishable by `dataset_id`, which is what migration should rely on (no new manifest fields; deployed web/demo reject unknown fields).

---------------------------------------------------------------------------------------------------
## 2. C-HARM v3.1 and S-NPP

### 2a. ERDDAP metadata (fetched 2026-10-13T07:02:22–23Z, all HTTP 200)

https://coastwatch.pfeg.noaa.gov/erddap/info/wvcharmV3_0day/index.json (and `_1day`, `_2day`, `_3day`). Identical across leads except id/title/date_modified/time_coverage_end:

- `history` = "2026-10-12: DINEOF gap filling was applied to daily S-NPP NOAA VIIRS chlorophyll, 551nm reflectance, and 486nm reflectance fields extending back 180 days from today's date. / ROMS current forecasts were used to advect the NOAA VIIRS data 1, 2, and 3 days into the future. Data gaps resulting from the advection were filled with a second DINEOF."
- `satellite` = "S-NPP"; `platform` = "VIIRS"
- `product_name` = "C-HARM v3.1 model output with SNPP VIIRS and WCOFS data"; `product_version` = "3.1"
- `source` = "Satellite data, C-HARM model output, and WCOFS model output"
- `summary` = "... Inputs for the model include near real-time satellite NOAA S-National Polar-orbiting Partnership (NPP) Visible and Infrared Imager/Radiometer Suite (VIIRS) observations gap-filled chlorophyll a, 486nm reflectance, and 551nm reflectance fields from the S-NPP NOAA VIIRS sensor plus nowcast and forecast data of surface salinity, sea surface temperature, and surface currents from WCOFS ..."
- `contributor_role` = "Operationalizing, Lead PI, Co-PI, Co-PI, Source of level 2 data" (Dale Robinson; C. Edwards; R. Kudela; C. Anderson; NOAA NESDIS)
- `references` attribute: **not present** (no `references` in any lead).
- Variables include `chla_filled` (long_name "S-NPP NOAA VIIRS Chlorophyll Fields Gap-Filled with DINEOF", mg m^-3), `r486_filled`, `r551_filled` (same "S-NPP NOAA VIIRS ... Gap-Filled with DINEOF").
- `testOutOfDate = now-5days`; `time_coverage_end`: 0day 2026-10-11T12Z, 1day 10-12, 2day 10-13, 3day 10-14 (newest run issued 2026-10-12).
- `date_modified` 2022-11/12 (unchanged since v3 launch).
- Dataset list (allDatasets, 07:04:29Z): only `charmForecast*` (v1, v2, ended 2022) and `wvcharmV3_*` (+ `_LonPM180`). **No new C-HARM dataset id** (no v3.2/v4, no NOAA-20 variant).

### 2b. Public statements found

- Searched NOAA NESDIS, CoastWatch news, NASA Earthdata, SCCOOS, CeNCOOS (2026-10-13). **No statement found** about C-HARM moving to NOAA-20/NOAA-21 VIIRS, or about the S-NPP end affecting C-HARM. The SCCOOS bulletins found are older (v3 launch: WCOFS + "VIIRS ocean color products in place of MODIS Aqua"; VIIRS "pixelation" issue) — e.g. https://sccoos.org/?p=15711, https://sccoos.org/?p=15609 (dates not re-checked; UNVERIFIED whether any later bulletin exists).
- Note the C-HARM chain: STAR S-NPP NRT L2 → CoastWatch West Coast (`dalesLibViirsChl` per erdVHN history, same operator Dale Robinson) → DINEOF 180-day → C-HARM. The same West Coast node already ingests NOAA-20/21 elsewhere, so a switch is technically plausible, but that is inference, not a statement.

### 2c. Known vs unknown

Known: C-HARM v3.1 ocean-colour inputs today are S-NPP VIIRS only (chl, Rrs486, Rrs551), DINEOF-filled over 180 days; S-NPP VIIRS delivery stops 2026-11-02 13:00 UTC; no announced C-HARM change.
Unknown: whether NOAA/SWFSC/UCSC will (a) switch to NOAA-20 or NOAA-21 VIIRS, (b) keep running on a shrinking S-NPP window (DINEOF over 180 days could keep producing fields for a while from old data — that would be silently degraded), (c) pause C-HARM, (d) bump product_version or publish new dataset ids, (e) re-tune the model for NOAA-20/21 radiometric differences (NOAA-20 reads ~10–20 % lower than S-NPP over California in my two-day check, §3d — the empirical C-HARM models use chl and Rrs as predictors, so an un-recalibrated swap changes the probabilities). **CoastWatch cannot fix any of this**; it can only detect and label it.

### 2d. Signals CoastWatch can monitor (no new manifest fields; use `upstream_metadata` dict and health.json)

1. `history` first line: platform word ("S-NPP" → "NOAA-20"/"NOAA-21"/"JPSS") and its date stamp vs run date. Already recorded as `history_first_line`; add `satellite`, `platform`, `product_name`, `summary` (first 400 chars), `date_modified` to the kept keys at `charm.py:326` (it's an existing free-form dict) and alert on any change.
2. `product_version` != "3.1" → today a generic validation failure (`charm.py:228-229`); make it a distinct `charm_version_changed` health alert instead of a silent failed run.
3. Variable `long_name`s of `chla_filled`/`r486_filled`/`r551_filled` (would change platform wording).
4. Time gaps: no new nowcast for > 2 Pacific days while ERDDAP answers (time[last] stuck) → "paused upstream", not "no risk".
5. Stale-input heuristic after 2026-11-02: history line date advancing but the ocean-colour inputs not (needs `chla_filled` change check: e.g. day-to-day identical `chla_filled` field or valid-fraction/variance drop vs 14-day median).
6. New dataset ids: daily ERDDAP search `search/index.csv?searchFor=C-HARM` and `allDatasets` filter `charm`/`wvcharm` on pfeg; alert on any id not in the known list.
7. CoastWatch news page (https://coastwatch.noaa.gov/cwn/news.html) and SCCOOS/CeNCOOS HAB pages — human watch, weekly, until the outcome is known.

---------------------------------------------------------------------------------------------------
## 3. Replacement candidates (live checks 2026-10-13)

Search: ERDDAP `search/index.csv` on the four servers for "VIIRS chlorophyll", "NOAA-20 chlorophyll", "NOAA-21 chlorophyll", "N20 chlorophyll", "VIIRS chla", "N20VIIRS", "N21VIIRS", "NOAA-21", "multisensor", "VIIRS merged chlorophyll", "N20 chla", "JPSS" (07:03–07:04Z) plus `tabledap/allDatasets.csv` of each server (07:04:29Z: coastwatch.noaa.gov 176,554 B; pfeg 521,823 B; cwcgom 10,779 B; oceanwatch.pifsc 25,900 B).
Result: **NOAA-21 chlorophyll exists only as `noaacwN21VIIRSchlaSectorUSDaily` on coastwatch.noaa.gov.** cwcgom.aoml and oceanwatch.pifsc have **no** NOAA-20/21 chlorophyll (only S-NPP or deprecated). No NOAA-20/21 750 m product on pfeg. No `nppVIIRS`/`nesdisVHNnoaa21` ids exist.

### 3a. Candidate records

C1 — **`noaacwN20VIIRSchlaSectorUSDaily`** — https://coastwatch.noaa.gov/erddap/griddap/noaacwN20VIIRSchlaSectorUSDaily
- Title "Daily Merge chlora from NOAA-20 VIIRS"; source "NOAA-20_VIIRS_chlora"; institution NOAA NESDIS STAR; creator Veronica Lance.
- Processing: NOAA MSL12 NRT ("NRTNIR", per history `V2026278_D_N20_NRTNIR_US00_edgemask_chlora.hdf`), cwutils `cwcomposite --method explicit` daily merge of passes (L3, mapped). `chlor_a` long_name "Chlorophyll Concentration, **OC3 Algorithm**".
- Variables: `chlor_a` (mg m^-3, _FillValue -999, no valid_min/max attrs), `l2_flags` (int, no flag_meanings attr), `qa_score` (0–1), `swath_latitude`, `swath_longitude`, `graphics`.
- Grid: geographic, 0.0075° (~830 m N–S, ~670 m E–W at 36°N; "750 m" nominal); lat 0.00375–59.99625 (descending), lon −142.38375 to −52.39125; 8000×12000. **Same lattice as `erdVHNchla1day`** (verified: both return cell 42.49875, −127.00125 at stride 8).
- Time: one step per day, stamped at the merge centre time (e.g. 2026-10-04T19:02:17Z), not 12:00Z. **Rolling archive**: start 2026-02-01T19:34:24Z, 238 steps. Latest observed: **2026-10-05T18:38:24Z** (time axis 07:05:51Z); `date_created` 2026-10-06T05:39Z; `testOutOfDate now-15days`. Today that is ~7.5 d behind (it was ~2.9 d at the 2026-10-09 verification in docs/coastwatch/02-data-sources.md) — the NOAA 750 m sector pipeline appears to be lagging right now (erdVHN is also stuck at 10-06), while the S-NPP central sector `noaacwNPPVIIRSchlaSectorVYDaily` reached 2026-10-10T19:12Z. Latency is therefore 3–8 d and variable.
- NRT, not science quality.
- California subset: `chlor_a[(2026-10-04T12:00:00Z)][(0.0)][(42.5):8:(32.0)][(-127.0):8:(-116.0)]` .csv0 → **HTTP 200, 1,965,928 B** (07:06:14Z); 10-03 → 200, 1,954,001 B (07:06:26Z). Monterey port-style .nc request (stride 2, 10-01..last) → 200, 20,404 B (07:09:49Z).
- Flags: valid pixels still carry flags (Monterey box 10-04: valid with l2_flags 2048, 2432, 256, 2304, 2496, 655748; qa_score 0.2–1.0, median 0.8). Masked pixels are already NaN (most with fill flag). Bit meanings are not in the metadata — **UNVERIFIED**; need NOAA STAR MSL12 flag table before using flags.

C2 — **`noaacwN21VIIRSchlaSectorUSDaily`** — https://coastwatch.noaa.gov/erddap/griddap/noaacwN21VIIRSchlaSectorUSDaily
- Same structure/grid/variables as C1; source "NOAA-21_VIIRS_chlora"; history `.../viirs/n21/nrt/L3C/... V2026278_D_N21_NRTNIR_US00_edgemask_chlora.hdf`.
- `chlor_a` long_name "Chlorophyll Concentration, **OCI Algorithm**" — differs from C1's "OC3". Whether the algorithms really differ or one label is wrong is **UNVERIFIED**; ask the CoastWatch help desk (coastwatch.info@noaa.gov) — the user decides whether to contact anyone.
- Rolling archive from 2026-02-09T19:34:30Z, 231 steps; latest **2026-10-05T18:36:59Z**; date_created 2026-10-06T07:30Z.
- California subset 10-04 → **HTTP 200, 1,958,191 B** (07:06:15Z); 10-03 → 200, 1,949,740 B. Monterey flags box → 200, 235,874 B; valid pixels incl. flags 131072/131328 and qa_score down to 0.0 (median 0.6).
- NOAA names NOAA-21 the **primary** JPSS replacement.

C3 — `nesdisVHNnoaa20chlaDaily` (pfeg) = mirror of `noaacwN20VIIRSchlaDaily` (coastwatch.noaa.gov)
- Title "Chlorophyll, NOAA NOAA-20 VIIRS, Near Real-Time, Global 4km, Level 3, 2017-present, Daily"; platform NOAA-20; `chlor_a` "OC3 Algorithm" (O'Reilly 1998), mg m^-3, valid 0.001–1000, NaN fill; 0.0375° (~4 km), lat descending; summary still says "Suomi-NPP ... EXPERIMENTAL" (template text).
- Coverage 2021-08-26 → latest **2026-10-10T12:00Z** (07:05:52Z), daily 12:00Z stamps, ~3 d latency.
- California subset (stride 2) 10-04 → **HTTP 200, 1,256,480 B** (07:06:20Z); 10-03 → 200, 1,250,362 B.
- No flags variable. S-NPP twin `nesdisVHNchlaDaily` latest 2026-10-11 (rolling from 2025-10-07).
- No NOAA-21 4 km equivalent found.

C4 — `nesdisVHNnoaa20chlaWeekly` (pfeg) / `noaacwN20VIIRSchlaWeekly` (central)
- 4 km; summary "Geometric Mean Baseline of Chlorophyll ... Weekly"; time axis is a daily-stepped rolling window (2026-09-01..10-04 all present, 12:00Z); latest **2026-10-04**.
- Multi-time port-style .nc request → **HTTP 502** on pfeg twice (07:09:38Z, 07:10:04Z) and on central (07:10:22Z); single-time csv0 → 200, 5,401 B (07:10:32Z). Fragile for a 60-day history request; not recommended.

C5 — Gap-filled DINEOF: `noaacwNPPN20VIIRSDINEOFDaily` (central) = `nesdisVHNnoaaSNPPnoaa20NRTchlaGapfilledDaily` (pfeg)
- "Chlorophyll (Gap-filled DINEOF), NOAA S-NPP NOAA-20, VIIRS, NRT, Global 9km", platform "Suomi-NPP, NOAA-20", MSL12 → DINEOF, EXPERIMENTAL; `chlor_a` mg m^-3 valid 0.001–100; 0.0833° (~9 km); 2020-05-05 → latest **2026-10-11T12:00Z**.
- California subset 10-04 (stride 1) → **HTTP 200, 902,045 B** (07:06:24Z): 52 % of cells valid (vs ~20 % for daily L3).
- **Itself uses S-NPP**; what it becomes after 2026-11-02 is unknown. Science-quality 2 km DINEOF (`noaacwNPPN20S3ASCIDINEOF2kmDaily`, S-NPP+NOAA-20+S3A OLCI) ends 2026-09-27 (~2-week lag). Neither is a like-for-like replacement for an observation layer (CoastWatch rules: no gap-filling shown as observation). Not recommended for CoastWatch use beyond optional, clearly labelled context.

C6 — GIBS: `VIIRS_NOAA21_Chlorophyll_a` (to 2026-10-12) as fallback for the existing NOAA-20 GIBS layer. GIBS is visual only and not S-NPP today.

### 3b. Comparison with the current S-NPP product (`erdVHNchla1day`)

| Aspect | erdVHNchla1day (S-NPP) | C1 N20 US sector | C2 N21 US sector | C3 N20 4 km |
|---|---|---|---|---|
| Variable | `chla` | `chlor_a` | `chlor_a` | `chlor_a` |
| Algorithm label | "NOAA-MSL12 NRT" (algorithm not named) | OC3 | OCI | OC3 |
| Units | mg m^-3 | mg m^-3 | mg m^-3 | mg m^-3 |
| Valid range attr | 0.001–100.1 | none | none | 0.001–1000 |
| Fill | -999 | -999 | -999 | NaN |
| Grid | 0.0075°, same lattice | identical lattice | identical lattice | 0.0375° |
| Masks | pre-masked (LAND+CLDICE+HIGLINT+HISATZEN+HISOLZEN) | pre-masked edgemask + `l2_flags`, `qa_score` | same | pre-masked, no flags |
| Time stamp | 12:00Z per day | merge-centre time (~19Z) | ~19Z | 12:00Z |
| Archive | 2015-02-25 → | rolling from 2026-02-01 | rolling from 2026-02-09 | 2021-08-26 → |
| Latest seen | 2026-10-06 | 2026-10-05 | 2026-10-05 | 2026-10-10 |
| Server | pfeg | central | central | pfeg (mirror on central) |

California check (lat 32–42.5, lon −127 to −116), same-day, same cells:

| Day | Pair | n paired | median ratio (b/a) | MAD log10 | r(log10) | coverage a / b |
|---|---|---|---|---|---|---|
| 10-04 | N20 750 / S-NPP 750 | 1,995 | **0.82** | 0.026 | 0.978 | 20.1 % / 19.9 % |
| 10-04 | N21 750 / S-NPP 750 | 1,687 | **0.92** | 0.020 | 0.964 | 20.1 % / 16.5 % |
| 10-04 | N21 / N20 (750) | 4,915 | 1.11 | 0.022 | 0.967 | — |
| 10-04 | N20 4 km / S-NPP 4 km | 4,473 | 0.83 | 0.023 | 0.980 | 22.5 % / 23.9 % |
| 10-03 | N20 750 / S-NPP 750 | 1,470 | **0.89** | 0.043 | 0.980 | 17.0 % / 14.8 % |
| 10-03 | N21 750 / S-NPP 750 | 1,303 | **0.93** | 0.030 | 0.981 | 17.0 % / 12.9 % |
| 10-03 | N21 / N20 (750) | 3,399 | 1.07 | 0.034 | 0.981 | — |
| 10-03 | N20 4 km / S-NPP 4 km | 3,860 | 0.89 | 0.038 | 0.974 | 21.4 % / 20.0 % |

(750 m sampled every 8th cell ≈ 6.7 km spacing; 4 km every 2nd cell. Two days only; this is a smoke test, not a validation.)
Reading: patterns agree closely (r ≈ 0.97–0.98), but NOAA-20 reads ~11–18 % lower and NOAA-21 ~7–8 % lower than S-NPP; the N20/N21 sector pixels overlap the S-NPP composite on only ~⅓ of S-NPP's clear cells (different overpass times/clouds), while N20 ∪ N21 has more coverage than either alone (N20 only 1,534 + N21 only 434 + both 4,915 on 10-04). A colour-scale step of 10–20 % is visible on a log palette near class edges and would make the multi-sensor "Sentinel-3 vs VIIRS" agreement statistic shift. **Do not splice S-NPP history and NOAA-20/21 values in one series without saying so.**

Time continuity: overlap of S-NPP and N20/N21 750 m exists from 2026-02-01/09 to 2026-11-02 (~9 months) — but the N20/N21 sector archive is **rolling**, so pull the overlap period now if a cross-calibration record is wanted.

---------------------------------------------------------------------------------------------------
## 4. Recommendation

### 4a. Per use

| Use | Primary | Fallback | Notes |
|---|---|---|---|
| VIIRS 750 m map layer `viirs750_*` + multi-sensor secondary + SatelliteNear fallback | **NOAA-21 + NOAA-20 US-sector 750 m** (`noaacwN21VIIRSchlaSectorUSDaily` first, `noaacwN20VIIRSchlaSectorUSDaily` filling gaps), coastwatch.noaa.gov, `chlor_a` | Until 2026-11-02: keep `erdVHNchla1day` as a separately labelled fallback; after: `nesdisVHNnoaa20chlaDaily` 4 km on pfeg (different server, a real fallback) shown as its own product/legend label "NOAA-20 VIIRS 4 km", never merged pixel-wise with 750 m | Same lattice and native res → minimal code change. NOAA-21 primary follows NOAA guidance; but given the OC3/OCI label mismatch and a 7 % N21–N20 offset, a reasonable alternative is NOAA-20 first (C-HARM lineage/West Coast precedent). Decision for the user. Per-pixel: one platform per pixel (like OLCI S3A/S3B), never averaged. |
| Port/My Coast chlorophyll (8-day, 15 km median) | CoastWatch-computed **8-day window median of the daily NOAA-21/NOAA-20 750 m pixels** within 15 km (same pixels as the map; label "8-day window of NOAA-20/21 VIIRS daily observations, median of clear pixels") | `nesdisVHNnoaa20chlaDaily` 4 km over the same window | No NOAA-20/21 750 m 8-day composite exists on any server checked; the weekly 4 km product 502'd on multi-time requests. 60-day history fits inside the rolling archive. Keep erdVHNchla8day history points that are already published, marked S-NPP. |
| GIBS visual layer | keep `VIIRS_NOAA20_Chlorophyll_a` | add `VIIRS_NOAA21_Chlorophyll_a` as second choice | No S-NPP dependency today. |
| C-HARM | no replacement possible (NOAA product) | — | Monitor (§2d); label honestly. |

### 4b. Code/config changes (file:line at c638f274) — proposals only

Pipeline
1. `sources/satellite.py:154-168` — new Product, e.g. key stays `viirs750` (keeps the web selector working) or a new key `jpss750` (cleaner provenance; needs web changes — product decision). Fields: `server="https://coastwatch.noaa.gov/erddap"`, `platforms=(("NOAA-21", (("noaacwN21VIIRSchlaSectorUSDaily", -180, 180, 90),)), ("NOAA-20", (("noaacwN20VIIRSchlaSectorUSDaily", -180, 180, 90),)))`, `variable="chlor_a"`, `step=0.0075`, `source_name="NOAA-20/NOAA-21 VIIRS chlorophyll-a, 750 m daily merge (NOAA CoastWatch/NESDIS STAR)"`, `sensor_text="VIIRS on NOAA-21, with NOAA-20 filling gaps"`, license text from the dataset. Keep the existing S-NPP Product as a separate fallback until cut-off, with platform "S-NPP".
   - `:9-10, :83` doc/freshness note (latency 3–8 d); `:76` licence.
   - Time handling: per-day selection must use the overpass-time stamps (as OLCI already does, `run_product` per-scene logic `:593-649`); check `time_matches_listing` and the UTC-date → observation date mapping (19Z ≈ local noon, same date).
   - Fill/valid: -999 handled by `_FillValue` path (`:280-283`); add explicit upper bound check (no valid_max attr) — `IMPLAUSIBLE_MG_M3=200` QC already flags.
   - Flags: start with upstream masks as published (CoastWatch rule "masks from upstream preserved"); optionally later request `l2_flags`/`qa_score` once STAR's bit table is confirmed. Do not invent a qa threshold without documentation.
   - `:674-708` run order/notes ("OLCI unavailable: the VIIRS ... fallback") — wording to "NOAA-20/21 VIIRS".
2. `pipeline.py:208-209` source title → "Sentinel-3 OLCI 300 m (NOAA-20/21 VIIRS 750 m fallback)". Consider giving the S-NPP fallback its own error prefix so `layers.ts:68` still matches.
3. `sources/multisensor.py:229,242,250-251` — titles/labels from the member's platform ("VIIRS 750 m (NOAA-21/NOAA-20)"); bump `RULE_VERSION` so the multi-sensor hash changes; agreement text `:55-64` unchanged in method.
4. `sources/port_intel.py:8-9,48-49,171-215,326,341-349` — replace `erdVHNchla8day` fetch with the 8-day median of daily NOAA-20/21 750 m (or reuse the satellite source's fetched/published day grids to avoid extra requests); new provenance `source_id="viirs_jpss_chl_8day"` (new id), `dataset_id="noaacwN21VIIRSchlaSectorUSDaily,noaacwN20VIIRSchlaSectorUSDaily"`; method text says CoastWatch computed the window median. Keep `composite_days=8` semantics clear ("8-day window" rather than "8-day composite").
5. `verify_satellite.py:92,128` — `prod` is chosen by layer-id prefix; must select by the layer's own `provenance.dataset_id` so the verifier compares against the source actually used (S-NPP layers vs NOAA-20/21 layers).
6. `health.py:11-13,38,78` — VIIRS_MAX_AGE_DAYS=8 will trip with today's 7.5 d lag; set from observed latency (e.g. 10) and name the label "NOAA-20/21 VIIRS 750 m". Add a `snpp_ended` informational state after 2026-11-02 so a dead S-NPP fallback is not an alarm.
7. `sources/charm.py:326` — extend kept metadata keys (`satellite`, `platform`, `product_name`, `summary`, `date_modified`); `:228-229` version check → distinct alert (see data-health.md §improvement 3, which proposes the same).
8. `sources/gibs.py:57-72` — add `VIIRS_NOAA21_Chlorophyll_a` as a fallback GibsLayer (separate artifact id).
9. Fixtures/tests: `fixtures.py:85` regex, new recorded scenes (`pipeline/scripts/record_p1_fixtures.py:27`, `record_m2_fixtures.py:32`), `tests/fixtures/satellite/index.json`, `recorded/index.json`, tests `test_satellite.py:199,214,316`, `test_multisensor.py:75`. Add tests: platform order (N21 then N20 per pixel), overpass-time stamps, lattice equality with erdVHN, fill -999, no valid_max, S-NPP fallback labelled S-NPP, port 8-day median window boundaries.

Web
10. `lib/stamp.ts:10,18` — recognise "NOAA-20"/"NOAA-21"/"S-NPP" platform strings (today exact `"VIIRS"`), e.g. "VIIRS (NOAA-21) 750 m".
11. `lib/layers.ts:68` — error regexes include `noaacwN2[01]VIIRS|NOAA-2[01]`.
12. Copy: `LayerDock.tsx:373,496,527,661` ("VIIRS · NOAA" → "VIIRS · NOAA-20/21"; latency text), `SatelliteNear.tsx:58`, `content/copy.ts:65` (port text: "NOAA-20/21 VIIRS daily observations, 8-day window" and, for older points, "S-NPP"), `copy.ts:242` sources list, `PortPanel.tsx:183`.
13. Schema: **no new manifest fields** (demo + deployed web reject unknown fields). Use existing `platforms`, `provenance.dataset_id`, `source_name`, `upstream_metadata`.

### 4c. Validation plan (before switching)

1. Overlap pull now (rolling archive!): for every day 2026-08-01 → cut-over, fetch S-NPP `erdVHNchla1day`, N20 and N21 US-sector over the California coastal domain (`satellite.CALIFORNIA` + coastal mask). Same lattice → exact cell pairing; time window per pair ≤ the same UTC day.
2. Metrics per day and pooled, by region (north coast, central, SoCal Bight) and by distance-to-shore band (0–5, 5–20, 20–100 km) and chl class: n pairs, median log10 ratio (bias), MAD, RMSE in log10, r, Type-II slope, coverage (% coastal cells valid) for each and for the N21∪N20 union, fraction > 200 mg m^-3, fraction ≤ 0.
3. Accept if: |median log10 bias| documented (expect −0.03…−0.09) and stable over time; r(log) ≥ 0.9; union coverage ≥ S-NPP coverage on ≥ 80 % of days; no lattice offsets; implausible fraction ≤ 1 %. Do **not** bias-correct to S-NPP (CoastWatch publishes upstream values); instead state "NOAA-20 typically reads ~10–20 % lower than S-NPP here" in the method text if confirmed.
4. Units/range/flags: assert `units == "mg m^-3"`, fill = -999, values >0; record `l2_flags` histograms among valid pixels; get the bit table from STAR docs before any flag use.
5. Algorithm label check: confirm whether N20 (OC3) vs N21 (OCI) labels are real (look for STAR product documentation; contacting the help desk is the user's call).
6. Multi-sensor: recompute the Sentinel-3 vs VIIRS agreement on the overlap with S-NPP and with N21/N20 and record the shift, so the displayed "Sentinel-3 read X % lower than VIIRS" does not jump unexplained.
7. Port 8-day: compare old erdVHNchla8day port medians with the new window medians for all ports over 60 days; document ratio.

### 4d. Keeping provenance intact

- Existing published artifacts are immutable: they keep `dataset_id="erdVHNchla1day"`/`"erdVHNchla8day"`, `source_id="satellite_chl"`/`"viirs_chl_8day"`. Never rewrite them.
- New artifacts: new dataset ids in `provenance.dataset_id`, `platforms=["NOAA-21","NOAA-20"]` (actual per-day platforms used), new `source_name`; port intel new `source_id` (`viirs_jpss_chl_8day`). Consider a new layer key (`jpss750_*`) so per-day layers from the two lineages never share an id; if `viirs750` is kept for UI stability, the date-keyed day layers will still differ by provenance — make `_prev_layers` reuse (satellite PROCESSING_VERSION `satellite.py:33`) bump to `satellite-processing-3` so S-NPP days are not reused as if they were NOAA-20/21.
- UI: sensor stamp from `platforms` ("VIIRS · S-NPP" for old, "VIIRS · NOAA-21" for new); in the port history chart, mark the change point ("before 2026-11-02: S-NPP; after: NOAA-20/21, which reads ~x % lower").

### 4e. Staging steps (do not touch `coastwatch-data` prod; demo deploy only with explicit approval)

1. Feature branch + worktree; implement 4b; unit tests with re-recorded fixtures for N20/N21 (recorded once, offline afterwards).
2. Run the pipeline with `--only satellite_chl,port_intel` to the `staging` channel; run `verify-satellite` live (expect pfeg/central 403/502 episodes; verify-time upstream errors must be "inconclusive", not mismatches).
3. Compare staging manifest vs production: layer ids, `dataset_id`s, platforms, coverage, freshness; load staging in the private preview; check stamps, readout, multi-sensor seam text, port panel.
4. Simulate the cut-off: fixture override making `erdVHNchla*` return no new times / 404 "no matching results" after a date, and N20 or N21 failing individually; confirm messages (4f).
5. Run the overlap validation (4c) and attach the numbers to the PR.
6. Merge before ~2026-10-26 so a week of NOAA-20/21 runs exist before 2026-11-02; keep the S-NPP fallback until it goes stale, then remove it in a follow-up.

### 4f. What users should see on/after 2026-11-02

- VIIRS layer: if NOAA-20/21 is live, normal; the stamp shows the satellite. If a product stops: "No newer VIIRS observation since <date>. NOAA ended Suomi-NPP data on 2 Nov 2026; this layer now uses NOAA-20/21." Gaps remain "no observation", never "low chlorophyll" or "no bloom".
- S-NPP fallback after its last pixel: shown as historical with its own observation dates ("S-NPP, last observation <date>"), not as current.
- Port panel: history before the change labelled S-NPP; after, NOAA-20/21; if nothing new: "No clear satellite view near this port since <date>" (never "0" or "low").
- C-HARM: if runs stop: "C-HARM has not published a new forecast since <date> (NOAA). The last forecast is shown as historical." If metadata shows a platform/version change: "NOAA changed C-HARM's satellite input on <date> (now <platform>/<version>); forecasts before and after may not be directly comparable." If runs continue but `history` still says S-NPP well after 2026-11-02: "C-HARM's satellite input may be ageing (it still reports S-NPP)." Never "no risk" / "no bloom" from a missing or stale run; chlorophyll is never presented as toxin; observations are never presented as forecasts.

---------------------------------------------------------------------------------------------------
## 5. Open items / unverified

- OC3 (N20) vs OCI (N21) `chlor_a` labels — unverified which is correct.
- `l2_flags` bit meanings for these NESDIS L3 merges — not in metadata.
- Cause of the current ~7.5 d lag of the 750 m NOAA sector products (both N20/N21 and S-NPP erdVHN) — unknown; earlier verification saw ~2.9 d.
- Whether the N20/N21 US-sector archive stays rolling (start dates suggest ~8 months kept) — unverified.
- C-HARM's post-S-NPP plan — no public statement found (searched NOAA NESDIS, CoastWatch news, NASA Earthdata, SCCOOS, CeNCOOS on 2026-10-13).
- Comparison statistics rest on two days (2026-10-03/04) of strided samples; treat as indicative only.
