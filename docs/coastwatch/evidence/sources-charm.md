> **Evidence log.** Raw verification notes from live requests on 2026-10-08 (UTC times inside). Summarized in [`../02-data-sources.md`](../02-data-sources.md). Re-verify before relying on any value; sources change.

# C-HARM and California HAB observation sources: verification report

- **Checked:** 2026-10-08, 16:27–16:35 UTC. Every value in this report comes from a live request made in that window, unless it is marked otherwise.
- **Method:** ERDDAP queries used `curl`. Human-facing pages were read with WebFetch, so their content is a model summary of each page. Treat page quotes as "per page summary" and check them by hand before publishing them.
- **ERDDAP reliability:** CoastWatch ERDDAP returned intermittent **HTTP 503** errors during the session, on info/charmForecast3dayV2, the "harm" search, and one WMS GetMap. A retry 20–60 s later succeeded each time. Clients need retry with backoff.

---

## 1. C-HARM on NOAA CoastWatch West Coast ERDDAP

### 1.1 Discovery

These searches all returned the same C-HARM family:

- `https://coastwatch.pfeg.noaa.gov/erddap/search/index.json?searchFor=charm&page=1&itemsPerPage=100`
- `...searchFor=domoic...`
- `...searchFor=pseudo-nitzschia...`
- `...searchFor=harm...`. The first attempt returned 503. The retry added only the OSU MODIS chl/SST datasets, which are not C-HARM.

There are 23 C-HARM datasets, in three generations, each with a 0–360 longitude grid and a `_LonPM180` copy.

| Generation | Dataset IDs (0–360 lon) | `_LonPM180` copies | Coverage (time_coverage_start → end) | Status |
|---|---|---|---|---|
| v1 (product_version 1.0) | charmForecast0day, charmForecast1day, charmForecast2day, charmForecast3day | all four (0,1,2,3day) | 2018-06-19 → 2022-03-17 (0day); 1/2/3day end 2022-03-18/19/20 | **Frozen / historical** |
| v2 (2.0) | charmForecast0dayV2, charmForecast1dayV2, charmForecast2dayV2, charmForecast3dayV2 | all four | 2021-01-01 → 2022-11-30 (0day); 1/2/3day end 2022-12-01/02/03 | **Frozen.** The title still says "2022-present", which is misleading. |
| **v3.1 (3.1)** | **wvcharmV3_0day, wvcharmV3_1day, wvcharmV3_2day, wvcharmV3_3day** | **1day, 2day, 3day only.** `wvcharmV3_0day_LonPM180` returns **HTTP 404**. | 2022-11-01 → **2026-10-07** (0day); 1day → 2026-10-08; 2day → 2026-10-09; 3day → 2026-10-10 | **LIVE, updated today.** NC_GLOBAL `history` begins "2026-10-08: DINEOF gap filling was applied…" |

**Latest version:** C-HARM **v3.1** (`product_version = 3.1`). The internal `id` attributes are charmForecast0dayV3, charmForecast1dayV3, and so on, but the ERDDAP dataset IDs are `wvcharmV3_*`.

### 1.2 v3.1 dataset tables

Fields shared by all v3.1 datasets, read from `/erddap/info/<id>/index.json`:

| Field | Value |
|---|---|
| Endpoint | `https://coastwatch.pfeg.noaa.gov/erddap/griddap/<id>` |
| Resolution | 0.03° lat × 0.03° lon (~3 km); 391 lat × 351 lon |
| Bounds | lat 31.3–43.0; lon 232.5–243.0 (0–360), or −127.5 to −117.0 for `_LonPM180` |
| Variables | `pseudo_nitzschia`: "Probability of Pseudo-nitzschia > 10,000 cells/L", units "1"<br>`particulate_domoic`: "Probability of Particulate Domoic Acid > 500 nanograms/L", units "1"<br>`cellular_domoic`: "Probability of Cellular Domoic Acid > 10 picograms/cell", units "1"<br>`chla_filled`: VIIRS chl gap-filled with DINEOF, mg m^-3<br>`r486_filled`, `r551_filled`: VIIRS reflectances, gap-filled<br>`salinity`: WCOFS surface salinity, PSU<br>`water_temparture` [sic]: WCOFS surface temperature, degree_C |
| Fill value | `_FillValue = missing_value = -99999.0`. ERDDAP returns this as NaN in CSV. |
| Time axis | `long_name` "Centered Nowcast Time" (0day) or "Centered Forecast Time" (1–3day). `comment` "The day represented by the nowcasts/forecasts". Stamped 12:00Z. |
| Cadence | `time_coverage_resolution = PD1` (daily), but there are many missing days (see §1.4) |
| Inputs (per `summary`/`history`) | NOAA S-NPP VIIRS chl, Rrs486, and Rrs551, each DINEOF gap-filled over a 180-day window. Surface T and S come from **WCOFS**. ROMS (WCOFS) currents advect the satellite fields 1–3 days ahead, and a second DINEOF fills the gaps the advection creates. |
| License | "The data may be used and redistributed for free but is not intended for legal use, since it may contain inaccuracies. Neither the data Contributor, CoastWatch, NOAA, nor the United States Government … assumes any legal liability…" |
| Institution | NOAA/NMFS/SWFSC/ERD, CoastWatch West Coast. Publisher: NOAA NMFS SWFSC ERD, erd.data@noaa.gov |
| Creator | Clarissa Anderson, cra002@ucsd.edu, creator_url https://www.cencoos.org/observations/models-forecasts |
| Contributors | Dale Robinson, Christopher Edwards, Raphe Kudela, Clarissa Anderson, NOAA NESDIS |
| Project | "Advancing the West Coast Ocean Forecasting System through Assessment, Model Development, and Ecological Products" |
| Access constraints | None (public, no key). Server has intermittent 503s. |
| Metadata dates | date_created/issued/modified = 2022-12-07T17:56:54Z. This is static and does not reflect the latest update. |

Per-dataset values:

| Dataset ID | Lead | time_coverage_start | Last time step (verified) | nValues (time) | WMS | testOutOfDate | Verified |
|---|---|---|---|---|---|---|---|
| wvcharmV3_0day | Nowcast | 2022-11-01T12Z | **2026-10-07T12:00Z** | 1151 | **No.** "This dataset is never available via WMS" (404), because of the 0–360 longitudes. | now-5days | YES, 2026-10-08 16:28Z |
| wvcharmV3_1day | +1 d | 2022-11-02T12Z | **2026-10-08T12:00Z** | 1158 | No (0–360) | now-5days | YES |
| wvcharmV3_2day | +2 d | 2022-11-03T12Z | **2026-10-09T12:00Z** | 1158 | No (0–360) | now-5days | YES |
| wvcharmV3_3day | +3 d | 2022-11-04T12Z | **2026-10-10T12:00Z** | 1158 | No (0–360) | now+5day | YES |
| wvcharmV3_1day_LonPM180 | +1 d | 2022-11-02T12Z | 2026-10-08T12:00Z | 1158 | **Yes** | – | YES; GetMap returned a PNG |
| wvcharmV3_2day_LonPM180 | +2 d | – | 2026-10-09T12:00Z | – | Yes (listed in search) | – | info YES; GetMap not tested |
| wvcharmV3_3day_LonPM180 | +3 d | – | 2026-10-10T12:00Z | – | Yes (listed in search) | – | info YES; GetMap not tested |
| wvcharmV3_0day_LonPM180 | – | – | – | – | – | – | **Does not exist (HTTP 404)** |

Stale v1/v2 datasets (do not use for current risk):

| Dataset ID | Last time | Variables (differences) |
|---|---|---|
| charmForecast0day / 1day / 2day / 3day (v1) | 2022-03-17 / 18 / 19 / 20 | pn, pDA, cDA, chla_filled, r555_filled, r488_filled (MODIS bands) |
| charmForecast0dayV2 / 1dayV2 / 2dayV2 / 3dayV2 (v2) | 2022-11-30 / 12-01 / 12-02 / 12-03 | pn, cDA, pDA, chla_filled, rrs486_filled, rrs551_filled |

### 1.3 Time-axis semantics (inferred from metadata and the time axes)

- `time` is the **valid day**, the day the product represents, at 12:00Z. It is **not** the issue time.
- One daily run produces all four leads. Today's run has `history` dated 2026-10-08 and produced:
  - nowcast valid 2026-10-07 (the day before the run),
  - 1-day forecast valid 2026-10-08,
  - 2-day forecast valid 2026-10-09,
  - 3-day forecast valid 2026-10-10.
- So **issue date ≈ nowcast valid date + 1**, and lead *k* means valid = nowcast day + *k*.
- The missing-run pattern agrees with this. The nowcast axis goes 10-04 → 10-07, and the 1-day axis goes 10-05 → 10-08. Runs that would have produced nowcasts for 10-05 and 10-06 are absent.
- No issue-time or forecast-reference-time variable is exposed. Derive it as described above. This is an inference, not documented metadata.

### 1.4 Freshness and gaps

- Today's run (2026-10-08) is present. Latest nowcast is 2026-10-07; latest forecast reaches 2026-10-10.
- The nowcast time axis has 1151 steps over about 1437 days, with **146 gaps longer than 1 day**.
- Largest gaps:
  - 2024-09-08 → 2024-11-01 (54 d)
  - 2023-11-12 → 11-27 (15 d)
  - 2024-08-06 → 08-18 (12 d)
- In 2026 there have been 236 nowcasts so far, with frequent 2–6-day gaps, for example 2026-07-10 → 07-16 and 2026-10-04 → 10-07.
- The app should show "as of <valid date>" and handle missing days.

Request:

- `https://coastwatch.pfeg.noaa.gov/erddap/griddap/wvcharmV3_0day.csv0?time`
- `https://coastwatch.pfeg.noaa.gov/erddap/griddap/wvcharmV3_1day.csv0?time`

### 1.5 Real subset: Monterey Bay (lat 36.6–36.9, lon −122.2 to −121.8 = 237.8–238.2 in 0–360)

Requests (all HTTP 200). The URLs for the 1day, 2day, and 3day datasets are the same apart from the dataset ID:

```
https://coastwatch.pfeg.noaa.gov/erddap/griddap/wvcharmV3_0day.csv?pseudo_nitzschia%5B(last)%5D%5B(36.6):(36.9)%5D%5B(237.8):(238.2)%5D,particulate_domoic%5B(last)%5D%5B(36.6):(36.9)%5D%5B(237.8):(238.2)%5D,cellular_domoic%5B(last)%5D%5B(36.6):(36.9)%5D%5B(237.8):(238.2)%5D,chla_filled%5B(last)%5D%5B(36.6):(36.9)%5D%5B(237.8):(238.2)%5D
https://coastwatch.pfeg.noaa.gov/erddap/griddap/wvcharmV3_1day_LonPM180.nc?particulate_domoic%5B(last)%5D%5B(36.6):(36.9)%5D%5B(-122.2):(-121.8)%5D   (200, 10 KB NetCDF)
```

The subset is 11 lat × 14 lon = 154 cells (lat 36.61–36.91, lon 237.81–238.20).

| Dataset | Valid time | P(PN>10k cells/L) valid / NaN; min / mean / max | P(pDA>500 ng/L) valid / NaN; min / mean / max | P(cDA>10 pg/cell) valid / NaN; min / mean / max | chla_filled mean (max), mg m⁻³ |
|---|---|---|---|---|---|
| wvcharmV3_0day | 2026-10-07T12Z | 144 / 10; 0.381 / 0.929 / 0.975 | 128 / 26; 0.675 / 0.750 / 0.801 | 128 / 26; 0.208 / 0.305 / 0.385 | 1.055 (3.887) |
| wvcharmV3_1day | 2026-10-08T12Z | 144 / 10; 0.381 / 0.916 / 0.975 | 128 / 26; 0.665 / 0.744 / 0.808 | 128 / 26; 0.190 / 0.297 / 0.388 | 1.158 (3.887) |
| wvcharmV3_2day | 2026-10-09T12Z | 144 / 10; 0.110 / 0.906 / 0.975 | 128 / 26; 0.588 / 0.730 / 0.795 | 128 / 26; 0.211 / 0.312 / 0.471 | 1.340 (3.887) |
| wvcharmV3_3day | 2026-10-10T12Z | 144 / 10; 0.008 / 0.857 / 0.975 | 128 / 26; 0.621 / 0.743 / 0.801 | 128 / 26; 0.236 / 0.344 / 0.558 | 1.104 (3.887) |

Example rows from wvcharmV3_0day:

```
2026-10-07T12:00:00Z,36.61,237.81,0.90165746,0.7453551,0.3347065,0.5772194
2026-10-07T12:00:00Z,36.61,237.84,0.93271524,0.7345138,0.32729033,0.9258635
```

Fill behaviour for the nowcast:

- **No cloud NaNs.** The fields are DINEOF gap-filled, so open water is fully populated.
- The 10 NaN cells for PN and chl are **land** in the east and southeast corners.
- pDA and cDA have **16 extra NaN cells along the coastline**, where PN is valid. The toxin probabilities are masked one or two pixels further offshore than PN.

Map of the subset (north at top, west at left). `#` means all values are present, `P` means only PN and chl are present, `.` means land or NaN.

```
36.91 # # # # # # # # # # . . . .
36.88 # # # # # # # # # # # P P .
36.85 # # # # # # # # # # # # # P
36.82 # # # # # # # # # # # # # #
36.79 # # # # # # # # # # # # # #
36.76 # # # # # # # # # # # # # P
36.73 # # # # # # # # # # # # # P
36.70 # # # # # # # # # # # # P P
36.67 # # # # # # # # # # # # P .
36.64 # # # # # # # # # P P P P .
36.61 # # # # # # # P P P P . . .
```

As a result, the toxin layers do not cover piers and wharves such as Monterey Wharf (36.604, −121.889). Use the nearest valid offshore pixel.

### 1.6 WMS and tiles

- Datasets with 0–360 longitudes (`wvcharmV3_0day`, `wvcharmV3_1day`, `wvcharmV3_2day`, `wvcharmV3_3day`) are **not available via WMS**.
  - `https://coastwatch.pfeg.noaa.gov/erddap/wms/wvcharmV3_0day/request?service=WMS&request=GetCapabilities&version=1.3.0` returns 404 "This dataset is never available via WMS."
- `_LonPM180` datasets have WMS.
  - `https://coastwatch.pfeg.noaa.gov/erddap/wms/wvcharmV3_1day_LonPM180/request?service=WMS&request=GetCapabilities&version=1.3.0` returns 200.
    - Layers: `wvcharmV3_1day_LonPM180:{pseudo_nitzschia, particulate_domoic, cellular_domoic, chla_filled, r486_filled, r551_filled, salinity, water_temparture}`.
    - Time dimension default="2026-10-08T12:00:00Z", nearestValue="1".
  - GetMap returned 200 image/png, 256×256 RGBA, 3983 bytes. The first attempt returned 503. Request:
    `https://coastwatch.pfeg.noaa.gov/erddap/wms/wvcharmV3_1day_LonPM180/request?service=WMS&version=1.3.0&request=GetMap&layers=wvcharmV3_1day_LonPM180:particulate_domoic&styles=&crs=EPSG:4326&bbox=36.0,-122.5,37.5,-121.0&width=256&height=256&format=image/png&transparent=true&time=2026-10-08T12:00:00Z`
- **There is no WMS for the v3.1 nowcast.** For a nowcast map layer, render tiles yourself from griddap, or use the 1-day forecast WMS.
- `/erddap/files/wvcharmV3_0day/` returns 200, so file listing is available.

---

## 2. Human-facing C-HARM pages

| Page | Status (2026-10-08) | Version stated | Thresholds | Skill / caveats | ROMS named | Data link |
|---|---|---|---|---|---|---|
| https://habs.sccoos.org/hab-forecast (CalHABMAP WordPress; links resolve to calhabmap.org) | 200 | **None** | 10,000 cells/L; pDA 500 ng/L (0.5 µg/L); cDA 10 pg/cell. Notes that environmental cDA has not exceeded 200 pg/cell. | No skill metrics. "Probability of 0.7 = 70% chance of exceeding threshold". Concentrations below the thresholds can still cause shellfish toxicity or strandings. | Not named | **Links to `charmForecast0day` (v1, frozen since 2022-03-17). The link is stale.** (Confirmed by grep of the HTML.) |
| https://www.cencoos.org/observations/models-forecasts (C-HARM `infoUrl`) | 200 | None | None | Calls C-HARM "probabilistic", run by NOAA CoastWatch West Coast, "developed at CeNCOOS" | Not linked to C-HARM | Links to CoastWatch **`charmForecast0day` (stale v1)** and to THREDDS `HAB_CELLULAR_DOMOIC_ACID_NOWCAST`. The THREDDS link redirects to the top catalog, which lists "C-HARM HAB Forecast V3 (Current)". Freshness is **UNVERIFIED**. |
| https://ioos.noaa.gov/models/california-harmful-algae-risk-mapping-c-harm | 200 | None | None | None. Generic description: "nowcasts and forecasts of domoic acid risk", ROMS. Posted 2024-08-13. | Generic "ROMS" | None |
| https://sccoos.org/california-hab-bulletin/ | 200 | None | 500 ng/L caveat | "Experimental product." "These nearshore data do not always correspond with C-HARM predictions for the open coast." "C-HARM output may be more closely correlated with marine mammals that strand." Says C-HARM "is now a product of NOAA Coast Watch in collaboration with NOAA NCCOS". The background text still mentions "3-km ROMS" and MODIS Aqua, which is outdated. | Generic | – |
| California HAB Bulletin, Dec 2022 (https://sccoos.org/?p=15290) | 200 | "C-HARM version 3 is now operational at NOAA CoastWatch" | – | "new operational circulation model (WCOFS) for currents, salinity, and temperature"; "VIIRS ocean color products in place of MODIS Aqua"; "Issues with salinity accuracy persist and have notable consequences for the Particulate Domoic Acid predictions"; PN bloom predictions suggest "a large percentage of false positives"; VIIRS pixelation | **WCOFS** ("4DVAR data assimilative model run at NOAA CSDL") | – |

### WCOFS vs UCSC ROMS

The ERDDAP v3.1 metadata `history` dated **2026-10-08** says that WCOFS salinity, temperature, and currents were used for today's run. The `source` attribute is "Satellite data, C-HARM model output, and WCOFS model output". So **WCOFS is the active driver, and it is still running as of today** (inferred from the provenance of today's run). v1 used the UCSC/CeNCOOS 3-km ROMS and MODIS-Aqua. I did not find any page that documents the UCSC ROMS being retired.

### Skill and citation

- Peer-reviewed skill assessment:
  - Anderson, C.R., Kudela, R.M., Kahru, M., Chao, Y., Rosenfeld, L.K., Bahr, F.L., Anderson, D.M., Norris, T.A. (2016). *Initial skill assessment of the California Harmful Algae Risk Mapping (C-HARM) system.* Harmful Algae 59:1–18. https://doi.org/10.1016/j.hal.2016.08.006
  - Confirmed via https://repository.library.noaa.gov/view/noaa/33076.
  - The assessment covers v1 nowcasts at nearshore pier pixels for 2014–2015. Forecast lead times correlated best with SPATT DA and with marine-mammal strandings.
- A web search cited "Moreno et al. (2022)" for v3 and WCOFS. **UNVERIFIED**: I did not open the paper.
- **I found no published skill metrics for v3.1 (UNVERIFIED).**
- No official "recommended citation" for the dataset was found on any page (UNVERIFIED / not provided). Use the ERDDAP dataset citation (creator, institution, dataset ID, URL, access date) together with Anderson et al. 2016.

---

## 3. California HAB Bulletin

| Field | Value |
|---|---|
| URL | https://sccoos.org/california-hab-bulletin/ ; archive https://sccoos.org/archive-ca-hab-bulletins/ ; mirror index https://calhabmap.org/hab-bulletin |
| Publisher | SCCOOS, with CeNCOOS and HABMAP contributors; IOOS funded. Labelled "experimental". |
| Cadence | Stated as "Monthly to bi-monthly". In practice it has been **bimonthly**: the 2026 issues are Jan–Feb, Mar–Apr, and May–Jun. |
| Latest issue | "2026: May & June", https://t.e2ma.net/webview/dg9wjm/be4d9ba98937188c7c8cad298c8b1b6e (Emma email webview). calhabmap.org also lists it at sccoos.org/california-hab-bulletin/may-june-2026/. **No Jul–Aug 2026 issue is listed as of 2026-10-08.** |
| Separate monthly news | Monthly "California HAB News" posts on calhabmap.org; the latest is "California HAB News: September 2026". |
| Machine-readable | **No.** Issues are narrative web or email pages, with no API or feed of values. The calhabmap.org WordPress site has `/feed` (RSS) for its posts, but I did not test it. |
| Verified | Pages fetched 2026-10-08 via WebFetch (summaries) |

---

## 4. CalHABMAP shore-station observations (SCCOOS ERDDAP)

- **Discovery:** `https://erddap.sccoos.org/erddap/search/index.json?searchFor=HABs&page=1&itemsPerPage=200` returned 200 and 17 tabledap datasets.
- **calhabmap.org/datasites** lists 15 of them. It omits the two Morro Bay datasets.
- **CeNCOOS ERDDAP** (`https://erddap.cencoos.org/erddap/search/index.csv?searchFor=...`) has no relevant HAB data. Searches for "domoic", "pseudo-nitzschia", and "charm" returned 404 (no results). "HAB" returned only two 2015 ECOHAB glider datasets.

Fields shared by all 17 datasets (from the `HABs-MontereyWharf` info):

- **cdm_data_type:** TimeSeries
- **institution / creator:** CalHABMAP
- **infoUrl:** https://calhabmap.org
- **license:** Same "free to use, not for legal use, no warranty" text as CoastWatch.
- **summary:** "collects weekly phytoplankton and water quality data"
- **Variables:**
  - `time`, `Location_Code`, `SampleID`, `depth`, `Temp`, `Salinity`
  - Chl and phaeo, in mg/m3
  - Nutrients, in uM
  - **`pDA`, `tDA`, `dDA` in ng/mL**. 1 ng/mL = 1000 ng/L, so the C-HARM pDA threshold of 500 ng/L equals **0.5 ng/mL**.
  - `Pseudo_nitzschia_delicatissima_group` and **`Pseudo_nitzschia_seriata_group`** in cells/L
  - Alexandrium, Dinophysis, Akashiwo, Lingulodinium, Prorocentrum, Ceratium, Cochlodinium, Gymnodinium, other diatoms, other dinoflagellates, and total phytoplankton, in cells/L
- **Access:** Public, no key.

The table below gives the last non-NaN date per variable. It was checked with `orderByMax("time")`; for example:

`https://erddap.sccoos.org/erddap/tabledap/HABs-SantaCruzWharf.csv0?time&pDA!=NaN&orderByMax(%22time%22)`

A 404 means no non-NaN values, or the variable is never populated.

| Dataset ID | Lat, Lon | Record start | Last pDA | Last PN seriata | Last PN delicat. | Rows in 2026 |
|---|---|---|---|---|---|---|
| HABs-ScrippsPier | 32.867, −117.257 | 2008-06-30 | 2026-08-10 | 2026-09-14 | 2026-09-14 | 40 |
| HABs-NewportBeachPier | 33.6061, −117.9311 | 2008-06-30 | 2026-08-10 | 2026-06-22 | 2026-06-22 | 30 |
| HABs-SantaMonicaPier | 34.008, −118.499 | 2008-06-30 | 2026-08-03 | **2026-10-05** | 2026-10-05 | 39 |
| HABs-StearnsWharf | 34.408, −119.685 | 2008-06-30 | 2026-08-03 | 2026-09-28 | 2026-09-28 | 39 |
| HABs-CalPolyPier | 35.170, −120.741 | 2008-08-15 | 2026-08-03 | **2026-10-04** | 2026-10-04 | 40 |
| HABs-MorroBayFrontBay | 35.371, −120.859 | 2023-01-03 | none | 2026-09-01 | 2026-09-01 | 35 |
| HABs-MorroBayBackBay | 35.330, −120.845 | 2023-01-03 | none | 2026-09-01 | 2026-09-01 | 35 |
| HABs-MontereyWharf | 36.604, −121.889 | 2005-06-03 | **2022-08-31** | **2025-08-27** | 2025-08-27 | 39 (temperature only) |
| HABs-SantaCruzWharf | 36.958, −122.017 | 2011-10-05 | **2026-09-30** | **2026-09-30** | none | 39 |
| HABs-InnerTomalesBay | 38.118, −122.867 | 2021-01-29 | none | 2026-03-03 | 2026-03-03 | 2 |
| HABs-TomalesBayMouth | 38.231, −122.979 | 2020-06-30 | none | 2026-03-03 | 2026-03-03 | 2 |
| HABs-TomalesBayMid-ChannelBuoy | 38.190, −122.929 | 2021-01-29 | none | 2026-03-03 | 2026-03-03 | 2 |
| HABs-BodegaMarineLab | 38.316, −123.071 | 2020-01-20 | none | 2026-04-13 | 2026-04-13 | 16 |
| HABs-BodegaMarineLabBuoy | 38.313, −123.083 | 2020-02-20 | none | 2026-01-28 | 2026-01-28 | 1 |
| HABs-HumboldtSouthBay | 40.723, −124.223 | 2017-08-16 | 2026-03-30 | none | none | 4 |
| HABs-Humboldt | 40.778, −124.197 | 2017-09-05 | 2026-04-07 | none | none | 3 |
| HABs-TrinidadPier | 41.055, −124.147 | 2017-08-31 | 2026-04-17 | none | none | 6 |

Latest Santa Cruz Wharf values (2026-08-19 → 09-30), from `https://erddap.sccoos.org/erddap/tabledap/HABs-SantaCruzWharf.csv?time,pDA,Pseudo_nitzschia_delicatissima_group,Pseudo_nitzschia_seriata_group&time%3E=2026-08-15`:

| Date | pDA (ng/mL) | PN seriata (cells/L) |
|---|---|---|
| 2026-08-19 | 0.008 | 8150 |
| 2026-08-26 | 0.050 | 20300 |
| 2026-09-02 | 0.039 | 15900 |
| 2026-09-09 | 0.013 | 1550 |
| 2026-09-16 | 0.061 | 0 |
| 2026-09-23 | 0.0 | 100 |
| 2026-09-30 | 0.202 | 4900 |

Gotchas:

1. Cadence is **weekly** at the core piers. pDA lags the cell counts by weeks: at most southern piers the last pDA is from early August.
2. **Monterey Wharf has weekly rows through 2026-10-07, but they contain only temperature.** The last PN count is from 2025-08, and the last pDA is from 2022-08.
3. Northern sites (Humboldt, Trinidad, Bodega, Tomales) are sampled sporadically, with no data since spring 2026.
4. The `time_coverage_end` metadata lags the actual data. For example, Monterey shows 2026-09-30, but a row for 2026-10-07 exists. Santa Monica shows 09-28, but PN data exists for 10-05. Always query the data rather than the metadata.
5. A record whose HAB variables are all NaN is not a zero. Treat NaN as "not measured".

Other successful request:

`https://erddap.sccoos.org/erddap/tabledap/HABs-MontereyWharf.csv?time,pDA,tDA,dDA,Pseudo_nitzschia_delicatissima_group,Pseudo_nitzschia_seriata_group,Avg_Chloro&time%3E=2026-07-01`

---

## 5. NOAA NCCOS and other West Coast HAB forecasts

- The NCCOS HAB forecasts page (https://coastalscience.noaa.gov/science-areas/habs/hab-forecasts/) lists two "External Partner HAB Forecasts":
  - **California Forecast**, which links to the SCCOOS California HAB Bulletin
  - **Pacific Northwest Forecast**, provided by NANOOS (https://www.nanoos.org/products/habs/forecasts/home.php)
  - The page states no operational status for either.
- **NOAA has no separate NCCOS-run California forecast product.** C-HARM on CoastWatch is the operational model ("product of NOAA Coast Watch in collaboration with NOAA NCCOS", per SCCOOS).
- **PNW HAB Bulletin** (https://www.nanoos.org/products/habs/forecasts/bulletins.php):
  - Covers **Washington and Oregon** beaches only, with no California coverage.
  - Issued as PDFs. The latest is "PNW HAB Bulletin: 4 October 2026".
  - It matters here only because the C-HARM domain extends to 43°N (southern Oregon).
- The NCCOS 2018 item (https://coastalscience.noaa.gov/?p=36126) is the skill-paper summary. The site showed a "U.S. Government is closed… This site will not be updated" banner, so the NCCOS pages may be stale.

---

## 6. Could not verify / open items

- Published skill metrics for v3.1 (WCOFS + VIIRS): **UNVERIFIED** (none found). Moreno et al. 2022 was not opened.
- An official dataset citation string: **not found**.
- Freshness of the CeNCOOS THREDDS C-HARM V3 entries: **UNVERIFIED**. The catalog lists "C-HARM HAB Forecast V3 (Current)", but its contents were not queried.
- GetMap for the WMS of wvcharmV3_2day_LonPM180 and wvcharmV3_3day_LonPM180: not tested. Both datasets are listed as having WMS.
- The California HAB Bulletin RSS feed (calhabmap.org/feed): not tested.
- Whether the 16-pixel coastal mask on pDA and cDA is intentional (a model domain restriction) or a salinity/WCOFS edge effect: **UNVERIFIED**. It was observed, not documented.
- Human-page content was read through WebFetch summaries. Quoted strings should be spot-checked in a browser before they are reproduced in the app.
