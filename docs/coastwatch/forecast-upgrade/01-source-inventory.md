# 1. Verified source inventory and endpoints

Every row was checked against the live endpoint on **2026-10-09, about 15:00–17:00 UTC**. "Latest" is the
newest time step the server returned, not what documentation promises. Raw responses are in
[`evidence/`](evidence/).

**Status key**
- ✅ fetched real data
- 🟡 metadata only (data needs an account I don't have)
- ❌ failed or stale

## Satellite ocean colour

| Product | Endpoint (dataset id) | Native grid | Latest on 10-09 | Lag | Access | Status |
|---|---|---|---|---|---|---|
| **Sentinel-3A OLCI chlorophyll-a, NRT, 300 m sectors** | `coastwatch.noaa.gov/erddap/griddap/noaacwS3AOLCIchlaSector{CI,DI}Daily` (CI = 140–120 °W, DI = 120–100 °W; California needs both) | 0.0025° (≈ 280 × 220 m) | 2026-10-07 19:16 UTC overpass | ≈ 2 days | Open, no key | ✅ |
| Sentinel-3B OLCI, same sectors | `noaacwS3BOLCIchlaSector{CI,DI}Daily` | 0.0025° | 2026-10-01 / 10-03 | 6–8 days, irregular | Open | ✅ (lagging) |
| VIIRS chlorophyll-a, West Coast 750 m, 1/3/8-day | `coastwatch.pfeg.noaa.gov/erddap/griddap/erdVHNchla{1,3,8}day` | 0.0075° (≈ 750 m) | 2026-10-04 (1-day) | ≈ 5 days; time axis skips days | Open | ✅ |
| NOAA-20 VIIRS chlorophyll-a, NRT 4 km | `nesdisVHNnoaa20chlaDaily` (also S-NPP `nesdisVHNchlaDaily`) | 0.0375° | 2026-10-06 | ≈ 3 days | Open | ✅ |
| VIIRS + OLCI DINEOF gap-filled, science quality | `noaacwNPPN20S3ASCIDINEOF2kmDaily` | 0.0208° (≈ 2 km) | 2026-09-27 | ≈ 12 days | Open | ✅ (too late for "now") |
| VIIRS DINEOF gap-filled, NRT | `nesdisVHNnoaaSNPPnoaa20NRTchlaGapfilledDaily` | 0.083° (≈ 9 km) | 2026-10-07 | ≈ 2 days | Open | ✅ (coarse) |
| MODIS-Aqua chlorophyll, NRT | `erdMH1chla1day_R2022NRT` (variable `chlorophyll`) | 0.0417° | 2026-10-07 | ≈ 2 days | Open | ✅ (aging sensor) |
| **PACE OCI Level-2 biogeochemistry, NRT** (chlorophyll and more) | NASA CMR `PACE_OCI_L2_BGC_NRT` v3.2; files at `obdaac-tea.earthdatacloud.nasa.gov` | ≈ 1.2 km swath at nadir | granule 2026-10-08 21:02 UTC, online 23:30 UTC | **≈ 2.5 hours** | **NASA Earthdata Login** (download returned HTTP 401) | 🟡 |
| PACE OCI L3 mapped NRT | `PACE_OCI_L3M_BGC_NRT` v3.2 (the `…_L3M_CHL_NRT` short name returns no granules) | 4 km / 9 km | not tested | — | Earthdata Login | 🟡 |
| Copernicus GlobColour OLCI 300 m L3 NRT | Copernicus Marine `OCEANCOLOUR_GLO_BGC_L3_NRT_009_101` → `cmems_obs-oc_glo_bgc-plankton_nrt_l3-olci-300m_P1D` | 300 m | catalogue extent to 2026-10-08 | ≈ 1 day | Free Copernicus Marine account | 🟡 |
| NASA GIBS imagery: PACE OCI, S3A/S3B OLCI, VIIRS NOAA-20/21 chlorophyll | `gibs.earthdata.nasa.gov/wmts/epsg3857/best/…` layers `OCI_PACE_Chlorophyll_a`, `S3A_OLCI_Chlorophyll_a`, `VIIRS_NOAA20_Chlorophyll_a` | rendered tiles to zoom 7 | 2026-10-09 (PACE, NOAA-20/21), 10-08 (OLCI) | same day | Open | ✅ (pictures, not values) |

Notes:
- **NOAA's 300 m OLCI sectors keep only a rolling 90 days** ("90 days ago – present" in the dataset metadata). They serve current data, not a training archive.
- NOAA's central server search endpoint (`coastwatch.noaa.gov/erddap/search`) returned HTTP 502 all morning. The dataset index and the data requests worked, so I found the sectors through `griddap/index.json`.

## Harmful-algal-bloom forecasts

| Product | Endpoint | Grid | Latest on 10-09 | Horizon | Access | Status |
|---|---|---|---|---|---|---|
| **C-HARM v3.1** (pDA > 500 ng/L, cDA > 10 pg/cell, *Pseudo-nitzschia* > 10⁴ cells/L) | `coastwatch.pfeg.noaa.gov/erddap/griddap/wvcharmV3_{0,1,2,3}day` | 0.03° (≈ 3 km), 31.3–43 °N | nowcast 10-07, forecast to **10-10** | nowcast + 3 days | Open | ✅ |
| "C-HARM v4" | Named in the SCCOOS California HAB Bulletin for May–June 2026 (published 2026-08-19). It links the same `wvcharmV3_3day` dataset, whose metadata says `product_version 3.1`. | — | — | — | — | No separate v4 data found |
| C-HARM v1, v2 | `charmForecast{0..3}day`, `…V2` | — | frozen | — | Open | ❌ legacy |
| Other California / West Coast operational HAB outlooks | None found on NOAA, SCCOOS or IOOS servers. Research early-warning methods exist (molecular markers; copepodamide passive sampling, reported up to 7 weeks ahead) but are not operational products. | | | | | none operational |

The C-HARM metadata (`history`) says the 1–3-day forecasts are made by **moving the gap-filled VIIRS fields with
ROMS (WCOFS) forecast currents**, filling the gaps again with DINEOF, and re-running the statistical models
([`evidence/charm-v3-metadata.txt`](evidence/charm-v3-metadata.txt)). Its 3-day ceiling is the WCOFS
forecast length.

## Ocean circulation

| Product | Endpoint | Grid | Latest on 10-09 | Horizon | Access | Status |
|---|---|---|---|---|---|---|
| **NOAA WCOFS** (ROMS with 4D-Var assimilation of HF-radar currents, SST and SSH; operational since March 2021) | `noaa-nos-ofs-pds.s3.amazonaws.com/wcofs/netcdf/YYYY/MM/DD/wcofs.t03z.YYYYMMDD.{2ds,regulargrid,fields,…}.{nNNN,fNNN}.nc` | ≈ 4 km curvilinear (348 × 1016, 40 levels); `regulargrid` 0.04° | run t03z 2026-10-09, files written 04:01–05:52 UTC | **72 h, hourly** | Open, anonymous S3 | ✅ |
| **HF radar surface currents** (observed) | `coastwatch.pfeg.noaa.gov/erddap/griddap/ucsdHfrW{6,2,1,500}` | 6 km / 2 km / 1 km / 500 m, hourly | 2026-10-09 09:00 UTC | observation (≈ 6 h lag) | Open | ✅ (bursts of HTTP 503) |
| Copernicus Marine global physics analysis and forecast | `GLOBAL_ANALYSISFORECAST_PHY_001_024` (hourly surface currents `…merged-uv_PT1H-i`) | 1/12° (≈ 9 km) | catalogue to **2026-10-19** | 10 days | Free account | 🟡 |
| Copernicus Marine global biogeochemistry forecast | `GLOBAL_ANALYSISFORECAST_BGC_001_028` (chlorophyll, phytoplankton, nutrients, O₂) | 1/4° (≈ 25 km) | catalogue to **2026-10-18** | 10 days | Free account | 🟡 |
| CA ROMS (UCLA, 3 km) | GOODS / SCCOOS THREDDS links point to 2021 folders | 3 km | not live | — | — | ❌ (C-HARM replaced it with WCOFS in v3) |
| OSCAR currents on CoastWatch West Coast | `jplOscar` | 1/3° | ends 2014-09-26 | — | — | ❌ stale |

## Measurements and official status (already in CoastWatch)

| Source | Notes |
|---|---|
| CalHABMAP shore stations (SCCOOS ERDDAP) | Already ingested by M3: 17 stations; weekly pDA, *Pseudo-nitzschia*, chlorophyll, temperature. The only in-water toxin measurements and the natural validation target. |
| CDFW / CDPH notices | Already in M2, human-curated. The only authority on closures. |
| NOAA FOSS landings | Already in M3, statewide annual history. |

## Reliability observed during testing

Out of 252 regional ERDDAP requests:
- 8 failed on the server side: 5 × HTTP 503 and 3 × HTTP 502. Every one succeeded on retry.
- 39 MODIS requests failed with HTTP 500 because my script used the wrong variable name. That was my error, corrected and re-tested ([`evidence/transient-errors.txt`](evidence/transient-errors.txt)).

The production pipeline's existing retry/backoff is therefore necessary for every new source.
