# CoastWatch: additional ocean-condition and regulatory data sources for California

Research report. No accounts were created, nobody was contacted and nothing was written to any repository.

- **Checked:** 2026-10-12, US Pacific time (2026-10-13 05:09–05:35 UTC). All "latest observed" timestamps below are UTC, retrieved live during that window with `curl` from macOS. Raw probe outputs are kept next to this file (`out_*.txt`, `*.all.csv`, `w13.txt`, `dl_wcofs/`).
- **Scope:** daily-useful information for fishermen, shellfish harvesters and coastal operators in California, published by the existing Python pipeline as JSON, PNG or u16 grids.
- **Already in CoastWatch** (found with `git grep` on `main`): C-HARM v3 (`wvcharmV3_0..3day`), VIIRS chlorophyll (`erdVHNchla1day/8day`), OLCI 300 m sector CI (`noaacwS3AOLCIchlaSectorCIDaily`), HF radar 2 km (`ucsdHfrW2`), SCCOOS `HABs-*` tabledap, and ETOPO 2022 for relief.

> **Framing rule for everything below.** These datasets describe the *physical state of the ocean*: temperature, currents, waves, wind, upwelling and chlorophyll fluorescence. **None of them is a fish-catch forecast, and none is a toxin forecast.** Chlorophyll is not toxin. Warm water or upwelling is not "good fishing". Model output (WCOFS, C-HARM) is a forecast, not an observation. A buoy or shore-station reading is an observation, not a forecast. The UI must label each one accordingly (see CLAUDE.md "Science" rule).

---

## 0. Executive summary

| Rank | Source | Why | Effort |
|---|---|---|---|
| 1 | **NWS marine zone forecasts and alerts** (api.weather.gov, PZZ zones) | The highest-value daily decision input for boaters: wind, seas and Small Craft Advisories. Official, free, JSON, verified live. | Low |
| 2 | **Live observations from NDBC/CDIP buoys and NOAA CO-OPS stations**: water temperature, wind, waves, tide predictions | Real measurements that build trust, at about 60 California points. Simple text/JSON. Tide predictions are something harvesters use every day. | Low–medium |
| 3 | **Satellite SST: NOAA Geo-Polar Blended 5 km (gap-free)** plus **ACSPO L3S-LEO 2 km (cloud-masked)**; MUR as a fallback | A daily SST map and anomaly. Upwelling fronts are visible. Same ERDDAP pattern the pipeline already uses. | Low |
| 4 | **CeNCOOS/SCCOOS automated shore stations** (pier temperature, salinity and chlorophyll fluorescence, 4–15 min data) | The only near-real-time *nearshore and in-harbor* measurements, at the places shellfish harvesters work. They complement the existing HABs sampling. | Medium |
| — | Close runner-up: **CUTI/BEUTI upwelling indices** | Strong science context, but about 2 weeks of latency (observed). Better suited to a "context" panel than a daily one. | Low |
| — | **WCOFS**: LATER | Operational, but 4 km resolution. NOAA's own report says it "does not sufficiently resolve the coastal bays and estuaries". Mean in-situ SST nowcast RMSE is 1.15 °C. Files are 74–582 MB each. | High |
| — | **CDFW MPA polygons**: INTEGRATE as a *reference layer*, with a strong disclaimer | Official CC-BY polygons, but labelled "as of January 1, 2019", and the Fish and Game Commission's decision on 20 MPA petitions is pending (around October 2026). | Low build, high liability |
| — | Machine-readable fishing regulations/seasons | **They do not exist.** CDFW publishes HTML tables and PDFs only. Federal 50 CFR 660 text is available through the eCFR API, but as legal text, not structured seasons. | NO |

Key operational risk observed live: the `coastwatch.pfeg.noaa.gov` ERDDAP **restarted at 05:16 UTC during this check**. `ucsdHfrW2`, `ucsdHfrW6` and `ucsdHfrW1` returned `Not Found: Currently unknown datasetID` for about 10 minutes (back by 05:26 UTC). Earlier the same night `ucsdHfrW6` returned HTTP 503. A same-content mirror for HF radar exists on the NDBC THREDDS server and was serving data 4 h fresher. Section 4 has the details.

---

## 1. Master evaluation table

Legend for type: **OBS** = in-situ observation; **SAT** = satellite observation (L3 = swath/gridded with gaps, L4 = gap-filled analysis); **MODEL-A** = model analysis/nowcast; **FCST** = forecast; **REG** = regulatory/reference geometry.

### 1a. Ocean models and forecasts

| Field | **NOAA WCOFS** (West Coast OFS) | **NWS coastal/offshore zone forecasts** | **C-HARM v3** (already used) |
|---|---|---|---|
| Owner | NOAA NOS CO-OPS / OCS ([WCOFS page](https://tidesandcurrents.noaa.gov/ofs/wcofs/wcofs.html)) | NOAA NWS WFOs EKA, MTR, LOX, SGX (and MFR for the Oregon border) | NOAA CoastWatch West Coast Node / NCCOS / UCSC (dataset on [pfeg ERDDAP](https://coastwatch.pfeg.noaa.gov/erddap/griddap/wvcharmV3_0day.html)) |
| Endpoint | S3 (NODD): `https://noaa-nos-ofs-pds.s3.amazonaws.com/wcofs/netcdf/YYYY/MM/DD/wcofs.t03z.YYYYMMDD.{2ds.fNNN,fields.fNNN,regulargrid.fNNN,stations.forecast}.nc` ([AWS registry](https://registry.opendata.aws/noaa-ofs/)). OPeNDAP: `https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/WCOFS/MODELS/YYYY/MM/DD/wcofs.t03z.YYYYMMDD.regulargrid.f024.nc` | `https://api.weather.gov/zones/coastal/PZZ545/forecast`; alerts at `https://api.weather.gov/alerts/active?area=PZ` ([API docs](https://www.weather.gov/documentation/services-web-api)) | `https://coastwatch.pfeg.noaa.gov/erddap/griddap/wvcharmV3_{0,1,2,3}day` |
| Format | NetCDF4 (ROMS native curvilinear `2ds`/`fields`; interpolated `regulargrid`) | GeoJSON / JSON-LD (text periods) | NetCDF/CSV via griddap |
| Spatial res. | Native about 4 km, 40 σ-levels ([NOAA Tech Rep NOS CO-OPS 097](https://repository.library.noaa.gov/view/noaa/44306), p. 7; [coastaloceanmodels.noaa.gov/WCOFS](https://coastaloceanmodels.noaa.gov/WCOFS/)). The `regulargrid` file is 935×833 at 0.04° (18.42–55.78°N, 145.38–112.1°W), 41 z-levels starting at 0 m (verified via the OPeNDAP DDS) | Zone polygons: inner 0–10 nm, outer 10–60 nm, plus bays (SF Bay, Monterey Bay, Humboldt Bay and Bar) | about 3 km (0.0275°) |
| Temporal res. | Hourly `2ds` surface files f001–f072. 3-hourly `fields`/`regulargrid` (n003–n024 nowcast, f003–f072 forecast) | Roughly 12-hour forecast periods out to about 5 days | Daily, nowcast plus 1–3-day forecast |
| Type | MODEL-A (24 h 4D-Var nowcast) + FCST (72 h) | FCST (human-edited) + warnings | MODEL FCST (probabilistic HAB/toxin risk) |
| Cadence / latency | One cycle per day at 03z. The 2026-10-13 t03z files appeared on S3 between 04:03 and 04:12 UTC (observed `LastModified`) | Updated about 4×/day plus amendments. PZZ545 `updated` = 2026-10-12T20:10-07:00 | Daily |
| CA coverage / gaps | Full coast 24–54°N. **"With the 4-km resolution, the model does not sufficiently resolve the coastal bays and estuaries. The model results in these areas, if any, should be used with extra caution"** (Tech Rep 097, §2.1). No meaningful <1 km nearshore or harbor skill | 25 CA zones verified: PZZ410/415/450/455/470/475 (EKA), 530/531/535/540/545/560/565/570/571/575/576 (MTR), 645/650/655/670/673/676 (LOX), 740/745 (SGX). The Oregon-border waters (Cape Blanco–Pt St George) are MFR zones PZZ356/376. Harbors are not forecast individually | Offshore of about 3 km. Masked nearshore |
| Latest observed | Cycle 2026-10-13 t03z: f001–f072 present (165 objects). Regular-grid file f072 is 582 MB | PZZ545 issued 2026-10-12 20:10 PDT. **11 active PZ alerts**, including Small Craft Advisories for PZZ450/455/470/475 and a Hazardous Seas Warning at Cape Blanco–Pt St George | `wvcharmV3_0day` 2026-10-11T12:00Z. `_3day` 2026-10-14T12:00Z. **Missing days in the last 30: 09-17, 10-05 and 10-08** (time-axis gaps observed) |
| Units | `temp` °C, `u_eastward`/`v_northward` m/s, `zeta` m (`zetatomllw` relative to MLLW), `salt` PSU | Knots, feet, seconds (plain text: "NW wind 5 to 10 kt… Seas 4 to 6 ft. Wave Detail: NW 6 ft at 10 seconds") | Probability (0–1) |
| Validation | Tech Rep 097 (Xu, Zhang, Kurapov, Seroka, Bayler 2022, [doi:10.25923/a9mj-zm71](https://doi.org/10.25923/a9mj-zm71)), 1-yr assessment Mar 2020–Mar 2021. Day-1 SST RMSE vs satellite was 0.58 °C (0.79 °C without DA). **Nowcast in-situ SST RMSE was 0.36–2.69 °C, mean 1.15 °C, against 69 NDBC/NOS stations, about the same as the free run (1.18 °C).** DA reduces day-1 current RMSE by 1–3 cm/s (8–15 %). Water level was **not** assessed (no estuaries; de-tiding recommended). The report's own future-work list says 4 km cannot resolve slope processes and ROMS 4DVAR is too slow for 2 km | Not quantified per zone. NWS verification is internal | Anderson et al. 2016 (Harmful Algae) and later CoastWatch v3 notes (already documented in the repo) |
| Licence | US Government, public domain | Public domain. `User-Agent` header required | Public, NOAA ERDDAP disclaimer |
| Reliability | One cycle per day. If the 03z cycle fails, there is no fresher run that day. Files are large | Generally high. API has occasional 5xx responses and rate-limits. Text format is stable | Daily gaps as listed above |
| Integration cost | **High.** Curvilinear grid or 582 MB regular-grid files. Use OPeNDAP to subset the surface layer only (`temp[0][0][y0:y1][x0:x1]`). Requires a nearshore mask and a skill caveat | **Low.** One JSON call per zone (25 calls). Text is shown as-is | Already integrated |
| Python tooling | `xarray` + `netCDF4` (OPeNDAP), or `s3fs`/`fsspec` for S3; `xesmf` if regridding native | `requests` (custom `User-Agent`), `shapely` for zone polygons | `erddapy`/`requests` (existing) |
| **Recommendation** | **LATER.** Use only offshore surface currents/SST as "model forecast" context after the observed layers are live. Do not show it inside bays and harbors. Do not present it as better than satellite SST | **INTEGRATE NOW (staging)**. Rank 1 | Keep. Add gap detection |

### 1b. In-situ observations

| Field | **NOAA CO-OPS Tides & Currents** | **NDBC buoys + C-MAN** (incl. CDIP Waverider relays) | **CDIP direct** (Scripps) | **CeNCOOS/SCCOOS automated shore stations** | **SCCOOS ERDDAP (erddap.sccoos.org)** | **IOOS Glider DAC** |
|---|---|---|---|---|---|---|
| Owner | NOAA NOS CO-OPS | NOAA NDBC; owners incl. Scripps/CDIP, USACE, MBARI, NERRS, CeNCOOS | Scripps Institution of Oceanography CDIP, funded by USACE and CA Parks DBW | SCCOOS (Newport, Santa Monica, Scripps and Stearns piers), CeNCOOS partners (Humboldt State, MLML, Cal Poly, BML, MBA); served by Axiom (`erddap.operational.sensors.axds.co` proxied by CeNCOOS) | SCCOOS | NOAA IOOS, Spray/Seaglider operators (Scripps, UW, etc.) |
| Endpoint | Data: `https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?station=9413450&product=water_temperature&date=latest&units=metric&time_zone=gmt&format=json&application=<app>`. Station list: `https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json?type=watertemp` ([API](https://api.tidesandcurrents.noaa.gov/api/prod/), [MDAPI](https://api.tidesandcurrents.noaa.gov/mdapi/prod/)) | `https://www.ndbc.noaa.gov/data/latest_obs/latest_obs.txt`, `https://www.ndbc.noaa.gov/data/realtime2/{ID}.txt` (+ `.spec`), station list `https://www.ndbc.noaa.gov/activestations.xml` ([access FAQ](https://www.ndbc.noaa.gov/faq/rt_data_access.shtml), [field definitions](https://www.ndbc.noaa.gov/faq/measdes.shtml)) | OPeNDAP `https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/{stn}p1_rt.nc` ([cdip.ucsd.edu](https://cdip.ucsd.edu/)) | `https://erddap.cencoos.org/erddap/tabledap/{newport-pier-automated-shore-sta, santa-monica-pier-automated-shor-1, scripps-pier-automated-shore-sta-1, stearns-wharf-automated-shore-st-1, edu_humboldt_humboldt, monterey-wharf-real-time-samplin, edu_calpoly_marine_morro, san-luis-bay-cal-poly-pier-shore, bodega-bay-bml_wts, monterey-bay-aquarium-seawate, tiburon-water-tibc1}.csv` | `https://erddap.sccoos.org/erddap/tabledap/HABs-*` | `https://gliders.ioos.us/erddap/tabledap/allDatasets.csv` ([gliders.ioos.us](https://gliders.ioos.us/)) |
| Format | JSON/CSV/XML | Fixed-width text (NDBC realtime2), XML station list | NetCDF (CF), OPeNDAP | CSV/JSON/NetCDF (ERDDAP tabledap) with IOOS QARTOD `*_qc_agg` flags | ERDDAP tabledap | ERDDAP tabledap (profiles) |
| Spatial | Points. **16 CA water-temperature stations, 19 CA water-level stations** (MDAPI, verified) | Points. **106 active stations** in a 32.4–42.1°N, 117–126.5°W box: 14 NDBC 3-m buoys, 26 Scripps/CDIP buoys, 35 NOS/PORTS fixed, 13 NERRS, plus others (activestations.xml) | Points. Nearshore and offshore Waveriders | Points *on piers and in harbors and bays*, at 1–5 m depth | Points (weekly HAB grab samples) | Tracks, 0–500 m profiles |
| Temporal | 6-min | 10-min/hourly met; waves every 30 min | 30-min waves; SST | 4–15 min | Weekly | Profiles every few hours |
| Type | OBS (+ astronomical tide **predictions**, which are deterministic forecasts) | OBS | OBS | OBS | OBS (already used) | OBS |
| Latency | Minutes | Under 1 h | Under 1 h | Under 1 h | Days | Hours |
| CA gaps | No open-coast stations between Port San Luis and Monterey, or between Arena Cove and North Spit. **Four of 16 water-temperature stations returned "No data was found" at check time: San Francisco 9414290, Los Angeles 9410660, Redwood City 9414523 and Martinez 9415102.** SF and LA had no water temperature at all from 2026-09-01 to 10-12 | Mostly offshore (3-m buoys 10–120 nm out). Wind is not measured on CDIP Waveriders. **No "latest" data from 46053 (E. Santa Barbara), 46235, 46251 or 46273** at check time | Mostly southern CA dense, north sparse | Only about 10 active stations statewide. **Tiburon last reported 2026-09-24 (stale).** Newport Beach temperature was NaN with `qc_agg=2` (not evaluated) at 03:30–03:36Z, while chlorophyll was valid | 15 piers. Several stale for months (Tomales, Trinidad, BML) | Sparse, mission-based |
| Latest observed | 12/16 water-temp stations at 2026-10-13 05:06–05:12Z (e.g. Monterey 63.7 °F, Pt Reyes 57.4 °F, Crescent City 52.3 °F). Monterey water level 05:12Z. La Jolla hi/lo predictions returned for 10-13/14 | `latest_obs.txt` rows 04:00–04:50Z. 46026 realtime2 04:50Z (WVHT 1.8 m, DPD 17 s, WTMP 16.0 °C) | Station 201 (Scripps Nearshore): 2026-10-13T04:30Z, Hs 0.86 m, Tp 11.1 s | Newport, Santa Monica, Scripps and Stearns 04:52–05:00Z. Humboldt Bay pier 04:45Z (13.96 °C, chl 10.24 µg/L, S 33.49). Monterey wharf 02:40Z. Cal Poly pier 04:42Z | Monterey Wharf 10-07. Scripps Pier 10-05. CalPoly 10-04 | 8 gliders in CA waters, latest 2026-10-12/13 |
| Units | °F/°C, ft/m relative to a chosen datum (MLLW) | WSPD/GST m/s, WVHT m, DPD/APD s, MWD/WDIR °T, WTMP °C; `MM` = missing | m, s, °, °C | °C, PSU, µg/L (fluorometric chl, *not* lab-extracted), NTU, pH | cells/L, µg/L, ng/mL | °C, PSU |
| Validation / QC | CO-OPS QA/QC, preliminary vs verified flags (`f`, `q`) | NDBC QC ([handbook](https://www.ndbc.noaa.gov/faq/measdes.shtml)) | CDIP QC flags `waveFlagPrimary` | IOOS QARTOD: `flag_values 1,2,3,4,9 = PASS, NOT_EVALUATED, SUSPECT, FAIL, MISSING` (verified in metadata) | Lab methods per CalHABMAP | DAC QC |
| Licence | Public domain | Public domain | Free; credit CDIP | "May be used and redistributed for free but not intended for legal use" (ERDDAP license attribute) | Same | Public |
| Reliability | High overall, but individual sensors go dark for weeks (SF and LA, above) | High overall. **Individual records alternate**: the :40 record has WVHT but DPD=`MM`, and the :30 record has waves all `MM` (observed in 46026, 46042, 46011 and 46013). Code must take the *latest non-MM value per field* within a window, not the last row | High | Variable. Single-sensor biofouling, maintenance gaps | Lab-paced | Mission gaps |
| Integration cost | Low | Low–medium (fixed-width parse) | Medium (OPeNDAP indexes) | Medium (per-station variable names differ; QC flags) | Already integrated | Medium–high (profile UX) |
| Python | `requests` | `pandas.read_csv(sep=r"\s+", skiprows=[1], na_values="MM")` | `xarray.open_dataset(url)` | `erddapy` | existing | `erddapy` |
| **Recommendation** | **INTEGRATE NOW** (water temperature and tide predictions). Rank 2 | **INTEGRATE NOW** (wind, waves, SST). Rank 2 | **LATER** (only if NDBC relays are insufficient. NDBC already relays CDIP stations as 462xx) | **INTEGRATE NOW (staging)**. Rank 4 | Keep | **NO** for the daily UI. Science context only |

### 1c. Satellite and analysis grids

| Field | **MUR SST v4.1** | **NOAA Geo-Polar Blended SST 5 km** | **ACSPO L3S-LEO 2 km SST (NRT)** | **CoralTemp** | **OLCI 300 m chl (S3A, S3B sectors)** | **VIIRS chl (erdVHNchla) / DINEOF gap-filled** | **Sentinel-2 MSI** | **CUTI / BEUTI** |
|---|---|---|---|---|---|---|---|---|
| Owner | NASA JPL (PO.DAAC), served by ERD | NOAA NESDIS/STAR, CoastWatch | NOAA NESDIS/STAR | NOAA Coral Reef Watch | EUMETSAT/ESA, processed by NOAA CoastWatch | NOAA CoastWatch | ESA Copernicus | NOAA SWFSC ERD + UCSC |
| Endpoint | `https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41` (+ `jplMURSST41anom1day`) ([PO.DAAC](https://podaac.jpl.nasa.gov/dataset/MUR-JPL-L4-GLOB-v4.1)) | `https://coastwatch.noaa.gov/erddap/griddap/noaacwBLENDEDsstDNDaily` (day+night). Also `noaacwBLENDEDsstDaily` (night) and `noaacwBLENDEDCsstDaily` ([product page](https://coastwatch.noaa.gov/cwn/products/noaa-geo-polar-blended-global-sea-surface-temperature-analysis-level-4.html)) | `https://coastwatch.noaa.gov/erddap/griddap/noaacwLEOACSPOSSTL3SnrtCDaily` ([OSPO](https://ospo.noaa.gov/products/ocean/sst/acspo/L3S/), [PO.DAAC](https://podaac.jpl.nasa.gov/dataset/L3S_LEO_DY-STAR-v2.81)) | `https://coastwatch.noaa.gov/erddap/griddap/noaacrwsstDaily` | `https://coastwatch.noaa.gov/erddap/griddap/noaacwS3AOLCIchlaSectorCIDaily` (CI = central/south CA sector, already used), `…SectorCHDaily`, `noaacwS3BOLCIchlaSector{CI,CH}Daily` | `https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdVHNchla1day`, `noaacwNPPN20S3ASCIDINEOF2kmDaily` | Copernicus Data Space | `https://oceanview.pfeg.noaa.gov/erddap/griddap/erdCUTIdaily`, `erdBEUTIdaily` ([intro](https://oceanview.pfeg.noaa.gov/products/upwelling/intro)) |
| Res. | 0.01° grid (about 1 km). *Effective* feature resolution is coarser where only microwave or older data are available | 0.05° (about 5 km) | 0.02° (about 2 km) | 0.05° | 0.0025° (about 300 m) | 750 m / 2 km | 10–60 m | 1° latitude bins, 31–47°N, coastal strip |
| Temporal | Daily | Daily | Daily (also AM/PM, day/night variants) | Daily | Daily per overpass | Daily / 8-day | 5-day revisit | Daily |
| Type | SAT L4 (gap-filled analysis) | SAT L4 (gap-filled OI analysis) | SAT L3S (**gaps where cloudy**) | SAT L4 | SAT L3 (gaps) | SAT L3 / L4 (DINEOF) | SAT L1C/L2A | MODEL-A-derived index (ROMS 4DVAR state estimate) |
| Latest observed | 2026-10-11T09:00Z (about 1.8 days lag at check). Anomaly 2026-10-10 | 2026-10-11T12:00Z (all three variants) | NRT: 2026-10-11T12:00Z. *Science-quality `L3SCDaily` stopped at 2026-08-06, and `noaacwN20ACSPOSSTL3GCDaily` at 2026-01-13: do not use those IDs* | 2026-10-10 | S3A CI 2026-10-11T19:12Z, CH 10-11T18:26Z. **S3B CI/CH last 2026-10-01** (about 11 days stale). S3B global 4 km 10-04 | `erdVHNchla1day` 2026-10-06 (6-day lag), 8-day 10-03. **DINEOF 2 km 2026-09-27 (15 days stale)** | n/a | **2026-09-27** (CUTI and BEUTI daily, about 15 days lag) |
| CA nearshore gaps | Interpolated right to the coastline. Near-shore values in bays and in upwelling cells are smoothed or biased. Koutantou et al. 2023 Saildrone validation off California ([Remote Sens. 15:2277](https://doi.org/10.3390/rs15092277)) | 5 km, so cells closer than about 5 km to shore are mixed or masked | Coastal pixels and fog/stratus (summer marine layer) cause large gaps. Cloud-leak residuals | 5 km | Best coastal resolution. Gaps from cloud, glint, adjacency and Case-2 turbid water; standard OC algorithm biased in turbid bays | Clouds; DINEOF smooths | Not a calibrated ocean-colour product | Not spatial. One number per degree of latitude |
| Units | `analysed_sst` °C (ERDDAP). `analysis_error` K | °C | °C, plus `quality_level` 0–5 and `sses_bias` | °C | `chlor_a` mg m⁻³ | mg m⁻³ | reflectance | CUTI m² s⁻¹ (per m coastline). BEUTI mmol NO₃ m⁻¹ s⁻¹ |
| Validation | Chin et al. 2017 (RSE). Off-CA Saildrone validation (Koutantou 2023) | Maturi et al. 2017 BAMS [doi:10.1175/BAMS-D-15-00002.1](https://doi.org/10.1175/BAMS-D-15-00002.1). About 0.3 K RMS vs Argo (NOAA poster, preliminary) | Jonasson, Ignatov et al. SPIE 2019–2021 (refs on PO.DAAC). Dataset DOI 10.25921/d8ef-8597 | CRW docs | NOAA CoastWatch OLCI processing | CoastWatch | — | Jacox et al. 2018, JGR Oceans [doi:10.1029/2018JC014187](https://doi.org/10.1029/2018JC014187) |
| Licence | PO.DAAC free; cite | GHRSST free and open | GHRSST free and open | Public | **"Contains modified Copernicus Sentinel data"** attribution required (ERDDAP license) | Public | Copernicus free/open | Free, ERD disclaimer |
| Reliability | Good; occasional multi-day NRT delays | Good | Good; dataset IDs churn (two variants found stale) | Good | S3A fine; S3B sectors currently stalled | erdVHNchla lagging ~6 d | — | Lagging ~2 weeks |
| Cost | Low (ERDDAP subset, already-familiar pattern) | Low | Low | Low | Already integrated | Already integrated | High | Low |
| Python | `requests` + `xarray` (as the existing pipeline does) | same | same | same | existing | existing | `sentinelhub`/`odc-stac` | `requests` CSV |
| **Recommendation** | **Fallback** for SST map | **INTEGRATE NOW** as the gap-free SST map. Rank 3 | **INTEGRATE NOW** as an "observed clear-sky pixels" overlay or SST-front layer (rank 3, same job) | NO (redundant) | Keep S3A. Add S3B **LATER**, with a freshness gate | Keep. **Flag the DINEOF staleness** if it is used anywhere | **NO** for daily ops | **LATER**: a weekly "upwelling context" card |

### 1d. Surface currents (HF radar)

| Field | **ERD ERDDAP `ucsdHfrW{1,2,6}`** (already used: W2) | **NDBC THREDDS `hfradar_uswc_{1,2,6}km`** | **HFRNet THREDDS (hfrnet-tds.ucsd.edu)** |
|---|---|---|---|
| Owner | HFRNet/SIO data, served by NOAA ERD | HFRNet data, served by NOAA NDBC ([hfradar.ioos.us](https://hfradar.ioos.us/) points here) | Scripps CORDC |
| Endpoint | `https://coastwatch.pfeg.noaa.gov/erddap/griddap/ucsdHfrW2` | `https://dods.ndbc.noaa.gov/thredds/dodsC/hfradar_uswc_2km` (also `_6km`, `_1km`) | `https://hfrnet-tds.ucsd.edu/thredds/…` |
| Res. | W2 0.018°×0.021° (2 km); W1 1 km; W6 6 km. **Rolling window only: time coverage starts 2026-07-11 (about 90 days)** | Same grids. `_6km` is 367×234 | — |
| Latest observed | W6 2026-10-13T00:00Z, W2 03:00Z, W1 02:00Z (05:10Z check). W2 had 748 hourly steps since 09-12 (no gap > 1.5 h) | `_6km` **2026-10-13T04:00Z** (fresher than ERD) | **Connection failed (HTTP 000)** at 05:12Z; also `hfrnet.ucsd.edu` |
| Coverage | Snapshot W6 at 00Z: 5,402 valid vectors of 24,249 cells in a 32.5–42°N, 117–125.5°W box. Coverage is patchy and varies hourly with site outages. No data inside harbors; the first 1–2 km near shore are often missing | same | — |
| Units | `water_u`/`water_v` m/s, `hdop`, `number_of_sites`, `number_of_radials` | same | — |
| Validation | Kim, Terrill and Cornuelle 2008 (JGR) for the US West Coast network. HFR is assimilated into WCOFS (Tech Rep 097) | same | — |
| Reliability evidence (this check) | `ucsdHfrW6` returned **HTTP 503** at about 05:12Z. ERD ERDDAP **restarted at 05:16Z**, and W1/W2/W6 then returned `Currently unknown datasetID` from 05:19Z to about 05:23Z; they were back by 05:26Z. The 30-day time axis is complete (W2 748, W6 745 hourly steps, no gaps > 1.5 h) | Up | Down |
| **Recommendation** | Keep, but make **NDBC THREDDS the automatic fallback** (same variables) | **INTEGRATE NOW as the fallback** | NO |

### 1e. Regulatory and reference geometry

| Field | **CDFW Marine Protected Areas [ds582]** | **NOAA WCR Groundfish Conservation Areas (RCA, YRCA, CCA, GEA)** | **CDFW Groundfish Recreational Boundaries [ds3143] / 20-fathom line [ds3160] / RCA lines [ds3144]** | **Regulations and seasons (CDFW web, CCR Title 14, 50 CFR 660)** | **Bathymetry: NCEI CRM/CUDEM, GEBCO, ETOPO 2022** |
|---|---|---|---|---|---|
| Owner | CDFW Marine Region GIS Lab | NOAA Fisheries West Coast Region | CDFW | CDFW / FGC / NOAA | NOAA NCEI / GEBCO |
| Endpoint | FeatureServer `https://services2.arcgis.com/Uq9r85Potqm3MfRV/arcgis/rest/services/biosds582_fpu/FeatureServer/0/query?where=1%3D1&outFields=*&f=geojson` ([BIOS metadata](https://filelib.wildlife.ca.gov/Public/BDB/GIS/BIOS/metadata/DS0582.html), [data.gov](https://catalog.data.gov/dataset/california-marine-protected-areas-ds582)) | `WCR_GCA.gdb` download ([NOAA page](https://www.fisheries.noaa.gov/resource/map/groundfish-conservation-areas-maps-and-gis-data-west-coast-region)); waypoints ([closed areas](https://www.fisheries.noaa.gov/west-coast/sustainable-fisheries/west-coast-groundfish-closed-areas)) | `…/biosds3143_fpu/FeatureServer/0` ([DS3143](https://filelib.wildlife.ca.gov/Public/BDB/GIS/BIOS/metadata/DS3143.html)); ds3144 ([data.gov](https://catalog.data.gov/dataset/rockfish-conservation-area-r7-cdfw-ds3144)); ds3160 ([DS3160](https://filelib.wildlife.ca.gov/Public/BDB/GIS/BIOS/metadata/DS3160.html)) | [Groundfish summary](https://wildlife.ca.gov/Fishing/Ocean/Regulations/Groundfish-Summary), [Sport fishing regs](https://wildlife.ca.gov/Fishing/Ocean/Regulations/Sport-Fishing); eCFR API `https://www.ecfr.gov/api/versioner/v1/structure/{date}/title-50.json?part=660` | [CRM](https://www.ncei.noaa.gov/products/coastal-relief-model); [GEBCO](https://www.gebco.net/data_and_products/gridded_bathymetry_data/); ETOPO already used |
| Format | GeoJSON/PBF/Esri JSON, EPSG:3857 | File geodatabase | GeoJSON/shapefile | HTML tables + PDF; eCFR JSON/XML (legal text) | NetCDF/GeoTIFF |
| Verified | HTTP 200. **155 features.** Fields: `NAME, CCR, Type, CCR_Int, SHORTNAME, Acres…`. `dataLastEditDate` = 2024-01-09. Description: *"all of California's marine protected areas (MPAs) as January 1, 2019… not intended for navigational use or defining legal boundaries."* Licence CC BY 4.0 | Page HTTP 200. NOAA: GIS is "an approximate representation and is not an official record"; the legal definition is 50 CFR 660 | Search-verified; metadata updated Nov 2025 (data.gov) | Groundfish summary page "Updated June 23, 2026", HTML tables plus "Printable PDF". **No API, CSV, RSS or GIS for seasons or depth limits.** "Subject to in-season change." eCFR API HTTP 200 (Title 50 up to date as of 2026-10-08) | CRM 1–3 arc-sec; CUDEM 1/9 arc-sec in places |
| Pending change | **FGC decadal-review MPA petitions (20 petitions) are pending.** OPC recommendations dated July 2026 ([PDF](https://opc.ca.gov/wp-content/uploads/2026/07/OPC-Recommendations-on-MPA-Petitions-July-2026-508.pdf)); FGC decision expected around October 2026 (news: [Independent](https://www.independent.com/2026/05/01/changing-the-central-coasts-protected-marine-areas/), [KCBX](https://www.kcbx.org/environment-and-energy/2026-05-04/state-officials-to-hold-hearings-over-ocean-protections-in-central-coast)). Boundaries may change after adoption and before the GIS layer is updated | Federal rule changes each biennium | — | Seasons and depths change in-season | — |
| **Recommendation** | **INTEGRATE NOW (staging) as a reference overlay only**, with the dataset date shown, a "not legal boundaries – check CDFW" disclaimer and a link to the CDFW MPA page. Never compute "you may fish here" | **LATER** (RCA lines matter for commercial groundfish but change by rule; approximation caveat) | **LATER**: management-area polygons only as a "which area am I in → link to CDFW table" helper | **NO** machine-readable seasons. Link out to CDFW only. Never restate bag or season rules | Keep ETOPO; **LATER** CRM/CUDEM for nearshore shading |

### 1f. Other sources considered

| Source | Notes | Recommendation |
|---|---|---|
| **Copernicus Marine (CMEMS)**, e.g. global physics analysis/forecast, OLCI L3 | Free, and commercial use is allowed. **Registration is required.** Attribution "Generated using E.U. Copernicus Marine Service Information" plus DOIs must be visible on the home or product page. Licensee must keep usage records; French law governs; the licence can be revised ([licence](https://marine.copernicus.eu/user-corner/service-commitments-and-licence)). Tooling: `copernicusmarine` toolbox ([install](https://help.marine.copernicus.eu/en/articles/7970514-copernicus-marine-toolbox-installation)). Global 1/12° physics adds little over WCOFS/HFR/SST for California nearshore | **NO** for now (account plus attribution obligations; US sources cover the need) |
| **SCCOOS ROMS** (`roms_fcst`, `roms_ncst` on erddap.sccoos.org) | Last data 2022-12-01, so discontinued | **NO** |
| **CDPH marine biotoxin program / mussel quarantine** | The authoritative source for shellfish toxin information ([CDPH](https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Marine-Biotoxin-Monitoring-Program.aspx)). Already in scope of CoastWatch's official-notices system | Keep through the existing `cwp review-official` human process |
| **CDIP MOP nearshore wave model** | THREDDS catalog `cdip/model/MOP_alongshore` HTTP 200; about 100 m alongshore wave nowcast/forecast for southern CA | **LATER** (high value for surf-zone or launch decisions; needs its own validation review) |

---

## 2. Cross-cutting findings

1. **The ERD ERDDAP (`coastwatch.pfeg.noaa.gov`) is a single point of failure** for C-HARM, HF radar and VIIRS chlorophyll. During this check it was restarted (status page: "Startup was at 2026-10-12T22:16:38-07:00"), and the HF radar and C-HARM datasets were *unknown* for about 10 minutes while `LoadDatasets` ran. The pipeline should retry with backoff over at least 30 minutes before falling back. It should keep the last good artifact and stamp it as stale rather than publish an empty layer.
2. **Freshness varies by product, and several "daily" products are days to weeks behind**: erdVHNchla1day 6 d, CUTI 15 d, DINEOF 15 d, S3B OLCI 11 d, MUR about 1.8 d. Every layer needs a freshness threshold and a visible "as of" date.
3. **Dataset IDs churn.** For example, the ACSPO science-quality and N20 L3 IDs stopped updating months ago, while the `…nrt…` ID is current. Pin IDs in config, and run a daily `time[(last)]` probe in the existing data-refresh alerting.
4. **Nearshore under 1 km and estuaries are poorly served by every grid product.** Only shore stations, CO-OPS gauges and the OLCI 300 m chlorophyll reach into harbors. Be explicit about this in the UI.
5. **Observations vs forecasts.** Only NWS (official forecast), WCOFS (model forecast), C-HARM (model forecast) and tide predictions (astronomical) are forward-looking. Everything else is an observation or analysis of the past 1–48 h.

---

## 3. Ranked shortlist: minimal staging ingestion designs

These designs assume the existing pipeline pattern: Python fetch → validate → write JSON/PNG/u16 to the `staging` data channel, with manifests. Per memory, **do not add new fields to existing manifest schemas.** New sources go into new, separately versioned files that the web app reads behind a staging flag.

### Rank 1: NWS marine zone forecasts and marine alerts

- **Endpoints**
  - `GET https://api.weather.gov/zones/coastal/{ZONE}/forecast` for each of the 25 CA zones above, plus PZZ356/376 if the north-coast border is in scope.
  - `GET https://api.weather.gov/alerts/active?area=PZ`, filtered to the zone IDs.
  - Zone geometry once per month from `GET https://api.weather.gov/zones/coastal/{ZONE}`. Note: `/zones?type=coastal&area=CA` returned **an empty FeatureCollection** in testing, so enumerate the list explicitly.
  - Headers: `User-Agent: CoastWatch (contact URL)` (required by the API) and `Accept: application/geo+json`.
- **Variables:** `properties.updated` and `periods[].name/detailedForecast`. From alerts: `event, severity, onset, ends, areaDesc, headline, description`.
- **QC rules**
  - The HTTP status must be 200 and the response must contain at least 1 period.
  - Reject the forecast if `updated` is more than 18 h old.
  - Do not parse numbers out of the text. Show it verbatim; a parsed "max seas" is a derived claim that could be mis-extracted.
  - Expire alerts with `ends < now`.
- **Freshness:** green if `updated` < 8 h old, amber 8–18 h, hidden with a "forecast unavailable – check weather.gov" message after 18 h. Run several times a day if the pipeline allows; otherwise daily, with a client-side live fetch as a later option.
- **Fallback:** if the API fails, show a link to the NWS marine forecast page for the zone. Never show yesterday's text without a prominent "issued X h ago" label.
- **User-facing copy:** "National Weather Service marine forecast for *Coastal Waters from Point Reyes to Pigeon Point (out to 10 nm)*, issued 8:10 pm Mon. Small Craft Advisory in effect until …".
- **Do NOT claim:** that conditions are "safe"; any rewording of the forecast; any relationship to fishing quality or blooms.

### Rank 2: Live buoy, shore-gauge and tide data (NDBC + CO-OPS)

- **Endpoints**
  - NDBC `https://www.ndbc.noaa.gov/data/realtime2/{ID}.txt` for a curated list of about 35 CA buoys and C-MAN stations: 3-m buoys 46011/12/13/14/22/25/26/27/28/42/47/54/69/86, plus CDIP relays 46213/14/15/18/19/21/22/24/25/32/36/37/39/44/53/54/56/58/68/74/75/86. Optionally use `latest_obs.txt` as a single-file prefetch.
  - CO-OPS `datagetter` with `product=water_temperature` (`range=24`) for the 16 CA water-temperature stations.
  - CO-OPS `product=predictions&interval=hilo&datum=MLLW&range=72` for the 19 CA water-level stations (tide tables).
  - Always pass `application=CoastWatch`.
- **Variables:** NDBC WDIR, WSPD, GST, WVHT, DPD, MWD, WTMP, ATMP, PRES. CO-OPS `t, v, f` (flags) and `q`.
- **QC rules**
  - Parse `MM` as missing. For each field, take the **latest non-missing value within the last 3 h**. Rows alternate between wave and no-wave records, so the last row is often missing waves.
  - Range checks: WTMP 5–30 °C, WVHT 0–15 m, WSPD 0–40 m/s.
  - Spike check against the previous hour: drop if |ΔWTMP| > 3 °C.
  - Drop CO-OPS values whose flag string has any non-zero flag, and mark `q:"p"` as preliminary.
  - Mark a station *offline* if it has no valid value in 6 h. As of this check that applies to SF, LA, Redwood City, Martinez, 46053, 46235, 46251 and 46273.
- **Freshness:** live if < 2 h old, "stale" for 2–6 h, hidden after 6 h with an "offline" badge. Tide predictions are valid until their horizon.
- **Fallback:** keep the last good per-station JSON and mark it stale. NDBC `latest_obs.txt` can back up `realtime2`.
- **User-facing copy:** "Measured 2:40 pm at NOAA buoy 46026 (18 nm W of San Francisco): water 16.0 °C (61 °F), waves 1.8 m (6 ft) at 17 s from WNW, wind NNW 2 m/s (4 kt)." "Predicted tides (NOAA), La Jolla: low 2.1 ft 9:13 pm."
- **Do NOT claim:** that a buoy reading applies at the user's location (show the distance); that water temperature indicates fish presence; that tide predictions include storm surge (they are astronomical only).

### Rank 3: Daily satellite SST map (Geo-Polar Blended 5 km, ACSPO 2 km overlay, MUR fallback)

- **Endpoint (primary):**
  `https://coastwatch.noaa.gov/erddap/griddap/noaacwBLENDEDsstDNDaily.nc?analysed_sst[(last)][(32.0):1:(42.5)][(-126.0):1:(-116.5)],analysis_error[...]`
- **Overlay (optional):** `…/noaacwLEOACSPOSSTL3SnrtCDaily.nc?sea_surface_temperature[(last)][(32.0):1:(42.5)][(-126.0):1:(-116.5)],quality_level[...]`. Keep only `quality_level == 5`; this layer shows where the satellite actually saw the ocean.
- **Fallback:** `https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41` (0.01°) for the same box. This is a different ERDDAP host from the primary, which adds resilience.
- **Variables:** analysed SST (°C), `analysis_error`. Optionally compute an anomaly against `jplMURSST41anom1day` or a stored climatology. Keep it simple at first and show absolute SST.
- **QC rules**
  - Time must be ≤ 3 days old.
  - Valid-cell fraction over a fixed ocean mask must be ≥ 95 % for the L4 product.
  - Values must be within 5–30 °C.
  - Mask cells closer than 1 grid cell to land, and draw no colour inside bays for the 5 km product.
  - `analysis_error` > 1 K → hatch the cell.
- **Freshness:** green ≤ 2 d, amber 2–4 d, then switch to the fallback, then hide.
- **Output:** u16 grid (scale 0.01 °C) plus PNG, using the existing grid machinery.
- **User-facing copy:** "Sea surface temperature (satellite analysis, Oct 11). Gaps from clouds are filled by interpolation; near-shore and bay values are approximate."
- **Do NOT claim:** that this is a forecast; that this is the temperature at depth; that fronts mean fish; any temperature for harbors and bays from the 5 km product.

### Rank 4: Nearshore automated shore stations (CeNCOOS/SCCOOS)

- **Endpoint:** `https://erddap.cencoos.org/erddap/tabledap/{id}.csv?time,sea_water_temperature,sea_water_temperature_qc_agg,sea_water_practical_salinity,sea_water_practical_salinity_qc_agg,mass_concentration_of_chlorophyll_in_sea_water[_eco],…_qc_agg&time%3E%3Dnow-2days`. Note that `>=` **must be URL-encoded**: the server returned HTTP 500 for a raw `>=` because it proxies to `erddap.operational.sensors.axds.co`.
- **Station list:** Newport, Santa Monica (ECO variant), Scripps, Stearns, Humboldt Bay pier, Monterey wharf, Morro Bay, Cal Poly pier, BML intake, Monterey Bay Aquarium intake, Tiburon (currently stale).
- **Per-station variable map:** names differ by station (`_ctd`, `_eco` and plain suffixes, verified), so a config map is required.
- **QC rules**
  - Use only rows with `*_qc_agg == 1` (PASS).
  - Show `2` (not evaluated) as "unverified", or exclude it; Newport temperature was `2`/NaN at check time.
  - Drop rows flagged 3, 4 or 9.
  - Reduce to an hourly median.
  - Chlorophyll is in situ fluorescence in µg/L: biofouling causes drift, so use a 7-day median for display and flag a > 5× jump as "check".
- **Freshness:** live < 3 h, stale 3–24 h, offline > 24 h (Tiburon at 19 days must show as offline).
- **Fallback:** last good value marked stale. CO-OPS water temperature where co-located (Monterey, Santa Monica).
- **User-facing copy:** "Pier sensor at Humboldt Bay (Humboldt State / CeNCOOS), 9:45 pm: water 14.0 °C, salinity 33.5. Chlorophyll fluorescence 10 µg/L (a plant-pigment estimate, not a toxin test)."
- **Do NOT claim:** that chlorophyll or fluorescence means a harmful bloom or toxins; that values are lab-verified; that values apply away from the pier. Link to CDPH or the existing official-notice records for shellfish safety.

### Honourable mention for the next milestone

**HF radar resilience.** Add `https://dods.ndbc.noaa.gov/thredds/dodsC/hfradar_uswc_2km` as an automatic fallback to the current `ucsdHfrW2` source. It has the same variables, and it was 1 h fresher (04Z vs 03Z) while ERD was down. This is a small change that removes a verified single point of failure, but it is a robustness fix rather than new user value, which is why it is not ranked.

---

## 4. Live-check evidence log (UTC, 2026-10-13)

| Time | Request | Result |
|---|---|---|
| 05:09 | `jplMURSST41/index.json` | 200 |
| 05:10 | `jplMURSST41.csv?time[(last)]` | 200, 2026-10-11T09:00Z |
| 05:10 | `ucsdHfrW6/W2/W1 time[(last)]` | 200: 2026-10-13T00Z / 03Z / 02Z |
| 05:10 | `oceanview tabledap cciea_OC_CUTI` with time filter | 400 (monthly dataset; switched to griddap `erdCUTIdaily`) |
| 05:10 | CO-OPS MDAPI watertemp / waterlevels | 200 (16 / 19 CA stations) |
| 05:10 | CO-OPS 9414290 (SF) water_temperature latest | `No data was found`; also none for 2026-09-01→10-12 |
| 05:10 | CO-OPS 9410230 hilo predictions | 200, 7 events |
| 05:10 | NDBC latest_obs / realtime2 46026 / activestations.xml | 200 / 200 / 200 |
| 05:10 | `api.weather.gov/zones/forecast/PZZ545/forecast` | 200 but returned zone geometry; correct path is `/zones/coastal/PZZ545/forecast` (200, periods present) |
| 05:10 | `api.weather.gov/zones?type=coastal&area=CA` | 200 but empty FeatureCollection |
| ~05:12 | `ucsdHfrW6 time[(2026-09-12):1:(last)]` | **503 Service Unavailable** |
| ~05:12 | `hfrnet-tds.ucsd.edu`, `hfrnet.ucsd.edu` | connection failed (000) |
| ~05:13 | `dods.ndbc.noaa.gov/thredds/dodsC/hfradar_uswc_{6,2,1}km.das` | 200. `_6km` last time 1791864000 = 2026-10-13T04:00Z |
| ~05:13 | S3 `noaa-nos-ofs-pds` list `wcofs/netcdf/2026/10/13/` | 200; 165 objects, LastModified 04:03–04:12Z |
| ~05:14 | CO-OPS THREDDS WCOFS regulargrid f024 `.dds`/`.ascii` | 200 |
| 05:16 | ERD ERDDAP restart ("Startup was at 2026-10-12T22:16:38-07:00") | — |
| 05:19:48 | `ucsdHfrW2.csv?time[(last)]` ×3 | `Not Found: Currently unknown datasetID` |
| 05:22:49 | `ucsdHfrW6/W2` | still unknown; `wvcharmV3_0day` back (27 days since 09-12, gaps 09-17, 10-05, 10-08) |
| ~05:15 | CeNCOOS ERDDAP allDatasets | 200; 1,078 datasets, 255 with maxTime ≥ 2026-10-10 |
| ~05:15 | SCCOOS ERDDAP allDatasets | 200; 35 datasets |
| ~05:16 | gliders.ioos.us allDatasets in CA box | 200; 8 active |
| ~05:17 | CDFW ds582 FeatureServer `?f=json`, `returnCountOnly` | 200; 155 features; dataLastEditDate 2024-01-09 |
| ~05:17 | eCFR `titles.json` / `structure/…/title-50.json?part=660` | 200 / 200 |
| ~05:20 | CDIP `201p1_rt.nc` | 200; last waveTime 2026-10-13T04:30Z |
| ~05:21 | coastwatch.noaa.gov S3B OLCI sectors CI/CH | last 2026-10-01 |

| 05:26:17 | `ucsdHfrW6/W2 time[(2026-09-12):1:(last)]` | Back: W6 745 hourly steps and W2 748, **no gaps > 1.5 h in 30 days**; latest W6 00Z, W2 03Z. The ERD outage lasted about 10 min (LoadDatasets still running, 204 failed responses logged) |

**HF radar verdict:** over the 30-day window the *time axis* of the ERD-hosted HF radar products is complete. The observed failures were short server-side unavailability (503, a restart, then unknown-dataset errors for about 10 minutes), plus large hour-to-hour *spatial* gaps from individual radar site outages. Retry with backoff and use the NDBC THREDDS mirror as a fallback.

## 5. References

- NOAA Tech. Rep. NOS CO-OPS 097 (2022), *Implementation of WCOFS and the semi-operational nowcast/forecast skill assessment*: https://repository.library.noaa.gov/view/noaa/44306 (PDF read: §2.1 estuary caveat; §5 RMSE values; future-work list)
- WCOFS science page: https://coastaloceanmodels.noaa.gov/WCOFS/ ; CO-OPS WCOFS: https://tidesandcurrents.noaa.gov/ofs/wcofs/wcofs.html ; NODD: https://registry.opendata.aws/noaa-ofs/
- NWS API: https://www.weather.gov/documentation/services-web-api
- CO-OPS API: https://api.tidesandcurrents.noaa.gov/api/prod/ ; MDAPI: https://api.tidesandcurrents.noaa.gov/mdapi/prod/
- NDBC: https://www.ndbc.noaa.gov/faq/rt_data_access.shtml ; https://www.ndbc.noaa.gov/faq/measdes.shtml
- CDIP: https://cdip.ucsd.edu/
- IOOS HF radar: https://hfradar.ioos.us/ ; NDBC THREDDS: https://dods.ndbc.noaa.gov/thredds/hfradar.html
- MUR: https://podaac.jpl.nasa.gov/dataset/MUR-JPL-L4-GLOB-v4.1 ; Koutantou et al. 2023 https://doi.org/10.3390/rs15092277
- Geo-Polar Blended: https://coastwatch.noaa.gov/cwn/products/noaa-geo-polar-blended-global-sea-surface-temperature-analysis-level-4.html ; https://ospo.noaa.gov/products/ocean/sst/blended_sst_5km.html ; Maturi et al. 2017 https://doi.org/10.1175/BAMS-D-15-00002.1
- ACSPO L3S-LEO: https://ospo.noaa.gov/products/ocean/sst/acspo/L3S/ ; https://podaac.jpl.nasa.gov/dataset/L3S_LEO_DY-STAR-v2.81
- CUTI/BEUTI: https://oceanview.pfeg.noaa.gov/products/upwelling/intro ; Jacox et al. 2018 https://doi.org/10.1029/2018JC014187
- CeNCOOS ERDDAP: https://erddap.cencoos.org/erddap/ ; SCCOOS ERDDAP: https://erddap.sccoos.org/erddap/ ; IOOS gliders: https://gliders.ioos.us/
- CDFW MPA ds582: https://filelib.wildlife.ca.gov/Public/BDB/GIS/BIOS/metadata/DS0582.html ; https://services2.arcgis.com/Uq9r85Potqm3MfRV/arcgis/rest/services/biosds582_fpu/FeatureServer/0
- CDFW ds3143 / ds3144 / ds3160: https://filelib.wildlife.ca.gov/Public/BDB/GIS/BIOS/metadata/DS3143.html ; https://catalog.data.gov/dataset/rockfish-conservation-area-r7-cdfw-ds3144 ; https://filelib.wildlife.ca.gov/Public/BDB/GIS/BIOS/metadata/DS3160.html
- NOAA groundfish GIS: https://www.fisheries.noaa.gov/resource/map/groundfish-conservation-areas-maps-and-gis-data-west-coast-region
- CDFW regulations: https://wildlife.ca.gov/Fishing/Ocean/Regulations/Groundfish-Summary ; eCFR API: https://www.ecfr.gov/api/versioner/v1/titles.json
- OPC MPA petition recommendations (July 2026): https://opc.ca.gov/wp-content/uploads/2026/07/OPC-Recommendations-on-MPA-Petitions-July-2026-508.pdf
- Copernicus Marine licence: https://marine.copernicus.eu/user-corner/service-commitments-and-licence
- NCEI CRM: https://www.ncei.noaa.gov/products/coastal-relief-model ; GEBCO: https://www.gebco.net/data_and_products/gridded_bathymetry_data/
- CDPH biotoxin program: https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Marine-Biotoxin-Monitoring-Program.aspx

*Citation caveats:* Chin et al. 2017 (MUR), Anderson et al. 2016 (C-HARM) and Kim et al. 2008 (HFR) are cited from domain knowledge and were not re-fetched in this session. The 0.3 K Blended-SST figure comes from a NOAA poster, not a peer-reviewed paper. The MPA-petition timeline comes from news and advocacy sources plus the OPC PDF; confirm against fgc.ca.gov before relying on dates.
