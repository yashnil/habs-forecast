> **Evidence log.** Raw verification notes from live requests on 2026-10-08 (UTC times inside). Summarized in [`../02-data-sources.md`](../02-data-sources.md). Re-verify before relying on any value; sources change.

# Satellite ocean color + model-forcing data sources — live verification

**Verification window:** 2026-10-08 16:27–16:45 UTC (all checks were live HTTP requests with curl/python from this machine).
**Scope:** California coastal HAB decision-support app (`coastwatch-web`) and future model inference.
**Rule used:** if I did not get a live response for something, it is marked **UNVERIFIED**. Nothing below comes from memory alone.

Latency = 2026-10-08T16:30Z minus the last time step reported by the server.

---

## 0. TL;DR

| Role | Dataset | Server | Res | Last obs (verified) | Latency |
|---|---|---|---|---|---|
| **Primary (map + point values, fresh, nearshore)** | `noaacwN20VIIRSchlaSectorUSDaily` (NOAA-20 VIIRS, 750 m, US sector, NRT) | coastwatch.noaa.gov | 0.0075° | 2026-10-05T18:38:24Z | ~2.9 d |
| Primary companion (second satellite, same grid) | `noaacwN21VIIRSchlaSectorUSDaily` (NOAA-21) | coastwatch.noaa.gov | 0.0075° | 2026-10-05T18:36:59Z | ~2.9 d |
| **Fallback, different server (750 m, S-NPP, 1/3/8-day)** | `erdVHNchla1day` / `erdVHNchla3day` / `erdVHNchla8day` | coastwatch.pfeg.noaa.gov | 0.0075° | 10-04 / 10-03 / 10-01 | ~4 d |
| Gap-free field (clouds filled, coarse) | `noaacwNPPN20VIIRSDINEOFDaily` (NRT DINEOF, S-NPP+NOAA-20) | coastwatch.noaa.gov (mirror `nesdisVHNnoaaSNPPnoaa20NRTchlaGapfilledDaily` on pfeg) | ~0.0833° (9 km) | 2026-10-06T12:00Z | ~2.2 d |
| Global 4 km NRT (lightweight) | `noaacwN20VIIRSchlaDaily` = `nesdisVHNnoaa20chlaDaily` (pfeg mirror) | both | 0.0375° | 2026-10-04T12:00Z | ~4.2 d |
| Science-quality / retrospective | `noaacwNPPN20S3ASCIDINEOF2kmDaily` (gap-filled VIIRS+OLCI, 2 km), `noaacwNPPVIIRSSQchlaDaily` (S-NPP SQ 4 km) | both | 0.0208° / 0.0375° | 09-27 / 09-28 | ~10–11 d |

**Surprises:**
1. **GIBS problem that affects the current app.** For the newest date that GetCapabilities reports (`<Default>`), tiles come back as HTTP 500 errors or as broken, striped grayscale RGBA PNGs whose content changes between identical requests. Valid palette tiles start one day earlier. The app passes `<Default>` straight into the tile URL (`/api/gibs-chl-meta`), so it is probably showing broken tiles right now.
2. **The legend URLs hard-coded in `gibs.ts` return 404.** The GetCapabilities file lists different legends for these layers (see §3).
3. **Dead datasets:** `noaacwNPPN20VIIRSchlociDaily` (title says "2019-present", but it ends 2021-09-02); `erdVH2018chla*` (NASA S-NPP R2018 on pfeg, ends 2022-07); `erdVHNchlamday` (monthly, last 2026-06-16, so ~4 months stale).
4. **Rolling archives:** the NOAA-20/21 Sector-US datasets only go back to 2026-02, and the S-NPP NRT global daily dataset only to 2025-09-30. They are not suitable for long history.
5. **PACE OCI is not on either CoastWatch ERDDAP.** It is available from OB.DAAC/CMR (needs Earthdata Login) and as a GIBS layer.
6. **Aqua dependency:** the web app no longer uses MODIS-Aqua. However, the **trained model's inputs** (`log_chl`, `Kd_490`, `nflh` on the MODIS-Aqua 4 km grid; see README.md, `new_ds/*`) do. **Neither CoastWatch ERDDAP has a VIIRS FLH (nflh) product.** The only FLH found is Aqua MODIS (`erdMH1cflh*`). The README files still describe GIBS "MODIS Aqua L3S 8-day" tiles, which is out of date.
7. **Reliability:** coastwatch.noaa.gov ERDDAP returned **502 Proxy Error on every endpoint** from about 16:40Z during this session (see §6). pfeg returned intermittent 503s on some `time[last]` queries. The app needs to handle both servers failing, and probably cache server-side.

---

## 1. NOAA CoastWatch VIIRS chlorophyll-a on ERDDAP

### 1.1 Discovery

Searches I ran (both returned 200):
- `https://coastwatch.pfeg.noaa.gov/erddap/search/index.csv?searchFor=viirs%20chlorophyll&page=1&itemsPerPage=200`
- `https://coastwatch.noaa.gov/erddap/search/index.csv?searchFor=viirs%20chlorophyll&page=1&itemsPerPage=200`
- plus `noaa-21 chlorophyll`, `pace chlorophyll`, `oci pace`, `PACE`, `chlorophyll merged`, `Daily Merge chlora`, `fluorescence line height`, `GFS`

Chlorophyll dataset families found (Lon0360 duplicates and per-tile "Sector XX" sets left out):

| Family | coastwatch.noaa.gov ID | coastwatch.pfeg.noaa.gov ID | Notes |
|---|---|---|---|
| NOAA-20 NRT global 4 km daily | `noaacwN20VIIRSchlaDaily` | `nesdisVHNnoaa20chlaDaily` | identical data (same subset values) |
| NOAA-20 NRT global 4 km weekly | `noaacwN20VIIRSchlaWeekly` | `nesdisVHNnoaa20chlaWeekly` | daily time steps → rolling 7-day composite |
| S-NPP NRT global 4 km daily | `noaacwNPPVIIRSchlaDaily` | `nesdisVHNchlaDaily` | archive starts 2025-09-30 (rolling) |
| S-NPP NRT global 4 km weekly | `noaacwNPPVIIRSchlaWeekly` | `nesdisVHNchlaWeekly` | not inspected in detail |
| S-NPP Science Quality 4 km daily/weekly/monthly | `noaacwNPPVIIRSSQchlaDaily/Weekly/Monthly` | `nesdisVHNSQchlaDaily/Weekly/Monthly` | "15-day latency" per summary |
| **NOAA-20 750 m US sector daily merge (NRT)** | `noaacwN20VIIRSchlaSectorUSDaily` | — | covers CA; includes `l2_flags`, `qa_score` |
| **NOAA-21 750 m US sector daily merge (NRT)** | `noaacwN21VIIRSchlaSectorUSDaily` | — | **only NOAA-21 chl dataset found** on either server |
| S-NPP 750 m North Pacific 1/3/8-day/monthly | — | `erdVHNchla1day/3day/8day/mday` | ERD-hosted; summary says S-NPP, MSL12 NRT |
| S-NPP 750 m sector tiles (NRT + SCI) | `noaacwNPPVIIRS[SCI]chlaSector{UW..ZZ}Daily` | — | per-tile; not mapped to CA (Sector-US covers CA anyway) |
| S-NPP 750 m VV00 "Pacific Region" | `noaacwNPPVIIRSchlaDailyVV00` | — | **Hawaii region** (lat −20..34, lon 140..209). Not CA. |
| DINEOF gap-filled NRT 9 km (S-NPP+N20) | `noaacwNPPN20VIIRSDINEOFDaily` | `nesdisVHNnoaaSNPPnoaa20NRTchlaGapfilledDaily` | "EXPERIMENTAL" |
| DINEOF gap-filled SQ 9 km | `noaacwNPPN20VIIRSSCIDINEOFDaily` | `nesdisVHNnoaaSNPPnoaa20chlaGapfilledDaily` | last 2026-09-10 (~4 wk lag) |
| DINEOF gap-filled SQ 2 km VIIRS+OLCI S-3A | `noaacwNPPN20S3ASCIDINEOF2kmDaily` | `noaacwNPPN20S3ASCIDINEOF2kmDaily` | public-domain license string |
| Chl anomaly ratio/difference 2 km (N20, S-NPP; NRT+SCI) | `noaacwN20VIIRSchlanomratDaily` etc. | — | variable `chlor_a_pdif`; useful for "bloom anomaly" layer |
| OCI merged S-NPP+N20 4 km | `noaacwNPPN20VIIRSchlociDaily` | — | **DEAD: last 2021-09-02** |
| NASA R2018 S-NPP 4 km | — | `erdVH2018chla1day/8day/mday` | **DEAD: last 2022-07-25 (daily)** |
| OLCI S-3A/S-3B 300 m sectors | `noaacwS3[AB]OLCIchlaSector..Daily` | — | "90 days ago – present"; CA sector not identified (not checked) |
| Multi-sensor OLCI-VIIRS merge | `noaacwecnOLCImultisensorCHLeastcoast7Day` | — | **Chesapeake Bay only**, so not useful for CA |

There is no global or West-Coast "multi-sensor merged" VIIRS chlorophyll product apart from the DINEOF gap-filled ones and the NOAA-20/21 "Daily Merge" (which merges swaths, not sensors).

### 1.2 Shortlist, with verified metadata

The common license text for NOAA ERDDAP is: "The data may be used and redistributed for free but is not intended for legal use, since it may contain inaccuracies…". The NESDIS 4 km sets also cite the NASA Earth science data policy URL and ask for the acknowledgement "These data were provided by NOAA's Center for Satellite Applications and Research (STAR) and the CoastWatch program." No access constraints apply to any ERDDAP below: no key and no login.

#### A. `noaacwN20VIIRSchlaSectorUSDaily` (PRIMARY)

| Field | Value (verified) |
|---|---|
| Endpoint | `https://coastwatch.noaa.gov/erddap/griddap/noaacwN20VIIRSchlaSectorUSDaily` |
| Title | Daily Merge chlora from NOAA-20 VIIRS |
| Institution | NOAA NESDIS STAR |
| Resolution | 0.0075° lat × 0.0075° lon (~750 m); 8000 × 12000 grid; lat descending |
| Bounds | lat 0.00375 – 59.99625, lon −142.38375 – −52.39125 |
| Time | time_coverage_start 2026-02-01T19:34:24Z → **end 2026-10-05T18:38:24Z** (238 steps, ~1/day; timestamps are overpass times, not 12:00) |
| Last 7 steps | 2026-09-29 … 2026-10-05, daily with no gaps |
| Variables | `chlor_a` (mg m^-3), `l2_flags` (int), `qa_score` (double), `swath_latitude`, `swath_longitude`, `graphics` |
| Processing | NRT, L2→L3 swath merge via cwutils `cwcomposite` (history); `processing_level` attr not set |
| Cadence | daily; `date_created` 2026-10-06T05:39:22Z for the 10-05 file, so it posts ~11 h after the overpass |
| License | NOAA "may be used and redistributed for free…" boilerplate |
| Latency | **~2.9 days** at check time |
| WMS | yes. GetCapabilities 200 (layers `…:chlor_a`, `…:l2_flags`, …). GetMap PNG 200, 42 KB, renders CA coast correctly |
| Monterey subset | OK, 2214 cells, 599 valid, mean 1.21 mg m^-3 (0.46–9.69) |
| Status | VERIFIED 2026-10-08 16:33Z (server returned 502 from ~16:40Z, see §6) |
| Caveat | Archive is rolling (starts 2026-02-01), so it is not a climatology source. Daily single-sensor data has many cloud gaps. |

#### B. `noaacwN21VIIRSchlaSectorUSDaily` (PRIMARY companion / redundancy)

Same structure as A. Title "Daily Merge chlora from NOAA-21 VIIRS"; NOAA NESDIS STAR; 0.0075°; same bounds; time 2026-02-09T19:34:30Z → **2026-10-05T18:36:59Z** (231 steps); `date_created` 2026-10-06T07:30:28Z; same variables. Monterey subset OK: 529 valid of 2214, mean 1.58. WMS listed in search ("wms" column). Latency ~2.9 d. VERIFIED 16:33Z.
→ Taking the per-pixel nanmean or median of N20 and N21 gives more cloud-free coverage than either alone.

#### C. `erdVHNchla1day` / `erdVHNchla3day` / `erdVHNchla8day` (FALLBACK on a separate server)

| Field | Value (verified) |
|---|---|
| Endpoint | `https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdVHNchla{1day,3day,8day}` |
| Title | Chlorophyll a, North Pacific, NOAA VIIRS, 750m resolution, 2015-present (1 Day / 3 Day / 8 Day Composite) |
| Institution | NOAA NMFS SWFSC ERD |
| Sensor | Summary says VIIRS on **Suomi-NPP**, NOAA-MSL12 NRT processing, with flags LAND+CLDICE+HIGLINT+HISATZEN+… |
| Resolution | 0.0075° (11985 lat × 9338 lon), lat descending |
| Bounds | lat −0.10875 – 89.77125, lon −180.03375 – −110.00625 |
| Time | 1day: 2015-02-25 → **2026-10-04T12:00Z**; 3day: → **2026-10-03T12:00Z**; 8day: → **2026-10-01T00:00Z** |
| Variable | `chla` (mg m^-3); note the name differs from `chlor_a` |
| Cadence | daily (1-day has a gap on 09-27/28); 3- and 8-day are rolling composites with daily steps |
| Latency | ~4.2 d (1-day) |
| WMS | yes. 8day GetMap 200, 120 KB, good coverage of the CA coast |
| Monterey subset | 1day 1576/2200 valid, mean 1.82 (max 64.1); 8day 2026/2200 valid, mean 2.40 |
| Monthly | `erdVHNchlamday` **STALE: last 2026-06-16** |
| Status | VERIFIED 16:32–16:33Z (some `time[last]` calls hit 503 and succeeded on retry) |
| Caveat | Depends on S-NPP, the oldest VIIRS (launched 2011). It is still flowing today. |

#### D. `noaacwNPPN20VIIRSDINEOFDaily` (gap-filled, freshest)

| Field | Value |
|---|---|
| Endpoint | coastwatch.noaa.gov; pfeg mirror `nesdisVHNnoaaSNPPnoaa20NRTchlaGapfilledDaily` (same tce) |
| Title | Chlorophyll (Gap-filled DINEOF), NOAA S-NPP NOAA-20, VIIRS, Near Real-Time, Global 9km, 2020-present, Daily |
| Institution / level | NOAA NESDIS CoastWatch / L3 Mapped; summary says "EXPERIMENTAL" |
| Resolution | ~0.08333° (2160 × 4320) |
| Bounds | global, lat ±89.958 |
| Time | 2020-05-05 → **2026-10-06T12:00Z**; daily, last 7 consecutive; created 2026-10-07T21:15Z |
| Variable | `chlor_a` (mg m^-3) |
| License | NOAA boilerplate |
| Latency | **~2.2 d** |
| WMS | yes. GetMap 200. Visual check: smooth gap-free ocean, with artifacts over inland lakes |
| Monterey subset | 24/24 cells valid, mean 1.98 |
| Status | VERIFIED 16:32Z |
| Caveat | At 9 km resolution, Monterey Bay is only a handful of pixels, so it is too coarse for the nearshore. Use it for the regional context or as a model input with no cloud gaps. |

#### E. `noaacwN20VIIRSchlaDaily` (= pfeg `nesdisVHNnoaa20chlaDaily`), global 4 km NRT

Title "Chlorophyll, NOAA NOAA-20 VIIRS, Near Real-Time, Global 4km, Level 3, 2017-present, Daily". NOAA NESDIS CoastWatch. 0.0375° (4788 × 9602), global. **Note:** the title says 2017-present but time_coverage_start is 2021-08-26. Last **2026-10-04T12:00Z** (~4.2 d). `chlor_a` (mg m^-3). `processing_level` attribute is "L2" (sic). License is the NASA data policy URL plus NOAA boilerplate. Monterey subset: 59/96 valid, mean 2.83, and identical values on both servers. WMS yes. The weekly sibling (`…Weekly`) ends 2026-09-26 with daily steps (rolling 7-day). VERIFIED 16:32Z.

#### F. Science-quality / retrospective (for training and validation, not live display)

| ID (server) | Res | Coverage | Last | Notes |
|---|---|---|---|---|
| `noaacwNPPN20S3ASCIDINEOF2kmDaily` (both) | 0.02083° (8640×17280) | 2018-01-01 → **2026-09-27** | ~11 d lag | VIIRS S-NPP+N20 + Sentinel-3A OLCI, DINEOF L4, NOAA NESDIS STAR. License: "produced by NOAA and are not subject to copyright protection in the United States…" Monterey 306/320 valid, mean 2.45 |
| `noaacwNPPVIIRSSQchlaDaily` (= pfeg `nesdisVHNSQchlaDaily`) | 0.0375° | 2012-01-02 → **2026-09-28** | ~10 d | "science quality data with a 15-day latency", MSL12 v1.2 / OC-SDR v04. Monterey 71/108 valid, mean 1.84. Weekly → 2026-08-27; Monthly → 2026-08-01 |
| `noaacwNPPN20VIIRSSCIDINEOFDaily` (both) | 0.0833° | 2018-05-30 → **2026-09-10** | ~4 wk | SQ DINEOF |

### 1.3 ERDDAP request patterns (all run successfully)

griddap axis order is `[time][altitude][latitude][longitude]`. Altitude is a single 0.0 level. Latitude is **descending** in all NESDIS/ERD VIIRS sets, so request `(north):(south)`. Square brackets must be URL-encoded (`%5B` `%5D`). Unencoded brackets returned HTTP 400 from python urllib.

Template:
```
https://{host}/erddap/griddap/{datasetID}.{csv|json|nc|png}?{var}%5B{(ISO)|last}%5D%5B(0.0)%5D%5B({latN}):({latS})%5D%5B({lonW}):({lonE})%5D
```
Run successfully (Monterey Bay, latest step):
```
https://coastwatch.noaa.gov/erddap/griddap/noaacwN20VIIRSchlaSectorUSDaily.csv?chlor_a%5Blast%5D%5B(0.0)%5D%5B(36.9):(36.6)%5D%5B(-122.2):(-121.8)%5D
https://coastwatch.noaa.gov/erddap/griddap/noaacwN21VIIRSchlaSectorUSDaily.csv?chlor_a%5Blast%5D%5B(0.0)%5D%5B(36.9):(36.6)%5D%5B(-122.2):(-121.8)%5D
https://coastwatch.noaa.gov/erddap/griddap/noaacwN20VIIRSchlaDaily.csv?chlor_a%5Blast%5D%5B(0.0)%5D%5B(36.9):(36.6)%5D%5B(-122.2):(-121.8)%5D
https://coastwatch.pfeg.noaa.gov/erddap/griddap/nesdisVHNnoaa20chlaDaily.csv?chlor_a%5Blast%5D%5B(0.0)%5D%5B(36.9):(36.6)%5D%5B(-122.2):(-121.8)%5D
https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdVHNchla1day.csv?chla%5Blast%5D%5B(0.0)%5D%5B(36.9):(36.6)%5D%5B(-122.2):(-121.8)%5D
https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdVHNchla8day.csv?chla%5Blast%5D%5B(0.0)%5D%5B(36.9):(36.6)%5D%5B(-122.2):(-121.8)%5D
https://coastwatch.noaa.gov/erddap/griddap/noaacwNPPN20VIIRSDINEOFDaily.csv?chlor_a%5Blast%5D%5B(0.0)%5D%5B(36.9):(36.6)%5D%5B(-122.2):(-121.8)%5D
https://coastwatch.noaa.gov/erddap/griddap/noaacwNPPN20S3ASCIDINEOF2kmDaily.csv?chlor_a%5Blast%5D%5B(0.0)%5D%5B(36.9):(36.6)%5D%5B(-122.2):(-121.8)%5D
https://coastwatch.noaa.gov/erddap/griddap/noaacwNPPVIIRSSQchlaDaily.csv?chlor_a%5Blast%5D%5B(0.0)%5D%5B(36.9):(36.6)%5D%5B(-122.2):(-121.8)%5D
https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWchla8day.csv?chlorophyll%5Blast%5D%5B(0.0)%5D%5B(36.6):(36.9)%5D%5B(237.8):(238.2)%5D   (Aqua; lat ascending, lon 0-360)
```
JSON with explicit time (200):
```
https://coastwatch.noaa.gov/erddap/griddap/noaacwN20VIIRSchlaSectorUSDaily.json?chlor_a%5B(2026-10-05T18:38:24Z)%5D%5B(0.0)%5D%5B(36.80):(36.79)%5D%5B(-122.00):(-121.99)%5D
```
Latest time only (cheap freshness probe):
```
https://{host}/erddap/griddap/{id}.csv?time%5Blast%5D
https://{host}/erddap/griddap/{id}.csv?time%5Blast-6:1:last%5D
```
Metadata: `https://{host}/erddap/info/{id}/index.json` (or `.csv`).

### 1.4 ERDDAP WMS (map tiles)

All shortlisted datasets advertise WMS. Verified:
```
https://coastwatch.noaa.gov/erddap/wms/noaacwN20VIIRSchlaSectorUSDaily/request?service=WMS&request=GetCapabilities&version=1.3.0      → 200
https://coastwatch.noaa.gov/erddap/wms/noaacwN20VIIRSchlaSectorUSDaily/request?service=WMS&version=1.3.0&request=GetMap&bbox=32,-125,42,-117&crs=EPSG:4326&width=400&height=500&bgcolor=0x808080&layers=Land,noaacwN20VIIRSchlaSectorUSDaily:chlor_a,Coastlines&styles=&format=image/png&time=2026-10-05T18:38:24Z   → 200 image/png
https://coastwatch.pfeg.noaa.gov/erddap/wms/erdVHNchla8day/request?...layers=Land,erdVHNchla8day:chla,Coastlines...&time=2026-10-01T00:00:00Z   → 200 image/png
https://coastwatch.noaa.gov/erddap/wms/noaacwNPPN20VIIRSDINEOFDaily/request?...layers=...:chlor_a...&time=2026-10-06T12:00:00Z   → 200 image/png
```
Layer name format: `{datasetID}:{variable}`; the extra layers `Land`, `Coastlines`, `LakesAndRivers`, `Nations` are available. I only tested EPSG:4326. **UNVERIFIED:** EPSG:3857 GetMap support, which Mapbox raster sources need (`bbox={bbox-epsg-3857}`). Test it before relying on it. ERDDAP renders WMS images on demand, so it will be much slower than GIBS's pre-rendered tiles. Put a cache or proxy in front of it.

### 1.5 MODIS-Aqua datasets still live on pfeg (for awareness)

- `erdMWchla1day` (Aqua MODIS, 0.0125°, West US, EXPERIMENTAL): last **2026-10-05**, `chlorophyll` (mg m-3), lon 0–360, lat ascending.
- `erdMWchla8day`: last 2026-10-02.
- `erdMH1chla1day_R2022NRT` (NASA OBPG 4 km global NRT): last 2026-10-05; `erdMH1chla8day_R2022NRT` tce 2026-09-26.
- `erdMH1cflh{1day,8day,mday}_R2022{NRT,SQ}`: **these are the only FLH (nflh) products found on either server**, and they are Aqua.

These still work today, but they all depend on MODIS-Aqua. Treat them as legacy and continuity-only inputs.

---

## 2. NASA PACE OCI chlorophyll

| Item | Result |
|---|---|
| CoastWatch ERDDAP (both) | **Not present.** Searches for `PACE`, `pace chlorophyll`, `oci pace` returned no PACE OCI datasets (pfeg only matched unrelated HadISST; noaa.gov only an eddy product). |
| NASA CMR collections (public, no auth) | `PACE_OCI_L3M_BGC_NRT` v3.2 (`C4184125829-OB_CLOUD`), `PACE_OCI_L3M_BGC` v3.2 (`C4184125847-OB_CLOUD`), `PACE_OCI_L2_BGC_NRT` v3.2 (`C4124887054-OB_CLOUD`), `PACE_OCI_L2_BGC` v3.2, plus AOP/IOP variants. Start 2024-03-05. |
| Latest L3M NRT granule | `PACE_OCI.20261007.L3m.DAY.BGC.V3_2.4km.NRT.nc` (also `0p1deg`), updated 2026-10-08T13:19Z, so **~1 day latency** for daily L3 mapped |
| Latest L2 NRT granule | `PACE_OCI.20261008T143914.L2.OC_BGC.V3_2.NRT.nc`, ingested 15:59Z (about 80 min after acquisition) |
| Download | `https://obdaac-tea.earthdatacloud.nasa.gov/ob-cumulus-prod-public/<file>` → **302 to urs.earthdata.nasa.gov** (Earthdata Login required, free). `https://oceandata.sci.gsfc.nasa.gov/getfile/<file>` → 302 to its URS login. |
| Query used | `https://cmr.earthdata.nasa.gov/search/granules.json?collection_concept_id=C4184125829-OB_CLOUD&sort_key=-start_date&page_size=200` |
| Variable list inside BGC files | **UNVERIFIED** (I could not open a file without credentials). It is likely to include `chlor_a`, but whether it includes `nflh` is not confirmed. |
| GIBS layer | `OCI_PACE_Chlorophyll_a` exists (see §3) |

---

## 3. NASA GIBS layers used by the app

App code (read only, not modified):
- `coastwatch-web/src/lib/gibs.ts`: `GIBS_SATELLITE.layerId = "VIIRS_NOAA20_Chlorophyll_a"`, `GIBS_PACE_CHL.layerId = "OCI_PACE_Chlorophyll_a"`, `tileMatrixSet = "GoogleMapsCompatible_Level7"`, caps URL `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/1.0.0/WMTSCapabilities.xml`, legend URLs `…/legends/VIIRS_NOAA20_Chlorophyll_a_H.svg` and `…/legends/OCI_PACE_Chlorophyll_a_H.svg`. Tile template `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/{layer}/default/{YYYY-MM-DD}/{TMS}/{z}/{y}/{x}.png`.
- `coastwatch-web/src/app/api/gibs-chl-meta/route.ts`: fetches the caps, parses `<Default>` per layer, and returns it as `viirsDate`/`paceDate` (fallback is today−2).
- `coastwatch-web/src/lib/geo.ts`: only a deprecated `gibsChlDate()` helper; no layer IDs.
- No MODIS-Aqua layer is referenced in `src/` (the comment in gibs.ts confirms the MODIS Aqua L3S 8-day layer was dropped). **However**, `README.md` and `coastwatch-web/README.md` still say the app uses "MODIS Aqua L3S 8-day" GIBS tiles. That documentation is out of date.

GetCapabilities (both the KVP `wmts.cgi?SERVICE=WMTS&REQUEST=GetCapabilities` and the REST `1.0.0/WMTSCapabilities.xml` returned 200, 5.84 MB) at 16:28Z:

| Layer | Exists | Default | Last time interval | TMS | Legend in caps |
|---|---|---|---|---|---|
| `VIIRS_NOAA20_Chlorophyll_a` (app) | yes | 2026-10-07 | 2018-02-24/2026-10-07/P1D | GoogleMapsCompatible_Level7 | `legends/VIIRS_Chlorophyll_H.svg` |
| `OCI_PACE_Chlorophyll_a` (app) | yes | 2026-10-08 | 2026-03-21/2026-10-08/P1D (gaps earlier in 2024–26) | GoogleMapsCompatible_Level7 | `legends/MODIS_Chlorophyll_H.svg` |
| `VIIRS_NOAA21_Chlorophyll_a` | yes | 2026-10-07 | 2025-10-25/2026-10-07 (gap 2025-10-24) | L7 | VIIRS_Chlorophyll |
| `VIIRS_SNPP_L2_Chlorophyll_A` | yes | 2026-10-07 | 2026-07-16/2026-10-07 (several 2024–26 gaps) | L7 | VIIRS_Chlorophyll |
| `S3A_OLCI_Chlorophyll_a` / `S3B_OLCI_Chlorophyll_a` | yes | 2026-10-07 | …/2026-10-07 | L7 | VIIRS_Chlorophyll |
| `MODIS_Aqua_L2_Chlorophyll_A` | yes (Aqua) | 2026-10-08 | 2025-12-29/2026-10-08 | L7 | MODIS_Chlorophyll |
| `MODIS_Terra_L2_Chlorophyll_A` | yes | 2026-10-08 | | L7 | |
| MERIS, SeaWiFS GAC/MLAC | historical only | | ends 2012 / 2010 | | |

No MODIS-Aqua L3 chlorophyll layer exists in `epsg3857/best`. Only the L2 swath layer is left.

**Legend check:**
- `…/legends/VIIRS_NOAA20_Chlorophyll_a_H.svg` → **404**
- `…/legends/OCI_PACE_Chlorophyll_a_H.svg` → **404**
- `…/legends/VIIRS_Chlorophyll_H.svg` → 200
- `…/legends/MODIS_Chlorophyll_H.svg` → 200

**Tile health check:** tiles 6/24/9, 6/24/10 and 6/25/10 cover central California. A good tile is a palette-mode ("P") PNG with about 100–200 colours. A bad tile is an HTTP 500 or an RGBA PNG with more than 15,000 colours and a striped grayscale look.

| Layer | Date | Result (16:28–16:38Z) |
|---|---|---|
| VIIRS_NOAA20 | 2026-10-08 | 404 (not in caps, as expected) |
| VIIRS_NOAA20 | **2026-10-07 (= Default)** | 500 on some tiles. Others return 200 but the RGBA images are **garbage**, and sizes vary between identical requests (141 KB, 106 KB, 96 KB, 32 KB, 151 KB, 67 KB) |
| VIIRS_NOAA20 | 2026-10-06, 10-05, 10-04 | 200, valid palette tiles, stable across repeats |
| OCI_PACE | **2026-10-08 (= Default)** | 500 on 2 of 2 tiles at 16:30Z; earlier a 200 with garbage RGBA |
| OCI_PACE | 2026-10-07, 10-06, 10-05 | 200, valid palette tiles |
| VIIRS_NOAA21 | 2026-10-07 (Default) | mix of 500 and a non-palette "LA" image; 10-06 valid |
| S3A_OLCI | 2026-10-07 (Default) | garbage RGBA; 10-06 valid |

The same pattern showed up again on a recheck at 16:38Z. **The newest advertised date is not safe to render.** Recommendation for `/api/gibs-chl-meta`: use `Default − 1 day`. Alternatively, probe one known-ocean tile and check that it is palette-mode or under ~40 KB before using a date. Fix the legend URLs too.

**Aqua note:** the GIBS layers the app uses (NOAA-20 VIIRS, PACE OCI) do not depend on Aqua. The Aqua dependency is in the **model**: README.md says the grid is "the MODIS-Aqua Level-3 mapped grid" and the inputs are "MODIS-Aqua L3 (NASA OB.DAAC / Earthdata) `log_chl`, `Kd_490`, `nflh`" (also `new_ds/*.py`, `docs/coastwatch/01-codebase-audit.md`). Replacement options for operations: VIIRS chl (above), VIIRS Kd490 (`noaacwNPPN20VIIRSkd490Daily`, found by search but **metadata UNVERIFIED** because the server returned 502 when I queried it), and for nflh possibly PACE OCI (**UNVERIFIED**).

---

## 4. Model-forcing sources (brief)

### 4.1 Copernicus Marine — Global Ocean Physics Analysis and Forecast

| Field | Value (verified 16:34–16:36Z) |
|---|---|
| Product ID | `GLOBAL_ANALYSISFORECAST_PHY_001_024` ("Global Ocean Physics Analysis and Forecast"), Level 4 |
| STAC | `https://stac.marine.copernicus.eu/metadata/GLOBAL_ANALYSISFORECAST_PHY_001_024/product.stac.json` → 200; temporal interval 2019-01-01 → **2026-10-18** (so ~10-day forecast horizon); license field "proprietary" (Copernicus Marine service terms) |
| Datasets (daily means, 1/12°) | `cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_202406` (`uo`, `vo` m s-1); `cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m_202406` (`thetao` °C); `cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m_202406` (`so` 1e-3); `cmems_mod_glo_phy_anfc_0.083deg_P1D-m_202406` (`zos` m, `mlotst` m, `tob`, `sob`, `pbo`, sea-ice vars). All have time extent 2022-06-01 → 2026-10-17. Also 6-hourly (`PT6H-i`) cur/thetao/so, hourly merged surface currents `cmems_mod_glo_phy_anfc_merged-uv_PT1H-i_202211`, and hourly sea level `…merged-sl_PT1H-i_202411`. |
| Access | `copernicusmarine` Python toolbox (`copernicusmarine.login(...)`, `subset`/`open_dataset`). The official toolbox docs (`toolbox-docs.marine.copernicus.eu/en/stable/usage/quickoverview.html`, fetched 200) say: "To register, you can obtain credentials for free by creating an account at Copernicus Marine website." **So a free account is required.** |
| Other observed endpoints | ARCO Zarr `https://s3.waw3-1.cloudferro.com/mdl-arco-time-007/arco/GLOBAL_ANALYSISFORECAST_PHY_001_024/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_202406/timeChunked.zarr/.zmetadata` → 200 anonymously (technically readable without login; terms of use still apply). WMTS `https://wmts.marine.copernicus.eu/teroWmts/GLOBAL_ANALYSISFORECAST_PHY_001_024/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_202406?service=WMTS&request=GetCapabilities` → 200 |
| Not done | I did not run a toolbox subset (no credentials here). **UNVERIFIED:** the actual latest analysis day. |

### 4.2 ERA5 / ERA5T latency

CDS catalogue `https://cds.climate.copernicus.eu/api/catalogue/v1/collections/reanalysis-era5-single-levels` → 200. Temporal extent ends **2026-10-02T00:00Z** (~6.7 days before the check). The description says "ERA5 is updated daily with a latency of about 5 days … (called ERA5T) … could be different from the final release 2 to 3 months later." CDS needs a free account and API key (standard CDS behaviour; I did not test the download endpoint).

### 4.3 NOAA GFS (near-real-time winds, radiation, precip)

| Route | Verified |
|---|---|
| NOMADS directory | `https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.20261008/` → 200 (needed `--http1.1` and a User-Agent; HTTP/2 without a UA returned an empty 200 body). Cycles 00/06/12 present at 16:36Z. `gfs.t12z.pgrb2.0p25.f000.idx` → 200. Grib filter `https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl` → 200. |
| AWS Open Data (no auth) | `https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20261008/12/atmos/gfs.t12z.pgrb2.0p25.f000.idx` → 200. Contains `UGRD:10 m above ground`, `PRATE:surface`. Byte-range GRIB2 access is possible with the .idx. |
| ERDDAP | `https://coastwatch.pfeg.noaa.gov/erddap/griddap/NCEP_Global_Best` → **302 redirect to PacIOOS** `https://pae-paha.pacioos.hawaii.edu/erddap/griddap/ncep_global`. 0.5°, 3-hourly, 2022-12-01 → **2026-10-15T15:00Z** (forecast to +7 d). Vars `ugrd10m`, `vgrd10m` (m s-1), `pratesfc`, `tmp2m`, `tmpsfc`, `rh2m`, `prmslmsl`, `dswrfsfc`, `dlwrfsfc`, `uswrfsfc`, `ulwrfsfc`. Point query succeeded (follow redirects): `https://coastwatch.pfeg.noaa.gov/erddap/griddap/NCEP_Global_Best.csv?ugrd10m%5B(2026-10-08T12:00:00Z)%5D%5B(37.0)%5D%5B(238.0)%5D,vgrd10m%5B…%5D,dswrfsfc%5B…%5D` → u=−0.15, v=−1.13 m/s, dswrf=0 (night). Lon is 0–360. |
| Also seen | `pifscCcmpDailyV21NRT` (CCMP NRT ocean surface winds, 6-hourly) on pfeg. Not inspected. |

### 4.4 NWS API (api.weather.gov), sent with a User-Agent header

| Endpoint | Result |
|---|---|
| `https://api.weather.gov/zones?type=marine&area=PZ` | 200, 68 PZ marine zones. CA examples: `PZZ535` Monterey Bay; `PZZ530` (SF Bay, forecast verified); `PZZ540/545/560` coastal out to 10 nm; `PZZ570/571/575` 10–60 nm; `PZZ650`, `PZZ673`, `PZZ676` (SoCal); `PZZ455/475` (Cape Mendocino–Pt Arena) |
| `https://api.weather.gov/zones/forecast/PZZ535/forecast` | 200, updated 2026-10-08T08:48-07:00, 10 periods ("NW wind around 5 kt… Seas 3 to 4 ft…") |
| `https://api.weather.gov/zones/marine/PZZ535/forecast` | 200 |
| `https://api.weather.gov/zones/marine/PZZ535`, `/zones/forecast/PZZ535` | 200 |
| `https://api.weather.gov/alerts/active?area=PZ` | 200, 27 active alerts (Gale Watch, Dense Fog Advisory… mostly WA at check time) |

Free, no key. NWS asks for an identifying User-Agent.

---

## 5. Recommended architecture (data-source side)

1. **Live map chlorophyll:** keep GIBS for fast pre-rendered tiles, but fix the date logic (`Default − 1`, or probe and validate) and the legend URLs. Offer the NOAA-20 / NOAA-21 / PACE layers so one cloudy sensor can be swapped for another.
2. **Point values / zone summaries:** ERDDAP griddap on `noaacwN20VIIRSchlaSectorUSDaily` + `noaacwN21VIIRSchlaSectorUSDaily` (750 m, nanmean). If that fails, use pfeg `erdVHNchla1day` / `erdVHNchla3day` (750 m, different server). Use `noaacwNPPN20VIIRSDINEOFDaily` (or the pfeg mirror) when a gap-free value is required, and label it "gap-filled, 9 km".
3. **History / climatology / anomaly:** `erdVHNchla*` (2015→) or the SQ products. Do not use the Sector-US sets, because their archives are rolling.
4. **Model inputs:** plan the MODIS-Aqua → VIIRS/PACE transition. nflh has no VIIRS equivalent on CoastWatch.

## 6. Reliability observations during this session

- coastwatch.noaa.gov ERDDAP: all calls OK 16:27–16:34Z. From about **16:40Z, every request (index.html, info, griddap) returned 502 Proxy Error**. Re-check result is in the addendum below.
- coastwatch.pfeg.noaa.gov ERDDAP: intermittent **503** on `time[last]` queries for `erdVHNchla8day` and `erdMH1chla8day_R2022NRT`. Retries succeeded for erdVHNchla8day.
- GIBS: HTTP 500s and corrupt tiles only on the newest date; older dates were stable.
- NOMADS: returned an empty 200 body over HTTP/2 without a User-Agent; worked over HTTP/1.1 with a UA.

### Addendum: re-check at 2026-10-08 16:44:30Z
- `https://coastwatch.noaa.gov/erddap/index.html` → still **502 Proxy Error**. The outage lasted at least ~5 minutes (16:40 to 16:44Z).
- `noaacwNPPN20VIIRSkd490Daily` → 502, so its metadata, variable name and latency remain **UNVERIFIED**. (The `kd_490` variable name was a guess and is not confirmed.)
- Everything on coastwatch.noaa.gov listed as VERIFIED above was confirmed between 16:27 and 16:34Z, before the outage.
