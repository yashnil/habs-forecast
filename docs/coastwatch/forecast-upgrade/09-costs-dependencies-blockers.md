# 9. Costs, API dependencies and blockers

**How each figure was obtained**
- *Measured* figures come from the tests in this folder (2026-10-09).
- *Estimated* figures are calculations from measured grid sizes, labelled as such.

## 9.1 Per-day data transfer and processing

| Source | Request | Bytes per day | Time | Basis |
|---|---|---|---|---|
| OLCI 300 m, statewide (sectors CI + DI, S3A) | 1 overpass, chlorophyll only | ≈ 45 MB (uncompressed NetCDF from ERDDAP; 4 B × ~11 M cells) | ≈ 30–60 s | Estimated from measured regional sizes (Monterey 0.47 MB, SoCal 1.85 MB) |
| OLCI 300 m, S3B fallback | same | ≈ 45 MB | ≈ 30–60 s | Estimated |
| VIIRS 750 m statewide | 1 day | ≈ 7 MB | ≈ 10 s | Estimated |
| HF radar 2 km, California, 24 hourly fields | u, v | ≈ 50 MB (or ≈ 6 MB at 6 km) | ≈ 30 s | Estimated |
| WCOFS surface, 12 steps (+3 … +72 h) | regulargrid, surface level by range reads | **≈ 0.7 GB** (57 MB of range reads per step) | **≈ 2.5 min** (12 s per step) | Measured per step |
| C-HARM (unchanged) | 12 grids | ≈ 2 MB | seconds | Measured (M1) |
| **Total** | | **≈ 0.85–0.9 GB/day** | **≈ 5 min/run** | |

- **Compute:** GitHub-hosted runners are free for public repositories. The existing pipeline already runs every 6 hours, and the new sources only need to run once a day (WCOFS and OLCI update daily).
- **Egress costs:** ERDDAP is free, and the AWS Open Data bucket costs nothing to read.

## 9.2 Published artifacts and hosting

| Artifact | Size | Basis |
|---|---|---|
| Latest clear-view chlorophyll PMTiles, zoom 5–10, statewide | ≈ 50–150 MB | Estimated (coastal tiles only, 20–60 KB PNG per tile) |
| Age layer for the composite | ≈ 10–30 MB | Estimated |
| u16 value chunks for the readout, 6 regions | ≈ 5–10 MB | Estimated |
| Currents textures (12 steps) + arrows | ≈ 5 MB | Estimated from the lab (2 MB per step as NetCDF before quantization) |
| 14-day rolling satellite days (single-day layers for the slider) | ≈ 0.7–2 GB | Estimated |

**GitHub Pages limits:**
- **Size:** published sites are limited to 1 GB.
- **Bandwidth:** soft limit of 100 GB a month.

The latest composite plus currents fit. A 14-day rolling archive of 300 m days does not.

**Recommendation:** keep the manifest and small artifacts on Pages. Put PMTiles and the rolling days in **Cloudflare R2** (S3 API, no egress fees). At about 2 GB stored that is within R2's free 10 GB, so roughly **$0/month** at today's scale. This would be a production-infrastructure change and needs your approval first.

## 9.3 API dependencies and licences

| Dependency | Authentication | Licence / terms | Risk |
|---|---|---|---|
| NOAA CoastWatch West Coast ERDDAP (C-HARM, VIIRS, HF radar) | none | Free use and redistribution; "not intended for legal use" | Bursts of HTTP 503 and 502 (about 3 % of requests in testing); single host |
| NOAA CoastWatch central ERDDAP (OLCI 300 m) | none | NOAA open data; Sentinel-3 data © Copernicus/EUMETSAT, free and open | Search endpoint returned HTTP 502 throughout; data endpoints fine. **90-day rolling window.** |
| NOAA OFS on AWS (WCOFS) | none | NOAA Open Data Dissemination | Large files; layout changed Jan 2025 (`YYYY/MM/DD`) |
| NASA Earthdata (PACE) | **Earthdata Login (free); token stored as a CI secret** | NASA open data | Account needed. User tokens expire (60 days at the time of writing) and need rotating. |
| Copernicus Marine (10-day physics/BGC, 300 m OLCI archive) | **Free account; `copernicusmarine` toolbox credentials** | Copernicus Marine Service licence: free, attribution required | Account needed; coarse for coastal use |
| NASA GIBS (imagery) | none | NASA open data | Pictures only; no values |
| OpenFreeMap tiles | none | ODbL / OpenMapTiles | Already used |

## 9.4 Blockers and open questions

| # | Blocker | Who can resolve it | Effect |
|---|---|---|---|
| 1 | **No validated 7-day HAB forecast exists for California.** | Science, not engineering | The 7-day goal cannot be met honestly with existing operational products. Days 4–7 stay "no forecast". |
| 2 | **PACE needs a NASA Earthdata account** (download returned HTTP 401). | You: create an account and add a token as a repository secret | Without it, no PACE values (GIBS imagery only) |
| 3 | **Copernicus Marine needs an account.** | You | Without it, no 10-day physics and no OLCI 300 m archive for training |
| 4 | OLCI 300 m on NOAA ERDDAP keeps only 90 days | Archive elsewhere (EUMETSAT/Copernicus/OB.DAAC) | Training features from OLCI need blocker 3 or an Earthdata account |
| 5 | C-HARM version labelling: the SCCOOS bulletin says "v4"; the data say `product_version 3.1` | NOAA CoastWatch / SCCOOS (Clarissa Anderson's group) | CoastWatch must label the version the data declare (3.1) and should ask before claiming v4 |
| 6 | No public skill numbers for C-HARM v3.x | Same team, or CoastWatch's own backtest (§6) | Product page cannot quote skill |
| 7 | Fog-season gaps in high-resolution imagery (Monterey: 1 usable OLCI day in 14) | Physics; mitigated by clear-view composites with age | The 300 m layer will often be days old in the central and northern coast |
| 8 | Hosting > 1 GB for rolling 300 m days | Your approval for R2 (or keep only the latest composite on Pages) | Phase 2 design |
| 9 | Independent HAB-scientist review | Still pending from M1–M3 | Required before any experimental forecast is public |
