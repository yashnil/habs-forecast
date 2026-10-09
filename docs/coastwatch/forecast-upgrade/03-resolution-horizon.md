# 3. Resolution and forecast-horizon comparison

| Layer | What it is | Native spacing | Cells across Monterey Bay (≈ 40 km) | Horizon | Typical age | Gaps | Skill evidence |
|---|---|---|---|---|---|---|---|
| OLCI 300 m (S3A) | Observation: chlorophyll-a | 0.0025° ≈ 250 m | ≈ 160 | none (observation) | 2 days | Severe in fog season | NASA/ESA algorithm validation; nearshore case-2 water is less reliable |
| PACE OCI L2 | Observation: chlorophyll-a, hyperspectral reflectance, phytoplankton optics | ≈ 1.2 km | ≈ 33 | none | hours | Clouds | NASA validation programme. Hyperspectral phytoplankton-type products are new. |
| VIIRS 750 m | Observation: chlorophyll-a composite | 0.0075° ≈ 750 m | ≈ 53 | none | 5 days | Clouds | Operational NOAA product |
| VIIRS 4 km | Observation: chlorophyll-a | 0.0375° ≈ 4 km | ≈ 10 | none | 3 days | Clouds | Operational |
| DINEOF 2 km | Statistically gap-filled chlorophyll-a | 0.021° ≈ 2 km | ≈ 19 | none | 12 days | none (filled) | Filled values are estimates |
| **C-HARM v3.1** | Model: probability of pDA > 500 ng/L, cDA > 10 pg/cell, PN > 10⁴ cells/L | 0.03° ≈ 3 km | ≈ 13 | **nowcast + 3 days** | 2 days | Nearshore mask (§2.3) | Anderson et al. 2016 (earlier version, 2014–15 pier data: DA models useful, PN model high false positives). No public v3/v3.1 validation found. |
| WCOFS | Model: currents, temperature, salinity, sea level | ≈ 4 km | ≈ 10 | **72 h**, hourly | hours | Nearshore and bays poorly resolved at 4 km | 4D-Var assimilation; NOAA skill assessment reports exist. I didn't extract numbers. |
| HF radar | Observation: surface currents | 6, 2, 1 km, 500 m | ≈ 20 at 2 km | none | ≈ 6 h | Partial: 45 % of the Monterey box at 2 km on 10-09 | Direct measurement |
| Copernicus PHY | Model: global currents, temperature, salinity | 1/12° ≈ 9 km | ≈ 4–5 | **10 days** | daily | Coastal processes under-resolved | Copernicus quality information documents (not reviewed here) |
| Copernicus BGC | Model: global chlorophyll, phytoplankton, nutrients | 1/4° ≈ 25 km | ≈ 1–2 | **10 days** | daily | Coastal upwelling under-resolved | Global validation only |

## Reading the table

- **Spatial detail and forecast horizon trade off.** The fine layers (OLCI, VIIRS 750 m, HF radar) are observations with no horizon. The layers with horizons beyond 3 days (Copernicus) are 9–25 km: too coarse to say anything about a bay or a pier.
- **No source offers a validated 7-day HAB forecast for California.** C-HARM's 3 days is the operational ceiling, and it inherits WCOFS's 72 hours.
- **The 4 km currents limit transport detail.** A 4 km current model cannot move a 250 m chlorophyll feature with 250 m accuracy. Advected high-resolution imagery looks precise but isn't. Drawing it at 300 m after 2–3 days of advection would fabricate precision.
- **Native resolution must be visible.** On the map, the native spacing of every layer is shown as a chip (`native 0.0025° (~250 m)`, `native 0.03° (~3 km)`), as in the map lab.
