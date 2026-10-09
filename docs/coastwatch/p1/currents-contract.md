# Ocean currents: data contract and readiness (P1 phase D)

Status:
- **contract defined;**
- **proof of concept run;**
- **not published;**
- **the map shows the group as "Next phase", disabled, drawing nothing.**

## 1. Sources

| Source | Kind | Endpoint | Grid | Cadence and horizon | Access |
|---|---|---|---|---|---|
| NOAA WCOFS | Model forecast (ROMS, 4D-Var assimilation of HF radar, SST and SSH) | `noaa-nos-ofs-pds.s3.amazonaws.com/wcofs/netcdf/YYYY/MM/DD/wcofs.t03z.YYYYMMDD.regulargrid.fHHH.nc` | 0.04° regular grid (model about 4 km curvilinear), 41 depths | One run a day (t03z, files about 04:00–06:00 UTC); 3-hourly regular-grid steps to +72 h | Open, anonymous S3 |
| HF radar (IOOS / UCSD) | Observation | `coastwatch.pfeg.noaa.gov/erddap/griddap/ucsdHfrW{6,2,1,500}` | 6 km / 2 km / 1 km / 500 m | Hourly. Lag varied from about 6 h to about 15 h on 2026-10-09 | Open ERDDAP |

## 2. Contract (in `pipeline/coastwatch_pipeline/models.py`)

A currents layer is an ordinary `LayerArtifact` with `group_id = "currents"`. It also carries:

- **`product_class`**: `official_forecast` for WCOFS (a NOAA operational forecast), `observation` for HF radar.
- **`vectors: VectorField`**:
  - `u_grid`, `v_grid`: `ValueGrid`, eastward and northward velocity in m s⁻¹, quantized over a fixed −2.5…2.5 range;
  - `depth_m`: 0 for surface;
  - `speed_max`;
  - `arrows_url`: thinned arrows GeoJSON for static display;
  - `texture`: optional `RasterImage` with u and v packed into the R and G channels, for a WebGL particle layer.
- **`time`**:
  - WCOFS: `issued_date` (run), `valid_time` (step) and `lead_days`;
  - HF radar: `observed_date` and `valid_time`.
- **`native_resolution_m`**: 4000 (WCOFS) or 2000 (HF radar 2 km), shown as the native-resolution chip.
- **`freshness`**:
  - WCOFS: issued_date basis, current ≤ 1 day, stale ≤ 3 days;
  - HF radar: observed_date basis, current ≤ 1 day, stale ≤ 3 days.
- **`caveats`** must include:
  - "Surface water movement. Not where a bloom will be, not bloom growth, not toxin."
  - "Model currents are least reliable inside bays and near the shore."
  - HF radar: "Coverage varies hour to hour; gaps are not calm water."

**Front-end rules (the `LayerDock` "Ocean currents" group):**
- The tab stays disabled until a manifest lists `group_id = "currents"` layers.
- Arrows and particles never use the forecast or chlorophyll colour ramps. Speed is shown by line opacity and width only.
- Particles never move beyond the forecast's last valid time. A particle stops at land or at the grid edge, with no extrapolation.
- Drift paths, if ever shown, carry "where surface water may move · not where a bloom will be". Under the PR #10 validation gates they stay experimental until hindcast skill beats HF-radar persistence.

## 3. Proof of concept results (2026-10-09)

[`pipeline/scripts/wcofs_poc.py`](../../../pipeline/scripts/wcofs_poc.py), with evidence in [`evidence/`](evidence/):

- **Ingest:** the California surface subset of each 582 MB regular-grid file costs about **34 MB of HTTP range reads and about 5 s**. Encoded u and v are about **60 KB each** per step (265 × 238 cells). Twelve 6-hourly steps cost about 0.4 GB of reads and about 1 minute per day.
- **Encoding:** the WCOFS fields round-trip through `VectorField` and `ValueGrid` with a quantization error under 4 × 10⁻⁵ m s⁻¹.
- **Check against observations:** each WCOFS forecast from the 2026-10-08 run was compared with HF-radar 2 km currents at the same hour over Monterey Bay (651–723 overlapping cells):

| Valid (UTC) | Lead | Mean speed HFR / WCOFS (m s⁻¹) | Vector RMSE (m s⁻¹) | Complex correlation | Mean direction difference |
|---|---|---|---|---|---|
| 10-08 15:00 | +12 h | 0.165 / 0.163 | 0.282 | 0.45 (∠ −162°) | −46° |
| 10-08 18:00 | +15 h | 0.189 / 0.148 | 0.291 | 0.39 (∠ 149°) | −37° |
| 10-08 21:00 | +18 h | 0.171 / 0.143 | 0.266 | 0.42 (∠ 130°) | −66° |
| 10-09 00:00 | +21 h | 0.140 / 0.122 | 0.258 | 0.25 (∠ 177°) | 161° |

- **Speeds:** WCOFS speeds are realistic.
- **Hour-by-hour pattern:** it doesn't match the radar: errors are larger than the currents themselves.
- **Not a sign bug:** the direction difference changes from hour to hour, so this isn't a sign or rotation error in the convention. Tides, diurnal sea-breeze forcing and sub-grid bay circulation all contribute.
- **Sample size:** four hours of one day is far too small a sample for a skill estimate.
- **Not independent:** WCOFS assimilates HF radar, so even a forecast isn't fully independent of the observations.

## 4. Recommendation for the currents phase

1. **Lead with observations.** Ship HF-radar observed currents first, as the 2 km "currents now" layer: hourly, with a coverage mask, a 24 h mean option to suppress tides, and its real observation time.
2. **WCOFS as context, with its skill shown.** Add WCOFS forecast arrows at regional zoom (shelf scale) only after a **30-day hindcast check** against HF radar (daily-mean, de-tided currents, by region and lead), with the result published in the layer's methods.
3. **No drift or particle "outlook"** until the PR #10 gate (positive skill over HF-radar persistence at 24–48 h) is met.
4. **Size:** about 0.4 GB a day of read bandwidth; published artifacts under 2 MB a day (u and v grids plus arrows for 12 steps).
