# Coastal intelligence v2 · Stage 0: the baseline (15 October 2026)

What the product actually is before v2 touches it: what is measured, modelled, retrieved or derived; the true
support of every product; coverage at the 22 ports; the nearshore gap; the test and CI state. Facts come from
the code on `main` (fc448079), the production dataset (run 38774047981, 2026-10-15 17:03 UTC) and the design
audit in PR #63. Nothing here is a plan.

## 1. How every displayed value is obtained

| Product | Kind | Source and endpoint | Native support | Cadence · latency | Where it appears |
|---|---|---|---|---|---|
| C-HARM v3.1 (Pseudo-nitzschia > 10⁴ cells/L; particulate DA > 500 ng/L; cellular DA > 10 pg/cell) | **Published statistical forecast** (NOAA; logistic models on gap-filled S-NPP VIIRS + WCOFS ROMS) | `coastwatch.pfeg.noaa.gov/erddap/griddap/wvcharmV3_{0..3}day` (NetCDF) | 0.03° ≈ 3 km, 351 × 391 cells, lat 31.3–43.0, lon −127.5 to −117.0 | daily run, nowcast + 3 days; ~20 % of days missing since 2022-11; issue date **inferred** (nowcast day + 1) | Ocean Map forecast layer, My Coast model section, port summaries |
| Chlorophyll-a, Sentinel-3 OLCI | **Satellite retrieval** | `coastwatch.noaa.gov/erddap` `noaacwS3{A,B}OLCIchlaSector{CI,DI}Daily` | 300 m (0.0025°) | daily overpass, 1.5–2 d latency; 7-day "latest clear view" composite with a per-pixel age grid | Ocean Map satellite, My Coast chlorophyll |
| Chlorophyll-a, VIIRS | **Satellite retrieval** | `noaacwN21VIIRSchlaSectorUSDaily` (NOAA-21) with NOAA-20 fill; S-NPP `erdVHNchla1day` only until 2026-11-02 | 750 m | daily, ~1.5 d | same, fallback when OLCI fails |
| Multi-sensor chlorophyll | **Derived (display only)** | pipeline `multisensor.py` | each pixel is one sensor's own value; nothing averaged | per run | Ocean Map |
| NASA GIBS chlorophyll browse | **Satellite retrieval (visual only, values not read)** | GIBS WMTS, VIIRS NOAA-20 and PACE OCI | tiles to z7 | daily, 1–3 d | Ocean Map imagery |
| Surface currents, HF radar | **Direct measurement** | HFRNet `ucsdHfrW2` on CoastWatch ERDDAP, fallback NDBC THREDDS `hfradar_uswc_2km` (in use today) | 2 km, 408 × 540 | hourly, 4–15 h latency; 24 h mean derived | Ocean Map currents |
| CalHABMAP shore stations | **Direct measurement** (pDA, dDA, tDA, Pseudo-nitzschia seriata/delicatissima counts, chlorophyll, T, S) | SCCOOS ERDDAP `HABs-<Station>` tabledap | 17 points | ~weekly; lab latency 1 week to 2 months | My Coast evidence, Bloom page |
| NDBC stations (82 CA, includes NOS tide stations and CDIP buoys) | **Direct measurement** (wind, waves, water T) | `ndbc.noaa.gov/data/realtime2/<id>.txt`, fetched live by the web server | points | 6–60 min | My Coast conditions |
| CO-OPS tide predictions | **Published astronomical prediction** | `api.tidesandcurrents.noaa.gov` hilo | 193 stations | — | My Coast conditions |
| NWS coastal marine zone forecast | **Published forecast (forecasters' words)** | `tgftp.nws.noaa.gov` CWF text (api.weather.gov only with `CW_NWS_API=1`) | zone | 4×/day | My Coast conditions |
| Official notices (CDFW, CDPH) | **Official regulatory, transcribed, unverified** | curated `data/curated/official_notices.json` + page watcher + CDPH ArcGIS polygons | polygons / latitude bands | watcher every 6 h | everywhere |
| Port summaries | **Derived** (C-HARM cells within 15 km; VIIRS 8-day median) | pipeline `port_intel.py` | per port | per run | Ocean Map port panel, My Coast |
| Fisheries landings | **Historical context** | NOAA FOSS + BLS CPI | statewide annual | weekly refresh | Fisheries page |

Not displayed and not integrated: any reanalysis or reconstruction (DINEOF gap-filled chlorophyll is a C-HARM
input only), WCOFS fields, Copernicus Marine products, IFCB images, CDPH shellfish toxin results, PSP.
Neritic's own experimental predictions: none in the product (the PINN/ConvLSTM research stays in `research/`).

## 2. The nearshore gap, measured

- C-HARM toxin values cover **12.6 % of coastal water within 1 km of shore** and 31.5 % within 3 km (bloom
  probabilities: 48 % and 64 %). Stable across 41 nowcasts: the gaps are structural (doc 27 §1).
- Ten distinct causes, kept separate in the product: land inside a 3 km cell (22 % of water within 1 km); the
  undocumented toxin mask 3–6 km from shore (35 %); the model's own no-value cells and bay mask (30 %; 79 % of
  SF Bay); fog and cloud (OLCI: median 5 % of water within 1 km valid per October day); C-HARM outages (286 of
  1,440 days); old observations; monitoring-network gaps; the end of S-NPP on 2026-11-02; and the absence of
  any validated nearshore estimate.
- Within 15 km of Monterey the model has 27 cells (6.6–15 km out) and none at the harbour; Santa Cruz 24 cells,
  nearest 7.2 km; San Diego 13 cells, nearest 8.4 km; Crescent City 40 cells, nearest 2.4 km.

## 3. Coverage at the 22 ports (production data, 15 Oct)

CalHABMAP station within 40 km, the nearest NDBC/NOS station, HF-radar 24 h-mean cells within 15 km, and the
dataset the port's chlorophyll came from.

| Port | CalHABMAP ≤ 40 km (last toxin / cells) | Nearest NDBC/NOS | HFR cells | Port chl |
|---|---|---|---|---|
| Crescent City | **none** | cecc1 (NOS, in harbour), 46027 | 0 | S-NPP 8-day |
| Trinidad | Trinidad Pier 0.6 km (pDA 2026-04-17) | tdpc1 | 0 | S-NPP 8-day |
| Eureka | Humboldt 3.6 km (pDA 04-07), South Bay 9.7 km | hbxc1 | 0 | S-NPP 8-day |
| Shelter Cove | **none** | 46213 (65 km) | 0 | S-NPP 8-day |
| Fort Bragg | **none** | 46014 (27 km) | 0 | S-NPP 8-day |
| Point Arena | **none** | anvc1 | 0 | S-NPP 8-day |
| Bodega Bay | BML 2.7 km (PN only, 04-13); Tomales (PN only) | bdxc1 | 35 | N21/N20 |
| San Francisco | **none** | pxsc1 | 30 | N21/N20 |
| Half Moon Bay | **none** | rtyc1 (24 km) | 39 | N21/N20 |
| Santa Cruz | Santa Cruz Wharf 2.2 km (pDA + PN 09-30) | 46092 (25 km) | 74 | N21/N20 |
| Moss Landing | Monterey Wharf 24 km; Santa Cruz 27 km | mlsc1 | 72 | N21/N20 |
| Monterey | Monterey Wharf 0.6 km (pDA ended 2022-08; PN 2025-08) | meyc1 (NOS, in harbour) | 69 | N21/N20 |
| Morro Bay | Morro Bay front/back 1–4 km (PN only, 09-01); Cal Poly 24 km | mbxc1 | 23 | N21/N20 |
| Avila / Port San Luis | Cal Poly Pier 1.4 km (pDA 08-03, PN 10-04) | pslc1 | 75 | N21/N20 |
| Santa Barbara | Stearns Wharf 0.4 km (pDA 08-03, PN 09-28) | ntbc1 | 68 | N21/N20 |
| Ventura | Stearns 38.8 km | ntbc1 (39 km) | 63 | N21/N20 |
| Port Hueneme | **none** | 46025 (44 km) | 91 | N21/N20 |
| San Pedro | Santa Monica 35.8 km; Newport 36.4 km | ohbc1 | 70 | N21/N20 |
| Newport Beach | Newport Pier 1.0 km (pDA 08-10) | 46253 (24 km) | 28 | N21/N20 |
| Dana Point | Newport 26.6 km | 46285 | 67 | N21/N20 |
| Oceanside | Scripps 38.4 km | 46224 | 58 | N21/N20 |
| San Diego | Scripps Pier 20.5 km (pDA 08-10) | sdbc1 (NOS, in harbour) | 0 | N21/N20 |

Seven ports have no HAB shore station within 40 km; only Santa Cruz has a toxin sample under 30 days old within
25 km. Bodega Bay and Morro Bay have cell counts, no toxin. Monterey's pier has had no toxin sample since 2022.

## 4. Source health today

- `coastwatch.pfeg.noaa.gov` answers HTTP 403 to cloud runners intermittently (10-09, 10, 11, 13). C-HARM has no
  alternate host; HF radar falls back to NDBC THREDDS automatically.
- Sentinel-3A OLCI: the time listing returned 404 in the production run (dataset id to re-verify, see
  `01-sources.md`); Sentinel-3B silent since 2026-10-01 (health alert open). Layers carried forward with their
  own dates (latest 2026-10-12).
- CalHABMAP: all 17 stations carried forward in the production run (300 s budget used up on SCCOOS reloads).
- S-NPP data end 2026-11-02; VIIRS is already on NOAA-21/NOAA-20; C-HARM's own input is still S-NPP and NOAA has
  published no plan.

## 5. Architecture facts that constrain v2

- The manifest schema is frozen for the released web app and the demo: `extra="forbid"` in the pipeline models,
  Ajv with `additionalProperties:false` in the web, and a frozen released schema in
  `tests/unit/compat/manifest.released.schema.json`. **No new manifest fields.** New information must either fit
  existing free-form fields (`provenance.upstream_metadata`, `notes`, `platforms`) or be served by the web
  server's own live routes (`/api/conditions`, now `/api/evidence`).
- The web server already fetches live in-situ data outside the manifest (NDBC, CO-OPS, NWS), with recorded
  fixtures for tests. That route pattern is the low-risk path for any new in-situ source.
- C-HARM is drawn tile by tile from the published value grid (exact cell edges, nearest neighbour, clipped to the
  basemap's water); nothing is interpolated. Cell outlines exist as a GeoJSON layer from z 8.5.

## 6. Design audit (PR #63) findings carried into Stage 1

- P0: the offshore model drawn as the boldest evidence at the harbour; notice status repeated 4–7 times; forecast
  day labels contradicting each other ("issued Oct 14", "nowcast Oct 13", "+1 day", "yesterday"; the map opened
  on yesterday); distance hidden behind the headline (Santa Cruz Wharf 41 km from Monterey).
- P1: routine data latency shown in alert colour; 73–79 % rendered as one tone; 15–18 font sizes per page; most
  phone controls under 44 px; My Coast 1,325 words / 7.5 phone screens.

## 7. Tests and CI (before v2)

- Web: 310 Vitest unit tests (32 files), with scientific invariants in `gridtiles`, `grid`, `hab`, `safety`,
  `multisensor`, `design`; 26 Playwright specs (~281 tests) on four `next start` servers (fixture, failed,
  nodata, compat-m2) with axe checks in 13 specs; `test:published` against the live dataset (not in CI).
- Pipeline: 245 offline pytest tests (~3 min with `-n 4`), one live test; schema-drift and fixture-reproducibility
  checks.
- CI (`coastwatch-ci.yml`): scope job, `web-static` (gen:types, lint, typecheck, unit), `web-e2e` in 2 shards on
  ubuntu-latest with bundled Chromium, `pipeline`.
- Performance and Lighthouse: not measured in CI. The design audit recorded 0 console errors on 60 captures.

## 8. Visual baseline

The 60 captures from the design audit are in
`docs/coastwatch/design/2026-10-neritic-excellence/review/img/base-*.webp` (1440, 1024, 768, 390 and 320 px).
The v2 "after" captures are in `review/img/` of this folder.
