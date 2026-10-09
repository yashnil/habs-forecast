# 14 — P1: Ocean Map with high-resolution satellite observations

Status: **draft PR #11, not merged or deployed.** P0 (PR #9) is merged into `main`. PR #7 (design) and
PR #10 (forecast-upgrade research) are the references. Production data (`coastwatch-data`) is unchanged;
new data was tested only through the staging channel (`coastwatch-data-staging`).

## 1. What is implemented

### Pipeline (`pipeline/`)

- **New source `satellite_chl`:**
  - **Sentinel-3 OLCI chlorophyll-a at 0.0025° (about 300 m)** from NOAA CoastWatch (sectors CI + DI, Sentinel-3A first, Sentinel-3B filling gaps).
  - **VIIRS 750 m** (`erdVHNchla1day`) as an independent fallback layer.
- **Values:** every published value is an upstream value, log10-quantized to uint16 (relative error under 0.02 %).
- **Grids:** stored in 512 × 512 chunks so a browser reads only the chunk it needs. Cloud, land and quality masks stay no-value; nothing is gap-filled, smoothed or interpolated.
- **Time:**
  - per-day layers carry the upstream overpass timestamps verbatim;
  - the **7-day latest clear view** composite carries a **per-pixel age grid** and age tiles.
- **Display:** XYZ palette-PNG tile pyramids, z5–10 for 300 m, nearest cell per pixel, rendered from the published grid so a tile colour always equals the readable value.
- **Coverage per region:** the share of C-HARM ocean cells with at least one valid pixel, plus the observed area in km².
- **Alignment:** requests use exact cell centres on the source lattice; a scene off the lattice is rejected, and edge cells beyond the domain are trimmed.
- **Coastal scope:** pixels more than about 12 km (4 C-HARM cells) from the C-HARM ocean domain are left out, so inland lakes and reservoirs are never drawn as ocean. The count dropped is recorded as a QC check per day.
- **Reuse and outages:**
  - unchanged days are reused, not re-downloaded, but only if they were made by the same processing version (`satellite-processing-2`); older days are rebuilt;
  - an OLCI outage (the "unknown dataset" seen on 2026-10-09) falls back to VIIRS;
  - if only one product fails (VIIRS got HTTP 403 in staging), its previous layers are kept with their real dates;
  - a total outage keeps the previous layers with their real dates;
  - transient HTTP 403 and 429 from ERDDAP are retried.
- **`cwp verify-satellite`:** live ERDDAP point checks of values *and* cell centres, tile colour against grid value, composite against source day, and transparent empty cells. It now runs in the data workflow.
- **`check-published`:** validates tiles, every grid chunk, age grids and coverage.
- **C-HARM:** the PNGs use the approved stepped palette **`cw-probability-classes-v1`** (display classes, not risk levels); the value grids are unchanged. Staging showed that re-rendered images kept their old addresses, so the palette is now part of the content address, and a new rendering is reported "updated" and verified.
- **Schema (additive, `schemas/v1`):** stepped and log palettes; chunked `ValueGrid` with `transform`; pipeline-rendered `TileLayer`; `Coverage`; `CompositeInfo`; `native_resolution_m`; `platforms`; `observed_times`; and the `VectorField` contract for currents.

### Web (`coastwatch-web/`)

- **Navigation card:** the official row comes first and opens the notices drawer. Then port search, then regions → ports → breadcrumb. Monterey Bay is framed from Año Nuevo to Point Sur.
- **Layer dock** with three groups:
  - **HAB forecast:** C-HARM nowcast to +3 days only, stepped legend, "display steps, not risk levels", issue and valid dates, freshness, run and failure notes, the essential qualifiers inline, caveats and provenance one click away.
  - **Satellite:** OLCI 300 m latest clear view; per-day buttons with coverage bars; a fully clouded day says so and draws nothing; a per-pixel observation-age view; VIIRS 750 m fallback; GIBS imagery as pictures.
  - **Ocean currents:** disabled, labelled "Next phase", nothing drawn.
- **Every layer** shows its native resolution, its observation or valid time, and its freshness.
- **Inspector** appears only for a selected port or point, official notices first. A new "Satellite chlorophyll, latest clear view" section reads one grid chunk to give the exact value and its observation date. If cloud covers the point, it reports the nearest clear pixel within about 1 km, with its distance.
- **Basemap:** lighter land, coastline above all data, opaque nearest-sampled rasters over hatched no-value water, paper panels over the dark sea.
- **Mobile:** a place button, a collapsible dock above the tab bar, a places sheet, and a port sheet.

### Currents readiness (phase D)

The contract is defined, and a WCOFS proof of concept has been run and compared with HF radar; see [p1/currents-contract.md](p1/currents-contract.md).

## 2. Data sources and verified coverage (staging run 37982717189, 2026-10-09 19:48 UTC)

| Layer | Newest data | Native | Coverage of reference ocean cells |
|---|---|---|---|
| C-HARM v3.1 | issued 10-08: nowcast 10-07, forecast to 10-10 | 3 km | as published (nearshore masked) |
| OLCI 300 m latest clear view (7 d, 10-02 … 10-08) | overpass 2026-10-08 | 300 m | **23 % statewide**. North Coast 29 %, Mendocino–Sonoma 16 %, SF & Farallones 0.8 %, Monterey Bay 31 %, Central Coast 10 %, Southern California 99 % |
| OLCI single days 10-02 … 10-08 | | 300 m | 0.7–17 % of the domain per day (fog season) |
| VIIRS 750 m latest clear view | 2026-10-04 | 750 m | 75 % statewide; SF & Farallones 94 %, Monterey Bay 94 % |

An earlier local figure of 43 % statewide was inflated: ERDDAP snaps a time-range start to the nearest overpass, so the window had picked up a clear 10-01 day (8 days old). That is now filtered and tested; the figures above are the corrected ones.

**Sentinel-3B lags the 3A sectors by a week**, so in practice the latest clear view here is Sentinel-3A. When available, 3B fills only cells 3A left empty.

**Outage handling was exercised for real.** NOAA's central ERDDAP dropped the OLCI sector datasets ("Currently unknown datasetID", then HTTP 502) for more than an hour during development on 2026-10-09, then came back. A test covers exactly that case: OLCI fails and VIIRS is still published.

## 3. Performance and artifact sizes

| Item | Measured |
|---|---|
| Satellite source, statewide live run (OLCI 7 days × 2 sectors + VIIRS 3 days) | 2 min 44 s, of which about 22 s is CPU; the rest is downloads |
| Full pipeline in staging (all sources) | about 4.9 min |
| Satellite artifacts (staging) | **28 MB, 2,882 files** (tiles plus chunks for the composite, 7 days and VIIRS; varies with cloud cover) |
| Whole staging dataset | **35 MB, 2,943 files**, well under the 1 GB GitHub Pages limit, so no R2 needed yet |
| Tiles per layer | 694 (composite z5–10), 46–488 per day, 243 (VIIRS z5–9) |
| Statewide tile pyramid render (synthetic, 3,880 × 3,400 cells) | 2.6 s |
| Browser cost of a satellite readout | one 512 × 512 chunk (tens of KB gzipped) |

### Artifact inventory and what is checked

- **What the counts mean:** `files_checked` counts every manifest-referenced file except tiles: the manifest, JSON datasets, C-HARM images and grids, every satellite grid chunk and age grid, plus the sample tiles. Tiles are counted separately in `tiles_validated`.
- **Before publishing,** `check-published` now validates *every* tile:
  - each tile directory holds exactly the manifest's `n_tiles` files;
  - each one decodes as a 256 px palette PNG.
- **At the public URL,** the sample tiles confirm the files are served. The content is the same single commit.
- **Staging after run 37993419432:** 228 files checked plus 1,614 of 1,614 tiles validated, 0 problems.
- **Previous run kept:** the branch also keeps the previous run's artifacts (the prune policy is current plus previous manifest, for clients and CDN caches mid-load). After run 6 that was 1,167 files: the previous composite, the 10-01 day and the previous C-HARM rendering. That is why "2,882 satellite files" was larger than the checked count.
- **GitHub Pages size:**
  - typical 35–60 MB;
  - worst case a cloud-free week of 300 m layers, about 150 MB, or about 300 MB with the previous run kept. That is about 30 % of the 1 GB limit.
  - The repository is 242 MB; force-pushed data commits leave unreferenced objects until GitHub collects them, so repository size needs watching; the next phase adds it to the pipeline health check.

## 4. Tests and validation evidence

| Suite | Result |
|---|---|
| Pipeline `pytest` | **148 passed**, including 18 satellite tests:<br>• dates, coverage and age on three real regions;<br>• every value traced upstream;<br>• no-data preserved and transparent;<br>• lattice alignment;<br>• off-lattice rejection;<br>• OLCI-outage fallback;<br>• total-outage carry-over;<br>• day reuse, and rebuild after a processing change;<br>• one product failing while the other updates;<br>• inland water excluded;<br>• the 7-day window holds although ERDDAP snaps the start bound;<br>• pruning;<br>• implausible-value flag;<br>• fixtures are real ERDDAP files.<br>Also tested: palette steps, web/pipeline palette parity, and new renderings getting new addresses. |
| Web unit (`vitest`) | **82 passed**: chunked-grid sampling (log decoding, age, edge chunks, nearest pixel, missing chunks), stepped and log palettes, age-colour parity, resolution labels. |
| Playwright e2e (fixture build) | **65 passed**, including the new `p1.spec.ts`:<br>• layer groups and the disabled currents tab;<br>• opaque nearest raster and hatching;<br>• legend equal to the palette;<br>• satellite tiles and dates;<br>• clouded day;<br>• age;<br>• point readout equal to the verification report;<br>• inspector only on selection, official first;<br>• no panel overlap at 1280 and 1440.<br>Accessibility (axe) is clean on all pages. |
| Live satellite verification (local) | **108/108 checks**. 72 ERDDAP point comparisons all match in value *and* cell centre. ([evidence](p1/evidence/satellite-points-live-local.json)) |
| Staging run 37962710352 | Success. `verify-satellite` 108/108 in CI; `check-published` 333 files OK before publish and at the public staging URL with CORS; publish guard OK. |
| Staging run 37964087582 | ERDDAP answered HTTP 403 to C-HARM; carried over correctly. Led to the 403 retry. |
| Staging run 37964870549 | C-HARM re-rendered with the stepped palette: **156/156** point comparisons; satellite verification passed. |
| Staging run 37966225212 | Success, 331 files OK; screenshots showed inland lakes from days reused from before the coastal scope, which led to the processing-version gate. |
| Staging run 37967595146 | Success. All OLCI days rebuilt under `satellite-processing-2` (inland pixels gone); `verify-satellite` **96/96**; 220 files OK. ERDDAP returned 403 to every C-HARM and VIIRS request even after retries: C-HARM was carried over correctly, but VIIRS was dropped, which led to the per-product carry-over fix. |
| Staging run 37981549465 | Success, every source updated. C-HARM **156/156**; satellite **120/120**; 253 files OK publicly. VIIRS back. Revealed the 8-day-old pixel (window fix). |
| Staging run 37982717189 | **Final.** Success, every source updated. C-HARM **156/156**; satellite **108/108**; 217 files OK before publish and at the public URL with CORS. Composite 10-02 … 10-08, maximum age 7 days. |
| `tsc`, `eslint` | clean |

## 5. Screenshots

These show the running application against the **staging dataset** (real data, staging run 37982717189). They are in [p1/screenshots/](p1/screenshots/). Missing-data and stale-data states from recorded fixtures are in [p1/screenshots/states/](p1/screenshots/states/).

| View | 1440 × 900 | 1280 × 720 | 390 × 844 |
|---|---|---|---|
| HAB forecast | [01](p1/screenshots/desktop-1440/01-map-forecast.png) | [01](p1/screenshots/laptop-1280/01-map-forecast.png) | [01](p1/screenshots/mobile-390/01-map-forecast.png) |
| OLCI 300 m latest clear view | [02](p1/screenshots/desktop-1440/02-map-satellite.png) | [02](p1/screenshots/laptop-1280/02-map-satellite.png) | [02](p1/screenshots/mobile-390/02-map-satellite.png) |
| Port selected (inspector) | [03](p1/screenshots/desktop-1440/03-map-port.png) | [03](p1/screenshots/laptop-1280/03-map-port.png) | [03](p1/screenshots/mobile-390/03-map-port.png) |
| Observation age | [04](p1/screenshots/desktop-1440/04-map-satellite-age.png) | | [04](p1/screenshots/mobile-390/04-map-satellite-age.png) |
| Statewide 300 m | [05](p1/screenshots/desktop-1440/05-map-statewide-satellite.png) | [05](p1/screenshots/laptop-1280/05-map-statewide-satellite.png) | |
| Places sheet | | | [06](p1/screenshots/mobile-390/06-map-places.png) |
| Bloom / Fisheries / notices drawer | [07](p1/screenshots/desktop-1440/07-bloom.png) · [08](p1/screenshots/desktop-1440/08-fisheries.png) · [09](p1/screenshots/desktop-1440/09-official-drawer.png) | | [07](p1/screenshots/mobile-390/07-bloom.png) · [08](p1/screenshots/mobile-390/08-fisheries.png) · [09](p1/screenshots/mobile-390/09-official-drawer.png) |
| A clouded day | [10](p1/screenshots/desktop-1440/10-map-satellite-clouded-day.png) | | |

## 6. Remaining blockers and limits

- **No production publication yet.** The new source is verified in staging only. Publishing it to production requires merging PR #11 (the scheduled run would then publish it); that decision is yours.
- **Fog season limits 300 m coverage** in the central and north coast (SF & Farallones 0.5 % in 7 days). The VIIRS fallback and the age view keep this honest, but coverage will vary by season.
- **NOAA central ERDDAP reliability:**
  - the 2026-10-09 OLCI outage lasted more than an hour;
  - from GitHub's runners, ERDDAP intermittently answered **HTTP 403** to whole sources (C-HARM, VIIRS) for an entire run, while the same requests worked from a desktop. That looks like rate limiting or IP filtering of cloud runners on NOAA's side; it isn't something the pipeline can fix.
  - The fallbacks work, and the layers stay correctly dated, but a long run of 403s would make the scheduled data stale. If it persists, options are fewer requests per run (already reused where possible), a later schedule slot, or asking NOAA CoastWatch about runner access. None of these needs paid infrastructure.
- **Nearshore quality:** OLCI pixels next to shore can be affected by bottom reflectance and land adjacency. The caveat is shown; flag-level masking beyond NOAA's L3 product isn't available on this endpoint.
- **PACE and Copernicus** remain deferred (accounts).

## 7. Recommendation for the next phase: ocean currents

Details in [p1/currents-contract.md](p1/currents-contract.md).

1. **Ship HF-radar observed currents first** (2 km, hourly, with a 24 h mean option, a coverage mask and real observation time) as `observation`.
2. **Hold WCOFS forecast arrows** until a 30-day hindcast against HF radar (daily mean, de-tided, by region and lead) is published. In the PoC, four hourly comparisons in Monterey Bay showed realistic speeds but weak pattern agreement (correlation 0.25–0.45, vector RMSE 0.26–0.29 m s⁻¹, larger than the currents).
3. **No drift or particle outlook** until it beats HF-radar persistence (PR #10 gates).
