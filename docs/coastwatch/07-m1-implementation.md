# 07 — Milestone 1 implementation: honest baseline + live C-HARM forecast

Branch `feat/coastwatch-m1-baseline` (not merged, not deployed). Built and verified 2026-10-08.

## 1. What works

| Area | Built | Evidence |
|---|---|---|
| Cleanup | Removed the synthetic snapshot/overlay, canned fishing and economic advice (`recommendations.ts`), the hand-placed harbour list, `fisheries_context.json`, the GIBS newest-date API route, the dead CDPH link, the Streamlit dashboard and its weekly synthetic-data workflow | `git log`; `tests/unit/safety.test.ts` forbids the dead link, within-map percentiles and the 8.3% figure in app code |
| Public claims | README: unsupported 8.3% PINN improvement and unquantified 2019-heatwave claim replaced with same-pipeline numbers; research files and results untouched | `README.md` §1, §5 |
| Ports | 22 principal ports from CDFW ds3081 by port code (one identity for Princeton / Half Moon Bay; Santa Barbara 34.407°N; San Pedro −118.29) | `pipeline/tests/test_gibs_ports_safety.py::test_ports_from_cdfw_fix_known_errors` |
| Pipeline | `pipeline/` package: C-HARM v3.1, NASA GIBS chlorophyll, CDFW ports; validation; Web-Mercator rendering; value grids; manifest; per-source failure isolation; last-known-good | 45 offline tests + 1 live test |
| Schemas | Pydantic → `schemas/v1` → generated TypeScript; Ajv validation on the server | drift checks in CI and tests |
| C-HARM | All four leads of the newest run; issue date derived and labelled; missing leads reported, never back-filled; three probability variables with thresholds | live point verification below |
| Satellite tiles | Date = newest date whose three probe tiles over California are HTTP 200 palette PNGs with ≥ 2% data; legends from GetCapabilities and checked | On 2026-10-08 the newest VIIRS date returned 500s, then empty tiles; the pipeline used the previous day |
| Freshness | current / stale / historical / unavailable computed in the browser per source policy; failed updates shown in a banner and on `/sources` | unit tests at day boundaries; e2e with a frozen clock |
| Hierarchy | Official-status card first (closures not tracked yet; official links and hotlines), then official forecast, then observation | e2e DOM-order test |
| Design | Navy MapLibre basemap (OpenFreeMap), magenta single-hue probability ramp (monotone lightness, 0% at 2.16:1 against water), Geist type, desktop floating panel, mobile stacked layout, Upcoming nav items not linked | screenshots in [`m1/`](m1/) |
| Scheduling | `coastwatch-data.yml` every 6 h → `coastwatch-data` branch; refuses to publish if live verification fails. `coastwatch-ci.yml` runs all checks | inert until merged (scheduled workflows run from the default branch) |

## 2. C-HARM geographic verification (live)

`uv run cwp verify-charm` decodes the published value grid at each point, checks that the reprojected image pixel at that cell is exactly the palette colour of that value (transparent where there is no value), and queries ERDDAP directly for the same cell.

Result for run issued 2026-10-08 (manifest generated 2026-10-08T17:34:07Z): **12 layers × 13 points = 156 comparisons, 0 failures**; 8 comparisons are cells with no value in the source (the Monterey Wharf pier cell for the two toxin variables, all four leads), matched by "no value" in the artifacts. Full report: [`evidence/m1-charm-verification-2026-10-08.json`](evidence/m1-charm-verification-2026-10-08.json).

Lead +1 (valid 2026-10-08):

| Point | Cell (lat, lon) | P(PN) published / ERDDAP | P(pDA) published / ERDDAP | P(cDA) published / ERDDAP | Image pixel = palette(value) |
|---|---|---|---|---|---|
| Off Crescent City | 41.71, -124.35 | 0.8188 / 0.8188 | 0.3479 / 0.3479 | 0.4848 / 0.4848 | yes |
| Off Eureka | 40.84, -124.29 | 0.8399 / 0.8399 | 0.5383 / 0.5383 | 0.6968 / 0.6968 | yes |
| Off Fort Bragg | 39.46, -123.90 | 0.8388 / 0.8388 | 0.6749 / 0.6749 | 0.4664 / 0.4664 | yes |
| Off Bodega Bay | 38.26, -123.15 | 0.9909 / 0.9909 | 0.6488 / 0.6488 | 0.5308 / 0.5308 | yes |
| Gulf of the Farallones | 37.69, -122.79 | 0.9694 / 0.9694 | 0.8149 / 0.8149 | 0.3783 / 0.3783 | yes |
| Off Half Moon Bay | 37.45, -122.61 | 0.9658 / 0.9658 | 0.7186 / 0.7186 | 0.5184 / 0.5184 | yes |
| Off Santa Cruz | 36.91, -122.10 | 0.9475 / 0.9475 | 0.7709 / 0.7709 | 0.2460 / 0.2460 | yes |
| Monterey Bay (mid-bay) | 36.79, -121.95 | 0.9438 / 0.9438 | 0.7690 / 0.7690 | 0.3248 / 0.3248 | yes |
| Monterey Wharf (nearshore) | 36.61, -121.89 | 0.9507 / 0.9507 | — / — | — / — | yes |
| Off Morro Bay | 35.35, -120.99 | 0.8398 / 0.8398 | 0.7705 / 0.7706 | 0.1637 / 0.1637 | yes |
| Santa Barbara Channel | 34.30, -119.79 | 0.1618 / 0.1618 | 0.7516 / 0.7516 | 0.1619 / 0.1619 | yes |
| San Pedro Channel | 33.61, -118.29 | 0.0918 / 0.0918 | 0.8084 / 0.8084 | 0.1068 / 0.1068 | yes |
| Off San Diego | 32.71, -117.36 | 0.1603 / 0.1603 | 0.7835 / 0.7835 | 0.0687 / 0.0687 | yes |

Published values are quantized to uint16 (maximum error 7.6 × 10⁻⁶); the comparison tolerance is that error plus 10⁻⁶.

**Bug found by this check and fixed:** an earlier build rendered images from raw floats while the grid stored quantized values, so one pixel (San Pedro Channel, lead 1) was one colour unit off. Images are now rendered from the published values, and `test_every_image_pixel_is_the_palette_colour_of_the_published_value` checks every pixel of every image.

Placement: images are resampled onto pixels uniform in Web-Mercator y, so corner placement is exact. Without this, a 31.3–43.0°N image would be misplaced by more than 10 km in the middle (regression test `test_naive_corner_placement_would_misplace_by_kilometres`).

## 3. Tests

| Suite | Command | Result (2026-10-08) |
|---|---|---|
| Pipeline (offline, recorded fixtures) | `cd pipeline && uv run pytest` | 45 passed |
| Pipeline (live ERDDAP) | `uv run pytest -m live` | 1 passed |
| Web unit (vitest) | `cd coastwatch-web && npm test` | 40 passed |
| Web lint / types | `npm run lint && npm run typecheck` | clean |
| Production build + e2e (Playwright, 3 servers) | `npm run test:e2e` | build OK; 15 passed |

Coverage of the requested areas: schema validity (Python + Ajv, malformed manifests); ingestion and malformed responses (HTML error page, missing variable, out-of-range values, irregular grid, outside domain, wrong product version, time mismatch, fill values, mostly-empty field); valid-time calculation (issue-date derivation, lead mixing, missing leads); missing data (first-ever failure, total failure keeps last good, no-data server); stale classification (day boundaries, frozen-clock e2e: current / stale / historical); projection and raster alignment (Mercator round trip, per-pixel sampling, corner placement, map source coordinates in the browser); numerical checks against C-HARM (fixture and live); broken satellite tiles (HTTP 500, corrupt RGBA, empty tiles, all dates bad, dead legend); attribution (every layer's provenance, citation, request URLs); safety invariants (copy and artifact text, hierarchy order, official domains, no percentile tiers, chlorophyll never official/toxin); TypeScript type checking; production build.

## 4. Known issues and unverified assumptions

1. **C-HARM issue date is inferred** (nowcast valid day + 1). Derived from metadata patterns on 2026-10-08, not documented by the producer. Labelled "(inferred)" everywhere.
2. **No published skill assessment for C-HARM v3.1** was found; the app cites the v1 assessment (Anderson et al. 2016) and states this.
3. **Official closures and advisories are not ingested.** The card says so and links to CDFW/CDPH. Curated records are M2.
4. **CI has not run.** Workflows were written and their commands run locally, but GitHub Actions has not executed them. Linux headless Chromium uses software WebGL; locally, SwiftShader did not draw vector layers (rasters and map sources work). The e2e tests do not depend on vector-layer pixels, but this is unverified on CI.
5. **Data publishing to the `coastwatch-data` branch is untested end to end** (requires merge to the default branch, and a public repository for raw.githubusercontent.com access).
6. **Basemap depends on OpenFreeMap** (free, no SLA). If it is unreachable, the forecast still renders on the land-coloured background without coastline or labels.
7. **NASA GIBS legend SVGs are ~600 KB** and drawn for a light background (shown on a white chip). Satellite chlorophyll also appears over inland lakes, as NASA provides it.
8. **Mobile:** the panel stacks below a 58vh map rather than a bottom sheet; the expanded attribution can overlap the scale bar on narrow screens.
9. **Accessibility:** text contrast checked (≥ 5.27:1 on all text tokens); no automated axe audit or screen-reader pass yet; the map itself is not keyboard-inspectable (the panels are).
10. **CDFW ds3081 port points** are CDFW's reference locations for landing records; CDFW notes they may not be exact harbour positions.
11. **C-HARM raster covers the full domain to 127.5°W.** Faithful to the source, but visually dominant; a nearshore crop or opacity default may be revisited with users.

## 5. Run it

```bash
# pipeline (Python 3.11+, uv)
cd pipeline
uv sync
uv run cwp run                    # live -> ../coastwatch-web/public/data/v1
uv run cwp verify-charm           # live point verification against ERDDAP
uv run pytest                     # offline tests

# web (Node 22)
cd ../coastwatch-web
npm install
npm run dev                       # http://localhost:3000 (reads public/data/v1)
# offline:
npm run data:fixture && CW_DATA_BASE_URL=/data/fixture/v1 npm run dev
# checks:
npm run lint && npm run typecheck && npm test && npm run test:e2e
```

## 6. Merge readiness

Ready for review, not for public deployment. Before merging: run CI on GitHub (item 4) and have a domain reviewer read the forecast copy. Before deploying publicly: the steps in `06-development-plan.md` §6, including scientific review of the copy, confirming the data branch publishes, and deciding whether the curated official-status records (M2) must ship first.
