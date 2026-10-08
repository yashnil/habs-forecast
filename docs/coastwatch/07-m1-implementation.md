# 07 — Milestone 1 implementation: honest baseline + live C-HARM forecast

Branch `feat/coastwatch-m1-baseline` (not merged; website not deployed). Built and verified 2026-10-08. Integration gate (GitHub CI + real data publishing) completed the same day — see §7.

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
| Scheduling and publishing | `coastwatch-data.yml` every 6 h: build (read token) → publish (write token, no repo code) → check public URL (read token). `coastwatch-ci.yml` runs all checks | Both ran green on GitHub (§7) |

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
4. *(Resolved §7)* CI runs green on GitHub, including Playwright on Linux headless Chromium.
5. *(Resolved §7)* Publishing to the `coastwatch-data` branch tested end to end on GitHub.
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

## 7. Integration gate (2026-10-08)

### 7.1 GitHub CI

| Workflow | Run | Result |
|---|---|---|
| CoastWatch CI (pipeline + web) | [37827522709](https://github.com/yashnil/habs-forecast/actions/runs/37827522709) | ✅ pipeline: 53 passed, 1 deselected (the live test; its equivalent runs in the data workflow) · schema and fixture drift checks clean · web: lint, typecheck, 40 vitest, production build, 15 Playwright e2e on headless Chromium |
| CoastWatch data refresh | [37827522650](https://github.com/yashnil/habs-forecast/actions/runs/37827522650) | ✅ build → publish → public-URL check (details below) |

The final commits were re-run on CI after this report; see the pull request checks for the latest runs.

Permissions: repository default is read-only (`default_workflow_permissions: read`). CI declares `contents: read` and does not persist credentials. The data workflow declares `permissions: {}` at the top; only the `publish` job has `contents: write`, and it checks out no repository code and runs no third-party build tools (it downloads the verified artifact and pushes it). No tests are skipped; the only deselected test is the live end-to-end check, which the data workflow covers with `cwp verify-charm`.

### 7.2 Real data publishing

Path verified on GitHub runners: ERDDAP / NASA GIBS / CDFW → `cwp run` → validation → `cwp verify-charm` (156/156 against ERDDAP) → `cwp check-published` (26/26 files) → `cwp guard-publish` → artifact → force-replace of the `coastwatch-data` branch (single commit) → `cwp check-published` against the public URL with the expected run id (26/26, CORS OK).

**Public data URL** (no account changes were needed; the repository was already public):
`https://raw.githubusercontent.com/yashnil/habs-forecast/coastwatch-data/v1/manifest.json`

| Check | Result |
|---|---|
| Manifest and assets load from published URLs | ✅ HTTP 200; independently re-checked from a workstation with `cwp check-published` |
| CORS | ✅ `access-control-allow-origin: *` on manifest, PNGs and grids; PNGs served as `image/png`; manifest as `text/plain` (fine for server-side `fetch().json()`) |
| Every manifest reference exists and decodes | ✅ 12 images (sizes match), 12 grids (decoded length matches), ports |
| Atomic versions | ✅ assets are content-addressed (`charm/<issued>/lead<k>-<sha>/…`, `ports-<sha>.geojson`); a new manifest references only new paths, and the previous manifest's files are kept one more run to cover the 5-minute CDN cache, so a client never mixes versions. The branch update is a single ref update. Tests: `pipeline/tests/test_publish.py` |
| Failed refresh preserves last valid data without pretending it is current | ✅ Drill on the real published dataset with ERDDAP unreachable: C-HARM `outcome: failed`, `last_success_at` kept, issue date still 2026-10-08, same asset paths, dataset still passes `check-published`; verification is skipped (nothing new) so the failure status is still published. The browser then classifies by date (stale after 2 days, historical after 7) and shows the failure banner (e2e) |
| Cannot expose unrelated repository files | ✅ The publish tree is built in a separate directory from pipeline output only; `guard-publish` rejects anything not `v1/*.{json,geojson,png,u16.gz}` (test: research code, `.env`, notes are refused); the publish job also refuses staged paths outside `v1/`. Published branch: 1 commit, 27 files, all under `v1/` |
| Research repo not bloated by data | ✅ The data branch is a single squashed commit each run (bot force-push), so history does not accumulate in clones |
| Production-equivalent frontend | ✅ `npm run build` + `next start` with `CW_DATA_BASE_URL` = the public URL; `npm run test:published` (Playwright): no data-unavailable or fixture banner, map image source = published content-addressed URL, inspector values decoded in the browser equal the values verified against ERDDAP, no failed cross-origin requests |

**Limitation:** raw.githubusercontent.com is fine for validation and light use, but it is not a production CDN (rate limits, `text/plain` for JSON, no SLA). For public launch, publish the same artifact to **GitHub Pages** (one repository setting) or Cloudflare R2. Setup steps are in §7.5.

### 7.3 Scientific and UI review outcomes

Verified against the ERDDAP metadata: variable names and thresholds (`pseudo_nitzschia` > 10,000 cells/L, `particulate_domoic` > 500 ng/L, `cellular_domoic` > 10 pg/cell), probability scale 0–1 shown as 0–100% on a fixed legend, lead/valid dates (nowcast = 2026-10-07, +3 = 2026-10-10), inferred issue date labelled, freshness by issue date. Monterey mid-bay inspector (94% / 77% / 32%) equals ERDDAP.

Changes made in this review:
- Badge "Official forecast" → **"Agency forecast"** so that a NOAA model forecast cannot be confused with an official closure or advisory.
- Heading "Bloom and toxin forecast" → **"Bloom and domoic acid forecast"** (C-HARM predicts DA in the water column, not toxicity of seafood).
- **All caveats are visible without expanding** ("Read before using" box for C-HARM; caveat list for satellite chlorophyll); source and provenance stay in a collapsible section. Test: every published caveat must be visible.
- Freshness badge says what the age refers to ("Current · issued today", "observed 1 day ago").
- Mobile: upcoming experiences collapse to "+ 3 upcoming"; scale bar hidden on narrow screens to avoid overlapping the attribution.

Checked: no copy asserts toxicity of seafood, safety, or fishing productivity (automated phrase checks over copy, artifacts and components). Reviewer checklist: [`08-scientific-review-checklist.md`](08-scientific-review-checklist.md).

Final screenshots (production build reading the published dataset): [`m1/01-live-map-desktop.png`](m1/01-live-map-desktop.png), [`02-inspector-monterey.png`](m1/02-inspector-monterey.png), [`03-satellite-chlorophyll.png`](m1/03-satellite-chlorophyll.png), [`04-mobile-nearshore.png`](m1/04-mobile-nearshore.png), [`05-data-and-sources.png`](m1/05-data-and-sources.png).

### 7.4 Remaining blockers before public launch

1. **Scientific sign-off** on the C-HARM copy (checklist 08). Required before the site is public.
2. **Production data host:** move from raw.githubusercontent.com to GitHub Pages or R2 (requires a repository setting or an account; see 7.5).
3. **Web hosting** not set up (deliberately not deployed).
4. **Official closures and advisories** are linked, not ingested (M2). Decide whether launch waits for M2.
5. Known issues 1–3 and 6–11 in §4 remain (inferred issue date, no v3.1 skill paper, OpenFreeMap dependency, GIBS legend size, mobile panel layout, no full accessibility audit).

### 7.5 Steps that need you (not done here)

**GitHub Pages as the data host (recommended before launch):**
1. Repository → Settings → Pages → Build and deployment → Source: **GitHub Actions**.
2. Tell me when that's done; the data workflow then gets a `deploy-pages` job (`actions/upload-pages-artifact` + `actions/deploy-pages`, with `pages: write` and `id-token: write` on that job only). The data URL becomes `https://yashnil.github.io/habs-forecast/v1/`, and the app's `CW_DATA_BASE_URL` changes to match. Note that this creates a public `github.io` site containing only the data artifacts.

**Web hosting (when you decide to launch):** connect `coastwatch-web/` to Vercel or Cloudflare Pages, set `CW_DATA_BASE_URL` to the data URL, and build with Node 22.

### 7.6 Commits in the integration gate

See `git log main..feat/coastwatch-m1-baseline`. Gate-specific commits: atomic publishing and published-data checks; review fixes and production-equivalent test; this report.

### 7.7 Merge recommendation

All technical gates pass: CI green on GitHub, real publishing verified end to end, failure drill passed, production-equivalent frontend verified. **Ready to merge into `main`** once the PR checks pass. Merging enables the six-hourly data refresh on `main`, which publishes data only, not the website. **Not ready for public launch** until scientific sign-off and a production data host are in place (§7.4).
