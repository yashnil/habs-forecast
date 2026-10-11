# 18 — M5 plan: the Ocean Map as the centrepiece

Status:
- **Branch:** `feat/coastwatch-p5-map-experience`, cut from `main` at `88d6aef` (after #16 and #15).
- **Stage:** plan only. No map component has been changed yet.
- **Brief:** the design audit and M5 brief are in [17-roadmap-and-design-audit.md](https://github.com/yashnil/habs-forecast/blob/docs/coastwatch-roadmap/docs/coastwatch/17-roadmap-and-design-audit.md) (§4.1, §6), on PR #17.

## 0. Ownership and starting point

**Ownership.** This branch is the only place where Ocean Map components change. Those components are:
- `LiveOceanMap.tsx`;
- `components/map/*`;
- `MobileSheet.tsx`, `Banners.tsx` and `shell/OfficialShell.tsx`;
- the map's style and its basemap assets.

**Rules for the other refs:**
- **Demo:** the demo branch `demo/bases-rc` (PR #18) and the tag `demo-release/2026-10-10` are frozen. Nothing is merged into them or from them.
- **Responsive fixes from the RC:** port `a96ec1f` ("give the map most of a phone screen") and the label part of `15fe6e2` by hand, as the first commit. Leave out the preview-only code (`NEXT_PUBLIC_CW_DEMO`, `DemoIntro`, `RegulatoryGap`).

## 1. Scope, in build order

Each step is a separate commit with its own tests.

| # | Step | Main files | Done when |
|---|---|---|---|
| 1 | **Full-bleed layout.** The map fills the viewport under the masthead. A slim left control rail: 64 px collapsed, 320 px expanded, holding layer group, product and time. The inspector becomes a right sheet. The map's fit area is recomputed from the rail and the sheet, so the selected place is never covered. | `LiveOceanMap`, new `map/ControlRail`, `map/Inspector`, `MapCanvas` (fit padding) | At 1440 × 900 the map's unobstructed area is ≥ 75 % (today about 55 %), measured by the e2e test from element boxes |
| 2 | **One status line.** The official disclosure, data-failure and freshness notices merge into a single expandable line under the masthead. The unverified-notice wording stays unchanged. | `Banners`, `shell/OfficialShell` | Never more than one banner row. The disclosure text is still present (existing safety tests pass unchanged) |
| 3 | **Mobile sheets.** A bottom sheet with three snap points: peek (legend and time), half (controls), full (inspector). Drag handle, keyboard and screen-reader operable. Places open as a full-screen search sheet. | `MobileSheet`, `map/LayerDock` split into rail and sheet content | At 390 × 844, peek leaves ≥ 55 % of the map visible. Every snap point is reachable by drag, tap and keyboard. Axe is clean |
| 4 | **Compact legend and hover readout.** The legend is a single-row ramp with three to five ticks, expanding to class detail and caveats. On desktop, pointing at the map shows the exact value and date read from the published grid (spec §3.1). No readout where there is no data. | `map/LayerDock` (legend), new `map/HoverReadout`, `lib` grid sampling | The readout equals the inspector value at the same pixel (e2e, on real tiles). No value is shown over no-data |
| 5 | **Marine basemap.** A custom MapLibre style: land and water tones, coastline weight by zoom, a label hierarchy (ports, then towns, then roads), and latitude-limit labels placed offshore with collision rules. Bathymetric relief is added only if the licence and size check passes (§3). | `public/` style JSON, `MapCanvas` | The coastline stays above all data. Legend colours still equal map colours (pixel test unchanged). Labels don't collide at the three viewports |
| 6 | **Richer inspector.** Model section: the 4-day min–max strip and the 30-day nowcast sparkline from published C-HARM history. "Measured nearby": the nearest CalHABMAP station with its last sample and age, linking to Bloom. Satellite and currents as compact dated rows. | `map/Inspector`, `SatelliteNear`, `CurrentsNear`, new `map/MeasuredNearby` | Every number traces to a published file (unit test per row). Missing history is shown as missing, not interpolated |
| 7 | **Region framing.** Hand-set views for every region, not only Monterey Bay. Keyboard navigation of ports. | `map/NavCard`, region config | Each region's coastline fills ≥ 60 % of the free map area at 1280 × 720 |
| 8 | **Typography and motion.** A type scale for map UI (numbers in tabular figures). 150–200 ms ease-out for panels and sheets. A raster crossfade only when the layer changes, never between time steps of observed data. All motion is off under `prefers-reduced-motion`. | `globals.css`, the components above | The reduced-motion e2e shows no transitions. No crossfade between time steps (test) |

**Not in M5:**
- Bloom (M6) and Fisheries (M7);
- forecast currents or drift (the WCOFS gate is not met);
- the full-site deployment (M9).

## 2. Rules that don't change

- **No fake interactions:** every control does something real on published data. Anything not yet implemented is absent, not shown disabled or mocked.
- **Data:** no invented or decorative values. Observation and model are distinguished everywhere, and every layer shows its own time.
- **Official notices:** shown first, with the "not verified" disclosure.
- **Chlorophyll:** stays opaque and nearest-sampled; the combined view stays opt-in.

## 3. Open checks before step 5

- **Bathymetry source:** GEBCO grid or NOAA relief tiles. Confirm the licence and the attribution text. Measure the tile size for the California extent at zoom ≤ 10, with a budget of ≤ 3 MB on first view.
- **Hosting:** the tiles must be static files on the existing Pages or Vercel hosting. No new paid tile service.

## 4. Testing, performance and visual QA

**Tests:**
- **Suites:** pipeline `pytest`, web `vitest`, Playwright e2e and axe, `tsc`, `eslint`, all green on every commit.
- **New e2e checks:** the measurable "done when" criteria in §1.

**Performance:** measured on production data with a production build. Desktop at 1440; phone at 390 with 4× CPU slowdown. Compare against the P3 baseline ([16-p3-map-refinement.md](16-p3-map-refinement.md) §4).

| Metric | Budget |
|---|---|
| First map idle | no worse than P3 + 10 % |
| Layer switch | ≤ 1 s |
| Sheet and panel transitions | 60 fps |
| JS added to the map route | ≤ 40 KB gzip |

**Visual QA:**
- **Screenshots:** every view at 1440 × 900, 1280 × 720 and 390 × 844, on production data.
- **Pass 1:** list every shortcoming, then fix.
- **Pass 2:** fresh eyes, compared against the design-reset prototypes and the P3 screenshots.
- **Report:** both passes go in an M5 report, with before and after images and the shortcomings that remain.

## 5. Merge and deploy

- **Before merging:** the draft PR is ready only when §1–§4 are met. Merging needs your approval.
- **Deploying:** nothing is deployed to `coastwatch-demo`, and the full site waits for M9.
