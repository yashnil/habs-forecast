# 20 — M5: Ocean Map

**Status:** built on `feat/coastwatch-p5-map-experience` (draft PR #20). Not merged or deployed. No production data has been published, and the BASES demo has not been touched.

**Starting points:**
- Images: [m5/review/](m5/review/). Start with [1440-opening-forecast](m5/review/1440-opening-forecast.jpg), then [390-opening-forecast](m5/review/390-opening-forecast.jpg) and [satellite-cells-z11](m5/review/satellite-cells-z11.png).
- Background: the decision record is [19](19-m5-map-direction.md).

**What the "before" and "after" images show:**
- **Before:** P3 (main) on production data published 2026-10-10.
- **After:** M5 on the same data, with the Sentinel-3 latest view re-tiled at 512 px from its published values. That re-tiling is the only difference between the two datasets.

## 1. What changed

The five decisions approved after doc 19, as built:

| Decision | Built |
|---|---|
| Bathymetric basemap | NOAA ETOPO 2022 relief, the Coastal Relief Model in Monterey Bay, isobaths at 200–3000 m. Always on, under all data, darker and greyer than every data colour. Credited in the map attribution. |
| Freshness-aware opening | `lib/opening.ts`; thresholds in §2 |
| Fine-dot no-data | 1 px dots on a 3 px diagonal lattice, shown only under a data raster. The legend swatch matches the map. |
| 512 px tiles, one more zoom | Pipeline and web; staging and verification in §3 |
| Every-cell arrows | One arrow per observed 2 km cell from z8.5, thinned 2×, 4× and 8× when zoomed out. Direction and speed class unchanged. Animated flow is an option, off under reduced motion. |

**Also built:**
- **Place chip:** the place and its official-notice count with verification state ("4 notices · Not verified"). It opens the place list. It replaces the 320 × 330 px place card.
- **Compact dock:** what is drawn and when, the time steps, a legend with units, and the warnings that must stay visible. Options, conversions, method and provenance sit under Details.
- **Forecast warnings:** stale, historical and failed-update now share one line beside the freshness badge.
- **Hover readout (desktop):** the exact published value of the cell under the pointer, with its sensor or run and its date. It never shows the nearest cell or an interpolated value, and says when a cell has no value and why.
- **Inspector:**
  - order: official notices (compact rows), the four C-HARM days at the place, satellite, currents, then "Measured nearby";
  - the four days form a strip that also sets the map day;
  - "Measured nearby" shows the latest CalHABMAP samples within 30 km, each with its sample date;
  - 30-day nowcast history and 60-day chlorophyll collapse behind a sparkline summary.
- **C-HARM cell edges:** drawn faintly from z8.5. Geometry only; no value is changed or smoothed.
- **Transitions:** switching layer fades the new layer in over 220 ms. A new time step of the same layer cuts, never cross-fades. Reduced motion turns both off.
- **Phone:** the sheet peek is just tall enough for the layer tabs, what is drawn and when, and the whole legend; it grows when the stamp wraps. The place button is only as wide as its text. The port sheet opens at the peek so the map stays visible.
- **Tile index:** tiles that were never written are answered locally and never requested, so a 512 px tile set causes no 404s.
- **Keyboard:** Escape closes the place list, then the inspector. The rail is a tab list driven by the arrow keys.

**Removed:** the review switches (`?relief`, `?nodata`, `?arrows`).

## 2. Opening layer

Which layer the map opens on when the link names none. A layer in the link, or one the user picks, always wins. The link records the layer only once someone has chosen one, so a bookmark of the plain map keeps choosing by freshness.

1. **C-HARM**, when its latest run is current by the source's published freshness policy (issued within 1 day) and the opening variable and day exist.
2. **Otherwise satellite chlorophyll**, when a latest clear view meets both conditions:
   - it is current by its published policy (newest pixel within 3 days);
   - it observed **at least half** of the region's ocean (the whole domain, statewide).

   Products are tried in order: multi-sensor, Sentinel-3, then VIIRS. The dock says why it opened on satellite and links back to the forecast.
3. **Otherwise no data layer.** The bathymetric map shows with a message:
   - when the newest forecast was issued;
   - how much of the region satellites last saw, and when;
   - "This does not mean conditions are normal";
   - buttons for the last forecast and the latest satellite view, each with its date.

   See [m5-nothing-current](m5/review/m5-nothing-current.jpg).

**Where the thresholds come from:** the day limits are the pipeline's own `FreshnessPolicy`, not new numbers. The one new number is half the region's ocean: below it, the map would show more gap than observation.

**Tests:** 10 unit tests cover fresh, stale, missing, cloudy, partially covered, the inclusive threshold at exactly 0.5, a stale but well-covered satellite view, nothing published, the statewide domain fraction, and a current run missing the requested day. Browser tests cover the current, nothing-current and link-wins cases.

On the live data at the time of writing, C-HARM was issued 2026-10-10, so the map opens on the forecast.

## 3. Satellite tiles

**Format.** Tiles are now 512 px images in the standard 256 px Web Mercator grid:
- zoom limits: z5–11 for Sentinel-3 and the multi-sensor view, z5–10 for VIIRS;
- each image pixel takes the one source cell under its centre, so values, georeferencing and no-data masks are unchanged;
- each tile set also writes `index.json`, listing the tiles that exist.

**Compatibility, found while testing.** The first version added two manifest fields. The deployed web and the BASES demo validate the manifest with a schema that rejects unknown fields. Publishing that version would have made the demo show "Live data unavailable", because the demo reads the production dataset.

The format was changed so that no field is added:
- the existing `tile_size` field says 512;
- the index is found by convention;
- older clients ignore `tile_size` and draw the larger images in their 256 px slots.

A unit test validates new manifests against the released schema. That schema is identical in `demo-release/2026-10-09-submitted`, `demo-release/2026-10-10` and main before M5. P3 was checked against both old and new data.

**Detail at zooms where single 300 m cells matter.** Measured on a 2× screen offshore Monterey Bay, on the same published values ([satellite-cells-z11](m5/review/satellite-cells-z11.png), [z11.5](m5/review/satellite-cells-z11.5.png), [z12](m5/review/satellite-cells-z12.png), [numbers](m5/evidence/satellite-cell-edges.txt)):

| Zoom | True cell width | 256 px tiles (P3): widths drawn, edges off by > 2 px | 512 px tiles (M5): widths drawn, edges off by > 2 px |
|---|---|---|---|
| 11 | 14.6 px | 8 or 16 px · 77 % | 14 or 16 px · **0 %** |
| 11.5 | 20.6 px | 11–23 px · 86 % | 19–23 px · **21 %** |
| 12 | 29.1 px | 16 or 32 px · 95 % | 28 or 32 px · **33 %** |

**Verification against NOAA.** A full staging build of the satellite source was run with this code: the production dataset as the starting point, re-fetched from ERDDAP.
- Live comparison: **408 of 408 checks** pass (values, cell centres and 512 px tiles, including 384 multi-sensor pixels) ([evidence](m5/evidence/staging-local-satellite-verification.json)).
- The publish check validated **8,203 tiles** against their indexes, with no problems ([evidence](m5/evidence/staging-local-check-published.txt)).

**GitHub staging channel** ([run 38103638833](https://github.com/yashnil/habs-forecast/actions/runs/38103638833)). Published to `coastwatch-data-staging` on 2026-10-11. The two earlier runs stopped in currents verification while NOAA's HF-radar dataset `ucsdHfrW2` was offline.

| Check | Result |
|---|---|
| C-HARM against ERDDAP | 13 points × 12 layers, 0 failures |
| Satellite against ERDDAP | 420 / 420 |
| HF radar against ERDDAP | 32 / 32 |
| Manifest references | 8,499 tiles validated |
| Public URL | Served, complete and CORS-enabled |

The staged data has 512 px tiles for all three latest views, with their indexes served. Further checks on the staged data:
- the staged manifest validates against the released (demo) schema;
- in a browser, M5 on staging makes no 404 requests and logs no console errors;
- P3, standing in for the deployed demo code, loads the same staging data and draws the 512 px tiles correctly. It still requests absent tiles, as it does today.

**Storage**, measured on the staging build:
- satellite tiles grow from 14.9 MB to 35.2 MB;
- the dataset grows from 50 MB to 63 MB;
- single-day layers become 512 px as they roll over (about 8 MB more within a week).

The data branch is replaced on each run, so history does not grow.

## 4. Map area

Unobstructed share of the map, measured on a 4 px grid against the boxes of every floating control. Live data, ordinary unselected state ([M5](m5/evidence/map-area-m5.txt), [P3](m5/evidence/map-area-p3.txt)):

| Viewport | P3 | M5 | Target |
|---|---|---|---|
| 1440 × 900, forecast | 70 % | **82.6 %** | ≥ 75 % |
| 1440 × 900, multi-sensor | 64 % | **81.9 %** | ≥ 75 % |
| 1440 × 900, currents | 62 % | **81.7 %** | ≥ 75 % |
| 1280 × 720, forecast | 63 % | **76.4 %** | ≥ 75 % |
| 1280 × 720, multi-sensor | 57 % | **75.3 %** | ≥ 75 % |
| 1280 × 720, currents | 57 % | **75.0 %** | ≥ 75 % |
| 390 × 844, share of the screen showing map (forecast / satellite / currents) | 19 % | **60.1 / 57.8 / 57.8 %** | ≥ 55 % |

**Not weakened to reach these numbers:**
- the unverified-notices disclosure, its agency links and the notices count with verification state;
- the "not a toxin measurement or a closure decision" line;
- the "algae biomass, not toxin" line;
- stale and failed-update warnings;
- the "not a trajectory" line while flow is on.

**Framing:**
- opening a port fits the region beside the inspector;
- clicking a point pans it clear of the inspector;
- browser tests check that the rail, chip, place list, dock and inspector never overlap at 1280 or 1440.

## 5. Visual review passes

Both passes used live data at 1440 × 900, 1280 × 720 and 390 × 844 (57 views per pass).

**Pass 1 findings and fixes:**
- **Blue rectangles.** Satellite views showed solid blue wherever a tile is absent. The embedded "transparent" stand-in image was really half-transparent blue (0, 0, 255, 127). Replaced with a verified transparent PNG, and a unit test decodes it.
- **Graph-paper cells.** C-HARM cell edges read as graph paper at z9.5–10. Opacity was cut to 0.10–0.16 from z9.6.
- **Hover label.** The label truncated its quantity; it now wraps.
- **Chevron.** The place-chip chevron pointed the wrong way.
- **Arrow warnings.** Arrow layers mounted before their images were registered ("Image cw-arrow-N could not be loaded"). They now mount after load.
- **Notices too long.** Official notices filled the first screen of the port inspector; they are now compact rows that expand.
- **Phone legends.** The sheet put the satellite and currents legends below the fold, and the currents legend wrapped; both fixed.

**Pass 2 findings and fixes:**
- **Water labels.** Italic water labels had a hard dark outline over the forecast and chlorophyll; now a soft translucent halo.
- **Visible caveats.** The collapsed dock had lost two caveats: "biomass, not toxin" was clamped away, and the forecast's "not a toxin measurement" was missing. Both are always visible again.
- **Animated flow.** The flow legend now says "Not a trajectory" without expanding.
- **1280 × 720 below 75 %.** Satellite and currents were under the target. The time slider now shares a row with Hourly / 24-hour mean, display options moved under Details, and repeated lines were removed. All six desktop cases now meet the target.
- **Empty header.** The nothing-current dock showed an empty header; hidden.
- **Phone margin.** CI measured the phone at 53.8 %, against 56.1 % locally, because Linux fonts wrap differently. The peek now fits its content (136–184 px) and the place button is only as wide as its text, so the phone is 60.1 % locally, with margin for font differences.

**Images** in [m5/review/](m5/review/):
- 16 before/after pairs across the three viewports;
- 9 views of new features (hover readouts, place list, nothing-current state, C-HARM cells, phone sheet and places);
- 3 satellite-cell pairs.

## 6. Tests

| Suite | Result |
|---|---|
| Pipeline (pytest, offline fixtures) | 186 passed, including a test that each 512 px pixel is the cell under its centre, the mask holds, and the index lists the tiles written |
| JSON Schema in sync, web fixtures reproducible | Regenerated and committed |
| Web unit (vitest) | 118 passed (16 new: opening policy, cell edges, transparent tile, nearby stations, released-schema compatibility) |
| Browser (Playwright, 4 fixture servers) | 117 passed, including 23 new M5 tests |
| Accessibility (axe, in the browser suite) | No serious or critical violations: map with place list and inspector open; phone map and Bloom |
| Typecheck, lint, production build | Clean |
| Live C-HARM verification | First staging run: 13 points × 12 layers passed |
| Live satellite verification | 408 / 408 (§3) |
| Console during the 57 review views | No errors or warnings from M5. On the same data, P3 logs tile 404s and missing-image warnings. |

**What the new browser tests cover:**
- the opening layer (current, nothing current, link wins);
- map area: ≥ 75 % at 1440 for each layer, and ≥ 55 % of a phone screen;
- no overlap at 1280 and 1440;
- notice access in every state;
- Escape and the keyboard rail;
- phone sheet snap points from the keyboard;
- reduced motion;
- time steps cut, layer switches fade;
- C-HARM cell edges, and no-data dots only under data;
- no 404 tile requests;
- hover value equals the published C-HARM cell;
- the four-day strip sets the map day;
- Measured nearby;
- axe.

`CW_E2E_PORT_BASE` now moves the four test servers, so parallel checkouts can run the suite side by side.

## 7. Performance

Same data, served locally, Chrome. Median of 3 runs ([raw](m5/evidence/perf-p3-vs-m5.jsonl)).

| 1440 × 900 @2x | P3 | M5 (256 px data) | M5 (512 px data) |
|---|---|---|---|
| Forecast: load to map idle | 690 ms | 640 ms | 630 ms |
| Forecast: transfer | 804 KB | 1,231 KB | 1,230 KB |
| Multi-sensor: tile requests / 404s | 58 / 19 | 32 / 12 | 47 / **0** |
| Multi-sensor: tile bytes | 187 KB | 109 KB | 142 KB |
| Fly-to and pan: mean fps / p95 frame | 60 / 16.7 ms | 60 / 16.8 ms | 60 / 16.7 ms |
| Total blocking time | 0 ms | 0 ms | 0 ms |

The phone at 390 × 844 @3x shows the same pattern.

**Where the extra transfer comes from:** about 280 KB of relief basemap (cached after the first visit) and the C-HARM grid used for cell edges and hover values. The grid is the same file the inspector already reads.

**Caveat:** local serving hides network latency; these numbers compare builds, not networks.

## 8. Scientific and regulatory safeguards

**C-HARM**
- Values are never smoothed or interpolated.
- The readout uses the exact cell. The inspector says "nearest cell, N km away" when it substitutes one, and never presents that as the value at the point.
- Cell edges are geometry only.

**Satellite**
- Only the tiling changed. Values, masks and dates are re-verified against ERDDAP.
- Gaps are dots, never colour.
- "Algae biomass, not toxin" is always visible.

**Currents**
- Observed, never a forecast.
- One arrow per observed cell, with gaps visible as the radar's footprint.
- Animated flow is optional and says "Not a trajectory".

**Measured nearby**
- Each sample keeps its own date. A long gap is flagged ("176 days ago").
- "Measured at the station on the date shown, not today and not at this point."

**Official notices**
- Never marked verified.
- The count and verification state are on screen in every state.
- The full list is one click away.

## 9. Limitations

- **Old tile sets.** Tile sets published before M5 (today's production data) have no index, so M5 still requests their absent tiles until the 512 px tiles are published.
- **Measured nearby** shows the latest sample per variable; series are on the Bloom page. Stations more than 30 km away are not shown.
- **Hover** is desktop only (fine pointer). On touch, tapping opens the inspector with the same values.
- **Map area** is measured against floating-control boxes. It does not count small overlays such as the scale bar text.
- **Water labels** come from OpenFreeMap and are not curated; some bays show no name at some zooms.
- **Integration.** The pipeline change becomes production behaviour on the first scheduled run after M5 merges. Merging is safe for the demo (§3), but the order matters (§10).

## 10. Integration recommendation

**Recommend merging M5 after your review of the images**, in this order:

1. **Review the PR, then merge to main.** Scheduled data runs use main's code. The next run publishes 512 px tiles. The released web, including the BASES demo, still reads them; this was tested.
2. **Watch the first scheduled production run:** satellite verification, the publish check and the served-index checks.
3. **Deploy the web** that reads the indexes and opening policy, with separate approval.
4. **Do not redeploy the BASES demo** as part of this; it keeps working on the new data as it is.

**Before merging, if you want extra margin:** dispatch one more staging run and open the M5 preview against `coastwatch-data-staging`.

## 11. Recommendations for M6

1. **WebGL value-grid layer** for exact cell edges at z12+ and colour-mode switches without new tiles (doc 19 §1 option 2).
2. **Inspector series for points:** 30-day nowcast at any point, and satellite history at the pixel. This needs a small per-cell history artifact.
3. **Station markers** for CalHABMAP on the map, with last-sample age as the marker state.
4. **Shareable view state:** map centre and zoom in the link, not only region and layer.
5. **Offline-tolerant phone mode:** cache the last dataset and say so.
6. **Curated water-body labels** (Monterey Canyon, Gulf of the Farallones, Santa Barbara Channel) from NOAA charts, with source credit.
7. **Bloom and Fisheries pages** brought to the Ocean Map's layout and density (doc 17 audit).
