# 16 — P3: Ocean Map refinement and the combined currents + chlorophyll view

Status:
- **Branch:** `feat/coastwatch-p3-map-refinement`, draft PR. **Not merged.**
- **Website:** not deployed.
- **Production:** P2 (PR #13) is merged and publishing (section 1).
- **Companion documents** in this phase:
  - [p3/regulatory-review-2026-10-10.md](p3/regulatory-review-2026-10-10.md): issue #12 checklist;
  - [p3/noaa-outreach-draft.md](p3/noaa-outreach-draft.md): unsent email draft.

## 1. P2 integration and production

**Pre-merge checks** (all passed):
- **CI and branch:** CI is green; the branch is current with `main`.
- **Artifacts:** the full local check of the final staging dataset found 271 files plus 2,842 of 2,842 tiles OK. All 50 HF-radar u/v grids decode.
- **C-HARM:** 12 of 12 layers are identical between `main` and P2 on the same live data: grid bytes, PNG bytes, metadata, time, caveats, palette, QC.
- **Currents can't read as forecasts:** every currents layer is `observation`, with no lead time and no time later than when the data were produced.
- **Missing coverage is explicit:** shown in the panel and the inspector, and now on the map too.
- **Multi-sensor:** verified pixel by pixel against its member layers (384 pixels per run).
- **Monitoring:** `actionlint` is clean, and the issue lookup creates an issue when none exists.
- **Size:** the dataset is 38 MB (currents 0.8 MB).

**Merged** as `1ef54b5` (merge commit).

**Production run 38019952037:**
- all 9 sources updated;
- C-HARM 156/156, satellite 408 checks, currents 32/32;
- 271 files plus 2,842 tiles validated before publishing, 258 at the public URL;
- health clear.

GitHub Pages deploy 38020732643 (deploy and check jobs passed) serves it, including 25 currents layers, the multi-sensor layer and `health.json`.

**Upstream revision.** After its reload, ERDDAP's `ucsdHfrW2` no longer lists 2026-10-09T21:00Z. Staging had published that hour; production's newest is 20:00. Upstream can retract hours, and the pipeline follows what upstream lists.

## 2. What changed in the Ocean Map

| Problem found in the screenshots | Change |
|---|---|
| C-HARM forecast percentages next to every port and region also showed with Satellite or Currents selected, where they read as chlorophyll or current values | Shown only while the HAB forecast is on the map, and labelled "C-HARM forecast …" |
| Observation and forecast times lived only in the dock, which is collapsed on phones | A **timestamp chip on the map**: one line per layer drawn, marked as observation or model, with its observation or valid time and age |
| Missing radar coverage was only a "0 %" line at the bottom of the dock | The chip says "No radar observations in North Coast this hour: no data, not calm water" |
| Arrow size and opacity scaled continuously, so speed couldn't be read off | **Five speed classes** (< 0.1, 0.1–0.25, 0.25–0.5, 0.5–1, ≥ 1 m/s): glyph length grows with class at constant line width; a light fill with a dark outline that reads over the dark sea and over bright chlorophyll; centred on the 2 km cell; the legend shows the same five glyphs |
| The map fit subtracted the dock's height, so tall panels zoomed the map out | The region is fitted into whichever free area is larger, beside the left column or above the dock. Monterey Bay is now framed larger at 1280 and 1440 |
| The dock could overlap the navigation card, and on phones cover the map | The navigation card's height follows the measured dock. The dock is capped at 70 % of the map height on desktop and 55 % of the screen on phones, scrolling inside. Collapsed panels show short versions |

**Not added:** no decorative effects. Chlorophyll stays opaque and nearest-sampled, so legend colours equal map colours. The coastline stays above all data.

## 3. Combined currents + chlorophyll (opt-in prototype)

- **How it works.** In the currents panel, "Show satellite chlorophyll underneath" (off by default; `chl=1` in the URL). The view draws the published Sentinel-3 300 m latest-clear-view tiles unchanged (native resolution, no-data hatching, per-pixel dates), with the HF-radar arrows one density level sparser on top.
- **What the panel states:**
  - chlorophyll is ocean colour (algae biomass), not toxin;
  - currents are observed surface motion, not a forecast;
  - **different times**, with both date ranges and the median gap in days;
  - arrows do not show where a bloom will travel.
  A collapsed panel shows the same four points in one sentence.
- **On the map:** the timestamp chip shows both layers' times and "Different times: chlorophyll a median N days older than the currents".
- **Inspector:** the satellite and currents readouts appear side by side, each with its own value, resolution and time.

**Explored:**
- **Chlorophyll opacity:** kept at 100 %. At lower opacity the colours no longer match the legend, which would undermine reading values.
- **Arrow density:** one level sparser over the imagery, so the 300 m pattern stays readable.
- **Zoom-dependent detail:** statewide, arrows every 32 km (64 km in the combined view); one per cell at zoom ≥ 9.6.
- **Contrast:** the dark-outlined light arrows stay legible over the brightest chlorophyll (screenshots 20–22).

**Does it improve interpretation?** Measured on production data, run 38019952037 ([evidence](p3/evidence/combined-view-time-gap.json)):
- **Half the radar cells have no chlorophyll:** of 10,351 radar cells in the newest hour, only **53 %** have any Sentinel-3 pixel in its 7-day window.
- **Where they overlap, the chlorophyll is old:** **94 %** of it was observed **3 or more days before** the currents hour (median 3 days; none on the same day).
- **Currents change in a day:** compared with 12–23 h earlier, the arrows turn a median **25–29°**, and **12–15 %** of cells turn more than 90°. Correlation is 0.68.

**Conclusion.** The two layers are rarely contemporaneous. The arrows show the water's motion at one hour, not the motion that shaped a chlorophyll pattern observed days earlier. The combined view is useful for spatial context, for example where radar and satellite overlap, or where fronts sit relative to a bay. But it invites a "the bloom is moving that way" reading the data cannot support. **Kept opt-in, not a default,** with the time gap stated on the map and in the panel. No user study was run; this is an evidence-based heuristic review.

## 4. Tests and performance

| Suite | Result |
|---|---|
| Pipeline `pytest` | 184 passed |
| Web unit (`vitest`) | **99 passed**, including timestamp lines (observation wording, gap lines, model marking) |
| Playwright e2e | **89 passed**, including `p3.spec.ts`: forecast values only with the forecast; the map states what and when; speed-class arrows match the legend; combined view opt-in with all four statements, published tiles unchanged, opaque nearest raster, sparser arrows; inspector reads both layers; missing coverage stated on the map. Axe clean on all checked pages |
| `tsc`, `eslint` | clean |

**Performance**, production data with a production build. The phone row uses a 4× CPU slowdown. Headless Chrome; frame rates are indicative.

| | Desktop 1440 | Phone 390 (4× slower CPU) |
|---|---|---|
| First map idle (cold) | 3.2 s | 1.0 s |
| Switch to satellite / currents | 0.15 s / 0.70 s | 0.23 s / 0.82 s |
| Combined view on | 0.69 s | 0.72 s |
| Previous hour | 0.78 s (14 KB) | 0.80 s |
| Statewide combined | 2.6 s (29 KB) | 1.8 s |
| Fly animation | 60 fps | 60 fps |
| Arrow points built in the browser (statewide hour) | 10,351 | 10,351 |

## 5. Screenshots (production data, 2026-10-10)

All are in [p3/screenshots/](p3/screenshots/):

| View | 1440 × 900 | 1280 × 720 | 390 × 844 |
|---|---|---|---|
| Satellite observations | [02](p3/screenshots/desktop-1440/02-map-satellite.png) | [02](p3/screenshots/laptop-1280/02-map-satellite.png) | [02](p3/screenshots/mobile-390/02-map-satellite.png) |
| Multi-sensor | [11](p3/screenshots/desktop-1440/11-map-multisensor.png) | [11](p3/screenshots/laptop-1280/11-map-multisensor.png) | [11](p3/screenshots/mobile-390/11-map-multisensor.png) |
| Currents alone | [14](p3/screenshots/desktop-1440/14-map-currents.png) | [14](p3/screenshots/laptop-1280/14-map-currents.png) | [14](p3/screenshots/mobile-390/14-map-currents.png) |
| Currents, 24 h mean | [15](p3/screenshots/desktop-1440/15-map-currents-mean.png) | | [15](p3/screenshots/mobile-390/15-map-currents-mean.png) |
| Combined currents + chlorophyll | [20](p3/screenshots/desktop-1440/20-map-combined.png) | [20](p3/screenshots/laptop-1280/20-map-combined.png) | [20](p3/screenshots/mobile-390/20-map-combined.png) |
| Combined, statewide | [21](p3/screenshots/desktop-1440/21-map-combined-statewide.png) | [21](p3/screenshots/laptop-1280/21-map-combined-statewide.png) | |
| Monterey Bay, port selected | [03](p3/screenshots/desktop-1440/03-map-port.png) · [18](p3/screenshots/desktop-1440/18-map-currents-point.png) · [22](p3/screenshots/desktop-1440/22-map-combined-point.png) | [03](p3/screenshots/laptop-1280/03-map-port.png) · [18](p3/screenshots/laptop-1280/18-map-currents-point.png) · [22](p3/screenshots/laptop-1280/22-map-combined-point.png) | [03](p3/screenshots/mobile-390/03-map-port.png) · [18](p3/screenshots/mobile-390/18-map-currents-point.png) · [22](p3/screenshots/mobile-390/22-map-combined-point.png) |
| Missing current coverage (North Coast) | [19](p3/screenshots/desktop-1440/19-map-currents-north-coast-gap.png) | | [19](p3/screenshots/mobile-390/19-map-currents-north-coast-gap.png) |
| Mobile layer controls, expanded | | | [23 currents](p3/screenshots/mobile-390/23-mobile-controls-expanded.png) · [24 satellite](p3/screenshots/mobile-390/24-mobile-satellite-expanded.png) |
| Forecast | [01](p3/screenshots/desktop-1440/01-map-forecast.png) | [01](p3/screenshots/laptop-1280/01-map-forecast.png) | [01](p3/screenshots/mobile-390/01-map-forecast.png) |

## 6. Remaining limitations

**Scientific:**
- **Coverage:** HF radar has none on the North Coast today. Sentinel-3 coverage is fog-limited, 29 % of Monterey Bay in 7 days.
- **Sensor disagreement:** Sentinel-3 and VIIRS differ by a factor of 2–5 on the same day.
- **Forecast currents:** WCOFS has no skill over persistence, and no transport or bloom outlook exists.
- **Combined view:** its layers are typically about 3 days apart.

**Product:**
- At 1280 × 720 with a port selected, three panels compete with the map. The dock collapses on short screens, but the bay is still framed small.
- Statewide, the arrows are sparse by design.
- The combined view uses Sentinel-3 only, not the multi-sensor view, to keep one native resolution.

**Regulatory:** two new razor-clam notices for Del Norte County, from CDPH SN26-020 and CDFW, are **not in the registry** and need a human (see the checklist). The app still shows every record as not verified.

**Reliability:** NOAA's ERDDAP 403s to runners, dataset reloads and retracted hours. The outreach draft is ready; it's yours to send.
