# Research: v11 → v12 capability checklist

v11 sources: `coastwatch-web/src/v4/research/**` (`Research.tsx`, `exports.ts`, `ObsTools.tsx`, `Lineage.tsx`,
`CoastLadder.tsx`, `Stations.tsx`), the orphaned `/bloom` page, and research C §3 (capability inventory) and §3
"Deep links that must keep resolving". v12 code: `coastwatch-web/src/nx/research/**`, route
`src/app/(neritic)/research/page.tsx`. Proof: `tests/unit/nx/research/*.test.ts` (31) and
`tests/e2e/journeys/research.spec.ts` (5; the journey is recorded as `journey-research-1440.webm`).

Result: **29 of 31 kept or improved, 2 kept in part** (listed at the end). Nothing was dropped silently.

| # | v11 capability | v12 | Status |
|---|---|---|---|
| 1 | Station record chart, one quantity at a time (`LinkedSeries`) | Every quantity the station measures as strips on one time axis, one SVG (`RecordStrips`) | Improved |
| 2 | Quantity selector (pDA, dDA, tDA, two Pseudo-nitzschia groups, chlorophyll, temperature) | Strip names are the selector (`aria-pressed`); the focus drives the brush overview, the log, the model strip and `var` | Kept |
| 3 | Range presets 90 d / 1 y / 3 y / All, custom window | Same presets (ds `Segmented`); custom windows by brush | Kept |
| 4 | Brush strip under the chart | ds `Brush`, aligned to the strips' plot (same 84 px gutter); keyboard sliders; preset and keyed changes `settle`, drags follow the hand | Improved |
| 5 | Crosshair shared with the model panel; pinned day (`day`) | One crosshair through every strip by pointer, arrow keys, Home/End; the day flagged on the time axis; a log row pins `day` | Improved |
| 6 | Linked C-HARM model panel: history, partial days open, hatched before history | Model strip: hollow, dashed, joined day to day only, partial days open circles, hatched "No model history kept before …"; follows the focus (P(pDA > 500 ng/L) or P(Pseudo-nitzschia > 10,000 cells/L)) | Kept |
| 7 | "No C-HARM history" note with the pipeline's error | Model row says so and quotes the pipeline's error | Kept |
| 8 | C-HARM's own thresholds as reference lines | Dashed line on its own water scale, named in the strip head (T6) | Kept |
| 9 | Fixed axes, never rescaled to the data | Whole-decade log axes fixed to the station's record; the two Pseudo-nitzschia groups share one | Kept |
| 10 | Reported 0 drawn apart, rejected as cross, flagged high ringed | Same three marks, in each strip's own lane; one legend per view lists only the marks in view | Kept |
| 11 | "In this window" stats: samples, reported 0, median, highest | Window row: highest with its date beside the median (T8), reported 0, rejected, flagged | Kept |
| 12 | Provenance: method, matrix, units, zero policy, detection note, QC rules, retrieved, pipeline run | Source row (plus request URL and licence) and the ds `Stamp` → `ProvenanceSheet` with the caveats | Improved |
| 13 | Station checks with pass/fail and details | Checks row, "Passed"/"Failed" in words, the QC rules, the station's request status | Kept |
| 14 | Citation and licence | Source row (licence) and Export row: a one-line citation of exactly the window with a link back, copyable | Improved |
| 15 | Table view of the window | Samples row: dated log of the focused quantity, tabular figures, sortable by date or value, row ↔ crosshair | Improved (see partial A) |
| 16 | Export: this window (CSV, one quantity, `#` provenance header) | Primary action: every quantity in the window, one row per sample, source columns on every row, `#` header | Improved |
| 17 | Export: model history (CSV) | Model days in the window (CSV); "All" gives the whole kept history | Kept |
| 18 | Original data at SCCOOS | Export row link to the station's own request | Kept |
| 19 | Native grid: C-HARM cell, transect, area; cell edges; the run's days | "Grid" row opens v11's instrument in a dialog (`grid/NativeGrid.tsx`), loaded only then | Kept (see partial B) |
| 20 | Native grid on satellite chlorophyll and HF radar (pixel, transect, area, dates, sensors) | Same, in the dialog | Kept |
| 21 | Cell, transect and area CSVs | Same functions (`src/v4/research/exports.ts`) | Kept |
| 22 | Lineage and citation of a grid layer | In the dialog, under "Lineage and citation of this grid" | Kept |
| 23 | Station picker: coast ladder north to south with last sample; locator map with the 15 km ring | Stations written on the persistent chart where they sample (names are buttons); the north-to-south list with each newest sample; the dashed 15 km model ring around the open station; the chart pans to a new station | Improved |
| 24 | "Arrived from X · nearest station Y, N km"; Back (`from=/path`) | Same; past 60 km it says no station is near (research C problem 10); Back kept | Improved |
| 25 | A requested quantity silently swapped when not measured | Said in words: "pDA isn't measured at …"; an unreadable `var` is quoted | Fixed |
| 26 | Open in Explore; My Coast for the nearest port | Station row: "… in Explore" (`/explore?sel=station:…`), "Plan a trip from …" | Kept |
| 27 | "How to read this" popover | The lead sentence, one legend, words in each row and strip head (no popover) | Kept in other form |
| 28 | Section hashes `#r-stats #r-tools #r-provenance #r-lineage #r-exports` | Open the matching margin row | Kept |
| 29 | Deep links `station place variable day start end range from` | All resolve (`parseQuery`), then the address is rewritten to v12's `station from to var` (+ `place`, `day`, `back`) | Kept |
| 30 | `/bloom?station&var&range&year` | `bloomRedirect` (shell) → `/research?…variable…`, which Research reads | Kept |
| 31 | Landings history (from `/fisheries`, research C target "Research") | Landings row links to `/fisheries` (statewide, so not on the chart) | Kept |

Also kept: keyboard everywhere (crosshair, brush sliders, sortable headers, rows), reduced motion (no ink, no
settle, no pan), dusk and night modes, the designed unavailable state when the dataset can't be read.

Not carried: v11's readiness reporting into its trust store (`useReportSource`) on the record itself; the v12
shell has no such store (the grid dialog still reports inside its own provider).

## Kept in part

- **A. Model value per row in the table.** v11's table had a model column. The v12 log lists the focused
  quantity only; the model value of any day is read with the crosshair, and the model CSV holds every day.
- **B. Native grid styling.** The instrument is v11's code with the v4 theme tokens mapped onto the chart mode's
  (`grid/grid.module.css`), and its own small MapLibre map in the dialog. Restyling it natively, and moving cell
  reads onto the persistent chart, is a follow-up (it needs the chart host to accept click and overlay layers).

## The address Explore links with

`/research?station=<id>&from=YYYY-MM-DD&to=YYYY-MM-DD&var=<quantity>`. `station` takes the dataset id
(`HABs-SantaCruzWharf`), the id without `HABs-`, or the name. `var` takes a station quantity (`pDA`, `dDA`, `tDA`,
`pn_seriata`, `pn_delicatissima`, `chl_extracted`, `temp`), a C-HARM variable (`particulate_domoic`,
`cellular_domoic` → pDA; `pseudo_nitzschia` → seriata group) or an Explore chlorophyll or SST layer
(`chl`, `olci300`, `viirs750`, `multi`, `chl_estimate`, `outlook` → extracted chlorophyll; `sst` → temperature).
A single day (`from` = `to`) opens 90 days around it. Add `place=<port-N | point-lat,lon>` to have the page name
the nearest station and its distance.
