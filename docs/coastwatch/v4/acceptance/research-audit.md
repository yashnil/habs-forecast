# D — Research workspace audit (read-only)

App root: `scratchpad/v4/coastwatch-web`. All paths below are relative to it.
Files read: `src/v4/research/Research.tsx` (432 l), `src/v4/research/transect.ts`, `src/app/research/page.tsx`,
`src/v4/explore/{Explore,Inspect,reads}.tsx/ts`, `src/v4/charts/SeriesChart.tsx`, `src/lib/{observations,bloom,grid,layers,currents,readout}.ts`,
`tests/e2e/research.spec.ts`, fixture `public/data/fixture/v1/{manifest.json,observations-*.json,verification/*}`.

## What data actually exist (fixture v1, schema frozen)

| Data | Content | Usable for |
|---|---|---|
| `observations-*.json` | 17 CalHABMAP stations; per station `sample_times`, 7 series (pDA, dDA, tDA ng/mL; pn_seriata, pn_delicatissima cells/L; chl_extracted mg/m3; temp degree_C), `qualifiers` map (index → `reported_zero`/`rejected_negative`/`flag_high`; fixture has only 8 364 `reported_zero`), `summaries`, station `qc` (6 checks), `source_url`, `request_url`, `retrieved_at`, `depths_m` (all null). Dataset `method`, `caveats`, `provenance` (citation, licence). Variable meta: method, matrix, fraction, units, zero_policy, detection_limit (null), plausible_max. | Station histories, QC, provenance, CSV export |
| `station.charm` | `radius_km` 15, `nearest_cell_km`, `history_days` 180, `history.{pseudo_nitzschia,particulate_domoic,cellular_domoic}` = daily `{date, value, n}` (nowcast median of cells within 15 km). Present for **2 of 17** stations in the fixture (Santa Cruz, Monterey: 147 days 2026-04-11…10-07); the rest carry `error`. | Model-beside-observation timeline |
| C-HARM layers | **One run** (issued 2026-10-08, derived), leads 0–3 × 3 variables = 12 native grids (0.03°, 51×81, uint16, `max_quantization_error` 7.6e-6). Each layer: `provenance` (source, dataset `wvcharmV3_0day`, version 3.1, request URLs, upstream metadata incl. *"DINEOF gap filling was applied to daily VIIRS chlorophyll…"*), `time` (issued_date_derived + method), `qc.checks` (17), `caveats` (no v3.1 skill assessment; many false positives), `threshold_text`. | Native grid inspection, lineage, QC |
| Satellite chl | `olci300_chl_latest` (chunked 300 m grid + `composite.age_grid` + per-day pixels_used), `olci300_chl_2026-10-0{1,2,5,6}` daily grids (10-03/04/07 published with `grid:null`, 0 % valid), `viirs750_chl_latest`, `multisensor_chl_latest` (tiles + rule), GIBS tiles. Units mg m-3, log10 transform. QC (e.g. `plausible_values` failed for 10-01). `coverage` by region. | Point/area/time inspection of observations |
| HF-radar currents | 24 hourly u/v grids (2 km, 41×44) 2026-10-07T13Z…10-08T12Z + `mean24h` (≥18 of 24 h rule), m s-1. | Point time series (24 h), lineage |
| `verification/*.json` | Pipeline self-consistency (tile vs grid, charm points); `live_source_comparison:false`. Not skill. Not referenced by the manifest or the app. | Lineage/QC evidence only, not skill |
| Skill numbers | None in data. C-HARM caveat: *"Published skill assessment covers C-HARM v1 (Anderson et al. 2016); no published skill assessment for v3.1 was found."* | Quote the caveat, no number |

## Requirement-by-requirement status

| # | Requirement | Status | Evidence | Interactive? |
|---|---|---|---|---|
| 1 | Spatial point/line/area | **PARTIAL** | Research: tool toggle cell/transect `Research.tsx:144-146,333-339`; click → `setCellAtPt` or 2-point line `:359-362`; transect sampler `transect.ts:11-27`, chart `Research.tsx:389-432`. Area exists **only in Explore** (`Explore.tsx:244-246,278-294,470,560-577`; `AreaView` `:785`; `areaStats` `Inspect.tsx:414`). All selections are C-HARM only; satellite/currents never sampled in Research (`satellite={null}` `Research.tsx:344`; Explore area says "Satellite and current statistics … are not computed"). | Yes for cell + transect (C-HARM, lead 0). No area in Research. Transect samples at 1/6-cell steps (`transect.ts:13`), so a cell clipped only at a corner can be skipped. |
| 2 | Valid source-specific time ranges | **PARTIAL** | Presets 90d/1y/3y/All anchored to the station's own last sample and clamped to its first (`Research.tsx:33-38,125-129`); model history before its 180-day start hatched (`:67-74`). Grid fixed to lead 0 (`:148`), no lead control; no satellite day / currents hour navigation in Research. No custom/brushed range; the selected window's dates are not printed. | Presets yes; grid time no. |
| 3 | Unit-bearing charts | **PARTIAL** | `SeriesChart` fixed decade log axes (`bloom.ts:24-32`), units in readout/aria title/table header (`SeriesChart.tsx:100-101,128,170`), but **no y-axis title** on the plot. Research passes raw `vmeta.units` (`Research.tsx:241,260,280`) so temp shows "degree_C" and chl "mg/m3" (`unitLabel` exists, `observations.ts:57`, unused). ModelPanel: % ticks but **no time-axis labels** and no readout (`Research.tsx:58-86`). Transect: % ticks, 0/total km only, no endpoint coordinates (`:399-425`). | SeriesChart yes (pointer + arrow keys); ModelPanel/transect static. |
| 4 | Station histories | **IMPLEMENTED** (core) | 17-station list with last date + visit count (`Research.tsx:176-204`, URL `?station=` `:139`), variable picker restricted to variables with data (`:118,226-232`), chart + table, window stats incl. reported zeros (`:245-271`). Gaps >21 d not joined (`bloom.ts:127`). Missing: per-sample `flag_high`/`rejected_negative` marks (chart only special-cases `reported_zero`, `SeriesChart.tsx:154-155,177`); one variable at a time. | Yes |
| 5 | Model vs observation | **PARTIAL (valid design)** | `ModelPanel` puts C-HARM nowcast median (15 km) on the same x-window, never on the obs axis, "never compared with them as numbers" (`Research.tsx:43-90`); partial days open circles (`bloom.ts:226-266`). Only pDA/PN show it (`:242`); variable mapping `:138`. Data exist for 2/17 stations. No synchronised cursor; no PN 10,000 cells/L reference (pDA 500 ng/L ref only, `:41`). No skill metric — correct. | Static panel |
| 6 | Provenance + lineage | **PARTIAL (observations only)** | Variable method/matrix/units/zero policy/detection, retrieved time (`Research.tsx:272-291`), SCCOOS original-CSV link (`:320-324`, `observations.ts:100`). **Not shown anywhere in v4:** C-HARM `layer.provenance` (dataset id, version, request URLs, upstream DINEOF gap-filling note), `issued_date_derived/method`, layer QC, manifest `pipeline_version/run_id/generated_at`, dataset citation/licence (`grep` of `src/v4` finds no `.provenance`/`.caveats` use; only legacy `components/panels/ForecastPanel.tsx:194-236`). | Text only |
| 7 | QC flags + uncertainty | **PARTIAL** | Station QC checks list (`Research.tsx:292-303`); zero policy; partial model days. Missing: per-sample qualifiers in chart/table, layer `qc.checks` (incl. failed `plausible_values` on OLCI), C-HARM caveats (false positives, salinity, no v3.1 skill), quantisation error, `n` cells per model day. Cell value rounded to whole % (`Inspect.tsx:298-300`). | No |
| 8 | Data exports | **PARTIAL** | One CSV: station × variable × window with provenance header, empty = not measured, qualifier column (`Research.tsx:92-102,305-319`; tested `research.spec.ts:25-34`). Missing: C-HARM history, cell (4 leads), transect, area, native-grid artifact + decoding block; header lacks lat/lon, request URL, units conversion, pipeline run id, licence. | Yes (1 export) |
| 9 | Native forecast grid inspection | **PARTIAL (good core)** | Map renders lead-0 C-HARM image + grid (`Research.tsx:342-363`); cell click → `CellBody` (`Inspect.tsx:302-378`): value, footprint polygon, all 4 leads of the run, gap + nearest valued cell, harbour relation, row/col/centre. Missing: lead and C-HARM-variable selectors (cellular_domoic unreachable; variable is derived from the obs variable `:138`), cell-edge overlay (Explore passes `cellEdges`, Research does not), raw probability + quantisation error. | Yes |

### Honest judgement
Research today is a **competent station-record viewer plus a C-HARM point/transect probe**, not yet a scientific workspace. What is there is scientifically careful (fixed axes, gaps kept, zeros qualified, no model/obs arithmetic, no interpolation), and genuinely interactive for station history, cell inspection and a two-point transect. It falls short on: lineage/QC for the gridded products (the most "scientific" metadata in the manifest is never surfaced in v4), per-sample flags, time control over the grids (lead fixed; satellite/currents absent), area selection, unit labelling on axes, and exports beyond one CSV. About 4 of 9 requirements are usable; none of the gaps needs new data.

## Plan: close the gaps within the existing data (ordered by scientific value)

Each item: user action · data/helper · files · guardrail wording · test.

**1. Lineage, QC and uncertainty for every product shown (req 6, 7).**
- User: opens a "Sources and lineage" panel beside the grid and the station record; expands each product.
- Data: `layer.provenance` (source, dataset_id, product_version, institution, retrieved_at, request_urls, `upstream_metadata.history_first_line`), `layer.time.issued_date_derived/issued_date_method`, `layer.qc.checks/n_valid/valid_fraction`, `layer.caveats`, `grid.max_quantization_error`, `manifest.pipeline_version/pipeline_run_id/generated_at`, `observations.provenance` (citation, licence), `observations.method`.
- Files: new `src/v4/research/Lineage.tsx`; mount in `Research.tsx` (provenance section ~`:272`, grid section ~`:329`).
- Wording: "Issue date inferred by Neritic (nowcast valid day + 1); C-HARM publishes no issue time." "Upstream: the producer gap-fills its satellite inputs (DINEOF) before running the model; Neritic does not fill gaps." "No published skill assessment for C-HARM v3.1; the published assessment covers v1." Show failed checks as failed, never hide them.
- Test (`research.spec.ts`): the panel contains `wvcharmV3_0day`, "inferred", "DINEOF", every `qc.checks` name for the shown layer, and the pipeline run id.

**2. Per-sample quality flags and proper units (req 3, 4, 7).**
- User: sees flagged-high samples ringed, rejected samples as a distinct floor mark labelled "rejected (QC)" (never as a value or zero); the table gains a qualifier column; y-axis titled "pDA (ng/mL, log)"; °C and mg/m³.
- Data: `series.qualifiers`, `formatObs`/`unitLabel` (`observations.ts:42-59`), `plausible_max`.
- Files: `src/v4/charts/SeriesChart.tsx` (marks, table, axis title), `Research.tsx:241,260,280` (use `unitLabel`).
- Wording: "Kept but flagged: above the plausibility ceiling (value as published)"; "Rejected by QC (negative concentration): not a measurement."
- Test: unit test rendering `SeriesChart` with a synthetic station holding `flag_high` and `rejected_negative` (fixture has none); e2e: a station's temp chart readout shows "°C" and not "degree_C".

**3. Native grid controls (req 9, 2).**
- User: picks C-HARM variable (all three, defaulting to the one matching the measured variable) and a lead from the run's own leads (labelled nowcast/forecast + valid date); toggles cell edges; the cell readout shows the probability to 3 decimals with ± quantisation error.
- Data: `charmRun().leads_available`, `charmLayer()`, `cellEdges` (`grid.ts:231`), `useCellRead` (already loads all leads).
- Files: `Research.tsx:143-170,329-379`; `Inspect.tsx` `CellBody` (optional `precise` prop).
- Wording: "Model output for one 3 km cell, not a toxin measurement. Forecast days are only those this run published; nothing beyond its horizon."
- Test: choose lead 2 → heading shows its valid date and "forecast"; `cellular_domoic` selectable; precise value present.

**4. Area selection in Research (req 1).**
- User: third tool "Select an area": drag a box; sees counts of valued/unvalued cells, min/median/max of published probabilities, stations and harbours inside.
- Data: `areaStats` (`Inspect.tsx:414`), Explore's drag overlay (`Explore.tsx:278-294,560-577`).
- Files: extract the overlay into `src/v4/explore/AreaDrag.tsx` (shared), reuse `AreaView` body in `Research.tsx`.
- Wording (already in Explore): "A summary of the published cell probabilities inside the box: not an average for any harbour and not a measurement. Cells without a value are counted, never filled."
- Test: drag a box → `inspect-area` shows "of N grid cells have a published value".
- Also fix the transect sampler to an exact grid traversal (or step ≤ 1/20 cell) so every crossed cell is listed; unit test a diagonal line that clips a cell corner.

**5. Exports for every selection (req 8).**
- User: "Export" on the model history, the cell (4 leads), the transect, the area cell list, plus "Native grid file" (link to the `.u16.gz` and a JSON of its manifest `grid` block for decoding).
- Data: `st.charm.history` (date,value,n,partial), `CellRead.leads`, `TransectCell[]`, area cell list, `layer.grid`.
- Files: new `src/v4/research/exports.ts` (one header builder shared with `csvFor`), buttons in `Research.tsx`.
- Wording in every header: product class ("model output" vs "observation"), units, source dataset id + licence, pipeline run id, "empty = no published value (never zero, never interpolated)", "model probability, not a toxin measurement", "not a seafood test". Station CSV header adds lat/lon and the SCCOOS request URL.
- Test: unit tests on each CSV header/columns; e2e download of the transect CSV contains `km,row,col,lat,lon,value,in_domain` and an empty value for a gap.

**6. Model beside observation, done properly (req 5).**
- User: one shared time cursor across the measurement chart and the model panel (readout shows "Measured: 0.52 ng/mL on Sep 30" and separately "C-HARM nowcast that day: 71 % (median of 26 cells within 15 km)" or "no model value that day"); ModelPanel gets month labels; PN charts get C-HARM's own 10,000 cells/L reference line (from `threshold_text`).
- Data: `modelOnAxis` (`bloom.ts:226`), station samples.
- Files: `SeriesChart.tsx` (lift cursor to a prop), `Research.tsx:43-90,241-242`.
- Wording: "Shown on the same dates, never compared as numbers: a pier water sample and a model probability over 3 km cells within 15 km are different quantities. No skill score is computed; no published skill assessment exists for C-HARM v3.1."
- **Invalid, do not build:** correlation, hit rate/contingency table, ROC/AUC, residuals or "model was right/wrong" between C-HARM cell probability and pDA/PN samples (point vs 15 km median, probability vs concentration, no same-pipeline skill number). A paired "model vs measured" table is also discouraged — the shared cursor gives the same lookup without inviting arithmetic.
- Test: cursor readout contains both labelled lines; the page contains no `/correlat|skill score|accuracy|hit rate|R²/i`.

**7. Observation grids in the workspace (req 1, 2, 9 for observations).**
- User: switches the map product to satellite chlorophyll (latest + each published OLCI day) or HF-radar currents (24 hourly frames + 24 h mean); clicks a point to read the exact pixel/cell value with its own observation date/age; sees a per-pixel day strip (Oct 1–7, days with no grid or no value shown as gaps) and a 24-hour current speed/direction strip.
- Data: `readout.ts` (exact cell, never nearest), `sampleChunked`/`chunkedCode` (`grid.ts:146-195`), `composite.age_grid`, `satelliteDays()`, `currentsHourly()`/`loadField`/`sampleField` (`currents.ts`).
- Files: new `src/v4/research/ObsGridPanel.tsx`; `Research.tsx` map props.
- Wording: "Observation, not a forecast." "Chlorophyll is algal biomass, not toxin and not Pseudo-nitzschia." "A cloud gap is no observation, not low chlorophyll; gaps are never filled." "Observed surface currents; not where a bloom will go."
- Test: click a water pixel → value in mg m⁻³ with observation date; a day with `grid:null` appears as a labelled gap.

### Needs new data, or is invalid
- **Forecast verification (lead-k forecast vs later nowcast)** — needs archived C-HARM runs; the manifest holds one run. New data.
- **C-HARM history at the other 15 stations** — the fixture has it at 2; elsewhere the pipeline reports an error/outside domain. Show the published `charm.error` reason; don't recompute.
- **C-HARM uncertainty intervals** — not published; show only quantisation error + caveats.
- **Satellite chl vs station `chl_extracted` matchups** — no temporal overlap in the fixture (last sample 09-30, satellite 10-01…10-06); pier vs 300 m pixel adjacency bias; it would be a derived metric not from the pipeline. Side-by-side only.
- **Vertical profiles** — `depths_m` all null. **Detection-limit lines** — no limits published.
- **Any skill number** — none in data; quote the v1/v3.1 caveat verbatim, no figure.
