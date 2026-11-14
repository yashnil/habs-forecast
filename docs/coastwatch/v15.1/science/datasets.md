# Neritic v15.1: scientific dataset audit

Audited 2026-11-13 by reading the published manifests over HTTPS (not the fixtures):

- production: `https://raw.githubusercontent.com/yashnil/habs-forecast/coastwatch-data/v1/manifest.json`,
  `generated_at` **2026-11-12T07:14:30Z**, 48 layers.
- staging: `https://raw.githubusercontent.com/yashnil/habs-forecast/coastwatch-data-staging/v1/manifest.json`,
  `generated_at` **2026-11-09T09:57:26Z**, 48 layers. Staging is three days *older* than production: production
  runs every 6 h (`17 */6 * * *`, `.github/workflows/coastwatch-data.yml`); staging is written only by manual
  `channel: staging` runs (last run 42597191999, pipeline 33411500b447).
- `derived/index.json`: **200 on staging** (generated 2026-11-09, four layers), **404 on production**.
- the observations file named by each manifest (`observations_url`), the bundled NOAA station lists
  (`coastwatch-web/src/content/stations/`), `src/lib/conditionsServer.ts`, `src/nx/explore/**`,
  `src/nx/chart/**`, and the pipeline (`pipeline/coastwatch_pipeline`, read only).

Status words: **live** = in the production manifest and read by the app; **staging only** = published only on
`coastwatch-data-staging`; **not published** = only on a branch or in a fixture; **live read** = fetched by the
app's server at request time from the agency (nothing published by our pipeline).

## Gridded and tiled layers

| Dataset (layer ids) | Scientific meaning | Units | Coverage / grid | Cadence | Temporal resolution | Source | Latest (production / staging) | Status | In Explore |
|---|---|---|---|---|---|---|---|---|---|
| Satellite chlorophyll-a, Sentinel-3 OLCI 300 m, latest clear view (`olci300_chl_latest`) | Near-surface algae biomass (chlorophyll-a concentration). Not toxin, not Pseudo-nitzschia, not fish. | mg m⁻³ (log10-encoded u16) | 125.5–117.0 °W, 32.4–42.1 °N; 3400 × 3880 cells at 0.0025° (≈ 280 m), chunked 512²; coastal ocean only (bays within ≈ 12 km kept, lakes out); 80 % of C-HARM ocean cells observed in the 7-day window | Pipeline every 6 h; one satellite pass per day per platform (S-3A, S-3B) | Composite: each pixel its own newest clear day within 7 days (per-pixel age grid); clouds and fog are gaps, never filled | NOAA CoastWatch ERDDAP `noaacwS3[A/B]OLCIchlaSector[CI/DI]Daily` (Copernicus/EUMETSAT) | pixels 2026-11-05 … **2026-11-10** (ref. 2026-11-12) / pixels … 2026-11-07 | **live** | yes: Chlorophyll, frame "Latest" |
| Same sensor, single days (`olci300_chl_YYYY-MM-DD`, 6 per run) | One day's clear pixels | mg m⁻³ | Same grid; day coverage varies 0–100 % by region (e.g. Nov 8: S. California 0 %) | as above | One calendar day | as above | 2026-11-05 … **2026-11-10** / 2026-11-02 … 2026-11-07 | **live** | yes: day frames on the time rail, playable |
| Satellite chlorophyll-a, VIIRS 750 m, latest clear view (`viirs750_chl_latest`) | as above, coarser sensor, different algorithm | mg m⁻³ | 1133 × 1293 at 0.0075° (≈ 830 m); 95 % of C-HARM ocean cells | daily composite | 7-day latest clear view, per-pixel age | NOAA CoastWatch West Coast `erdVHNchla1day` (Suomi-NPP; fallback while NOAA-20/21 are unavailable) | pixels 2026-11-05 … **2026-11-07** / … 2026-11-07 | **live** | yes: second track "VIIRS 750 m" |
| Multi-sensor chlorophyll display (`multisensor_chl_latest`) | OLCI where present, VIIRS elsewhere; display only | mg m⁻³ | Tiles z5–11 over 125.5–117.0 °W, 32.4–42.1 °N; no value grid | daily | 7-day latest | Rendered by Neritic from the two above | 2026-11-10 / 2026-11-07 | **live** (tiles only) | no (Explore reads values; this has none) |
| NASA GIBS VIIRS NOAA-20 chlorophyll (`gibs_viirs_noaa20_chl`) | Browse imagery for visual context | mg m⁻³ (NASA legend; values not readable) | Global WMTS, native z7 (≈ 1 km) | daily | One day | NASA GIBS `VIIRS_NOAA20_Chlorophyll_a` | **2026-11-10** / 2026-11-07 | **live** (browse tiles) | no (no values: cannot carry a readout) |
| NASA GIBS PACE OCI chlorophyll (`gibs_pace_oci_chl`) | as above | as above | as above | daily | One day | NASA GIBS `OCI_PACE_Chlorophyll_a` | **2026-11-11** / 2026-11-08 | **live** (browse tiles) | no |
| C-HARM v3.1 Pseudo-nitzschia (`charm_pseudo_nitzschia_lead0..3`) | **Model** probability of a Pseudo-nitzschia bloom (> 10⁴ cells/L). Not a measurement, not toxin in seafood. | probability 0–1 | 127.5–117.0 °W, 31.3–43.0 °N; 351 × 391 at 0.03° (≈ 3 km); 57 % of cells valued | NOAA runs daily, often missing; ingested every 6 h | One day per lead: nowcast (lead 0) and +1, +2, +3 days | NOAA CoastWatch ERDDAP `wvcharmV3_1day` | run issued **2026-11-03** (inferred: nowcast + 1 d), valid 2026-11-02 … 11-05 / same run | **live** — but a **past run** on 2026-11-13: every lead is before today; freshness policy says historical after 7 days | not drawn (Explore draws the pDA variable only) |
| C-HARM v3.1 particulate domoic acid (`charm_particulate_domoic_lead0..3`) | **Model** probability that particulate DA > 500 ng/L | probability 0–1 | as above; no values in many cells within ≈ 3–6 km of shore | as above | as above | as above | as above | **live**, past run | yes: "C-HARM" layer, leads as rail frames, drawn as classes, lighter when past |
| C-HARM v3.1 cellular domoic acid (`charm_cellular_domoic_lead0..3`) | **Model** probability that cellular DA > 10 pg/cell | probability 0–1 | as above | as above | as above | as above | as above | **live**, past run | not drawn |
| HF-radar surface currents, hourly (`hfr2km_currents_YYYYMMDDTHHZ`, 24 per run) | **Observed** surface water movement (top ≈ 1 m), u (east) and v (north) | m s⁻¹ | 125.49–118.2 °W, 32.41–42.1 °N; 408 × 540 at 0.018° (≈ 2 km); 5 % of grid cells valued per hour; no North Coast coverage, 12 % of C-HARM ocean cells | Hourly snapshots; pipeline keeps the last 24 | One hour (instantaneous snapshot) | HFRNet real-time vectors via NOAA CoastWatch ERDDAP `ucsdHfrW2` | 2026-11-11T06Z … **2026-11-12T05Z** / 2026-11-08T09Z … 2026-11-09T08Z | **live** | yes: "Currents" switch, animated streaks, hour rail, playable |
| HF-radar 24-hour mean (`hfr2km_currents_mean24h`) | Mean of each cell's valid hours (≥ 18 of 24); suppresses tide/sea breeze | m s⁻¹ | as above | each run | 24 h mean | as above | to 2026-11-12T05Z / to 2026-11-09T08Z | **live** | yes: drawn still, labelled "24 h mean" |
| SST, JPL MUR v4.1 (`derived/index.json` → `sst_latest`) | Gap-free **analysis** of sea-surface temperature (blends satellites and buoys) | °C | statewide | daily, ≈ 2 days latency | one day | NOAA CoastWatch West Coast ERDDAP `jplMURSST41` | — / **2026-11-07** | **staging only** | shown only where derived layers are enabled (see below) |
| 8-day chlorophyll outlook (`chl_outlook_8d`) | Neritic **experimental** statistical outlook of mean chlorophyll; not observed, not a HAB/toxin forecast | mg m⁻³ | statewide 0.0025° chunked | per staging run | 8-day mean window | Neritic, from NASA MODIS-Aqua 8-day | — / issued 2026-11-09, window 11-05 … 11-12 (already past on 2026-11-13) | **staging only** | as SST |
| Gap-filled chlorophyll estimate (`chl_estimate_latest`) | Estimated chlorophyll where clouds/shore hide it; not observed | mg m⁻³ + uncertainty | statewide | per staging run | latest | Neritic optimal interpolation over OLCI | — / issued 2026-11-09 | **staging only** | no (not read by v15 Explore) |
| C-HARM nearshore extension check (`charm_nearshore_extension`) | Validation record (nearest C-HARM cell vs pier samples) | — | — | — | — | Neritic | — / 2026-11-09 | **staging only** | no |

## Point datasets (stations)

| Dataset | Scientific meaning | Units | Coverage | Cadence | Temporal resolution | Source | Latest (production / staging) | Status | In Explore |
|---|---|---|---|---|---|---|---|---|---|
| CalHABMAP shore stations (`observations-*.json`, 17 stations) | **Measured** water samples at piers: particulate / dissolved / total domoic acid, Pseudo-nitzschia seriata and delicatissima groups, extracted chlorophyll, temperature | DA ng/mL; PN cells/L; chl mg/m³; °C | 17 piers, Trinidad (41.05 °N) to Scripps (32.87 °N); none on the coast between Tomales Bay and Santa Cruz or north of Trinidad | Typically weekly per station; many stations silent for months | One sample (time-stamped) | CalHABMAP via SCCOOS ERDDAP (`HABs-*`) | newest pDA 2026-10-28 (Santa Cruz), PN 2026-11-08 (Cal Poly), chl 2026-11-08; North Coast pDA last 2026-03/04; Monterey Wharf pDA last 2022-08-31 / staging newest PN 2026-11-03 | **live** | yes (v15.1): "Stations" layer, latest dated pDA / PN per station; the slim station list carries pDA and the two PN groups only |
| CDPH shellfish biotoxin sampling (`biotoxin/` sidecar) | Measured PSP/DA in shellfish tissue | µg/100 g, ppm | sampled sites | weekly-ish | sample | CDPH recorded PDF | fixture only on this branch (pilot sidecar) | **not published** (pilot sidecars, staging) | no |
| NDBC buoys and C-MAN (82 stations bundled, `src/content/stations/ndbc-ca.json`) | **Measured** wind, waves, water temperature, pressure | kt, ft/m, s, °C | California coast and offshore | Hourly–10-min; the app reads the newest 6 h | one observation | NOAA NDBC `realtime2/*.txt` | read at request time (10 min cache) | **live read** (`/api/conditions`) | yes (v15.1): "Stations" layer, each buoy clickable; the readout is that station's newest observation, or names the nearest reporting station when it has none in 6 h |
| CO-OPS tide predictions and water temperature | Predicted tide (MLLW); measured water temperature at tide gauges | ft; °C | tide gauges | 6 min / predictions | — | NOAA CO-OPS API | request time | **live read** | no (Today/My Coast only) |
| NWS marine forecasts / alerts | Forecast text, warnings | — | marine zones | per issuance | periods | NWS api / tgftp text | request time | **live read** | no (Today/My Coast only) |
| CDFW ports, official notices, port intel, FOSS landings | Regulatory / economic records | — | — | 6 h (landings monthly) | — | CDFW, PSMFC | 2026-11-12 (landings 2026-10-27) | **live** | no (not ocean science) |

## What Explore California Waters shows (v15.1)

Live and drawn from real published values only: OLCI and VIIRS chlorophyll (latest view and day frames), C-HARM
particulate-DA probability (leads), HF-radar currents (hours, 24 h mean), CalHABMAP stations and NDBC buoys. Each
says *observation* or *model*, its units, and its own dates.

Not shown as live: SST, the 8-day outlook and the gap-filled estimate. `src/nx/explore/load.server.ts` reads
`derived/index.json` only when `CW_DERIVED_LAYERS=1`, or from a local fixture path (the e2e fixture keeps the v13
tests of those layers). On any deployed data (staging or production URL) without the flag, they sit in the
switcher's "Not published yet" note.

## Publishing SST / the outlook (owner decision, not done here)

1. Review the staging derived products (`derived/index.json`, run 42597191999): the outlook's skill vs
   climatology-AR1, the MUR freshness policy (index has none; app falls back to 3 / 7 days), the estimate's
   validation.
2. Merge the derived-layer pipeline (data PR #74, branch `feat/coastal-derived-layers`) so the scheduled
   production run writes `derived/` (today only manual staging runs do).
3. Run a `staging` publish from the merged pipeline, verify the app against it with `CW_DERIVED_LAYERS=1`, then
   let the scheduled production run publish; set `CW_DERIVED_LAYERS=1` on the deployment only after that.

Nothing in v15.1 publishes data or changes the pipeline.

## Findings worth acting on

- C-HARM: the newest run in both channels was issued 2026-11-03; on 2026-11-13 it is ten days old and every lead is
  past. Explore labels it "past run" and draws it lighter; it is never presented as current. NOAA has not posted a
  newer run that the pipeline could read (`forecast_runs[0].leads_missing` is empty for that run).
- Staging lags production by three days; a staging preview shows older frames than the public data.
- Monterey Wharf's last pDA is 2022-08-31 although temperature is sampled weekly: a variable not collected, not a
  pipeline gap (owned by the Research coverage audit).
- HF radar has no cells north of Cape Mendocino in either channel (radar coverage, not a pipeline fault).
