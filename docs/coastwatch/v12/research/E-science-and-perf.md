# Neritic v12, research track E: scientific integrity and performance constraints

Read-only audit, 2026-11-10. Companion: `E-real-data-snapshot.json` (real Monterey / Moss Landing values, each with `source` and `asOf`). Paths are relative to the repo; web paths are under `coastwatch-web/src/`.

## 0. Findings that change the design brief

1. **C-HARM is 7 days old today.** Newest run is issued 2026-11-03 (derived date), valid Nov 2-5. Every lead is in the past. The real state on the day of the mock is "past run", and a design with a "next 3 days" hero shows nothing true. Design the past-run state first.
2. **The 8-day outlook is pure climatology at Moss Landing this run.** All 20 cells within 15 km equal the climatology (ratio 1.000), because the MODIS composite had no clear value. Domain-wide 69.6 % of cells are climatology only (composite values in 485 of 1590 cells). The app prints "near usual" (`src/pilot/facts.ts`, `outlookRead`), which is true by construction and says nothing. The value is about 10 mg/m3 (a 4 km MODIS near-shore climatology) next to an observed OLCI 0.69 mg/m3 (300 m). Never set them side by side. Detect `value/typical == 1` per cell and say "no recent view; this is the usual for this time of year".
3. **C-HARM at the shore has no demonstrated skill.** Against CalHABMAP piers (same-day v3.1 nowcast, nearest cell): pDA > 500 ng/L AUC 0.30 (CI 0.23-0.39), BSS -11.5, n=1058, 46 events; mean model probability 0.68 against a 4.4 % base rate. At Moss Landing the pDA median sat at 0.70-0.81 on all 17 days with a run in the last 30. **A 75 % C-HARM reading here is near-constant background and must never read as an alarm.**
4. **Three of the pilot's sections exist only on staging:** CDPH crab tests, fishery status + RAMP, derived layers (SST, chlorophyll estimate, outlook). Production has none of them and must stay that way. Loaders already return `null` for a missing sidecar, so each section needs a designed unavailable state.
5. **Two active contradictions at the Monterey ports.** SN26-018 (Monterey bivalves) is transcribed active but flagged disputed by SN26-021; the mussel quarantine is `active` in production and "ended Oct 31" on staging. "Sources differ · not verified" must be a first-class state of the status row.

## 1. Data inventory

Dataset: `v1/` on `coastwatch-data` (production, commit 7a0d173e3, generated 2026-11-10T07:21Z; served by GitHub Pages) and `coastwatch-data-staging` (ee87eebef, generated 2026-11-09T09:57Z; never on Pages; read via raw.githubusercontent). Pipeline cron every 6 h at :17 (`.github/workflows/coastwatch-data.yml`). The app reads `CW_DATA_BASE_URL`. Grids decode as `raw*scale_factor+add_offset`, `10**` if `transform: log10`, `65535` = no data; large ones are chunked (`{row}_{col}.u16.gz`, 512x512).

Channels: **P+S** both; **S** staging only; **Live** fetched by the server at request (not in the dataset).

| Product | Source | Space | Cadence, latency | Coverage gaps | Kind | Where | Ch |
|---|---|---|---|---|---|---|---|
| C-HARM v3.1: P(Pseudo-nitzschia > 10,000 cells/L), P(pDA > 500 ng/L), P(cDA > 10 pg/cell); leads 0-3 | NOAA CoastWatch ERDDAP `wvcharmV3_0day..3day` | 0.03 deg, about 3 km, 351x391 | Daily, often skipped; issue date derived (nowcast + 1). Freshness: current 1 d, stale 7 d | Coastal mask; harbours in the gap (Moss Landing nearest cell 1.98 km); valid fraction 0.56 | Model (official forecast) | `manifest.layers[charm_*_lead{0-3}]`; `charm/<date>/lead<n>-<h>/<var>.{u16.gz,png}` | P+S |
| OLCI chlorophyll, latest clear view + 6-7 daily layers | Sentinel-3A/B OLCI via ERDDAP | 0.0025 deg, 300 m, 3400x3880, log10 | Daily; 1.5-2 d; current 3 d, stale 7 d | Cloud gaps; per-pixel age grid (`composite.age_grid`); pixels from different days; unreliable near shore | Observed | `olci300_chl_latest`, `olci300_chl_<date>`; `satellite/olci300/{latest,day}-*/grid,tiles` | P+S |
| VIIRS chlorophyll, latest | S-NPP (NOAA-20/21 merge when newer) | 0.0075 deg, 750 m | Daily; 2 d (Nov 10 run: newest Nov 7) | Clouds; platform changes by run, name it | Observed | `viirs750_chl_latest` | P+S |
| Multi-sensor chlorophyll | Neritic pick of OLCI/VIIRS per pixel | Tiles only, no grid | With its members | Display product, no values to read | Observed composite | `multisensor_chl_latest` tiles | P+S |
| GIBS VIIRS NOAA-20, PACE OCI chlorophyll | NASA GIBS WMTS | Level 7 tiles, NASA legend | Daily; 1-3 d | NASA-rendered colours, no numbers | Observed imagery | `gibs_*` (external URL) | P+S |
| HF-radar surface currents, 24 hourly fields + 24 h mean | HFRNet totals, ERDDAP `ucsdHfrW2` | 2 km (0.018 x 0.021 deg) | Hourly; 4-15 h (this run newest 05Z, run 07:21Z); hourly fields rolled every run | Monterey Bay 0.79 observed; North Coast 0.00; statewide 0.12; cells need 2+ radars, HDOP under 1.25, under 2 m/s | Observed | `hfr2km_currents_<hour>`, `hfr2km_currents_mean24h`; `vectors.{u_grid,v_grid}` | P+S |
| Chlorophyll estimate (gap fill) | Neritic OI on OLCI + per-pixel background | 300 m, with `uncertainty_grid` and `class_grid` | Per pipeline run | Gaps with too little data stay empty | Estimated, validated | `derived/index.json#chl_estimate_latest` | S |
| 8-day chlorophyll outlook | Neritic: climatology + AR(1) of newest MODIS-Aqua 8-day | 4 km, 240x240, open coast 25-30 km only; bays empty | One 8-day mean; issued Nov 9, window Nov 5-12 (starts before issue) | 70 % of cells climatology only this run | Statistical outlook | `derived/index.json#chl_outlook_8d` + `typical_grid`, `uncertainty_grid` | S |
| SST | JPL MUR v4.1 via ERDDAP `jplMURSST41` | 1 km, 851x971 | Daily; about 2 d | None by construction: it is a blended analysis | Analysis, not observation | `derived/index.json#sst_latest` | S |
| Shore-sample water quality (pDA, tDA, Pseudo-nitzschia, chlorophyll, temp) | CalHABMAP via SCCOOS ERDDAP, 17 stations | Point | Typically weekly, days late | Santa Cruz and Monterey wharfs nearest; blank = not measured, 0 = "reported 0" | Observed sample | `observations-<h>.json` | P+S |
| CDPH crab/anchovy domoic results | CDPH CrabDAResults.pdf, parsed (sha256 kept) | Named site/BMA | Irregular; as_of Nov 2 | Moss Landing: Monterey Bay Dungeness and rock crab, Sep 28 only | Observed lab sample | `biotoxin/index.json` | S |
| Fishery status + RAMP zones | Curated from CDFW/CDPH; every record `pending_human_review` | Fishery + zone lat bounds | checked Nov 8 | Zones 1-5 listed; no polygons (ds3120 terms unstated, latitude only) | Official, unverified | `fisheries-status/index.json` (`records`, `ramp`) | S |
| Official notices | CDFW/CDPH transcribed; `registry.review.status=pending_human_review` | Notice area + `geometry` | Watcher each run (`official-watch.json`) | 11 records; SN26-024 seen upstream unreviewed; `conflicts[]` | Official, unverified | `official-<h>.json` | P+S |
| Port neighbourhood stats, C-HARM 30-day medians, VIIRS 8-day median | Neritic from above | 15 km around 22 ports | Per run | Medians, not at the dock | Derived summary | `port-intel-<h>.json` | P+S |
| CDFW landings, statewide 2015-2024, CPI-adjusted | CDFW + BLS | Statewide | `foss_landings` last success Oct 27 (3 consecutive failures) | 2025 unavailable; values statewide only | Reference | `fisheries-<h>.json` | P+S |
| NWS coastal waters forecast, alerts | `tgftp.nws.noaa.gov/.../pzz535.txt` (api.weather.gov only with `CW_NWS_API=1`, robots) | Marine zone | Several per day | Near known places only | Official forecast text | `/api/conditions` | Live |
| Buoys | NDBC realtime2 (46236 waves, 15 km; 46092 wind, 22 km) | Point | 30-60 min | Wave buoy has no wind; station may go silent | Observed | `/api/conditions` | Live |
| Tide high/low | CO-OPS predictions, 9413617 (0.3 km) | Point | Prediction | Astronomical only | Prediction | `/api/conditions` | Live |
| Water temp | CO-OPS 9413450 Monterey (23.6 km); SCCOOS shore stations | Point | 6 min | Distance to port is material | Observed | `/api/conditions` | Live |
| Relief, isobaths | NOAA ETOPO 2022 15" statewide; CRM v6/7 3" Monterey inset | About 450 m; about 90 m | Static | Inset only -122.72..-121.58, 36.22..37.33 | Reference | `public/basemap/*`, `relief.json` | Bundled |

**Staging is not simply ahead.** On 2026-11-10 production's chlorophyll (Nov 8) is newer than staging's (Nov 7). A preview built on staging shows staging's dates.

**Snapshot highlights (Moss Landing, port-592; full detail and arrays in the JSON).**

| Value | Reading | asOf | Source | Ch |
|---|---|---|---|---|
| SST, MUR analysis, 15 km median | 19.21 C (18.78-19.53) | Nov 7 | derived `sst_latest` | S |
| Water temp, in situ | 19.2 C Monterey; 19.1 C 46236 | Nov 10 15:42 / 15:00Z | CO-OPS / NDBC | Live |
| Chlorophyll, OLCI, 15 km median | 0.69 mg/m3 (p10 0.42, p90 2.46; n 4626) | Nov 8 | `olci300_chl_latest` | P+S |
| Chlorophyll, VIIRS 8-day median | 0.62 mg/m3 | centred Nov 4 | `port-intel` | P+S |
| C-HARM P(pDA) medians, 15 km | 75 %, 74 %, 71 %, 72 % (valid Nov 2/3/4/5) | issued Nov 3, 7 d old | `port-intel` | P+S |
| C-HARM P(P-n), same days | 67 %, 70 %, 62 %, 66 % | issued Nov 3 | `port-intel` | P+S |
| Crab DA, Monterey Bay, viscera | Dungeness avg 5.5, max 9.8 ppm; rock crab avg 3.8, max 20 ppm (action 30) | sampled Sep 28 | `biotoxin` | S |
| Commercial Dungeness | Delayed to at least Jan 1, 2027 (whales), Zones 1-5, not verified | checked Nov 8 | `fisheries-status` | S |
| Recreational Dungeness | Open since Nov 7, traps prohibited, hoop nets/snares | checked Nov 8 | `fisheries-status` | S |
| Monterey anchovy | No take for human consumption since Sep 11, bait only | checked Nov 8 | `fisheries-status` | S |
| RAMP Zone 4 | Pigeon Point to Lopez Point; next assessment Dec 15-18 | assessed Oct 30 | `fisheries-status.ramp` | S |
| Monterey Bay marine forecast | Today N 5 to 10 kt, backing NW; seas 4 to 6 ft | Nov 10 02:30 PST | NWS PZZ535 | Live |
| Next tides | High 5.71 ft 09:53; low -0.39 ft 17:27 (MLLW, PST) | Nov 10 | CO-OPS 9413617 | Live |
| HF radar, bay box | 374-383 valid cells every hour of 24; newest 05Z Nov 10, 376 vectors | Nov 10 05Z | `hfr2km_currents_*` | P+S |

## 2. Truth rules

Apply to any graphic, any surface, any size. Each rule has its check. Existing guard tests: `tests/unit/safety.test.ts`, `scienceWording.test.ts`, `flow.test.ts`, `charmDays.test.ts`, `todayFacts.test.ts`.

**Kind and time**
- T1 Every reading shows kind, source and an absolute date, in the same visual unit as the number. Kind words only: `measured`, `analysis`, `estimated`, `model`, `outlook`, `prediction`, `official · not verified`. *Check: cover the value's neighbours; can you still name its kind and date?*
- T2 Observed and forecast never share a mark. Forecast/model: hatched or outlined mark, or a distinct glyph; observed: solid. A shared axis has a visible "now" divider, and a past lead stays on the past side. *Check: greyscale print still separates them.*
- T3 Future tense only for a valid date on or after today (Pacific). A run whose leads are all past reads "past run (Nov 3)", steps behind newer evidence, and uses no forward-looking layout. Live example: C-HARM today.
- T4 One hero, one kind. A composite statement mixing official + model + observed needs each part separately labelled.

**Chlorophyll and toxin**
- T5 Chlorophyll is "algae, not toxin": the words appear with every chlorophyll figure, legend or outlook, including phone peeks. Never "bloom risk", "toxic", "danger", or a warning hue on the chlorophyll ramp.
- T6 Toxin appears only as a named lab result: species, tissue, site, date, ppm, action level. Water-sample pDA (ng/mL) is never drawn on a seafood scale (ppm) or compared with one.
- T7 `<2.5` and `ND` are printed as such (`values_raw`) and drawn as a distinct censored mark, never as 0 or as 2.5.
- T8 An average over six crab never hides the maximum or the share over the action level. Print max next to avg. An older test is older: show age.

**C-HARM**
- T9 Name the event and threshold: "P(pDA > 500 ng/L)". Never "toxicity", "toxin forecast", "risk of poisoning".
- T10 Mandatory words near it: "probability of conditions in the water, not toxin in seafood".
- T11 It may not imply: location of a bloom at the shore, seafood state, closure likelihood, or good/bad fishing. Harbours are inside its coastal gap; a port number is "median of cells within 15 km", never "at the harbour".
- T12 No alert styling from probability alone. Colour classes are display intervals, not risk tiers (`src/lib/palette.ts`). The `CHARM_WATCH` line is a display threshold; do not name it. Validation shows near-constant high nearshore probabilities (finding 0.3).
- T13 Do not show a min-max across cells as a confidence interval. It is spatial spread; label it so or omit it.

**Uncertainty**
- T14 Every model or estimated number shows its typical error in plain words: outlook "typical error about a factor of 2"; estimate from `uncertainty_grid`; "read as higher or lower than usual, not as a value".
- T15 Precision follows error. Outlook: one significant figure ("about 10"). C-HARM: whole percent. Chlorophyll observed: two significant figures. *Check: no 3-decimal mg/m3 anywhere.*
- T16 Point forecasts without uncertainty are not shown (safety doc R16).

**Stale and gaps**
- T17 Freshness is computed from `freshness.current_max_age_days` / `stale_max_age_days` per layer (C-HARM 1/7, OLCI 3/7, HFR 1/3, GIBS 4/10), never hardcoded. "(old)" sits beside the value itself. Beyond the stale limit the value leaves the headline.
- T18 Latest-clear-view maps state the range of observation dates and offer per-pixel age on inspect. A single "latest" date is false for them.
- T19 A gap is drawn: a distinct no-data pattern, labelled "no view", never a low colour, never blank water that could be read as calm. Summaries state valid fraction. No inpainting except the validated estimate layer, which is always labelled `estimated` and visually separate from observed pixels.
- T20 Zero is different from missing, everywhere: "No value is not a low value".

**Colour**
- T21 Sequential, perceptually uniform, lightness monotone, CVD-checked. No rainbow, no red-yellow-green. Chlorophyll: `cw-chlorophyll-log-v1`, log10, 0.05 to 50 mg/m3, labelled ticks at 0.1, 1, 10. SST: sequential, or diverging only around a stated climatology. C-HARM: `cw-probability-classes-v1`, stepped.
- T22 One colour system per quantity: chlorophyll, probability and temperature never share a ramp. Green may not mean go: status uses red for official closure, amber for a rising signal, neutral blue-grey for "no restriction on record".
- T23 Legend drawn at the opacity and over the ground the data uses (C-HARM drawn at 0.72 over relief).

**Animation**
- T24 Currents only where measured. Particles follow one observed hour, frozen, nearest-cell, die entering a gap, never cross a gap (`src/lib/flowSim.ts`). Screen speed is relative, so print "relative speed".
- T25 No decorative flow: no ambient particles on landing or empty water, no flow in cells or hours without observations, no streaks on a forecast or chlorophyll layer.
- T26 Time steps are discrete frames, each stamped. Any tweening between frames is off by default, labelled "interpolated between frames", and never between products or sensors. A new hour is a cut.
- T27 Motion never implies transport or prediction: no advecting chlorophyll by currents, no "drift of the bloom", no animating C-HARM leads as if a field moved.
- T28 `prefers-reduced-motion`: static arrows or a static frame, same information.

**3D, relief, bathymetry**
- T29 Vertical exaggeration above 1x is stated on screen ("depth exaggerated 4x"). Current hillshade is 4x sea, 1.6x land. Extruding a data field into height is banned; colour carries the value.
- T30 Bathymetry is ETOPO 15" (about 450 m), or CRM 3" (about 90 m) only inside the Monterey inset. Isobaths are Gaussian-smoothed and simplified: 200, 500, 1000, 2000, 3000 m. Nothing finer may be implied; no canyon detail outside the inset.
- T31 Relief is "not for navigation" (NOAA terms). No soundings, no depth at a tapped point unless sourced to the resolution above. Terrain may never suggest a fishing ground.

**Skill claims.** Quote only these, exactly, with period:
- T32 Outlook: RMSE ln 0.76 vs climatology 0.86 and persistence 0.89, held-out MODIS 2019-2021, n=185,364. Typical error about a factor of 2.
- T33 Estimate: RMSE log10 by band, 0-0.6 km 0.61, 0.6-3 km 0.51, over 3 km 0.42, bays 0.25 (vs older view 0.61/0.60/0.50/0.32), 2026-08-19..10-16.
- T34 C-HARM: "Published assessment covers v1 (Anderson 2016); none found for v3.1. Producer notes many false positives." Nearshore: "no demonstrated skill at the shore" with AUC/BSS only on the Research surface.
- T35 PINN: only `copy.ts`: "matches, but does not beat, a standard ConvLSTM baseline". No PINN or ConvLSTM number in product, including 0.80 test RMSE or the 8.3 % validation figure (a different pipeline run). Experimental model output stays off current-condition surfaces (R12-R16). No "AI-powered", "accurate", "% accurate", or any comparative claim against NOAA.

**Forbidden words and verdicts**
- T36 Never, in any state, tone or alt text: safe, all clear, clear, go, no-go, good day, good to fish, best spot, hotspot, recommended area, expected catch, price forecast, green light, low risk (as verdict), open (as Neritic's own status). "Open" appears only inside a quoted, attributed agency sentence. Green badges, check marks and "OK" pills next to a fishery are the same claim in shape.
- T37 Low means low, not safe: "lower probability", "no record", with "does not mean open or safe" within one glance.

**Official notices**
- T38 Registry state is `pending_human_review`. Never display verified, confirmed, cleared, lifted or expired unless a person ran `cwp review-official`. The minimum text is exactly **"not verified"**, as words: never icon-only.
- T39 One marker may cover a group only if it stays with the group, is computed ("5 of 7 not verified"), each quote outside the group (lede, alert, Ask) carries its own, and the detail sheet states the state. "No record" rows need the marker too.
- T40 "Sources differ" is a status, never resolved by Neritic: SN26-018/021; mussel quarantine. Quote agency wording, link the source, show `checked` date.

**Regulatory correctness**
- T41 RAMP is CDFW's whale-entanglement programme for commercial Dungeness: its tools are delays, depth limits, gear reductions (15-50 %), closures, Fleet Advisories. The 2026-27 delay is a whale decision, not domoic acid. Never colour or word it as a toxin event.
- T42 CDPH issues health advisories and quarantines and runs tests; CDFW acts on fisheries. Attribute each. Never combine "CDFW and CDPH say". Commercial and recreational are separate rows; a tapped port must not get the recreational list for a commercial reader (IA risk 4).
- T43 Zones are latitude bands (`ramp.zones[].lat_north/lat_south`), drawn as lines, not polygons. Action levels: 30 ppm crab viscera, 20 ppm anchovy/sardine. "Two consecutive clean sets" is "usually", and CDFW/CDPH decide. No whale sightings data exists for Monterey; do not imply it.
- T44 A clean test never softens a closure of the same fishery. Hierarchy: official status, then measured tests and observations, then models.

**Economics**
- T45 Computable from user inputs only: gross, fees, fuel, trip cost, crew pay, net to boat, per crew, per person, break-even catch/price/fuel, fixed ±20 %/±30 % grids. Every output is labelled "hypothetical".
- T46 Never computable or shown: expected catch, expected price, profit, best day or place, opportunity score, any link from chlorophyll, C-HARM, SST or sea state to dollars.
- T47 Defaults: nothing numeric is pre-filled. A sourced reference (NOAA 2008 fleet survey, CDFW 2024 statewide average, EIA on-road diesel) sits beside its field and enters only on a tap, then stays labelled "your entry, from <source, year>". Species without a source stay blank.

## 3. Trip ledger audit

Code: `src/lib/tripEconomics.ts` (`computeTrip`, arithmetic), `src/lib/tripEvaluation.ts` (`evaluateTrip`, engine of record: reasons, gaps, formulas; calls `computeTrip` for net and break-evens), `src/lib/tripEconomicsStore.ts` (device store), `src/content/fisheryPresets.ts`, `src/content/referenceValues.ts`, UI in `src/v4/coast/economics/`. Tests: `tripEconomics` 31, `tripEvaluation` 35, `tripEconomicsStore` 15, `economics.spec` 14.

**Inputs.** Round trip nm, cruise kt, steam gph, fishing hours and gph, or gallons directly; fuel $/gal, shared flag; cost lines (shared or boat-paid); pots and bait per pot (become a shared cost); fees % and $/lb; catch lines (species, lb, $/lb); crew model.

**Formulas (hand-checked in an audit script against all three crew models, loss, two-line, zero-price cases; break-evens plugged back to net 0).**
- Fuel gal = nm/kt x steam gph + hours x fishing gph, or as entered. Fuel $ = gal x price.
- R = sum lb x $/lb; fees = R x tau + u x Q; D = shared costs; B = boat-paid.
- After expenses: N = R(1-tau) - uQ - D; crew = s x max(N,0); net = N - crew - B.
- Of gross: crew = sR; net = R(1-s-tau) - uQ - D - B. Day rate: crew = people x $/day x days; net = R(1-tau) - uQ - D - B - crew.
- Break-even root of the unclamped linear net, correct because the root is always at N >= 0 for B >= 0.

**Arithmetic is correct. Issues found.**

| # | Severity | Issue | Where | Fix for the rebuild |
|---|---|---|---|---|
| E1 | Medium | A blank leg is quietly dropped: steaming entered, fishing hours blank gives 23.4 gal instead of 71.4 and an overstated net (+$180 in the test case); only an assumption note says so | `tripEvaluation.ts` fuel block | Show a "partial" state: fuel known for one leg, net not worked out until the other is 0 or filled |
| E2 | Medium | `fuelShared` defaults to true, against "nothing is assumed"; moves net by about $104 in the test case; absent from `assumptions` | `tripEconomics.ts`, `tripEvaluation.ts` `clean()` | Make it a required choice or add it to the assumptions |
| E3 | Low | Waterfall: when costs exceed gross and net is still waiting, `lo` stays 0 and bars run below the baseline into the labels | `Results.tsx`, `y = (v) => ...`, `lo = Math.min(0, net ?? 0)` | Use the running minimum for `lo` |
| E4 | Low | Sensitivity grid holds fuel, bait and day-rate fixed as catch moves ±30 %, unstated | `computeTrip`, `Results.tsx:304` | Say so in the grid heading |
| E5 | Low | Worked sums always read in pounds (`Gross = 1,000 lb x $0.54`) while the worksheet may be in tons or kg; the per-lb fee field stays `$/lb` for squid | `evaluateTrip` formulas, `Worksheet.tsx:386,491` | Render formulas in the worksheet's unit |
| E6 | Low | Break-even fuel can exceed the $50/gal input limit (53.88 in the test case) | `LIMITS.usdPerGal` | Print "above $50/gal" |
| E7 | Note | Two engines. `computeTrip` is imported only by `tripEvaluation` and its tests | | Keep `evaluateTrip` as the only public API |

**Missing assumptions the UI should state** (all outside the arithmetic): "net to boat" is before insurance, permits, maintenance, gear loss, loan payments, depreciation, and captain's pay where the owner skippers; it is not profit or take-home. Landing taxes and buyer deductions only if entered. One steaming burn for both legs. No weather, soak time, pot loss or multi-day logic. Reference caveats: NWFSC-121 is a 2008 fleet survey; crew shares are "owner as captain" variants; CDFW prices are statewide 2024 averages and Dungeness "mixes two seasons"; diesel `DIESEL_REF` is week of Oct 19 on-road retail including road taxes ($8.242), 22 days old on Nov 10 and past its 14-day stale limit.

## 4. Performance baseline

**Method.** Current build on `localhost:3410`, fixture data, installed Chrome headless, cold load (cache disabled, service worker blocked), 9 s settle (22 s throttled). Metrics from `PerformanceObserver`; TBT = sum of long-task time over 50 ms after FCP; JS = CDP `encodedDataLength` (over the wire); map-ready = first frame `window.__cwMap.loaded()`. Throttled = CPU 4x + DevTools Slow 4G (562.5 ms, 1.47 Mbps down). Medians of 3 runs desktop, 2 throttled. Scripts and raw rows: `scratchpad/audit-E/perf*.mjs`, `perf-results*.json`.

**Caveats.** The machine carried other sessions (load average 11-22), so treat TBT and map-ready as upper bounds and re-measure before/after on the same box. Fixture data is October-dated and its payload is not production's. This server has the pilot flag: `/` is a 307 to `/today`, so I measured Explore separately at `/?view=map`. Bare `/today` is the port picker.

| Route | Desktop LCP / TBT / map | Desktop 4x + Slow 4G: LCP / TBT / map | Phone 390, 4x + Slow 4G: LCP / TBT / map | JS KB (files) | Fonts / data KB |
|---|---|---|---|---|---|
| `/` (307 to picker) | 3904\* / 0 / 928 | 2766 / 144 / 14879 | 4404 / 102 / 14281 | 571 (18) | 333 / 522 |
| `/today` picker | 3860\* / 0 / 961 | 2164 / 108 / 14870 | 3834 / 110 / 14302 | 571 (18) | 333 / 522 |
| `/today?place=port-592` | 152 / 0 / 1242 | 3822 / 350 / 19435 | 3836 / 140 / 16288 | 571 (18) | 333 / 867-992 |
| `/?view=map` Explore | 196 / 0 / 936 | 5758 / 212 / 17169 | 9242 / 265 / 14511 | 669 (19) | 333 / 561-640 |
| `/coast` landing | 144 / 0 / none | 3778 / 103 / none | 3782 / 86 / none | 358 (19) | 333 / 20 |
| `/coast?place=port-592` | 252 / 0 / 1139 | 3792 / 371 / 18756 | 3782 / 388 / 17394 | 683 (23) | 333 / 971-1051 |
| `/research` | 172 / 0 / none | 2710 / 163 / none | 2694 / 124 / none | 619 (18); phone 293 (14) | 333 / 518; phone 13 |

Times in ms. \*LCP here is a late caption of the opening ("Streaks: measured currents"), one run 6080; the opening's text is setting LCP on an idle desktop.

**Read.**
- The longest task under throttle was 220 ms; the main thread is not the problem. On a phone connection the map arrives in **14-19 s**, mostly transfer: 571-683 KB JS, 333 KB fonts, 0.5-1 MB data.
- Text LCP sits near 3.8 s throttled, set by the network and fonts. Explore on a phone is 9.2 s because its largest text waits for the map state.
- **Bug: `/research` at 390 px has CLS 0.36** (every other route is at or under 0.04).
- Fonts are 333 KB on every route, more than the whole `/coast` data and about the size of its JS.

**Budgets the redesign must not regress** (measured the same way; numbers are ceilings, over-the-wire):

| Budget | Ceiling |
|---|---|
| Route JS | `/today` 571, Explore 669, `/coast` 358, briefing 683, `/research` 619 KB (phone 293). Any new library is paid for by removals in the same route |
| Three.js, deck.gl, or any second GL stack | Not in first-load JS on any route. Lazy, opt-in, and counted against the route once loaded; one WebGL context per page (`motion.spec`, `currents.spec` assert one canvas) |
| Fonts | 333 KB maximum, target 200 KB; at most the current 5 preloads |
| Text LCP, phone 4x + Slow 4G | 3.9 s on Today, `/coast`, briefing; 2.8 s on `/research`; Explore must improve to 4.5 s (statement text may not wait for the map) |
| LCP, unthrottled desktop | 300 ms on content routes. The opening must not be the LCP element, or must be under the current 3.9 s and skipped on repeat visits |
| Map-ready, throttled | Today/briefing 19 s, Explore 17 s, picker 15 s; first-paint content never depends on it |
| TBT, throttled | 400 ms; longest task 250 ms |
| CLS | 0.1 on every route and width (fixes `/research` phone) |
| Animation | No canvas or particle loop before LCP; 60 fps at idle zoom on desktop, 30 fps floor at 4x; paused off screen and under reduced motion; no added main-thread long task during pan |
| Graphic data | Inline downsampled data under 30 KB per first-screen graphic; native grids only on demand, through the chunked grid reader |

**Offline system (must keep working).**

| Piece | What it does |
|---|---|
| `public/sw.js` (`neritic-sw-4`, 349 lines), `sw-push.js` | `_next/static` cache-first in `neritic-static-<build>`; old builds' code dropped; packs never deleted by the worker. Pages, data and API network-first, pack as fallback; with a pack and slow network the answer waits at most 5 s. PMTiles from the pack as 206 byte ranges. Unsaved page offline: saved Explore or briefing for those two paths, else the offline page; never another page's content. Pages from a pack carry `self.__NERITIC_SAVED__` with the saved time |
| Field packs, `src/v4/offline/packs.ts` | `PACK_VERSION` 2. One pack: Monterey, port 550, bbox -122.124..-121.676, 36.44..36.8; `neritic-pack-monterey`. Six groups: app pages and code, basemap (PMTiles 6.7 MB bundled, checksum verified, OSM/OFL licences saved), C-HARM 4 leads x 3 variables, latest satellite with pixel age, every published HF hour and the mean, briefing and evidence snapshot. `record.json` states what, when, from what data, complete or not |
| Today save, `src/pilot/today/offline.ts` | `neritic-pack-today`: port page, CSS, fonts and files referenced by the HTML, the `/api/conditions` answer |
| Kept briefing, `src/v4/offline/kept.ts` | `neritic-kept-v1`, `/coast/kept`, device storage |

Constraints for the rebuild: pack keys (`/__pack__/page/explore`) and route shapes stay, or `plan()` and the worker change together. **Today's save takes only what `staticRefs(html)` finds in the HTML and CSS**, so a lazily imported chunk, worker, texture or inline-fetched graphic data is not saved unless added explicitly. A pack only ever shows its own dates ("nothing is invented"); new graphics read saved time and stale state the same way. Service-worker changes bump `VERSION` and keep packs.

Guards: `tests/e2e/offline.spec.ts` (1, plan, download, verify, remove), `offline-recovery.spec.ts` (7, quota, older format, damaged record, new worker with renamed code), `mycoast-kept-offline.spec.ts` (1), `pilot-journey.spec.ts` (`save-offline`), `tests/unit/offline-pack.test.ts` (4), `keptOffline.test.ts` (2), and **`scripts/offline-proof.mjs` (`npm run test:offline`)**: server stopped, browser restarted, every request sent to a dead proxy, every layer, sensor and day walked at several zooms inside the pack's box, zero same-origin misses. Playwright offline emulation cannot stop a worker's own requests, so only this script proves it. Not run in this audit (build left untouched); run it on every rebuild milestone.

## 5. Tests and coupling

Totals: Playwright **51 specs, 386 tests**; Vitest **79 files, 828 tests**; pipeline pytest **297 functions, 23 files**; no screenshot baselines (0 `toHaveScreenshot`).

Coupling: e2e uses 2,129 `getByTestId` calls over 876 unique ids (251 of them built dynamically, so missed by a static grep); 995 text assertions (`toContainText`/`toHaveText`) and 1,053 text/role locators overall; 119 `boundingBox`/`data-area` geometry checks. Source has about 1,186 static test ids. Unit tests are DOM-free (no `render()`); only source-scanning guards depend on file layout (`safety`, `motion`, `design`, `kit`, `v6x`).

| Plan | Specs (tests) | Why |
|---|---|---|
| **Preserve, behaviour-level** | `hydration` (1), `offline` (1), `offline-recovery` (7), `mycoast-kept-offline` (1), `gridtiles` (2), `coastline` (2), `coastline-fallback` (3), `parity` (13), `states` (5), `readiness` (5), `sync` (4), `a11y` (5), `regulatory` (3), `pilot-account` (11), `pilot-account-live` (2), `ask` (5), `pilot-front-door` (6), `splash` (10), `motion` (6) = **19 specs, 92 tests** | Assert data, offline, state or a11y truth. Port selectors only. |
| **Port, truth encoded in DOM** | `today` (17), `pilot-journey` (1), `today-picker` (5), `trust` (18), `briefing-trust` (11), `briefing` (8), `discrepancy` (4), `evidence` (3), `evidence-v3` (6), `since` (10), `economics` (14), `fishing` (9), `research` (22), `currents` (10), `explore` (10), `explore-v11` (14), `navigation` (9), `m7` (15) = **18 specs, 186 tests** | Keep the claims, especially wording and honesty assertions; re-express against new structure. Encode new rules from §2. |
| **Retire or rewrite with the new layout** | `explore-rework` (12), `explore-shell` (7), `explore-v10` (5), `mycoast-v10b` (7), `mycoast-pass` (4), `v5-map` (12), `v8` (7), `ux-final` (6), `layout` (7), `ergonomics` (6), `hardening` (7), `m3` (14), `m6` (9), `brand` (5) = **14 specs, 108 tests** | Versioned layout, geometry and copy-position checks. Mine for any honesty assertion before deleting (notably `m3`, `m6`, `hardening`). |

**Keep unchanged (engine of truth):** all unit suites for `facts`, `freshness`, `charmDays`, `chlnear`, `flow`, `official`, `tripEvaluation`, `safety`, `scienceWording`, and pipeline pytest. Banned-word regexes exist in two places (`safety.test.ts`, `economics.spec.ts`); consolidate.
