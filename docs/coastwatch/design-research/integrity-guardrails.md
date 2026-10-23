# v10 integrity guardrails: science, attribution, accessibility, performance

Status: baseline for the v10 redesign, written 2026-10-22 against worktree commit `6d3f32f3e` (v9) with
staging data (manifest `0bb19831d0be`, generated 2026-10-21T03:52Z; derived index of the same run).
Every concept and the final build is reviewed against §5. File references are relative to
`coastwatch-web/` unless they start with `docs/` or `pipeline/`. Measurements: `scratchpad/v10-audit/`.

The rules here restate rules the project already has (docs/coastwatch/05-science-and-safety.md R1–R26,
CLAUDE.md "Science", the v4–v9 decisions) and the guarantees the code enforces today. A redesign
may change how things look and move. It may not weaken any of these guarantees, and where it
re-opens a past decision (for example particle currents, removed in v4) it must say so.

---

## 1. Data semantics inventory

Product classes and their colour families are fixed: official = amber, model = violet (C-HARM and the
outlook), measured = teal (satellite chlorophyll, shore samples), conditions = sky (SST, currents,
NWS/NDBC/CO-OPS) (`src/v4/theme.css:10-12`, `src/v4/explore/LayerPanel.tsx:24-31`). About copy promises
"the same four colours, and they never stand for anything else" (`src/content/copy.ts:223-224`).
Precedence is official > official forecast > observation > experimental/derived > history
(`docs/coastwatch/05-science-and-safety.md` §1, R1–R4).

### 1.1 C-HARM v3.1 bloom and domoic-acid probability (forecast/nowcast, model)

- **What:** the probability that each of three quantities exceeds a threshold. Particulate domoic acid
  > 500 ng/L, *Pseudo-nitzschia* > 10,000 cells/L, cellular DA > 10 pg/cell (`src/v4/explore/layers.ts:32-36`).
  These are probabilities, not concentrations.
- **Footprint:** 0.03° cells, about 3 km (manifest `resolution_deg: 0.03`). The map shows them with
  nearest-neighbour resampling (`src/components/map/MapCanvas.tsx:182`, `raster-resampling: "nearest"`).
  There are no toxin probabilities for many cells within about 3–6 km of shore, and none in enclosed bays
  (`src/lib/readout.ts:31-36`).
- **Time:** one run gives leads 0–3 (`src/lib/layers.ts:5`). Lead 0 is labelled "Nowcast" (`layers.ts:27-29`).
  Every layer carries `valid_date` and `issued_date`. The issue date is **inferred** (valid day + 1)
  because C-HARM publishes no issue time (manifest `issued_date_derived: true`; copy `src/content/copy.ts:55`).
  Freshness basis is `issued_date`: current ≤ 1 day, stale ≤ 7, then historical (manifest `sources[charm].freshness`;
  `src/lib/freshness.ts:34-42`).
- **Required labels:** the threshold sentence, "valid <day>", the issue date marked as inferred, and
  "A forecast probability, not a measurement of toxin in seafood and not a closure decision" plus
  "A low probability does not mean an area is safe" (`src/content/copy.ts:48-57`). The display classes
  are ten 10-point intervals. They are reading aids, not risk categories (`src/lib/palette.ts:3-24`).
  Skill was published only for v1 (Anderson et al. 2016), and the producer warns of many false positives
  (manifest caveats).
- **Validation that limits use:** the pipeline checked the nearest C-HARM cell against CalHABMAP pier
  samples (n = 1,058, 46 events). For pDA > 500 ng/L the AUC was 0.30 (95% CI 0.23–0.39), worse than
  chance, with BSS −11.5. The published readout is: "Context only: the nearest C-HARM cell offshore has no
  demonstrated skill at the shore" (derived index `charm_nearshore_extension`; `src/lib/derived.ts:299-323`).
- **QC:** the pipeline checks each value against its verification points
  (`tests/e2e/trust.spec.ts:198-206`). A cell with no value tells the reader why: outside the model,
  toxin mask, or not modelled (`src/lib/readout.ts:30-58`).
- **Never:** interpolate or smooth between cells, or tween between leads or days. A day step cuts; it
  does not fade (`tests/e2e/motion.spec.ts:60`, v9 rule 1). Never extend values into the masked nearshore
  strip or bays as if modelled. Never call a low class "safe", "clear" or green-for-go (R3). Never put a
  probability above an active notice for the same place (R1). Never show the issue date as published
  upstream. Never relabel the 10 classes as low/medium/high risk. Never blend C-HARM colours with
  chlorophyll, outlook or SST ramps in one legend (R7).

### 1.2 Satellite chlorophyll-a (observation)

- **What:** near-surface chlorophyll-a in mg/m³ on a log scale (`src/lib/derived.ts:217-234`). It is algae
  biomass. It is not toxin, it does not identify *Pseudo-nitzschia*, and it does not show where fish are
  (`src/content/copy.ts:68-72`, R5).
- **Products and footprint:** Sentinel-3A/3B OLCI at 300 m (0.0025°) and VIIRS NOAA-21/NOAA-20 at 750 m
  (0.0075°). S-NPP is used only when it holds a strictly newer day, and S-NPP data end on 2026-11-02
  (`pipeline/coastwatch_pipeline/sources/satellite.py:1-25`). The "multi-sensor view" is a Neritic
  display that puts OLCI 300 m where it exists and VIIRS 750 m elsewhere. A colour change at a sensor
  boundary may come from the sensors, not the water (manifest caveat). There are also two NASA GIBS
  browse layers (NOAA-20 VIIRS, PACE OCI). These are pictures: "values are not read from these tiles"
  (manifest caveat).
- **Time:** there are single-day layers (`observed_date`) and a "latest clear view" composite, in which each
  pixel is its newest valid observation within 7 days and keeps its own date through a per-pixel age
  grid (`src/lib/layers.ts:44-55`, `src/lib/readout.ts:66-92`). A coverage figure names the span of days
  it counts (`src/v4/explore/layers.ts:184-196`). Freshness basis is `observed_date`: current ≤ 3 days,
  stale ≤ 7 (manifest). On a failed update the previously published layers stay, with their own dates
  (`src/lib/layers.ts:70-76`).
- **QC and flags:** upstream cloud, land and quality masks are kept as no-value
  (`satellite.py:21-25`). Place rules set aside pixels that are on land, against the shore, or a lone
  spike more than 10× the water around them. These pixels stay on the map but never become a place's
  headline, and the readout says why (`src/content/copy.ts:77-103`, `src/lib/readout.ts:94-102`).
  There is no fixed ceiling, because real blooms have exceeded 300 mg/m³ (`copy.ts:80-81`).
- **Required labels:** sensor, native resolution, "observed <date>" for each pixel, the biomass
  sentence, and "Gaps are clouds or missing passes, not low chlorophyll". An empty pixel reads "No
  observation: cloud, fog, land or no overpass. Not low chlorophyll" (`src/lib/readout.ts:72`).
- **Never:** fill, smooth or interpolate gaps in the observed layer. That is the estimate's job, and the
  estimate is drawn separately (§1.3). Never animate a "loop" that implies one synoptic image per day
  when pixels come from different days. Never present chlorophyll as toxin, as bloom risk or as a fish
  indicator. Never rank pixels by percentiles of the current view (R6). Never treat GIBS picture tiles
  as values.

### 1.3 Gap-filled chlorophyll estimate (derived, "Estimated, not observed")

- **What:** `chl_estimate_latest` holds values only where the OLCI latest clear view has none. It is
  built from each pixel's recent typical value plus today's departure, spread on a 3 km scale (optimal
  interpolation). Observed pixels are drawn above it and never overwritten (`src/lib/derived.ts:4-15`;
  derived index `method`). Per-pixel class grid: 1 = filled from nearby clear water, 2 = extrapolated
  toward shore and never observed (borrowed from water farther out and raised ×5.4), 3 = bay on a
  borrowed background (`src/lib/derived.ts:47-66`, published `class_legend`).
- **Footprint/time:** 0.0025° grid (3400 × 3880). `issued` 2026-10-21, `based_on_days` 2026-10-13..16.
  It is shown only under the latest chlorophyll view of its own product, and it is on by default
  (`est=0` turns it off) (`src/v4/explore/layers.ts:19-24, 121-126`).
- **Uncertainty (quote as published only):** held-out RMSE in log10 by band is 0.606 (0–0.6 km), 0.512
  (0.6–3 km), 0.42 (> 3 km) and 0.245 (bays), against an "older view" baseline of 0.609 / 0.604 / 0.503 /
  0.319, with n = 1,252,982. Its caveat: "Next to the shore and in bays the error is larger (typically a
  factor of 2.5-4)". The range shown is value ÷/× 10^error from the uncertainty grid
  (`src/lib/derived.ts:236-258`). The web never hard-codes a multiplier or an error (`derived.ts:48-54`).
- **Never:** draw it with the observed layer's treatment so that the two read as one surface. Never let
  it overwrite or blend into observed pixels. Never present class-2 shoreline values as measured. Never
  use it for a harbour headline or for a place value without the words "Estimated, not observed".
  Never ship a derived layer that lacks its validation: the parser drops it (`src/lib/derived.ts:135-175`).
- **Decision to record:** R12 says experimental layers are "off by default". The estimate is on by
  default under the latest view. A concept that makes it more prominent must name this tension. Only
  the owner can decide it.

### 1.4 8-day chlorophyll outlook (derived forecast, "Outlook, not observed")

- **What:** a forecast of the 8-day **mean** chlorophyll per 4 km cell: MODIS-Aqua climatology
  (2003–2021) plus the newest 8-day composite's anomaly, damped by AR(1). It is "not a day-by-day map and
  not toxin" (derived index `method`). It is offered only with skill against at least one plain
  baseline (`src/lib/derived.ts:158-163`).
- **Footprint/time:** 1/24° (about 4.6 km), open coast only, roughly the first 25–30 km. There is no
  outlook for bays and harbours (caveat). `valid_from`–`valid_to` is 2026-10-20..27, issued 2026-10-21.
- **Skill (same pipeline, quote exactly):** RMSE ln(mg m⁻³) is 0.7596 for the model, 0.857 for
  climatology and 0.8921 for persistence, over n = 185,364 held-out cells × composites, 2019-01..2021-06.
  The pipeline's words: "a typical error of a factor of about 2". The model **is** climatology + AR(1)
  (`src/lib/derived.ts:331-335`). Per CLAUDE.md, it must never be described as the PINN or ConvLSTM, or
  as beating them. Research numbers stay same-pipeline (the PINN matches, does not beat, the ConvLSTM;
  `src/content/copy.ts:264-265`).
- **Never:** split it into 8 daily frames or animate it day by day. It is one mean for the whole window.
  Never show it in the observation family's colours. It currently shares the chlorophyll ramp but sits
  in the model family (`LayerPanel.tsx:28`). Never place it on the same time scrubber as observations
  without a hard visual boundary at "now". Never put it in bays.

### 1.5 Sea-surface temperature (JPL MUR v4.1, analysis)

- **What:** a daily gap-free L4 **analysis** that blends infrared, microwave and in-situ data
  (`src/lib/derived.ts:271-273`; derived index `product_class: "analysis"`). It is labelled "Analysis",
  not "Observed" (`LayerPanel.tsx:29-30, 41-46`).
- **Footprint/time:** a 0.01° grid (about 1 km). Under cloud the effective detail is about 25 km
  (microwave), and near shore it leans on interpolation (caveat). The analysis day is about 1 day behind
  (`observed_dates`: 2026-10-19). The value is surface skin/foundation temperature, "not the temperature
  at depth where fish may hold".
- **Display:** a cmocean "thermal"-derived ramp, 8–22 °C, °F beside °C (`src/lib/derived.ts:275-297`).
- **Never:** call it observed or a satellite measurement. Never promise 1 km detail under cloud. Never
  combine SST and chlorophyll into a "fishing score" or a "go" signal. Never label fronts as fish
  locations.

### 1.6 HF-radar surface currents (observation)

- **What:** hourly HFRNet total vectors on a 2 km grid (0.018°), the top metre or so of water. Speed
  and direction come from u/v. The data are read from NOAA CoastWatch ERDDAP `ucsdHfrW2`, or from the
  identical NDBC THREDDS copy `hfradar_uswc_2km` (`pipeline/coastwatch_pipeline/sources/currents.py:3, 48-70`).
  There is also a 24-hour mean, published where a cell has enough valid hours (`currents.py:375`).
- **QC (per hour, published in `qc.checks`):** at least 2 radars per cell, HDOP ≤ 1.25, speed ≤ 2.0 m/s,
  and grid alignment (`currents.py:248-259`). On 2026-10-20T03Z only 4.5% of cells had a valid value
  (`n_valid` 9,906 of 220,320). Hours whose coverage collapsed stay selectable but are not the default
  (`src/lib/currents.ts:21-55`).
- **Time:** each hour is a separate snapshot (`valid_time`). Stale beyond 18 h (`src/lib/currents.ts:135-142`).
  The source policy is current ≤ 1 day, stale ≤ 3 (manifest).
- **Required labels:** "HF radar, native 2 km · observed <time PT>", or "24-hour mean to <time>". A gap
  reads "No radar observation here. No data, not calm water" (`src/lib/readout.ts:104-112`). Manifest
  caveat: "not where a bloom will be … Each hour is a separate snapshot; the map does not show what
  happened between hours, and nothing here is a forecast."
- **Current rendering:** arrows, one per observed cell, thinned by zoom (`src/lib/currents.ts:12-13, 90-108`).
  v4 removed particle flow ("No particle flow", `docs/coastwatch/v4/README.md`). A v2 particle
  component still exists but is not mounted by v4 (`src/components/map/FlowParticles.tsx:1-30`,
  `src/v4/explore/Explore.tsx:332` mode `"arrows"`). Its rules are a good template if animation returns:
  a fixed field, the nearest cell with no interpolation, particles that die in gaps, 1–2 s lives, speed
  independent of zoom, and none at all under reduced motion.
- **Never (for any animated-currents concept):** interpolate in time between hours or in space across
  cells or gaps (earth/Windy-style bilinear advection does exactly this). Never let particles cross
  land, gaps or the coverage edge. Never draw long trails that read as trajectories or drift forecasts.
  Never imply a forecast ("where the bloom will go"). Never animate under `prefers-reduced-motion`.
  Never animate without a pause control (§3). The readout must stay exact: `currentsAt` reads the cell
  that contains the point (`search = 0`), never a neighbour (`src/lib/readout.ts:109`).

### 1.7 Shore-station samples (CalHABMAP; CeNCOOS shore stations)

- **What:** laboratory values from one water sample at one pier on one day: pDA, *Pseudo-nitzschia*,
  chlorophyll and temperature. "A blank means not measured … a reported 0 means not quantified, not
  absent"; "Toxin in seawater is not toxin in seafood" (`src/content/copy.ts:126-143`). CeNCOOS stations
  show only values flagged PASS and credit the operator (`src/lib/shoreStations.ts:11-20, 73-74`).
- **Time:** the sample date. Policy: current ≤ 14 days, stale ≤ 45 (manifest `calhabmap`).
- **QC display:** a value flagged high is ringed and kept as published. A rejected value is a cross on
  the floor, never a value. Fixed whole-decade axes never rescale to the data. Lines join samples only
  when they are less than 21 days apart (`src/v4/charts/SeriesChart.tsx:8-17`).
- **Never:** compare a sample numerically with a C-HARM probability (`copy.ts:129-130`). Never
  interpolate between samples across gaps, or smooth them. Never auto-scale an axis so that a quiet year
  looks like a bloom. Never extend a pier sample to "nearby beaches or fishing grounds" (`copy.ts:133`).

### 1.8 Buoy, tide and harbour-temperature observations; NWS text

- **NDBC buoys:** each field takes its own latest valid row with its own time, and nothing older than
  6 h is shown (`src/lib/conditions.ts:63-68`).
- **CO-OPS tides:** **predictions** (astronomical), not observed water levels (`conditions.ts:9-10`).
  The day curve is a half-cosine drawing between predicted highs and lows. The mirrored edges are
  flagged `edge` (`src/lib/tideCurve.ts:3-15`). A concept must keep the turning-point times as the facts
  and the curve as an illustration.
- **CO-OPS harbour water temperature:** 6-minute readings from the one thermistor within 6 km (no
  fallback to another station). NOAA's X/N/R flags are shown in words. A reading is stale after 1 h and
  rejected after 3 h (`conditions.ts:155-200`).
- **NWS coastal waters forecast:** the zone's text from tgftp, quoted as published, with its issue time
  and headlines. The zone is assigned from known places, never guessed from an arbitrary point
  (`src/lib/nwsText.ts:3-16`). It is a forecaster's text for a whole zone. Doc drift: the comment in
  `conditions.ts:5` still names api.weather.gov, but `nwsText.ts` says that host is not used. Settle
  this before quoting it in a provenance line.
- **Never:** turn NWS text into icons or "good/bad day" verdicts that drop the forecaster's words.
  Never present tide predictions as observed. Never show a buoy value without its own time. Never use a
  far thermistor for a harbour. Conditions say nothing about toxin or fish (`conditions.ts:15-16`).

### 1.9 CDFW / CDPH official notices (transcribed copies, not verified)

- **What:** closures, advisories and quarantines quoted from agency pages, with agency links and
  hotlines (`src/content/copy.ts:7-46`). Verification is computed in the browser from the review date,
  the watcher fingerprints, disputes and failures. Anything other than `verified` must read as NOT
  verified (`src/lib/official.ts:5-55`). Staging today: `reviewed_date` 2026-10-08, so the state is
  "Not verified".
- **Applicability:** decided only from each notice's published geography (statewide, latitude limits,
  agency outlines). The tiers are geographic / unresolved / outside, and no place name is ever matched
  as text (`src/lib/applicability.ts:5-25`). The wording controls over drawn outlines, and
  latitude-defined notices state no offshore limit (`copy.ts:45`).
- **Required:** "Not verified" next to the records, "may be incomplete", "No notice listed here does not
  mean an area is open or that seafood is safe", and the agency links. All are tested on every page
  (`tests/e2e/regulatory.spec.ts:8-38`, `tests/e2e/trust.spec.ts:36-95`). A notice past its expected end
  is "unconfirmed", never "lifted" (`trust.spec.ts:66`).
- **Never:** show a verified tick, a green state or "all clear". Never collapse "Not verified" into a
  tooltip or an icon alone. Never show "no advisories" when none are listed (R2). Never let any lower
  layer headline contradict an active notice (R1). Never mark anything verified, lifted or expired in UI
  or data (CLAUDE.md: only a person running `cwp review-official`). Never extend a county notice
  offshore by drawing.

### 1.10 Price and cost references (EIA diesel, CDFW landings $/lb)

- **What:** values shown **beside** worksheet inputs, never pre-filled (`src/content/referenceValues.ts:1-11`).
  The CDFW 2024 statewide value ÷ pounds is computed by Neritic, because CDFW publishes no $/lb, and it
  is labelled "CDFW history, not today's price" / "Past average, not today's dock price". EIA's weekly
  California No. 2 diesel retail price (read 2026-10-20) is a static, dated value that "goes stale and
  says so after 14 days". A reference is copied in only on a tap, and the field's basis is then shown
  ("basis: EIA" / "basis: CDFW history") (`tests/e2e/economics.spec.ts:49-83, 186-197`).
- **Never:** pre-fill a price or a catch. Never present a reference as a current market price or a
  forecast. Never invent a species-level reference where CDFW has no species-exact line (albacore and
  salmon have none, `referenceValues.ts:5-7`). Never hide the basis once a reference is in use.

### 1.11 Fishery presets and trip economics

- **What:** editable starting values, each with its source: NOAA NWFSC-121 (2008 survey, "Fleet average
  then, not a norm"), the FR 2024-28144 buyback fee of 2.25%, and FGC §8276.5 trap tiers. "No preset ever
  holds a catch or a price" (`src/content/fisheryPresets.ts:1-56`). The arithmetic is pure: models
  A/B/C, missing inputs listed and never guessed (`src/lib/tripEconomics.ts:1-16, 53-60`). The kicker
  reads "Scenario from your inputs, not a forecast." (`src/v4/coast/economics/TripEconomics.tsx:45-46`).
  The trap-tier hint defers to the reader's CDFW permit (`Worksheet.tsx:439`).
- **Never:** introduce an assumption without a source (a "typical catch", a "typical price", a fuel
  burn without a cited table). Never mix user inputs and references without showing which is which.
  Never present net or break-even as advice to fish, or show it beside or above a closure as if the
  trip were permitted. Never imply a recreational angler can sell catch (economics is a commercial
  tool). Never compute losses from closures (R17).

### 1.12 Historical landings (NOAA FOSS, BLS CPI-U)

- **What:** statewide annual commercial landings since 2015, with the 2015–16 domoic acid events marked
  as context only (`src/content/copy.ts:145-205`). They are history. They are not losses, not profit
  and not port-level (R17–R19).
- **Never:** draw a trend line into the current or a future season, or attribute a change in landings
  to a bloom.

---

## 2. Source attribution and licensing

### 2.1 Data and basemap sources in use

| Source | Licence / terms (as recorded) | Required credit and where it shows now |
|---|---|---|
| NOAA C-HARM v3.1 (CoastWatch West Coast ERDDAP `wvcharmV3_0day`) | "may be used and redistributed for free but is not intended for legal use … No warranty" (manifest provenance; `docs/coastwatch/evidence/sources-charm.md:46`) | Product name, version and NOAA; citation Anderson et al. 2016 (manifest `citation`). Shown in the layer panel, lineage and exports (`src/v4/research/exports.ts:14-40`) |
| Sentinel-3 OLCI via NOAA CoastWatch (`noaacwS3[AB]OLCIchlaSector[CD]IDaily`) | "Contains modified Copernicus Sentinel-3 data (EUMETSAT), processed by NOAA CoastWatch/OceanWatch; free and open" (manifest provenance) | The "contains modified Copernicus Sentinel-3 data" credit must travel with any image or export of OLCI. Direct Copernicus Marine/EUMETSAT access needs an account and the credit "Generated using E.U. Copernicus Marine Service Information". **Not allowed without the owner's decision** (`docs/coastwatch/v2/01-sources.md:108`) |
| VIIRS NOAA-21/NOAA-20 and S-NPP (CoastWatch) | NOAA boilerplate (`docs/coastwatch/evidence/sources-satellite.md:71`) | NOAA CoastWatch |
| NASA GIBS (VIIRS NOAA-20, PACE OCI browse) | "open, no restrictions on reuse; attribute NASA GIBS" (manifest) | NASA GIBS |
| JPL MUR SST v4.1 (GHRSST L4, CoastWatch `jplMURSST41`) | NASA/JPL PO.DAAC open data. The recommended citation names JPL MUR MEaSUREs Project, PO.DAAC, doi:10.5067/GHGMR-4FJ04 (not recorded in the repo: **verify**) | Credited in the layer panel from `source`. **The /sources page does not mention MUR** (checked 2026-10-22). Add it |
| MODIS-Aqua 8-day (`erdMH1chla8day_R2022NRT`) for the outlook | NASA OB.DAAC / NOAA CoastWatch, open | Named in the outlook's `basis`. Keep it in the outlook's provenance |
| HFRNet HF radar (Scripps/IOOS via CoastWatch or NDBC) | "Public domain (U.S. Government and IOOS data); credit HFRNet and the operating institutions" (manifest) | HFRNet and the operators (About providers, `copy.ts:274`) |
| CalHABMAP (SCCOOS ERDDAP), CeNCOOS shore stations | SCCOOS/NOAA boilerplate. CeNCOOS: "may be used and redistributed…", credit the operator, CeNCOOS and NOAA (`src/lib/shoreStations.ts:11-20`; `src/lib/harborEvidence.ts:682`) | The operator, credited in context (`shoreCredit`). **The /sources page does not mention CeNCOOS** |
| NWS (tgftp text), NDBC, CO-OPS | U.S. Government works | Named in context. The /sources page does not mention CO-OPS or NWS by name |
| CDFW (notices, ports ds3081, MLDS, "By The Numbers"), CDPH | CDFW Conditions of Use: public domain "unless otherwise indicated", attribution needed (`docs/coastwatch/evidence/sources-economics.md:103, 368`). CDFW ports ds3081 is CC BY 4.0 (`src/content/fisheries-geo.json`) | Agency name, link and "as of" date on every quote and reference |
| NOAA Fisheries FOSS, BLS CPI-U | U.S. Government | About providers |
| EIA diesel series EMD_EPD2D_PTE_SCA_DPG | U.S. Government (EIA asks that it be cited as source) | "EIA diesel, week of …" (`Worksheet.tsx:398`) |
| NOAA NWFSC-121, Federal Register, FGC | Government publications | Shown per field (`fisheryPresets.ts:20-56`) |
| OpenFreeMap / OpenMapTiles / OSM | OSM data ODbL 1.0 ("credit wherever its map is seen"); OpenMapTiles schema/design CC BY 4.0; the pack extract is redistributed under ODbL (`public/offline/monterey/pack.json`) | A one-line credit on the map at every width (`src/v4/explore/Explore.tsx:1140-1165`) |
| NOAA NCEI ETOPO 2022 and Coastal Relief Model | U.S. Government | "Relief: NOAA NCEI ETOPO 2022 and Coastal Relief Model" (`src/lib/basemap.ts:58-59`) |
| CA Dept. of Technology / CDTFA counties | Public domain (`copy.ts:349-354`) | Credit line |
| Fonts: Atkinson Hyperlegible Next, Source Serif 4, IBM Plex Mono (next/font, self-hosted); Noto Sans map glyphs | SIL OFL 1.1. The glyph copy carries OFL.txt into the field pack (`src/v4/offline/packs.ts:75-77`) | Ship OFL.txt with self-hosted glyphs. No credit is needed in the UI |
| cmocean "thermal" (SST ramp derived from it) | MIT (Thyng et al. 2016, doi:10.5670/oceanog.2016.66) | **Not credited anywhere outside a code comment** (`src/lib/derived.ts:276`). Add the citation to About/sources, and keep the MIT notice if any cmocean data file is copied verbatim |

### 2.2 Techniques and libraries other agents may propose

| Candidate | Licence | Verdict and conditions |
|---|---|---|
| cambecc/earth (particles, globe) | MIT (code) | The code and ideas may be adapted with the MIT notice kept. Its **method** (bilinear interpolation in space, continuous advection, long trails, model wind and ocean fields) breaks §1.6. Adapt only with nearest-cell sampling, death in gaps, short lives, and no time interpolation. Do not copy its data sources (GFS/OSCAR are not Neritic sources) |
| Windy.com / Windy API | Proprietary. API keys under Windy's terms, attribution and their branding required. Visuals and code may not be copied | **Do not copy** UI, palettes or code. Do not embed: a Windy embed shows ECMWF/GFS forecasts that are not same-pipeline data |
| deck.gl / luma.gl | MIT | Allowed, but use it only interleaved in MapLibre's context (`MapboxOverlay({interleaved: true})`) to keep **one WebGL context**. Measure its bundle cost against §4.3. A second full-screen canvas is a budget failure |
| maplibre-gl (current, v6), pmtiles, react-map-gl, motion | BSD-3 / BSD-3 / MIT / MIT | In use |
| cmocean colormaps | MIT | Allowed with citation (see above). Each ramp must stay in its family (thermal for SST, algae-like for chlorophyll). Never use one ramp for two products |
| Google Fonts (OFL families) | SIL OFL 1.1 | Allowed. Self-host through next/font with no runtime request to Google (`src/app/layout.tsx:9`) |
| OriginKit, Casberry, or any template, UI kit or "inspiration" site with unknown terms | Unknown | **Do not copy** code, assets, illustrations or distinctive layouts. Describe the idea in our own words and build it from scratch |
| Copernicus Marine / EUMETSAT direct, GlobColour gap-free, NOAA DINEOF | Account and record-keeping terms; gap-filled data | Not without the owner's decision. Gap-filled products only as "reconstructed", never as observation (`docs/coastwatch/v2/01-sources.md:108-109`) |
| UCSC ROMS/WCOFS nests, IFCB class scores | CC BY 3.0 "not for navigational/life safety"; no licence | Excluded (`01-sources.md:107, 110`) |
| Mapbox GL JS v2+, Mapbox tiles | Proprietary | Do not use (it would replace an open stack with a metered one) |

---

## 3. Accessibility baseline

### 3.1 What exists today

- **Automated:** `tests/e2e/a11y.spec.ts` runs axe with tags `wcag2a, wcag2aa, wcag21a, wcag21aa`. It
  fails on serious or critical violations, and the WebGL canvas is excluded. Scope: Explore with a
  harbour open, the notices panel, Research's map tools and the My Coast briefing, each at desktop
  1440 and phone 390 (`a11y.spec.ts:5-50`), plus daylight mode (`:54`). Ten more specs call axe on their
  own surfaces (briefing, economics, fishing, explore, m3, m6, m7, since, mycoast-pass, research).
  **Gaps:** no `wcag22aa` tag; nothing checks /about, /sources, /bloom or /fisheries in v4 chrome; minor
  and moderate impacts are ignored; the opening/splash and the field-pack dialog are covered only by
  functional focus tests (`tests/e2e/offline-recovery.spec.ts:281`).
- **Contrast** (token pairs computed 2026-10-22, WCAG relative luminance):
  - Light (`:root`): ink-3 `#526773` is 5.92:1 on white and 4.83 on surface-3. Family colours on white
    are 5.66–7.03:1. **`--nx-gap` `#5d707b` on surface-3 is 4.21:1. That fails 4.5:1 if used for small
    text there.** `--nx-line-strong` `#c3ced4` is 1.6:1 on white, which fails 3:1 if it is the only
    boundary of an input (1.4.11).
  - Dark instrument (`.env-instrument`): ink-3 `#7c95a4` is 5.69:1 on surface but **4.41:1 on
    surface-3**, below 4.5. The family colours are 6.7–9.9:1. The short attribution plate is 12.8:1.
  - Daylight mode (`html[data-daylight]`, `src/v4/theme.css:683-720`) promises "at least 7:1" for data
    and state colours.
  - There is a forced-colors focus fallback (`theme.css:340-343`).
- **Focus:** `.nx-focus:focus-visible` draws a 2 px accent outline, offset 2 px (`theme.css:327-339`).
  The pack and notices dialogs trap focus and return it on Escape (`offline-recovery.spec.ts:281`;
  `src/components/shell/OfficialShell.tsx:57`).
- **Keyboard:** the series charts are focusable and arrow keys step through samples. They have a
  `<title>` text alternative and a "Show the samples as a table" disclosure
  (`src/v4/charts/SeriesChart.tsx:140-160, 225-240`). Escape closes the area tool and selection, and
  "a" toggles the area tool when the map has focus (`src/v4/explore/Explore.tsx:537-553`). Map
  pan and zoom use MapLibre's keyboard handlers. **Gap:** an arbitrary water point can be inspected only
  by pointer or by URL (`sel=point:`, `inspect=`). Search reaches harbours, stations and sites, not open
  water. 2.1.1 has no exception for point selection. Proposal: Enter on the focused map inspects the
  map centre under a visible crosshair.
- **Reduced motion:** `MotionConfig reducedMotion="user"` (`src/components/AppShell.tsx:37`). CSS
  sets `--nx-shift: 0` (`theme.css:660-680`). Rasters cut and the camera jumps (`motion.spec.ts:88`).
  The opening becomes a still lockup (`motion.spec.ts:119`). Particles are disabled and arrows shown
  (`src/components/map/LayerDock.tsx:698-751`). v9 rule 7: "Reduced motion is a designed version".
- **Colour independence:** each state ships an icon and a word (`theme.css:10-12`, R25). The age and
  sensor views are categorical tiles with legends (`src/lib/palette.ts:95-98`).
- **Target size:** 44 px targets are tested on phone economics (`economics.spec.ts:146`) and `min-h-11`
  is used throughout.

### 3.2 WCAG 2.2 AA requirements specific to this redesign

1. **1.1.1 / 1.3.1 Maps:** every data layer has a text equivalent next to the canvas: the layer name,
   product class, valid or observed time, legend values in text, and a "what is here" readout for the
   selection. The canvas may stay `aria-hidden` only if this holds. Legends are real text, not pixels.
2. **1.4.1 Colour:** forecast classes, the estimate classes and stale/historical states need a
   non-colour cue (the printed value, a pattern for no-data hatch and estimate, a word for state).
   Estimate versus observed must differ by more than hue, for example a texture or stipple on the
   estimate.
3. **1.4.3 / 1.4.11 Contrast:** 4.5:1 for text on **any** raster it can sit on. Text over the map
   gets a backing plate, like the short credit (`Explore.tsx:1146`). Legend ramps, focus rings, arrows
   and particles need ≥ 3:1 against adjacent colours where they carry meaning. Fix the two
   sub-4.5 pairs above before adopting them in new surfaces.
4. **2.1.1 / 2.1.2 Keyboard:** every map action works without a pointer: layer, day/hour, inspect point
   (the gap above), the area tool, and closing panels. No keyboard trap in the map or sheet.
   **Charts:** a crosshair or scrubber must be focusable, step with arrows, jump with Home/End, and
   announce its value through a polite live region or a `<title>` update. Keep the data-table
   alternative.
5. **2.2.2 Pause, Stop, Hide:** any animation that starts automatically, lasts more than 5 s and runs
   in parallel with other content needs a visible pause/stop control. Animated currents qualify, and so
   do a looping time-lapse and an ambient background. The control must work with keyboard and screen
   reader, and the choice must persist. Off under `prefers-reduced-motion` (2.3.3, AAA, adopted as
   project policy).
6. **2.3.1 Flashes:** no flashing above 3 per second. Particle shimmer or a rapid frame loop must not
   flash large areas.
7. **2.4.7 / 2.4.11 (2.2) Focus visible and not obscured:** a focused control may not be hidden under
   the phone sheet, a sticky results bar, the dock or a toast. Test focus at 320 and 390 px with the
   sheet at each snap.
8. **2.4.13 is AAA (not required).** Keep the 2 px ring at 3:1 or better on both themes.
9. **2.5.7 (2.2) Dragging:** every drag (sheet, time scrubber, area box, map pan) has a single-pointer
   alternative: tap targets, step buttons, arrow keys or zoom buttons.
10. **2.5.8 (2.2) Target size:** at least 24 × 24 CSS px. The project standard is 44 px on touch.
11. **3.2.6 (2.2) Consistent help:** the notices/agency links and "not verified" disclosure stay in the
    same place on every page (already tested in `regulatory.spec.ts`).
12. **3.3.7 / 3.3.8 (2.2):** worksheet scenarios are not re-asked. Keep no login and no cognitive
    test.
13. **4.1.3 Status messages:** loading, stale, offline, "saved copy" and pack progress go through
    `role="status"` or `aria-live="polite"` without moving focus.
14. **Dark themes:** the instrument theme is dark by default on the map. Test both themes and daylight
    mode with axe. Check the raster-overlay text plates at the darkest and lightest palette ends. Respect
    `forced-colors`.
15. **1.4.10 Reflow / 1.4.12 Text spacing:** no horizontal scroll at 320 px (tested in economics) and
    no clipped readouts with 1.5 line height and 0.12 em letter spacing.

---

## 4. Mobile performance and offline budgets

### 4.1 Measured baseline (v10-base, 2026-10-22)

Phone profile: 390 × 844, DPR 2, 4× CPU throttle, no service worker, production `next start` on :3417
with staging data, median of 3 runs. Two scripts were used: `scratchpad/perf.mjs` and
`scratchpad/v10-audit/perf-extra.mjs`. JS (dec) is the decoded script bytes. JS (wire) is the compressed
bytes on the wire. "Content" is the time until the page's own content is visible and no launch screen
covers it.

| Page | JS dec KB | FCP ms | Content ms | Long tasks ms | CLS |
|---|---|---|---|---|---|
| Explore, first visit (opening) | 1654 | 344 | 3853 | 321 | 0.001 |
| Explore, repeat | 1667 | 260 | 1035 | 292 | 0.001 |
| My Coast | 172 | 300 | 528 | 80 | 0.011 |
| Briefing (Monterey) | 172 | 300 | 566 | 200 | **0.146** |
| Research | 224 | 192 | 553 | 66 | 0 |

The v9 numbers (`scratchpad/perf-v9.json`) were 3739 / 1163 / 573 / 566 / 624 ms content. The briefing's
CLS was 0.134 at v9 and 0.146 now. It is **above the 0.1 "good" threshold** and is the one existing
budget failure.

Supplementary measurements (`perf-extra.mjs`, same profile, repeat visit, map pans 4 × 700 ms and
zooms once at 4× throttle):

| Page | JS wire KB | LCP ms | JS heap MB | DOM nodes | Canvases / WebGL maps | Frame p50 / p95 / max ms (pan + zoom) |
|---|---|---|---|---|---|---|
| Explore, C-HARM | 625 | 1184 | 17.0 | 691 | 1 / 1 | 16.7 / 16.7 / 83 (1 frame > 50) |
| Explore, currents (arrows) | 625 | 1296 | 16.2 | 680 | 1 / 1 | 16.7 / 16.7 / 17 |
| Explore, OLCI satellite | 625 | 1424 | 15.8 | 802 | 1 / 1 | 16.7 / 16.8 / 17 |
| Briefing (Monterey) | 311 | 712 | 11.5 | 3540 | 0 | — |
| Research | 260 | 272 | 11.5 | 2215 | 0 | — |

Caveats. CDP CPU throttling slows main-thread JS but not the GPU, so these frame times are a best case
for a real mid-range phone. The rule they support: the main thread is idle while the map pans today, and
any animation must keep it that way. Wire bytes include headers. On the briefing and Research, "JS wire"
is larger than perf.mjs's decoded figure, so perf.mjs under-counts scripts whose body it cannot read.
Use wire KB for budgets. The briefing's 3,540 DOM nodes are the largest DOM, which matters for
any scroll-linked effect.

### 4.2 Proposed budgets for the redesign (phone profile above; each is a CI-checkable number)

| Metric | Budget | Notes |
|---|---|---|
| JS, Explore route | ≤ 700 KB on the wire (today 625) and ≤ 1.75 MB decoded (today 1.65 MB). Any new visual layer adds **≤ 40 KB wire** to the first load. Heavier libraries are lazy-loaded with the layer | MapLibre is most of it. deck.gl or a particle engine loads on demand |
| JS, briefing / Research / My Coast | ≤ 340 KB wire each (today 311 / 260) | The economics chunk stays lazy (`offline-recovery.spec.ts:321`) |
| Content visible | Explore repeat ≤ 1.2 s. First visit with opening ≤ 3.9 s, against the v9 hard cap of 4.2 s. My Coast / briefing / Research ≤ 700 ms | Opening scope stays the first Explore visit only |
| LCP | ≤ 1.6 s on Explore (today 1.18–1.42 s), ≤ 1.0 s on briefing and Research (today 0.71 / 0.27 s); never above 2.5 s | |
| Long tasks (sum to settle) | Explore ≤ 350 ms; other routes ≤ 200 ms. No single task > 200 ms after content is visible | |
| CLS | ≤ 0.05 per route, ≤ 0.1 hard. **The briefing must get back under 0.1** | Statements settle together (v4) |
| Frame time while animating (currents or time scrub) | p95 ≤ 33 ms (30 fps) at 4× throttle; p50 ≤ 20 ms; no frame > 100 ms; pan/zoom with the animation running p95 ≤ 22 ms (today 16.7) | Particles: cap the count by viewport (v2 used 600 phone / 1400 wide), and pause on `visibilitychange` and while the map moves if needed |
| Memory | JS heap after settle ≤ 30 MB on Explore (today 15.8–17.0) and ≤ 20 MB elsewhere (today 11.5); no growth > 5 MB over 10 layer/day switches | Grids are cached per URL. Evict on layer change for large derived grids |
| WebGL | **Exactly one WebGL map context** (`motion.spec.ts:78` already asserts one map). No second full-screen canvas. A 2D overlay canvas is allowed only for particles, and only when on screen | |
| Battery | No requestAnimationFrame loop when nothing animates. Animation stops in background tabs and when the sheet covers the map | |
| Raster transitions | Layer switch fade 220 ms. Day/hour steps cut. Values never tween (v9 rules 1, 3) | |

### 4.3 Offline guarantees (field pack)

Current pack (Monterey Bay, format v2; `src/v4/offline/packs.ts:8-80`; `public/sw.js:1-35`):

- **Contents:** the Explore and briefing pages and their code chunks, including the lazy ones; the
  MapLibre worker; a 7.0 MB regional PMTiles extract (OSM 2026-10-05, SHA-256 verified); relief WebP
  images, isobaths, the water fallback and counties; Noto Sans glyphs with OFL.txt; the C-HARM run (all
  4 days × 3 variables); the latest satellite views with age grids; every published HF-radar hour plus
  the 24 h mean; and the briefing/evidence snapshot (notices, shore samples, harbour conditions as
  saved).
- **Not in the pack:** the derived index (estimate, outlook, SST). `loadDerived` returns null offline,
  and those products are simply not offered (`src/lib/derived.ts:195-200`). Research, Fisheries and About
  are also not packed.
- **Service-worker rules:** hashed static assets are cache-first in one cache per build. Pages, data and
  API are network-first, with the pack as fallback. With a pack present, the pack answers after a 5 s
  deadline. The pack's PMTiles are served with byte ranges. A page the pack never held gets the short
  offline page, never another page. A saved answer keeps its original time and is marked
  (`x-neritic-saved-for`, `__NERITIC_SAVED__`). Packs survive worker updates.

A redesign must keep all of these:

1. A saved copy says it is a saved copy and gives its save time and data dates. It is never "now", and
   it says its age once outdated (`offline-recovery.spec.ts:244`).
2. A notice in a pack shows the same "Not verified" state and the date it was last checked
   (`PackRecord.noticesCheckedAt`).
3. No new layer or animation may need a network request to render offline. If a concept adds a layer
   (SST, outlook, estimate, animated currents) to the pack, it must budget its bytes and show its dates.
   Otherwise the layer must say "not saved for offline" and must not show as empty water.
4. Pack size: about 5–10 MB per region for the basemap (v4 README), and ≤ 25 MB total per region
   including data. The plan dialog shows the total before download. A full disk is reported and partial
   packs are kept (`offline-recovery.spec.ts:62`).
5. Any service-worker change keeps the per-build code cache, keeps packs across versions, and passes
   `offline-recovery.spec.ts` (older format, renamed chunks, slow network).
6. Offline animations follow the same reduced-motion and pause rules. The particle field comes from the
   saved hour only.

---

## 5. Review checklist (apply to each concept and to the final build)

Mark each item **Pass**, **Fail** or **N/A**, with evidence (a screen, file:line or test). Any Fail in
A–C blocks the concept.

**A. Science and data semantics**
1. Every value on screen carries its product class (official / model / observation / estimate /
   outlook / analysis / history) in words, not only in colour.
2. Every value carries its own time with the right semantics: valid (C-HARM), observed per pixel
   (satellite), analysis day (MUR), window (outlook), hour (radar), sample date (stations), issue time
   (NWS).
3. Cells are drawn and read nearest, never interpolated across cells, hours, days or gaps (§1.1, §1.6).
4. Data values never tween or animate between states. Day/hour steps cut.
5. Gaps read as "no data", in a distinct pattern, never as low values or calm water.
6. Estimate and outlook are visually and verbally distinct from observations ("Estimated, not
   observed" / "Outlook, not observed"), never overwrite observed pixels, and show published skill and
   uncertainty only.
7. Chlorophyll is never presented as toxin, as bloom risk or as a fish indicator. SST and fronts are
   never presented as fish locations.
8. Forecasts are never presented as observed, and observations never as forecasts. Animated currents
   never imply a trajectory or a forecast.
9. One product per legend. No blended colour ramps across families. Each family keeps its colour.
10. No safety verdicts ("safe", "clear", "go", green-for-go), and no low-probability headline above an
    active notice (R1, R3).
11. Stale, historical and failed states show in place with real dates (freshness policy from the
    manifest, not hard-coded).
12. Research claims quote same-pipeline numbers only. The outlook is climatology + AR(1). The PINN
    matches and does not beat the ConvLSTM.

**B. Official notices and regulatory presentation**
13. "Not verified" and "may be incomplete" are visible wherever notices appear, including phone,
    offline and shared views.
14. Nothing renders as verified, lifted or expired. Past-end notices read "unconfirmed".
15. "No notice ≠ open or safe" stays next to every notice list or empty state, with agency links and
    hotlines one tap away.
16. Notice geography follows the published wording (geographic / unresolved / outside). No county
    notice is drawn offshore.
17. Commercial and recreational rules and audiences stay distinct. A purpose choice changes order and
    defaults only, never values, notices or caveats (`src/lib/purpose.ts:6-10`).
18. Nothing is a legal or safety determination. Seasons, permits and closures defer to CDFW/CDPH with
    links.

**C. Economics truthfulness**
19. The worksheet is framed as a hypothetical "Scenario from your inputs, not a forecast", and nothing
    implies the trip is allowed. A season closure or notice for the species shows above the economics.
20. No invented assumptions. Every default has a cited source and date, and catch and price are never
    pre-filled.
21. References (CDFW history, EIA diesel) are visibly distinct from user inputs, copied only on a tap,
    show "basis: …" once used, and go stale with a date.
22. Missing inputs show as "—" plus what is needed, never as a number not given.
23. Historical landings are never shown as losses, forecasts or port-level values.

**D. Attribution and licensing**
24. The OSM/OpenMapTiles credit is visible on the map at every width. Relief credit and "not for
    navigation" are present.
25. Each layer's credit and licence is reachable from the layer (Copernicus Sentinel-3 modified-data
    line, NOAA, NASA GIBS, HFRNet operators, JPL MUR, CeNCOOS operators). /sources lists MUR,
    CeNCOOS, CO-OPS, NWS and EIA, and cmocean is cited.
26. No code, assets or distinctive design copied from Windy, OriginKit, Casberry or any source of
    unknown terms. MIT notices are kept for adapted MIT code.
27. Fonts are OFL and self-hosted, with no runtime Google request. OFL.txt ships with packed glyphs.

**E. Accessibility (WCAG 2.2 AA)**
28. axe (`wcag2a/aa`, `wcag21a/aa`, **`wcag22aa`**) has no serious or critical issue on every new
    surface, in the dark, light and daylight themes, at 390 and 1440 px.
29. Text is ≥ 4.5:1 and meaningful graphics ≥ 3:1 on every background, including over rasters (with
    backing plates).
30. Full keyboard path: layers, time, inspect a point on the water, area tool, sheet, dialogs. Focus
    is visible and never obscured.
31. Charts and scrubbers: arrow/Home/End stepping, announced values, a table alternative.
32. Any motion lasting more than 5 s has pause/stop. Reduced motion is a designed still version. There
    is no flashing.
33. Targets are ≥ 44 px on touch, and every drag has a single-pointer alternative.

**F. Performance and offline**
34. The budgets in §4.2 hold on the phone profile (JS, content, LCP, long tasks, CLS ≤ 0.1, frame
    p95), and the briefing CLS is fixed.
35. There is one WebGL context, no idle rAF loop, and animation pauses when hidden.
36. Offline: every packed page opens with dated, marked saved data. Unpacked layers say so. Pack size is
    shown and budgeted. `offline*.spec.ts` passes.
37. A failed or missing source degrades to its last good data with real dates, or to an explicit
    absence. It never falls back to synthetic values.
