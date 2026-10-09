# CoastWatch design reset: design specification

Status: **concept for review.** Nothing here is implemented in `coastwatch-web/`. The prototypes in
`prototype/` are the reference rendering of this spec; where the two disagree, the spec wins and the
prototype is a bug.

---

## 1. What we are fixing

From the M3 screenshots (`docs/coastwatch/m3/`):

| Problem | Root cause | Answer in this design |
|---|---|---|
| Map drowned in an opaque pink raster | Continuous single-hue ramp; C-HARM particulate-DA values cluster at 0.45–0.80 statewide, so a smooth ramp renders as one flat colour at full opacity | 10-class stepped ramp with equal lightness steps (§5.1). The same real values now show eddies and fronts. Opacity drops with zoom so the coastline stays legible. |
| Official cards push controls off-screen | Every notice rendered as a card at the top of a single tall left column | Official status is a **component family** (§4.3): a masthead pill on every page, a compact block (max three rows) on the map, an amber section in the inspector, a one-line strip on the analysis pages, and a full drawer. Always first in reading order, never more than ~150 px. |
| Bloom page dense, charts tiny | Five equal small charts plus methodology text inline | One large primary chart with measurement tabs, demoted small multiples, a season heatmap, methodology behind disclosures (§3.2). |
| Fisheries monotonous | Three identical bar-chart cards, caveats in boxes at the top | Narrative hero, three typographic figures, one stacked chart, an interactive table with sparkbars, a designed "not available" state, methods at the end (§3.3). |
| Mobile looks like squeezed desktop | Same DOM, narrower | Mobile has its own structure: bottom tab bar with a **Notices** tab, map bottom sheet, station picker, edge-to-edge charts (§7). |
| Weak hierarchy, typography, contrast | One sans size range (12–16 px), low-contrast greys on navy everywhere | Three-family type system, a 44 px display size, reading done on paper surfaces with AA-tested ink (§2). |
| Product doesn't feel cohesive | Dark admin-console chrome on every page; "UPCOMING" nav items | One navy masthead, consistent page grammar, semantic colours that mean the same thing on every page, no placeholder navigation. |

## 2. Visual principles

1. **Navy frame, paper surfaces, dark sea.** A deep-navy masthead frames every page. Reading and analysis
   happen on warm paper (`--paper`) and white (`--surface`) because long text, tables and charts are more
   legible on light ground, including on a phone in daylight on a dock. Only the map canvas is dark,
   because the forecast field reads best as light-on-dark and the sea should look like the sea. This is
   the hybrid the brief asked us to consider. Dark everywhere was the cause of several M3 legibility issues.
2. **One job per surface.** Each panel answers one question: *what applies here*, *where*, *what is the
   selected place like*, *which forecast and when*. No panel mixes official status with model numbers.
3. **Semantic colour is reserved.** Amber is official. Violet is model. Teal is measured. Nothing
   decorative may use those hues (§2.2). A user who learns this on one page can read every page.
4. **Precedence is spatial, not just textual.** Official information is always first in reading order and
   always in the same amber container with a shield. The model never sits above it.
5. **Show the data's shape before its caveats.** Caveats stay, but they come after the thing they qualify.
   Disclosures, not paragraphs, for methods. Short qualifiers sit inline where a number could mislead,
   e.g. "reported 0 (not quantified, not absent)".
6. **Editorial, not dashboard.** Prefer typographic figures over KPI cards, one great chart over six
   small ones, tables with sparklines over grids of charts. Serif headlines give each page a voice.
7. **No decoration.** No gradients except data ramps, no animated flourishes, no glassmorphism, no
   drop-shadow stacks. Shadows only on surfaces floating over the map.

### 2.1 Typography

Three families, all SIL OFL and on Google Fonts. Production should self-host them with `next/font`.

| Role | Family | Use |
|---|---|---|
| Display | **Newsreader** (opsz 6–72, 400/500) | Page titles, station and port names, section titles, ledes |
| Text and UI | **IBM Plex Sans** (400/500/600) | Body, labels, controls, figures. Tabular lining numerals are on globally (`font-feature-settings: "tnum", "lnum"`) |
| Data | **IBM Plex Mono** (400/500) | Coordinates, codes, IDs only |

Why: Newsreader gives the editorial voice that separates CoastWatch from an admin console. Plex Sans is
engineered, legible at small sizes, and has true tabular figures for aligned numbers. Plex Mono shares
its metrics.

Scale (tokens in `prototype/tokens.css`):

| Token | Spec | Example |
|---|---|---|
| `--t-display-1` | Newsreader 500, 44/48 | "Santa Cruz Wharf", page H1 |
| `--t-display-2` | Newsreader 500, 32/37 | Inspector title, mobile H1 |
| `--t-display-3` | Newsreader 500, 24/29 | Section titles |
| `--t-lede` | Newsreader 400, 19/29 | Fisheries narrative |
| `--t-title` | Plex Sans 600, 18/25 | Chart titles |
| `--t-heading` | Plex Sans 600, 16/22 | Panel headings, disclosure summaries |
| `--t-body` | Plex Sans 400, 16/26 | Prose |
| `--t-ui` | Plex Sans 400, 14/20 | Controls, table cells |
| `--t-small` | Plex Sans 400, 13/19 | Secondary lines, captions |
| `--t-axis` | Plex Sans 400, 12 | **Chart axes only.** Nothing else is below 13 px |
| `--t-figure` | Plex Sans 600, 40–52 | Headline numbers (76%, $53M) |
| eyebrow | Plex Sans 600, 12, +0.08em, uppercase | Section category labels (`OFFICIAL · 3 MAY APPLY`) |

Minimum sizes: body copy 16 px, interactive labels 14 px (13 px only in mobile chips), axes 12 px.

### 2.2 Colour

Frame and surfaces:

| Token | Hex | Role |
|---|---|---|
| `--navy-950` | `#06111e` | Masthead |
| `--navy-900` | `#0a1a2c` | Selected rows, tooltips, selected time step |
| `--sea` / `--land` / `--coast` | `#0b1d33` / `#1c2a3c` / `#a8bfd6` | Map basemap |
| `--paper` | `#f5f3ee` | Page background |
| `--surface` | `#ffffff` | Panels, cards, tables |
| `--line` / `--line-strong` | `#e4e0d7` / `#cbc4b6` | Hairlines |
| `--ink` / `--ink-2` / `--ink-3` | `#0d1b2a` / `#3d4b5c` / `#5f6b79` | Text hierarchy |
| `--accent` | `#0b6a85` | Links, focus, selection (non-semantic interaction colour) |

Reserved semantic colours (never decorative):

| Meaning | Text | Background | Line | On navy |
|---|---|---|---|---|
| **Official** (CDFW/CDPH notices) | `#9a4a00` / strong `#7a3a00` | `#fdf2e0` | `#eeb565` | `#f6bb5c` |
| **Model** (C-HARM) | `#6a3fb0` | `#f3eefb` | `#cdbcef` | `#c3aef3` |
| **Measured** (CalHABMAP, satellite) | `#0d756c` | `#e5f3f0` | `#a6d6cf` | `#3fc1b0` (map dots) |
| Historical economics | `#34495f` | `#eef1f5` | `#cfd8e3` | — |

Freshness always pairs a colour with a **word** and a **glyph**. Current is a filled dot `#1d7f53`, stale a
half-filled dot `#9a6a00`, historical an open ring `#6b7480`. Colour is never the only signal.

Measured contrast (WCAG 2.x):

| Pair | Ratio |
|---|---|
| `ink` on `paper` | 15.7:1 |
| `ink-2` on `paper` | 8.0:1 |
| `ink-3` on `paper` / `surface` | 4.9:1 / 5.4:1 |
| `official-strong` on `official-bg` | 7.8:1 |
| `official` on `official-bg` | 5.7:1 |
| `model` on `model-bg` | 6.3:1 |
| `measured` on `surface` | 5.6:1 |
| `accent` on `surface` | 6.2:1 |
| freshness current / stale / historical on `surface` | 5.0 / 4.7 / 4.7:1 |
| `on-navy` / `on-navy-2` on `navy-950` | 16.3 / 9.4:1 |
| `official-on-navy` on `navy-950` | 11.0:1 |
| forecast class 0–10 % on `sea` | 2.4:1 (non-text; keeps "low" distinct from "no value") |

Fisheries categorical palette: Tier 1 groups are blues (`#173f66`, `#3a75ab`, `#86b6dc`) and Tier 2
groups are sage (`#4f7a64`, `#86a996`, `#c3d3c9`). Hue family encodes tier, so the stack reads by tier
before by species.

## 3. Page specifications

All three pages share: masthead → official precedence element → page content. Desktop layouts are
designed at 1440 × 900 and verified at 1280 × 720. Mobile is designed at 390 × 844.

### 3.1 Ocean Map

**Purpose:** see the agency forecast along the whole coast, find your port, read it, and see what official
notices apply, without leaving the map.

Desktop layout (screenshots `01`–`03`):

```
┌ masthead ──────────────────────────────────────── [⛨ 8 official notices | Not verified] [● Data status] ┐
│┌ Coast panel 344 ─┐                                                         ┌ Inspector 384 ──────────┐│
││ OFFICIAL block   │                                                         │ PORT · MONTEREY BAY     ││
││  ≤3 rows + All 8 │                  MAP (full bleed)                       │ Santa Cruz              ││
│├──────────────────┤                                                         │ OFFICIAL · 3 MAY APPLY  ││
││ Coast, N → S     │                                                         │ MODEL FORECAST · C-HARM ││
││  region rows w/  │                                                         │  76%  + 4-day strip     ││
││  port ticks      │   ┌ Forecast dock (bottom-left of map area) ─┐          │  30-day sparkline       ││
││  ports (expanded)│   │ [Particulate DA|P-n|Cellular DA] [Layers] │          │ MEASURED NEARBY         ││
│└──────────────────┘   │ [Wed 7 Nowcast|Today 8|Fri 9|Sat 10]     │          │ SATELLITE               ││
│                       │ threshold text · 10-class ramp · Model ● │          └─────────────────────────┘│
└───────────────────────┴───────────────────────────────────────────┴─────────────────────────────────────┘
```

- **Map framing.** California's coast runs north-west to south-east. The coast panel covers open ocean,
  the inspector covers Nevada, and the dock sits over the ocean south-west of the Bight. The statewide
  view is fitted between the measured panel edges, never behind them. Region views add about 0.5° of
  context so 3 km model cells stay small on screen.
- **Coast index** (the left panel's lower half) replaces the region tab bar. Each region row shows the
  min–max of its ports' median probabilities and a strip of port ticks coloured with the map ramp, north
  to south. The coast's shape thus appears in the list. Expanding a region lists its ports, each with a
  bar and a value.
- **Forecast dock.** Holds the quantity tabs, a four-day timeline (Nowcast, Today, +2, +3; dates in
  Pacific time; "Today" when the valid date is today), the exact threshold sentence from the manifest,
  the stepped legend with a "No model value" swatch, the `Model` chip, the source and issue date, and
  freshness.
- **Inspector order** (fixed): identity → **Official** (amber, every related record with its relation:
  statewide / port within the notice's latitudes / same county / named area nearby) → **Model** (big
  median, 4-day strip with min–max range bars and median tick, 30-day nowcast sparkline, the
  "probability for nearby water, not a measurement and not a closure decision" line) → **Measured
  nearby** (the nearest CalHABMAP station, its last particulate DA with date and freshness, link into
  Bloom Intelligence) → **Satellite** (8-day VIIRS chlorophyll median, "algae biomass, not toxin",
  cloud-free fraction) → the port caveat.
- **Map layers, bottom to top:** land, landcover, water, forecast raster (below the graticule and
  coastline), graticule with degree labels, coastline, roads, state line, official geometry (amber,
  dark-cased lines and dashed polygons, latitude-limit labels from the geometry), offshore region labels
  (low zoom only), cities (zoom ≥ 6.6), CalHABMAP stations (teal dots), ports (white dots, labels at
  zoom ≥ 6.6), selection halo.
- **Not in the prototype, required in production:** click-to-read a cell value (the existing M2
  inspector reading the published u16 grid), satellite chlorophyll as an alternative layer under
  "Layers", keyboard navigation of ports, and the URL state that M2 already has.

### 3.2 Bloom Intelligence

**Purpose:** follow measured toxin and Pseudo-nitzschia at one shore station, know how fresh the data are,
and see the agency model alongside the measurements without confusing the two.

Desktop structure (screenshots `04`–`06`):

1. **Official strip:** a one-line strip with the notices related to the station's nearest port, the
   verification state, and "All 8 notices".
2. **Station rail** (sticky, 296 px): 17 stations grouped by region, north to south. Each row shows the
   name, a freshness glyph for the latest sample, and one sub-line giving the latest sample date and the
   particulate DA status ("pDA 0.20 ng/mL", "pDA reported 0", "pDA last measured Aug 2022", "No
   particulate DA since 2014").
3. **Station header:** eyebrow `SHORE STATION · REGION · CALHABMAP`, the name in display serif,
   coordinates in mono, location code, distance to port, and "Open on map".
4. **Freshness card** (the most prominent element after the name): the latest sample date in display
   type, the freshness word and age, a **120-day sampling strip** (one tick per sample, the gap since the
   last sample shaded, "today" marked), and the sampling cadence and lab-latency sentence. Its top rule
   takes the freshness colour. If the latest sample is historical, a full-width banner says the values do
   not describe current conditions.
5. **Readouts:** five cells in one bordered row, one per quantity. Each cell is also the selector for the
   main chart. A value older than the stale limit **never gets headline size**: the cell reads "Not
   measured since Aug 2022", with the last value in small text. This prevents a four-year-old number from
   looking current (Monterey Wharf, screenshot `05`).
6. **Primary chart** (≈ 340 px tall): measurement tabs; a 3 mo / 1 yr / 3 yr / Since 2014 range; fixed
   whole-decade log axes; points and joining lines that break at gaps over 21 days; reported zeros in a
   separate `0*` lane as open rings; a tick for every sampling visit (so "not measured" is visible);
   "Highest in view" annotated with value and date; a crosshair tooltip. Summary line: "Measured in 49 of
   49 sampling visits in view · 10 reported 0".
7. **Other measurements, same period:** a 2 × 2 grid of small multiples, each with its own axis and
   latest value. Selecting one makes it the primary chart. A quantity with no data in the period gets a
   hatched placeholder with its last-measured date, not an empty axis.
8. **Historical context:** a calendar heatmap (years × weeks on desktop, years × months on mobile) of the
   highest value per period since 2014. Teal ramp with **decade bins, labelled "not risk levels"**.
   Reported-0-only periods and unsampled periods have their own swatches.
9. **Agency forecast band:** a violet-tinted, full-width band with `MODEL, NOT A MEASUREMENT` and the
   `C-HARM v3.1 · NOAA` chip. It shows the nowcast median within 15 km on the **same time axis** as the
   primary chart (capped at 12 months). The period before CoastWatch kept model history is hatched and
   labelled. Partial-coverage runs are drawn as open markers (§5.4). The "never compared numerically"
   sentence stays.
10. **About these data:** three short columns (what a value means / blanks and zeros / seawater is not
    seafood), then disclosures for quantities, station QC and processing, then source, licence and the
    review-pending note.

### 3.3 Fisheries & Economic Exposure

**Purpose:** understand the historical commercial value of the species that marine toxins affect, framed as
history and never as a forecast of loss.

Structure (screenshots `07`, `08`):

1. **Official strip:** the notices referenced by the species groups (rock crab, anchovy, bivalves).
2. **Hero:** a `Historical` chip, scope eyebrow, serif H1 ("What California's toxin-affected fisheries have
   landed"), the published terminology sentence as the lede, and a side note "Annual data through 2024.
   2025 not yet published by NOAA."
3. **Controls bar:** Species (Tier 1 / Tiers 1 + 2) and Dollars (2024 dollars / Nominal), between two
   hairlines.
4. **Three figures,** typographic with no card chrome: latest-year value; 10-year average with the lowest
   and highest years; share of NOAA's state total with the withheld amount stated.
5. **Primary chart:** stacked annual bars by group, totals on top, and a "share of NOAA's state total" row
   under the year labels. The 2015–16 bracket ("Dungeness season delayed by domoic acid (CDFW)") comes
   from the group's published `tier_basis`. A legend that also acts as a filter.
6. **By species group:** one table with each group, its tier, latest value, 10-year average, 10-year
   sparkbars and share. Out-of-set groups are dimmed, not hidden. Selecting a row highlights the group in
   the chart and opens a **detail panel** with the tier basis, linked official notices (amber), pounds
   (meat weight for bivalves), NOAA market categories, and the aquaculture caveat for bivalves.
7. **Port-level, not available:** a designed empty state, not a warning box. It shows the published
   reasons beside the nine CDFW port areas, north to south, as locked, hatched rows, with "Statewide
   values are never divided among ports". The future layout is visible, so its absence explains itself.
8. **How these numbers are made:** four short columns (source with the NOAA courtesy line, inflation,
   confidentiality, duplicates), then disclosures for all caveats, the full values table (including the
   withheld row) and processing steps, then the licence line.

## 4. Navigation, layout and component hierarchy

### 4.1 Navigation

- Desktop masthead: logo and wordmark, "California", primary nav **Ocean Map · Bloom Intelligence ·
  Fisheries**, then on the right the **official pill** (count + verification state, opens the drawer) and
  **Data status** (the existing Sources page). Remove "My Coast — UPCOMING": unbuilt features do not
  belong in primary navigation.
- Mobile: compact masthead (logo + official pill) and a bottom **tab bar**: Map, Blooms, Fisheries,
  **Notices** (amber, with a count badge). On a phone, official notices are one tap from anywhere.
- Cross-links carry context: port → station (`bloom.html?station=…`) and station → port on the map
  (`map.html?port=…&region=…`).

### 4.2 Layout grid

- 4 px spacing base. Panel padding 16–24, section gaps 28–36, page gutters 32 desktop and 16 mobile.
- Analysis pages: 296 px rail + fluid main (Bloom), or a 1120 px editorial column (Fisheries).
- Map: floating panels 16 px from the edges. Coast panel 344 px, inspector 384 px, dock ≤ 600 px. With a
  port open, the dock never extends under the inspector.
- Radii: 6 (chips), 10 (controls), 16 (panels and cards). Shadows only on map panels.

### 4.3 Component hierarchy

```
AppShell
├─ Masthead ─ Brand · PrimaryNav · OfficialPill → OfficialDrawer · DataStatusLink
├─ OfficialDrawer (global) ─ VerificationNote · NoticeArticle* (by agency) · Statement* · Hotline*
├─ TabBar (mobile) ─ Map · Blooms · Fisheries · Notices(badge)
├─ Official family: OfficialPill · OfficialStrip · OfficialBlock (map) · OfficialSection (inspector) · NoticeRow
├─ Status: FreshnessChip · VerificationLine · ProductChip(Model | Measured | Historical) · AgencyChip
├─ Controls: Segmented · TabList · Timeline(day steps) · RegionChips (mobile)
├─ Map: MapCanvas · CoastPanel(OfficialBlock, CoastIndex(RegionRow, PortRow)) · ForecastDock(Legend) · PortInspector · BottomSheet (mobile)
├─ Bloom: StationRail(StationRow) · StationHeader · FreshnessCard(SamplingStrip) · ReadoutRow(Readout) ·
│         ObservationChart(primary | compact) · SeasonHeatmap · ModelBand(ModelTrack) · AboutData
└─ Fisheries: Hero · ControlsBar · FigureRow(Figure) · StackedBars · BreakdownTable(SparkBars) · GroupDetail ·
              UnavailablePorts · MethodsGrid · ValuesTable
```

Every chart component takes **already-computed** series from `lib/` helpers (as M3 does), so the
presentation layer cannot change scientific handling.

## 5. Data visualisation conventions

### 5.1 Forecast probability on the map

- **Ten classes of ten percentage points**, stepped and not interpolated. Hex in `tokens.css` (`--p0`…`--p9`)
  and `scripts/render_rasters.py`.
- Built in OKLCH. Lightness rises in equal steps from 0.47 to 0.95; adjacent classes differ by ≈ 1.2:1;
  hue turns indigo → violet → magenta → rose → shell. Order survives greyscale and colour-vision
  deficiency.
- The lowest class keeps 2.4:1 against the sea, so "low probability" never reads as "no value". No value
  is transparent (the sea shows) and has its own legend swatch.
- The domain is fixed at 0–100 % and never stretched to the data.
- Raster opacity 0.84 at zoom ≤ 6, 0.72 at 8 and 0.55 at ≥ 10, so the coast and labels read through at
  harbour scale. Resampling stays **nearest**: a pixel is a model cell, never interpolated.
- The legend title is the manifest's `threshold_text` verbatim ("Probability that particulate domoic acid
  exceeds 500 ng per litre").
- **Production change:** the pipeline palette `cw-probability-magenta-v2` would be replaced by a new
  palette id (e.g. `cw-probability-classes-v1`) in `process/palette.py`. Only the PNG colouring changes;
  the u16 value grids, schema and port statistics are untouched. See the implementation plan.

### 5.2 Measurements

- Log axes over **fixed whole decades** per quantity: pDA 10⁻⁴–10² ng/mL, Pseudo-nitzschia 1–10⁷ cells/L,
  chlorophyll 10⁻²–10³ mg/m³. Temperature is linear 5–30 °C. Axes never rescale to hide or exaggerate.
- Points are teal. Lines join consecutive samples and break at gaps over 21 days. Reported zeros sit in a
  separate `0*` lane as open rings, never on the log axis. Flagged-high values are open teal markers.
- Every sampling visit gets a tick, so "not measured" is visible without reading a table.
- Values are shown to two significant figures with trailing zeros kept (0.20, not 0.2), whole counts with
  thousands separators, and temperature to 0.1 °C.

### 5.3 Heatmaps and bins

Decade bins for log quantities (pDA < 0.01, 0.01–0.1, 0.1–1, 1–10, ≥ 10 ng/mL) use a teal ramp, because
these are measurements. The caption says **"Decade bins, not risk levels."** No regulatory action levels
are drawn on seawater data, since action levels apply to seafood tissue.

### 5.4 Model tracks

Violet line and area on a fixed 0–100 % axis with a 50 % gridline. Days without a run are left blank, not
interpolated. The span before CoastWatch kept history is hatched and labelled. A run whose neighbourhood
had fewer than 60 % of the usual valid cells is drawn as an open marker and not joined. In the current
data no Santa Cruz run meets that test: its day-to-day swings are full-coverage model output, and the
chart states they are not smoothed.

### 5.5 Economics

Stacked bars with totals on top and a share-of-state row. The dollar basis is always named in the chart
subtitle. Groups are lower bounds and the withheld amount is never attributed to a group (both stated
next to the table). Figures use `$53M`-style abbreviations, with whole millions above $10M and one
decimal below.

### 5.6 Interaction

Hover and keyboard tooltips (the M3 arrow-key and Escape behaviour carries over), crosshair to the
nearest sampling visit, and tooltips that state "Not measured in this sample" or "Reported 0 (not
quantified)" explicitly.

## 6. Scientific and regulatory presentation rules

These are acceptance criteria for implementation. Each maps to an existing M1–M3 safety rule or test.

1. **Official first.** On every page, official information precedes model and measurement content in
   reading order and is reachable in one interaction (masthead pill / Notices tab). The e2e
   `compareDocumentPosition` test pattern applies to every page.
2. **Verification is never upgraded by design.** "Not verified" appears wherever a notice list or row
   group appears, until `review.status` says otherwise. The drawer explains what that means.
3. **Agency wording is quoted.** Notice titles are the registry titles, the drawer quotes `official_text`,
   and statements are quoted with their agency.
4. **No safety implication.** No green "all clear", no "safe", "open" or "loss" except in negated
   sentences. Low forecast values carry "does not mean it is safe to fish or harvest". The existing
   `assertNoUnsafeClaims` check runs on every page.
5. **Observation vs model is visual, not only verbal.** Teal vs violet, separate containers, `Model` and
   `Measured` chips, and "never compared numerically". No chart mixes the two on one y-axis.
6. **Freshness is explicit and computed in Pacific calendar days** from the dataset's own dates. The
   prototype uses the snapshot time as "now"; production uses the request time as M3 does. Stale and
   historical states change the colour, the word and the glyph.
7. **Missing is not zero.** Blank, reported 0, rejected, not sampled and no model value each have their
   own visual treatment and legend entry.
8. **Old values never look current** (readout rule in §3.2).
9. **History is not forecast.** The fisheries page carries the `Historical` chip, the terminology
   sentence as its lede, and "Past landings only, not a forecast" in the chart subtitle.
10. **Attribution travels with the data:** C-HARM v3.1 · NOAA on the map and the model band, CalHABMAP via
    SCCOOS ERDDAP with the licence on Bloom, "Courtesy: National Oceanic and Atmospheric Administration"
    on Fisheries, OpenFreeMap / OpenMapTiles / OSM on the map.
11. **Review state is visible.** "These pages have not yet been reviewed by an independent HAB scientist"
    and the tier-review note remain until sign-off. This design does not claim it.

## 7. Mobile interaction design

Designed at 390 × 844; the layout switches at ≤ 720 px.

- **Tab bar** (64 px): Map, Blooms, Fisheries, **Notices** (amber, count badge). Notices opens the
  drawer full-screen.
- **Map:** the map fills the screen between masthead and tab bar. The **bottom sheet** peeks with, in
  order: an official row ("8 official notices in California · Not verified ›"), quantity tabs, region
  chips (horizontal scroll), the day timeline and the legend. Selecting a port raises the sheet to ~72 %
  and turns it into the inspector. The official section comes first there, condensed to one line per
  notice. The map re-frames using the sheet's measured height, and offshore region labels are hidden
  because the chips replace them.
- **Bloom:** an official line under the masthead, then a **station picker** button (name, region,
  "17 stations") that opens the rail as a full-screen list. Then the freshness card, readouts as a 2 × 2
  grid plus one full-width cell, an edge-to-edge primary chart with scrollable measurement tabs,
  single-column small multiples, a years × months heatmap, the model band and accordions.
- **Fisheries:** narrative first, full-width segmented controls with short labels ("Tier 1", "Tiers
  1 + 2"), stacked figures, an edge-to-edge chart with two-digit years, a two-column table (value,
  share; average and sparkbars hidden), a detail panel shown only after selection, the port empty state
  and the methods.
- Touch targets ≥ 44 px for primary controls; no hover-only information; no horizontal page scroll
  (checked by the screenshot script and the existing e2e overflow test).

## 8. Known gaps in the prototypes

The prototypes are static HTML/JS for evaluating the design, not production code. Not implemented:
click-to-read map cells, the satellite layer switch, the full-screen mobile station list styling, the
Sources page, the M2-data and failed-update states (production already has them; the patterns in §6
apply), the error states, and full keyboard support beyond what the HTML provides. The "Layers" button
is inert.
