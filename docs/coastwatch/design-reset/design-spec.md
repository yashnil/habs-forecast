# CoastWatch design reset: design specification

Status: **approved direction, revision 2 (2026-10-09).** The overall design system was approved in the
PR #7 review with targeted refinements, all folded into this revision (summary below). Implementation
starts with phase P0 of the implementation plan. The prototypes in `prototype/` are the reference
rendering of this spec; where the two disagree, the spec wins and the prototype is a bug.

### Review decisions

| Decision | Outcome |
|---|---|
| Light editorial analytical pages, dark ocean map | **Approved** |
| Banded forecast colours | **Approved**, on condition that the bands are presented as visual display intervals, not scientifically validated risk categories, with exact source values and thresholds preserved (§5.1, rule 12 in §6) |
| Newsreader / IBM Plex Sans / IBM Plex Mono | **Approved**, self-hosted in production |
| Remove "My Coast" from navigation until it exists | **Approved** |
| Optional "Monitoring stations" map toggle | **Deferred** until after the primary redesign |

### Revision 2 refinements

- **Ocean Map:** no longer opens with two full side panels. A compact navigation card (official summary,
  port search, region list) replaces the full-height coast panel; the inspector opens only when a port
  is selected, and the navigation card then shrinks to a breadcrumb. New calmer palette, opaque raster
  so legend colours equal map colours, lighter land, a brighter coastline drawn above the forecast,
  hatching for "no model value", a pointer readout of the exact cell value, and tighter region framing
  (Monterey Bay now runs Año Nuevo to Point Sur).
- **Bloom Intelligence:** the five latest-value cards become the measurement selector, each with a small
  preview on the chart's period and axis. The four secondary full charts are gone. On mobile the
  selector is one swipeable row, and the history heatmap and the about notes are collapsed. The page is
  about 38 % shorter on a phone (4,490 → 2,770 px) and 18 % shorter on desktop.
- **Fisheries:** the port-level section is reduced to one quiet line that expands on demand, and a
  sentence directly under the headline figures says past landings are not losses or a forecast of
  losses. The page is about 15 % shorter.

---

## 1. What we are fixing

From the M3 screenshots (`docs/coastwatch/m3/`):

| Problem | Root cause | Answer in this design |
|---|---|---|
| Map drowned in an opaque pink raster | Continuous single-hue ramp; C-HARM particulate-DA values cluster at 0.45–0.80 statewide, so a smooth ramp renders as one flat colour at full opacity | 10-class stepped, low-chroma ramp whose common 60–80 % classes are mid tones (§5.1). The same real values show eddies and fronts, and the coastline, land and labels stay legible. |
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
| `--sea` / `--land` / `--coast` | `#0b1d33` / `#2a3646` / `#d3dfeb` | Map basemap. Land is a lighter slate than the sea (1.4:1) and the coastline is drawn above the forecast |
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
| forecast classes, adjacent | ≈ 1.2:1 each (non-text, equal OKLCH lightness steps) |
| coastline on forecast class 60–70 % / 70–80 % | 2.4 / 2.0:1 (non-text) |

Fisheries categorical palette: Tier 1 groups are blues (`#173f66`, `#3a75ab`, `#86b6dc`) and Tier 2
groups are sage (`#4f7a64`, `#86a996`, `#c3d3c9`). Hue family encodes tier, so the stack reads by tier
before by species.

## 3. Page specifications

All three pages share: masthead → official precedence element → page content. Desktop layouts are
designed at 1440 × 900 and verified at 1280 × 720. Mobile is designed at 390 × 844.

### 3.1 Ocean Map

**Purpose:** see the agency forecast along the whole coast, find your port, read it, and see what official
notices apply, without leaving the map. The map is the visual centre: by default only one compact card
and the dock float over it.

Desktop layout (screenshots `01` statewide, `02` region, `03` port selected, `04` exact value, `05`
drawer):

```
┌ masthead ──────────────────────────────────────── [⛨ 8 official notices | Not verified] [● Data status] ┐
│┌ Navigation 312 ─┐                                                                                     │
││ ⛨ 8 official    │                                                                                     │
││   notices · NV ›│                       MAP (full bleed, the centrepiece)                              │
││ [Find a port   ]│                                                                                     │
││ Coast, N → S    │                                                                                     │
││  North Coast  › │                                                                                     │
││  …6 regions     │                                                                                     │
│└─────────────────┘                                                                                     │
│┌ Dock 624 ─────────────────────────────────────────────┐                                              │
││ [Particulate DA | P-n | Cellular DA]  [Wed 7|Today|Fri 9|Sat 10]                                      │
││ threshold text · day, lead · 10-step ramp · ▨ No model value · "display steps, not risk levels"       │
││ Model · C-HARM v3.1 · NOAA · issued Oct 8                                          ● Current          │
│└────────────────────────────────────────────────────────┘                                             │
└───────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

- **Three states, never two full panels.**
  - *Statewide (default):* the navigation card shows the official summary row (count, scope, "Not
    verified", opens the drawer), port search, and the six regions with the range of their port
    medians. No inspector.
  - *Region:* the card shows a back link, the region name, its ports with bars and values, and the
    official row scoped to the region ("4 official notices may apply in Monterey Bay"). The map frames
    the region.
  - *Port selected:* the inspector opens on the right and the navigation card shrinks to search plus a
    breadcrumb ("‹ Monterey Bay · 3 ports"). The inspector's official section, first in reading order,
    carries the notices, so they are not repeated in the card. The masthead pill still shows all 8.
- **Map framing.** Each region has a hand-set view tight on the coast that matters (Monterey Bay: Año
  Nuevo to Point Sur, Santa Cruz to Monterey filling the visible area). Views are fitted inside the
  measured panel edges: left of the navigation card, above the dock, left of the inspector.
- **Exact values.** Pointing at the water (tap on mobile) shows the model value of that 3 km cell to one
  decimal, read from the published u16 grid with the same nearest-cell logic as production
  `lib/grid.ts`: "70.6 % · model cell at 36.78°N 122.12°W". Where there is no value it says so, and
  whether it is outside the model area or land / masked nearshore water, giving the nearest cell within
  three cells and its distance instead of pretending it is the value at the point.
- **Forecast dock.** Quantity tabs and the day steps on one row (Wed 7 is the nowcast; the others are
  forecasts, named in the legend line and in each step's tooltip), the manifest's threshold sentence
  verbatim with the valid day and lead, the 10-step ramp with boundary labels, the hatched "No model
  value" swatch, the "10-point colour steps for display, not risk levels" note, and the `Model` chip with
  source, issue date and freshness.
- **Inspector order** (fixed): identity → **Official** (amber, every related record with its relation, and
  "Agency notices decide what is open. This page does not.") → **Model** (big median, 4-day strip with
  min–max range bars and median tick, 30-day nowcast sparkline, the "probability for nearby water, not a
  measurement and not a closure decision" line) → **Measured nearby** (nearest CalHABMAP station, last
  particulate DA with date and freshness, link into Bloom) → **Satellite** (8-day VIIRS chlorophyll
  median, "algae biomass, not toxin", cloud-free fraction) → the port caveat.
- **Map layers, bottom to top:** land, landcover, water, **no-value hatching**, forecast raster (opaque),
  graticule, roads, state line, **coastline (above the forecast)**, official geometry (amber, dark-cased
  lines and dashed polygons, latitude-limit labels at zoom ≥ 7.4), graticule labels (desktop only), cities
  (zoom ≥ 7.2), CalHABMAP stations (teal dots, zoom ≥ 6.5), ports (white dots, labels at zoom ≥ 7.4),
  selection halo. The offshore region labels of revision 1 were removed: the navigation card names the
  regions.
- **Not in the prototype, required in production:** satellite chlorophyll as an alternative layer,
  keyboard navigation of ports, and the URL state M2 already has.

### 3.2 Bloom Intelligence

**Purpose:** follow measured toxin and Pseudo-nitzschia at one shore station, know how fresh the data are,
and see the agency model alongside the measurements without confusing the two.

Desktop structure (screenshots `07`–`10`):

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
5. **Measurement selector:** five cards, one per quantity, under the heading "Latest measurement of each
   quantity". Each shows the latest value, its date and freshness, the 12-month sample count, and a
   **preview**: the samples in the chart's current period on the same fixed axis, with reported zeros as
   open dots on the baseline and gaps left empty. Selecting a card makes it the primary chart. A value
   older than the stale limit **never gets headline size**: the card reads "Not measured since Aug
   2022", with the last value in small text. This prevents a four-year-old number from looking current
   (Monterey Wharf, screenshot `09`). A quantity never measured at the station says so and has an empty
   preview. The Pseudo-nitzschia and chlorophyll caveats sit under the row.
6. **Primary chart** (≈ 340 px tall): a 3 mo / 1 yr / 3 yr / Since 2014 range; fixed whole-decade log
   axes; points and joining lines that break at gaps over 21 days; reported zeros in a separate `0*` lane
   as open rings; a tick for every sampling visit (so "not measured" is visible); "Highest in view"
   annotated with value and date; a crosshair tooltip. Summary line: "Measured in 49 of 49 sampling
   visits in view · 10 reported 0". The selector cards replace revision 1's measurement tabs.
7. *(Removed in revision 2: the 2 × 2 small multiples. The selector previews cover what they showed
   without drawing four more full charts.)*
8. **Historical context:** a calendar heatmap (years × weeks on desktop, years × months on mobile) of the
   highest value per period since 2014. Teal ramp with **decade bins, labelled "not risk levels"**.
   Reported-0-only periods and unsampled periods have their own swatches. Open on desktop; a collapsed
   disclosure on mobile, with its title and "not risk levels" caption still visible.
9. **Agency forecast band:** a violet-tinted, full-width band with `MODEL, NOT A MEASUREMENT` and the
   `C-HARM v3.1 · NOAA` chip. It shows the nowcast median within 15 km on the **same time axis** as the
   primary chart (capped at 12 months). The period before CoastWatch kept model history is hatched and
   labelled. Partial-coverage runs are drawn as open markers (§5.4). The "never compared numerically"
   sentence stays.
10. **About these data:** three short columns on desktop (what a value means / blanks and zeros / seawater
    is not seafood), disclosures on mobile, then disclosures for quantities, station QC and processing,
    then source, licence and the review-pending note. The chart legend keeps the inline qualifiers
    ("Gaps are not zeros", "0* reported 0 (not quantified, not absent)") visible at all times.

### 3.3 Fisheries & Economic Exposure

**Purpose:** understand the historical commercial value of the species that marine toxins affect, framed as
history and never as a forecast of loss.

Structure (screenshots `11`, `12`):

1. **Official strip:** the notices referenced by the species groups (rock crab, anchovy, bivalves).
2. **Hero:** a `Historical` chip, scope eyebrow, serif H1 ("What California's toxin-affected fisheries have
   landed"), the published terminology sentence as the lede, and a side note "Annual data through 2024.
   2025 not yet published by NOAA."
3. **Controls bar:** Species (Tier 1 / Tiers 1 + 2) and Dollars (2024 dollars / Nominal), between two
   hairlines.
4. **Three figures,** typographic with no card chrome: latest-year value; 10-year average with the lowest
   and highest years; share of NOAA's state total with the withheld amount stated. Directly under them:
   "Past landings show what was at stake in earlier seasons. They are not losses, not a forecast of
   losses, and say nothing about the current season."
5. **Primary chart:** stacked annual bars by group, totals on top, and a "share of NOAA's state total" row
   under the year labels. The 2015–16 bracket ("Dungeness season delayed by domoic acid (CDFW)") comes
   from the group's published `tier_basis`. A legend that also acts as a filter.
6. **By species group:** one table with each group, its tier, latest value, 10-year average, 10-year
   sparkbars and share. Out-of-set groups are dimmed, not hidden. Selecting a row highlights the group in
   the chart and opens a **detail panel** with the tier basis, linked official notices (amber), pounds
   (meat weight for bivalves), NOAA market categories, and the aquaculture caveat for bivalves.
7. **Port-level, not available:** one quiet line under the table, "Port-level values are not available
   yet. Everything here is statewide and is never divided among ports. Why", which expands to the
   published reasons and the nine CDFW port areas. Revision 1's full-width locked-rows section gave an
   absence more space than the data; it is gone.
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
- Map: floating panels 16 px from the edges. Navigation card 312 px (top left, height to content, never
  overlapping the dock), dock 624 px (bottom left), inspector 368 px (right, only with a port). With a
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
├─ Controls: Segmented · DaySteps · PlaceSearch · QuantitySelect (mobile)
├─ Map: MapCanvas · NavCard(OfficialSummaryRow, PlaceSearch, RegionList | PortList | Breadcrumb) ·
│       ForecastDock(Legend) · ValueReadout · PortInspector · BottomSheet + PlaceButton (mobile)
├─ Bloom: StationRail(StationRow) · StationHeader · FreshnessCard(SamplingStrip) ·
│         MeasurementSelector(SelectorCard(MiniPreview)) · ObservationChart · SeasonHeatmap(collapsible) ·
│         ModelBand(ModelTrack) · AboutData
└─ Fisheries: Hero · ControlsBar · FigureRow(Figure) · NotLossNote · StackedBars · BreakdownTable(SparkBars) ·
              GroupDetail · PortLevelNote · MethodsGrid · ValuesTable
```

Every chart component takes **already-computed** series from `lib/` helpers (as M3 does), so the
presentation layer cannot change scientific handling.

## 5. Data visualisation conventions

### 5.1 Forecast probability on the map

- **Ten display classes of ten percentage points**, stepped and not interpolated. They are colour steps
  for reading the map, **not risk categories**: no class has a name, the legend says "10-point colour
  steps for display, not risk levels", and nothing in the UI attaches meaning to a class boundary. The
  exact value of any cell is one pointer move away (§3.1), and port values are printed as numbers.
- Hex in `tokens.css` (`--p0`…`--p9`) and `scripts/render_rasters.py`:
  `#3a385b #4c436a #5f4e79 #735986 #886492 #9c709c #af7ea4 #c28cab #d39cb3 #e5abbc`.
- Built in OKLCH. Lightness rises in equal steps from 0.36 to 0.80 (adjacent classes ≈ 1.2:1); hue turns
  dusk violet → mauve → rose at low chroma (≤ 0.085). Order survives greyscale and colour-vision
  deficiency. Revision 1's ramp ran to near-white and high chroma, so the common 60–80 % range painted
  whole bays bright pink and buried the coastline. Here that range is a mid tone, and the coastline,
  land and white port labels stay legible over it. Seven candidates were compared on the real data at
  statewide and harbour zoom before choosing.
- **The raster is opaque** at every zoom, so the legend swatches are exactly the colours on the map.
  Revision 1 faded the raster with zoom, which made the map disagree with its legend.
- **No value is hatched, never a colour.** Water without a model value (outside the model area, or
  nearshore cells the producer masks) shows diagonal hatching. Because the raster is opaque, the
  hatching appears only where there is no value, so a low probability can never be read as missing
  data, or the reverse. It has its own legend swatch and its own readout text.
- The domain is fixed at 0–100 % and never stretched to the data. Resampling stays **nearest**: a pixel
  is a model cell, never interpolated.
- The legend title is the manifest's `threshold_text` verbatim ("Probability that particulate domoic acid
  exceeds 500 ng per litre"), followed by the valid day and lead. The 500 ng/L and 10,000 cells/L
  thresholds are the producer's and are never restated or rounded.
- **Production change:** the pipeline palette `cw-probability-magenta-v2` would be replaced by
  `cw-probability-classes-v1` in `process/palette.py`. Only the PNG colouring changes; the u16 value
  grids, schema and port statistics are untouched. See the implementation plan.

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
12. **Display bands are not risk levels.** Forecast colour classes and heatmap bins are labelled as
    display intervals wherever they appear. Exact values remain available (map readout, port numbers,
    chart tooltips), and source thresholds are quoted verbatim.

## 7. Mobile interaction design

Designed at 390 × 844; the layout switches at ≤ 720 px.

- **Tab bar** (64 px): Map, Blooms, Fisheries, **Notices** (amber, count badge). Notices opens the
  drawer full-screen.
- **Map:** the map fills the screen between masthead and tab bar. A **place button** floats at the top
  ("All California ▾"); it opens the navigation (official row, port search, regions, ports) as a sheet.
  The **bottom sheet** peeks at about 250 px with, in order: the official row ("8 official notices in
  California · Not verified ›"), a native quantity select beside the day steps, and the legend (threshold
  sentence, ramp, "No model value", `Model` chip, freshness, "Display steps, not risk levels. Tap water
  for exact value."). Tapping water pins the exact-value readout above the sheet. Selecting a port
  raises the sheet to about 74 % and turns it into the inspector, official section first. The map
  re-frames using the sheet's measured height.
- **Bloom:** an official line under the masthead, then a **station picker** button (name, region,
  "17 stations") that opens the rail as a full-screen list. Then the freshness card, the measurement
  selector as one **swipeable row** of cards with previews, the edge-to-edge primary chart, the history
  heatmap collapsed, the model band (kept open: forecast and observation must stay visibly distinct),
  and accordions.
- **Fisheries:** narrative first, full-width segmented controls with short labels ("Tier 1", "Tiers
  1 + 2"), stacked figures, an edge-to-edge chart with two-digit years, a two-column table (value,
  share; average and sparkbars hidden), a detail panel shown only after selection, the one-line port
  note and the methods.
- Touch targets ≥ 44 px for primary controls; no hover-only information; no horizontal page scroll
  (checked by the screenshot script and the existing e2e overflow test).

## 8. Known gaps in the prototypes

The prototypes are static HTML/JS for evaluating the design, not production code. Not implemented: the
satellite layer switch, the full-screen mobile station list styling, the Sources page, the M2-data and
failed-update states (production already has them; the patterns in §6 apply), the error states, and full
keyboard support beyond what the HTML provides. Port search is a simple name filter.
