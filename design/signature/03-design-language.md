# 03 — CoastWatch design language

Live specimen: `#/system`. Source of truth: [`prototypes/src/styles/tokens.css`](prototypes/src/styles/tokens.css). It uses the same `--cw-*` names as `coastwatch-web/src/app/globals.css`, and every new token is additive.

## The organising idea: four marks

CoastWatch exists to show the difference between what is **official**, **measured**, **modelled** and **not known**. The identity is built on that difference. Each kind of knowing has its own *graphic mark*, and the same mark appears on the map, in sentences and in charts.

| Kind | Mark | On the map | In text | In charts | Colour (paper / sea) |
|---|---|---|---|---|---|
| Official | **Line** (rule) | Latitude limits, area outlines; solid, labelled with the latitude and agency | A 14 px rule before the agency name | A vertical rule at the effective date | `#8f4700` ink, `#d9922e` line / `#f2b04c` |
| Measured (in situ) | **Point** that carries its date | Filled when sampled in the last 14 days, a ring when older | A dot | Dots; lines only between samples ≤ 21 days apart | `#0c6f62` / `#4fd1ae` |
| Measured (satellite) | **Pixel** | Native-resolution pixels, nearest-neighbour, never smoothed | A small square | n/a | as measured |
| Modelled | **Field** of cells | Exact 3 km cells; veil, lens or class lines | A 2×2 cell glyph | A band on its own axis, "model probability, not a measurement" | `#6a3fb0` / `#b58cf0` |
| Not known | **Hatching** | Diagonal grey hatch, only where a field exists and has no value | A hatched square | Hatched intervals with no samples | `#606b78` / `rgba(176,196,220,.42)` |
| Context (conditions) | Small diamond | Not drawn as a field | Accent ink | n/a | `#0a6680` / `#74cfe6` |

**Why marks, not just colours.**
- Marks survive colour-vision deficiency and greyscale printing.
- They stop a model being read as an observation: a *point* can't be mistaken for a *field*.
- They give CoastWatch a recognisable, purpose-derived look without decoration.

## Colour

### Rules

- **No safety colours.** No red, green or amber implies seafood safety, and there is no "all clear" colour. Amber means *an agency said this*. Green-teal means *someone measured this*. Violet means *a model computed this*.
- **Freshness has no colour of its own.**
  - "Current" is ink.
  - "Stale" and "Historical" recede to grey.
  - "Not confirmed" uses official ink, because it is a fact about the *source*.
- **Text colours must pass AA.** Text colours pass 4.5:1 on the surface they sit on (measured live in `#/system`). Map line colours pass 3:1 against the sea.

### Marine basemap

| Role | Value | Note |
|---|---|---|
| Sea (neutral) | `#0b1d33` | Under every data layer; darker than every data colour, so it is never read as a value |
| Depth tint | `#2c6c8a` (shelf) → `#0a1a2e` (abyss) | ETOPO/CRM tint, **only with no data field on**. It cross-fades (420 ms) to the neutral relief when a field is shown; it is never toggled |
| Land | `#1b2531` + hillshade at 85 %, −35 % saturation | One step darker and less blue than before, so the sea carries the colour |
| Coastline | `#d6e2ee`, 0.5 → 2 px with zoom | The most important reference; always above data |
| Isobaths | 200 m at 34 % opacity, others at 15 % | Drawn *above* data fields, so depth stays readable over the forecast |
| Labels | Ports white (only white labels); cities `#9fb0c2`; towns `#8394a8`; water italic, letter-spaced | Ports are where people start |

### Data

- **Forecast classes:** the published palette `cw-probability-classes-v1` is kept unchanged, so the data contract and the screen agree.
- **Chlorophyll:** the published `cw-chlorophyll-log-v1`, log scale.

#### Forecast compositing (new)

Each mode changes only how loudly the field is drawn. No mode changes a value, a class or a cell boundary.

| Mode | How | When |
|---|---|---|
| **Veil** (default) | Class colour pre-mixed with the sea at 60 %. The cell is opaque, so the no-value hatch underneath can only show in cells without a value | Whole-region reading |
| **Lens** | Cells whose centre lies within 16 km of the chosen point at 100 %, others at 20 %. The edge follows cell edges, as a staircase | "What does the model say *here*?" Default proposal for Coastal mode with a selected point |
| **Class lines** | Faint 24 % field plus lines along exact cell edges where the 10 % class changes ("isobaths of probability") | Research; calm overview |

**Rejected:** alpha proportional to probability. It makes a low probability look like *no data*, or like *safe*.

### Interaction

| Role | Value |
|---|---|
| Selected water point | White ring and dot with a navy halo; never a data colour |
| Focus | `0 0 0 2px page, 0 0 0 4px #1a8fb3` (dark: `#74cfe6`) |
| Selected control | Ink fill, paper text (dark surfaces: inverted) |
| Hover | Surface step (+1) or a 6 % ink wash; never a colour change of data |
| Unavailable | Grey text and the gap mark; the control stays focusable with an explanation |

### Surfaces

- **Navy frame.** `#0b1a2b` masthead with the status line inside it: one band, 3 px amber left rule.
- **Paper.** `#f4f2ec` page. Sheets are `#fbfaf7`, not stark white, so cards don't glare against the paper.
- **Dark map chrome.** `rgba(11,26,43,.88)` with a 10 px blur. Only floating map controls use it; reading happens on paper.

### Charts

- Samples use the measured colour.
- Notices are official rules.
- The model is violet on a separate right axis.
- Gaps are hatched.
- Grid lines are hairline (`--cw-hairline`).
- Axis text is `t-micro`, 12 px.

## Typography

**Audit of the current use:**
- Newsreader is loaded at 400/500 static only.
- IBM Plex Mono is used for IDs, coordinates *and* some values.
- Uppercase tracked eyebrows are the most frequent style.
- Several meta lines are 11–12 px in amber on cream.

**System:** two families with strict roles.

| Role | Style | Notes |
|---|---|---|
| Display (page headline that states an answer) | Newsreader 600, opsz 72, clamp 32–44 px / 1.06 | Variable font with optical sizing: sharper at display sizes than the static 500 now used |
| Title (place name) | Newsreader 500, opsz 48, 24–30 px | "Off Santa Cruz", "Monterey" |
| Headline (card answer) | Newsreader 500, opsz 24, 20 px | Only for answers, never for absences (audit P6) |
| Lead | Plex Sans 400, 17/26 | The Level 1 sentence |
| Body | Plex Sans 400, 15/23 | |
| UI | Plex Sans 500, 14 | Buttons, tabs, chips |
| Label | Plex Sans 500, 13 | Agency names, kind labels |
| Meta | Plex Sans 400, 13 | Dates, sources, distances |
| Micro | Plex Sans 500, 12 | Chart axes, attributions, map labels only. Floor for any text |
| Value | Plex Sans 500, 28 (17 small), tabular lining numerals | Numbers are never mono |
| Unit | Meta, ink-3, 0.25 em gap | "0.20 ng/mL" |
| Coordinate / ID | Plex Mono 400, 12 | Only coordinates, station IDs, layer IDs |
| Map labels | Noto Sans (self-hosted glyphs) | Existing; unchanged |

**Rules:**
- Sentence case everywhere.
- Kind labels are words led by a mark, never tracked capitals.
- Dates are "Mon, Oct 12", with relative words only within a week.
- No text below 12 px.
- Disclaimers that change meaning are body text, not micro text.

## Layout

| Pattern | Standard |
|---|---|
| Page header | Kind label → Title (place) → one coordinate/meta line → Level 1 sentence. 24 px padding, 14 px gaps |
| Primary content on desktop map pages | Full-bleed map; one floating paper panel, 420 px (360 at ≤ 1100 px, 440 in Research), 16 px inset, 18 px radius |
| Map controls | Two zones only. **Top:** search, then layers, with Centre at the right. **Bottom:** time, with field mode beside it. Legend top-right. Zoom by gesture/keyboard; buttons optional |
| Information sheet | The ledger: one row per kind of knowing, fixed order, hairline separated, expand in place (no cards inside cards) |
| Notices | Agency · action → title → since/source → limit sentence with a rule glyph. Dashed hairline between notices |
| Charts | Full content width; controls above as segmented radiogroups; readout below; "Show as table" toggle |
| Tables | Hairline rows, tabular numerals, scope headers, no zebra |
| Details drawers | `grid-template-rows: 0fr → 1fr` (works in Safari); chevron rotates; content never animates height by measuring |
| Responsive grid | ≥ 1101 px panel 420; 761–1100 panel 360; ≤ 760 sheet. Reading pages max 1120 px, 16 px gutters on phones |
| Mobile bottom sheet | Non-modal. Peek (answer + one row) / half (55 %) / full; above a 5-item tab bar; controls ride above the sheet at peek and hide otherwise |
| Full-screen exploration | Masthead plus map only. The sheet can collapse to peek and the map stays interactive above it |

## Motion tokens

| Token | Value | Use |
|---|---|---|
| `--cw-d-micro` | 110 ms, standard | Press (scale 0.97), hover, toggles |
| `--cw-d-short` | 180 ms, standard | Tooltips, chips, small reveals |
| `--cw-d-medium` | 280 ms, settle | Sheets, selection marker, disclosure |
| `--cw-d-long` | 460 ms, standard | Chart window and quantity transitions |
| `--cw-d-swap` | 100 ms out + 140 ms in (delay 100) | Data layer swaps: **dip through the basemap**, never blending two datasets |
| `--cw-d-thread` | 520 ms + 90 ms stagger | Provenance threads, once per selection |
| Camera: journey | 900–2600 ms (distance-scaled), along the coast; < 40 km ease; > 450 km fly | Port and place changes; interrupted by any input |
| `--cw-ease-standard` | `cubic-bezier(.2,0,0,1)` | Entering and moving |
| `--cw-ease-exit` | `cubic-bezier(.4,0,1,1)` | Leaving |
| `--cw-ease-settle` | `linear()` spring, ≈ 3 % overshoot | Sheets and the selection ring only. Never data marks |
| Reduced motion | All spatial motion → 120 ms opacity; threads instant; camera jumps; progress bar static | |

Nothing loops continuously except the indeterminate progress line, which appears only after 400 ms of real waiting and is static under reduced motion.

## Copy: representative rewrites

The rule: lead with the place-specific fact, keep the one caveat that changes the meaning beside it, and move method to Level 3.

| Today (redesign or `main`) | Proposed | Why |
|---|---|---|
| OFFICIAL NOTICES · THESE WATERS / 2 notices apply here for anchovy or sardine / Not verified | — Official<br>**2 notices cover these waters for northern anchovy**<br>From CDFW and CDPH · not verified by CoastWatch | Sentence case; "cover" is the geographic claim the data supports; verification stays on the same line |
| No notice in CoastWatch's list (24 px serif) | **No notice's official limits include these waters**<br>That is not the same as safe or open | States the limit of the claim, not an absence; never set larger than a finding |
| Current · issued 1 day ago … Last update failed … showing the last successful run | C-HARM · for Mon, Oct 12 · **Not confirmed**, issued Oct 11 | One line, both dates, one settled freshness word |
| Probability that particulate domoic acid exceeds 500 ng per litre · 10-point colour steps for display, not risk levels | **60–70 % chance** in this 3 km cell<br>(open:) Particulate domoic acid above 500 ng/L. A forecast of the water, not a test of seafood. A low chance does not mean safe. | The answer first; the definition one level down |
| Hatched or empty water has no value. C-HARM has no toxin values for many cells within about 3–6 km of shore… | **No forecast for this nearshore cell**<br>(open:) C-HARM leaves out many cells within about 3–6 km of shore, and bays. Nearest cell with a value: 70–80 % chance, 2.9 km away — it describes that cell, not this one. | Specific to the point, spatial, with the nearest real value kept in its place |
| Toxin (domoic acid) in seawater 0.052 ng/mL, over 4 years ago | **Monterey Wharf, 4.1 km southeast**<br>Last sampled Oct 7 · toxin last measured Aug 31, 2022 | Separates "visited" from "measured this" |
| (nothing) | **What isn't known here:** Toxin in seafood — only CDPH testing tells you that, and CoastWatch does not have those results | The most important unknown is always stated |
