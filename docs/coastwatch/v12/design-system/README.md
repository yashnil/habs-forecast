# Neritic Chart: the design system

Owner: the lead creative director. Every screen, component and agent follows this document. Where code and
this document disagree, this document wins until it is changed on purpose.

Direction chosen by the owner on 2026-11-10: **Concept A, Chart**, with two grafts (the dated agenda and mark
system from Tide; the lit canyon as a one-time opening from Canyon). Evidence behind every rule is in
`../research/` (A fishing, B visual, C IA, D motion, E science rules T1–T47) and `../concepts/`.

## 1. What Neritic is, for the person using it

> For your port and fishery: is the season open, what do the tests say, what are the whale rules, and when is the next decision. Next to what the water is actually doing. On one chart.

Neritic doesn't predict fish, give go/no-go calls, or replace CDFW, CDPH or NWS. It saves the eight-page round
trip and shows the evidence. Every screen answers: **what am I looking at, why does it matter to me, what can I
do next**, through composition plus one sentence.

## 2. Principles

1. **The chart is the product.** One sheet, one projection, one camera, persisting across pages. Pages are views of that sheet plus a margin. We never put the map behind a panel as wallpaper.
2. **Readings live where they were measured.** A buoy figure sits at the buoy, a test figure at its test area. If a value has no place, it goes in the margin, never floats.
3. **Letter style carries class.** As on a nautical chart: italic serif names water, upright sans names land and interface, the serif at size makes statements. Colour is not the first tool.
4. **One foreign hue.** Magenta means regulation and nothing else. Everything else is ink, paper and the depth tints. Data ramps live only on the water.
5. **Every number has one home and carries its kind, source and date.** A number shown twice on a screen is a bug.
6. **Truth beats drama.** T1–T47 are design rules. Motion never depicts what wasn't measured.
7. **Quiet chrome, rich water.** Effort and beauty go into cartography, type and motion of real things, not into containers.

## 3. Anti-vibe-coding rules (enforced by `npm run lint:design`)

These are the tells the owner rejected. Each has an automated check where possible; the rest are review items.

| Never | Check |
|---|---|
| Colour literals outside `src/nx/ds/tokens.css` and `src/nx/ds/palette.ts` | lint: hex/rgb/hsl in `src/nx/**` except token files |
| Text under 14 px; body under 16 px | lint: `font-size` values; component props |
| `box-shadow`, `backdrop-filter`, gradients as decoration | lint |
| Radius other than `--r-0` (0) or `--r-1` (2 px); pills | lint |
| `text-transform: uppercase`, letter-spaced caps labels, mono caps | lint |
| Emoji, decorative icons, icon-plus-tint pills | lint for emoji; review |
| Forbidden words (T36): safe, all clear, go, no-go, good day, hotspot, best spot, recommended, expected catch, low risk… | lint over `src/nx/**` copy and `src/content` |
| KPI tile rows, card grids, identical section templates, row-plus-chevron lists | review |
| Stagger-on-everything, text scramble, count-up on arrival, ambient glows | review; motion only through `ds/motion` presets (lint: raw `transition:` longer than 200 ms or keyframes outside `ds/`) |
| "X, not Y" disclaimers repeated per figure | one legend per view states kinds; review |
| Filler copy: taglines inside the product, encouragement, "Was this useful?" × n | review |
| Hard-coded dates, values or statuses in components | lint: ISO dates in `src/nx/**` outside tests |

Review gate: the independent visual critic may reject any screen regardless of tests.

## 4. Colour

Three chart modes, each separately specified like S-52's day, dusk and night tables, never an automatic
inversion of another: **Day** (default, sunlight-first), **Dusk** (evening wheelhouse, reduced luminance),
**Night** (dark-adapted, very low emission). Mode follows a toggle and `prefers-color-scheme`; Night is opt-in
only.

**Which end of a data ramp is "more" (decided 2026-11-11).** On Day's paper, high is dark. On the dark grounds of
Dusk and Night, high is **light**: every data ramp (`--chl`, `--prob`, `--sst`) runs dark to light, lightness
still monotone, the way the depth tints there run shoal-light to deep-dark, as S-52's night tables do. High-is-dark
on a near-black sea would sink the highest values into the water. So switching to Dusk at sunset turns the bay's
highest chlorophyll from darkest to lightest, on purpose, and every ramp legend says which end is more in words
("Darker is more." on Day, "Lighter is more (dark chart)." on Dusk and Night), driven by the mode's own tokens
(`--ramp-says-darker` / `--ramp-says-lighter`).

Day tokens (contrast on paper `#FFFEFA`):

| Token | Day | Role |
|---|---|---|
| `--paper` | `#FFFEFA` | Sheet and margin ground |
| `--ink` | `#0E1417` | Text, interaction, rules (18.4:1) |
| `--ink-2` | `#2F4A5B` | Secondary text (9.2:1). There is no third text level |
| `--water-name` | `#1F4C74` | Italic water names only |
| `--reg` / `--reg-text` | `#A83BA6` / `#7B2279` | Regulation: zone lines, rails, delay/closure statements. The only hue in chrome |
| `--closure` | `#8A3307` | Official closure statements only |
| `--depth-0…8` | `#7CB5F0` → `#EBF6F2` | Depth tints, shoal saturated to deep palest |
| `--land` | `#DFD8AE` | Land, with 1× relief |
| `--rule` / `--rule-soft` | ink / `rgba(14,20,23,.26)` | 1.5 px heads and tallies / 1 px hairlines |
| `--focus` | ink | 2 px outline, 2 px offset |
| `--chl-0…7` | `#F6F4DC` → `#052327` | Chlorophyll, log10 0.05–50 mg/m³, high is dark on Day's paper (light on Dusk and Night, see above), lightness monotone |
| `--prob-0…9` | `#F4F2FB` → `#34315D` | C-HARM probability in cw-probability-classes-v1's ten 10-point display intervals (not risk tiers, T12): a quiet slate-violet clear of `--reg`, drawn at 0.72 under a model hatch; L* 95.9 → 22.7, min ΔL* 8.0 |
| `--sst-0…7` | `#F8F2DC` → `#853E19` | Sea temperature, sequential cream to rust (its own hue family, T22); L* 95.4 → 35.0, min ΔL* 8.4. Explore draws temperature as isotherms; the ramp is there for a filled field or a legend that needs one |

Dusk and Night tables are defined in `tokens.css` with their own computed contrast; Night keeps text ≥ 4.5:1 at
its low luminance and drops depth tints to three steps.

Semantic marks (shape first, colour second): **measured** solid; **predicted / model** hollow or hatched; **official, not verified** ring with the words; **no data** ink hatch labelled "no view"; **stale** the word "(old)" next to the value. These five marks and one legend per view replace repeated caveats.

## 5. Type

| Role | Face | Sizes (desktop / phone) | Notes |
|---|---|---|---|
| Statement | Literata (opsz) 500 | 64/40, 48/32, 36/28 | Page titles, status, the big date |
| Water names | Literata italic 400 | 15–22, letter-spaced 0.04em on the chart | Water features only |
| Interface and land | Public Sans 400/500/600/700 | 16 body, 18 lead, 22 heads | All UI |
| Sounding | Public Sans 600 tabular | 48/36 figure, unit at 0.46em | Figures on the chart |
| Small | Public Sans 400 | 14 | Provenance, ticks. Floor for anything read |

Both OFL, self-hosted with `next/font`. No monospace in the product. Sentence case everywhere. Tabular numerals
in columns and soundings. Line length 45–70 characters. Text on the chart is set on a 3 px knockout halo.

## 6. Space, grid and structure

- Space scale 4, 8, 12, 16, 24, 32, 48, 64 (`--s-*`). Nothing else.
- **The sheet**: full-bleed chart inside a quiet neatline (a 4 px graduated band with real latitude and longitude ticks, 3 px on phone).
- **The margin column**: 340–372 px on desktop, ruled, no container. It holds the page's statement, notes and actions. On phone it becomes a sheet that rests at a meaningful detent (status visible) and drags up.
- **Shell**: one 52 px bar (wordmark, port, four destinations, Ask, Alerts). Phone: top bar with port; four destinations in a bottom bar, 56 px, 48 px targets.
- Radius 0 everywhere; 2 px only on inputs. No shadows: layering by knockout halos, rules and paper.

## 7. Iconography

A small custom set drawn on a 24 grid at 1.5 px stroke, from chart symbology: port (circled dot), buoy, test
area, tide station, radar coverage, zone line, search, layers, time, close, chevron, share, offline, alert. Ink
only. No icon without a word unless it is a universal control (close, zoom) with an aria-label. No icon library.

## 8. Motion (Motion for React, `motion@12.43`)

Motion says where something came from and never holds up a click. All motion goes through `src/nx/ds/motion.ts`
presets; components don't invent timings.

| Preset | Use | Value |
|---|---|---|
| `ink` | Lines drawing (isobaths, tide curve, series) | 0.9–2.1 s, `[0.65,0,0.35,1]`, path length |
| `pan` | The sheet's camera between pages | 0.62 s, same ease; map camera via `flyTo`/`easeTo` with Motion-driven progress |
| `settle` | Figures and bars changing (ledger, readouts) | spring, stiffness 260, damping 30 |
| `register` | A sounding appearing at its place | 0.24 s fade + 4 px scale from 0.96 |
| `sheet` | Phone margin sheet, readouts | spring, stiffness 380, damping 36; drag with velocity |
| `press` | Every control | scale 0.97, 0.12 s |
| `cut` | Time steps between data frames | none: a new frame is a cut (T26) |
| `layout` | Shared elements: port name from picker into Today's title; a margin row into its detail | Motion `layoutId` within a page |

Route changes: the persistent chart pans; the margin cross-fades (React `ViewTransition` on the margin only; the
map is excluded from the snapshot). Reduced motion: final states, no pans, static streamlines. GPU tiers
from `src/lib/capability`: full (WebGL map, opener, streamlines), reduced (map, no opener, 30 fps lines),
static (baked chart image of the same view, same notes).

## 9. Components (`src/nx/ds/`)

Each has states: default, hover, focus, pressed, disabled, loading, empty, stale, offline, error.

| Component | Purpose |
|---|---|
| `Sheet` (persistent chart host) | One MapLibre instance across routes; pages declare a `ChartView` (bounds, layers, notes) |
| `Neatline` | Graduated frame with real ticks |
| `Sounding` | A figure placed at a lon/lat, with unit, kind mark and a leader line; collision-aware |
| `ChartNote` | A short note on the water with knockout halo; drops by priority when space is short |
| `Margin` / `MarginRow` / `MarginSheet` | Ruled column; rows expand in place; phone bottom sheet with detents (Motion drag, `sheet` spring) |
| `Statement` | Page title plus status sentence, Literata |
| `BigDate` | The next date that matters, with what it is and its source |
| `Agenda` | Dated items: past solid, future hollow, now line (from Tide) |
| `Mark` + `Legend` | The five semantic marks; one legend per view |
| `Stamp` | Kind · source · date · verification, the provenance line, opens `ProvenanceSheet` |
| `TimeRail` | Discrete stamped frames per product with availability bars (GFW-style) |
| `RampLegend` | Legend-as-title for the active layer, log ticks, "no view" hatch |
| `Readout` | Tap result: value, kind, date, linked transect chart |
| `SeriesChart` / `SmallMultiples` / `Brush` | d3-scale/d3-shape SVG, crosshair readout, censored marks (T7) |
| `Worksheet` | Ruled ledger: inputs, live waterfall (gross and net solid ink, costs a lighter solid `--ink-2`, never hatch or hollow), sensitivity table with both axes named (catch down, price across), assumptions list |
| `Stepper` / `Field` / `Segmented` / `Toggle` | Inputs with units, 48 px targets |
| `Button` (primary ink, secondary ruled, text link) | One primary per view |
| `PortFinder` | Ranked search (port name prefix, then word prefix, then substring; county and code as keywords) in one flat list while typing, Enter opens the best match, no match said in words; north-to-south directory and recent ports when empty |
| `Popover` / `Dialog` / `Tooltip` | Radix primitives styled here |
| States: `Unavailable`, `Stale`, `Offline`, `SourcesDiffer`, `Loading` | Designed, worded, never a spinner alone |

## 10. Words

Plain, specific, active. One word per thing: **port** (not harbour/place), **season status**, **test** (a CDPH
sample result), **whale measures** (RAMP), **forecast** (NWS), **model** (C-HARM, outlook), **measured**.
Dates absolute ("Dec 15–18"), times Pacific with "PT" where ambiguous. Units fishermen use: kt, ft, °F with °C
on request, fathoms for depth limits.

## 11. Libraries (all permissively licensed, pinned)

| Package | Version | Licence | Why |
|---|---|---|---|
| `motion` | 12.43.0 (installed) | MIT | All UI motion; React `layoutId`, springs, scroll |
| `maplibre-gl` | 6.x (installed) | BSD-3 | The persistent chart |
| `@paper-design/shaders-react` | 0.0.82 | Apache-2.0 | `paper-texture` for the sheet's paper and `image-dithering` for the opening's canyon still; never on water at runtime |
| `@radix-ui/react-dialog`, `-popover`, `-tooltip` | 1.3.1 / 1.3.1 / 1.4.1 | MIT | Accessible primitives |
| `cmdk` | 1.1.1 | MIT | Port finder |
| `d3-scale`, `d3-shape` | 4.0.x, 3.2.x | ISC | Chart maths; we render SVG ourselves |
| Literata, Public Sans | via `next/font` | OFL-1.1 | Type |

Not used, on purpose: vaul (1.1.2 always opens a modal Radix dialog: it hides the page from assistive tech and traps focus, which a page margin must never do; the sheet is Motion's drag with the `sheet` spring), three.js and deck.gl (one scene doesn't justify the weight), visx, icon libraries,
Tailwind component kits, runtime fluid shaders.

## 12. Budgets

First-load JS per route at or below v11's; no second GL stack; LCP ≤ 2.5 s desktop, ≤ 4 s throttled phone for
the margin text; opening ≤ 5 s, skippable, once per visitor, statement legible at frame one; offline field pack
keeps every page readable with the baked chart.
