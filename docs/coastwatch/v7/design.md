# Neritic v7: design reset (the design owner's decisions)

Base: `feat/neritic-v6` @ `e17d30d` (draft PR #75). Branch `feat/neritic-design-reset`.

Inputs:
- the owner's four reference screenshots;
- four parallel audits (visual, fisherman workflow, scientific content, frontend);
- before captures at 1440 / 390 / 320.

This document is the single composition authority. Implementation tracks build to it. They do not invent their own layouts.

## The idea: one instrument per page, the document on demand

v6 pages read as documents with a map attached:
- small ALL-CAPS label rows carry the hierarchy;
- provenance is set at the same weight as decisions;
- amber is a disclaimer colour;
- the same warning appears 3–6 times on one screen.

v7 makes each page **one instrument**, with a single dominant surface:

| Page | Dominant surface | One primary action | Essential context | On demand |
|---|---|---|---|---|
| Explore | the map, full bleed | pick a place (search or tap) | the layer, one legend, the day | layer list, layer science, source history, full credits, evidence |
| My Coast (first visit) | one place search | choose a place | five starting places | every place via search |
| My Coast (briefing) | the trip: conditions + trip economics | adjust the trip numbers | official position for this species, what changed, what is unknown | full sea forecast, seasons, checks, evidence, notices, map |
| Research | the station's primary chart | choose a station / variable | the station's position on the coast, value + model same day | provenance & QC, lineage, grid tools, exports |

## Shell (all pages)

- **Header**: brand, workspaces, (Explore: search), then a right cluster:
  - **Notices** control: shield + count + "Not verified".
  - **Status**: an icon only when everything is ready; words only when something is not.
  - **More** menu.
- **One disclosure line** (`registry-disclosure`): `Research preview · Notices not verified, may be incomplete · CDFW · CDPH`.
  - Neutral ink, 12.5–13 px, a single small official tick before "Notices not verified". No amber sentence.
  - Chart pages: directly under the header, aligned to the content column.
  - Explore: in the map's bottom margin with the map credits (desktop), or one line under the header (phone).
- Paragraph-level restatements of the same obligation are removed:
  - the briefing provenance paragraph is replaced by one quiet line in the briefing footer, keeping the test id;
  - the glance "· not verified" suffix goes; the "Not verified" tag on each notice record stays;
  - the econ notices line goes;
  - Explore's purpose toast and floating preview capsule go.
- The My Coast tab opens the saved home trip (place + activity + species). A bare `/coast` with a home port opens its briefing.

## Explore: immersive map instrument

```
1440 ────────────────────────────────────────────────────────────────────────────
[◭ neritic] [Explore My Coast Research] [⌕ Search a harbour, station, species  /]  [⛨ 10 Not verified] [●] [⋯]
                                                                         ┌ inspect (on selection) ┐
                       FULL-BLEED MAP                                    │ Monterey               │
                                                                         │ Monterey County        │
                                                                         │ [Open the briefing  →] │
                                                                    [+]  │ 3 short facts          │
                                                                    [−]  │ Evidence and sources ▸ │
                                                                    [◎]  └────────────────────────┘
 ┌ instrument dock ───────────────────────────────────────────────┐ [⋯]
 │ [▦ NOAA bloom forecast · Domoic acid ▾] │ ramp 0–100 % · units │ ◂ Mon 19 [Tue 20] Wed 21 Thu 22 ▸ │
 │  one-line coverage / what-it-is-not                    ⓘ       │  run Oct 20 (inferred)             │
 └────────────────────────────────────────────────────────────────┘
 Research preview · Notices not verified, may be incomplete · CDFW · CDPH     © OSM · OpenMapTiles · NOAA relief · not for navigation ⓘ
```

- Default state has three things over the map: the header row, the instrument dock and the tool stack. Nothing else floats.
- **Instrument dock** (`layer-capsule` testid on the layer button; the dock is `timebar-deck` when time exists):
  - left: the active layer button. It opens the layer popover (`layer-deck`) upward: the layer list, the variable and options, then "About this layer" (provenance, reference lines, full credits) folded.
  - middle: the legend, once: ramp + units + date/source + a one-line coverage/"what it is not" statement. The longer caveat stays in the DOM behind ⓘ.
  - right: the day steps (Timebar compact) when the layer has time.
- **Inspect** (`inspect-panel`): right side, 380 px, top-aligned, max height to the dock line. Contents:
  - the title and the one primary action (open briefing / research);
  - at most three short facts;
  - everything else in "Evidence and sources", mounted and folded.
- Tools: zoom ±, locate, area, plus share and field pack in one stack at bottom-right. With a selection, the stack sits left of the panel.
- Phone: header (mark, search, notices, more) plus the one-line disclosure; the map; a dark sheet whose peek is the dock (layer button + legend line + days, about 132 px); the navigation dock. A selection lifts the sheet to half height.
- Identity: the isobaths stay on the basemap. The dock's top edge is a single contour hairline (SVG), the one decorative line per page.

## My Coast: operations briefing

First visit (no place, no home):
```
              (faint contour field behind, static SVG)
     Where are you fishing?                          ← serif display
     [⌕  Harbour, beach or town                    ] ← the one control (coast-place-search)
     Start with   Monterey · Santa Cruz · Half Moon Bay · Eureka · Crescent City · San Diego   (text links; Monterey keeps coast-start-port-550)
     One line: what a briefing holds.
```
- No species chips: species and activity are chosen in the briefing's trip bar.
- No directory: every place is reachable from the search.

Briefing (1440):
```
[header] / disclosure line
Monterey ▾        [Commercial ▾] [Rock crab ▾] [Today, Oct 20 ▾]   ⚓ Home port   ⋯
──────────────────────────────────────────────────────┬─────────────────────────────────
 Since you last checked: nothing new · Oct 19          │ TRIP ECONOMICS  (open, primary)
 WIND        SEAS        WAVES        WATER            │ Scenario from your inputs, not a forecast.
 W 5–10 kt   3–5 ft      5.9 ft·13 s  67 °F             │  Net to boat   Cost    Break-even
 NWS tonight · your limits: not set ›   measured …     │  —            $412    — lb
 ─────────                                             │ [Vessel ▾] Fuel $/gal [−][4.85][+]
 Rock crab is on CDFW's sport list only.                │ Catch lb [−][   ][+]  Price $/lb [−][  ][+]
 Commercial: CDFW commercial ↗ · in-season page ↗       │ Full worksheet ▸ (crew pay A/B/C, costs, fees, scenarios, share)
 3 notices tied to these waters, none name rock crab › │
 ─────────                                             │
 What matters now (≤ 3 statements)                     │
 5 things not known ›                                  │
──────────────────────────────────────────────────────┴─────────────────────────────────
 Folded notes: Sea forecast and tides · Seasons · Before you go · Evidence · Notices · Map · Sources
```
- The duplicate rail map is removed. The map is one "On the map" note: the locator opens on demand, and Explore is linked from it.
- Trip economics is the right column on desktop: sticky and open. The top of the worksheet is the quick block:
  - the live result line (aria-live);
  - vessel, fuel price, catch, price.
  The full worksheet follows in the same panel, with its advanced fieldsets folded.
- On a phone it comes right after the official position, with a sticky result bar.
- Conditions are four **soundings**: label, large tabular value, one quiet source line.
- The labelled-row "At a glance" table is retired.

## Research: calm scientific workspace

```
[header] / disclosure line
Santa Cruz Wharf ▾   36.958 N 122.017 W · sampled 2014–Sep 30            [Export ▾] [Open in Explore]
┌ coast ladder ─┐┌ primary chart ───────────────────────────────────────────────────────────┐
│ stations at   ││ [pDA | Pseudo-nitzschia | Chlorophyll | Temp]   90 d · 1 y · 3 y · All   │
│ true latitude ││ value 0.20 ng/mL Sep 30 · NOAA model same day 70 %                        │
│ ● selected    ││ chart, model chart on the same axis                                       │
└───────────────┘└──────────────────────────────────────────────────────────────────────────┘
 Tabs: In this window · Provenance & QC · Native grid · Lineage · Exports
```
- The station list becomes a **coast ladder**: each station sits at its true latitude along a stylised coastline. It works as a map and a picker in one, without a second WebGL map.
- Provenance, QC and lineage move to tabs. They stay mounted, so their safety wording is still in the DOM.

## Component language (eight primitives, `src/v4/ui/Kit.tsx`)

1. **Field**: a full-bleed map or chart.
2. **Rail**: a side column or sheet.
3. **Sounding**: label + tabular value + unit + one-line source.
4. **Statement**: a family tick (official / model / measured / conditions / unknown) + one sentence + an optional disclosure.
5. **Dock**: legend + time.
6. **TripBar**: place + activity + species + day.
7. **Note**: a folded disclosure row (title, count, one-line summary).
8. **Control**: one primary filled button, quiet text buttons, a segmented control and a stepper.

Banned: outlined pill navigation, ALL-CAPS eyebrows as hierarchy, coloured status words, nested cards, amber sentences.

## Type, space, colour

- Type:
  - Atkinson Hyperlegible Next for UI and reading;
  - Source Serif 4 for one display moment per page;
  - Plex Mono only for tables and coordinates;
  - numbers in text use `tabular-nums` Atkinson.
  - IBM Plex Sans and Newsreader are unused and no longer loaded.
- Scale: display 40/44 (phone 30/34), title 21/28, sounding 26/30, body 16/24, label 13/16 (sentence case), note 12.5/17.
- Space: 4 px base (8 / 12 / 16 / 24 / 32 / 48). Hairlines separate bands; a box is reserved for the one interactive instrument per page.
- Colour:
  - navy field and cyan for selection/focus/primary;
  - data-family colours only as a 3 px tick or 8 px glyph;
  - amber text only for a restriction that names the reader's species and waters, or a disputed notice.

## Motion

- Content first. No opening animation delays ready content.
- Panels 240 ms ease-out with 8–12 px travel; folds by height + fade, 180 ms; reduced motion means fades only.
- Opening effect: Explore only (moved out of the root layout). The Monterey Canyon isobaths draw and resolve into the mark *inside the map area while it loads*. The header is interactive throughout.
  - It exits when the map is ready, capped at 1.2 s, with no minimum hold.
  - It is skippable by any input; reduced motion shows a still frame.
  - It is CSS + SVG only (no particle renderer, no dependency).
- OriginKit (proprietary licence, the relevant component is paid, WebGL) and Casberry (no licence, three.js + MediaPipe) were reviewed and not used.

## What must not change

- Every layer and its coverage semantics: observed / estimated / extrapolated / outlook / forecast words.
- NOAA model boundaries, source metadata and timestamps.
- Not-verified notices and "a missing notice is not open".
- Trip economics:
  - all three crew-pay models;
  - scenarios, compare and the share link;
  - offline calculation;
  - nothing pre-filled for catch or price.
- The saved home port and species.
- Historical price caveats.
- Research tools and exports.
- Forecast validity and data gaps.
- Accessibility and offline behaviour.

Safety wording stays in `src/content/copy.ts`. Folded content stays mounted.
