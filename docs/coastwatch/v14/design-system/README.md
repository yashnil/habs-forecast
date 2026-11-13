# Neritic v14 design system: the synthesis

v14 implements the owner-approved **synthesis** direction (prototype `art/concept-4-synthesis`, its `SYSTEM.md` and
`AUDIT.md`). The chart is the page on every screen. Each screen has one object that sits beside its place and says one
thing, one filled button, and caveats drawn as marks. Code: `coastwatch-web/src/nx/ds` (tokens, components, motion),
`src/nx/shell` (the bar, the workspace), `src/nx/chart` (the persistent chart). Import components from `@/nx/ds`.

v12 and v13 parts are still exported and take the new tokens, so pages can move one at a time (§7). Where the
prototype is silent (Dusk and Night, Explore's data layers, legacy pages), v14 extends its language as little as
possible. Each such place says so below.

## 1. Tokens (`src/nx/ds/tokens.css`)

| Role | Day value | Token |
|---|---|---|
| Objects on the chart (tickets, docks, bars) | `#FFFFFF` | `--paper` |
| A field, a quiet surface, a hover | `#F1F4F6` | `--paper-2`, `--wash` |
| Text, measured data / secondary / hairlines only | `#0C2233` / `#47616F` / `#8497A2` | `--ink`, `--ink-2`, `--ink-3` |
| Anything you can press or drag | `#2B3BD6` (deep `#1D2AA8`, press `#18238F`, wash `#E8EBFF`) | `--ultra`, `--ultra-deep`, `--ultra-press`, `--ultra-wash` |
| Regulation: lines / text on paper / small text on water | `#F0502A` / `#BF360F` / `#6E1903` | `--coral` (`--reg`), `--coral-ink` (`--reg-text`), `--coral-map` |
| The lit sea, under 10 m → over 2,000 m (shelf edge between 4 and 5) | `#E9F7F2 … #4C86B3` | `--depth-0` … `--depth-8` |
| Land / coast | `#EFF1E6` / `#234E62` | `--land`, `--coast` |
| Trip columns: gross / costs / crew / net hatch | `#1E4459` / `#CBD5DC` / `#BFC7F6` / ultramarine hatch | `--gross`, `--cost`, `--crew`, `--hatch-net` |
| Elevation | the prototype's two shadows | `--shadow`, `--shadow-sm`, `--shadow-ticket`, `--shadow-pass` |
| Hatches (hypothetical, illustrative) | | `--hatch-ultra`, `--hatch-coral` |

**Type.** Figtree only, from next/font (`src/nx/ds/fonts.ts`), upright and italic in one family. The scale is
`--fs-11 / 13 / 15 / 18 / 25 / 40 / 54 / 68` and the weights are `--w5 / --w6 / --w7` (500 / 650 / 750). Figures are
tabular. Micro-labels are 11 px capitals tracked `--track-micro` (0.08em); use `<Micro>`. The map voice is italic for
water names and depth labels, spaced capitals for zones, capes and latitudes. The v13 role tokens now point at the
scale: `--type-display` 68 (54 on a phone or a short screen), `--type-title` 25, `--type-head` 18, `--type-body` 15,
`--type-small` 13. `--font-sans` and `--font-serif` are aliases of `--font`.

**Space and shape.** The base unit is 4 px (`--s-2 … --s-64`). Objects pad 24 to 28 px (`--pad-obj`, `--pad-obj-x`).
Objects stand `--gutter` from the screen's edges: 24 px, 20 on a phone. Radii are `--r-card` 20, `--r-ctl` 12,
`--r-sm` 9, `--r-bar` 14, `--r-sheet` 24. Phone targets are at least 44 px.

**Modes.** Day is the prototype's palette. Dusk and Night are not in the prototype. They keep v13's dark chart tables
and map the synthesis roles onto them: a lighter ultramarine and coral on the dark ground. Night uses outlines instead
of shadows, and the deep-water texture is off. The chart-mode menu works as before.

**The data-state grammar** (`<Glyph kind>`, `GLYPH_WORDS`):

| Glyph | Meaning |
|---|---|
| `measured` ● filled | measured |
| `old` ◐ half-filled | measured, more than 30 days ago |
| `unverified` ○ ring (`unverified-coral` for regulation) | official record, not verified by a person |
| `zero` □ | "Reported 0": at or below the lab's limit, never a hero numeral |
| `predicted` ┄ dashed | predicted or modelled (NWS seas, the next whale look, C-HARM) |
| `hypothetical` ▨ ultramarine hatch / `illustrative` coral hatch | hypothetical (the trip) / illustrative (a sample change, an advisory area) |
| `none` ⋯ dotted | no data (no test site, a blank input, not measured); `<Slot>` is the same as a box |
| `unread` / `read` | an alert's state |

## 2. Shell and layout

One persistent full-bleed MapLibre chart lies behind every page in the route group, with a single instance across
every navigation. The rest floats on top of it:

- **Top bar** (`nx-bar`): the brand pill (mark, "Neritic", then the port with its finder, `nx-port-control`) on the
  left. On the right are the nav pill and the menu (`nx-menu-button`). The nav pill holds Today / My Coast /
  Research / Alerts with the unread count (`nx-alerts-link`, `nx-unread`); a dark pill slides to the current section
  on the `navPill` spring. Explore lives in the menu and keeps Today current, and so do Notices.
- **Layer chips** (`nx-layers`, `role="group" aria-label="Map layers"`) sit under the brand. Each chip carries
  `aria-pressed` and `data-layer`, and its one-line key opens on hover or focus.
- **Map controls** sit bottom-right: zoom, back to the page's place (`nx-zoom-in`, `nx-zoom-out`, `nx-refit`), and
  the scale with "Not for navigation". They rise above a dock or pass that reaches that corner.
- **Phone (≤ 640 px)**: the map is a strip at the top and the page reads in one bottom sheet (`nx-margin`) whose
  resting detent leaves the strip useful. The sections move to a 62 px tab bar (`nx-tabs`); the chips become one
  "Layers" button. Zoom and scale are hidden, since a pinch zooms.

A page declares its layout with `PageLayout`:

| `mode` | For | Wide screen | Phone |
|---|---|---|---|
| `stage` | v14 pages | The page is a transparent stage; its `Ticket`, `Dock` and `Pass` place themselves on the chart | The sheet. Objects are blocks in reading order (a `Pass` shows its stub first) |
| `chart` (or no PageLayout) | v13 pages not yet moved | The page's column floats as a card on the right (404 px) | The sheet, resting at min(44 %, 340 px) |
| `desk` | v13 desk pages not yet moved | The desk floats as a wide panel on the left; the map shows to its right | v13's chart strip over a scrolling page |

`PageLayout` also takes `phoneMap` for stage pages: the share of a phone's screen the map strip keeps above the
resting sheet (default 0.27, the Today composition). Use 0.4 or more where the strip must show a zone or a station
with its neighbours (design issue 2), and give the scene the meaningful features to fit.

The phone sheet's detents are rest, half when there is room, full, and a `peek` that only a page asks for:
`openSheet("half")` from `@/nx/shell/Margin` (the old `openMargin` still works).

## 3. Components

### Objects on the stage (`ds/stage`)

```tsx
<PageLayout mode="stage" phoneMap={0.27} />
<MapLayers page="today" defaults={{ regulation: true, depth: true, sea: true }} />
<ChartScene scene={{ id: `today:${port.key}`, view: { bounds: zoneBounds }, focus: port }} />
<ChartRegulation spec={{ zones: { highlight: "2", caption: "Season delayed · Fleet advisory" } }} />

<Ticket key={port.key} at={{ lon: port.lon, lat: port.lat }} anchorY={170} labelledBy="today-h">
  <TicketMain>
    <Statement as="h1" id="today-h" tone="regulation" scope="Fort Bragg · Dungeness crab, commercial"
               lede="Season delayed to at least" hero="Jan 1, 2027" />
    <StampLine>CDFW · checked Nov 8 · not verified</StampLine>
  </TicketMain>
  <TicketStub>
    <Tabs label="Detail" idBase="t" value={tab} onChange={setTab} items={[{ id: "status", label: "Status" }, { id: "fisheries", label: "Fisheries", count: 8 }]} />
    <TabPanel idBase="t" id={tab} active>…</TabPanel>
    <Button variant="primary" href={tripHref}>Plan a trip</Button>
    <FollowButton name="Fort Bragg" following onChange={setFollow} variant="toggle" />
  </TicketStub>
</Ticket>
<Dock label="Sea, next four days" width={840}>…</Dock>
```

| Component | Props | Behaviour |
|---|---|---|
| `Ticket` | `at: {lon,lat} \| null`, `anchorY = 160` (px from its top level with the place), `width` (404 Today, 472 Alerts), `side = "east" \| "west"`, `label` or `labelledBy`, `lead = true`, `inline`, `data-*` | Stands 28 px east (or west) of its place, with a ≤ 24 px leader stub drawn when the gap is ≤ 30 px. The camera brings the place to its edge, and the lead ticket also reserves its side of the chart for the camera. It follows the place when the reader pans and stops at the screen edges (`--gutter`, under `--stage-top`). It grows from its pin at 62 % of the camera's flight, and when its page or place changes, a copy shrinks back toward the pin (170 ms) before the camera leaves. **Give it a `key` that changes with the place.** On a phone it is a block in the sheet. `inline` puts it in its container's flow (the catalog) |
| `TicketMain`, `TicketStub` | div props | The main part and the perforated stub, joined by notches and a dashed perforation. The stub scrolls when the ticket is taller than the screen allows |
| `Dock` | `label`, `align = "left" \| "right" \| "full"`, `width` (840 for the sea dock), `cover = true`, `inline` | The low panel along the chart's foot where time lives. It springs up 70 ms after its ticket, and the camera keeps the place above it. On a phone it is ruled off under the ticket |
| `Pass` | `label`, `at?`, `stub`, `stubWidth = 320` (Research 340), `inline`, `className`, `stubClassName` | The wide boarding pass along the bottom (the trip, Research). The main part holds the work and the perforated stub holds the result and the one filled action. A caret on its top edge stands under its place. On a phone the stub comes first |
| `Sheet` | `map` (0.16 to 0.7) | Sets the phone sheet's resting detent from inside a page (`PageLayout phoneMap` does the same) |

### Controls (`ds/Button`, `ds/synth`)

| Component | Props | Behaviour |
|---|---|---|
| `Button` | `variant: "primary" \| "ink" \| "secondary" \| "quiet"` (`"text"` = quiet), `glyph: "→" \| "↓" \| "↑" \| "+" \| "×" \| "✓" \| "↗" \| null`, `size`, `icon`, `iconEnd`, `busy`, `busyLabel`, `disabled`, `full`, `href`, `external`, `title` | A primary is a small ticket: words, a dashed perforation, and a 40 px stub whose glyph says what happens (default →, ↗ for an external link). Hover slides the glyph (120 ms); press sinks on a spring (700 / 18). The secondary is paper with an ink edge. The quiet variant is ultramarine words with a dashed underline. **One filled button per screen** |
| `FollowButton` | `name`, `following`, `onChange(next)`, `variant: "primary" \| "toggle"`, `disabled` | Follow → Following: paper sweeps across (260 ms), then a tick draws in the stub (340 ms). `toggle` is the quiet confirmed toggle that stands beside the screen's real primary once the port is followed (design issue 1) |
| `Chip`, `LayerChips` | `pressed`, `onPress`, `swatch`, `keyText`, `data-layer` | The shell renders `LayerChips`. Pressed is an ultramarine wash |
| `Tabs`, `TabPanel` | `label`, `items: {id,label,count?,disabled?}[]`, `value`, `onChange`, `idBase`, `look: "seg" \| "words"` | The ticket's Status / Record / Fisheries 8 / Notices 2 (`seg`) or Research's variable tabs (`words`). Roving focus with Left/Right/Home/End. The panel fades and slides in (220 ms) |
| `RollingNumber` | `value`, `format`, `empty`, `cents`, `pace: "fast" \| "slow"`, `srSuffix` | Digits roll in clipped columns from the old value (260 ms, or 360 for a sample's value), then settle to the plain string. A figure of a new shape is written as it is; it never counts up from zero |
| `Statement` (poster) | `lede`, `hero`, `tone: "ink" \| "regulation" \| "closure"`, `scope`, `as`, `id` | The date is the hero: 25 px lead words over a 68 px date (54 on a phone or a short screen), coral-ink for regulation. `title` still gives the v12 form |
| `Glyph`, `Micro`, `StampLine`, `Slot`, `Pennant`, `WindArrow`, `LimitGauge` | see §1 | The grammar's parts. `LimitGauge value limit label` is the "9.8 of 30 ppm" bar with its caliper |
| `Toggle`, `Segmented`, `IconButton`, … | v13 props | Take the new tokens |

### Charts (`ds/synth/charts`)

All four are plain SVG sized to their container, with values written where they belong. Each animates with the
presets in §5 and resolves at once under reduced motion.

| Component | Props | Notes |
|---|---|---|
| `Waterfall` | `label`, `steps: CascadeStep[]` (`{ id, label, basis?, amount: number \| null, kind: "start" \| "cost" \| "share" \| "end", valueText? }`), `format`, `hypothetical = true`, `breakEven?: { at, label }`, `height = 168` | Cascades gross → net on arrival (70 ms apart) and tweens on every input. Values sit at the column ends; a null amount is a dotted slot and the levels after it wait. Put the net dollars in the stub and the share in `valueText` ("72% of gross"). The break-even caliper is drawn only here. The v12 ledger is now `LedgerWaterfall` |
| `Dumbbell` | `label`, `rows: { id, label?, lo, hi }[]` with ends `{ value, label, text?, mark: "filled" \| "hollow" \| "hatched", tone, onPress?, pressLabel? }`, `caliper?`, `heading?`, `delta?`, `slide?`, `domain?`, `ticks?` | What-if: each end is a 44 px button that sets its input, and the caliper is "as entered"; ends slide on change (360 ms). Alert: `slide` brings "now" out from "was" with an overshoot, then `delta` ("+14 days") |
| `DatesLine` | `label`, `domain`, `span?: { from, to, fromLabel, toLabel, openEnd }`, `window?`, `today?`, `ticks?` | The delay is hollow coral (official, not verified), dotted past "at least". The next assessment is a dashed box; today is an ink rule. Draws in 700 ms |
| `SeaBars` | `label`, `domain`, `periods: SeaPeriod[]`, `measured?`, `advisories?`, `days?`, `keyText`, `max = 24` | NWS seas per period as dashed bars (predicted), each value inside its top. The buoy is the one filled mark (hovering it fires `nx-buoy-hl`). Advisories are coral spans in NWS's words. Bars rise in 620 ms; hovering or focusing a period shows NWS's words |

The v13 **plot kit** keeps its API and interaction contract and takes the synthesis look. The hover crosshair is a
dashed ink rule whose ring glides between samples on a spring (`crosshairGlide`, 620 / 44). The pinned sample is
ultramarine: a solid rule and a ringed dot. The tip is ink, the brush window an ultramarine wash with rounded handles,
the threshold a dashed `--ink-2` line.

## 4. The map

**Cartography** (`chart/style.ts`, from the bundled data):

- The lit sea is the depth areas painted deepest first. Every area's edge is a fine light isobath, from the seamless
  `public/chart/contours.geojson` built by `scripts/build-contours.mjs`; the index isobaths are brighter.
- The shore has a pale surf band and a white casing under a `--coast` line.
- Land is warm paper with a whisper of relief, and is drawn past the data's east edge too.
- A graticule is light on the water and faint on the land. Its parallels are named at the right edge
  (`GraticuleLabels`).

**Layers.** The page sets the default chips and the reader's choices are kept per page for the visit:

```tsx
<MapLayers page="trip" defaults={{ regulation: true, depth: true, sea: true, toxin: true }} />
const seaOn = useMapLayer("sea");      // read
setMapLayer("regulation", true);       // switch (a fishery picked that has an area)
<ChartMarks marks={[{ id: "buoy-46014", layer: "sea", lon, lat, priority: 60, leader: true, content: <Reading … /> }]} />
```

| Layer | Drawn by | What |
|---|---|---|
| `regulation` | the chart (`RegulationLayer`) from `<ChartRegulation>` | Whale-zone limits at their real latitudes: a coral line drawn from the shore out (800 ms), the latitude ("40°10′ N") at the shore end and the cape or line name on the land. The port's zone is tinted, with a ribbon inside each limit and a tag "WHALE ZONE 2" over `caption`. Neighbours are quiet ("Zone 3"). Notice bands work as in v13 |
| `depth` | the chart | The 20 / 50 / 100 / 200 m edges as ink contours (the 200 m shelf edge heavier), labelled along the line in true fathoms with metres small ("11 fm 20 m", "27 fm", "55 fm", "109 fm"). The 1,000 and 2,000 m lines ("547 fm", "1,094 fm") are always labelled. Labels place when the camera settles and ride with it while it flies |
| `sea`, `toxin` | the page, as `ChartMarks` with `layer` | Drawn only while the chip is on, fading in 320 ms and out 220 ms |
| Explore's data layers | the page | `<MapLayers extra={[{ id, name, question, key, swatch: "data" }]}>` adds chips after the four; read them with `useMapLayer(id)` (an extension of the prototype) |

**Ports** are drawn by the chart: the current port is a large ringed dot on a paper halo whose ring expands once on
arrival; the home port is tagged "Home port"; a followed port is ultramarine and tagged "Following". **Water names**
are Figtree italic and letter-spaced: big and light on the deep, smaller and dark with a halo elsewhere.

**Camera.** `ChartScene` works as before. Between places the camera flies (`map.flyTo`, 0.5 to 1.15 s by distance,
on the `EASE_FLIGHT` curve [.45, 0, .15, 1]) after any leaving ticket has shrunk. A ticket's place is brought to its
edge (`chart/anchor.ts`). Names and lines stay on the chart through the flight.

**Ambient** (`chart/DeepWater.tsx`): Paper Shaders' `water` caustics run inside the water deeper than 200 m only. They
load lazily after the chart has drawn, never on the static or reduced tier or in Night, and hide while the camera
moves. They pause in a hidden tab, hold still under reduced motion, and render at 1x on a phone.

**For QA**, `nx-sheet` carries `data-layers` (the layer ids actually drawn).

## 5. Motion (`ds/motion.ts`): the prototype's inventory

| Moment | Preset | Value |
|---|---|---|
| Screen change: the old ticket shrinks to its pin → the camera flies → the new ticket grows at 62 % → the dock follows | `ticketOut`, `flightSeconds(px)` + `EASE_FLIGHT`, `FLIGHT_NEAR`, `ticketIn` from `TICKET_FROM` with `ticketOpacity`, `dockIn` | 170 ms; 0.5 to 1.15 s; spring 340/26 from 0.86 with a 70 ms opacity snap; spring 260/24 + 70 ms |
| Nav pill | `navPill` | spring 420/34 |
| Today: dates draw, seas rise, zone limits draw, port ring | `datesDraw`, `seaRise`, `zoneDraw`, `portRing` (`buoyPulse` for a page's buoy) | 700 ms; 620 ms; 800 ms; 2.6 s once |
| Trip: cascade, tween, roll, what-if, field pop | `cascade(i)`, `cascadeTween`, `roll`, `whatIfSlide`, `fieldPop` | 420 ms + 70 ms stagger; 300 ms; 260 ms; 360 ms; spring 500/22 |
| Research: line draws, crosshair glides, brush eases, value rolls | `lineDraw`, `crosshairGlide`, `brushEase`, `rollSlow` | 1.0 s; spring 620/44; 550 ms; 360 ms |
| Alerts: now slides | `alertSlide` | 800 ms [.34, 1.25, .64, 1] after 180 ms |
| Controls: press, glyph, sweep, tick, toggle, layers, panels | `pressSink`, CSS 120 ms, `sweep`, `tickDraw`, `--ease-toggle`, `layerIn` / `layerOut`, `panelIn` | spring 700/18; 260 + 340 ms; 320 / 220 ms; 220 ms |
| Ambient | `DEEP_SPEED` | 0.22 |

Reduced motion resolves every item to its final state at once: `MotionPrefs` skips Motion animations, the camera
jumps, and the shader holds still. The v13 presets remain for v13 parts.

## 6. How to build a page in v14

1. Render `<PageLayout mode="stage" phoneMap={…} />`, `<MapLayers page=… defaults=… />`, `<ChartScene>` (fit the
   meaningful features: the port's zone, a station with its neighbours) and `<ChartRegulation>` where rules apply.
2. Lead with **one** `Ticket` beside the place (Today, Alerts) or one `Pass` along the bottom (Trip, Research). Put
   time in a `Dock`. Key the ticket on its place.
3. Say one thing per screen with the poster `Statement` or one lead figure (`RollingNumber`). Use **one** filled
   `Button`. Caveats are glyphs (`Glyph`, `StampLine`, `Slot`), not sentences.
4. Put readings on the chart with `ChartMarks` and a `layer` (`sea`, `toxin`). A number lives once: at the buoy, at
   the column end, or in the stub.
5. Draw figures with the chart kit (`Waterfall`, `Dumbbell`, `DatesLine`, `SeaBars`) or the plot kit. No raw d3 in
   pages.
6. Check the phone: objects become the sheet's blocks in reading order. A `Pass` shows its stub (the result and the
   action) first.
7. Don't add `position: fixed` panels or your own button styles to a page.

## 7. What stays, what changed for v13 code

- Every v12 and v13 export still works and takes the new tokens. `Waterfall` (v12 ledger) is now `LedgerWaterfall`;
  `Waterfall` is the synthesis chart.
- `PHONE_QUERY` and the shell's phone breakpoint are now **640 px** (was 720). A page stylesheet written for 720
  should move to 640.
- The nav is Today / My Coast / Research / Alerts. `activeDestination("/explore")` is `/today`.
- The desk's divider, rail and pane sizes are gone: legacy desk pages float as a panel until they move to `stage`.
  `useChartPane().toggle` still opens the chart on narrow screens and the phone strip.
- Vitest stubs CSS modules (class names are their keys), so a unit test no longer lists every stylesheet.

## 8. Lint (`npm run lint:design`), amended for the synthesis

- **Text:** nothing under 11 px.
- **Shadows and hatches:** shadows are allowed when they use token colours. Gradients and hatches come from tokens
  (`--hatch-*`, `--notch-*`, `--sw-*`), and backdrop filters stay out.
- **Radii:** a page uses the radius tokens; the design system and the chart may draw their own small shapes.
- **Capitals:** capitals and spaced labels belong to the design system (`Micro`) and the chart's map voice. A page
  writes sentence case.
- **Spacing:** a page's spacing sits on the 2 px grid up to 64 px, or uses a `--s-*` token.
- Everything else is unchanged from v12 §3 and v13 §10: token colours, Figtree, the motion presets, truth words, ISO
  dates, and the page rules.
