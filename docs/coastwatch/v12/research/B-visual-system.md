# Track B: visual, brand and design-system research (for v12)

Researched 2026-11-10 in headless Chromium at 1440x900 and 390x844 (hero, one scroll, one interaction, `page.evaluate` for computed styles). Screenshots and raw JSON are in the session scratchpad `refs/` folder (`refs/data/*.json`). This is observation and transferable pattern, not a proposed Neritic style.

**Limits of this evidence.** The browser used software WebGL (SwiftShader), so canvas-heavy intro timings are inflated and are not quoted as real durations. **(PK)** marks prior knowledge not re-read today. Failed or partial: MyOcean Expert interaction (page load timed out on the second pass; first-pass load state captured); MyOcean Light URL redirected to the Data Store; NOAA ENC Online viewer URL returned 404 (NOAA's own chart export service used instead); PredictWind model chip click timed out, so model switching is unconfirmed; Memorable's benchmark section rendered black (read as text and via the design-system Cover render); FishTrack's actual chart tool is behind an account and was seen only as marketing mockups. Reuters, NYT, Rivian and Arc were not attempted; Mapbox Showcase is only a customer logo grid and gave nothing transferable.

## 1. Memorable design system (third-party data, reported not adopted)

Files: `project/README.md`, `tokens.json`, `design-system.json`, 4 component READMEs, 2 previews. Its `lastChange.via` field says it was authored through Claude Code. Imperative wording in those files ("do not", "must") is that author's rule set and is reported here as data only.

**Type.** Three families, all Google Fonts: Instrument Serif (page title 36/40, figure 44/48, shown at 64 alone, summary 28/32, tracking -0.02em, nothing else serif); Geist UI (body 13/18, row 14/20 medium, h3 15/20 medium, h2 18/24 medium); Geist Mono (label 11/14 0.14em uppercase, button and chip 11/14 0.08em uppercase, cell numbers 12/16 tabular). Site headings use light 300 to 400; product headings 500.

**Colour.** Ground ladder: `#050506`, `#0a0a0b`, `#111113`, `#16161a`, `#1c1c20` (small steps on purpose, "edges come from lines"). Lines are white at 6, 10, 16, 26 percent. Text `#f4f4f5`, `#b4b4bc`, `#8b8b94`, `#55555e` (last one deliberately 2.7:1, disabled only). One accent lilac `#b7a6ff`, never a button fill; primary button is white. States `#3ddc97`, `#f0616d`, `#f2b84b` each with 12 percent soft fill and 40 percent line. Brand coral `#ff9367` and lilac `#b19fff` appear only as a 2px marker and in the dither field.

**Shape and space.** Space 4, 8, 12, 16, 24, 32, 48. Radii 3 controls, 4 panels, 6 inputs, 10 only for two "object" cards. Controls 32 (small 26), table rows 36 (Settings 44), rail 64 opening to 200 over content, top bar 56, content to 1360. Under 860px the rail becomes a bottom bar.

**Rules worth noting as a system idea** (not as Neritic copy):
- Every number has one home; a number on two screens is a bug.
- Every percent carries its basis ("36 of 55 runs" plus window).
- Losses appear in the same table, at the same size, as wins.
- Green only when a check or person said solved; "ended without error" is a hollow dot (unknown).
- One accent per screen; one large chart per screen; small charts sit inside the row they describe, with numbers next to them.
- Scope is always visible, first item in the top bar.
- No invented copy; labels name data. One word per thing, with a banned-synonym table.
- Signature effects each get exactly one location ("using one somewhere else dilutes it").

**Signature inventory.** Bayer dither 4px in coral and lilac (rail foot only, 22 to 52 percent alpha); glyph field `@#$%&8SX0B...` (sign-in only); 2px coral-lilac marker; hatched bar (115 degree hatch, 5px pitch, solid fill to value, caliper tick); node boxes with travelling dot; one white mono button per screen.

**Motion table.** Hover 120ms; logo bars rise 380ms, 50ms apart; first-session logo fill 640ms once; page groups rise 6px and fade 500ms, 40ms stagger; first list skeleton 340ms; numbers count from zero 900ms; title scramble 420ms first visit only; bars grow and lines draw 400 to 900ms; active marker slides 360ms; new rows slide with accent wash 600ms. Eases `cubic-bezier(0.16,1,0.3,1)` entrances and `(0.34,1.4,0.64,1)` small springs. Instant under reduced motion.

**Data grammar.** Hollow dot and muted mono for "without", accent dot and bright figure for "with", then delta coloured by meaning; hatched comparison bars; stacked bars with one baseline, no gridlines, three ticks; dumbbell per task; steps as numbered mono nodes on a 1px spine.

**Caution for the lead.** This system's serif-title-on-near-black plus Geist Mono caps is itself close to the 2025-26 "AI-tool" look (section 4). Its discipline (one-home numbers, one-place effects, unknown-is-hollow) transfers better than its surface.

## 2. Direct observations

Key: T type, L layout, N nav, C colour (hues doing real work), I first-timer, S simplification, M motion, G data graphics, X what fails.

### memorable.sh (`memorable-intro-12s`, `memorable-depth-*`, `memorable-ds-cover`)
- T: Geist H1 69.12/72.58 weight 400, -2.42px; H2 48.96 weight 300 to 400; body 15.84/23.76; secondary 12.96; Geist Mono 10.08 labels. Instrument Serif is loaded but not used for headings on the site. Fractional sizes come from a root scale of 0.72/1.08 rem. Phone H1 39.39.
- L: 16,563 px page, staged as full-viewport scenes. Floating hairline nav bar inset about 15px. Left-aligned hero, then centred set pieces.
- N: 3 mono links, ghost "Book a call", white "Dashboard". Logo left.
- C: `#060608`/`#09090b` ground; text `#f4f4f5`, `#9f9fa9`, `#71717b`. Effective hues about 2 (violet field, coral spark); small dots `#3bc4ff`, `#b19dff`, `#3bd89d`, `#d8764e` carry categories.
- I: "How It Works" is an exploded isometric stack of 4 plates (traces, workflow synthesis, graph assembly, retrieval), each with a numbered white mono tag and a leader line to a short note. Best first-timer device in the set.
- S: One concept per viewport; FAQ answers hidden behind "Show answer +".
- M: Glyph field canvas behind a per-glyph headline reveal; three-bar logo draws in first; hero shrinks into a frame as the next headline slides past with a ticker ("8h 1m, 2.88m tokens"); logo extrudes into a plate; steps tilt in 3D. 15 canvases, 1 video, about 20 running animations.
- G: Benchmarks are big delta (-19%, -40%), then without/with figures, then a basis line ("454 runs, every one passed").
- X: Long stretches of near-empty black (depth 58, 72, 86). Phone was still black at 14 s under software GL. Text at 10.08px `#71717b` is 4.19:1. Content is gated by the choreography, so a slow device sees nothing. Would be punishing on marine data links.

### FishTrack (`fishtrack-d-*`, `fishtrack-charts`)
- T: Linearsans H1 40/44 700, body 16/26, `#333`/`#555` on white; photo CTA 58/70 900. Icon font for app icons.
- N: "SST Charts" opens a mega-menu of 8 world regions and 40-odd named grounds, including Southern California, Northern California, Pacific Northwest. Geography is the IA.
- C: White plus teal `#67d3df` (1.76:1 on white, fails); charts are full rainbow.
- I: App-screen mockups; copy describes a "bullseye" at screen centre: move the chart under it to read lat/long, water depth and temperatures.
- G: SST, chlorophyll, true colour, SSH, bathymetry, currents as separate thumbnails; overlays offered on any SST image.
- X: Page reads as a 2015 template; the product itself is not visible without an account.

### PredictWind (`predictwind-d-hero`)
- T: Adobe acumin-pro. No H1 element at all on the home screen.
- L: Full-bleed animated wind field is the whole home page (scrollHeight equals viewport). Row of 8 model chips, then 5 variable icons, then one white CTA, stacked at bottom centre.
- C: Saturated rainbow ramp (5+ hues) under white streamlines.
- I: Marketing claim is literally the working field. A first-time visitor sees the product before any sentence.
- X: Rainbow overwhelms country labels, chips float on data, and nothing says what the colour means.

### earth.nullschool.net (`nullschool-ca-sst`, `nullschool-click`, `nullschool-menu`)
- T: M PLUS 1 13/19.5 UI, 18/27 for "earth". 26 DOM elements visible at rest.
- N: One 37px button bottom-left (`rgba(0,0,5,.85)`, radius 8).
- C: Land is flat dark grey `#303030` (sampled) with thin white coastlines, so the only saturated thing on the page is data.
- S: Click drops a green ring and prints, in a card bottom-left: coordinates, then one line per layer: name, value, unit. Nothing else changes.
- G: Menu is a two-column list: Date, Data, Source (names CMEMS and NOAA OISST), Scale (colour bar), Control, Mode, Animate, Overlay, Projection. Legend, provenance and controls are one object.
- M: Streamlines move only because the data moves. No UI motion at all.
- X: No legend visible until the menu opens. Discoverability relies on prior knowledge. Same hue cue for anomaly bands is hard to read without the scale.

### Copernicus Marine (`copernicus-d-*`, `myocean-d-hero`)
- Home: Arial, institutional logo stack, 18/23.4 body, `#22689b` blue with orange headings, cookie panel. Reads as a government portal.
- MyOcean Pro: Lato 16/20.8, 10/13 axis. Floating white layer panel at top-left holding 3 stacked layers, each with download, info, style, log icons and its ramp inside the layer row. Black land. Time bar along the bottom spanning 2022 to 2027; depth slider bottom right.
- X: On phone the panel covers about 60 percent of the map. Icon row meaning is not guessable.

### Global Fishing Watch (`gfw-d-*`, `gfwmap-d-hero2`, `gfwmap-m-hero`)
- Home T: Roboto 55/55 700, Open Sans body. C: `#1b4b87` plus CTA red `#e22e12`. Ordinary NGO site.
- Map L: Light-blue left panel of grouped layer cards (Activity, Detections, Events); dark ocean; tool column; timebar.
- C: Hue per dataset: AIS teal, VMS orange. The ramp sits under the layer title with values 407 to 28K in "hours / 32,000 km²".
- G: The timebar is drawn as a stream of data volume in the layer's own hue. A gap in coverage is visible as a gap in the shape.
- I (phone): "What would you like to do today?" with two illustrated task cards: search for a vessel, run a report on an area.
- X: A news popup sits over the map. Light panel against dark map feels split. Coarse cells at default zoom.

### Motion (`motion-intro-*`, `motion-d-*`)
- T: TASA Orbiter (OFL, wght 400 to 800) H1 34.56/37.3 700, -1.2px; H2 40/44; body 15/23.25; Geist Mono 11 uppercase +1.21px for nav and buttons.
- L: H1 is deliberately small and sits low-left under a large graphic. CTA pair sits low-right.
- C: `#080808`, `#edebe5`, `#979d97`, one bright yellow CTA. Hue count 1.
- M: Point cloud sphere resolves into the logo, then idles. Sections are live demos with code directly under them. CSS transitions 150 to 200ms.
- X: 13,600 px of demos; dim text on black in long runs.

### Paper Shaders (`paper-dithering`, `paper-halftone-dots`, repo)
- Licence Apache-2.0 (GitHub API). 30 effects: paper-texture, fluted-glass, water, image-dithering, halftone-dots, halftone-cmyk, lens-distortion, heatmap, liquid-metal, gem-smoke, mesh-gradient, static gradients, dithering, grain-gradient, dot-orbit, dot-grid, warp, spiral, swirl, waves, neuro-noise, perlin, simplex, voronoi, pulsing-border, metaballs, color-panels, smoke-ring, god-rays.
- Control vocabulary is reusable: `colorBack`, `colorFront`, `shape`, `type` (random, 2x2, 4x4, 8x8), `size`, `speed`, `scale`. Halftone has `grid` hex, `radius`, `contrast`, grain controls.
- X: A catalogue of decoration. Metaballs, god rays and mesh gradient are exactly what now reads as templated. The 2-colour dither and halftone effects are the ones with print and chart ancestry.

### Linear (`linear-menu`, `linear-deep`)
- T: Inter Variable at non-standard weights 510 and 590. H1 64/64 -1.4px; H2 48/48; UI 13 to 15. Berkeley Mono for code.
- C: `#08090a`, `#0f1011`, `#f7f8f8`, `#d0d6e0`, `#8a8f98`, `#62666d`. Hue count 0 in chrome; product screenshots carry small status colours. `#62666d` is 3.45:1.
- L: Real product UI as the hero, fading into the page at the bottom edge.
- M: 160ms `(0.25,0.46,0.45,0.94)` for UI, 700ms `(0.32,0.72,0,1)` for large reveals.
- X: Its surface has been imitated so often that the imitation is now the generic look.

### Coinbase CDS (`cds-d-hero`, `cds-components-*`, `cds-colors`)
- T: Inter 400 display, 40/48 and 28/36. Not heavy.
- C: Semantic tokens point at a 0 to 100 spectrum (Foreground GRAY100, Muted GRAY60, Primary BLUE70, Positive GREEN60, Negative RED60, Warning ORANGE70, Elevation 1 GRAY5, 2 GRAY10), and each status has a Wash (BLUE0, RED0, GREEN0, ORANGE0).
- G: LineChart docs name the right primitives: Scrubber, ReferenceLine, Period selector, Missing Data, Live Updates, Gradients. RollingNumber rolls each digit and can pulse colour.
- I: Hero is a split-flap style letter board spelling a sentence.
- X: Pill and rounded-rect cards everywhere; fintech friendliness does not fit a working tool.

### Windy (`windy-waves-ca`, `windy-picker`, `windy-point`)
- T: System font for UI, custom `bariolnumbers` for numerals, map labels 10 to 11.
- L: Layer list with thumbnails on the right edge, timeline bottom, ramp strip bottom right.
- C: Ramp strip prints values inside the colour band (ft 1.6, 3.3, 5, 6.6, 20, 30).
- S: Point page is a five-day hourly table: one row per variable, wind cells filled by value, day headers with a predictability percent. Tabs swap Basic, Waves and Tides, Meteogram, Wind, Airgram; source selector in the same strip.
- G: Picker on a draggable pin prints "Waves 8 ft W, period 10 s". Wave particles differ from wind particles.
- X: Geo-IP city plus article teaser over the map, 12+ right-edge items, magenta wave field over everything, red close button.

### Felt (`felt-d-hero`, `felt-1000`, `felt-2300`)
- T: GT Alpina light serif H1 69.12/70.5, H2 46/48.76, intro 26/33.8; Atlas Grotesk 14/21.7; Publico Text 16.5/26.4 in FAQ; SF Mono labels. All commercial.
- C: Paper `#f4f2ea`, ink `#1c1b18`, `#69685f` (5.0:1), one rust `#a7531b` (4.81:1) on mono eyebrows. Hue count 1.
- L: A black 29px telemetry strip above the nav (measured): LAT, LON, BRG, ZOOM, CRS and a scale bar in mono; the values differed between captures, so they update as the page runs.
- G: Illustrated aerial scene with pins, a small count legend placed by the data ("Open 4, Filled 3") and comment cards anchored near the road.
- X: Scenes are illustration, not live data. The mono eyebrow is the one template habit on the page.

### Stripe (`stripe-d-hero`) and Vercel (`vercel-d-hero`)
- Stripe: sohne-var weight 300, H1 48/55.2. Headline is a two-tone sentence: first clause ink, the rest in muted blue. Live counter "Global GDP running on Stripe: 1.74726465%" above it. Ribbon gradient is its signature and its most-copied element.
- Vercel: Geist 64/64 -3.84px on black, triangle spotlight, two pill buttons, logo row. Hue count 0. Reads as a template only because so many copied it.

### Teenage Engineering (`te-d-hero`, `te-product`)
- T: Custom te-20 at weight 100 and TechnoType. Nav items are line drawings beside a word and three small links.
- L: One object in black with a spotlight, nothing else in the fold. Product page: full-bleed photograph, outlined display type, katakana in orange.
- X: Weight 100 and 9px labels, weak findability, weak phone IA. Art direction at the cost of use.

### The Pudding (`pudding-d-hero`, `pudding-map`)
- T: Gooper SemiCondensed 32/32, Atlas Grotesk 16/22.4, Atlas Typewriter caps 12.8 to 14.
- C: Dark ground, every story card its own saturated frame colour, serial number and date ("#228, Nov 2026").
- I: Story opens on an invented world map with one question card, "Start" and an equally visible "Skip story".
- X: Home is deliberately loud; it does not generalise to a tool.

### swisstopo (`swisstopo-zoom`)
- Chrome: Noto Sans 13.2, Bootstrap greys, dated. Map: dense but calm. One colour and one type style per class: grey rock, brown contours, blue water, green forest, spaced caps for regions, italic for water. Sampled: paper `#F2F5F6`, forest `#C6DFAF`/`#9BB08C`, lake `#457E9D`/`#6FA8C7`, contour `#614B2C`/`#B0925E`, road red `#943433`.
- X: Interface is far weaker than its cartography.

### NOAA ENC, Monterey Bay (`refs/src/noaa-enc.png`, NOAA MCS export)
- 41 percent `#DDF1EE` deep water, 29 percent `#CEC58A` land, 15 percent `#9ECEFC` shoal, `#82BDFF` very shallow, `#B09E56` built-up, `#8B9593` soundings, magenta for regulated areas. Depth is palest offshore, most saturated at the coast. Soundings are grey numbers with subscript decimals, used as texture. Exactly one foreign hue (magenta) carries the regulatory layer.

### Admiralty charts (Commons scans of No 3520, 1959; No 2941, 1952)
- Paper `#FFFCE7` to `#FEEFCA`, engraved near-black ink (`#000001` to `#2C261F`), a single purple overprint `#6D385E` for lights. Hierarchy is carried by engraved letter style, line weight and hachure, with almost no colour.

## 3. Transferable patterns

| # | Pattern | Seen | What it does | How it could apply to a marine product |
|---|---|---|---|---|
| 1 | Single resting control: one 37px button in a corner, all else inside it | nullschool | Page is 100% data at rest | Map view whose only persistent chrome is one labelled control |
| 2 | Click prints coordinates plus one line per layer (name, value, unit) in a fixed corner | nullschool | Readout never covers the point read | Tap water: fixed block with lat/long and each visible layer's value and date |
| 3 | Settings as a two-column rows list (label, then segmented options) | nullschool | One scan path for everything | Date, Layer, Source, Scale, Time step rows in one sheet |
| 4 | Provenance row inside the control panel, naming the producer | nullschool Source row | Source is a fact of the layer, not a footer | "Source" row with agency name beside the scale |
| 5 | Ramp strip with values printed inside the colour band | Windy | Legend is 20px tall and needs no ticks | Single strip with mg/m3 or ft inside the band |
| 6 | Hourly table, one row per variable, cells filled by value | Windy point page | Five days readable in one glance, tabular not chart | Wind, seas, swell period rows for a port with the same scale as the map |
| 7 | Tabs swap the same table's variable group; source selector in the same strip | Windy | Depth on demand without a new page | Marine, tide, water tabs on one forecast table |
| 8 | Draggable pin with a two-line caption | Windy picker | Value travels with the finger | "8 ft W / period 10 s" style pin label on phone |
| 9 | Colour hue assigned per dataset; ramp sits under that layer's title | GFW | Layer list doubles as legend | Each overlay's ramp lives in its row, never floating |
| 10 | Timebar drawn as data volume silhouette in layer hue | GFW | Coverage gaps are visible, not footnoted | Satellite pass coverage as shape; clouds become visible gaps |
| 11 | Onboarding as "What would you like to do today?" with two task cards | GFW phone | First choice is a goal, not a feature tour | Two entries: a trip question, or a place to look at |
| 12 | Geography is the top of the IA | FishTrack region menu | Users find themselves first | Harbour or port picker before any variable |
| 13 | Fixed centre crosshair, map moves underneath, values update | FishTrack (described in copy) | Precise reading on a phone without a tap target | Crosshair mode for one-handed use |
| 14 | Model chips as a row directly on the map | PredictWind | Disagreement is a first-class action | Compare products from a row, one tap each |
| 15 | One colour and one letter style per map feature class | swisstopo, ENC | Dense maps stay calm because classes do not share ink | A written feature-class table for the basemap |
| 16 | Few depth tints plus one foreign overprint hue | NOAA ENC | Regulatory layer pops by being the only hue of its family | Base in 3 tints; regulations in one hue absent from everything else |
| 17 | Palest at depth, most saturated at the coast | NOAA ENC | Attention goes where boats and risk are | Base ramp lightness inverted relative to ocean-as-dark |
| 18 | Numbers as map texture (soundings) | NOAA ENC, Admiralty | Values readable without a legend | Sparse printed values at buoy and station points |
| 19 | Subscript decimal (29 with small 2) | NOAA ENC | Compact, recognisable chart idiom | Wind or depth labels in a chart-native numeric style |
| 20 | Letter style carries hierarchy, not colour | Admiralty, swisstopo | Works in sun, in print, in grey | Spaced caps for sea names, italic for water, roman for land |
| 21 | Telemetry strip: mono LAT, LON, BRG, ZOOM, CRS, scale bar | Felt | Says "this is an instrument" in 29px | Thin strip that always reads the map state |
| 22 | Count legend placed next to the data it counts | Felt | Summary and map share one glance | "3 notices here" anchored at the harbour, not in the header |
| 23 | Comments and notes pinned to geography | Felt | Text gets a location | Notice text with a leader line to its boundary |
| 24 | Paper and ink neutrals with a single rust accent | Felt | Warm, print-derived and unlike SaaS dark mode | One accent on a paper field |
| 25 | Exploded isometric plates with numbered tags and leader lines | Memorable | Explains a layered system in one picture | Satellite, processing, product, what you see, as stacked plates |
| 26 | Hero shrinks into a frame while the next scene arrives | Memorable | Continuity between scenes | Globe to coast to harbour as one continuous camera |
| 27 | One signature effect, one location | Memorable system | Effect stays special | Any chart-print texture confined to a single place |
| 28 | Hatched bar with caliper tick (115 degrees, 5px pitch) | Memorable system | Compares two values on one scale with a print-like look | Tested value against action limit |
| 29 | Hollow dot equals unknown; filled only when checked | Memorable system | Absence of evidence never looks like good news | Unverified notice as hollow; verified as filled |
| 30 | Every percent carries its basis | Memorable system | Removes vanity numbers | "2 of 9 samples, last 30 days" alongside any rate |
| 31 | Every number has one home | Memorable system | Stops the summary repeating the detail | A wind value appears once; other places link to it |
| 32 | Dumbbell for a pair; numbers at row end | Memorable system | Pair comparison in row height | Observed against climatology per port, in a row |
| 33 | Semantic tokens point at a stepped spectrum, each status with a wash | CDS | Light and dark stay correct by construction | Status tokens as references, plus one wash per status |
| 34 | Scrubber, reference line and missing-data states as named chart primitives | CDS | Interaction is built in rather than bolted on | One chart kit with a defined gap state |
| 35 | Per-digit roll, optional colour pulse | CDS RollingNumber | Value change is noticed without layout movement | A live reading updating while the rest stays still |
| 36 | Split-flap letter board as the hero | CDS | A sentence as a physical object | Board-style arrivals display for harbour conditions |
| 37 | Point cloud resolving into the mark | Motion | Brand built from the medium itself | Real pixels or points assembling the coastline or mark |
| 38 | Small H1 low-left, large graphic above | Motion | Graphic is the subject; type is a caption | Map first, modest type as its caption |
| 39 | Two-tone sentence headline | Stripe | One sentence, two levels of emphasis | "Season delayed." in ink, reason continuing in muted tone |
| 40 | Live counter above the headline | Stripe | Single true number signals a live system | Last satellite pass or last check time as the top line |
| 41 | Real product UI as hero, fading at its edge | Linear | Shows the thing, avoids a device mockup | Actual harbour view, not illustration, as the opening |
| 42 | Nav items as drawings plus word plus three links | Teenage Engineering | Navigation becomes part of identity | Hand-drawn chart symbols as section marks |
| 43 | Poster composition: photo, outlined type, single colour accent | Teenage Engineering | Art-directed moment in a commercial page | One editorial page for the about or season story |
| 44 | Serial number and date on each item | The Pudding | Editorial, archival feel | Notices numbered like Notice to Mariners |
| 45 | Skip beside Start at equal weight | The Pudding | Experts are not held hostage | Any intro has an equal-weight skip |

## 4. What reads as "AI-generated dashboard"

Cross-checked against v11 (`docs/coastwatch/v11/after/`). Frequencies are counted from those PNGs.

| Tell | In v11 (file, evidence) | Contrast in references |
|---|---|---|
| Near-black navy ground with teal, lilac, amber, sky status hues | `theme.css`: `#04090e`, `#0a141c`, teal `#3fd6bd`, lilac `#b39cff`, sky `#74c7ff`, amber `#f0a94a`, coral `#f08a74`. Five semantic hues, all visible in `today-desktop-full` | Memorable, Linear, Motion use 0 to 2 hues; ENC uses 1 foreign hue |
| Semantic hue colliding with the data ramp | Measured teal sits only 23 CIE76 units from the chlorophyll ramp (ramp sampled from the `explore-desktop` legend), so status teal reads as data | ENC magenta is 61 units from the same ramp |
| High-contrast serif headline on black over a darkened map | `today-desktop`: Newsreader place name about 90px over a vignetted map | Felt uses serif on paper; nullschool and Windy do not put a headline over data |
| Map as wallpaper | `today-desktop`, `coast-brief-desktop-day`: map fades under text and sheet, labels collide with text and the bay name sits on data | nullschool, Windy: map is the page, chrome on top is tiny |
| Unboxed KPI row (label, value, subtext x 4) | `today-desktop` Fisheries, Sea, Tests, Bloom; `today-phone` as 2x2 | Windy table; ENC printed numbers; Memorable "one lead figure" |
| Identical section template repeated | `today-desktop-full`: Fisheries, Sea, Toxin tests, Bloom, Keep-alert-ask all use left title, right content | Memorable scenes each have a different composition; Windy gives the table most of the weight |
| Icon-plus-tint pill on every row | `today-desktop`: Delayed, Season closed, Restricted, Status unclear | Windy cells carry value by fill; ENC symbols |
| Row plus chevron plus hairline, four times | `today-desktop-full` rows | Windy tabs, nullschool rows |
| Repeated disclosure rows | `today-desktop-full`: "All forecast periods...", "Every site and sample", "What these are...", "None on record..." | nullschool: one menu, one place |
| Caveat as boilerplate | First fold of `today-desktop`: "Not verified" x 4, "(old)" or "old" x 4, "not toxin" x 2; full page has "not toxin" or equivalent at least 5 times | MyOcean, GFW state caveats once per layer; Memorable's one-home rule. Honesty need remains; the repetition is the tell |
| Same number repeated across the page | `today-desktop-full`: "5.5 of 30 ppm" appears 3 ways; wind and wave appear in the top row and as giant figures | Memorable rule: a number on two screens is a bug |
| Frosted dark pills over the map | `today-desktop` status capsule and legend card; `explore-desktop` bottom capsules | MyOcean's single panel; nullschool's single button |
| Rounded-tile gradient app-icon logo | `today-phone` top left | Felt wordmark; Vercel triangle; Memorable bars |
| One line-icon set for everything | `today-phone` bottom tabs; shield, clock, bell, ellipsis throughout | Teenage Engineering's drawn nav; chart symbols |
| Thin line with dots, dashed threshold, mono axis | `research-desktop`, `coast-brief-desktop` | Windy cells, ENC labels, Memorable hatch |
| Italic serif side-annotations | `research-desktop`, `today-desktop-full` | Admiralty uses italic for a class (water), not a tone of voice |
| Sentence case 13 to 15px everywhere with no real mid-level | `today-desktop-full` meta text (huge title, then 15 to 18 for everything) | Swisstopo's many label classes; Windy's cell/row split |
| Data boundary that looks like a bug | `explore-desktop`: data stops at a hard vertical at about x=683 next to diagonal hatch | ENC and swisstopo end at a neatline; GFW shows gaps as shape |
| Light mode as inverted tokens | `coast-brief-desktop-day`: same composition, swapped values | ENC day, dusk, night are separately specified palettes |

General tells seen across templates (and to check against any v12 draft): bento grids of equal radius; hero with eyebrow, headline, two buttons, logo row (Vercel, Mapbox); mono caps eyebrow on every section (Felt uses it, sparingly); mesh gradient, god-ray or aurora glow (Paper catalogue); text scramble on every heading; staggered fade-up on every load; `rounded-xl` plus hairline plus 12px padding on every element; stat counters animating from zero on arrival; rail plus topbar plus KPI row plus chart plus table in that exact order.

**What the serious references share instead:** an identifiable ancestor outside software (chart, engraving, print, instrument, poster); one dominant element per view; unequal section sizes; colour only where it carries a variable; and at least one idiosyncratic detail that a template would not include (subscript decimals, telemetry strip, split-flap board, drawn nav).

## 5. Type candidates (self-hostable)

All verified in `github.com/google/fonts` `METADATA.pb` as OFL unless stated. Metrics are `OS/2` values at the default master, computed with fontTools: x-height/em, x/cap ratio, default digit behaviour, `tnum` and `zero` feature presence. File sizes are full variable TTF, before subsetting. v11 ships IBM Plex Sans, IBM Plex Mono and Newsreader (`coastwatch-web/src/app/layout.tsx`).

| Family | Weights / axes | Italic | x-height, x/cap | Digits | Features | TTF | Notes |
|---|---|---|---|---|---|---|---|
| Barlow | 100 to 900 static | Yes | .501, .72 | prop | tnum | 99KB | Barlow Condensed also 100 to 900. Designer's stated source is California plates and highway signs (PK). |
| Overpass | wght 100 to 900 | Yes | .511, .73 | prop | tnum, zero | 311KB | Highway Gothic lineage (PK) |
| Archivo | wdth 62 to 125, wght 100 to 900 | Yes | .526, .77 | prop | tnum, zero | 643KB | Condensed labels and wide display from one file |
| Public Sans | wght 100 to 900 | Yes | .517, .72 | prop | tnum | 100KB | US government typeface, neutral |
| Atkinson Hyperlegible Next | wght 200 to 800 | Yes | .496, .74 | prop | tnum | 111KB | Mono sibling (wght 200 to 800, zero). Letterforms built to differ (PK) |
| Schibsted Grotesk | wght 400 to 900 | Yes | .527, .75 | prop | tnum, zero | 171KB | News publisher face |
| Instrument Sans | wdth 75 to 100, wght 400 to 700 | Yes | .510, .71 | prop | tnum | 189KB | Pairs with Instrument Serif; both near Memorable's look |
| Mona Sans | wdth 75 to 125, wght 200 to 900 | Yes | .517, .71 | prop | tnum | 340KB | GitHub family |
| Source Sans 3 | wght 200 to 900 | Yes | .478, .72 | uniform | zero | 631KB | Small x-height; long-form feel |
| TASA Orbiter | wght 400 to 800 | No | .500, .74 | prop | tnum, zero | | Used by Motion |
| IBM Plex Sans (current) | wdth 75 to 100, wght 100 to 700 | Yes | .516, .74 | uniform | zero | 524KB | Mono sibling .516; no change in read if retained |
| Literata | opsz 7 to 72, wght 200 to 900 | Yes | .507, .72 | prop | tnum, zero | 932KB | Book serif with optical sizes; highest x-height of the non-slab serifs measured |
| Source Serif 4 | opsz 8 to 60, wght 200 to 900 | Yes | .475, .71 | uniform | tnum, zero | 1181KB | Optical sizes, Plex-free serif route |
| Bitter | wght 100 to 900 | Yes | .522, .75 | prop | tnum, zero | 320KB | Slab; sturdy on low-res screens |
| Newsreader (current) | opsz 6 to 72, wght 200 to 800 | Yes | .426, .64 | uniform | tnum | 441KB | Lowest x-height in the set |
| Spectral | 200 to 800 static | Yes | .450, .68 | uniform | tnum, zero | 239KB | Small x-height |
| JetBrains Mono | wght 100 to 800 | Yes | .550, .75 | uniform | zero | 182KB | Largest conventional mono x-height |
| Martian Mono | wdth 75 to 112.5, wght 100 to 800 | No | .600, .75 | uniform | | 144KB | Very large x-height; wide |
| Red Hat Mono | wght 300 to 700 | Yes | .488, .70 | uniform | zero | 66KB | Text sibling Red Hat Text |
| Sometype Mono | wght 400 to 700 | Yes | .470, .72 | uniform | | 62KB | Numerals and data feel |
| Geist, Geist Mono | wght 100 to 900 | Yes | .530, .75 | prop / uniform | tnum / none | 165KB/167KB | Memorable, Vercel; widely read as AI-tool default |

Commercial faces observed and unavailable for this licence condition: GT Alpina, Atlas Grotesk, Publico, Söhne, Matter, Berkeley Mono, Acumin.

**Pairings to evaluate** (candidates only, the lead decides):
1. **Barlow Condensed + Barlow + JetBrains Mono.** Regional provenance, condensed labels hold wide numbers on phones.
2. **Archivo (one file, two widths) + JetBrains Mono.** Single family with narrow and wide range; no serif.
3. **Literata (display opsz) + Public Sans + Red Hat Mono.** Serif route with a far higher x-height than v11's Newsreader.
4. **Bitter + Public Sans.** Slab plus government neutral; print chart feeling, low-res robust.
5. **Overpass + Overpass Mono.** Highway-sign lineage, one design family.
6. **Atkinson Hyperlegible Next + Atkinson Mono.** Legibility-first; distinct but quiet.
7. **Schibsted Grotesk + Source Serif 4 for long text.** News-register pair.

**Sunlight and legibility factors.** Model below is effective luminance contrast with screen reflection added: `(L_a*1000 + E*0.045/pi) / (L_b*1000 + E*0.045/pi)`, assuming a 1000 nit phone and 4.5 percent reflectance. It is a model, not a measurement.

| Text on ground | 1k lux | 10k | 30k | 100k |
|---|---|---|---|---|
| v11 dark ink / ink-2 / ink-3 on `#0a141c` | 43 / 26 / 15 | 6.9 / 4.5 / 3.0 | 3.0 / 2.2 / 1.7 | 1.6 / 1.4 / 1.2 |
| v11 light ink / ink-2 / ink-3 on `#f5f7f8` | 42 / 21 / 11 | 7.1 / 6.1 / 5.0 | 3.1 / 2.9 / 2.7 | 1.6 / 1.6 / 1.6 |

- Top-level ink is about equal in either polarity. The gap is in secondary text: at 30k lux, dark-ground third-level text is 1.7 against 2.7 on light. Positive polarity also has literature support for legibility (Buchner and Baumgartner 2007; Piepenbrock et al. 2013, PK).
- Any low x-height face loses first at distance and in glare. Newsreader's .426 is the lowest measured here.
- Numerals decide this product. Check `4/1/7`, `3/8`, `5/6`, `0/O` at 16px, plus units `kt ft ppm`. Prefer `font-variant-numeric: tabular-nums` only in aligned columns.
- On dark ground, avoid weights below 400 (halation, PK). Teenage Engineering's weight 100 and Memorable's 10.08px mono are the failure examples.
- WCAG 4.5:1 is not a sunlight standard. Text that must be read on deck deserves 7:1 or better on its own ground, and a value at size 16 or larger.
- Thin 1px chart lines and hairline dividers disappear under glare; ENC uses solid fills and 1.5 to 2px strokes where they matter. Touch targets for wet or gloved use at 48px minimum (PK).

## 6. Palette territories

Contrast is WCAG 2 relative luminance, computed in `refs/final.py`. "Ramp" is the current chlorophyll ramp sampled from the `explore-desktop` legend (`#102B4E` to `#D5DD5E`). dE is CIE76, smallest distance to any ramp step. Provenance states if a value is sampled, from S-52 data, or derived.

Common finding: the current ramp runs dark to light, which works on a dark ground (top end 12.7:1 on `#0a141c`) and fails on any light one (top end 1.2 to 1.4:1, so the highest values vanish). Every light territory below needs its own ramp, specified separately. The official S-52 tables make the same point: Day Bright, Day Blackback, Dusk and Night are separately defined colour tables (OpenCPN `chartsymbols.xml`), not inversions.

### A. Day ENC (NOAA electronic chart)
Sampled from NOAA's Monterey Bay ENC render; symbol colours from S-52 Day Whiteback.

| Role | Hex | On ground `#DDF1EE` | On shoal / land | Source |
|---|---|---|---|---|
| Ground (deep water) | `#DDF1EE` | | | sampled |
| Shoal | `#9ECEFC` | | | sampled |
| Very shallow | `#82BDFF` | | | sampled |
| Land | `#CEC58A` | | | sampled |
| Built-up | `#B09E56` | | | sampled |
| Ink | `#070707` | 17.2 | 12.2 / 11.5 | S-52 |
| Ink 2 | `#2F4A5B` | 7.9 | 5.6 / 5.3 | derived |
| Contour line | `#616866` | 4.9 | 3.4 / 3.3 | sampled |
| Sounding (texture only) | `#8B9593` | 2.6 | 1.9 / 1.8 | sampled |
| Regulatory magenta (symbol) | `#A83BA6` | 4.7 | 3.3 / 3.1 | S-52 |
| Regulatory magenta (text) | `#8E2C8C` | 6.2 | 4.4 / 4.1 | derived |
| Alert text | `#8A3307` | 7.0 | 5.0 / 4.7 | derived from `#C76B2E` |

dE to ramp: magenta 61, alert 76. Reads as: an actual chart; instantly familiar to a fisherman. Risk: blue and buff feel official-utility.

### B. Engraved Admiralty (paper, black, one overprint)
Sampled from Admiralty scans (No 3520 and No 2941, Wikimedia Commons).

| Role | Hex | On ground `#FFFCE7` | On tone / land / buff | Source |
|---|---|---|---|---|
| Paper | `#FFFCE7` | | | sampled |
| Aged buff | `#FEEFCA` | | | sampled (No 2941) |
| Tone | `#F2ECD9` | | | sampled |
| Mid tone (stipple and land mix) | `#E8E2D0` | | | sampled |
| Ink | `#141310` | 18.0 | 15.7 / 14.3 / 16.3 | derived within sampled ink range |
| Ink 2 | `#5A564B` | 7.1 | 6.2 / 5.7 / 6.4 | derived |
| Overprint | `#6D385E` | 8.6 | 7.5 / 6.8 / 7.7 | sampled |
| Alert | `#A5281B` | 7.0 | 6.1 / 5.6 / 6.3 | derived |

dE to ramp: overprint 31 (muted, so it needs a stronger variant for symbols on data), alert 81. Reads as: archival, authoritative, least SaaS. Risk: nostalgic if line art is not contemporary; colour is near-absent so hierarchy rests on letter style.

### C. Swiss topo (grey relief, class colours)
Sampled from the swisstopo national map colour layer (`ch.swisstopo.pixelkarte-farbe`, zoomed capture). Sampled values are clustered pixels, so tints are approximate.

| Role | Hex | On ground `#F2F5F6` | On pale / forest / rock | Source |
|---|---|---|---|---|
| Paper | `#F2F5F6` | | | sampled |
| Pale | `#EBEEDD` | | | sampled |
| Forest tint | `#C6DFAF` | | | sampled |
| Rock tint | `#BAC5C2` | | | sampled |
| Ink | `#1D2424` | 14.4 | 13.4 / 11.0 / 8.9 | sampled |
| Ink 2 | `#3F4A49` | 8.4 | 7.8 / 6.4 / 5.2 | derived |
| Water | `#457E9D` | 4.1 | 3.8 / 3.1 / 2.5 | sampled |
| Contour | `#614B2C` | 7.5 | 7.0 / 5.7 / 4.6 | sampled |
| Red | `#943433` | 6.8 | 6.4 / 5.2 / 4.2 | sampled |

dE to ramp: water 11 (collides with the chlorophyll ramp), contour 47, red 62. Reads as: precise, cool, engineered. Risk: mountain language, water hue overlaps the ramp.

### D. Tanaka slate (illuminated contours on mid-tone)
Method from Tanaka 1950 (PK). Ground chosen so white and black lines both clear about 4:1; panels are separate grounds.

| Role | Hex | Value | Source |
|---|---|---|---|
| Map ground | `#767C80` | white line 4.2, black line 5.0 | computed |
| Shadow / lit lines | `#000000` / `#FFFFFF` | as above | method |
| Dark panel | `#262B2E` | white ink 14.3, ink 2 `#B8BEC2` 7.6, signal `#FFD24A` 9.9 | derived |
| Light panel | `#E4E7E9` | ink `#262B2E` 11.5, ink 2 `#5B6469` 4.9, alert `#A5281B` 5.8 | derived |

Text is never placed directly on the slate (ink 5.0, white 4.2, sunlight worst of the set: at 30k lux ink 1.5, white 2.3). Signal yellow is 24 dE from the ramp top. Reads as: dimensional, unusual, map-as-object. Risk: mid-grey can look muddy; only as good as the bathymetry.

### E. S-52 black-back (night bridge)
Exact S-52 Day Blackback values (OpenCPN `chartsymbols.xml`).

| Role | Hex | On `#070707` | On `#262B2E` / `#3F5265` / `#436B8C` / land | Name |
|---|---|---|---|---|
| Deep | `#070707` | | | DEPDW |
| Medium | `#262B2E` | | | DEPMD |
| Shallow | `#3F5265` | | | DEPMS |
| Very shallow | `#436B8C` | | | DEPVS |
| Land | `#867B51` | | | LANDA |
| Ink | `#D4EAEE` | 16.1 | 11.5 / 6.4 / 4.5 / 3.4 | CHWHT |
| Ink 2 | `#A3B4B7` | 9.4 | 6.7 / 3.7 / 2.6 / 2.0 | CHBLK in this table |
| Magenta | `#DDADF4` | 10.9 | 7.7 / 4.4 / 3.1 / 2.3 | CHMGD |
| Orange | `#DD7633` | 6.4 | 4.6 / 2.6 / 1.8 / 1.4 | CHCOR |
| Red | `#F15469` | 6.0 | 4.2 / 2.4 / 1.7 / 1.3 | CHRED |
| Green | `#68E456` | 12.3 | 8.7 / 4.9 / 3.4 / 2.6 | CHGRN |

dE to ramp: magenta 57, orange 61, red 84, green 28. The official S-52 Dusk and Night tables are very low contrast by design (Night CHBLK on background 1.26:1, Dusk 1.79:1) to protect dark adaptation; they are not web-usable as text but prove the principle that night means dimming, not just inverting. Reads as: a real bridge instrument, close to v11's dark but with a sourced logic. Risk: nearest to what already reads as generic dark UI.

### Current v11 hues for comparison
`#0a141c` ground: official `#f0a94a` 9.3, model `#b39cff` 8.1, measured `#3fd6bd` 10.2, conditions `#74c7ff` 10.1, down `#f08a74` 7.6. Pairwise CIE76 distances are as low as 35 (official vs down) and 41 (model vs conditions), so hue alone does not reliably separate those pairs at chip size.
