# Integrating the Signature Experience (PR #43) into the coastal redesign (PR #41)

**Date:** 2026-10-13. **Inputs:**
- the owner's decisions on the three Signature questions (A, B, C below);
- PR #43's audit, design language, 12 interaction specs, evaluation and handoff (`design/signature/`, kept there as the reference; nothing from its prototype directory is copied into the app, and no dependency is added).

**Test for every idea:** does it make one coastal question easier to answer, without adding a control, a colour or a claim the data cannot support? Ideas that failed it are listed as rejected, with the reason.

## The owner's decisions, as built

| Decision | What it became |
|---|---|
| **A.** Fold the cream warning bar into a navy masthead status; keep active advisories conspicuous in location briefings; source links and uncertainty accessible | The status line is the second row of the navy masthead: a 3 px amber left rule, the same words, CDFW and CDPH links on every page, "More" for the reasons. The location briefing still leads with the official card (amber rule, not an amber fill) and the answer line names the notices first |
| **B.** Sentence-case labels; the four-marks grammar; text labels beside marks; missing ≠ loading | Every tracked-capital eyebrow is sentence case. Marks (`components/ui/Mark`): rule = official, point = measured (filled within 14 days, ring when older), cells = model, hatch = not known, diamond = conditions. Each mark sits beside words. The map's hatch now waits until a field exists; loading words appear only after 400 ms; a failure says so |
| **C.** Coastal: bathymetric-first, no always-on C-HARM raster; immediate access to the full native field; no cropping; Research has full controls | Coastal opens on the seafloor, coast, ports and official lines. One tap shows C-HARM's full native field (exact 3 km cells, nothing cropped or dimmed). Research opens on the data layer and keeps every control. The "lens" (full strength only around the chosen point) was **not** adopted: it crops the field to a neighbourhood |

## Accepted (built and tested)

| # | Idea (PR #43 reference) | Where | Notes |
|---|---|---|---|
| 1 | Status line inside the masthead (handoff #1) | `Banners`, `AppShell` | One band; phones show "Notices not verified. Check CDFW and CDPH." on one row |
| 2 | No amber card fills; official = an amber rule (P1) | every official card, the drawer, the inspector | Amber again means "an agency said this" |
| 3 | Sentence-case kind labels led by marks (handoff #5) | `Mark`, `Kinds`, all eyebrows | No text under 12 px was added |
| 4 | Freshness without colour (handoff #6, P7) | `FreshnessBadge` | Current = ink, stale/historical = grey, "Not confirmed" = official ink; the glyph's shape carries the state |
| 5 | Answer before controls (handoff #8, P2) | My Coast | Three mark-led lines under the place name: official, measured (with its age in the dot), model or the gap. Activity and species are two selects after it; the forecast day moved beside the forecast |
| 6 | Absences state the limit of the claim (P6) | My Coast, Fisheries | "No notice in CoastWatch's list covers these waters … not the same as safe or open", never in the large serif |
| 7 | Next steps as rows, reason on demand (P8) | `NextSteps` | Five rows; conditions readings are not repeated as steps under the conditions card |
| 8 | Five mobile tabs; About in a masthead menu (P10) | `TabBar`, `MastheadMenu` | The menu also gives phones the Coastal/Research switch on every page |
| 9 | Hatch only where a field exists (P11, the hatch-flash bug) | `MapCanvas` | Regression test holds the grid back and checks the hatch stays off |
| 10 | Loading and failure words for the map field (§11) | `MapCanvas` | "Drawing the data…" only after 400 ms; "could not be loaded. Nothing is drawn in its place." on a source error |
| 11 | Isobaths above data; depth tint cross-fades (handoff #2) | `basemap`, `MapCanvas` | Labels stay under the field so none hides a value |
| 12 | Provenance threads + exact cell outline (handoff #4, §1) | `lib/provenance`, Ocean Map inspector, My Coast | Model cell (or the nearest valued cell), the nearest shore station (fainter beyond 25 km), and the nearest official latitude limit as a ruler when within about 30 km. Threads draw once (520 ms, 90 ms stagger), are never data-coloured, end at the source's real location, and are instant under reduced motion |
| 13 | Coastal journeys, adapted to the device (handoff #10, §2) | `lib/camera` | Distance-scaled flight with a GPU; a 300 ms ease on software WebGL or after a measured median frame over 33 ms; a jump under reduced motion. Any input stops it |
| 14 | Transform-only mobile sheet with the settle spring (handoff #7, §9) | `MobileSheet` | No re-layout while dragging; projected velocity picks the snap |
| 15 | Notices as amber rules on the station chart (handoff #11, P12) | `StationChart`, Bloom | At each active notice's own effective date, with the same facts in the chart's description |
| 16 | Disclaimers that change meaning are sentences (P5) | conditions card | "Wind, waves and tides only: not a toxin or fishing forecast." |
| 17 | Variable Newsreader with optical sizing; warmer paper sheets; motion tokens | `layout`, `globals.css` | — |

## Rejected or deferred, and why

| Idea | Decision | Why |
|---|---|---|
| Lens compositing (full strength only near the point) | **Rejected** | Decision C: it crops the field to a neighbourhood and can mislead about coverage |
| Veil (class colours pre-mixed toward the sea) | **Rejected** | The panel bars, lead strip, nav card and legend would all need the same mixed colours, mode by mode, or a reader matching a colour would be misled. With the field off by default in Coastal (C) and isobaths drawn above it, the published palette stays one set of colours everywhere |
| Class lines ("isobaths of probability") for Research | Deferred | Research already shows exact cell edges from zoom 8.5; another rendering mode adds a control for a small gain |
| One persistent map across Map, My Coast and Blooms (handoff #14) | Deferred | An app-architecture change (a layout-level map) with real risk to tested behaviour; the threads and journeys give most of the geographic continuity within a page |
| Coast-following spline path for journeys | Rejected | MapLibre's interruptible `flyTo`, scaled by distance and gated by the device, gives the context with far less code |
| Map controls rebuilt into two zones (top: search and layers; bottom: time) | Deferred | The Ocean Map's controls already sit in one left column with the map clear; rebuilding the 1,000-line layer dock now risked more than it gained. Listed as a remaining weakness |
| Time bar with separate For and Issued rows (handoff #3) | Partly | The dock already states "for Mon, Oct 12 · issued Oct 12" and a freshness word; a separate Issued row was not added |
| Station chart transitions and the five-step "season story" | Rejected for now | Motion and a story mode on the analysis chart add a mode and code for a page Coastal readers rarely open; the notice rules carry the key relation |
| Swipe comparison between days | Rejected | Two map instances (twice the GPU memory) for a comparison the per-day values already give |
| Satellite pixel thread | Deferred | The satellite reader does not expose the pixel's position yet |
| Purpose onboarding "where the map answers" and the three-stop tour | Deferred | The welcome (#39) already sets defaults by purpose; changing it needs stakeholder input |
| Prototype directory, Vite build, d3 | Not copied | Kept in PR #43 as the reference; no new runtime dependency |

## Measurements

See the [readiness record](readiness.md) for the design passes, screenshots and remaining weaknesses, and the performance table below.

## Performance (production builds, production data, 1440 × 900; before = #41 at `9349aefa`)

Measured on an M1 Max under heavy background load (load average ~23), so treat small differences as noise.

| | Before | After |
|---|---|---|
| **GPU** start-up long tasks / blocking time | 0 / 0 ms | 0 / 0 ms |
| GPU flight frame time (p50 / p95) | 17 / 17 ms (60 fps) | 17 / 17 ms (60 fps) |
| GPU: click on water → panel shown | 62 ms | 69 ms |
| **Software WebGL** (SwiftShader) start-up blocking time | 1,084 ms (6 long tasks, max 483 ms) | 1,103 ms (7, max 677 ms) |
| Software WebGL flight | 20 fps (p50 50 ms) | 20 fps → **the app now uses a 300 ms ease instead** |
| Software WebGL: click on water → panel | 154 ms | 166 ms |
| Software WebGL + CPU 4×: blocking time | 1,416 ms | 1,514 ms |
| Phone sheet drag, CPU 4× (p50 / p95 frame) | 17 / 17 ms | 17 / 17 ms |

- **What the start-up long tasks are:** not the relief images. Replacing every relief image with a 1-pixel image left software-WebGL blocking unchanged (about 1.1 s in all three variants, three runs each). The cost is MapLibre's initialisation and first renders on a CPU-emulated GPU; the integration adds none. With a GPU there are none.
- **What changed for weak devices:** journeys are chosen per device. The renderer is read once (SwiftShader, llvmpipe and "software" renderers get the short ease), and a flight that measures a median frame over 33 ms downgrades the rest of the session.
- **The transform-only sheet** showed no measurable gain on this machine (60 fps both ways at CPU 4×); it removes per-frame layout, which matters most on slower phones than this one.
- Scripts: `perf.mjs`, `startup.mjs`, `sheet.mjs` in the session record; not part of the app.

## Matched screenshots (production data; pre-integration = #41 at `9349aefa`, integrated = this branch)

| View | Pre-integration | Integrated |
|---|---|---|
| Ocean Map, Coastal default | [before](img/pre-integration-d-map.jpg) | [after](img/integrated-d-map.jpg) |
| Ocean Map, forecast on | [before](img/pre-integration-d-map-forecast.jpg) | [after](img/integrated-d-map-forecast.jpg) |
| Ocean Map, a chosen point (threads) | [before](img/pre-integration-d-map-point.jpg) | [after](img/integrated-d-map-point.jpg) |
| My Coast, Monterey | [before](img/pre-integration-d-coast-monterey.jpg) | [after](img/integrated-d-coast-monterey.jpg) |
| Bloom, Research | [before](img/pre-integration-d-bloom-research.jpg) | [after](img/integrated-d-bloom-research.jpg) |
| Fisheries | [before](img/pre-integration-d-fisheries.jpg) | [after](img/integrated-d-fisheries.jpg) |
| Notices drawer | [before](img/pre-integration-d-drawer.jpg) | [after](img/integrated-d-drawer.jpg) |
| Phone map | [before](img/pre-integration-m-map.jpg) | [after](img/integrated-m-map.jpg) |
| Phone map, a chosen point | [before](img/pre-integration-m-map-point.jpg) | [after](img/integrated-m-map-point.jpg) |
| Phone My Coast, Monterey | [before](img/pre-integration-m-coast-monterey.jpg) | [after](img/integrated-m-coast-monterey.jpg) |
| 320 px My Coast, Santa Cruz | — | [after](img/integrated-s-coast-santa-cruz.jpg) |
