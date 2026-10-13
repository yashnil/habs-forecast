# 06 — Handoff to the redesign

**For:** the session that owns `feat/coastal-redesign` (PR #41).
**From:** branch `design/signature-experience`.
**Nothing in `coastwatch-web/` was changed.** Everything below is a proposal to adopt once your architecture stabilises.

**Your settled decisions are kept:**
- mode ids and storage (`coastal` / `research`, `cw-mode-v1`, `?mode=`);
- the kind colours (official amber, conditions teal, measured green, model violet, gap ink-3 grey);
- the freshness words;
- tint only with no data raster;
- no green all-clear;
- proximity never shown as a notice applying;
- "not a toxin or fishing forecast" on conditions;
- 320 px and axe clean.

## Ranked roadmap (highest impact first)

| # | Change | Impact | Effort | Depends on | Status |
|---|---|---|---|---|---|
| 1 | **Fold the cream status bar into the masthead** (one navy band, amber left rule, same words and links) and **stop using amber backgrounds** for cards | Removes 80 px on phones; amber regains its meaning (audit P1, P2) | S | Nothing | Integrate now |
| 2 | **Forecast drawing: veil pre-mix, isobaths and coastline above data, dip-through-basemap day swaps** | The field stops flooding the map; no blended days (P4, P9) | S–M | The pipeline PNG can pre-mix, or render client-side as in `map/field.ts` | Integrate now |
| 3 | **Time bar with separate "For" and "Issued" rows** | Removes the issue-vs-valid confusion | S | LeadStrip | Integrate now |
| 4 | **Provenance threads plus exact cell and pixel outlines** on selection | The signature interaction; it shows where every statement comes from | M | Selected point in MapCanvas; `associationsAt` tiers | Integrate now |
| 5 | **Sentence-case kind labels led by marks** instead of uppercase eyebrows; Newsreader variable (`opsz`) | Readability, wrap at 320, identity (P5) | S | A tweak to the settled card pattern: your call | Integrate now if you agree |
| 6 | **Freshness without colour**; measured dot filled vs ring by age | No green near toxin values (P7) | S | FreshDot, Badges | Integrate now |
| 7 | **Mobile: custom transform-only sheet, five tabs, controls riding above the sheet** | One-handed map (P2, P10) | M | MobileSheet, tab bar | Integrate now |
| 8 | **Answer before controls**: Level 1 sentence under the place name; plan controls after it | First screen answers (P2) | M | Briefing (in flux) | After Briefing settles |
| 9 | **Four-marks ledger row anatomy** for official / measured / model / not known | Consistent progressive disclosure | M | Briefing, CoastSections | After Briefing settles |
| 10 | **Coastal journeys** for place changes | Geographic continuity | S | PlacePicker / map routing | Integrate now |
| 11 | **Station chart transitions plus notice rules plus season story** | Bloom page becomes explanatory | M | StationChart; add `d3-scale` / `d3-shape` | Integrate now |
| 12 | **Loading system**: reserved layouts, per-row loading words after 400 ms, saved-copy labelling, calm basemap until a field exists | Perceived performance, honesty (P11) | M | Data hooks | Integrate now (incrementally) |
| 13 | **Purpose onboarding where the map answers**, plus the three-stop tour | First-visit delight | S | Welcome (#39) | After review |
| 14 | **One persistent map across Map, My Coast and Blooms** | Camera continuity between pages (P3) | L | App architecture (layout-level map) | Depends on the redesign architecture |
| 15 | Lens compositing mode | Focus on the reader's water | M | Client field rendering | Exploratory, with the dimmed outside always kept |
| 16 | Swipe comparison between lead days | Low; the time bar and per-lead table cover it | L | Two map instances | Exploratory, not recommended |

## Component map (prototype → coastwatch-web)

| Prototype (`prototypes/src/…`) | Target | Notes |
|---|---|---|
| `styles/tokens.css` | `app/globals.css` | Same `--cw-*` names; new: the `--cw-map-*`, `--cw-p0..9` and motion tokens, and `.mark-*` glyphs (`styles/base.css`). Map the tokens into Tailwind 4 `@theme` if you want utilities |
| `map/field.ts` | `lib/` (new) and MapCanvas | Pure functions: `renderField`, `classEdges`, `footprint`. Takes the published grid; no new dependency. Add a `base` colour parameter for light maps (the Directions worker copied it for that reason) |
| `map/threads.ts` | MapCanvas | Needs the map instance (`useMap().current.getMap()` in react-map-gl) |
| `map/journey.ts` | MapCanvas / PlacePicker | Pure; takes a map and a coast path |
| `map/style.ts` | `lib/basemap.ts` | Changes: isobaths above data (`BEFORE_DATA = "isobaths"`), land −35 % saturation, tint cross-fade via `raster-opacity-transition` |
| `map/MapStage.tsx` | Reference only | Shows the hold-and-swap A/B image-source pattern; your MapCanvas should own this |
| `ui/Sheet.tsx` | `components/MobileSheet.tsx` | Drop-in shape; `bottom` is the tab-bar height |
| `ui/TimeBar.tsx` | `components/map/LeadStrip.tsx` | Props are the run, leads and freshness |
| `ui/Ledger.tsx` | Briefing / CoastSections | Adopt the anatomy, not the file: it predates your Briefing logic |
| `lib/provenance.ts` | `lib/` | `modelReading`, `stationReading` and `satelliteReading` are new. **Use your `associationsAt` for notices**; the prototype's `noticeGeometry` is a simplified mirror |
| `data/store.ts` `gunzipIfNeeded` | `lib/grid.ts` | Some static servers (Vite dev) send `.gz` with `Content-Encoding: gzip`; check the magic bytes instead of assuming |
| `charts/*`, `pages/Station.tsx` | `components/charts/StationChart.tsx`, Bloom | Replace `charts/scale.ts` with d3-scale/d3-shape |

## Libraries (from [`research/02-library-research.md`](research/02-library-research.md))

| Need | Use | Avoid |
|---|---|---|
| Enter and exit, popovers, dialogs | CSS `@starting-style`, `allow-discrete`, `linear()` | — |
| Springs | Pre-generated `linear()` tokens | Motion's `motion.div` (41 kB). Use `m` with LazyMotion only if a gesture needs velocity |
| Route transitions | `document.startViewTransition` behind a feature check, never around the map canvas | React `<ViewTransition>` until React 19.3 and a Next release without the flag |
| Map camera, fades, hover | MapLibre `flyTo`/`easeTo`/`fitBounds` with padding, `stop()`, `originalEvent` checks, `raster-opacity-transition`, feature-state | — |
| Primitives | Native elements; Base UI for new complex widgets | vaul (unmaintained), react-modal-sheet (no accessibility) |
| Charts | d3-scale and d3-shape (about 14 kB) | Recharts (112 kB), GSAP (about 45 kB, proprietary licence) |

## ⚠ Animation patterns that would mislead — never do these

1. **Never tween between two forecast days or two observations.** Interpolated colours are invented data. Swap through the basemap ("dip"), not by cross-fading.
2. **Never animate flow, drift or "bloom movement"** from C-HARM or chlorophyll. Neither field is a velocity, and HF-radar currents are separate observations with their own times.
3. **Never draw water, wave or shimmer effects** on the sea. On a scientific map, any movement on the water reads as an observation.
4. **Never move a value onto the reader's point.** The nearest cell or sample stays where it is; a thread connects it.
5. **Never smooth or contour between cells** (no bilinear resampling, no marching squares). Class lines follow exact cell edges.
6. **Never map alpha to probability.** A low probability then looks like no data, or like safe.
7. **Never let the loading state look like a gap.** No hatch until a field exists, and no skeleton that resembles a value.
8. **Never animate to conceal failure.** Spinners and shimmer must not run in place of an error; say what failed.
9. **Never use motion or colour that implies "all clear"**, such as a green tick, a calm fade-out of notices, or a celebratory transition when no notice is found.
10. **Camera journeys must be interruptible** and never imply travel time or routing. They are context, not navigation.

## Coordination notes

- Ports used here: 3460 (dev), 3461 (preview). Your 3200–3203 and 3420–3424 were never touched.
- The SN26-018 discrepancy is shown with a **labelled fixture** in PR #40's exact `conflicts` format, because the pinned production snapshot predates that pipeline change. No record is edited or marked lifted.
- The data snapshot is pinned to `coastwatch-data@0353c1f3` (run 38328551760) and is git-ignored; `npm run snapshot` reproduces it.
