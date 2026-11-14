# D. Motion, shaders, graphics and geospatial feasibility (v12 research)

Worktree read at `c02f460cb`. Paths are relative to `coastwatch-web/` unless noted. Spikes are throwaway, outside the repo, in the session scratchpad folder `spikes-D/`.

## 0. How to read the numbers

**Measured** means run this session: Chrome 157 (Apple M1 Max, Metal), Playwright WebKit 27.2 (desktop), Firefox 157, against production builds of Next 16.4.2, React 19.3.0, Motion 12.43.0, MapLibre 6.13.0. **SwiftShader** is Chrome on software GL on that same CPU: good only for comparing scenes with each other, neither a floor nor a phone estimate. **Estimate** and **not tested** are said where they apply.

**No real phone was tested.** Nothing here measures a mid Android GPU or iOS Safari. Playwright WebKit is desktop WebKit. M1 Max results show no pathology and say little else: every GPU run sat at the 16.7 ms vsync cap. Section 12 names the device checks that have to pass before the opener is committed.

## 1. Answers first

| # | Question | Recommendation |
|---|---|---|
| 1 | One map or one per page | **One persistent MapLibre instance in a route-group layout.** Pages declare a scene; they do not own a map. Proved across routes in three engines. |
| 2 | Shared element, port to briefing | **React `<ViewTransition>`**, not the hand-rolled `startViewTransition` in `src/v4/motion/routes.tsx`. The hand-rolled one freezes the page for the whole network wait (850 ms measured). |
| 3 | Map continuity | The camera is the transition. Never snapshot or stretch the canvas. |
| 4 | Canyon opener | **MapLibre terrain plus hillshade** from a baked CRM Terrarium set: no new JS, same canvas and camera into the live coast. Gated, once only, and conditional on a phone test. |
| 5 | Currents | Keep `FlowLayer` and its CPU simulation. Improve the drawing and add a real per-cell quality signal. No GPU advection. |
| 6 | Paper Shaders | **Not on or near the ocean.** The fluid shaders look like measurements. |

## 2. What exists today

- **Map:** `src/components/map/MapCanvas.tsx` (1154 lines), created per page through `OceanStage`/`StageScene`, `HarbourMap` and `Explore`. Each route makes a new WebGL context. `window.__cwMap` has 130 references across `src` and `tests`, so it must keep working.
- **Layout:** one root layout, no nested layouts, each page renders its own `TopBar`.
- **Route motion:** `src/v4/motion/routes.tsx` intercepts link clicks and runs `startViewTransition` with `ROUTE_WAIT_MS = 900`. `.nx-vt-stage` names the stage `nx-stage`, which is a stretched snapshot and the source of v11's "not a true morph".
- **Poster:** `StagePoster`, inline SVG of real Monterey isobaths (`MARK_FIELD`), zero requests.
- **Capability checks:** three separate ones. `stageCanRun` (`src/v4/stage/fallback.ts`), `cameraTier` (`src/lib/camera.ts`), `flowFps` and the slow-frame check (`FlowLayer.tsx`).
- **Context loss:** only `StageScene` listens, and it falls back to the poster permanently. Explore has no handler.
- **Bathymetry in the repo** (`pipeline/scripts/build_relief.py`, `public/basemap/relief.json`): statewide pre-shaded `relief-*.webp` from ETOPO 2022 at 15 arc-second (about 450 m, 59–145 KB); a Monterey inset `relief-sea-monterey*.webp` from NOAA CRM vols 6 and 7 at 3 arc-second (about 90 m, 21–36 KB); `isobaths.geojson` from ETOPO (5 depths, 3,240 points, 66 KB); `MARK_FIELD` from CRM (inline).

There is **no gridded relief in the shipped app**: only pre-shaded images and coarse lines. `build_relief.py` reads CRM over OPeNDAP, so the source is one step away. Licence and attribution are already handled (`RELIEF_ATTRIBUTION`; not for navigation).

## 3. The persistent map (the main architecture question)

### Recommendation

Mount one MapLibre instance in `app/(ocean)/layout.tsx`, covering Today, Explore and My Coast. A route group changes no URLs. App Router keeps a layout mounted across navigation between its children, so the canvas is never rebuilt.

Research keeps its own small station map, and leaving the group frees the context, which is what we want on a phone.

### Measured (Chrome figures; WebKit and Firefox matched unless noted)

| Check | Result |
|---|---|
| Canvas element survives `/a → /b → /c` | Same element, 1 map init, 1 map context |
| Camera follows the route | Reached the page's `center` and `pitch` |
| First camera movement after tap (server delay 120 ms) | 128–135 ms, so roughly 10 ms after commit |
| Flight frames | Median 16.7 ms; 17 ms in WebKit |
| Per-page comparison, tiny style | New canvas and context each route, 252 ms from tap to idle (WebKit 218, Firefox 331) |

The per-page figure flatters per-page: the spike style has four layers and local tiles. The real `MapCanvas` adds rasters and `cwgrid://` decoding. **I did not measure the real app's re-init cost.**

### How it works

```
app/(ocean)/layout.tsx          server: poster + <OceanHost/> + {children}
  OceanHost (client, ssr:false) one MapCanvas, position:fixed, inset:0, height:100lvh, z-index:0
  {children}                    page HTML above it, z-index:1
page → <MapScene .../>          client, renders null, writes the scene to a module store in useLayoutEffect
```

1. **A scene is data:** `{ center, zoom, pitch, bearing, padding, field, currents, marks, lines, focus, interactive }`. `MapScene` replaces `OceanStage` and `HarbourMap`. `OceanHost` subscribes and applies the difference.
2. **Writing in `useLayoutEffect` starts the camera in the same commit as the new page.**
3. **The canvas never resizes between routes.** Today's 68svh hero and Explore's full bleed differ by camera `padding` and by how much opaque page covers the map. "The hero grows into Explore" becomes the sheet leaving while `easeTo({ padding })` runs. No canvas reallocation and no stretched snapshot.
4. **Use `100lvh` and set `trackResize: false`**, resizing only on real orientation or width changes, so the iOS address bar does not reallocate the drawing buffer each time it collapses. Not tested on iOS.
5. **Data still comes from the server.** Each page's `loadStageData` result passes through `MapScene` props.
6. **Interrupts:** a newer scene replaces the flight. MapLibre already stops a flight on input.
7. **Covered map:** when the sheet fully covers it, pause `FlowLayer` (it already pauses on `visible`). An idle MapLibre does not redraw.
8. **Offline:** unchanged. `stageCanRun` still returns false for a saved copy, so `OceanHost` is not loaded and the inline poster stands.

### Risks

- **Explore is the largest migration** and I did not audit how it drives `MapCanvas`. Move Today and My Coast first, Explore last. In pilot mode `/` redirects to Today, so `app/page.tsx` moves into the group unchanged.
- `.nx-vt-stage`, `motion.spec.ts`, `splash.spec.ts` and the "one canvas on Explore" check will need updating. Keep `window.__cwMap`.
- A long-lived map holds its tile cache: set `maxTileCacheSize` on phones.
- `StageScene`'s once-per-map arrival logic becomes per-scene.

## 4. Shared elements: View Transitions and Motion

Support (caniuse-lite 1.0.30001816): same-document View Transitions in Chrome 111, Safari and iOS 18.0, Firefox 144, Samsung 23, about 92.9% global. Transition types present in all three engines tested.

### Measured: the freeze

The route takes 800 ms on the server, which is an ordinary wait on a poor connection.

| Approach | Longest frame gap, Chrome | WebKit | Note |
|---|---|---|---|
| No transition | 33 ms | 22 ms | Old page stays live |
| Hand-rolled `startViewTransition` (the pattern in `routes.tsx`) | **850 ms** | **816 ms** | The callback holds rendering across the network wait; the map freezes too |
| React `<ViewTransition>` in App Router | 133 ms | 18 ms | The transition started at 812 ms, at commit |

**This is the most useful finding for the people we serve.** Today's wrapper can freeze the app for up to 900 ms on any navigation over a slow connection. React 19.3 exports `<ViewTransition>`, and the bundled Next 16.4 guide says it needs no configuration.

### Motion `layoutId` across routes

| Check | Result |
|---|---|
| `/a → /b → /c`, force-dynamic server pages, Chrome, WebKit, Firefox | **Morphs in all three**: 29–59 distinct intermediate widths |
| Back button | Morphs, with the same jump when the destination restores its scroll |
| Added bytes | None. The app imports the full `motion` component (43 KB gz measured, `domMax` 41 KB). |
| Source page scrolled 700 px | **Breaks.** The element jumps about 520 px at the start: Next resets scroll in the same commit and the morph starts from the wrong place |

### Choice

| Need | Use |
|---|---|
| Port to briefing, across routes | React `<ViewTransition name=...>` |
| Interruptible motion inside one page (row to sheet, thumb glide) | Motion, as now |
| The map | Neither: the camera |

- **Do not let the root snapshot capture the map.** Otherwise a frozen image of it cross-fades over the live one. With `html { view-transition-name: none }` and only `main` named, MapLibre kept rendering through the transition (5 frames at 250 ms, 21 at 500 ms; 8 and 23 in WebKit).
- A View Transition cannot be interrupted, and a second one aborts the first (the `AbortError` appeared in WebKit and Firefox). Catch it, as `routes.tsx` does today.
- `<ViewTransition>` will not read `mayTransition`'s conditions (splash, saved copy, offline). Those need a CSS gate on a root attribute. Not spiked.
- Avoid `AnimatePresence` page exits in App Router. Long-known to be unreliable because the outgoing tree receives the new router context. Not spiked.

## 5. Monterey Canyon opener

### Data: use CRM

| Source | Resolution | Status |
|---|---|---|
| **NOAA CRM vols 6 and 7** | 3 arc-second (about 74 × 93 m here) | **Use.** Fetched this session: 1441×1381 cells, -3330 to 1539 m. Public domain per `build_relief.py`; already attributed. |
| ETOPO 2022 | 15 arc-second | Already the statewide base; too coarse for the canyon |
| GEBCO | 15 arc-second | No gain. Terms not checked this session. |
| CUDEM 1/9 arc-second | about 3 m | Unnecessary for a 4–5 s scene. Catalog licence records are inconsistent, and I did not confirm a Monterey tile. |

### Baked sizes (measured, Terrarium, Web Mercator)

| Grid | Cell | PNG, 1 m steps | WebP lossless, 1 m | WebP lossless, 5 m |
|---|---|---|---|---|
| 256 | 383 m | 76 KB | 53 KB | 35 KB |
| 512 | 192 m | 256 KB | **175 KB** | 110 KB |
| 768 | 128 m | 511 KB | 343 KB | 208 KB |
| 1024 | 96 m | 825 KB | 553 KB | 328 KB |

- Lossless WebP beats PNG by about 30%. Lossy encoding is not usable: it corrupts decoded height.
- Naive gzipped 16-bit grids are bigger (256²: 123 KB; 512²: 476 KB). A real decimated quantized mesh might win but needs tooling and a loader. Not worth it here.
- Error against the 1024 grid: at 512, median 1.4 m, p95 18 m, max 237 m on canyon walls. Fine as a picture; do not quote a depth from it.
- MapLibre tile pyramid, 512 px tiles: z6–9 is 17 tiles, 762 KB; z6–10 is 2.28 MB. This camera needed 11 tiles, about 620 KB at 1 m. At 5 m I estimate about 390 KB (scaled from the single-image ratio).

### Renderer comparison

| Option | Added JS (gz, measured) | Into the live coast | Verdict |
|---|---|---|---|
| **MapLibre terrain + hillshade + color-relief** | 0 | Same canvas and camera, no hand-off | **Build with care** |
| Custom heightfield, standalone WebGL2 | about 3–5 KB (estimate) | Needs a second context and camera matching | Fallback only |
| Custom heightfield as a MapLibre custom layer | about 3–5 KB (estimate) | Same camera | Plan B if phones struggle |
| three.js | 132 KB minimum tree-shaken | Separate context and camera | No |
| deck.gl with `TerrainLayer` | 293 KB | Interleaves, but a second engine | No |

### Frames, 412×915 viewport, 4.5 s move from pitch 68 to flat

| Scene | M1 Max (DPR 2–3) | SwiftShader, DPR 1: median / p95 |
|---|---|---|
| Hillshade + color-relief, no terrain | not run | 16.7 / 16.8 ms |
| Terrain, fixed exaggeration, no color-relief | not run | 16.7 / 83 ms |
| Terrain, fixed exaggeration, color-relief | 16.7 / 16.7 ms | 33 / 183 ms |
| Terrain with `setTerrain` called every frame | 16.7 / 16.7 ms | **583 / 1567 ms** |
| Custom heightfield, 256² mesh, 512 DEM | 16.7 / 16.8 ms | 16.7 / 33 ms |
| Custom heightfield, 128² mesh | not run | 16.7 / 16.8 ms |

The custom mesh costs about 1 ms of JS a frame or less. Terrain figures include 11 tiles loading once.

### What the spike showed

- Negative elevations work. The canyon renders correctly and reads immediately as Monterey. Flat at the end, it looks like a finished map.
- **Never call `setTerrain` per frame.** It is the costliest thing measured. Step it 4–6 times in the last 600 ms, or drop terrain at pitch 0. Neither was tested for a visible pop.
- **The DEM's edge smears into long streaks** at the horizon, because edge values are clamped. Bake a wider extent or hide the horizon with `sky` fog.
- At ×4 exaggeration the Santa Cruz Mountains look absurd. Flatten land in the bake, or exaggerate sea only.
- **Say "Vertical exaggeration ×4" on the scene.** Depth exaggerated without a label misdescribes the seafloor.
- The custom mesh looked blocky with `NEAREST` sampling. Linear float textures need `OES_texture_float_linear`, so it would need manual filtering.

### The sequence

1. Server HTML and the inline SVG poster paint first. Nothing waits on the opener.
2. Tier full, online, first session only: load MapLibre and the DEM set after first paint.
3. Show a **baked still from the opening camera** as the bridge (measured WebP: 31 KB at 824 px q60, 24 KB at q40, 12 KB at 412 px).
4. Everything ready within about 1.5 s: run the terrain scene, 4–5 s, any input skips it. Otherwise go straight to the live flat map. Do not make someone wait for a film.
5. The camera eases to the port's scene, terrain drops, currents start. `StageScene` already holds currents until the camera settles, and that matters: `FlowLayer` draws at z=0 and would float off exaggerated terrain.

**Offline:** the opener never runs. The DEM set stays out of field and Today packs; the poster and the 7.0 MB Monterey pack are unchanged.

**Reduced motion or lower tiers:** the still, or the poster, then the settled map.

The opener competes with the existing splash. One of them must go; that is a creative-director decision.

## 6. Currents from HF radar

`FlowLayer` and `src/lib/flowSim.ts` are already honest: nearest cell, one frozen hour, a particle dies entering an unobserved cell, a new hour is a cut, footprint fill shows coverage, still frame under reduced motion. Up to 900 particles (360 on weak devices), instanced quads in the map's own context. The fixture Monterey grid is 41×44 cells at 2 km, 1,804 values.

**Do not move to GPU advection (windgl, nullschool style).** The grid is small enough that JS costs little. Tens of thousands of particles on 2 km cells would suggest detail the radar does not have. Screen-space fade buffers also have to reset on every pan, and float render targets are uneven on phones.

What would make it better while staying truthful, in order:

1. **Tapered trails.** Width and alpha falling along the trail. Today alpha carries it alone.
2. **Additive blending with a soft ceiling**, so dense flow glows rather than clips.
3. **Opacity from real data quality.** The pipeline already gates on `number_of_sites` and HDOP (`verify_currents.py`, `cur.MIN_SITES`, `cur.HDOP_MAX`) but does not publish them per cell. Publish a one-byte quality grid per hour. **It must be a sidecar file:** the demo and deployed web reject unknown manifest fields.
4. **A hairline at the real coverage edge**, not a soft fade, so "no streaks" reads as "no observation".
5. **Playing the last 24 observed hours** as labelled cuts, using the same rule as the time rail.

**Decision for the creative director:** nearest-cell sampling makes visible kinks at cell edges when zoomed in. Bilinear blending between adjacent observed cells would look better but breaks a rule pinned in `flowSim` and tests. I would keep nearest at harbour zoom. If it changes, it needs a label and must never cross a gap.

## 7. Paper Shaders

| Fact | Value |
|---|---|
| Licence, version | Apache-2.0, 0.0.82 (pre-1.0, updated 2026-11-08) |
| Size | All: 64.6 KB gz. Mount alone 4.9 KB; mount plus `water` 6.3 KB. No dependencies. |
| Requirement | WebGL2, **one context per mount**, beside the map's |
| Catalog | 29 shaders, 20 with a time uniform |

| Shaders | What they would say | Verdict |
|---|---|---|
| `water`, `waves`, `warp`, `swirl`, `perlin-noise`, `simplex-noise`, `smoke-ring`, `neuro-noise`, `gem-smoke`, `god-rays` | Flow, swell, plumes, turbidity | **Banned here.** Next to real currents and chlorophyll, a reader cannot tell them from data. Breaks v11's "Particles move only where water moves" and "Real geometry or nothing". |
| `mesh-gradient`, `static-*-gradient`, `grain-gradient`, `paper-texture` | Atmosphere | Conflicts with v11's "no gradients for decoration, no noise textures". A palette ramp over water also reads as a data legend. |
| `metaballs`, `voronoi`, `spiral`, `dot-*`, `liquid-metal`, `pulsing-border`, `fluted-glass`, `lens-distortion`, `color-panels` | Nothing relevant | No |
| `halftone-dots`, `halftone-cmyk`, `dithering`, `image-dithering` | A printed-chart texture **applied to a real image** | Only as a **build-time** render to a static file, for example on share cards. No runtime dependency. |

## 8. Tanaka and responsive contours

| Approach | Cost | Verdict |
|---|---|---|
| **Baked Tanaka WebP** in `build_relief.py`, beside `relief-sea-*` | Zero runtime. Size likely similar to the existing 21–36 KB inset (estimate). `packs.ts` would need the file added. | **Build.** Static, offline-capable, real geometry. |
| Vector contours (contourpy, Monterey inset only) | Measured: 100 m 146 KB gz, 50 m 288 KB gz | No, too heavy statewide |
| `maplibre-contour` (BSD-3, 0.1.1) | Contours from DEM tiles in a worker | Needs DEM at runtime, cannot light by aspect |
| **Fragment shader on the DEM** (custom layer) | One uniform per frame. The spike did contours, Tanaka and cursor proximity in about 25 lines, at the heightfield's frame times above. | **Build with care**, desktop first |

- Cursor or scroll response must only reveal what is real: light the isobath at the depth under the pointer, or reveal contours by depth as the page scrolls.
- Update **uniforms only**, never `setPaintProperty` per frame.
- A depth readout must say "approx." and "not for navigation", and read from a grid finer than 512.
- On phones there is no cursor. Scroll-reveal is the only variant, and only on tier full.

## 9. Charts

**Keep hand-rolled SVG.** `LinkedSeries.tsx` (660 lines) and `BrushStrip.tsx` (229) already do fixed decade axes, gaps, a reported-zero treatment and linked brushing. The fixture holds about 11.7k rows across 17 stations and all variables, so any one view has a few thousand points at most.

| Library | Measured gz | Verdict |
|---|---|---|
| d3-scale + d3-shape | 16.5 KB | Only if axes grow. Saves little. |
| d3-brush + d3-zoom | 20.2 KB | No. The brush exists. |
| visx scale, brush, shape | 21.3 KB | No. It fights the honesty rules' custom axes. |
| uPlot (canvas) | 23.0 KB | Keep in reserve for a view over about 10k points, such as all 17 stations overlaid. |

Brush cost at research scale was **not measured**. If a drag lags, throttle to one update per frame and translate a clipped group during the drag, redrawing on release.

## 10. Scroll-linked storytelling and parallax

- The `.nx-scroll-stage` pattern (CSS `animation-timeline`, opacity and transform only) is the right one: compositor-run, no JS. Safari added scroll-driven animations in 26.0. Confirmed available in WebKit 27.2 and Chrome 157, and **absent in Firefox 157**. Without it the page is static, an acceptable fallback. Keep to `opacity` and `transform`; no `backdrop-filter` over WebGL (v11 already bans it).
- **Do not scrub the map camera from scroll on phones.** Each step repaints the whole map, and iOS scroll is decoupled from the main thread, so the map lags the finger. Not measured on a device; I would not ship it without one.
- On desktop, tier full, allow at most one scroll-scrubbed camera passage.

## 11. GPU tiers, reduced motion and context loss

Combine the three existing checks into one `capability()` module and one session key.

| Tier | Entry condition | Gets |
|---|---|---|
| **Full** | WebGL2, hardware renderer, cores > 4 and memory > 4 GB where reported, not `saveData`, connection 4G or better | Opener once; flights; currents up to 900 at 60 fps; `pixelRatio` capped at 2 |
| **Reduced** | Cores or memory ≤ 4, 3G, or downgraded by measured frames | No terrain; 300 ms camera; currents up to 360 at 30 fps; `pixelRatio` capped at 1.5 |
| **Static** | No WebGL2, `saveData`, offline or a saved copy, or repeated context loss | Server HTML and poster. **The MapLibre chunk is never fetched.** |

- **Downgrade on evidence, once, and never flap.** Reuse existing thresholds: median flight frame over 33 ms (`camera.ts`), map frame over 8 ms (`FlowLayer`), eight-frame median over 125 ms.
- `deviceMemory` is missing on Safari, so never use it alone. Keep the software-renderer check from `camera.ts`. Avoid `detect-gpu`: a benchmark table download for a guess real frames answer better.
- **Do not plan on WebGPU.** MapLibre 6's `contextType` is restricted to `webgl2`, and WebGL2 is at about 97.1% global.

**Reduced motion** is independent of tier. Existing rule stays: fades stay, nothing travels, nothing loops, the camera jumps, currents are one still frame, no opener, and transitions off.

**Context loss (measured in Chrome and WebKit with `WEBGL_lose_context`):** MapLibre restored the style and terrain, but **custom layers do not come back**: `onAdd` did not rerun and the custom layer rendered 0 times after restore. The MapLibre source warns of this. So `FlowLayer` silently vanishes unless re-added.

Required handling:
1. On `webglcontextlost`, show the poster on the next frame.
2. On `webglcontextrestored`, re-add custom layers and restore the camera.
3. A second loss in one session drops to static.

`StageScene`'s permanent fallback becomes wrong once the map persists, because the map must survive a backgrounded tab.

## 12. Performance budget, mid Android (about Pixel 6a / Galaxy A54, slow 4G)

| Budget | Target | Today |
|---|---|---|
| First-load JS, Today | **≤ 240 KB gz** | 231 KB (measured, worktree build) |
| Map JS, lazy, after first paint | ≤ 430 KB gz | 285 KB main + 144 KB worker (measured) |
| New runtime dependencies in first load | **0** | 0 |
| LCP | ≤ 2.5 s, from server text and inline SVG, no WebGL in the path | Not measured |
| Longest route-tap freeze | **≤ 100 ms** | **Up to 900 ms** by design (850 ms measured in the pattern) |
| Opener data | ≤ 400 KB, cancelled at 1.5 s | New |
| Canvas | `pixelRatio` ≤ 2 (full), ≤ 1.5 (reduced) | No cap found in `MapCanvas` |
| Flight frames | Median ≤ 33 ms, else downgrade | Existing check |
| Terrain frames | Median ≤ 33 ms over the opener, else off for good | New |
| GPU memory | Estimate ≤ 150 MB, tile cache bounded | Not measured |
| Offline | Today pack still opens with the poster; no DEM in packs; field pack stays 7.0 MB | Holds |

**Gate before committing to the opener:** run the spike on one real mid Android and one older iPhone (iOS 18). Pass means a median frame at or under 33 ms through the 4.5 s move, no context loss across 3 runs, and tiles ready within 1.5 s on throttled 4G. Fail means Plan B (custom-layer heightfield from the 175 KB single image) or dropping the opener for the still.

## 13. Ranked

**Build:** (1) persistent map in an `(ocean)` layout with a scene store, section 3; (2) React `<ViewTransition>` for port to briefing, retiring the `routes.tsx` wrapper there and keeping the root out of the snapshot, section 4; (3) one capability module, three tiers, custom-layer recovery after context loss, section 11; (4) currents polish: tapered trails, soft additive, coverage hairline, section 6; (5) baked Tanaka relief and opener still in `build_relief.py`, sections 5 and 8; (6) keep hand-rolled charts, section 9.

**Build with care:** (7) terrain opener, tier full, once, skippable, labelled exaggeration, after the phone gate, section 5; (8) per-cell HF radar quality as a sidecar, section 6; (9) shader contours on cursor or scroll, desktop first, uniforms only, section 8; (10) CSS scroll-linked hero as now, at most one scrubbed camera passage on desktop, section 10; (11) Motion `layoutId` inside one page only, section 4.

**Don't build:** (12) Paper Shaders at runtime near water, section 7; (13) three.js or deck.gl for one scene, section 5; (14) GPU particle advection or interpolation across gaps, section 6; (15) visx, or statewide vector contour files, sections 8 and 9; (16) scroll-scrubbed camera on phones, or anything depending on WebGPU, sections 10 and 11.

## Sources

Local evidence: packages inspected in `node_modules`, the Next 16.4.2 bundled guide `01-app/02-guides/view-transitions.md`, `caniuse-lite` 1.0.30001816, npm registry for versions and licences, and bundles measured with esbuild (minified, gzip -9).

- [Safari 26.0 release notes](https://developer.apple.com/documentation/safari-release-notes/safari-26-release-notes.md)
- [WebKit Features in Safari 26.0](https://webkit.org/blog/17333/webkit-features-in-safari-26-0/)
- [NOAA Coastal Relief Model](https://www.ncei.noaa.gov/products/coastal-relief-model)
- [ETOPO 2022](https://www.ncei.noaa.gov/products/etopo-global-relief-model)
- [CUDEM 1/9 arc-second, data.gov record](https://catalog.data.gov/dataset/continuously-updated-digital-elevation-model-cudem-1-9-arc-second-resolution-bathymetric-t)
