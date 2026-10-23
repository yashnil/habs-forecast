# Explore as a living geophysical instrument: research, feasibility, spike

Status: research and a measured spike (branch `spike/v10-viz`, local only). Nothing here changes the app on
`main`, and nothing here is approved for build. Date: 2026-10-22. Data: the staging channel
(`coastwatch-data-staging/v1`, manifest generated for the C-HARM run of Oct 20 and the HF-radar hours from
Oct 20 03Z to Oct 21 02Z).

Before-state screenshots: `scratchpad/v10-before/explore-currents-{d,p}.png`,
`explore-chl-d.png`, `explore-forecast-d.png`. Spike outputs: `scratchpad/v10-viz/` (frames, videos,
`perf-flow.json`, the measuring script, copied here as `flow_perf_measure.mjs`).

**Bottom line.** Neritic can honestly animate exactly one thing: one observed HF-radar hour, as
short-lived particles that show direction and relative speed inside the observed cells and die at the
edge of coverage. Every other "living" effect has to come from real frames shown in turn (playback,
crossfades between published frames) and from interaction (an exact cell footprint, a point's time
series), never from fabricated intermediate states. The spike shows that the particles cost about
**0.3 ms of JavaScript per frame** as a MapLibre custom layer in the map's own WebGL2 context (versus
about 1.6 ms for the existing canvas overlay), and about **3 KB gzip** of code.

---

## Part 1. Reference products

Sources were read on 2026-10-22. Anything not read from a primary source is marked *unverified*.
The cambecc/earth source is saved in `scratchpad/earth/` (products.js, micro.js, earth.js, globes.js).

### 1.1 Earth Nullschool and cambecc/earth

**Licence and code.**
- [cambecc/earth](https://github.com/cambecc/earth/blob/master/LICENSE.md) is MIT ("Copyright (c) 2014
  Cameron Beccario").
- The [about page](https://earth.nullschool.net/about.html) links that repo as the "Free Version of
  Source" and says to contact inquiries@nullschool.net for licensing.
- The live site's current code is not published. Treat it as proprietary; that is an inference.
- The live site's currents are OSCAR v2.0 and Copernicus Marine global physics, "updated every day".
- The open repo's OSCAR product is a 0.33° grid (`oscar0p33Path`), stepped about 5 days at a time.

**What it does exceptionally well.**
- A calm, legible field of motion over a semi-transparent colour field (`OVERLAY_ALPHA = 0.4·255`).
- The motion reads as "the ocean is moving" without a single label.
- Permalinks keep the full state.

**Why it feels polished** (from the source):
- *Fading trails.* Each frame multiplies the canvas alpha by 0.97 (`globalCompositeOperation =
  "destination-in"` with `fillStyle "rgba(0,0,0,0.97)"`). Trails are long and soft.
- *No synchronised pulsing.* Particles start at random ages (`MAX_PARTICLE_AGE = 100` frames).
- *Speed-binned brightness.* There are 18 grey buckets (`windIntensityColorScale`,
  `INTENSITY_SCALE_STEP = 10`), and each bucket is one `beginPath`/`stroke`, so a whole frame is ~18 draw
  calls.
- *Projection-correct direction.* `distortion()` in micro.js uses finite differences with `H = 0.0000360`.
- *Interpolation off the main path.* It interpolates the screen field in 100 ms time slices
  (`MAX_TASK_TIME`, `MIN_SLEEP_TIME = 25`), then draws at a fixed `FRAME_RATE = 40` ms.

**Gaps are honest in the open code.**
- `buildGrid` interpolates bilinearly only when all four corners are values; otherwise it returns `null`.
- The currents builder writes `null` for any missing u or v.
- `evolve` sets `particle.age = MAX_PARTICLE_AGE` when a particle enters an undefined vector, so it is
  reborn elsewhere.
- No time interpolation: `lookupOscar` returns the newest file at or before the requested date.

**What Neritic currently does worse.**
- The currents view (`v10-before/explore-currents-d.png`) is a static arrow field. In the before shot it
  is an empty sea plus one line of caveat (see §2.1, the default-hour problem).
- The particle renderer that exists (`FlowParticles.tsx`) is not reachable from v4 Explore at all (§2.1).

**Technique worth adapting.**
- Speed buckets batched into few draw calls.
- Random initial ages.
- Null as a first-class value that kills a particle.
- Particle density proportional to screen area (`PARTICLE_MULTIPLIER = 7` per pixel of width; 0.75× on
  mobile).

**What NOT to copy.**
- *Bilinear interpolation of the vectors.* At 0.33°/1° that is presentation; at Neritic's 2 km radar
  cells it would invent sub-cell structure. Keep the nearest cell. Note that earth only interpolates
  inside fully valued quads, which is the strictest form we could ever allow.
- *The 2 px screen-field upsampling presented as resolution.*
- *`velocityScale` / `PARTICLE_MULTIPLIER` treated as physics.* The code itself calls them "arbitrary".
- *Very long trails* (0.97 fade gives ~100-frame tails). On a 2 km field with 1–2 cell gaps, long tails
  read as trajectories, which hourly Eulerian snapshots cannot support.
- *Nullschool's current proprietary code and visual identity.*

### 1.2 Windy

**What it provides.**
- The [Map Forecast API](https://api.windy.com/map-forecast/docs) is Leaflet 1.4-based, with "layers,
  particles, legend, picker, and isolines".
- It allows one map per page, uses localStorage heavily, and warns that undocumented features may break.
- The [Point Forecast API](https://api.windy.com/point-forecast/docs) returns one shared `ts` array
  (epoch ms) plus parameter arrays aligned to it, with `null` for no data.

**Terms.**
- The free tier is development-only and GFS-only.
- [Point testing data](https://api.windy.com/point-forecast/pricing) is "randomly shuffled and slightly
  modified".
- Professional costs 990 €/year.
- A staff post (thread now deleted, so secondary) says public weather apps may not use the API.
- The docs read "© Windyty, S.E. – all rights reserved". `windycom/windy-plugins` declares ISC in
  package.json but ships no LICENSE file. The renderer is not public.

**What it does exceptionally well / why polished.**
- One integrated instrument: the colour field, the particles, the picker and the timeline are one system.
- The point forecast is a first-class object.
- I did not instrument windy.com's own UI, so I make no claims about its internal techniques.

**What Neritic does worse.**
- Neritic's timebar (`explore-forecast-d.png`, `explore-currents-d.png`) steps frames well (v8 playback,
  1.2 s per frame).
- But a point's history is a separate page (Research), not a strip under the map.

**Technique worth adapting.**
- The point-series shape `{ts[], var[] with nulls}` as the contract for a point inspector strip.
- A picker that follows the pointer and reads the exact cell.
- Playback that is clearly "frame N of M" with model run and valid time separated.

**What NOT to copy.**
- Embedding Windy (licence, and its testing data is deliberately falsified).
- Hourly *forecast* animation semantics: Neritic has no hourly forecast. The "run vs forecast time"
  timeline is a C-HARM concept only, and C-HARM is daily.

### 1.3 Copernicus Marine MyOcean Pro (and MyOcean Learn)

**What it does exceptionally well.**
- The [geometry-to-graph model](https://help.marine.copernicus.eu/en/articles/7992399-what-information-can-we-obtain-from-myocean-pro-viewer-object-graphs):
  - a **point** gives a time series and depth profile;
  - a **line** gives a transect (value vs distance) and value vs distance × time;
  - an **area** gives an area-averaged time series and a histogram.
- Each graph shows average, minimum and maximum *of the plotted data*.
- Export is "the exact data that is displayed on the graph" (CSV, NetCDF, SVG).
- The [product (i) panel](https://help.marine.copernicus.eu/en/articles/6482737-introduction-to-myocean-pro-viewer)
  leads to the product ID, DOI, resolution and the Quality Information Document. A Notifications tab
  lists incidents.

**Why it feels polished.**
- Every geometry has one obvious analytical answer.
- Provenance is one click from any layer.
- Share links keep geometries and layer settings.

**Licence.**
- [Copernicus Marine licence](https://marine.copernicus.eu/user-corner/service-commitments-and-licence):
  free, worldwide, perpetual, derivative works allowed.
- It requires the credit "Generated using E.U. Copernicus Marine Service Information" plus DOI.
- The viewer's code licence was not found (*unverified*; assume proprietary).
- MyOcean Learn: no help-center documentation found (*unverified*).

**What Neritic does worse.**
- Neritic already has point, area and station inspection (Inspect.tsx, AreaDrag.ts) and an exact
  footprint highlight.
- But Explore's inspection is a value *now*, while history lives on `/research` (`research-d.png`).
- There is no transect.
- An area's spread (histogram, valid-cell count) is not shown on the map.

**Technique worth adapting.**
- Point → series and area → mean ± spread with **the valid-cell count shown**.
- "Export what is plotted".
- A per-product incident line (Neritic already has stale/failed notices; keep them per layer).

**What NOT to copy.**
- An area mean shown without its valid-pixel count. With clouds and radar gaps, the mean of 12 cells
  and the mean of 400 cells must not look alike.
- Any gap-filling colour map on daily satellite frames.

### 1.4 Observable Plot

- [Pointer transform](https://github.com/observablehq/plot/blob/main/docs/interactions/pointer.md):
  shows the nearest point within `maxRadius` (default 40 px). `pointerX` biases to x for time series;
  click-to-stick locks it.
- [crosshair](https://github.com/observablehq/plot/blob/main/docs/interactions/crosshair.md) and
  [tip](https://github.com/observablehq/plot/blob/main/docs/marks/tip.md) (pooled, at most one tip).
- **Brushing is not in a release**: the latest is 0.6.17, [issue #5](https://github.com/observablehq/plot/issues/5)
  is open, and brush PRs are unmerged to main. Do not plan on Plot brushing.
- Licence: ISC (Observable, Inc.). observablehq.com rate-limited the fetch (429), so these come from the
  repository docs.

**What it does exceptionally well.**
- Exact-value inspection with almost no chrome: one pooled tip, a crosshair, and a stick-on-click.

**What Neritic does worse.** Research charts (`research-d.png`) are static series; there is no
crosshair/tip pairing between the map and the series.

**Technique worth adapting.** `pointerX` + `crosshairX` + `tip` for the point time-series strip. Either
use Plot (ISC, about 70 KB gzip *unverified for the current version*; measure before adopting) or copy the
interaction pattern into the existing SVG charts. Gaps must stay gaps: Plot's `line` breaks on
`null`/`NaN` by default, so never pass interpolated values.

**What NOT to copy.** Implicit curve smoothing (`curve: "basis"`, etc.) on observations; linear is the
ceiling.

### 1.5 Colour scales and libraries

- cmocean is [MIT](https://github.com/matplotlib/cmocean/blob/main/LICENSE.txt) (Thyng 2015). Neritic
  already uses a cmocean-"thermal"-like SST ramp (`lib/derived.ts`, `neritic-sst-thermal-v1`).
  - `speed` (sequential) suits current speed.
  - `algae` suits chlorophyll. Keep the published `cw-chlorophyll-log-v1` until the pipeline changes it,
    because legend values must match the pipeline.
- d3-scale-chromatic is ISC, plus an Apache-2.0 notice for the ColorBrewer schemes. Keep both notices if
  it is used.
- deck.gl is MIT. Gzip sizes from bundlephobia: core ~218 KB, layers ~58 KB. `MapboxOverlay` with
  `interleaved: true` shares the map's WebGL2 context, but it is still ~275 KB gzip for something the
  spike does in 3 KB.

---

## Part 2. Feasibility

### 2.1 Animated currents

**What the data are** (staging manifest and all 25 published grids, decoded in
`hfr_coverage_analysis.py` beside this doc):

**Source and grid.**
- Source: HFRNet US West Coast 2 km hourly totals, via NOAA NDBC THREDDS (`hfradar_uswc_2km`). Public
  domain; credit HFRNet and the operators.
- Grid: 408 × 540 cells, 0.01798° lat × 0.02082° lon, which is 2.0 km N–S × 1.86 km E–W at 36.8° N.
- Stored as u/v `uint16le+gzip` with quantisation error ±0.00004 m/s.
- Each hour is ~13 KB gzip per component, so **~26 KB per hour, ~650 KB for all 24 hours**.

**Time.**
- 24 hourly snapshots (Oct 20 03Z … Oct 21 02Z), plus a 24 h mean (cells with ≥18 of 24 valid hours).
- Published about 4–15 h after observation.
- The app calls an hour stale past 18 h (`HFR_STALE_HOURS`).
- In the spike the newest hour was 46 h old, so it showed "Stale".

**QC, done in the pipeline per hour.**
- At least 2 radar sites.
- HDOP ≤ 1.25.
- Speed ≤ 2.0 m/s.
- Grid alignment checked.

The published grid holds only cells that passed. There are **no per-cell QC flags** in the artefact, only
pass/fail counts.

**Coverage.**
- 8,500–10,400 valid cells per hour (~3.9% of the grid; 9–11% of the C-HARM ocean domain).
- 12,776 cells were ever valid; 5,677 were valid in all 24 hours.
- **~1,240 cells change state between consecutive hours (mean; max 2,501)**: about 13% of the valid
  field flickers hour to hour.
- In the newest hour, 1,084 of 8,532 valid cells are on an edge (fewer than 4 valid neighbours), and 102
  interior holes are surrounded on ≥3 sides.

**Speeds.**
- Median 0.19–0.29 m/s, p95 0.42–0.57 m/s, maximum ~1.0 m/s.

**The default-hour problem (a real bug, found here).**
- The newest hour (Oct 21 02Z) has **zero** valid cells in Monterey Bay (0 of 575 in a 36.55–37.0° N box;
  the hours before it had ~420). The region's coverage entry says 0.0.
- `defaultHour()` accepts it because its *domain-wide* coverage (0.0926) is above half the median
  (~0.054).
- So `/?layer=currents&region=monterey_bay` opens on an empty sea (`v10-before/explore-currents-d.png`)
  while the hour before it is full.

Fix (not done here): pick the default hour by the *visible region's* coverage
(`coverage.regions[].observed_fraction`, already published), and say "newest hour with Monterey Bay
coverage: 6 PM; the 7 PM hour has none here".

**Can particles honestly encode direction and relative speed?** Yes, within limits:
- One hour, frozen. A particle shows the instantaneous (hour-averaged) surface vector of the cell it is
  in. It is not a trajectory: HF-radar totals are Eulerian hourly snapshots, and advecting a particle for
  longer than a cell crossing would imply a Lagrangian path the data do not contain.
  - At 0.25 m/s, a real water parcel takes ~2 h to cross one 2 km cell. Any visible motion is
    therefore a *symbol*, not the water's real displacement. Say so on screen: "Motion shows direction
    and relative speed; not real speed or a path".
- Screen speed ∝ current speed, the same at every zoom (spike: 30 px/s per m/s). A legend chip "fast ·
  slow" ties it to the m/s arrow legend.
- Nearest cell, no interpolation. A particle takes the u/v of the cell containing it.
  - Bilinear interpolation only inside fully valued 2×2 quads (earth's rule) would be defensible
    visually but adds no information at 2 km.
  - Readouts must stay on the exact cell (`currentsAt`, `sampleField(…, 0)`).
- Never interpolate across missing coverage. A particle dies the moment it enters a cell without a value
  and is reborn in a random *observed* cell in view.
  - Gaps therefore stay visibly empty.
  - The spike adds a faint fill of each observed cell's true 2 km footprint, so "no particles" visibly
    means "no observation", not calm water.
- Lives are short (1.2–2.4 s) and trails ~0.5 s, so no streak spans more than a few cells.
- Always show the hour and source next to the motion ("HF radar · observed 6 PM PDT Oct 20 · 46 h ago,
  stale"). The current dock already does; keep it pinned while animating.
- Keep native sampling for inspection: tapping reads the exact cell (u, v, speed, compass, cell centre),
  and the arrows remain one per cell at harbour zoom.
- **Static mode and reduced motion.** With `prefers-reduced-motion` the spike draws one still frame of
  short streaks and never animates. Arrows stay the default static mode, and the dock should toggle
  "Arrows / Flow".
- **Hour changes** (playback or step). Never morph particles between hours. Swap the field and respawn
  every particle (the layer is keyed by hour id). A hard cut is the honest transition; a 150–200 ms
  opacity dip is acceptable decoration.

**FlowParticles.tsx: audit.**
- *Good:*
  - honest by construction (frozen hour, nearest cell, death in gaps, short lives);
  - respawn only in valid cells;
  - pauses when the tab is hidden;
  - documented intent.
- *Wrong or weak:*
  1. **Dead code in v4.** Explore always sets `mode: "arrows"` (`Explore.tsx`). Only the legacy
     `LayerDock`/`LayerSwitch` offered "particles", and nothing imports `LayerSwitch` now.
  2. **A second canvas over the map.** It sits at `z-index 1` above every map layer, so particles are
     drawn *over* labels, the official-notice lines and the selection ring. It cannot be ordered under
     labels.
  3. **It blanks on every move** (`movestart` clears and stops drawing; `moveend` respawns everything),
     so the field vanishes while panning: the opposite of a living instrument.
  4. It calls `map.project` and `map.unproject` per particle per frame (2,800 projections per frame on
     desktop). That is the bulk of its 1.6 ms per frame.
  5. It makes one `beginPath`/`stroke` per particle (1,400 draw calls per frame).
  6. The full-canvas `destination-in` fade costs fill-rate on a 3× phone canvas. DPR is capped at 2.
  7. `spawn` tries 20 random *global* valid cells per particle and gives up if none are in view. When
     zoomed into a small area most attempts miss, so density drops. The spike precomputes the visible
     valid cells on `moveend` instead.
  8. No footprint of coverage. "No particles" and "slow water" look alike at a glance.
  9. The `resize` handler resets the canvas without redrawing (harmless, given the per-frame redraw).
- *Must change:* move into the map's GL context (a custom layer) below labels; keep georeferenced
  particles during camera moves; draw the observed footprint; batch the drawing.

**Options compared.**

| Option | Second WebGL context | Under labels | Moves with map | JS per frame (desktop, measured) | Added JS | Licence |
|---|---|---|---|---|---|---|
| Canvas 2D overlay (FlowParticles) | no (2D canvas) | no | no (blanks) | ~1.6 ms | ~1.4 KB gz | own |
| **MapLibre custom layer (spike)** | **no, shares the map's WebGL2** | **yes (`beforeId`)** | **yes** | **~0.3 ms** | **~3.0 KB gz** | own |
| deck.gl `MapboxOverlay` interleaved | no (when interleaved) | yes | yes | not measured | ~275 KB gz (core+layers) | MIT |

**Recommendation: the MapLibre custom layer.**
- Its main cost is that MapLibre repaints the whole map every animation frame (`triggerRepaint`). The
  measurements below show what that costs.
- A GPU particle-state version (ping-pong textures, as in mapbox/webgl-wind, ISC) is only worth it above
  ~10k particles. At 600–1,400 particles the CPU simulation is ~0.1–0.2 ms.

### 2.2 Temporal behaviour

The rule: **show only published frames; between them, cut or crossfade the pixels of two real frames,
never compute an intermediate field.**

| Layer | Native time step | Appropriate between-frame behaviour | Why |
|---|---|---|---|
| C-HARM (nowcast + 3 days) | daily, one run | cut, or a ≤250 ms crossfade of two real days | daily model fields; an interpolated "noon" field does not exist, and blending probabilities across days is not a probability |
| Satellite chlorophyll (S3 300 m, VIIRS 750 m) | daily overpass, clouds | cut; never crossfade into cloud gaps | a gap in day N must stay a gap; a crossfade would paint day N−1 values into day N's clouds |
| Multi-sensor latest-clear | one composite, per-pixel dates | no timeline; per-pixel age view (exists) | each pixel has its own date |
| 8-day outlook | one derived product | none | not a time series |
| SST (MUR L4) | daily analysis (gap-filled upstream by the producer) | cut or short crossfade | already an analysis; say "analysis", not "observed" (`sstIsAnalysis`) |
| HF-radar currents | hourly snapshots | cut; particles respawn per hour | ~13% of cells change state each hour; no Lagrangian continuity |

- The current app already does this right for rasters: a new *layer* fades in over 220 ms, and a new *day*
  of the same layer cuts (MapCanvas `RASTER_FADE_MS` comment). Keep it.
- *Watch one existing effect:* below zoom 8 the data rasters use `linear` resampling ("display only,
  about one texel wide"). That blends a valued texel into a transparent no-value texel at every gap edge:
  a one-texel soft edge, not a fabricated value, but it is a smoothing over gaps. It is acceptable only
  because it is sub-cell. Do not widen it, and do not apply it to the radar footprint.

**Playback and preloading.**
- Playback is 1 frame / 1.2 s today (`PLAY_MS`).
- For currents, fetch all 24 hours on play: ~650 KB gzip, ~26 KB per hour; decode ~0.6 MB of uint16 per
  hour with `DecompressionStream`.
- For raster days, prefetch the next frame's tiles for the viewport. MapLibre has no API for that, so add
  a hidden source with `raster-opacity: 0` for frame N+1, or `fetch()` the tile URLs.
- A frame that has not arrived holds the previous frame **with its own time label still shown** and a
  "loading 7 PM…" chip. Never advance the label before the data.
- Disabled (empty or failed) frames stay in the strip, marked, and playback skips them (already true for
  `disabled` steps).
- Stale: per-layer, from the published `freshness` rule. The newest radar hour at 46 h is "Stale" today.

### 2.3 Geographic selection

**Highlight the true footprint.**

| Selection | Footprint to draw |
|---|---|
| C-HARM | the 0.03° cell polygon (already: `v4-hl-fill`/`-line`, kind `cell`) |
| Satellite | the 300 m or 750 m pixel (`pixel`, already in Research) |
| Radar | the 2 km cell (same polygon builder; centre ± half `lat_step`/`lon_step`) |
| Station | a ring at its true position |

- When the readout falls back to a *nearest* value (C-HARM nearshore), draw both footprints and the
  connector (already done in Coast mode, `coast-cell-*`).

**Link to history without losing the location.**
- Explore already writes `sel`, `at` and the layer to the URL (`writeUrl`), and Inspect builds
  `/research?station=…&place=…&variable=…&day=…&from=…`.
- Add the selection's exact cell id (`grid:row:col` per product) so Research can draw the same footprint
  and My Coast can open the same point (`/coast?place=…` already exists).
- A point time-series strip in Explore (Copernicus point → series, Observable pointerX + crosshair)
  should read the same published grids:
  - the 24 radar hours (~650 KB, cached);
  - the 4 C-HARM days;
  - the satellite days present.
- The strip shows gaps as gaps.

### 2.4 Map aesthetics within MapLibre

**Bathymetric relief.**
- Already shipped as pre-rendered images: NOAA NCEI ETOPO 2022, and the Coastal Relief Model at 3″ for
  Monterey Bay (`lib/basemap.ts`, `public/basemap/*.webp`, ~350 KB). Isobaths come from GeoJSON.
- MapLibre ≥5.6 has a `color-relief` layer (hypsometric tint from terrain-RGB) and `hillshade`.
- A dynamic DEM would need a terrain-RGB tile set:
  - AWS Terrain Tiles are ETOPO1-based, so coarse offshore;
  - MapTiler Ocean RGB restricts redistribution and requires its logo.
- Recommendation: keep the pre-rendered NOAA relief and extend the 3″ CRM to the other regions in the
  pipeline. It gives better coastal detail than any free tile service, at zero runtime cost.

**Coastline precision.**
- OpenFreeMap/OpenMapTiles water polygons.
- Data rasters are clipped by the vector water, so land never shows ocean colour.

**Colour scales.**
- Speed: a sequential, lightness-monotone ramp (cmocean `speed`, MIT) for particle brightness and arrow
  tint. Keep white on dark sea for legibility, and bucket into ~8 levels as earth does.
- Chlorophyll: the pipeline palette (legend values must match the pipeline).
- C-HARM: the display classes over the published palette.
- Never use a rainbow.

**Blending and opacity.**
- Particles over chlorophyll: the arrows already thin one level over imagery (`coarser`). Particles
  should drop to ~60% count and 0.7 alpha over a data raster.

**Missing coverage.**
- Keep the existing no-data hatch for rasters.
- For radar, the observed-footprint fill (spike) plus words.

**Labels.** Draw the custom layer `beforeId` the overlay group, so labels, notices and the selection
ring stay on top.

### 2.5 Performance (measured)

Spike: `src/components/map/FlowLayerGL.tsx` (a custom layer), wired behind `?flow=gl` (custom layer) and
`?flow=canvas` (the existing FlowParticles, instrumented) in Explore.

Setup:
- URL: `/?layer=currents:20261021T01Z&region=monterey_bay` (the newest hour *with* Monterey Bay coverage;
  9,609 observed cells).
- Particles: 1,400 on desktop (1440×900), 600 on phone (390×844 at DPR 3, CPU throttled 4× through CDP).
- Each run: 6 s idle animation, then 4 s of panning (`panBy` out and back). Median of 3 runs.
- Measured: rAF interval, long tasks, the renderer's own JS per frame (`__flowStats`), and main-thread
  busy ms per second (CDP `TaskDuration`).
- Headless Chrome on an Apple-silicon Mac. GPU time is not separated, so treat these as main-thread
  numbers.

Main-thread busy time is ms per second of wall time (CDP `TaskDuration`). "Renderer JS" is the
particle code's own time per animation frame.

| Mode | Device | Idle: main busy ms/s | Idle: renderer JS / frame | Pan: main busy ms/s | Draws during pan | Long tasks |
|---|---|---|---|---|---|---|
| Arrows (today, static) | desktop | 10 | n/a | 93 | arrows, yes | 0 |
| Canvas overlay (FlowParticles) | desktop | 106 | 1.49 ms | 105 | **no (blank)** | 0 |
| **Custom layer (spike)** | desktop | **99** | **0.29 ms** | 119 | **yes** | 0 |
| Arrows (today, static) | phone 4× | 13 | n/a | 261 | arrows, yes | 0 |
| Canvas overlay (FlowParticles) | phone 4× | 173–176 | 2.0 ms | 274–280 | **no (blank)** | 0 |
| Custom layer, 60 fps | phone 4× | 236–256 | 0.42 ms | 276–290 | yes | 0 |
| **Custom layer, 30 fps cap (spike default under 768 px)** | phone 4× | **125–153** | 0.36 ms | 264–272 | **yes** | 0 |

Raw data: `scratchpad/v10-viz/perf-flow*.json`.

Reading the numbers:
- **The particle code is cheap.** About 0.3 ms per frame on desktop and 0.4 ms on a 4×-throttled phone.
  The canvas overlay costs 5× more (1.5–2.0 ms), mostly `project`/`unproject` and 1,400 separate strokes.
- **The real cost of the custom layer is MapLibre repainting the whole map every animated frame.** On
  the throttled phone at 60 fps that is ~240 ms/s of main-thread time (~24%), against ~175 ms/s for the
  overlay and 13 ms/s for static arrows.
- **A 30 fps cap halves it** to ~125–150 ms/s, below the canvas overlay. The motion stays smooth because
  particles move ~0.25–0.5 px per frame at 30 fps.
- **While panning, every mode costs about the same**, because MapLibre is repainting anyway. Only the
  custom layer keeps showing the field during the move.
- **No long tasks in any mode.** Building the observed-footprint buffer (9,609 cells) and the
  visible-cell list on `moveend` stays below the 50 ms long-task threshold.
- **The rAF frame rate was 60 in every case, so it does not discriminate here.** Headless Chrome's CPU
  throttling slows script, not vsync, and GPU time is not included. Busy ms/s is the meaningful number.
  Confirm on a real mid-range Android before shipping (budget: ≤150 ms/s of main thread while
  animating, 0 long tasks).

**JS cost.**
- `FlowLayerGL` is ~3.0 KB gzip (esbuild, minified); FlowParticles is ~1.4 KB.
- No dependency. deck.gl would add ~275 KB gzip.
- **Data cost:** ~26 KB gzip per radar hour; all 24 hours are ~650 KB.

**Battery and visibility rules for the real build.**
- Pause on `document.hidden`.
- Pause when the currents layer is off-screen (no visible observed cells).
- Pause after ~60 s without input, with a "Flow paused · tap to resume" chip.
- 30 fps under 768 px and on `navigator.hardwareConcurrency <= 4`.
- Count scales with the visible observed area, not a constant (earth: 7 per px of width, 0.75 on mobile).
  Monterey Bay at phone zoom has ~430 observed cells, and 600 particles is already dense.

**Spike artefacts** (`scratchpad/v10-viz/`):
- `video-gl-desktop.webm`, `video-canvas-desktop.webm`, `video-gl-phone.webm`: 6 s idle, then a pan out
  and back.
  - The canvas video shows the field vanishing during the pan.
  - The GL videos show it staying.
- Frames: `frame-*.png`, `gl-mb-01Z-{d,p}.png`.

**What the spike found about MapLibre 6.13.** `CustomRenderMethodInput.modelViewProjectionMatrix` maps
**world-pixel** coordinates (Mercator × 512·2^zoom), not Mercator 0..1 as the docs' Null-Island example
implies. The spike scales the matrix so its buffers stay in Mercator 0..1.

The layer must also disable the stencil, depth and cull state left by earlier raster layers. MapLibre
only guarantees blending.

**Spike limits** (not production code):
- Speed buckets and the colour ramp are not done (alpha by speed only).
- The fps cap is by width, not by device.
- Hour changes respawn by remounting the layer (correct, but it recompiles shaders; the real build
  should swap buffers).
- The particle count is not yet scaled to the visible observed area.
- No dock toggle: `?flow=gl` / `?flow=canvas` only.
- No legend line "Motion shows direction and relative speed".

---

## Part 4. Recommended architecture and expected cost

| Feature | Architecture | Expected cost |
|---|---|---|
| Animated currents | MapLibre custom layer (spike), CPU simulation, instanced quads, footprint fill, below labels, 30 fps on phones, reduced motion = one still frame, arrows stay the static mode | ~3 KB gz; 0.3–0.4 ms/frame JS; +110–140 ms/s main thread on a throttled phone vs arrows; 2–3 days to productionise with tests |
| Default radar hour | choose by visible region's `coverage.regions[]`; say which hour and why | small; fixes the empty Monterey Bay view today |
| Playback | real frames only; cut per day/hour; prefetch N+1 (radar: all 24 h, ~650 KB; rasters: hidden source); hold the old label until data arrive | 1–2 days |
| Exact footprint selection | existing `v4-highlight` kinds + radar `cell` polygon; cell id in URL `sel` | <1 day |
| Point time-series strip | published grids only (24 radar hours, 4 C-HARM days, satellite days present); gaps as gaps; pointerX/crosshair/tip pattern (Plot ISC, or the same pattern in the existing SVG charts) | 2–4 days; Plot adds JS (measure first) |
| Area statistics | mean, min, max **and valid-cell count / total** (Copernicus-style histogram optional) | 1–2 days |
| Relief and coastline | keep pre-rendered NOAA ETOPO/CRM; extend CRM 3″ to other regions in the pipeline | pipeline work; zero runtime cost |
| Colour | cmocean `speed` (MIT) for flow brightness, 8 buckets; pipeline palettes unchanged | small |


---

## Part 3. What must NOT be built

1. **Interpolated time.** No "smooth" hourly-in-between currents, no noon-of-day C-HARM, no morphing
   between satellite days. Crossfade pixels of two real frames at most, labelled with the frame being
   faded *to*.
2. **Trajectories, drift tracks or "where will the bloom go".** Particles from one Eulerian hour are not
   paths. Multi-hour advection through flickering 2 km coverage (13% of cells change each hour) would
   cross gaps by construction. A drift product needs a model, and Neritic has none.
3. **Gap filling of any kind** in radar, satellite or C-HARM fields: no inpainting, no smoothing across
   no-data, no "filled" particle field over Monterey Bay when the hour has no coverage there.
4. **Particles on the 24 h mean** presented as "now". If offered, it must be labelled "24-hour mean
   motion", and arrows are the better encoding.
5. **Particles for chlorophyll, C-HARM or SST.** These are scalar fields; animating them invents motion.
6. **Real-time-speed claims.** Screen speed is a scaled symbol; never print "real speed" or animate in
   "real time".
7. **Windy or Nullschool embeds or code.** Licence and terms forbid it, and our data are not theirs.
8. **An area mean without the valid-cell count**, or a time series that bridges missing hours with a line.
