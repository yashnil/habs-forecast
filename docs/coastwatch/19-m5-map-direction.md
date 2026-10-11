# 19 — M5: rendering audit and map direction

**Status:** decided. All five recommendations in §7 were approved and are built; see [20 — M5: Ocean Map](20-m5-ocean-map.md). The review switches described below have been removed. What follows is the decision record, written at the comparison stage. All screenshots use one build on the **live production dataset** (2026-10-10). Images are in [m5/comparison/](m5/comparison/).

**What is built so far:**
- The new layout: a full-bleed map, a control rail, a compact layer dock, a snap-point sheet on phones, and one status line.
- The relief assets.
- Review-only switches: `?relief=0`, `?nodata=hatch|stipple|veil`, `?arrows=dense`. These will be removed once a direction is chosen.

**What is not built yet:**
- the inspector redesign;
- hover readouts;
- the raster fixes listed in §1.

Start with [p3-before-forecast](m5/comparison/p3-before-forecast.jpg) beside [b-bathymetric-satellite](m5/comparison/b-bathymetric-satellite.jpg).

## 1. Raster rendering audit

The audit traced Sentinel-3 OLCI from ERDDAP through ingest, value grids, tiling, colouring and MapLibre to the screen.

| Stage | What happens | Detail lost? | Artifact? |
|---|---|---|---|
| Ingest | 0.0025° lattice (223 × 278 m at 36.8° N), copied cell for cell; coastal-scope mask | No | No |
| Value grid | uint16 log10, gzip | No (0.01 % rounding) | No |
| Tiles | grid → Mercator z5–10, 256 px, nearest; all 13,161 Monterey cells present at z10 | No | **Yes**: 1.82 × 2.27 tile px per cell, so cells become 1 or 2 px wide |
| Colour | 9-stop log ramp, 254 classes, lossless 8-bit PNG | No visible banding | No |
| MapLibre | second nearest resample; tiles stop at z10 | 0.7 % of cells dropped at DPR 1 only | **Yes**: uneven widths compound. At z11, DPR 2, cells are 8 or 16 px instead of 14–15 ([audit-olci-z11-cell-widths](m5/comparison/audit-olci-z11-cell-widths.jpg)) |
| C-HARM | 0.03° (~3 km) grid, one Mercator image, 4 px per cell | No | **25 × 32 px blocks** at the default zoom, and a 3 km staircase along the masked coast ([audit-charm-blocks](m5/comparison/audit-charm-blocks.jpg)) |

**Root causes, ranked:**
1. **The "blocky, low-resolution" impression comes from C-HARM, not the satellite.** Its cells are 3 km; at Monterey Bay every cell falls in the 70–80 % class, so the bay is one pink slab ([forecast-monterey](m5/comparison/forecast-monterey.jpg)). This is the model's real resolution. It must not be smoothed.
2. **Two non-integer nearest resamples** (grid → z10 tiles → screen) make OLCI and VIIRS cells uneven once zoomed past about z10.5. VIIRS is uneven at every zoom.
3. **Missing tiles are requested anyway:** 404s for empty tiles (4 of 9 at z10 in the bay).

**Fixes. None changes a value or the no-data mask:**

| Option | Effect | Cost |
|---|---|---|
| @2x tiles (512 px, declared 256) plus maxzoom 11 (OLCI) / 10 (VIIRS) | Even cells at every zoom | About +4.8 MB statewide per OLCI layer set; Monterey +34 KB |
| Draw the value grid directly in a WebGL layer | Exact cell edges at every zoom; uses the grid the inspector already reads | No extra bytes; more engineering |
| Display-only bilinear inside valid cells | Smooth, but implies detail below 300 m | **Not recommended**; it would need disclosure |
| Skip empty tiles (use the manifest's tile index) | Removes the 404s | Small |

**Recommendation:**
- @2x + maxzoom 11 now, in the pipeline, published to staging first.
- The WebGL grid layer later, if exact edges are wanted at z12.
- C-HARM keeps its cells. A faint cell outline from z9 makes them read as model cells, not pixelation.

## 2. Three treatments, same data

| | A. Minimal marine | B. Bathymetric marine | C. Satellite-first |
|---|---|---|---|
| Basemap | flat sea and land, coastline only | ETOPO 2022 relief, plus NOAA Coastal Relief Model 3″ in Monterey Bay; isobaths 200–3000 m | as B |
| Opening layer | C-HARM | C-HARM | multi-sensor chlorophyll (Sentinel-3 300 m + VIIRS 750 m) |
| Screens | [satellite](m5/comparison/a-minimal-satellite.jpg) · [currents](m5/comparison/a-minimal-currents.jpg) | [satellite](m5/comparison/b-bathymetric-satellite.jpg) · [currents](m5/comparison/b-bathymetric-currents.jpg) | [1440](m5/comparison/c-satellite-first-statewide.jpg) · [390](m5/comparison/c-satellite-first-390.jpg) |
| Spatial detail | Data only; empty sea reads as nothing | Shelf, shelf break, and Monterey and Carmel canyons under currents and in cloud gaps | Fronts, nearshore enrichment and the bay's structure at 300 m |
| Readability | Calm, but generic | Coast and depth orient the eye. The relief is subtle and sits under all data, so it never competes | Strongest first impression; needs the "biomass, not toxin" line kept visible |
| Scientific risk | Lowest | Low: the relief is real NOAA data, darker and greyer than every data colour, and hidden wherever data exist | Medium: a first-time user may read green as "toxic bloom" |

Licences:
- **ETOPO 2022** (NOAA NCEI, doi:10.25921/fd45-gt74) and the **Coastal Relief Model** (NOAA NCEI): free to use and redistribute.
- **Assets:** 263 KB total, built by [pipeline/scripts/build_relief.py](../../pipeline/scripts/build_relief.py).
- **Attribution:** in the map credits.

## 3. No-data treatment

Compared statewide ([hatch](m5/comparison/nodata-hatch-statewide.jpg) · [stipple](m5/comparison/nodata-stipple-statewide.jpg) · [veil](m5/comparison/nodata-veil-statewide.jpg)) and in Monterey Bay ([hatch](m5/comparison/nodata-hatch-monterey.jpg) · [stipple](m5/comparison/nodata-stipple-monterey.jpg) · [veil](m5/comparison/nodata-veil-monterey.jpg)).

- **Hatch:** unmistakable, but it textures most of the screen on a cloudy week. It also covered the whole sea in the currents view, where no raster was drawn. That is now fixed for every option: no-data is shown only under a raster layer.
- **Stipple:** a 1 px dot every 3 px. Still clearly "not a colour", quiet, and the relief shows through. **Recommended.**
- **Veil:** the calmest, but it lifts the whole ocean to grey and weakens the coastline contrast.

None of the three uses a hue from any data palette, so a gap never reads as a low value.

## 4. Opening state: C-HARM or observation first

| | C-HARM (today) | Multi-sensor chlorophyll |
|---|---|---|
| Statewide coverage | Complete offshore; nearshore 3–6 km masked ([statewide](m5/comparison/forecast-statewide.jpg)) | 95 % of Monterey Bay this week; gaps elsewhere ([statewide](m5/comparison/multisensor-statewide.jpg)) |
| Today's data | **Stale** (issued Oct 8; the latest update failed) | Newest pixel Oct 8; each pixel dated |
| At the default Monterey view | One colour class over the whole bay | Real spatial structure |
| What it answers | "How likely is domoic acid over 500 ng/L?", the closest thing to the user's question | "Where is algae biomass?", which is not toxin |

**Recommendation: a hybrid.**
- **Keep C-HARM as the opening layer.** The decision cannot rest on looks: the forecast is the only layer that speaks to toxin.
- **Draw it so its information shows:**
  - faint cell outlines;
  - the nearshore mask as stipple, not slabs;
  - its stale state on the map itself, which is already on the dock.
- **Make satellite chlorophyll one tap away** on the rail.
- **Open on satellite automatically only when C-HARM is unavailable or historical**, and say so.

If you prefer observation-first, it is a one-line change. The stamp and dock already carry "chlorophyll is biomass, not toxin".

## 5. Layer controls (already in this branch)

**Desktop:**
- A 64 px rail picks the layer group.
- The dock keeps only what changes the map:
  - what and when;
  - the time steps;
  - a one-row legend with units;
  - the colour mode;
  - freshness and the essential warnings.
- Options, caveats, method and provenance sit under **Details**.
- **Measured unobstructed map** (element boxes on a 4 px grid, production data, with C-HARM stale and two failed updates):

  | Viewport | Map area unobstructed | Target |
  |---|---|---|
  | 1440 × 900 | 65.5 % (P3: about 55 %) | ≥ 75 % |
  | 1280 × 720 | 52 % | — |
  | 390 × 844, peek | 54 % of the screen | ≥ 55 % |

- **Not yet met.** The dock grows to about 300 px when stale and failed-update warnings stack, and the place card is 320 × 330 px.
- **Planned fixes:**
  - fold the warnings into one line with the freshness badge (still visible, never hidden);
  - collapse the place card to a one-line place chip with the notices count, which opens the list;
  - give the phone peek the legend and time instead of the product picker.

**Phone:**
- The dock is a sheet with peek, half and full.
- Official notices stay in the status line, the masthead pill and the place pill.

## 6. Current fields

Compared:
- [arrows at the current density](m5/comparison/b-bathymetric-currents.jpg);
- [an arrow at every radar cell](m5/comparison/currents-every-cell.jpg);
- [particles](m5/comparison/currents-particles.jpg);
- [every cell over chlorophyll](m5/comparison/currents-every-cell-over-chlorophyll.jpg).

At the default zoom today's arrows show **26 %** of the radar cells: every second row and column, as in P3's ARROW_ZOOMS. That is why the field looks sparse and detached.

**Arrow at every cell:**
- one smaller arrow per observed 2 km cell from z8.5;
- it reads as a measured field;
- its outline is the radar footprint, so missing coverage is visible without a hatch.

**Recommended:**
- the every-cell arrows as the default;
- particles as an option (off under reduced motion);
- the stamp keeps "observed, not a forecast; not where a bloom will go".

## 7. Decisions needed

1. **Treatment:** B (bathymetric marine) as the foundation? Recommended.
2. **Opening layer:** the C-HARM hybrid (recommended) or observation-first?
3. **No-data:** stipple? Recommended.
4. **Raster fix:**
   - Option 1: @2x tiles + maxzoom 11 (pipeline change, staging first).
   - Option 2: wait for the WebGL grid layer.
5. **Currents:** an arrow at every cell? Recommended.

After your choice:
- finish the inspector, hover readout, region framing and motion;
- remove the review switches;
- run the full test suite;
- complete both visual review passes.
