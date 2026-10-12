# C-HARM on the Ocean Map: coastline and georeferencing

**Status:** draft for review (PR #28). The change is to the web app only: published data and production images are unchanged, and the Research Preview is not touched. §10 covers the fallback for when the basemap's tiles fail or are slow.

**Data:** production run 38140958238; C-HARM v3.1 issued 2026-10-10. Before and after use the same data, viewports and URLs. Staging (run 38158950779, a later C-HARM issue) was used as a second dataset.

## 1. Questions

1. Where C-HARM meets the coast, which parts of the crude look are rendering artifacts, and which are real coverage limits?
2. Are the forecast's cell edges drawn where the source grid puts them, and if not, why?

## 2. Coverage: what is real

Every cell of the 351 × 391, 0.03° grid was classified against ETOPO 2022 15″ land and sea:

| Cells | Count | Verdict |
|---|---|---|
| Forecast cells straddling the shore, drawn as whole 3 km squares over land | 432 (168 mostly land) | **Rendering artifact** |
| Water with no value in any variable: next to the shore, and enclosed bays such as San Francisco Bay | 486 | **Genuine limit:** keeps the no-data hatch |
| *Pseudo-nitzschia* value but no toxin value (C-HARM's documented toxin mask, "within about 3–6 km of shore") | 347 | **Genuine:** keeps the hatch; the readout says why |
| The model-domain rectangle | — | Outside the map's bounds |

## 3. Georeferencing: verified, and two real causes found

The published PNG (1,404 × 1,970 px) was checked against the source grid:

| Check | Result |
|---|---|
| Image corners vs outer cell edges (centres ± half a step) | **0.0°** |
| PNG rebuilt from the value grid (pixel-centre sampling, Mercator-uniform rows) | No-data mask 100% equal; **0 of 1,536,220** valued pixels with the wrong class colour |
| Readout and cell outlines in the web app | Same convention (`floor((v − first)/step + 0.5)`; edges at centre ± step/2) |

There is no pixel-centre or pixel-edge convention error, and no transform or reprojection bug. The offsets have two distinct, real causes:

1. **The PNG's row resolution (pipeline image).**
   - East–west is exact at 4.00 px per cell. North–south, a cell is 4.7–5.5 rows, so each boundary lands on the nearest row.
   - Measured over all 390 boundaries: mean |offset| **164 m**, max **355 m**, signed mean **+9 m**. It is unbiased rounding, not a shift.
2. **MapLibre's image-source placement (renderer).**
   - MapLibre 6.13 (`image_source.ts`) converts an image's corners to coordinates in one tile, at zoom ⌊−log₂ extent⌋ = 4, and rounds them to that tile's 8,192-step grid (305.7 m).
   - Our corners moved: north +47 m, south −77 m, west −63 m, east −19 m (Mercator). This is a small systematic squeeze and shift.
   - The predicted shift matched the measured on-screen residual at 12 boundaries to within 0.9 px at zoom 12.

**Total effect on screen** (production, zoom 12, 12 boundaries): drawn edges are a mean of 11.9 px and at most 20 px from the true boundaries. That is where they disagree with the exact cell outlines and with the readout.

## 4. What changed

**The forecast is drawn from the published value grid, tile by tile (`src/lib/gridTiles.ts`):**
- Every tile pixel takes the grid cell containing its centre, which is the pipeline's own rule, at the zoom being viewed.
- Colours come from the layer's published palette, using the pipeline's step rule over all 65,535 codes.
- No value means transparent. Nothing is interpolated.
- Both causes above disappear:
  - **Production:** 12/12 boundaries drawn exactly on the true boundary (0.0 px at zoom 12).
  - **Staging:** the same, on a different C-HARM issue.
  - **Fixture:** the tiles equal the published PNG at the centre of every cell.

**Every data tile is clipped to the basemap's own water (`src/lib/waterMask.ts`):**
- For each forecast or satellite tile, the basemap's vector tile for the same z/x/y (OpenMapTiles water; usually already in the browser cache) is rasterised at the data tile's resolution.
- Data are kept only over water, with an anti-aliased edge where the basemap draws the shore.
- If the basemap tile cannot be read, or has not arrived after 3 s, the bundled copy of the same water is used instead (§10). Only if that cannot be loaded either is the tile drawn unclipped.

**Readout:** an empty cell says why ([screenshot](charm-coastline/readout-toxin-mask.jpg)). Either it is the toxin mask (named only when the same run's *Pseudo-nitzschia* layer has a value there), or land or water C-HARM does not model, or it is outside the model area.

**A first approach was rejected.** A raster "land cover" above the data (relief-land, clear over water) was validated and failed at close zoom. Its ~370 m pixels blurred the edge: it hid near-shore water, including no-value hatching, up to about 400 m out, and left forecast showing ~300 m inland. It was replaced by the per-tile clip and is not in this PR.

## 5. Validation (production data; 6 coastal places × zoom 8, 10, 12)

For every screen pixel, the exact grid says whether it lies in a valued cell, and the basemap's own water render says water or land. Pixels within 2 px of a shoreline or a cell edge are excluded, since anti-aliasing and pixel rounding live there. Then:

| Measure (worst view) | `main` | This PR |
|---|---|---|
| Forecast drawn over land | 61,652 px, up to ~3 km inland | **0** in all 18 views |
| Forecast drawn over water with no value (hiding a real gap) | 2,882 px | **0** in all 18 views |
| Ocean values not drawn | 3,509 px | ≤ 13 px of ~266,000 (≤ 0.005%) |

Places: Santa Cruz, Moss Landing, Half Moon Bay, Golden Gate, Morro Bay, Point Arena. The 0–13 px residual sits next to tiny water features the basemap draws differently from one zoom's tiles to the next.

**Performance** (median of 5; production data; re-measured for §10, with the corrected method below):

| | `main` | This PR |
|---|---|---|
| Map ready, desktop | 499 ms | 451 ms |
| Map ready, phone with CPU slowed 4× | 1,111 ms | 1,130 ms |
| Blocking time, phone 4× | 146 ms | 150 ms |
| Re-zoom (Monterey Bay to zoom 11), desktop and phone 4× | 333–334 ms | 331–332 ms |
| Ocean Map first-load JavaScript | 159 kB | 160 kB (+1 kB: the readout's explanations for empty cells; the grid tiles, clip and fallback load with the map) |
| All JavaScript (gzip) | 592 kB | 599 kB (+4.1 kB vector-tile decoder, +2 kB grid tiles, clip and fallback, +1 kB readout) |
| Download | 64 kB image | 150 kB grid (also used by the readout) |

Correction: the earlier re-zoom figures (835 vs 833 ms) included the measuring script serialising the map object it got back from `jumpTo`. The script now returns nothing; the comparison between `main` and this PR is unchanged (equal).

## 6. Before and after (left `main`, right this PR)

| View | Comparison |
|---|---|
| Statewide, 1440 | [statewide-1440](charm-coastline/statewide-1440.jpg) |
| Statewide, *Pseudo-nitzschia* | [statewide-pseudo-nitzschia-1440](charm-coastline/statewide-pseudo-nitzschia-1440.jpg) |
| Phone, statewide | [statewide-390](charm-coastline/statewide-390.jpg) |
| Default, desktop and phone | [default-1440](charm-coastline/default-1440.jpg), [default-390](charm-coastline/default-390.jpg) |
| Selected port: Santa Cruz, desktop and phone | [port-santa-cruz-1440](charm-coastline/port-santa-cruz-1440.jpg), [port-santa-cruz-390](charm-coastline/port-santa-cruz-390.jpg) |
| Selected port: Morro Bay | [port-morro-bay-1440](charm-coastline/port-morro-bay-1440.jpg) |
| Coast detail and close zoom | [coast-detail-monterey](charm-coastline/coast-detail-monterey.jpg), [close-zoom-santa-cruz](charm-coastline/close-zoom-santa-cruz.jpg) |

## 7. Unavoidable at 3 km

- The forecast is 3 km squares. Their edges against no-value cells are one-cell steps, and water within a cell or two of the coast (and, for toxins, about 3–6 km) has no value.
- These are shown honestly: the steps stay, the gaps keep their hatch, and the cell outlines from zoom 8.5 now coincide with the drawn edges.

## 8. Compatibility and deployment

- **Published data:** no change. The pipeline still writes the PNG, which the Research Preview and older clients use, and the value grid this PR draws from.
- **No production publication is needed for this correction.** The web app falls back to the PNG for any layer whose grid it cannot draw (missing grid, chunked grid, non-step palette).
- **Deployment:** this changes the full website only when it is deployed, which needs your approval.
- **Dependencies:** `@mapbox/vector-tile` 3.0.0 and `pbf` 5.1.2 (BSD-3-Clause), the versions MapLibre already ships. They are now direct dependencies.
- **External dependency:** the clip reads OpenFreeMap tiles, which the basemap already depends on. If they fail or are slow, the bundled water (§10) is used; it ships with the site.

## 9. Tests

- **Unit tests (`gridtiles.test.ts`):**
  - colours equal the pipeline's step rule for every code;
  - tiles equal the published PNG at every fixture cell centre;
  - north–south and east–west edges within half a tile pixel of the true boundary at zoom 12;
  - an outside tile is transparent;
  - the clip arithmetic, and a failed mask leaving data unclipped.
- **Browser tests:**
  - the forecast source is the grid;
  - drawn edges lie within 1 px of the true boundaries, and the readout matches the drawn cell on both sides;
  - straddling cells are drawn on water and never on land;
  - the readout reasons.
- **Fallback (§10):** see there.
- **Full results** (after merging `main`, which brought in M7): 151/151 browser tests in local Chrome and in CI mode; 178 unit tests; 198 pipeline tests.
- **Ported test:** the old map test asserting an image source at the PNG corners now asserts a grid-tile source bounded by the same corners and tied to the selected layer and grid.

## 10. When the basemap's tiles fail or are slow

**Problem.** The clip read the basemap's vector tiles. With the tile server down (or the device offline from it), every data tile was drawn unclipped, and C-HARM cells again covered land up to ~3 km inland. The map also lost its sea, coastline and no-data dots, so a gap looked like land ([Monterey Bay](charm-coastline/fallback-monterey-1440.jpg), left).

**Bundled water** (`public/basemap/water-fallback.json`, built by [`pipeline/scripts/build_water_fallback.py`](../../pipeline/scripts/build_water_fallback.py)):
- The same water the basemap draws: the OpenMapTiles `water` layer from OpenFreeMap. Every zoom-10 tile over the map's pannable area is read; tiles with both ocean and land are read again at zoom 11.
- **Ocean class only** (the sea, bays and estuaries up to the OSM coastline). Lakes, lagoons and rivers are left out: C-HARM is an ocean model, and the basemap keeps or drops them by size from one zoom to the next (at zoom 8 and 10 they were the only places where forecast showed on the basemap's land).
- Simplified (Douglas-Peucker, topology kept) to **8 m within one cell of a C-HARM ocean cell** (`data/reference/charm_ocean_mask.json`, which holds every valued cell of all 12 production grids) and 40 m elsewhere. Small islands are kept (Southeast Farallon, the Channel Islands).
- Cut along the zoom-8 tile grid, so a data tile only reads the coastline near it.
- 181 pieces, 53,280 vertices, 432 kB (128 kB gzipped). Fetched only when needed; normal page loads do not download it.

**Behaviour** (`src/lib/waterMask.ts`, `src/lib/waterFallback.ts`, `MapCanvas`):
- A data tile uses the basemap's water. If its vector tile cannot be read, or has not arrived after **3 s**, the tile uses the bundled water, rasterised the same way (even-odd scanline fill, 4 sample lines per row, exact horizontal coverage: anti-aliased, deterministic, no browser canvas needed).
- Only if the bundled file cannot be loaded either is the tile drawn unclipped (data are never hidden because of a failure).
- While the basemap is down, or slow for the current view, the map draws the bundled sea, its coastline and the no-data dots; they hide again once the basemap's tiles arrive. Values, masks, timestamps and the published PNG fallback are unchanged.
- The page now gets the map once its style is ready, not at MapLibre's `load` (which waits for every basemap tile), so the data layers, the dots and region navigation do not wait for a slow tile server.

**Validation** (production data; same 6 places × zoom 8, 10, 12 and same exclusions as §5; truth from a normal session's basemap):

| Measure (18 views) | Tile server reachable | Unreachable | Tiles 8 s late |
|---|---|---|---|
| Forecast drawn over land | 0 | **0** | **0** |
| Forecast drawn over water with no value | 0 | **0** | **0** |
| Ocean values not drawn | 22 px | 2,640 px of 2,723,522 (0.10%) | 2,513 px (0.09%) |

- The values not drawn sit almost all on Elkhorn Slough (1,778 px at Moss Landing, zoom 12) and the Garcia River mouth (664 px at Point Arena): water the basemap draws as river or lake, which the bundled ocean leaves out.
- Where the bundled water and the basemap disagree inside a valued cell (2,511 px), it is always "bundled: land, basemap: water". Never the reverse.
- A recorded basemap tile (Moss Landing, zoom 12) agrees with the bundled water to within 2 px of the shore on every pixel (< 0.5% of pixels differ, all at the shore; unit test).

**Screenshots** (left `main`, right this PR; tile server unreachable at the network level):

| View | Comparison |
|---|---|
| Monterey Bay, 1440 | [fallback-monterey-1440](charm-coastline/fallback-monterey-1440.jpg) |
| San Francisco Bay and the Farallones, 1440 | [fallback-sfbay-1440](charm-coastline/fallback-sfbay-1440.jpg) |
| Statewide, 1440 | [fallback-statewide-1440](charm-coastline/fallback-statewide-1440.jpg) |
| Phone 390: Monterey Bay and statewide | [fallback-phone-390](charm-coastline/fallback-phone-390.jpg) |
| Toxin mask off Santa Cruz (particulate domoic acid, nowcast) | [fallback-toxin-santacruz-1440](charm-coastline/fallback-toxin-santacruz-1440.jpg) |
| Close zoom, Santa Cruz | [fallback-close-santacruz-1440](charm-coastline/fallback-close-santacruz-1440.jpg) |
| Slow tiles (7 s in) vs loaded, this PR | [fallback-slow-monterey-1440](charm-coastline/fallback-slow-monterey-1440.jpg) |

**Performance.**
- Basemap reachable: unchanged (§5).
- Basemap unreachable: re-zoom is the same as `main` (332 ms). The bundled water costs once: +92 ms map ready and +50 ms blocking on desktop; +330 ms and +250 ms on a phone with CPU slowed 4× (reading it, and handing the sea to the map).
- Rasterising a 512 px tile from the bundled water: 1.4–2.6 ms from zoom 8 to 12, 13 ms for a statewide zoom-5 tile (desktop).

**Tests.**
- Unit (`waterfallback.test.ts`): the rasteriser (edges, anti-aliasing, holes, outside the domain); the file covers the pannable area, is ocean-only, is cut along the zoom-8 grid, and its drawn coastline has no grid cuts; sea, land and islands at known points; agreement with the recorded basemap tile; and the choice of water when the tile server is unreachable, slow (3 s, fake timers), the browser cannot read vector tiles, or nothing can be read.
- Browser (`coastline-fallback.spec.ts`), truth from a normal session:
  - tile server unreachable: no forecast on land and forecast on all ocean points in straddling cells; the bundled sea is shown;
  - tiles 20 s late: the same, checked before the tiles arrive; then the bundled sea gives way to the basemap;
  - unreachable: toxin-mask and unmodelled cells on the sea keep the no-data dots and no forecast colour, land has neither, valued cells are coloured, and the readout still names the toxin mask.

**Limits.**
- The bundled water is a copy: if OpenStreetMap's coastline changes, it is rebuilt with the script (the basemap path always uses the live tiles).
- While OpenFreeMap is slow, place names and port markers wait for its fonts (MapLibre holds a source's tiles until their labels' glyphs arrive). This predates this PR; serving the fonts from the site would remove it.
- Satellite tiles beyond the pannable area are not clipped by the fallback (the map cannot be panned there).
