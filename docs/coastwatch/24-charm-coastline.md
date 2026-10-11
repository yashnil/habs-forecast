# C-HARM on the Ocean Map: coastline and georeferencing

**Status:** draft for review (PR #28). The change is to the web app only: published data and production images are unchanged, and the Research Preview is not touched.

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
- If the basemap tile cannot be read, the data tile is drawn unclipped, so data are never hidden by a basemap failure.

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

**Performance** (phone with CPU slowed 4×, median of 3; the transfer column of the measuring script reads 0 for these cross-origin files):

| | `main` | This PR |
|---|---|---|
| Map ready | 1,099 ms | 1,201 ms |
| Blocking time | 107 ms | 93 ms |
| Re-zoom | 835 ms | 833 ms |
| Ocean Map first-load JavaScript | 159 kB | 159 kB (MapLibre stays in the lazily loaded map chunk) |
| Download | 64 kB image | 150 kB grid (also used by the readout) |

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
- **External dependency:** the clip reads OpenFreeMap tiles, which the basemap already depends on. If they fail, data are drawn unclipped.

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
- **Full results:** 133/133 browser tests in local Chrome and CI mode; 145 unit tests.
- **Ported test:** the old map test asserting an image source at the PNG corners now asserts a grid-tile source bounded by the same corners and tied to the selected layer and grid.
