# C-HARM at the coastline: investigation and refinement

**Status:** draft for review. The change is to the web app only: no pipeline change, and no change to published or production data. The Research Preview and its deployment are untouched.

**Data:** production run 38140958238; C-HARM v3.1 issued 2026-10-10, lead 1. Before and after use the same data, viewport and URL.

## 1. The question

Where the C-HARM forecast meets the coast, the map looks pixelated, stair-stepped and disconnected from the geography. Which parts are rendering artifacts that can honestly be fixed, and which are real limits of a 0.03° (about 3 km) model?

## 2. What the data are

- **Value grid:** the published grid is 351 × 391 cells at 0.03°.
  - It is published as uint16 values for exact readouts.
  - It is also published as a Web Mercator PNG (1,404 × 1,970 px, 4 px per cell east–west), drawn as one image source with nearest-neighbour sampling.
- **ETOPO 2022 15″ comparison.** Every cell was classified against ETOPO land and sea:

| Cells | Count | What it is | Treatment |
|---|---|---|---|
| Forecast cells that straddle the shore | 432 (168 mostly land) | A 3 km square is drawn whole, so its colour covers land | **Rendering artifact: fixed** (§3) |
| Water cells with no value in any variable | 486 (383 one cell from shore; two regions of 105 and 142 cells, mostly enclosed bays such as San Francisco Bay) | Outside C-HARM's ocean cells | **Genuine coverage limit:** keeps the no-data hatch |
| Cells with a *Pseudo-nitzschia* value but no toxin value | 347 | C-HARM's documented toxin mask ("not provided for many cells within about 3–6 km of shore") | **Genuine:** keeps the hatch, and the readout now says why (§4) |
| The model-domain rectangle (31.3–43.0° N, 127.5–117.0° W) | — | It lies outside the map's California bounds | Not visible |

[Classification method and figures](#5-method-and-reproducibility).

## 3. Fix: data rasters stop at the drawn shoreline

**The layer.** A new basemap layer, `coastal-land`, sits **above** the data rasters and **below** the coastline line and labels.
- It is the same land relief that lies beneath (`relief-land.webp`, same pixel grid), opaque over land and clear over water.
- It covers only a band about 10 km wide along the ocean coast.

**Its coastline.** The shoreline is the basemap's own: OpenFreeMap / OpenMapTiles water polygons (© OpenStreetMap contributors, ODbL) at zoom 10, from 177 coastal tiles. Coverage is computed at 4 × 4 sub-pixels, so the edge is anti-aliased rather than stair-stepped. It adds 62 kB (relief) or 27 kB (flat, relief off).

**Why it is honest:**
- No value changes.
- The forecast keeps full opacity and nearest-neighbour sampling.
- The part of a cell over water is drawn exactly as before; only the part over land is covered.
- Cells without a value keep their hatch, so gaps are not hidden.
- Inland, the land looks identical, because the pixels underneath are the same.
- True-colour imagery is drawn above the cover (it shows land too).
- The satellite chlorophyll layers benefit in the same way.

**Before and after** (left: main; right: this branch; same data and viewport):

| View | Comparison | What changes |
|---|---|---|
| Statewide, 1440 | [statewide-1440](charm-coastline/statewide-1440.jpg) | The sawtooth of forecast colour over land along the whole coast is gone, and San Francisco Bay reads as "no value" |
| Statewide, *Pseudo-nitzschia* | [statewide-pseudo-nitzschia-1440](charm-coastline/statewide-pseudo-nitzschia-1440.jpg) | The same, on the layer that reaches the shore |
| Phone, statewide | [statewide-390](charm-coastline/statewide-390.jpg) | The same, on a phone |
| Default (Monterey Bay), 1440 and phone | [default-1440](charm-coastline/default-1440.jpg), [default-390](charm-coastline/default-390.jpg) | Overhanging cells at Moss Landing and on the north bay shore stop at the coast |
| Selected port: Santa Cruz | [port-santa-cruz-1440](charm-coastline/port-santa-cruz-1440.jpg) | Small: at this zoom the edge is mostly the toxin mask, which is genuine |
| Selected port: Morro Bay | [port-morro-bay-1440](charm-coastline/port-morro-bay-1440.jpg) | Small, for the same reason |
| Coast detail, *Pseudo-nitzschia* | [coast-detail-monterey](charm-coastline/coast-detail-monterey.jpg), [close-zoom-santa-cruz](charm-coastline/close-zoom-santa-cruz.jpg) | Cells clipped at the shoreline; the hatch and the 3 km steps remain |

## 4. Explaining genuine gaps

Hovering or clicking a cell without a value now says which case it is:
- **"No toxin probability in this cell: C-HARM does not provide toxin probabilities for many cells within about 3–6 km of shore. The same run has a *Pseudo-nitzschia* value here."** This is shown only when the same run's *Pseudo-nitzschia* layer has a value in that cell ([screenshot](charm-coastline/readout-toxin-mask.jpg)).
- **"No forecast value: land, or water C-HARM does not model (such as enclosed bays and some cells next to the shore)."**
- **"Outside the C-HARM model area."**

## 5. What cannot be improved honestly, and what could be with approval

**Unavoidable at 3 km:**
- The forecast is 3 km squares. Their edges against no-value cells are steps of one cell, and nearshore water within one or two cells of the coast (and, for toxins, about 3–6 km) has no value.
- Smoothing, interpolating or extending values to the shore would invent a forecast that C-HARM does not make. The map keeps the steps, and the faint cell outlines from zoom 8.5 show them as cells.

**A rendering limit that needs a pipeline change (not done here):**
- The published image has whole-pixel Mercator rows, so north–south cell edges sit on average 164 m (at most 355 m) from the true cell boundaries:
  - invisible statewide;
  - about 1 px at zoom 8.5 and 3 px at zoom 10;
  - 6–12 px at zoom 11–12, where they show against the exact cell outlines and readouts.
- Fixes:
  - render the images at 8 px per cell, which halves the offset;
  - or publish them in latitude bands of about 10 cells, which brings the offset to about 16 m.
- Either changes published production images. Because the Research Preview reads production data, its map would change too, so this needs your approval.

**Considered and rejected:**
- **Linear (blurred) resampling of the forecast:** it mixes colours into values that are not in the legend.
- **Drawing the cells as vector shapes:** 11–19k shapes per layer, a large architectural change with real phone cost.
- **A flat land overlay:** it would erase the coastal relief.

**Method and reproducibility.**
- `pipeline/scripts/build_coastal_land.py` rebuilds the cover. It needs the ETOPO extract used by `build_relief.py` and network access to the OpenFreeMap tiles, which it caches.
- The cell classification used the published grid and ETOPO 2022 15″ (land = elevation ≥ 0 m) for each cell's land fraction.

## 6. Tests

- **Browser tests (`tests/e2e/coastline.spec.ts`):**
  - layer order: data, then land cover, then coastline and labels;
  - the forecast stays opaque and nearest-neighbour;
  - the land-cover assets are served and attributed, on the same grid as the relief;
  - the readout reasons on fixture cells.
- **Unit test:** the run-matched *Pseudo-nitzschia* sibling is used only for toxin layers.
- **Results:** full suite 132/132 (local Chrome and CI mode); web unit tests 137/137.
