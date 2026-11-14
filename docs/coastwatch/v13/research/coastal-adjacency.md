# Coastal adjacency in satellite chlorophyll: evidence and the rule

Agent C saw a continuous dark (high-chlorophyll) band hugging the shoreline in Explore's OLCI layer. This note
records what was measured at the source, what the pipeline now does about it, and what is left.

**Finding.** In NOAA CoastWatch's Sentinel-3 OLCI 300 m chlorophyll the water pixel that touches the product's
own land mask reads 8 to 50 mg/m³ on every clear scene measured, 35 to 83 % of it in the top display class, and
the next pixel 2 to 17 mg/m³ (9 to 38 %), against 0 to 1 % in open water. The datasets publish `chlor_a` only: no
flag marks these pixels. It is an artefact of the product (land and water in one pixel, and the land's stray
light), not of Neritic's pipeline, which passed the values through unchanged.

**Rule.** OLCI layers leave out the two water pixels beside the product's land mask (about 600 m). No value is
altered; the cells have no value and a QC check says how many were left out. VIIRS 750 m is not changed.

**What is left.** In Monterey Bay the third pixel is still high in the recorded week (10 to 19 % in the top
class); in the Southern California Bight it is clean (0 to 1 %). The fixture's shore-most cells went from a
median of 39 mg/m³ (68 % top class) to 9.7 mg/m³ (17 %). Whether Monterey's third pixel is the bay's own
nearshore water or more adjacency cannot be told from this product alone, so it is kept and the caveat about
values very close to shore stays.

## 1. Sources

| Product | Datasets (NOAA CoastWatch ERDDAP, `coastwatch.noaa.gov/erddap`) | Cell |
|---|---|---|
| OLCI | `noaacwS3AOLCIchlaSectorCIDaily`, `…DIDaily` (and S3B for the mask) | 0.0025°, about 300 m |
| VIIRS | `noaacwN21VIIRSchlaSectorUSDaily` | 0.0075°, about 750 m |

Boxes (lat N to S, lon W to E): Monterey Bay 37.05–36.5, −122.3–−121.7; Bodega and Point Reyes 38.45–37.9,
−123.3–−122.7; Santa Barbara Channel 34.55–34.0, −119.95–−119.3 (OLCI: the DI sector).

Request, one per scene: `…/griddap/<dataset>.csv?chlor_a[(<time>)][(0.0)][(<latN>):(<latS>)][(<lonW>):(<lonE>)]`,
for every time on the dataset's axis from 2026-10-12 to 2026-11-10 (read 2026-11-12).

## 2. Live scenes, by distance from the product's land mask

`pipeline/scripts/adjacency_table.py <dir>`. "Land" is a cell with no value in any scene of the box; distance is
in the sensor's own cells (1 touches land). Medians over the clear days (at least a quarter of the water valid).

### OLCI 300 m

| Coast | Clear days | 1 | 2 | 3 | 4–5 | 6–10 | 11–20 | 21+ | Share > 10 mg/m³: ring 1 / 2 / 3 |
|---|---|---|---|---|---|---|---|---|---|
| Monterey Bay | 13 | 18.09 | 3.79 | 1.88 | 1.39 | 0.93 | 0.67 | 0.50 | 62 % / 16 % / 4 % |
| Bodega, Pt Reyes | 8 | 17.07 | 10.83 | 9.05 | 8.27 | 5.17 | 2.24 | 0.78 | 75 % / 58 % / 47 % |
| Santa Barbara Channel | 12 | 1.03 | 0.56 | 0.41 | 0.35 | 0.28 | 0.24 | 0.23 | 29 % / 5 % / 0 % |

(mg/m³.) Ring 1 is 6.6, 2.1 and 2.4 times ring 3. At Santa Barbara the median is modest but 29 % of ring-1
pixels are above 10 mg/m³ where ring 3 has none: a population of extreme values that exists only beside land.
Bodega is the weakest case: Tomales Bay, Drakes Estero and the turbid shelf are high everywhere.

### VIIRS 750 m

| Coast | Clear days | 1 | 2 | 3 | 4–5 | 6–10 | 11–20 | 21+ | Share > 10 mg/m³: ring 1 / 2 / 3 |
|---|---|---|---|---|---|---|---|---|---|
| Monterey Bay | 8 | 3.82 | 1.86 | 1.23 | 0.99 | 0.87 | 0.75 | 0.72 | 5 % / 0 % / 0 % |
| Bodega, Pt Reyes | 9 | 8.17 | 6.19 | 5.47 | 3.29 | 1.98 | 1.46 | 0.78 | 37 % / 16 % / 8 % |
| Santa Barbara Channel | 11 | 1.65 | 0.46 | 0.37 | 0.33 | 0.30 | 0.27 | 0.27 | 1 % / 0 % / 0 % |

VIIRS's first pixel is higher than its neighbours (3.0, 1.6 and 4.2 times ring 3) but it has no population of
extreme values (1 to 5 % above 10 mg/m³ at two coasts), and at Monterey its value (3.8) is what OLCI reads in the
same water, 300 to 900 m out (3.8 and 1.9). **The rule is not applied to VIIRS.** Santa Barbara's factor of four
is not explained and is an open question; the existing caveat ("values very close to shore … are less
reliable") covers it.

## 3. Recorded scenes (the test fixtures), by ring of the reference

An independent set: the scenes recorded on 2026-10-09 for the tests, read against the reference of section 4
(ring 1 touches land, 8 neighbours; ring k touches ring k−1). Median mg/m³ and the share in the top display
class (21 mg/m³ and above).

| Scene | Ring 1 | Ring 2 | Ring 3 | Ring 4 | Ring 5 | Ring 6 | Beyond |
|---|---|---|---|---|---|---|---|
| Monterey, Oct 1 | 29.7, 58 % | 11.2, 24 % | 6.5, 10 % | 5.0, 4 % | 3.5, 2 % | 4.5, 3 % | 4.41, 4.0 % |
| Monterey, Oct 5 | 39.9, 69 % | 13.0, 38 % | 5.0, 19 % | 2.7, 4 % | 2.1, 0 % | 2.0, 0 % | 0.62, 0.1 % |
| Monterey, Oct 6 | 50.2, 83 % | 17.0, 33 % | 14.3, 12 % | 7.6, 9 % | 5.6, 5 % | 3.6, 2 % | 0.55, 0.0 % |
| S. California Bight, Oct 3 | 15.6, 45 % | 3.4, 12 % | 1.4, 1 % | 0.8, 0 % | 0.6, 0 % | 0.5, 0 % | 0.16, 0.1 % |
| S. California Bight, Oct 6 | 7.9, 35 % | 1.7, 9 % | 0.8, 0 % | 0.6, 0 % | 0.5, 0 % | 0.4, 0 % | 0.16, 0.0 % |

The other recorded scenes had fewer than 20 valid pixels in every ring (cloud). Ring 2 carries the top class in
9 to 38 % of its pixels in every scene, at both coasts; ring 3 does in Monterey Bay (10 to 19 %) and not in the
Bight (0 to 1 %). This is why the rule is two pixels and not one or three.

## 4. The reference

`data/reference/olci_land_adjacency.json`, built by `pipeline/scripts/build_land_adjacency_mask.py` (needs
network; `--web-only` rewrites the web copy without it).

- Land is a cell of the OLCI lattice with no value in any of 87 scenes (Sentinel-3A and -3B, both sectors,
  2026-10-10 to 2026-11-10), **and** within 4 cells of the OpenStreetMap shoreline (the committed basemap water
  layer). The second condition keeps sea that was under cloud in every scene from being called land.
- Ring 1: 21,894 cells that had a value and touch land (17,975 in the satellite scope). Ring 2: 14,712 cells
  that touch ring 1 (13,995 in scope). Together under 2 % of the scope's cells.
- Checked against section 2's boxes: the reference's ring 1 covers 99 % (Monterey), 97 % (Bodega) and 95 %
  (Santa Barbara) of the ring measured there.
- Bays: the rings follow a bay's shore and its open water is kept (San Francisco Bay is tested). A slough or
  channel four cells wide or less (1.2 km) has no cell left: every one is within two pixels of land.
- The web's copy (`coastwatch-web/public/reference/land-adjacency-v1.json`, 88 KB) holds the lattice and the
  merged runs only.

## 5. What changed

- `sources/satellite.py`: `Product.land_adjacent_rings` (OLCI 2, VIIRS 0); after the mosaic the cells are set to
  no value and counted in a QC check named `land_adjacency`; a caveat is added to OLCI layers;
  `PROCESSING_VERSION` is `satellite-processing-5`, so published days made before the rule are not reused. No
  schema field was added.
- Everything computed from the OLCI days follows: the latest view, the multi-sensor view, the estimate.
- Explore: a cell with no value in an OLCI frame whose QC carries `land_adjacency`, and that is in the
  reference, reads "Left out: beside land, where the satellite reads land and water together", with its own
  legend key. It is not "no view" (cloud) and not "outside the model". Frames made before the rule (staging
  today) and VIIRS frames never say it.
- Nothing was published. Staging and production data are as they were.

## 6. Before and after (the fixture's latest view, Monterey Bay)

| | Before | After |
|---|---|---|
| Cells with a value | 14,242 | 13,661 |
| Cells in the top class (≥ 21 mg/m³) | 382 (2.7 %) | 130 (1.0 %) |
| Shore-most cells with a value: median | 39.0 mg/m³ | 9.7 mg/m³ |
| Shore-most cells in the top class | 68 % | 17 % |

"Shore-most": before, the cells of ring 1; after, the cells that touch the left-out band.

## 7. Limits

- One month of scenes, autumn 2026, three coasts. The north coast was too cloudy to measure.
- The reference is static. If NOAA changes the product's land mask, rebuild it (the test compares the shoreline
  file's hash, not the product's mask).
- A two-pixel rule also removes real nearshore water: kelp beds, harbours, river mouths, sloughs under 1.2 km.
- The alternative for the lead, one line: `land_adjacent_rings=1` keeps ring 2 (and its 9 to 38 % top-class
  share); `3` would also take Monterey's third pixel, without support from the Bight.
