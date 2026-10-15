# C-HARM carto prototype: critique rounds

Each round: `node ramp.mjs` (prints the stops), `node shoot.mjs` (40 shots), then every shot was
looked at. Questions asked of every round: does the seafloor relief show through; is the 70–80 %
band differentiated; is the harbor zoom informative rather than enlarged pixels; do labels
collide; is the no-value hatch legible; what is the contrast of labels on the ramp.

Data in every shot: `charm_particulate_domoic_lead2`, valid 2026-10-15 (issued 2026-10-14).
Monterey Bay in this run sits at 73–82 %; Crescent City 62–96 %.

## Round 0 (first draft)

Ramp r0: OKLCH L 0.40→0.90, hue 285° → 345° → 75° (through pure red at 0.6), chroma up to 0.14.
L* 30.0, 36.3, 42.1, 48.2, 54.4, 60.7, 67.1, 74.0, 81.1, 88.2 (monotonic, min ΔL* 5.8).
Field opacity 0.72 (T1–T3), 0.8 (T4). Outlines light, 0.35 → 0.6 at z 10.5. Hatch: 1 px lines
every 8 px at 0.3 alpha (quiet), 0.62 (explicit).

Findings

- **Relief.** T0 (opaque purple) hides the seafloor entirely at every zoom: Monterey Canyon
  is invisible at Bay zoom. T1 at 0.72 lets the canyon, the shelf break and the isobaths read
  through at Regional and Bay; at Harbor there is little relief under the valued cells anyway.
  Good enough; not changed.
- **Alarm colour.** The r0 hue path crosses pure red (hue 0–15°) at the 0.5–0.6 classes with
  chroma 0.13–0.14, and the 0.7 class is a saturated coral. With most of the coast in 0.6–0.8,
  T1 Regional reads as a wall of red: an alarm, which a probability display must not be.
  → Round 1: chroma held at ~0.10 through the red hues (dusty rose / terracotta), rising only
  in the amber top.
- **70–80 % band.** Monterey Bay (73–82 %) is one flat tone in every treatment, as it must be
  with 10 display classes; only the z ≥ 11 value labels (T2/T3) and the click readout separate
  76 from 79. The 0.8 class patches (81–82 %) at the south of the bay do show as amber against
  the coral 0.7 class, so the class edge itself is visible. To be revisited in round 2 with a
  finer display-class experiment.
- **Harbor zoom.** T0/T1 at z 12: a straight cell edge, then hatch; the field is just big
  rectangles. T2 is better: the outlines make the rectangles read as *cells* and the value
  labels (77, 76, 79) make each one informative. T3 adds the point of the view: 6.6 km from the
  harbor to the nearest modelled cell, the gap named, the wharf station with its last sample.
- **Labels collide.** T3 Harbor: "Monterey harbor" sits on top of "Monterey Wharf · last
  sample", and the distance label on "Pacific Grove". T3 Crescent City: the gap note, placed
  30 % along a 3.4 km line, lands on the harbor label. → Round 1: data-driven anchors (harbor
  label below its point; distance label above the midpoint of the dashed line; gap note below
  it; station labels prefer right/left), station labels placed first.
- **Ring label.** The "15 km C-HARM context ring" label never appeared at Bay zoom:
  `symbol-spacing` 900 is longer than the ring's drawn length. → 250.
- **Hatch.** At Bay zoom the nearshore band of hatch is legible and quiet. At Harbor zoom (and
  on the phone at 2x) two thirds of the screen is hatch, and 1 px lines every 8 px at 0.3 read
  as loud stripes that compete with the field. → Round 1: 10 px spacing, 0.18 alpha for the
  quiet hatch (T1/T2), 0.5 for the explicit hatch (T3), 0.28 for the light map.
- **T4 light.** The dark ramp's amber top (L* 88) is as light as positron's sea
  (rgb(194,200,202), L* ≈ 80): the 0.9 class vanishes on the light map, and 0.8 nearly so.
  The dark cell outlines at 0.35 over the whole bay read as a net. → Round 1: a light variant of
  the ramp (same hue path, L 0.40→0.78) and outline opacity 0.2 → 0.45.
- **Label contrast.** White value labels on the 0.0–0.5 classes: contrast 9.4 → 3.1; from the
  0.6 class the dark ink (#0b1d33) is used instead: 6.8 → 12.5. All classes have a label colour
  above 3:1 (fine for 11–14 px numerals with a halo).
- **Phone.** The three control rows cover the top quarter, and the legend half the screen at
  Harbor zoom. → Round 1: short button labels ("T1", "T2", …) and a compact legend under 480 px.
- **Perf (Chrome, M-series laptop).** Grid gz 147 KB, fetch 11–51 ms, decode 4–13 ms; whole-grid
  image 1404×1970 px rendered in 14–19 ms, PNG 106 KB encoded in 18–74 ms; raster first paint
  24–60 ms after the style is applied for T0/T1 and 215–260 ms for T2–T4 (the cell-edge and
  cell-centre GeoJSON, 77k points, is built and tiled on the first apply; 22–34 ms to build).
  Viewport image (hatch + field) 2048×1280 px in 19–77 ms.

## Round 1

Ramp r1: L 0.42→0.89, chroma capped at ~0.10 through the red hues, hue 285° → 345° → 25° → 75°.
L* 32.5, 38.1, 43.9, 49.5, 55.4, 61.5, 67.6, 73.8, 80.3, 86.9 (min ΔL* 5.6). Light variant for T4:
L 0.40→0.78, L* 30.0 → 74.0. Hatch 10 px spacing, alpha 0.18 / 0.5 / 0.28. Harbor labels with
fixed anchors. Phone: short buttons, compact legend.

Findings

- **Alarm colour gone.** T1 Regional now reads as a dusty rose–terracotta field with amber only
  where the probability is highest (Southern California Bight, Crescent City); eddies and the
  shelf break show through at 0.72. This is the keeper.
- **Relief.** Unchanged and adequate at Regional and Bay (the Monterey Canyon axis and the
  200 m / 1,000 m isobaths read through the 0.7 class).
- **70–80 % band.** Still one tone, as expected; the amber 0.8 patches read as a distinct class
  but 0.8 and 0.9 were hard to tell apart (two tans). → Round 2: more chroma and a slightly
  yellower hue at the very top. A 20-class (5 %) experiment is added to see whether 73 / 76 / 79
  can be separated at Bay zoom without labels.
- **Labels.** Fixed anchors removed the Monterey harbor/wharf pile-up, but the collision
  engine then *dropped* the gap note at both harbors (it met "Pacific Grove" / the harbor
  label) and the harbor label itself at Monterey (it met "Monterey"). At Crescent City the
  distance label and the harbor label still touch over a 3.4 km (≈90 px) line. → Round 2: the
  three T3 annotations are always drawn (allow-overlap), the harbor label sits to the right of
  its point (over land), the distance label over the line at 75 % of the way to the cell, the
  gap note under the line at 40 %.
- **Hatch.** Quiet at Bay zoom; at Harbor on the phone (2x) the explicit hatch still dominates
  the view. → Round 2: 12 px spacing, explicit alpha 0.42, quiet 0.16.
- **T4 light.** The light variant restores the top classes (terracotta → amber on the grey-blue
  sea) and the 0.2 outlines no longer read as a net; the field at 0.8 looked heavy → 0.75.
  Measured: the light variant's 0.9 class is still only 1.2:1 luminance contrast against
  positron's sea; it separates by hue (amber vs grey-blue) and by the outlines, not by
  lightness, which is the inherent limit of a light-rising ramp on a light map.
- **Phone.** The bar is now two short rows; the legend is readable and leaves the harbor view
  free.

## Round 2 (final)

Ramp r2: as r1 with a little more chroma (0.135) and a yellower hue (80°) at the very top.
L* 32.5, 38.1, 43.9, 49.5, 55.4, 61.5, 67.7, 73.9, 80.5, 87.0 (min ΔL* 5.6). Hatch 12 px, alpha
0.16 / 0.42 / 0.28. T4 field opacity 0.75. T3 annotations always drawn, fixed anchors, and shown
only from z 10.5 (a later fix after the round-2 shots showed the Bay-zoom pile-up; the T3 Bay
shots were retaken). Experiment: `?classes=20` (5 % classes) shot for T2 at Bay, Harbor and
Crescent City (`shots/*-fine20.png`).

Findings

- **Relief shows through** in T1–T3 at Regional and Bay (canyon, shelf break, isobaths), and
  does not in T0. Settled.
- **Top classes.** 0.8 (apricot) and 0.9 (amber) now read as two classes at Crescent City
  (94–96 % cells amber, 89 % apricot) and in the Bight at Regional.
- **70–80 % band.** With 10 classes Monterey Bay is one tone; the z ≥ 11 value labels and the
  readout carry the within-class information, and at Harbor zoom (76 / 77 / 79 side by side)
  that is enough. The 20-class experiment at Bay zoom does separate 70–75 from 75–80 (the
  inner bay and the mouth become two visibly different tones, ΔL* ≈ 3) but the legend becomes
  twenty near-identical swatches and the step between adjacent classes is at the edge of what a
  screen shows reliably. Recommendation: keep 10 classes as the product's display classes; if
  the product wants a finer field, a *Bay-zoom-only* 5 % option is viable but should be a
  deliberate decision, not a default.
- **Harbor zoom is informative in T2/T3**: cells read as cells (outlines), each carries its
  value, the nearshore gap is named, the distance to the nearest modelled cell is explicit
  (6.6 km at Monterey, 3.4 km at Crescent City) and the wharf station shows when it was last
  sampled (2026-10-14: filled). T0/T1 at Harbor remain "enlarged pixels".
- **Labels.** Monterey Harbor: harbor label right of the point, wharf label above it, distance
  label over the dashed line at its seaward end, gap note on the hatched water: no collisions.
  Crescent City (3.4 km line ≈ 90 px at z 12): the distance label and the gap note touch;
  readable but crowded. Bay zoom (after the z 10.5 fix): only the ring, the stations and the
  nearest-cell marker; the ring label appears twice on the ring, which is fine.
  Phone T3 Harbor: the gap note sits at 40 % of a line that is partly off-screen, so its first
  word is clipped at the left edge. Known limitation; placing the note at the on-screen part of
  the line would need a viewport-aware placement.
- **Hatch.** Legible and quiet at every zoom on the desktop; on the phone at Harbor zoom the
  explicit hatch is still the dominant texture of the view because two thirds of the view is
  unmodelled water. That is the honest picture; the lighter "quiet" hatch of T1/T2 is the
  right default there, with the explicit hatch and the note reserved for T3.
- **Label contrast on the ramp.** White on the 0–40 % classes (8.5 → 4.6), dark ink on
  40–100 % (4.6 → 12.1); the switch happens at the 40–50 % class where both are 4.6:1. All
  classes clear 4.5:1, so the value labels are fine at 11–14 px with a 0.8 px halo.
- **T4 light.** Terracotta/amber field on positron, outlines 0.2/0.45, values in dark ink:
  legible; the 90–100 % class is still only 1.2:1 in luminance against the sea and relies on hue
  and outlines (measured, see README).
