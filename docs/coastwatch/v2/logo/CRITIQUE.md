# Neritic logo, round 2: critique log

Each round: regenerate (`gen.mjs`), screenshot (`shots.mjs`), look at `shots/board.png`, `shots/{1,2,3,R}-{1,2}x.png`
and `shots/favicons.png` (pixel-exact 16 and 32 px renders, blown up 6× nearest-neighbour), then write the notes
before changing geometry. The rejected Depth N is on the board as the reference column.

Reference check first: Neritic Diving (neriticdiving.com, Lauderhill FL) uses **two brush-stroke wave lines stacked
above hand-lettered all-caps "NERITIC"**. Rules derived from that: no stacked horizontal wavy lines, no calligraphic
caps, no word-under-lines lockup. Every contour here runs north–south, with the coast to the east, as California sits.

## Round 1 (first geometry)

**1 · Contour.** Four open isobaths, coast heavy (4.6 u) with a broad bay bulge, three deeper lines with similar tongues.
- Fails. The three deep tongues are the same shape at the same spacing, so the eye reads a repeated chevron texture:
  "fast-forward", "wind", "wifi on its side". Nothing says *chart*.
- Stepped line lengths (43 → 36 u) make it worse: that is the grammar of a wind/sound icon.
- Clearances measured by the generator: 2.6 / 1.8 / 1.1 u between neighbours, i.e. 0.87 / 0.59 / 0.35 px at 16 px. The
  two deepest lines fuse into one grey smear at 16 px on both grounds; at 32 px the teal lines are still a halftone.
- Ink box is 22 × 38 u, sitting right of centre (x 12–40). Narrow upright slab; in the 56 px header it collapses to a
  smudge next to a 9.6 px x-height.
- Change for round 2: three lines, not four. One *event*: the canyon as a long narrow finger on the deep line, the middle
  line cradling it, the coast nearly straight with a slight bay. Tune the two shoulder widths so the clear gap between
  neighbours is uniform (≥ 3 u = 1 px at 16 px). Same length for all three lines. Centre the ink box on 24.

**2 · Shelf.** Disc r 21, split once by the floor profile (deep flat left, 54° slope, shelf flat right), 4.4 u gap.
- Works at every size: a disc with a diagonal gap survives 16 px on both grounds (favicons.png row 2).
- But the two fillets are the same radius (4.2 u) and the slope is moderate, so the cut reads as a soft S: a generic
  fintech "split disc" rather than a shelf *break*. The geological signature is a flat shelf, an abrupt convex break,
  a steep slope, a soft concave foot.
- Shelf (right) is as long as the floor (left); the water piece is a thin cap on the right. Slightly top-left heavy.
- Change for round 2: per-corner fillets (break 2.8 u, foot 5.2 u), slope 60°, break moved right of centre so the
  shelf is the longer flat, gap 4.0 u. Keep the disc (the tile belongs to the app icon).

**3 · Wordmark-led.** Sora 600, hand-cut N with a stepped join, circular i-dots, first i-dot amber; the tick is a
single vertical isobath fragment with a tongue.
- The tick alone is a curly bracket. "} Neritic" in the header is unacceptable; in the app icon it is a bracket in a box.
- The amber i-dot reads as a typo or a notification badge, not as a sounding. Amber has not earned its place here.
- The cut N is good: the step at the top-left join and the foot gives the letter a crisp mechanical character without
  being a profile drawing. Diagonal thickness 1.02 × stem is right.
- i-dots at 1.1 × stem look light against Sora's heavy stems; the "ti" pair is a touch open.
- Change for round 2: remove amber. i-dots 1.2 × stem. Three tick candidates on the board: (a) vertical fragment, thin,
  cap-high; (b) a closed innermost-contour loop set after the word at x-height; (c) a notched rule under the word.
  Favicon and app icon become the N monogram knocked out of a tile; the tick is only for large lockups, per the brief.

**Reference (Depth N).** Still the most legible at 16 px of the four; its problem is conceptual (an N that is also a
profile is a pun, and the owner rejected it), not technical. That is the bar the new marks must clear at 16 px.

## Round 2

**1 · Contour** (three nested open tongues, auto-spaced to a uniform 3.3 u gap, same length).
- The spacing is now right (measured 3.30 / 3.30 u clear = 1.1 px at 16 px) and the ink box is centred (x 9.3–39.7).
- But the figure got *more* like an arrow, not less: a narrow finger with two lines cradling it is the "forward/share"
  glyph. In the dark header it is a play button. Nesting open tongues that all point east will always do this; the
  shoulders become chevrons however round the tips are. This is a dead end, not a tuning problem.
- Change for round 3: closed loops. Three nested isobaths around a bank, eccentric so the lines crowd on the east side
  (that crowding is how a chart shows a shelf break) and spread to the west, with one notch cut into the outer two
  loops from the west: the canyon. Closed loops cannot read as arrows or wind; the risk is "fingerprint", which at
  least is a topographic association. Weights 4.0 / 3.0 / 2.4 u, outer loop in ink, inner loops in sea.

**2 · Shelf** (per-corner fillets 5.4 / 2.6 u, slope 60°, gap 4.0 u).
- Now reads as a step, not an S: the tight convex break and soft foot are the geological signature. Mono is strong;
  16 px on both grounds is the cleanest of the four (favicons.png row 2).
- The water piece is a thin cap at the right (6 u of water over the shelf). Acceptable: the mark is about the ground.
- Still a split disc, a crowded genre. Its claim to ownership is the profile, so round 3 keeps it and only tunes the
  shelf length (break moved 0.6 u left so the flat shelf is clearly the longer run).

**3 · Wordmark-led** (N tile, i-dots 1.2 × stem, three tick candidates).
- The N tile is the most legible 16 px mark on the board after the disc; the stepped joins survive at 32 px and
  vanish gracefully at 16 px. Teal tile on the dark header is the right call; a light tile was too heavy.
- Wordmark: dots now sit with the stems; "Ne" could close by a further 0.004 em; "ic" is good.
- Tick candidates: (a) is still a bracket; (b) reads as a subset sign or a stray "c"; (c), the rule with one dip under
  the first i, is the only one that is cartographic rather than typographic noise. It also rhymes with direction 2's
  profile. Keep (c), set it at 0.4 × stem, one x-height below the baseline, dip 0.13 × x-height.

**Reference.** Depth N remains the 16 px benchmark; the N tile matches it, the disc beats it.

## Round 3 (closed loops; tick settled)

**1 · Contour** (three closed loops, notch on the outer two).
- It is a chart now: nobody will see wind or an arrow. Mono holds; the 4× blow-up of 16 px shows a bean with a ring.
- Two faults. The middle loop's notch turns it into a letter D, and the inner ring sits concentric with the middle oval,
  so the nest reads as an eye (bean, iris, pupil). Measured gaps show the "crowding" is not real yet: east 3.5 u clear,
  west 3.3 u, i.e. uniform.
- Change for round 4: push the middle and inner loops east so the east gaps sit at the 16 px limit (3.3 u) and the west
  gaps open to about 5 u; notch only on the outer loop with a 1.6 u echo on the middle; inner loop a vertical ellipse
  2.9 × 3.9 u, not a circle.

**2 · Shelf.** Break moved 0.6 u west; shelf flat is now 20 u against a 12 u floor. No further change.

**3 · Wordmark-led.** The rule tick at 0.4 × stem with a 0.13 x-height dip under the first i is quiet and cartographic.
"Ne" closed by 0.004 em more. The N tile at 16 px is as legible as Depth N. Settled.

## Round 4 (final tuning)

**1 · Contour.** East gaps 3.3 / 3.2 u, west 5.1 / 3.2 u, inner hole 3.4 × 5.4 u. The crowding on the east now reads as
a slope and the figure is clearly a chart excerpt. What remains, honestly: at 16 px it is a blob with a ring, closer
to a copyright sign than a chart; the favicon cut (two loops and a solid point) is better but still the weakest 16 px
mark on the board. The eye association is reduced, not gone. Mono is now one outlined path (Tiller–Hanson offsets of
each loop, even-odd), so the weight grading survives in a single black shape.

**2 · Shelf.** Unchanged. Best 16 px mark of the four on both grounds.

**3 · Wordmark-led.** Unchanged. The N tile is the second-best 16 px mark; the wordmark carries the brand at header size
without any symbol, which is the point of the direction.

## Round 5 (coordinator's request: replace Contour with 1b "Coast-cut contours", add 2b "Shelf with isobath")

Coordinator's read of round 4, accepted: closed loops read as an eye/fingerprint; the N tile is a generic app-letter
tile; Shelf is the cleanest but sits in the split-disc genre. Direction 1 (closed loops) is retired; its geometry
survives only in this log.

**1b · Coast-cut contours.** Built three times before it held together:
- First pass: three independent hooks side by side. Measured clearance between the mid and deep hairpins was −0.56 u,
  i.e. they overlapped: side-by-side hairpins cannot nest, because each line's north arm walls off the east. The only
  consistent nesting has each deeper hook passing through the opening of the shallower bump.
- Second pass: explicit nested hooks (tips 33.4 / 28.6 / 23.0 u, half-heights 13.5 / 8.6 / 2.8 u). Clearance 2.27 /
  0.92 u: the return curves had different slopes and converged under the deep tip. Squaring the hairpins made it worse
  (1.69 / 1.01 u). Parallelism cannot be tuned by hand here.
- Third pass: parallel offsets (Tiller–Hanson) of the deep hairpin. The first attempt degenerated (all hairpin radii
  2.8 u, clearance −2.4 u) because the base hook's shoulders were S-bends of radius 2.8 u, smaller than the 5.6 and
  10.45 u offsets. Rebuilt the base from arcs: straight north arm, an 11 u sweep shoreward, a 2.8 u semicircle around
  the canyon head, the sweep back, straight south arm. Offsets of 5.6 and 10.45 u (= half widths + 2.6 u clear) now
  stay valid; south arms splayed to exits 7.6 / 16 / 23.5 u so the lines spread south-west. Measured clear gaps
  2.19 / 2.14 u (1.46 / 1.43 px at 32 px; the shortfall from 2.6 is the Tiller–Hanson approximation on the inner side
  of the sweeps). Shallow line to coast 2.85 u. Tile corner radius dropped to 6 u so the deep line's bottom exit
  (band 5.85–9.35 u) still meets a straight edge.
- Result on the board: as a knock-out from the navy tile it reads as a chart excerpt (land, coast, nested isobaths) and
  the 16 px favicon cut (coast and one contour) is as legible as the N tile. On transparent, the land field floats as a
  capsule beside a striped ribbon, and the three parallel hooks read as a road or cable bend more than as isobaths;
  the tile version should be primary. Monochrome is one filled path (land + outlined strokes); the knock-out is one
  even-odd path.

**2b · Shelf with isobath.** One isobath knocked out of the water piece, parallel to the cut, 4 u above the gap.
- Built with the same offset-edge machinery (edges at −5.3 / −6.7 u from the profile centreline); the foot fillet had
  to open from 5.4 to 7.0 u, otherwise the isobath's offset of that convex corner has negative radius. That softens the
  ground's foot slightly relative to 2 (ground-edge radius 9.0 vs 7.4 u); acceptable, but 2 and 2b are not pixel-identical.
- At 128 and 48 px the line turns the mark from a split disc into bathymetry: the profile now has a neighbour and
  the eye reads depth. At 32 px, 1.2 u was a 0.8 px grey smear, so the line is 1.4 u (0.93 px at 32 px): visible,
  soft. At 16 px it is dropped (the favicon is plain 2), which is the right rule for the system.
- Mono: one even-odd path (ground + the three water offsets); the isobath is a hole.

**Board.** Five columns now; the 1b column has an extra cell for the knock-out tile at 96 / 32 / 16 px on a grey
ground. Column order fixed to 1b, 2, 2b, 3, R.
