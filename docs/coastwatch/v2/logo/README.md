# Neritic logo, round 2: bathymetric directions

Logo directions for Neritic drawn from the bathymetry of the California shelf, after the owner rejected "Depth N" (an
N drawn as a depth profile). Five iterations; the board now carries **four live directions** (1b, 2, 2b, 3) and the
rejected Depth N as a reference column. Every file is generated from hand-set geometry on a 48-unit grid by `gen.mjs`;
nothing is raster. The comparison board (`board.html`, screenshots in `shots/`) shows each direction at 16, 32, 48 and
128 px, in light and dark 56 px headers, in monochrome, as a horizontal lockup and as a 512 px maskable app icon.

Reference check: Neritic Diving (neriticdiving.com) uses two brush-stroke wave lines stacked above hand-lettered
all-caps "NERITIC". None of these marks uses wave lines, calligraphic caps or a word-under-lines lockup; every contour
runs north–south with the coast to the east, as California sits. Also avoided, by brief: radar arcs, compasses,
sparkles, arbitrary dots, decorative contour noise. The amber observation dot from the current `BrandMark` did not
earn its place (round 1 tried it as an i-dot: it read as a typo) and is dropped everywhere.

Retired: direction 1 "Contour" (closed eccentric loops) after round 4; it read as an eye or fingerprint. The round
log is in `CRITIQUE.md`.

## The directions

### 1b · Coast-cut contours (plan view)

The east third of a tile is land, a solid field with a slightly irregular near-vertical coastline. Three open isobaths
run parallel to the coast and all hook sharply inshore into one canyon head (the Monterey Canyon reading): they crowd
at the head and spread apart to the south-west. The lines are clipped flat by the tile's top and bottom edges and never
close. The deep line is drawn from arcs (an 11 u sweep shoreward, a 2.8 u semicircle around the head); the two
shallower lines are its exact parallels, which is what keeps the gap uniform where they crowd. Primary form: the
knock-out from a navy tile. Paired with Inter 600.

| | value |
|---|---|
| tile | 48 u, corner radius 6 u; land = east field, coast through (36.2,0) (35.3,9) (36.9,24) (35.2,39) (36.3,48) |
| isobaths, deep → shallow | strokes **3.5 / 2.5 / 2.0 u**; centreline offsets 0 / 5.6 / 10.45 u (= half widths + 2.6 u clear) |
| canyon head | deep hairpin radius 2.8 u at x 22.6; parallels 8.4 / 13.3 u at x 28.2 / 33.0; shallow tip to coast 2.85 u clear |
| arms | north arms x 8.6 / 14.2 / 19.0 u; south exits x 7.6 / 16.0 / 23.5 u (splayed, so spacing opens to the south-west) |
| measured clear gaps | 2.19 / 2.14 u at the head (1.46 / 1.43 px at 32 px) |
| colour | land ink, lines sea; dark header: land #eef3f8, lines #5fc0d8; knock-out tile: navy with land and lines transparent |
| monochrome | one filled path (land + outlined strokes); the knock-out tile is one even-odd path |
| favicon cut (≤ 16 px) | the coast and one contour (3.4 u, offset 5.2 u), knocked out of the tile on light grounds, plain on dark |
| lockup | symbol height 1.5 x-heights = **1.13 cap heights** of Inter; gap 0.5 x-height |
| wordmark | Inter 600, tracked −0.015 em; pairs Ne −0.004, ri −0.004, ti −0.002, ic −0.004 em; i-dots as circles at 1.10 ×, lifted 0.01 em |
| app icon | full-bleed: the tile is the icon (land and lines in light on navy) |
| minimum size | 32 px for three lines; 16 px with the favicon cut |

### 2 · Shelf (section view, negative space)

A solid disc cut once by the seafloor profile, coast to the east: the shelf (flat, shallow, right), the break, the 60°
slope and the deep floor (left). Above the cut is the water, below it the ground. The gap is a true offset of constant
width built from lines and circular fillets, so both pieces are closed filled paths and the monochrome is a single
path. Paired with Source Serif 4 600.

| | value |
|---|---|
| disc | r 21 u, centre (24, 24) (diameter 0.875 of the grid) |
| cut | gap **4.0 u** (0.095 of the diameter; 1.3 px at 16 px); centreline shelf y 17.4, floor y 31.2, rise 13.8 u |
| slope | 59.9°, from the foot (15.6, 31.2) to the break (23.6, 17.4) |
| fillets (centreline) | foot 5.4 u (soft), break 2.6 u (tight); ground edge 7.4 / 0.6 u, water edge 3.4 / 4.6 u |
| flats | shelf ≈ 20 u, floor ≈ 12 u |
| colour | water #1d7894 over ground #0b2238; dark: #5fc0d8 over #eef3f8; mono: one ink |
| lockup | disc height 1.9 x-heights = **1.35 cap heights** of Source Serif 4; gap 0.5 x-height |
| wordmark | Source Serif 4 600, tracked +0.002 em; Ne −0.006 em; i-dots as circles at 1.06 × |
| minimum size | 12 px |

### 2b · Shelf with isobath

The Shelf disc with one thin isobath knocked out of the water, parallel to the cut, 4 u above the gap, so the mark
reads as bathymetry rather than a split disc. Built from the same offset edges; the foot fillet opens from 5.4 to
7.0 u so the isobath's offset of that convex corner keeps a radius (the ground's foot is therefore slightly softer than
in 2). Dropped at 16 px: the favicon is plain 2.

| | value |
|---|---|
| isobath | **1.4 u** wide, edges at 5.3 and 6.7 u above the profile centreline (3.3 u of water between it and the gap) |
| fillets (centreline) | foot 7.0 u, break 2.6 u; on the isobath 0.3 / 9.3 u |
| 32 px | the line is 0.93 px: visible, soft. 48 px and up: clean |
| monochrome | one even-odd path: ground + three water offsets, the isobath is a hole |
| everything else | as 2 |

### 3 · Wordmark-led

The letterforms carry the identity. "Neritic" in Sora 600 with a hand-built N: two stems, a diagonal of 1.02 × stem
with flat cuts at cap height and baseline, and a step at each join (the diagonal's upper edge leaves the left stem
0.1 cap below the top, mirrored at the foot of the right stem). Both i-dots are true circles at 1.2 × stem. The
favicon and app icon are the same N knocked out of a tile. The complementary cartographic mark is one rule under the
word with a single dip under the first i, the canyon, used only in large lockups; the header carries the wordmark alone.

| | value |
|---|---|
| wordmark | Sora 600: cap 0.730 em, x-height 0.534 em, stem 0.139 em; tracked −0.02 em; pairs Ne −0.008, er −0.002, ri −0.004, ic −0.006 em |
| N | width = cap height, stem 0.19 cap, diagonal 1.02 × stem, step 0.10 cap |
| i-dots | circles, r = 0.6 × stem, lifted 0.012 em |
| tick (rule) | 0.4 × stem thick, 0.3 x-height below the baseline, dip 0.13 x-height over a 0.44 x-height span |
| tile | 48 u, corner radius 11 u; N cap 24 u (0.5 tile), stem 4.57 u (1.5 px at 16 px); ink tile light, sea tile on dark; mono: ink tile, N transparent (even-odd) |
| lockup | tile height 1.37 x-heights = **1.00 cap height**; gap 0.5 x-height |
| minimum size | tile 14 px; wordmark x-height 7 px |

### Clear space (all)

One x-height of the paired wordmark on every side at the size in use (0.67 × symbol height for 1b, 0.53 × disc for 2
and 2b, 0.73 × tile for 3). Nothing else inside it, including the nav.

## What works at 16 px (`shots/favicons.png`: pixel-exact 16 and 32 px renders, 6× nearest-neighbour)

- **2 / 2b** (2b falls back to 2): cleanest on both grounds; the gap stays open at 1.3 px, the step is still a step.
- **3** N tile: as legible as the rejected Depth N; the N stays an N at 1.5 px stems.
- **1b** favicon cut (tile, coast and one contour): legible as "a notch in a tile"; the three-line mark itself needs 32 px.
- **Depth N** (reference): still good at 16 px; its problem was conceptual.

## Ranking (all five, honest)

1. **2b · Shelf with isobath**, with 2 as its ≤ 24 px cut. One isobath is enough to move the disc out of the split-disc
   genre: at 48 px and above it is unmistakably a depth section, and the profile (flat shelf, tight break, soft foot)
   is the ownable feature. It keeps everything 2 had (single-path mono, best 16 px behaviour via the fallback) and costs
   only a softer foot. This is the one I would take forward. Check it next to fintech discs before committing.
2. **2 · Shelf.** The same mark without the line: simpler, cleaner at 32 px, but more generic. Second only because 2b
   is the same identity with the genre problem answered.
3. **3 · Wordmark-led.** The most refined typography and the most pragmatic system (no symbol in the header, one weight of
   one face). The N tile is a generic app-letter tile, as the coordinator says; the system holds because the wordmark,
   not the tile, is the identity. The rule tick is optional.
4. **1b · Coast-cut contours.** The most literal bathymetric chart and the strongest app icon of the set as a full-bleed
   tile; the 16 px cut works. It ranks below 3 because on transparent the land floats as a capsule beside a striped
   ribbon (the three parallel hooks read as a road or cable bend), it needs 32 px for the full figure, and the measured
   head clearance (2.2 u) is at the limit. If it goes forward: tile-only usage, and consider two isobaths instead of three.
5. **Depth N** (rejected): technically fine at every size; conceptually a pun the owner does not want. Reference only.

The critique of each round, with the measurements that drove the changes, is in `CRITIQUE.md`.

## Files

```
docs/coastwatch/v2/logo/
  gen.mjs            generator: geometry, outlining, type, board.html, out/*.svg
  shots.mjs          Playwright: shots/board.png, {1b,2,2b,3,R}-{1,2}x.png, fav-*-{16,32}.png, favicons.png
  board.html         comparison page (generated)
  CRITIQUE.md        round-by-round critique with measurements
  fonts/             OFL fonts fetched at runtime (gitignored)
  out/
    {1b,2,2b,3}-symbol.svg, -symbol-mono.svg, -symbol-dark.svg   48 × 48: colour, single black path, dark header
    {1b,2,2b,3}-wordmark.svg, -wordmark-dark.svg                 symbol + "Neritic" outlined to paths
    {1b,2,2b,3}-wordmark-only.svg                                the wordmark alone
    {1b,2,2b,3}-icon-512.svg                                     512 maskable app icon
    1b-symbol-tile.svg, 1b-symbol-tile-mono.svg                  the knock-out from a navy tile (one even-odd path)
    1b-favicon.svg, 1b-favicon-dark.svg                          16 px cut: coast and one contour
    2b-favicon.svg, 2b-favicon-dark.svg                          16 px cut: plain Shelf (isobath dropped)
    3-wordmark-tick.svg, 3-wordmark-tick-dark.svg                wordmark with the contour tick (large lockups only)
    R-*.svg                                                      rejected Depth N, reference only
  shots/             board, per-direction crops at 1× and 2×, pixel-exact 16/32 px favicon renders, favicons.png
```

## Regenerating

From `coastwatch-web` (Node 22, Playwright with the installed Chrome channel):

```sh
npm i --no-save opentype.js                     # not a project dependency
cd ../docs/coastwatch/v2/logo/fonts             # OFL fonts, fetched at runtime, never committed
curl -sLo sora-500.ttf https://fonts.gstatic.com/s/sora/v17/xMQOuFFYT72X5wkB_18qmnndmSdgnn-K.ttf
curl -sLo sora-600.ttf https://fonts.gstatic.com/s/sora/v17/xMQOuFFYT72X5wkB_18qmnndmSeMmX-K.ttf
curl -sLo inter-600.ttf "https://fonts.gstatic.com/s/inter/v20/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuGKYMZg.ttf"
curl -sLo sourceserif4-600.ttf https://fonts.gstatic.com/s/sourceserif4/v15/vEFy2_tTDB4M7-auWDN0ahZJW3IX2ih5nk3AucvUHf6OAVIJmeUDygwjisltrhw.ttf
cd ../../../../../coastwatch-web
node ../docs/coastwatch/v2/logo/gen.mjs && node ../docs/coastwatch/v2/logo/shots.mjs
```

(URLs from `https://fonts.googleapis.com/css2?family=Sora:wght@500;600`, `Inter:wght@600`, `Source+Serif+4:wght@600`;
refresh them from the CSS if Google rotates the file hashes.) Sora, Inter and Source Serif 4 are SIL Open Font License;
the outlined wordmarks may be used in a logo under the OFL.
