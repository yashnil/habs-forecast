# Neritic brand, round 2: Shelf band against Chart label

The owner asked us not to finalise B (Shelf band) yet, but to refine it against the strongest alternative and show the actual 16/24/32 px results. Every judgement below comes from Chromium screenshots (Playwright) at device scale 1 and 2. Enlargements use nearest-neighbour scaling, so each square is one screen pixel.

- `board.png` / `board.html`: the whole review, with favicons at true size and 8x, neighbouring tabs, one colour, dark grounds, 180 px app icon and nav lockups.
- `favicons.png`: every size (16 · 24 · 32) on light and dark tab colours at 8x, the retina render at 4x (the same physical size), and the same SVG rasterised without the hand-drawn masters.
- Sources: `src/gen2.py` writes everything; `src/masters.py` holds the hand-drawn 16/24 px grids; `src/render.cjs` takes the screenshots.
- Rebuild: `python3 src/gen2.py && node src/render.cjs`. Fonts are the Literata static instances in `../src/fonts` and stay out of git (download the `ital,opsz,wght` 0/1,24,600 and 1,24,500 TTFs from the Google Fonts CSS API).

## The strongest alternative is C, not A

The favicon is where B is being judged, and A loses there. After four passes, A's 16 px cut still reads as stripes with a plug; it needs three tints, and in one colour it becomes an H. A is the family's best illustration, not a better identity.

C is the opposite case. Its 16 px tile was the most legible thing on the round-1 board. It brings a complete typographic system: italic for water names and spaced italic capitals for editorial use. Its contour-break logic is real chart practice. Its weakness is exactly what makes it a fair test of B: at tab size C is a letter tile, the same genre as the rejected N tile. So the question for this round was whether B can be as findable as C at 16 px without becoming a letter, a road or a slash.

## Changes applied to every round-2 candidate (C2 included)

- **The tile fills the square.** Round 1 left 4 u of 48 as margin, which cost 2.7 px of drawing at 16 px.
- **The favicon tile is #1b5583,** one step lighter than brand navy #12395a. On a dark tab strip (#202124) round 1's navy tile edge disappeared and only the pale band showed. On B0 that is a bare backslash, which confirms round 1's fear. Brand navy is unchanged everywhere else.
- **The 16 px and 24 px favicons are hand-drawn pixel masters** (`icons/*-16.png`, `*-24.png`). The SVG serves 32 px and every 2x slot. **Shipping caveat:** Chrome and Firefox prefer an SVG favicon over an ICO when both are declared. The masters therefore only appear if the site serves `favicon.ico` (16/24/32/48) and declares no SVG favicon.
- **On dark grounds the B mark is always a tile:** pale solids with a sea-blue shelf. The loose two-shape version is retired.

## Candidates

### B0 · round 1 (reference)
- **16 px:** a pale stroke through a navy tile, read as a road, a backslash or a stylised N. On a dark tab the tile edge vanishes and what is left is a backslash.
- **24/32 px:** the same reading, larger. The bay and point appear only at 32 px and above.
- **Verdict:** the risks the owner named are real.

### R1 · Taper (the conservative fix)
The band runs top to bottom as before, but it widens from 2 px in the north to 4.5 px in the south. The deep-water edge is smooth, and the coast edge has one bay and one point, each pushed to a whole pixel.
- **16 px:** still one pale stroke corner to corner. The bay notch turns it into a zigzag or a lightning bolt.
- **24 px:** the taper is visible. It reads as a river or a crack.
- **32 px / 2x:** pleasant. The bay and point read as a coastline.
- **One colour:** an R or a stylised N.
- **Verdict:** taper and asymmetry do not change the genre at 16 px. Not enough.

### R2 · Bight (recommended B direction)
This uses the real geometry of California. The shelf is a narrow band in the north, pinched at Point Conception, then opens east into the Southern California Bight. The two solids are unequal: a large deep-water mass and a small land block in the top-right corner.
- **16 px:** a navy tile with a 2 px slit at the top that opens into a pale lower-right quarter. It is not a road, a slash or a letter, and it was the most distinctive silhouette among the neighbouring tabs on both tab colours.
- **24 px:** it starts to read as a coast that turns east.
- **32 px, 2x, 180 px:** a map. The Monterey-to-Bight story is legible, and it is the strongest app icon of the B family.
- **Risks:**
  - The pale Bight touches the tile's lower-right corner, so on a light tab that corner softens and can read as a folded or peeled page corner.
  - In one colour the two loose solids read as a lowercase **r**. The one-colour version must keep a 1 px tile rim; with the rim it is one shape with a cut, and the r is gone (board section 3).
  - Without a tile on dark grounds it also reads as an r, which is one more reason the tile is mandatory.
- **Verdict:** the only B variant that answers the road/backslash/N question at 16 px.

### R3 · Bight, closed south
R2 with deep water returning along the south edge, so the tile keeps a navy base and lower-left corner on light tabs.
- **16 px:** an L or boot shape, which reads more like a letter than R2 does.
- **24/32 px:** the shelf reads as a pipe or a sock.
- **One colour:** a reversed L. With the rim kept it becomes a tile with an L-shaped slot.
- **Verdict:** it fixes R2's corner but brings back a letter reading. Not preferred.

### R4 · Bight with Santa Cruz Island (tried, dropped; only in favicons.png)
- **16 px:** the island is a 3 x 1 px navy dash, which reads as a minus sign or a toggle.
- **Verdict:** noise at tab size. It could return as detail at 128 px and above.

### C2 · Chart label, refined (the alternative's fair pass)
The tile fills the square, the n is larger (46 u), the contour stubs run to the tile edge with a one-pixel step between left and right, and the 16/24 px masters put the n's stems on whole pixels. Each master needed three redraws: earlier cuts read as "rt" and then as π.
- **16 px:** an n with a dash on each side. It is the most legible favicon on the board and the quickest to find in a crowded strip. But at 16 px the italic has gone, so it is a pixel n, and the stubs read as dashes, not a contour.
- **24 px:** the italic returns.
- **32 px / 2x / 180 px:** the contour-break idea is clear. It is a handsome app icon.
- **One colour:** clean; the strongest one-colour mark here.
- **Nav lockup:** coherent by construction, but in a header the lowercase italic is quieter than the upright versions.
- **Risk:** it is a letter tile, the rejected genre. Beside the black serif "a" neighbour tile it belongs to the same family of marks.
- **Verdict:** the legibility benchmark B has to meet.

## Legibility test: 8 generic neighbouring tabs

The neighbours, chosen to be hard: a red mail tile, a navy circle with waves, a blue document, a sun, a black serif "a" tile, a green grid, a teal pin and a blue surf swoosh. Neritic sat fifth and inactive, with labelled tabs and crowded icon-only tabs on Chrome-like light and dark strips.

- **Distinguishable on both themes:** R2, R3 and C2. R2's half-pale tile is unlike any neighbour. C2 is findable because of its letter, although it resembles the "a" tile in kind.
- **Weak:** B0 and R1. Both are narrow pale strokes, and on the dark strip they shrink to their stroke.

## Recommendation

**Carry R2 · Bight forward as the B direction, but do not finalise it yet.** It is the only B variant that stops being a stroke at 16 px. It carries the name's meaning through real geography and is not a letter. **It does not beat C2 on raw 16 px legibility.** If that must win outright, C2 is the honest pick, at the price of being a letter tile.

One more short round is needed on R2 before an owner decision:
1. Make the rim-kept one-colour version the rule, and check it as a pinned-tab mask.
2. Try pulling the Bight 1 px off the tile's lower-right corner at 16 px, or deepen the shelf tone, so the outline holds on light tabs.
3. Cut a 32 px master to sharpen the point.
4. An informal glance test with a few people who have not seen the brief: what do the 16 px tab icon and the 60 px home-screen icon look like?

Nothing in this folder is installed in the app.
