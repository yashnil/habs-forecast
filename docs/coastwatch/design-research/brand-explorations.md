# Neritic identity explorations (v10)

Status: **exploration only, for the owner's decision. Nothing is installed in the app.** `src/v4/brand/*` and the icons are unchanged.

The owner has not approved the current mark: five nested Monterey Canyon isobaths in a rounded tile, with a Source Serif 4 wordmark. The brief asked for two or three stronger, original cartographic or bathymetric identities, and allowed refined uses of the canyon geometry. This document covers three. Each one is built from real NOAA bathymetry, rendered at every size the product uses, and compared with the current mark.

## Where to look

All files are in the session scratchpad, `…/scratchpad/v10-brand/`:

| File | What it shows |
|---|---|
| `compare-all.png` | **Start here.** The three identities and the current mark side by side: 16 px favicons (1× and 6×, light and dark tabs), 32 px, the desktop header at 1× (dark map and light page), the phone header at 30 px, the app icon, and one colour. |
| `I-headcut/board.png`, `II-sounding/board.png`, `III-gorge/board.png`, `current/board.png` | One full board per identity: favicons in tab strips at 1× and 8×; the header lockup at 28 px composited onto the real v10 header at 1× plus a 2× brand corner; the phone header; the 180 px apple-touch icon and the 512 px maskable icon, with an iOS squircle, an Android circle crop and the 80 % safe zone; the wordmark; one colour; light and dark masters; a hero. |
| `<identity>/*.svg` | SVG sources: symbol (light, dark, 16 px, mono), wordmark, lockup (light, dark, mono), `icon-180.svg`, `maskable-512.svg`. |
| `<identity>/png/*.png` | Every individual render. |
| `build.py` | Geometry and wordmark outlines, written to `geometry.json`. |
| `marks.mjs` | Colours and drawing rules. |
| `render.mjs` | Playwright renders and the boards. Rebuild with `python3 build.py && node render.mjs`. |

## Earlier rounds, and how these three differ

The repository already holds several logo rounds:
- **v4 round 1 and round 2:** Canyon tongues, the Shelf band, R2 Bight and C2 Chart label.
- **The design-excellence review:** A Shelf, B Depth N and C Soundings.
- **The current five-arch mark.**

All three new identities avoid repeating those:
- **Headcut** is not the five arches and not the round-1 "canyon tongues". It has one line, the 200 m isobath, used as the edge of a solid shape. It has no stroke and no tint layers.
- **Sounding** is not the earlier "Soundings", which showed an echo-sounder profile as dots along a schematic track. This is a north-south grid in plan view, and each dot's size is the real depth at that point.
- **Gorge** is not "A Shelf" or "Depth N", which were schematic coast-to-shelf profiles. It is a measured section *across* the canyon, and the 200 m horizon divides the water into neritic and oceanic. I did not redo a coast-to-shelf section: two earlier rounds covered that territory and the owner passed on both.

## Data and construction (shared)

- **Source:** NOAA NCEI U.S. Coastal Relief Model, vol. 6 (public domain).
- **Extract:** the same Monterey Bay extract the current mark uses, `coastwatch-web/scripts/brand/monterey-canyon-crm.csv`: 36.40–37.00 N, 122.45–121.75 W, every fourth 3″ cell, about 300 × 370 m.
- **Method:** smoothing is a Gaussian measured in km. Contours come from `contourpy`, polygon operations from `shapely`, and wordmark outlines from `fontTools` working on the variable fonts.
- **Stated generalisations:** every smoothing radius and every vertical exaggeration is listed below. No shape was redrawn by hand.

## I · Headcut (recommended)

**Idea.** The 200 m isobath is the edge of the neritic zone, and at the head of Monterey Canyon it cuts deep into the shelf. The tile is the shelf: water shallower than 200 m. The counter is the canyon: oceanic water cutting in. It reads as a lowercase **n** first and as a canyon second. It is one line drawn once, in one colour.

**Construction.** The frame is the current mark's frame: east up, turned a further 18°, a 9.2 km window centred on the canyon head. The 200 m surface is filled after 1.2 km smoothing for the full mark, or 1.6 km at 20 px and below. The outline is closed by ±0.9 unit and clipped to a tile with a 12/48 corner radius. The counter's apex sits 14.6/48 below the top. At 16 px the counter is 9.3 px wide near the bottom (y 40/48) and 2.5 px wide at y 20/48.

**Wordmark.** **Newsreader** SemiBold at the 72 pt optical size, tracked −6, outlined.
- Licence: SIL OFL 1.1 (Production Type), google/fonts commit 7b203a6.
- Why this face: its wedge serifs and display contrast echo the cut edges, it stays an editorial serif to match the reading pages, and it is firmer than Source Serif 4 at 28 px.

**System.**
- *Light master:* ink-navy tile `#0f2f4a`, with the counter open to the ground.
- *Dark master (the map):* the tile turns fog `#dbe9ee` and the dark map shows through the counter.
  - It reads as the same object, and the deep water really is dark.
  - Tile against the dark pill: **14.4:1**. The current mark's dark tile manages 1.9:1, so it sinks into the pill.
- *App icon:* a fog tile on navy, so the canyon opens onto deep water.
- *Signal cyan* (`#0b7a8f` light, `#6fdcec` dark) is kept out of the symbol and saved for UI accents.
- *Hero:* the bay's real isobaths with the symbol's window marked at the canyon head. This is its provenance device for About and the sharing cards.

**Tests (board I).**
- 16 px holds on all four tab colours and in one colour.
- 32 px is crisp.
- The 28 px lockup reads at 1× on both the map and the reading page.
- The 30 px phone button is the clearest of the four marks on both grounds.
- The 180 px and maskable icons keep the whole symbol inside the safe zone.

**Risks.**
- It belongs to the genre of an arch, doorway or tunnel glyph in a tile, and some viewers may see a ghost or a flame.
- The asymmetric right shoulder is real geometry and is what keeps it from being a generic arch.
- The lockup is 109 px wide at 28 px, against 91 px today, so the header pill grows by about 20 px (the composites show this).
- An n in a tile belongs to the letter-tile genre the owner rejected earlier with the N tile. The counter here is not a typed letter, which makes the risk smaller, but it is still there.

## II · Sounding

**Idea.** The canyon head as a survey sheet. Each dot is one sounding on a regular grid, and the dot's area is proportional to the real depth there: 47–487 m across the 5 × 5 grid. Soundings deeper than 200 m take the signal cyan. The identity is mostly typographic: the wordmark and every depth figure are set in the app's data face, and dot fields are the illustration language. The opening "Sounding" motion already uses this vocabulary.

**Construction.** It uses Headcut's frame, so the larger dots trace the counter. The full mark is a 5 × 5 grid over 36/48 units; 20 px and below use 3 × 3 over 33 units. Smoothing is 0.35 km, so each value stands for its cell. The minimum dot radius is 17 % of the grid pitch (27 % in the 3 × 3 grid).

**Wordmark.** **IBM Plex Mono** Medium, tracked −40, outlined.
- Licence: SIL OFL 1.1 (IBM), already the app's `--nx-font-data`.
- Why this face: the brand speaks with the voice of the instrument.

**System.**
- *Colours:* navy tile `#13324a`, shoal dots `#9fc6d2`, deep dots `#5fd6e9`. Both dot colours clear 7:1 on the tile.
- *Hero:* Monterey Bay as a 1.5 km sounding sheet with depth figures. It is the richest illustration system of the three.

**Tests (board II).**
- **It fails at 16 px.** The 3 × 3 grid turns into a blurred plaid on every tab colour, and in one colour it is a checkerboard. 32 px is acceptable.
- At 28 px the dots are 1–3 px and the mark reads as texture, not as a shape.
- The dark tile against the dark pill is 1.8:1, the same weakness as today.

**Risks.**
- Dot-matrix grids are a crowded genre among data and AI brands, and the geography only shows up from about 64 px.
- A 16 px master would have to be drawn by hand as a pixel pattern.

**Best use.** As the secondary system, whichever symbol is chosen: depth figures in Plex Mono, dot fields in the hero and the empty states.

## III · Gorge

**Idea.** Monterey Bay in section along 121.956° W, from shelf to canyon to shelf.
- The tile is the seabed and the water is the figure.
- The 200 m horizon divides the water: neritic water above it in fog, oceanic water below it in signal cyan.
- At large sizes it widens into a shore-to-shore section of the bay, from Santa Cruz to Pacific Grove.

**Construction.**
- The profile is sampled at 400 points over 18 km of latitude, with 0.3 km smoothing (0.6 km at 16 px).
- It maps 0–800 m into 41 units, a vertical exaggeration of about 19×. The hero is about 15×.
- Measured depths: north shelf 95 m, south shelf 108 m, canyon 686 m.

**Wordmark.** **Instrument Sans** SemiBold, tracked −12, outlined.
- Licence: SIL OFL 1.1 (Instrument), google/fonts commit 45e3d52.
- Why this face: a crisp grotesk, to match the engineered horizontal of the section.

**System.**
- *Colours:* seabed `#0c2c45`, neritic `#e3eef1` (12.2:1 on the seabed), oceanic `#4cc6dd` (7.1:1).
- *Device:* the dashed 200 m horizon can serve as a section divider in the reading pages.
- *Hero:* the strongest single image of the three.

**Tests (board III).**
- 16 px is legible.
- **But in one colour, and at 16–30 px, it reads as a letter "T" or a pen nib.** The neritic band is 95/800 of the depth, so only about 2 px thick at 16 px.
- The dark tile against the dark pill is 2.1:1.

**Risks.**
- The wrong letter.
- The two shelf blocks can read as a molar.
- If the vertical exaggeration is not stated, the section overstates the canyon.

## Resemblance check

I compared against the marks of the agencies and products named in the brief. The descriptions below are working notes, not a trademark search.

- **NOAA:** the emblem is a disc split by a white gull between dark sky and light sea. That is why no identity here uses a disc split by a line.
- **U.S. IOOS:** interwoven I-O-O-S letterforms, which its designers describe as an eye and a radiating signal, in blue and teal ([IOOS logo](https://ioos.noaa.gov/communications/the-ioos-logo/)).
- **Windy, Navionics (Garmin), Copernicus Marine:**
  - My working notes: these are wordmark-led brands. Windy uses a wind-flow motif; Copernicus uses an orbit and globe device ([Copernicus branding](https://atmosphere.copernicus.eu/branding-guidelines)).
  - I could not confirm their current marks from first-party pages in this pass. **Check them before anything is installed.**

None of these uses a figure-ground isobath cut, a depth-proportional sounding grid, or a measured cross-canyon section. The resemblance risks found are genres, not marks: arch tiles (I), dot matrices (II), and T or tooth shapes (III). The naming notes in `docs/coastwatch/brand/neritic.md` (Neritic Diving uses waves) still apply. None of the three uses wave artwork.

## Licences

- **Fonts:** Newsreader, IBM Plex Mono and Instrument Sans are all SIL OFL 1.1. Outlining a font into a logo is permitted. The SHA-256 of each font file is pinned in `build.py`.
- **Data:** NOAA CRM, public domain.
- **Work:** all geometry is original and generated by the scripts.

## Recommendation

**Adopt I · Headcut, and use II's typography and dot fields as the secondary system.**
- It is the only candidate that holds the same shape from 16 px to the hero.
- In one colour it is a single shape with nothing to drop out.
- Its dark-map master is the first in any round that does not sink into the pill (14.4:1 against 1.9:1).
- It refines the canyon idea the owner already responded to, with one real line instead of five.

Use Gorge's section as an editorial illustration, not as the mark.

**Decisions for the owner:**
1. Accept the arch-tile genre risk, or not.
2. Newsreader or keep Source Serif 4 for the wordmark. The geometry works with either.
3. Accept a header lockup about 20 px wider.

If Headcut is approved, installing it means regenerating `mark.geometry.ts` and changing `mark.ts`, `Brand.tsx`, `BrandMark.tsx`, the icon script and the brand e2e test, in a separate PR.
