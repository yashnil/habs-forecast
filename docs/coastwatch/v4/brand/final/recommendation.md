# Neritic logo: final recommendation (R2 Bight or C2 Chart label)

**Recommendation: adopt C2 · Chart label. Stop iterating on R2.**

Evidence: `summary.png` gives the decision views on one page. `board.png` has everything: the desktop bar on the dark map and on the light reading page, the phone bar and dock at true size and 2x, favicons in tab strips with 4x/8x enlargements, app icons and masks, one colour, and grounds. Both candidates are the round-2 masters exactly as committed (`../round2/svg`, `../round2/icons`). No new variations were made.

## Why C2

1. **Glare at sea.** On every ground the app uses, C2 is a whole tile with a familiar glyph inside it. The light master puts fog on navy at 10.2:1. The dark master puts fog on sea blue at 3.5:1 and sits on the instrument surface at 4.4:1. R2's dark master uses the same pair, so contrast inside the tile is a tie. The difference is the outline:
   - R2's pale Bight runs out to the tile edge.
   - Against the white brand pill it is 1.35:1. Against a light inactive tab it is 1.20:1.
   - So on every light surface the lower-right quarter of R2 dissolves into the ground, and glare makes it worse.
   - Under the 45 % white-veil stand-in, C2 still reads as "n in a tile". R2 reads as a navy scrap on the light page and as a white blob on the dark map.
2. **Tab findability.** In the crowded strips, both candidates can be found on both tab colours. C2 is found faster. Its "-n-" shape is unlike any of the eight neighbours, the "a" tile included, because of the stubs. R2 is distinctive on a dark strip. On a light strip its pale half merges with the strip and leaves a navy notch. That is the folded-corner reading.
3. **Meaning for this audience.** R2 tells the better story, the shelf pinched at Point Conception and opening into the Bight, but only from about 32 px up. In the places people actually see it (a 32 px phone button and a 16 px tab) it is an abstract two-tone shape. C2's idea (a label set in a break of its contour) is how depth contours are labelled on nautical charts. Boaters and fishermen already read that convention, and it works at every size from 24 px up. At 16 px the n survives even where the italic does not.
4. **Coherence with the two environments.** C2's dark master is the same drawing re-tinted: a sea-blue tile with a fog n and dark stubs. So the mark looks like itself in Explore (dark instrument) and on My Coast and Research (light chart). R2's dark master inverts its own geography. Deep water becomes the pale area and the shelf becomes the darker one. In the dark-map bar it reads as a different, white mark. The C2 lockup's italic lowercase wordmark also matches the typographic system it brings: italic for water names, as on charts.
5. **Genre risk is smaller than the outline risk.** C2's risk is that it is a letter tile, the genre of the rejected N tile. R2's risk is the folded or peeled corner on light grounds and a lowercase r in one colour (fixable only with a mandatory rim). The letter-tile risk can be managed by rules (below). R2's corner is a property of the drawing on exactly the light grounds where the app spends most of its reading time, and fixing it would mean another round of variations.

## Risks of C2 and how to manage them

| Risk | Mitigation |
|---|---|
| Reads as a generic letter tile (the rejected N genre) | The contour stubs are part of the mark. Never ship the n without them, never set it upright, and never use a capital N. Use the lockup, not the bare mark, wherever there is room (desktop bar, about page, sharing card). |
| At 16 px the italic is gone: it is a pixel n with dashes, which earlier cuts read as "rt" or π | Ship the hand-drawn 16/24 px masters through `favicon.ico`, and declare no SVG favicon (see below). Run a 2-minute glance test with 3–5 people who have not seen the brief, at 16 px in a tab strip and at 60 px on a home screen. If people read π or "rt", redraw the 16 px master. That is a pixel fix, not a new concept. |
| The stubs are lost in a maskable (circle safe-zone) icon: the board shows the safe-zone crop cuts them off | Export `maskable-512.png` with the symbol scaled into the centre 80 % circle on a navy bleed, so the stubs end inside the safe zone. This is an export rule, not a new design. |
| The italic lowercase wordmark is quieter in the header than the current upright serif | This is a deliberate match with the chart-label system. At 30 px in the 44 px pill it is still clearly legible (board section 1, 2x crops). Keep `BRAND.name` "Neritic" for text, titles and aria labels. The lowercase belongs only to the drawn wordmark. |
| The dark master is only 3.5:1 inside the tile | Same as R2, and above the 3:1 non-text minimum. Do not lighten the tile further. If a sun-readable mode is ever added, use the light master there. |
| The name is not legally cleared (from the earlier rebrand notes) | Unchanged by this choice. Clear the name before the identity is installed on the public demo. |

If the owner's objection to letter tiles outweighs all of the above, the fallback is R2 as it stands, with these rules: always a tile, the one-colour version always keeps its 1 px rim, and on light grounds the mark always sits on a navy or white pill with a visible border. Do not open a round 3 in either case.

## What installing C2 would touch (not done here)

- **`coastwatch-web/src/v4/brand/Brand.tsx`**:
  - `Mark` becomes the C2 symbol in two tones: navy tile with fog n and shelf stubs on `env-chart`, and the sea-blue dark master on `env-instrument`. Use a tone prop or CSS variables, so the one file still serves every v4 surface.
  - `Wordmark`/`Lockup` use the outlined italic wordmark from `c2-lockup.svg`. Outlining avoids loading Literata as a web font for one word.
  - Remove `data-brand="provisional"`.
- **`coastwatch-web/src/components/brand/BrandMark.tsx`** (v2 masthead) and **`src/app/opengraph-image.tsx`** (sharing card) carry the current mark and must change in the same PR.
- **`coastwatch-web/scripts/build_brand_assets.mjs`** draws today's mark into every icon. It has to:
  - pack `favicon.ico` from the hand-drawn `c2-16.png` and `c2-24.png` plus 32/48 renders of `c2-favicon.svg`;
  - render `apple-icon.png` (180) and `public/icons/icon-192.png` / `icon-512.png` from `c2-app.svg`;
  - render `maskable-512.png` with the safe-zone padding above.
- **`coastwatch-web/src/app/favicon.ico`**, **`src/app/apple-icon.png`** and **`public/icons/*`**: regenerated by that script.
- **`coastwatch-web/src/app/icon.svg`**: this is the catch. Next emits it as an SVG favicon, and Chrome and Firefox prefer an SVG over the ICO, so the hand-drawn 16/24 masters would never be shown. Remove it from `src/app` (or replace it and accept the vector 16 px). `src/app/manifest.ts` lists `/icon.svg` as well and needs the same edit.
- **`coastwatch-web/tests/e2e/brand.spec.ts`** currently requires exactly one `link[rel="icon"][type="image/svg+xml"]` and a served `/icon.svg`. Those assertions change with the ICO-only favicon decision. Its `brand-home` checks (aria label, visibility, position) stay as they are.
- The public research preview is deployed separately. Installing the identity in `main` changes nothing there until an explicitly approved `vercel deploy --prod`.

## Files here

- `summary.png`: the decision on one page (1600 × 1255).
- `board.png`: the full evidence board (1760 px wide).
- `board.html`, `summary.html`, `src/parts.html`, `src/common.js`: sources. `src/render.cjs` takes the Chromium screenshots into `shots/` and renders both boards: `node docs/coastwatch/v4/brand/final/src/render.cjs`. This needs network access for Google Fonts (Atkinson Hyperlegible Next, Source Serif 4, IBM Plex Mono, the v4 faces). The candidate wordmarks are outlined paths and need no font.
- The dark map behind the floating bar is an illustrative bathymetric backdrop drawn in a canvas for this board. It is not a screenshot of the v4 map and not data.
