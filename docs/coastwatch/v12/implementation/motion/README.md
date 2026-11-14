# v12 motion: the clips and the audit

Recorded from the production build on fixture data (clock 2026-10-13) with Playwright `recordVideo`, Chrome,
at 1440 × 900 and 390 × 844, then re-encoded to VP9. Reproduce: `npm run build && node scripts/motion-clips.mjs`.
Every motion goes through `src/nx/ds/motion.ts` (README §8).

| Clip | What it shows | Presets |
|---|---|---|
| `opening-to-today-1440.webm`, `-390` | First visit at `/`: the lit canyon still, isobaths inked, tints flooding, the neatline closing (≤ 5 s, statement legible at frame one), Home; Find your port, "moss", Moss Landing; the name travels from the list into the chooser's title; a fishery; Today assembles while the chart pans to the port and the two soundings register | opening phases, `layout` (`pan` timing, on a paper halo), `pan`, `register` |
| `route-1440.webm`, `-390` | Today → Explore → My Coast → Research by the tabs, then back three times: the one chart pans to each page's view (0.62 s) while only the margin swaps (`marginSwap`, 0.18 s), on tabs and on back alike | `pan`, `marginSwap` |
| `route-reduced-1440.webm` | The same route with the system set to reduce motion: every page arrives in its final state, the camera jumps, no fade | reduced motion = final states |
| `worksheet-scenario-1440.webm`, `-390` | My Coast: the reader types catch, price, fuel, crew; the waterfall and figures settle as each number lands; the ±30 % / ±20 % grid previews a scenario in the waterfall (hover on desktop, tap on a phone) and returns to the reader's numbers | `settle` |
| `explore-frame-readout-1440.webm`, `-390` | Explore at Moss Landing with measured currents; a tap on the water registers the readout and inks the transect; choosing Oct 5 is a cut (frames never tween, T26) and the readout's figure settles from 0.48 to 0.54 mg/m³; back to the latest view. On a phone the sheet rises to half on its spring | `register`, `ink`, `cut`, `settle`, `sheet` |
| `research-brush-1440.webm`, `-390` | Research at Monterey Wharf: the strips ink in, the crosshair reads one visit across them (desktop), the window's end moves a year by key and its start by hand; every strip follows the brush, the window settling as a camera would | `ink`, `settle` |

## What the audit changed (this pass)

- **Back and forward** swapped the margin with a cut: React finishes a history traversal without a view
  transition. The margin now fades in with `marginSwap` on back and forward, as on a tab (`src/nx/shell/Margin.tsx`).
- **Today's pan was cut short**: its soundings refine the same scene a moment after the page, and the host
  jumped to the refinement. A refinement while a pan travels now glides on to the refined camera (`ChartHost`).
- **In-page links were full page loads** (Plan a trip, the notices links, Back to Today): ds `Button` with an
  internal `href` is now a Next link, so they pan and swap like the tabs. `/notices` was also missing from the
  root's list of v12 paths, which remounted the whole chart on every visit; fixed.
- **Reduced motion** was honoured piecemeal (some components checked it, others did not). `ds/MotionPrefs`
  sets Motion's global skip from the system setting, so every register, settle, ink, sheet spring and layout
  travel is its final state; the camera and the margin's view transition already followed the setting.
- **Readout figures** changed without motion; they now `settle`, like Today's row figures.
- **The port name's layout travel** used three different timings; all use `pan` now, and the travelling name
  rides above the page on a paper knockout halo instead of tangling with the lines it crosses.
- A new page starts at the top of the margin (it is its own scroller, out of the router's scroll reset).

Measured with a probe (CDP-free, `movestart`/`moveend` and `startViewTransition` counts) on every pair of the
four pages and the notices page, both directions and on back/forward, at 1440 and 390: every route change
pans 0.62–0.64 s and swaps the margin; under reduced motion every move is 0 ms. No stagger outside the opening's
isobaths.
