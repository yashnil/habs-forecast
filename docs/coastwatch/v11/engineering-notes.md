# v11 engineering notes (facts the build depends on)

Collected from a read of `coastwatch-web/` at `feat/neritic-pilot` 9bbce0b7c. Paths are relative to `coastwatch-web/`.

## Map
- `src/components/map/MapCanvas.tsx` (client, 1138 lines) is the only map component. Embedded uses load it with `next/dynamic(..., { ssr: false })` after an IntersectionObserver fires: `src/v4/coast/HarbourMap.tsx` is the model to copy. It shows `ContourLoader` until `onReady`.
- One `MapCanvas` per page: it writes `window.__cwMap`, which the splash gate and tests read.
- Draws: C-HARM, outlook, SST and estimate rasters through the `cwgrid://` protocol (`src/lib/gridTileProtocol.ts`); satellite chlorophyll as XYZ tiles; currents as arrows or `FlowLayer`; official notice lines and labels; stations; ports; highlight marks; a footprint outline.
- Basemap (`src/lib/basemap.ts`): OpenFreeMap vector tiles plus static pre-shaded relief WebPs in `public/basemap/`. There is no DEM and no terrain. `isobaths.geojson` is coarse (5 depths).
- Pitch and bearing are not configured anywhere: MapLibre defaults apply. `minZoom` 4.2, `maxZoom` 12.5.
- Raster fade is `RASTER_FADE_MS = 220`; a day step cuts; reduced motion is 0. Tests pin these.
- Camera: `src/lib/camera.ts`. Tiers `none` (reduced motion), `short` (software GL or slow frames), `full`. A far pick must finish within 1850 ms (test).
- `src/components/map/FlowLayer.tsx` is a MapLibre custom WebGL2 layer in the map's own context: up to 900 streak particles, drops to 30 fps on weak devices, pauses itself when slow, hidden or out of observed cells, one still frame under reduced motion. It needs a `CurrentField` from `src/lib/currents.ts` (`loadFieldCached`).

## Today
- `src/app/today/page.tsx` is a server component (`force-dynamic`) that calls `loadToday(place)` and renders `TodayView` (one client island). `TodayView` renders its own `TopBar`.
- `PortTodayInputs` (`src/pilot/facts.ts`) has the place, notices, C-HARM days, outlook, fisheries and biotoxin, but not the manifest, ports collection or official geometry that `MapCanvas` wants. `loadData()` in `src/lib/data` provides those (see `src/app/page.tsx`).
- The hero's five curves (`TodayView.tsx` 213–219, `.today-hero-iso`) are invented. Remove them.
- `today.css` 228 has a shimmer skeleton; the system says no shimmer.

## Real geometry without a map
- `src/v4/brand/mark.geometry.ts`: `MARK_FIELD` (about 40 SVG path arms of real Monterey Canyon isobaths, viewBox `-126 -126 300 300`), `MARK_HERO`, `MARK_TIERS`. `src/v4/ui/ContourLoader.tsx` copies the 200 m arms; a unit test keeps them identical.
- `src/v4/coast/Landing.tsx` 231: the landing coast is 31 hand-picked points ("drawn, not measured").

## Motion that exists
- `src/v4/motion.ts`: `SPRING`, `T`, `EASE_OUT`, `EASE_IN`, `usePresence`, `exitProps`, `useSwapFade`, `markAppReady`/`onAppReady`.
- `src/v4/shell/MotionRoot.tsx` (in the root layout): `MotionConfig reducedMotion="user"`, the splash gate, the toaster.
- CSS in `src/v4/theme.css`: `.nx-rise`, `.nx-fade`, `.nx-panel` (12 px), `.nx-pop`, `.nx-veil`, `.nx-toast`, `.nx-late` (400 ms gate), `.nx-sound`, `.nx-chart-wait`/`.nx-chart-draw`, `.nx-saved-play`, `.nx-fold`, `.nx-press`, `.nx-tint`, `.nx-lift`, `.nx-focus`. `--nx-shift` is 0 under reduced motion.
- `TopBar` has its own nav pill glide (`useGlidePill`, WAAPI 260 ms, sessionStorage).
- No View Transitions anywhere. One root layout, no nested layouts: every page renders its own `TopBar`, and the map is re-created on every route.
- `src/v4/charts/SeriesChart.tsx` has `useDrawOnce` (IntersectionObserver, draws the line in 500 ms once).

## Tests that pin behaviour
- `tests/unit/motion.test.ts`: `--nx-t-press` 90ms, `--nx-t-release` 220ms, `--nx-t-layer` 220ms, `--nx-t-page` 200ms and their `T` mirrors; banned tokens `--cw-ease-settle`, `--nx-ease-in-out`, `--nx-t-quick`, `--nx-t-base`; spring overshoot ≤ 3%; `.nx-panel` 12 px and the literal `initial={{ opacity: 0, x: 12 }}` in Explore; no `M-20 ${` or `nx-sweep"`; `ContourLoader` equals `MARK_FIELD`.
- `tests/unit/design.test.ts`: palette monotonic, semantic colours differ in both themes, text tokens ≥ 4.5:1.
- `tests/unit/safety.test.ts`: forbidden phrases ("all clear", "safe to eat", "go fishing", …) anywhere in `src`.
- `tests/e2e/layout.spec.ts`: no sideways scroll at 320×640 and 844×390; phone controls ≥ 56 px; under reduced motion no running animation carries a transform, nothing loops, everything settles within 4 s.
- `tests/e2e/a11y.spec.ts`, `today.spec.ts`, `pilot-journey.spec.ts`: axe WCAG 2 A/AA; Today and the pilot journey allow zero violations of any impact. Daylight mode must pass too.
- `tests/e2e/motion.spec.ts`, `splash.spec.ts`: raster fade 220, day step cuts, one `canvas.maplibregl-canvas` on Explore, far pick ≤ 1850 ms, the splash's rules.
- `tests/e2e/today.spec.ts`: `[data-area]` order, `today-glance`, per-row `not-verified`, exact outlook words, offline reopen shows "Saved copy" and `sea-lanes`.
- Structure pinned elsewhere: `explore-shell.spec.ts` 31–62, `explore-rework.spec.ts`, `mycoast-v10b.spec.ts` 19–83, `v8.spec.ts` 56, `regulatory.spec.ts`, `navigation.spec.ts`, `research.spec.ts`. The full table is in `ia-audit.md` §8.
- A test that encodes a layout this pass changes on purpose is rewritten to assert the new layout and the honesty rule behind it. A test that encodes an honesty or accessibility rule is never weakened.

## Running tests
- `npm run lint && npm run typecheck && npm test` (vitest).
- Playwright: `npm run data:fixture && npm run build`, then `CW_E2E_PORT_BASE=<base> npx playwright test <spec>`. The config starts six `next start` servers on `base … base+5`. **Each worktree must use its own base** so parallel agents do not collide: system 3300 is taken by other sessions; use the base given in your brief.
- Fixture time is `CW_FIXED_NOW=2026-10-13T06:00Z`. Locally Playwright uses installed Chrome with a GPU.
- `next build` and `next dev` may leave an untracked `coastwatch-web/AGENTS.md`; delete it, never commit it.

## Offline
- `public/sw.js`: `/_next/static/*` is cached lazily; packs answer when the network fails.
- The Today pack (`src/pilot/today/offline.ts`) stores the page HTML, `/api/conditions` and the static files named in that HTML. A lazy map chunk and `public/` assets are not in it. Offline, Today must show the poster rather than a broken map.

## Fonts
- `src/app/layout.tsx` loads IBM Plex Mono, IBM Plex Sans, IBM Plex Sans Condensed and Newsreader through `next/font/google`.
- v2 components under `src/components/**` still use `--cw-*` tokens and v2 Tailwind colours, mapped onto `--nx-*` in `theme.css`.
