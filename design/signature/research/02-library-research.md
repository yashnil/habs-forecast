# CoastWatch motion and interaction tooling: research report (as of 2026-10-12)

**How the sizes were measured.** Unless a row cites a vendor figure, I measured the gzip sizes myself. I bundled each import set with esbuild 0.x using `--bundle --minify --format=esm` and React external, then ran gzip -9. That is a different method from bundlephobia or the vendors' Rollup numbers, so treat the figures as ±15% and use them to compare options. Versions, dates and licenses come from the npm registry and repo activity comes from the GitHub API, both queried today. No project files were touched.

## 1. Motion (motion.dev), recommendation: adopt narrowly

- **Current version:** `motion` 14.1.0, published 2026-10-09, MIT. The React peer range is `^18 || ^19`. Releases are frequent: 13.0.0 on 2026-08-05, 14.0.0 on 2026-10-02, 14.1.0 on 2026-10-09. 14.0 only removes internal APIs that 13.5.1 had restored for compatibility. 13.0 removed the optional `@emotion/is-prop-valid` dependency, which you now pass through `<MotionConfig isValidProp>`. 13.1.1 fixed `AnimatePresence` under React 19 strict mode. 13.4.0 added `AnimateView`, which builds on React 19.3's `<ViewTransition>`. Source: https://motion.dev/docs/changelog
- **Vendor sizes** (Rollup, from https://motion.dev/docs/react-reduce-bundle-size): the full `motion` component is 34 kB. `m` plus `LazyMotion` is about 4.6 kB for the first render. The `domAnimation` feature package adds 15 kB and `domMax`, which is needed for layout animations and drag, adds 25 kB. The `useAnimate` mini build is 2.3 kB.
- **My esbuild measurements, gzip:**
  - `motion.div`: 41.5 kB.
  - `m` + `LazyMotion` alone: 5.6 kB.
  - `m` + `LazyMotion` + `domAnimation` + `AnimatePresence` + `useReducedMotion`, imported statically: 29.4 kB. Load `domAnimation` with a dynamic import to keep it out of the main chunk.
  - `motion/mini` `animate`: 3.4 kB. `motion/react-mini` `useAnimate`: 3.5 kB.
- **Reduced motion:** `<MotionConfig reducedMotion="user">` turns off transform and layout animations when the OS setting is on, and keeps opacity and colour animations. `useReducedMotion()` returns a boolean. Source: https://motion.dev/docs/react-accessibility. The docs say nothing about what it returns during SSR (unverified), so assume `false` on the server and avoid hydration-dependent markup.
- **SSR:** these are client components, so they need `"use client"`.
- **Verdict:** use `m` + `LazyMotion` + `domAnimation` (loaded lazily) only where you need interruptible spring or presence animation, such as panel enter and exit or sheet drag physics. Do not use `domMax` or layout animations unless one specific screen needs them, because they are the expensive part. For one-off imperative tweens, `motion/mini` (about 3 kB, WAAPI-based) is enough. WAAPI and CSS transitions are already interruptible and run on the compositor for transform and opacity, so much of this needs no dependency.

## 2. View Transitions API and React `<ViewTransition>`, recommendation: adopt narrowly

**Browser support:**
- **Same-document transitions** are Baseline Newly Available since Firefox 144 shipped in October 2025. Chrome has supported them since 111 and Safari since 18. Source: https://web.dev/blog/same-document-view-transitions-are-now-baseline-newly-available
- **Cross-document transitions** are in Chrome and Edge 126+, Safari and iOS Safari 18.2+, and Firefox only partially (144–160). Global support is about 88.8%. Source: https://caniuse.com/cross-document-view-transitions
- **`view-transition-class`** is in Chrome 125, Safari 18.2 and Firefox 144, about 90% global. Source: https://caniuse.com/mdn-css_properties_view-transition-class
- **Nested groups (`view-transition-group`)** are Chrome 140+ only. Safari and Firefox do not support them, about 70% global. Source: https://caniuse.com/mdn-css_properties_view-transition-group. Do not depend on nested groups.

**React:**
- `<ViewTransition>` became stable in **React 19.3**, released 2026-09-09. It was not stable in 19.2, which shipped `<Activity>`. Source: https://react.dev/blog/2026/09/09/react-19-3
- It animates only updates marked as Transitions: `startTransition`, Suspense reveals and `useDeferredValue`.
- It supports `enter`, `exit`, `update` and `share` classes, plus `addTransitionType`.
- The release post does not cover reduced motion, so you have to handle it yourself, for example with `@media (prefers-reduced-motion: reduce) { ::view-transition-group(*), ::view-transition-old(*), ::view-transition-new(*) { animation: none } }`.

**Next.js:**
- Next 15.5 docs describe `experimental.viewTransition` as "experimental… not recommended for production" and import `unstable_ViewTransition`. Source: https://nextjs.org/docs/15/app/api-reference/config/next-config-js/viewTransition
- Next v16.3.0-canary.95 (2026-07-23) removed the flag as "inert". Source: https://releases.sh/release/rel_eHrDQECWkdHXLAOj-k6Wq-viewtransition-flag-removed-generatestaticparams-validation-hardened. I could not confirm whether a stable Next 16.3 has shipped.

**Verdict:** on your current stack, Next 15.5 with React 19.1, React's component is only available through the experimental flag. Do not adopt it there. Calling `document.startViewTransition` directly behind a feature check, for things like layer or tab swaps, costs nothing and is safe. Revisit `<ViewTransition>` once you upgrade to React 19.3+ and a Next version without the flag. Never wrap the MapLibre canvas in a view transition: snapshotting a WebGL canvas is costly and freezes the map.

## 3. Modern CSS motion features: browser support

| Feature | Chrome | Safari / iOS | Firefox | Global | Source |
|---|---|---|---|---|---|
| `@starting-style` | 117 | 17.5 | 129 | 93.1% | https://caniuse.com/mdn-css_at-rules_starting-style |
| `transition-behavior: allow-discrete` | 117 | 17.4 | 129 | 93.1% | https://caniuse.com/mdn-css_properties_transition-behavior |
| `linear()` easing | 113 | 17.2 | 112 | 94.0% | https://caniuse.com/mdn-css_types_easing-function_linear-function |
| Scroll-driven `animation-timeline: scroll()` | 115 | **26.0** | **160 only** | 87.8% | https://caniuse.com/mdn-css_properties_animation-timeline_scroll |
| `interpolate-size` | 129 | **no** | **no** | 71.7% | https://caniuse.com/mdn-css_properties_interpolate-size |
| `calc-size()` | 129 | **no** | **no** | 71.7% | https://caniuse.com/mdn-css_types_calc-size |

**Verdict:**
- **Ready to use:** `@starting-style`, `allow-discrete` and `linear()`. They cover enter and exit of `<dialog>`, popovers and `display:none` panels with no JavaScript.
- **Progressive enhancement only:** `interpolate-size` and `calc-size()`, because Safari and iOS do not support them. For height-to-auto accordions, use the `grid-template-rows: 0fr → 1fr` technique, or the CSS variables your primitives library exposes, such as Base UI's `--accordion-panel-height`.
- **Scroll-driven animations** now work on iOS 26+. Older iOS releases are still in use, so wrap them in `@supports (animation-timeline: scroll())` and keep content fully readable without them.

## 4. MapLibre GL JS, recommendation: adopt fully (no new dependency)

**Version and v6 changes:**
- **Current version:** 6.14.0, published 2026-10-11. 6.13.0 was published 2026-10-06 and 6.0.0 on 2026-07-22.
- **Breaking in 6.0:** the build is ESM-only, WebGL2 is required, and `Map` now composes `Camera` instead of extending it. `map.transform` was removed and events are now real classes. Feature-state was optimised, up to 3.4× faster. New `fill-layer-opacity` and `line-layer-opacity` properties were added. Source: https://github.com/maplibre/maplibre-gl-js/blob/main/CHANGELOG.md

**Changes in 6.x that matter for animation:**
- **6.14:** `flyTo` now keeps `offset` and `around` under reduced motion, so the reduced-motion jump lands where the animated flight would.
- **6.12:**
  - Raster tiles cross-fade when zooming out past a source's `maxzoom`.
  - New `absolutePadding` option for `fitBounds` and `cameraForBounds`.
- **6.11.x:**
  - Paint, light and sky properties that read `global-state` now transition from their previous value.
  - New `calculateAnchoredCameraOptions`.
  - The globe atmosphere fades with camera altitude.
  - Fixed `fitBounds` throwing on the globe when padding exceeds the viewport.
- **6.9.1:**
  - Raster tiles no longer stick at the start of a fade-in.
  - Setting one paint property no longer transitions all of them or delays `idle`.

Source: https://github.com/maplibre/maplibre-gl-js/releases

**Reduced motion:**
- `flyTo`, `easeTo` and the other camera animations behave like `jumpTo` when the OS reduced-motion setting is on, unless you pass `essential: true`. Source: https://maplibre.org/maplibre-gl-js/docs/API/type-aliases/AnimationOptions/ and https://maplibre.org/maplibre-gl-js/docs/API/classes/Map/
- Internally this reads `browser.prefersReducedMotion`, a lazy `matchMedia` call with a setter override, in `src/util/browser.ts`. I found **no public export** of that override in `src/index.ts`. If you want an in-app "reduce motion" toggle, pass `animate:false` or `duration:0` yourself.
- Inertia on Mac and the geolocate pulse also respect the setting.

**Camera options:**
- `flyTo` takes `curve`, `speed` (default 1.2), `screenSpeed` (ignored if `speed` is set), `maxDuration` (if the computed duration exceeds it, the duration becomes 0, which means a jump) and `minZoom`. Source: https://maplibre.org/maplibre-gl-js/docs/API/type-aliases/FlyToOptions/
- `freezeElevation` keeps the camera's sea-level height constant over 3D terrain.
- `fitBounds` and `cameraForBounds` take `padding` as a number or `{top,bottom,left,right}`. Pass the bottom-sheet height as `padding.bottom`, or use `offset`. `cameraForBounds` returns undefined and warns if the bounds cannot fit.

**Interruption:**
- `map.stop()` "Stops any animated transition underway."
- The handler manager calls `camera.stop(true)` when a user gesture starts, so any user gesture cancels a programmatic flight.
- `isMoving()` reports camera animations and gestures.

**Telling user moves from programmatic ones:** `MapMovementEvent` is typed `MapLibreEvent<MouseEvent | TouchEvent | WheelEvent | undefined>`. `originalEvent` is set for gesture-driven `movestart` and is undefined for `flyTo`/`easeTo`, unless you pass your own `eventData`. Confirmed in source `src/ui/events.ts` and `handler_manager.ts`. Use this to stop "follow" behaviour or URL sync loops.

**Paint transitions:**
- Paint properties marked "Transitionable" in the style spec accept a `*-transition: {duration, delay}` value, and `raster-opacity` is one of them. Set `raster-opacity-transition` before calling `setPaintProperty('…','raster-opacity',x)` to cross-fade C-HARM or chlorophyll layers on the GPU.
- `raster-fade-duration` controls the per-tile fade-in; the default is 300 ms per the spec, which I could not re-fetch, so treat that as unverified. Set it to 0 under reduced motion.
- **Hover:** `setFeatureState` plus `['feature-state','hover']` expressions in `fill-color`, `line-width` and `circle-radius` repaint without React re-renders. Features need ids, via `promoteId` or `generateId`. Source: https://maplibre.org/maplibre-style-spec/layers/

## 5. GSAP, recommendation: avoid

- **Current version:** 3.15.0, published 2026-04-13. The GitHub repo was last pushed 2026-04-13.
- **License:** the "Standard 'No Charge' GSAP License". After the Webflow acquisition it is free for commercial use including all plugins (SplitText, ScrollTrigger, MorphSVG), but it is **not OSI open source**. It also bars use in no-code visual animation builders that compete with Webflow. Sources: https://gsap.com/standard-license and https://cdn.jsdelivr.net/npm/gsap@3.15.0/README.md
- **Size, measured:** core is 28.3 kB gzip, ScrollTrigger 18.0 kB and SplitText 3.7 kB. Core plus ScrollTrigger plus `@gsap/react` bundles to 45.5 kB.
- **React integration:** `useGSAP` from `@gsap/react` is SSR-safe because it uses an isomorphic layout effect and needs `"use client"`. It cleans up automatically through `gsap.context`, and `contextSafe` handles animations created in event handlers. Source: https://gsap.com/resources/React/
- **Verdict:** about 45 kB plus a proprietary license is not justified for limited editorial storytelling. Use `IntersectionObserver` with CSS transitions, and add `animation-timeline` behind `@supports` as an enhancement.

## 6. UI primitives

**Current status of each library:**
- **Radix:** `radix-ui` 1.8.1 and `@radix-ui/react-dialog` 1.3.1, both published 2026-10-12, MIT. The repo is active. Secondary sources describe a slower pace since the WorkOS acquisition, with long-open Combobox issues. Source: https://www.pkgpulse.com/guides/shadcn-ui-vs-base-ui-vs-radix-components-2026
- **Base UI (`@base-ui/react`):**
  - 1.9.0 published 2026-10-09, MIT, released roughly monthly.
  - v1.0.0 shipped 2025-12-11 and Drawer became stable in v1.3.0 (2026-03-12). Source: https://unpkg.com/@base-ui/react@1.6.0/docs/react/overview/releases.md
  - shadcn/ui made Base UI the default for new projects on 2026-07-03, according to secondary sources I did not confirm against shadcn's changelog. Source: https://blog.openreplay.com/shadcn-ui-radix-base-ui-switch/
- **React Aria Components:** 1.22.1 published 2026-10-09, Apache-2.0, the strongest accessibility rigour of the three. All three accept React 19 as a peer.

**Measured gzip size, React external:**

| Import | Radix | Base UI | React Aria Components |
|---|---|---|---|
| Dialog only | 14.2 kB | 23.9 kB | 28.5 kB (with Modal, Trigger, Button) |
| Dialog + Popover + Tooltip + Tabs + ToggleGroup + Slider + Accordion | 40.0 kB | 65.5 kB | 56.0 kB (Disclosure in place of Accordion) |

**Verdict:**
- **Base UI** for new work. It is the actively developed successor by the original Radix authors and includes a Drawer with snap points, which unifies item 7.
- **Radix** is lighter and still maintained. Keep it wherever it is already used. Do not mix the two per component unless there is a reason.
- **Native elements** where they suffice: `<dialog>` for simple modals, the Popover API with `@starting-style` for simple non-anchored popovers, and native `<input type=range>` if a styled slider is not needed. Native `<details name=…>` gives you an exclusive accordion.

## 7. Bottom sheet for the map, recommendation: Base UI Drawer

**Options considered:**
- **vaul:** the README states "This repo is unmaintained." The last release was 1.1.2 on 2024-12-14 and the last push 2025-10-03. It depends on Radix Dialog. **Avoid.** Source: https://github.com/emilkowalski/vaul
- **react-modal-sheet:** 5.6.0, published 2026-03-27, MIT, with a peer dependency on `motion >=11`. It supports `snapPoints` and `detent`. It deliberately ships no accessibility or focus handling, has no documented non-modal mode, and its `prefersReducedMotion` prop defaults to false. Source: https://github.com/Temzasse/react-modal-sheet. **Avoid** for this use case.
- **Base UI Drawer** (stable since 1.3; measured at 38.6 kB gzip, sharing internals with Dialog):
  - `snapPoints`, where values 0–1 are viewport fractions and values above 1 are pixels, plus `snapToSequentialPoints`.
  - `swipeDirection`, `SwipeArea`, nested drawers, and `Drawer.VirtualKeyboardProvider`, added in 1.6.
  - `modal={false}` turns off focus trapping and `disablePointerDismissal` keeps the drawer open on outside clicks.
  - The docs do not explicitly say the content behind stays interactive with `modal={false}`, so treat that as **unverified** and prototype it against the MapLibre canvas.
  - There is no built-in reduced-motion handling; use `motion-reduce:` styles.
  - Source: https://unpkg.com/@base-ui/react@1.9.0/docs/react/components/drawer.md
- **CSS scroll-snap sheet:** a scroll container whose snap points are the peek, half and full positions. The browser does the physics, so it is smooth and accessible by keyboard and wheel. The reference implementation is pure-web-bottom-sheet 0.7.0 (MIT, a web component). Sources: https://dev.to/viliket/native-like-bottom-sheets-on-the-web-the-power-of-modern-css-ld6 and https://cdn.jsdelivr.net/npm/pure-web-bottom-sheet@0.7.0/README.md. The catch is non-modal use over a map: the transparent part of the scroller has to pass touches through with `pointer-events:none` on the container and `auto` on the sheet. That breaks scroll-dragging from the empty area, so I would treat this as an R&D option.

**Recommendation:**
1. Try the Base UI Drawer with `modal={false}` and three snap points first.
2. If the map under the peek state is not interactive, fall back to a custom pointer-events sheet. That means a fixed `transform: translateY` panel driven by Pointer Events with `setPointerCapture`, snapped with a CSS transition using a `linear()` spring. It costs about 2–3 kB of your own code and no dependency.
3. In either case, feed the sheet height into MapLibre `padding.bottom` so `fitBounds` and `flyTo` frame correctly.

## 8. Charts for ≤2,000 points with a log scale and annotations

**Current status:**
- uPlot 1.6.32 was published 2025-03-14, but the repo was pushed 2026-10-11, MIT.
- Observable Plot 0.6.17 was published 2025-02-14 and the repo pushed 2026-09-01, ISC.
- Recharts 3.10.1 was published 2026-07-25, MIT.
- visx 4.0.0 was published 2026-06-11, with React 19 as a peer.
- d3-scale 4.0.2 and d3-shape 3.2.0 are stable and effectively finished.

**Measured gzip size:**

| Option | Imports measured | Gzip |
|---|---|---|
| d3-scale + d3-shape | `scaleLog`, `scaleUtc`, `line`, `area` | 14.3 kB |
| uPlot | | 23.0 kB, plus about 1 kB of CSS (not measured) |
| visx, low-level | shape, scale, axis, event | 27.0 kB |
| visx, `@visx/xychart` | | 53.6 kB |
| Observable Plot | | 87.5 kB |
| Recharts 3 | LineChart, Brush, Reference* and ResponsiveContainer | 112 kB |

**Accessibility:** Recharts 3 turns on `accessibilityLayer` by default, giving keyboard tab and arrow navigation plus screen-reader support. Source: https://newreleases.io/project/npm/recharts/release/3.0.0. uPlot renders to canvas and has no built-in accessibility. Observable Plot and d3 output SVG, so you supply ARIA yourself.

**Recommendation:** use **d3-scale + d3-shape rendered as React SVG**, about 14 kB. At 2,000 points SVG is fine. It gives full control over the log axis, annotations (bloom thresholds, notice dates) and an accessible fallback such as a data table or `aria-describedby` summary. Implement the crosshair as a pointer handler plus a bisector, and keep hover state in a ref or CSS variable so React does not re-render on every pointer move. uPlot is the choice only if you need 10k+ points or brushing out of the box. Avoid Recharts on bundle grounds.

## 9. Spring-like CSS easing with `linear()`

- Jake Archibald and Adam Argyle's **Linear Easing Generator** at https://linear-easing-generator.netlify.app converts a JavaScript easing function or SVG path into a `linear()` stop list. It has Spring, Bounce and Elastic presets and a stop-count/precision trade-off. For springs it derives a duration from settle time; zero-friction springs are impossible. Sources: https://developer.chrome.com/docs/css-ui/css-linear-easing-function and https://joshwcomeau.com/animation/linear-timing-function
- I could not verify the "kvin" tool.
- **Practical approach:** generate two or three tokens once, for example `--ease-spring-snappy` and `--ease-spring-gentle`, each with about 30–40 stops and a paired `--dur-*`, and store them in the Tailwind 4 `@theme`. Use a `cubic-bezier()` fallback via `@supports not (transition-timing-function: linear(0,1))`, needed for only about 6% of browsers. Under `prefers-reduced-motion`, drop to `--dur: 0ms` or an opacity-only fade.
- Because CSS transitions retarget from the current value, these are interruptible, although velocity is not preserved. Use Motion springs only where preserving velocity during a drag matters, such as sheet release.

## Summary

| Capability | Recommendation | Dependency? | Approx. gzip | Rationale |
|---|---|---|---|---|
| Component enter/exit, popovers, dialogs | Native CSS: `@starting-style`, `allow-discrete`, `linear()` | No | 0 | About 93% support including iOS 17.5+; compositor-only and interruptible |
| Spring easing | Pre-generated `linear()` tokens | No | 0 | No runtime cost; interruptible; easy reduced-motion override |
| Gesture or presence physics | Motion `m` + `LazyMotion` + lazy `domAnimation`; or `motion/mini` | Yes, narrowly | 5.6 kB + 15–24 kB lazy, or about 3.4 kB | MIT, very active, `reducedMotion="user"`; avoid `motion.div` (41 kB) and `domMax` |
| Route and state transitions | `document.startViewTransition` behind a feature check now; React `<ViewTransition>` after React 19.3 + Next upgrade | No | 0 | Stable only in React 19.3; Next 15.5 flag is experimental; avoid nested groups (Chrome only) |
| Height to auto | `grid-template-rows` trick; `interpolate-size` as enhancement | No | 0 | No Safari or Firefox support for `interpolate-size` / `calc-size()` |
| Scroll storytelling | IntersectionObserver + CSS; `animation-timeline` under `@supports` | No | 0 | Safari 26+, Firefox 160 only |
| GSAP | Avoid | — | about 45 kB (core + ScrollTrigger + React hook) | Proprietary no-charge license; oversized for this need |
| Map camera, fades, hover | MapLibre `flyTo`/`easeTo`/`fitBounds` with `padding`, `stop()`, `originalEvent` checks, `raster-opacity-transition`, feature-state | No (already installed) | 0 | Reduced motion is built in; GPU transitions; no React re-renders |
| UI primitives | Base UI for new work (Radix acceptable where already used); native `<dialog>`/popover where they suffice | Yes | Base UI about 24 kB (dialog), 65 kB (7 widgets); Radix 14 / 40 kB | Base UI 1.x is active, from the Radix authors, and includes Drawer |
| Mobile bottom sheet | Base UI Drawer with `modal={false}` and snap points; fallback is a custom pointer-events sheet | Base UI (shared) | about 39 kB incl. dialog internals | vaul is unmaintained; react-modal-sheet has no accessibility; non-modal map interaction unverified, so prototype it |
| Time-series charts | d3-scale + d3-shape in React SVG | Yes, small | about 14 kB | Log scale, annotations, accessible; uPlot (23 kB) only for large data; avoid Recharts (112 kB) |

## Not verified

- What `useReducedMotion` returns during SSR.
- Whether a stable Next 16.3 without the view-transition flag has shipped.
- Whether Base UI Drawer with `modal={false}` lets pointer events reach the map.
- The default value of `raster-fade-duration`; I could not re-fetch the spec page.
- The "kvin" `linear()` tool.
- The shadcn default-switch date, which comes only from secondary sources.
- Firefox's "partial" cross-document view transition support: I did not confirm what it covers.