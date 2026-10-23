# Neritic v9: motion and art direction

Status: spec only (audit and specification). No app code is changed by this document.
Scope: the first-visit opening on Explore, plus the six motion behaviours used across the app.
Brief from the owner: "less, but better". Five or six well-made effects instead of thirty average ones.

Evidence: `scratchpad/review/opening-filmstrip-{d,p}.png` (300 to 6200 ms) and `scratchpad/review/fine-d-4100…5100.png`,
put together in `scratchpad/v9-audit/exit-fine-d.jpg`. Paths below are relative to `coastwatch-web/`.

---

## 1. Audit

### 1.1 The opening (Splash.tsx, theme.css 476–697, MotionRoot.tsx SplashGate)

The problem in one line: it runs about ten effects in five seconds, and the one moment that matters,
becoming the map, is a cut.

| # | Problem | Where | Seen in frames |
|---|---|---|---|
| O1 | **Too long.** SplashGate holds until 4700 ms whatever the map does, then exits for 700 ms. The map is ready long before that and stays hidden. That is about 5.4 s on a first visit. | MotionRoot.tsx:26–28, 44; theme.css:506 | The lockup is already complete at 3900 ms, but the map does not appear until 4500 ms. |
| O2 | **The handoff is a cut, not a transition.** The exit "dive" (stage scale 1.9, field scale 1.35, ease-in) zooms *in*, but the destination is the whole state zoomed *out*. The motion points the wrong way. Between 4300 and 4500 ms the lockup is gone and the statewide map is simply there. Nothing connects the canyon to Monterey Bay on the map. | theme.css:516–519, 675–676 | `exit-fine-d.jpg` frames 2 to 3 |
| O3 | **The field cannot line up with the map.** MARK_FIELD is drawn in the mark's frame, turned 72° counter-clockwise from north (build_mark.py: east becomes up, then rot −18°). No transform is ever applied to give it its geographic meaning. | Splash.tsx:56; scripts/brand/build_mark.py:58, 156–161 | – |
| O4 | **Generic "SVG line-draw".** Every isobath reveals from one end with `pathLength=1` dashoffset. Short and long lines take the same time, so short lines crawl and long lines whip. This is the most recognisable vibe-coded web effect, and it says nothing about soundings or surveys. | theme.css:552–575, 617–620 | The 1200 ms frames |
| O5 | **The same contour is drawn twice.** The hero's five isobaths redraw at their own weight on top of the field's identical isobaths. At 1200 ms this reads as a bright loading spinner with a stray dot (the 100 m line's start cap), not as bathymetry. | Splash.tsx:82–92 | 1200 ms d/p |
| O6 | **Sci-fi garnish.** Two sonar ping rings with `box-shadow` glow, a full-screen `filter: brightness(1.35)` "breathe", an index-isobath stroke-width "glow", and a radial "lit shelf" gradient. These are exactly the fake glows and random gradients the brief bans. The full-screen filter animation is also the most expensive paint in the sequence. | theme.css:521–534 (depth, breathe), 559–570 (glow), 577–596 (ping) | 3200 ms (ring) |
| O7 | **App icon on wallpaper.** At 2.6× scale the tile is a glossy ~230 px rounded square sitting on the contours. At 2400 ms it reads as an iOS icon on a contour wallpaper, not as a window onto the canyon. | theme.css:621–628 | 2400 ms |
| O8 | **A constant drift and parallax** (5.6 s scale 0.96→1 and rotate −1.5°). It moves while everything else moves, so nothing is ever still and nothing reads as intentional. | theme.css:535–550 | – |
| O9 | **Unreadable tagline.** It is on screen from 3850 to 4700 ms: 850 ms for 9 words. Nobody reads it, and it adds a fifth text beat. | Splash.tsx:99; theme.css:643–657 | 3900 ms |
| O10 | **An apologetic skip hint** ("Click or press any key to skip") from 1200 ms. On a sub-3.5 s opening it is noise. | Splash.tsx:101–104; theme.css:658–673 | All frames |
| O11 | **Skipping is slow.** A tap plays the full 700 ms dive. A skip means "get out of my way" and should take about 200 ms. | Splash.tsx:28, 30 | – |
| O12 | **The header is hidden throughout**, although the comments say it stays usable (z 44 < 45). Explore's root has `cw-fade-in`, an opacity animation with fill `both`. That creates a stacking context, so TopBar's z-45 is trapped beneath the splash. | Explore.tsx:951; globals.css:306; TopBar.tsx:162 | 300–4300 ms: no header |
| O13 | **Two clocks.** SplashGate counts from navigation start (`performance.now`). The CSS animations count from first style. If the HTML is slow, the gate fires before the CSS play ends and the opening is cut mid-wordmark. | MotionRoot.tsx:46 | – |
| O14 | **The whole Explore page fades in for 260 ms underneath an opaque splash**, then fades again on every later visit: the map canvas itself fades in. | Explore.tsx:951 | – |

What is good and stays:
- server HTML/CSS before JS;
- the first visit on Explore only (`nx-splash-seen-v2`), never on a field-pack copy or for automated browsers;
- skip on any input before hydration;
- base styles are the final frame, so reduced motion is the same markup with animations off;
- the instrument ground `#061016`, which is also the map ground, so nothing flashes;
- the geometry itself (real CRM isobaths). That geometry is the best asset Neritic has and the current opening hides it under effects.

### 1.2 App motion

| # | Problem | Where |
|---|---|---|
| A1 | **Two parallel token sets.** `--cw-ease-settle` is a different linear() spring (globals.css:108), used by MobileSheet (MobileSheet.tsx:97 also mixes `--cw-d-medium` and `--nx-t-sheet`). The aliases `--nx-ease-in-out`, `--nx-t-quick` and `--nx-t-base` (theme.css:132–134) are unnecessary. | globals.css:101–108; theme.css:132–134 |
| A2 | **The CSS spring overshoots about 11%** (linear() peak 1.108, theme.css:147). The JS SPRING is bounce 0.08 (motion.ts:13), so the two disagree. The spring drives the focus ring's `outline-offset` and press release, which makes the focus ring wobble. Springs belong to gestures, not focus rings. | theme.css:147, 295, 300, 332 |
| A3 | **Layer switching is inconsistent.** Forecast, outlook and SST rasters cross-fade over 220 ms. The observation raster snaps (fade 0). The same user action gives two different feels. | MapCanvas.tsx:177–183, 679 |
| A4 | **Seven different camera durations:** 300, 380, 450, 500, 700 and a flight of up to 2600 ms. A far harbour pick flies for up to 2.6 s while the panel is already open, so the panel and the map disagree for 2.6 s. | camera.ts:97–102; Explore.tsx:185, 407, 416; MapCanvas.tsx:482 |
| A5 | **Fake contours in loaders.** StartCard and the briefing map placeholder sweep procedural Béziers (`M-20 ${70+i*38} C…`) and present them as bathymetry. The brand's promise is real geometry. | Explore.tsx:1206–1210; Briefing.tsx:640 |
| A6 | **Panel slide distance** is 16 px in both CSS and JS. That is slightly loose for a 240 ms enter; 12 px reads tighter. The exit (8 px, 160 ms, ease-in) is right. | theme.css:377; Explore.tsx:1234–1236 |
| A7 | **A pack save ends in a text swap.** The bar fills with real bytes (good), then the line "Saved and verified" simply appears. The most consequential offline action has no moment of confirmation. | OfflinePanel.tsx:237, 259 |
| A8 | **The My Coast hero coast** reuses the 1400 ms splash draw curve on every visit (decorative, on a reading page). | theme.css:880–884 |

Keep as-is: `nx-late` (a 400 ms gate before any loading word, theme.css:415), the flat skeleton with no shimmer, the Timebar pill `layoutId` + SPRING (Timebar.tsx:256), the `nx-pop` popovers (180/120, scale .96), `nx-fold` disclosures, the toast, Glide (nav indicator), the `--nx-shift` reduced-motion switch, and the camera tiering by renderer and frame rate (camera.ts:34–72).

### 1.3 Remove list

1. Ping rings (both): `.nx-splash-ping`, its markup and keyframes.
2. `nx-splash-breathe`: the full-screen brightness filter.
3. `nx-splash-glow`: the index-isobath stroke pulse.
4. The radial "lit shelf" gradients in `.nx-splash-depth`. The ground becomes flat `#061016`, and depth is shown by line opacity by depth (data, not decoration).
5. `nx-splash-drift`: the continuous scale and rotate parallax.
6. The tagline in the opening (it stays on the My Coast front door and in metadata).
7. The visible skip hint. Keep the sr-only "Loading Neritic" status and add "press any key to skip" to it.
8. The redraw of the hero lines (`.nx-splash-line` dash draw), replaced by weight-up (§3).
9. The exit "dive" (stage scale 1.9, field scale 1.35).
10. The 4700 ms MIN_HOLD and the 700 ms skip exit.
11. `cw-fade-in` on the Explore root.
12. The procedural Bézier "contours" in StartCard and the briefing placeholder.
13. `--cw-ease-settle` and the alias tokens (A1).

---

## 2. Principles and tokens

1. **Real geometry or nothing.** Every line that moves is a real isobath or a real UI edge. Motion never invents data, and data values never tween (no tickers, no day-to-day interpolation).
2. **One thing moves at a time.** Beats hand over; they do not stack. At most one transform-driven element is in motion on screen, apart from incidental colour fades.
3. **Motion explains where things come from.** Panels enter from the edge they are anchored to, the opening lands where Monterey Canyon is, and the mark lands where the header mark is. Motion with no origin is decoration and is cut.
4. **Fast in, faster out, instant on skip.** UI enters in ≤240 ms with ease-out and leaves in ≤160 ms with ease-in. A user's interrupt always wins within one frame.
5. **No overshoot except under a finger.** Springs only settle gestures (sheet drag release, time pill). Everything else uses curves that do not overshoot.
6. **Stillness is part of the composition.** The opening has one deliberate 250 ms hold, nothing loops except a real "loading" indicator, and loading words wait 400 ms (`nx-late`).
7. **Reduced motion is a designed version, not a broken one.** Positions are final, opacity cross-fades of 150–250 ms are kept, and nothing translates, scales, rotates or draws.

### Token table

The tokens in theme.css:110–136 are reused and trimmed. ★ marks a changed value. + marks a new token. No parallel `--cw-*` motion tokens remain: in globals.css those become aliases or are deleted.

| Token | Value | Use |
|---|---|---|
| `--nx-t-press` | ★ 90 ms | press-in (replaces the hard-coded 90 ms at theme.css:306) |
| `--nx-t-hover` | 150 ms | colour/tint, reduced-motion fades |
| `--nx-t-ring` | 200 ms | focus ring (★ eased with `--nx-ease-out`, no spring) |
| `--nx-t-release` | ★ 220 ms | press release, lift (★ `--nx-ease-out`, no spring) |
| `--nx-t-pop` / `--nx-t-pop-exit` | 180 / 120 ms | popovers, rise, fade |
| `--nx-t-panel` / `--nx-t-panel-exit` | 240 / 160 ms | side panel, fold, dialogs |
| `--nx-t-layer` | + 220 ms | raster cross-fade (all rasters), legend swap 120 ms |
| `--nx-t-page` | ★ 200 ms | page content fade (opacity only, never on the map canvas) |
| `--nx-t-sheet` | 460 ms | phone sheet tap/key snap (drag release: SPRING) |
| `--nx-t-skip` | + 200 ms | opening skip, reduced-motion fade |
| `--nx-ease-out` | cubic-bezier(.23,1,.32,1) | every enter |
| `--nx-ease-in` | cubic-bezier(.4,0,1,1) | every exit |
| `--nx-ease-move` | cubic-bezier(.77,0,.175,1) | on-screen moves A→B (fold, settle, handoff) |
| `--nx-ease-drawer` | cubic-bezier(.32,.72,0,1) | sheet |
| `--nx-ease-spring` | ★ linear() with peak ≤ 1.03 (match SPRING bounce .08) | drag release and the time pill only |
| Distances | rise 6 px · ★ panel 12 px · toast 8 px · lift 2 px · press scale .965 · pop scale .96 | – |
| Camera (camera.ts constants) | near (<40 km) 600 ms ease · flight 700 + 3 ms/km, ★ cap 1800 ms · short tier 300 · none = jump | selection, region |

Opening-only tokens live inside the launch screen block (`--nx-o-*`, §3) and are never used elsewhere.

---

## 3. The opening: "Sounding"

**Concept.** Depth soundings appear as points along the real Monterey Canyon isobaths and join into contours, and the canyon head's contours thicken into the mark. The chart then turns to true north and pulls back into its real place on the California map, while the mark settles into the header. The opening does not cover the map; it becomes the map.

**Total.** 2.6 s to the formed lockup, plus 0.8 s handoff = **3.4 s** on a first visit (was about 5.4 s). Hard CSS cap: 4.2 s.
Why 3.4 s: the owner wants premium, not long. The shortest sequence that still lets each idea land is about 4 beats × 500–700 ms (soundings, contours, mark, lockup). The handoff needs about 800 ms to be legible as a pull-back. Anything longer than that is a hold, and holds were what made v8 feel mid. A cold first visit's map load (tiles plus style) typically takes 1.5–3 s, so the opening mostly overlaps real waiting.

### Geometry and transforms (all in tile units, from mark.geometry.ts)

- **Field:** MARK_FIELD (38 paths, 17 depths 0–2000 m, half = 150, so 300 units = 60 km; 1 unit = 0.2 km). The viewBox is centred on the tile centre (24, 24).
- **The tile centre's geographic position:** 36.790 N, 121.879 W. Derivation: build_mark.py WINDOW cx 0.6, cy 1.8 km in the turned frame, inverse-rotated, from 36.79 N 121.90 W. Ship it as `MARK_FIELD_GEO = {lat, lon, kmPerUnit: 0.2, northDeg: 72}` from build_mark.py. This is a new export, generated; do not hand-edit.
- **North:** the field's north points 162° CCW from screen-right. `rotate(72deg)` (CSS, clockwise) makes it north-up. QA must verify this with an overlay test (§5) before shipping.
- **Hero:** MARK_HERO (10 half-isobaths, 100–300 m). At hero scale `z` it coincides exactly with the field's 100–300 m lines, because the drift box is 6.25 × the hero box and 6.25 × 48 = 300.
- **Tile:** MARK_TILE. **Word:** MARK_WORD. The header targets are `Lockup size 28` (desktop, small tier) and `Mark size 30` (phone) in TopBar.tsx:167–172.

### Storyboard (ms from the splash's first style: one clock, see A/O13)

| t (ms) | Beat | What moves | Geometry | Property / timing |
|---|---|---|---|---|
| 0–100 | Ground | Nothing. A flat `#061016` ground, the same as the map. | – | – |
| 100–760 | **1. Soundings** | Dots appear along the isobaths, depth by depth, from the 200 m index outward (the existing `fieldDepths` order). | MARK_FIELD, no `pathLength` (user units) | `stroke-dasharray: 0 2.4` with round caps, so the dots sit 2.4 units (≈11 px desktop) apart. Opacity 0→target over 180 ms, stagger 35 ms × 17 depths, `--nx-ease-out`. Target opacity by depth: 0 m .45 · 25–150 m .32 · **200 m .70 in `#7fe3f5`** · 250–600 m .26 · ≥800 m falling linearly to .12 at 2000 m. |
| 420–1300 | **2. Contours** | The dots stretch into dashes and then into continuous lines, depth by depth, so the survey becomes a chart. | same | `stroke-dasharray: 0 2.4 → 2.4 0`, 600 ms `--nx-ease-move`, same stagger. The 200 m line completes first. |
| 1150–1800 | **3. Mark** | The canyon head's contours *thicken* into the mark; nothing redraws. The tile surfaces behind them and the field steps back. | MARK_HERO over the field's 100–300 m; MARK_TILE | The hero paths start at the field's weight and opacity (0.3, index 0.5) and are visible at 1150 (opacity 0→1 over 120 ms, so the identical lines show no seam). `stroke-width` grows to `strokeOf("full")` (1.7 / index 3.4) over 520 ms `--nx-ease-out`. Tile: opacity 0→1 over 400 ms from 1300, **no scale**. Field opacity ×0.55 over 500 ms from 1300. |
| 1800–2350 | **4. Lockup** | The mark settles to lockup size. The wordmark wipes in after it. | hero box; MARK_WORD | Hero `transform` from scale 1 to `scale(1/z) translateX(--nx-dx)`, 550 ms `--nx-ease-move` (the existing settle, retimed). Wordmark from 1950: `clip-path inset(0 100% 0 0) → inset(0 -2% 0 0)` + `translateX(−4px → 0)`, 400 ms `--nx-ease-out`. |
| 2350–2600 | Hold | Nothing moves: the only still frame with the full lockup. | – | 250 ms. |
| 2600–3400 | **5. Handoff: "the chart becomes the map"** | (a) The ground fades out and the live map shows through. (b) The field turns to north and pulls back to its true place and size on the statewide map. (c) The mark flies into the header mark and the wordmark into the header wordmark. (d) The header brand takes over. | MARK_FIELD + MARK_FIELD_GEO; hero → TopBar brand rect | (a) `.nx-splash` background opacity 1→0, 0–400 ms of the beat, `--nx-ease-move`. (b) The field wrapper gets `translate(--nx-land-x, --nx-land-y) rotate(72deg) scale(pow(--nx-land-s, --k))`, where `@property --k` (<number>) animates 0→1 over 800 ms `--nx-ease-move`. Scale is geometric, so the pull-back feels like a constant-speed camera. Its stroke opacity goes to 0 over the last 300 ms as the map's own bathymetry takes over. (c) Hero box to the header mark's rect over 700 ms `--nx-ease-move`. Over the last 40%, the 150 m and 250 m hero lines fade out so the mark arrives as the header's 3-line *small* tier. Desktop: the wordmark moves with it. Phone (header has the mark only): the wordmark fades out in 150 ms. (d) At 3300 the header brand opacity goes 0→1 over 100 ms (it was hidden by `html.nx-splash-on .nx-topbar-brand {opacity:0}`). The splash is removed at 3400. |

Landing values (SplashGate, under 1 KB of JS): `p = map.project([lon, lat])` and `q = map.project` of a point 10 km east. Then `--nx-land-s = (|q−p|/10 km) / (fieldPx/60 km)` and `--nx-land-x/y = p − viewport centre`.
Expected `s` at the statewide view: desktop 1440×900 ≈ 1/35 (the 60 km field lands as about 41 px); phone ≈ 1/22.
The map camera is **not** touched, and its initial view, deep links (`?region`, `?at`) and tiles are unchanged. If a deep link opens elsewhere and Monterey is off-screen, use (b′): the field scales to 0.6 and fades over 500 ms in place. Everything else stays the same.

**Rejected alternative:** start the map at Monterey (zoom ~11.5, bearing −72°) and `flyTo` statewide. It is literally "becoming the map", but it adds 1.5–2 s of tile loading, fights deep links and the camera tiers, and moves the user's map without their asking.

### Behaviour rules

- **First visit, Explore only:** unchanged decision script (`nx-splash-seen-v2`, saved-copy, webdriver, `nx-splash-off-v1`). Repeat visits, My Coast and Research: never shown and never in their HTML.
- **Gate:** SplashGate starts the handoff at `max(2600 ms, mapReady)` on the splash clock. Read the clock as `document.timeline.currentTime − splashStart`, where `splashStart` is recorded by the inline script at `nx-splash-on`; do not use `performance.now()` from navigation. If the map is not ready by 3900, there is no landing: the CSS cap fades it out (300 ms, gone by 4200) and StartCard is already there underneath.
- **Skip:** pointerdown or any key except Tab/Shift, at any time, gives `nx-splash-out`: opacity → 0 in `--nx-t-skip` (200 ms) `--nx-ease-in`, no landing, and the header brand shows at once. It works before hydration (inline script, `EXIT_MS = 200`).
- **Before JS:** beats 0–4 and the hold are pure CSS. The handoff needs JS (`map.project`). Without it, the CSS cap animation (`nx-splash-gone` at 3900 ms, 300 ms) ends the opening as a plain fade.
- **Header:** remove `cw-fade-in` from the Explore root (O12/O14), so TopBar (z 45) truly sits above the splash (z 44). The workspaces, search and notices stay usable throughout. Only `.nx-topbar-brand` is hidden until 3300.
- **Reduced motion:** the final lockup (beat-4 end frame: field static at its dimmed opacities, tile, mark, wordmark) is a still frame from first paint. It holds until `max(600 ms, mapReady)`, capped at 800 ms, then cross-fades out in 200 ms. The total is ≤ 1.0 s with no draw, rotate, scale or fly, and the header brand shows with the fade. Implement with the existing base-styles-are-final pattern (theme.css:692–696) plus `--nx-t-skip`.
- **Offline / field pack:** unchanged (no network asset, no service-worker change). `__NERITIC_SAVED__` keeps it off.
- **Cost:** no new assets. The field paths already ship (MARK_FIELD ≈ 10.9 KB raw). Removing the ping, depth and skip markup makes the HTML smaller. Beats 1–3 animate stroke properties (paint) on ≤ 48 small paths. Beats 4–5 are transform and opacity only (composited). `will-change: transform` applies only during beat 5.

---

## 4. Signature effects across the app (six)

| # | Effect | Surfaces | Trigger | What moves | Duration / easing | Reduced motion | Why |
|---|---|---|---|---|---|---|---|
| 1 | **Sounding** (the opening, §3) | Explore, first visit | first load | soundings → contours → mark → map | 3.4 s total; beats above | still lockup ≤ 1.0 s | Brand and orientation: this is a real chart of a real coast, and here is where you are. |
| 2 | **Contour loader** | StartCard (Explore.tsx:1197), briefing map placeholder (Briefing.tsx:640), Inspect loading (Inspect.tsx:234), My Coast hero coast (theme.css:880) | real waiting > 400 ms (`nx-late`) | **One real isobath** (MARK_FIELD's 200 m line, or the place's own coastline) cycles the Sounding beat-1→2 join: dots → line, 1.6 s period, `--nx-ease-move`, opacity .35. The My Coast coast plays one join (900 ms) and then stays still. | 1.6 s loop only while loading; stops on ready (fade 150 ms) | static solid line + words | It replaces fake Béziers (A5) and ties loading to the brand's one gesture. "Stale" and "empty" are **not** loading: no motion, a static dashed line (`2 2`) with the stale/empty words. Motion means "something is happening now". |
| 3 | **Instrument surfaces** (panels, sheets, popovers, dialogs, folds, page entry) | Inspect aside, LayerPanel popover, Notices, StatusCenter, search list, PackDialog, MobileSheet, Fold/details, route change | open / close / navigate | Anchored edge in: side panel x 12 px + opacity; popovers scale .96 from their trigger origin (`nx-origin-*`); dialog rise 6 px over a 180 ms veil; sheet translateY; fold by grid rows with the body fading 60 ms late. Reading pages fade content in over 200 ms (opacity only). Explore's map canvas never fades. The nav pill Glide stays. | enter 240 / 180 ms `--nx-ease-out`; exit 160 / 120 ms `--nx-ease-in`; sheet 460 ms drawer, drag release SPRING (one curve, A1). A new selection keeps the panel and cross-fades the body (120 ms). | opacity only, same durations capped at 150 ms | Where a surface comes from tells you what it belongs to; faster exits keep the map responsive. Keep as is apart from 16→12 px, the sheet tokens and the removed page fade. |
| 4 | **Layer cross-fade** | Explore layer deck, LayerPanel, time steps | layer or day change | All rasters (observation included) cross-fade with `raster-fade-duration: 220` (A3). The legend ramp and its words swap with 120 ms opacity. The Timebar pill glides (layoutId + SPRING). Colour families never blend into each other: the old layer fades out under the new one, and violet stays model. | 220 ms; legend 120 ms | raster fade 0 (an instant swap), legend instant | Shows *what* changed while keeping place. No value ever interpolates between days (the raster fades, the numbers do not tween). |
| 5 | **Journey** (selection: harbour, search, station, region) | Explore map + Inspect; search | pick | The camera eases or flies (camera.ts tiers, cap 1800 ms, near 600 ms). The panel opens at once with real content. On `moveend`, the selection outline draws once around the place (`line-dasharray` join, 300 ms, as in the Sounding join) and stays solid. | camera tiers; outline 300 ms `--nx-ease-out` | jump + outline shown solid | Causality: you picked *this*, and the map shows where it is. A shorter flight cap keeps panel and map in agreement (A4). Cluster click (450 ms) becomes the near tier (600 ms). |
| 6 | **Touch and confirm** (press, toggles, field-pack save, toast) | every control; OfflinePanel; Toaster | press / toggle / save completes | Press: scale .965 in 90 ms, release 220 ms ease-out with **no overshoot**. Switches: thumb 180 ms ease-out. Focus ring: offset 200 ms ease-out (no spring, A2). **Pack save:** the real-bytes bar reaches 100% and then morphs into a check. The bar's fill keeps its colour, the bar collapses to a 20 px circle (240 ms `--nx-ease-move`), the check strokes in (200 ms), and then "Saved and verified" fades in (180 ms). The toast "Pack saved" stays. | as listed | colour change only; the check appears without stroke draw | Hands get instant acknowledgement, and the one irreversible offline action gets a deliberate, earned moment (A7). Progress stays honest (bytes, not time). |

Leave as is: chart line draw once on first sight (500 ms, not a value tween), the `nx-late` gate, flat skeletons, hover lift 2 px on desktop cards, Glide, the toast, and the `--nx-shift` mechanism.

Never: number tickers, animated readings, looping ambient motion on reading pages, parallax, glass or blur over data, glow, or particles that are not data.

### Reference research (principles only)

- **particles.casberry.in.** An "AI Particle Simulator": a three.js/WebGL swarm of about 20 k particles that morphs between sphere, helix and text, with gesture control. **No licence is stated** on the page (only a Casberry credit and a GitHub link), so nothing is copied.
  - Transferable principle: a form is more memorable when it *resolves out of points*, because the eye completes the shape just before it finishes.
  - Rejected for Neritic: a WebGL runtime (that would be a second context beside the map), free-floating or random particles, physics and glow.
  - Adopted honestly: points that are **soundings on real isobaths** (beat 1). They never move, they only join, which is how a chart is actually made.
- **OriginKit** (originkit.com returned 503; originkit.dev renders client-side and only its tagline was retrievable: "Free animated component library"). No licence terms were visible, so nothing is copied and the reference was not evaluated beyond that.
- **From known practice:**
  - Windy: particles are legitimate only when they encode data (wind).
  - Apple Weather and Maps: background conditions are subtle; content comes first; camera moves are short and ease-out.
  - Felt and Mapbox intros: geometry arriving in place beats a splash that hides the map.
  - Linear and Vercel: 150–250 ms, one curve family, restraint.
  - Rauno Freiberg and Emil Kowalski: origin-aware popovers, no ease-in on enter, exits faster than entrances, everything interruptible, springs for gestures only, ask "would I notice if this were gone?" for each effect.

---

## 5. Acceptance checklist (QA)

**Opening timing**, measured from the splash's first style, Playwright with `nx-splash-force-v1`, desktop 1440×900 and phone 390×844:
- [ ] At 300 ms: dots visible on the field and no continuous lines except a partial 200 m line. At 1300 ms: all contours continuous.
- [ ] At 1700 ms: hero at full weight, tile visible, no doubled or offset lines (hero over field ≤ 0.5 px misregistration at beat 3 start).
- [ ] At 2400 ms: lockup complete, no tagline, no skip hint, no rings, no glow, no radial gradient (sample the ground: `#061016` ±2 at the corners and centre).
- [ ] Map ready before 2600: the handoff starts at 2600 ± 50 and the splash node is hidden by 3450.
- [ ] Map ready later (throttle): the handoff starts at mapReady + ≤ 50 ms. If the map is not ready by 3900, a plain fade runs and the splash is gone by 4250.
- [ ] **Landing registration:** freeze at the handoff end (pause animations at 100%). The field's 200 m line overlays the map's Monterey Canyon within 8 px desktop / 6 px phone, and north is up (verifies `rotate(72deg)` and MARK_FIELD_GEO).
- [ ] The mark lands on the header mark rect within 2 px; no double mark visible at any frame between 3000 and 3400.
- [ ] With JS disabled (or hydration blocked): beats 0–4 play and the splash is gone by 4250.
- [ ] Skip: a pointerdown or key at 200, 1500 and 3000 ms gives splash opacity 0 within 220 ms. Tab/Shift do not skip. Works before hydration.
- [ ] Header workspaces, search and notices are clickable during the opening (the click reaches them and also skips).

**Scope**
- [ ] The second visit to `/` and every visit to `/coast…` and `/research…` have no `.nx-splash-on` and no splash markup in My Coast/Research HTML. Their LCP is unchanged from v8 ±5%.
- [ ] A field-pack saved copy offline shows no opening. Service worker and field-pack e2e suites pass unchanged.

**Reduced motion** (`prefers-reduced-motion: reduce`)
- [ ] The opening shows a still lockup from the first frame, with no animation running (`document.getAnimations()` empty except the fade). The total is ≤ 1000 ms + 200 ms fade.
- [ ] Panels, popovers and sheets use opacity only (no transform delta in computed style). The camera jumps. Rasters swap instantly. The loader shows a static line. Pack save shows a check without stroke draw.

**App effects**
- [ ] All rasters, including observation, report `raster-fade-duration` 220. No data number changes over more than one frame on a layer or day switch (no tween).
- [ ] Camera flight ≤ 1800 ms for any pick. Near picks 600 ms. The panel content is present at t = 0 of the flight.
- [ ] Inspect enters x 12 → 0 in 240 ms and exits in 160 ms. Popovers: 180 / 120 ms. Measure via `getAnimations()` timing.
- [ ] No overshoot: sampled press release, focus ring and panel transforms never exceed their end value by more than 1% (spring allowed only on sheet drag release and the time pill, ≤ 3%).
- [ ] No `--cw-ease-settle`, `--nx-ease-in-out`, `--nx-t-quick` or `--nx-t-base` references remain.
- [ ] The loaders draw paths from MARK_FIELD (or the place's coastline). No procedural `M-20 …` curves remain in the source.

**Performance budgets**
- [ ] No new dependency. Added JS ≤ 1.5 KB min+gz (landing maths in SplashGate + `--k` property). Explore HTML is no larger than v8's (the removals pay for MARK_FIELD_GEO).
- [ ] Opening on a mid phone profile (Pixel 6a-class, or Chrome 4× CPU throttle): ≤ 5% dropped frames in beats 4–5, no long task > 50 ms caused by the splash, and beat 5 is composited (no paint in the Performance panel during 2600–3400 except map tiles).
- [ ] CLS = 0 on Explore with and without the opening. The map initialises once (one WebGL context; `document.querySelectorAll("canvas")` webgl count = 1).
- [ ] Map time-to-ready with the opening ≤ without it + 50 ms (the opening must not slow the map).

---

## 6. As built (after an independent critique and QA)

The build follows §§1–5 except where an independent visual critique and a QA pass found the result weaker than the spec intended. The changes:

| Spec | As built | Why |
|---|---|---|
| Handoff (b): the chart turns to north and pulls back onto the map's Monterey Canyon (MARK_FIELD_GEO) | The chart recedes in place (scale .96) and is gone in 220 ms, before the ground clears. MARK_FIELD_GEO was not kept. | At statewide scale the 60 km field lands as about 40 px, so the landing was invisible. On the way there, canyon-scale contours lay across the Bay Area and Nevada for about 300 ms: depth lines over land is a trust problem. Registration was verified correct before the change (the overlay matched the coast and the map's own isobath labels). |
| Handoff (c): the mark flies into the header mark | Kept, on its own layer above the header (z 46, transparent, no clicks), ending in a 120 ms cross-fade into the header's brand | Under the header (z 44 < 45), the flight slid beneath the top bar and the brand slot stayed empty for about 200 ms. |
| Beat 3: weight 520 ms ease-out from 1150; tile from 1300 | Weight 450 ms `--nx-ease-move`. The minor lines keep the chart's colour, and their cut ends fade into the chart (a mask), until the tile surfaces at 2000 during the settle. | Ease-out made the mark snap to full weight in about 100 ms and freeze as a large pale-and-cyan "rainbow" with flat-cut ends: the "app icon on wallpaper" moment again. |
| Beat 4: settle 1800, word 1950, hold to 2600 | Settle 1650–2200, word 2200–2550, hold to 2800 | One thing moves at a time: the word follows the settle instead of wiping over a still-large tile. |
| Total ≈ 3.4 s | ≈ 3.5 s (2.8 s to the handoff, 0.7 s handoff); `--nx-o` stretches it | – |
| Layer cross-fade on every layer or day change | Fade (220 ms) on a layer switch only; a day step of the same layer cuts | The old day's source is removed at once, so a day fade dips through the bare sea on every step, and on every frame of playback. |
| – | The depth-tinted seafloor stays until the first data field has drawn, then cross-fades to it | Switching from the seafloor to satellite dropped to near-black relief for the second the new tiles took. |
| – | A new selection in an open panel fades from 45 % opacity, not 0 | The panel read as an empty box between two places. |
| Effect 5: the selection outline draws once on `moveend` | Not built | The harbour ring is a bitmap icon and cell outlines use `line-dasharray`, which MapLibre cannot transition without redrawing the dash pattern every frame. |

QA defects fixed:
- When the CSS cap ended the screen (a slow map or a late script), the header's brand stayed hidden: the gate now ends the screen after the cap's fade. The brand's hiding is also time-bound in CSS (4.2 s), so it returns even if no script runs.
- A skip during the handoff now fades instead of cutting.

Measured (QA, local production build, staging data, 4× CPU phone profile, median of 3):
- Explore first-visit content: 3.6–3.7 s (v8 5.3 s).
- My Coast, briefing and Research content time unchanged (±7 %).
- CLS unchanged.
- Handoff at 2.63–2.67 s when the map is ready early (2.8 s after the critique loop).
- Mark lands on the header mark within 0.01 px.
- No dropped frames in the handoff (headless, approximate).
- Skip gone in ≤ 224 ms.
- Reduced motion gone at about 0.85 s.
- One WebGL canvas.
- Shared JS +1.6 KB gzip.
