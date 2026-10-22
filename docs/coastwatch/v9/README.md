# Neritic v9: opening and motion

Branch `feat/neritic-v9-motion`, stacked on `feat/neritic-v8-market` (#79). Spec, audit and as-built notes: [motion.md](motion.md). Screens: [screens/](screens/).

## The brief

> The opening / hero / motion system still does not feel premium. It feels mid. Create a high-quality, beautiful, distinctive opening and motion system that feels elegant, intentional, coastal/scientific, premium, memorable, not gimmicky, not "vibe-coded". Less, but better.

## How the pass was run

Four roles:
1. **Art direction**, an independent agent: audited the v8 opening and app motion frame by frame and wrote the spec. It looked at OriginKit and particles.casberry.in for principles only. No licence was visible on either, so nothing was copied.
2. **Implementation** in two parallel worktrees: the opening, and the app-wide effects.
3. **Independent visual critique** of the built result. It judged the soundings premium but the mark, the lockup and the handoff mid, and gave five refinements. Four were taken; for the fifth, the mark's flight was fixed rather than replaced.
4. **QA / performance** against the spec's acceptance list. It found two defects, both fixed.

## What changed

| Area | Before (v8) | After (v9) |
|---|---|---|
| **Opening** | About 5.4 s and ten effects: contour draws, ping rings, a brightness "breathe", an index glow, radial gradients, drift, a tagline, a skip hint, a zoom-in "dive". The header was hidden throughout. | **"Sounding", about 3.5 s.** Soundings appear as points on the real Monterey Canyon isobaths and join into contours. The canyon head's own contours grow into the mark and settle into the lockup; then the word appears and a short hold follows. Once the map is ready, the chart recedes, the ground clears to the live map, and the lockup flies into the header's brand. The header stays usable throughout. |
| Skip | 700 ms "dive" | 200 ms fade, before the app's script loads too |
| Reduced motion | still mark, then a fade | still lockup for 600–800 ms, then a 200 ms fade |
| Loaders | Made-up Bézier "contours" | One real isobath (Monterey Canyon's 200 m line): its soundings join and part while loading. Shown only after 400 ms of real waiting. |
| Layer switch | Forecast, outlook and SST faded; observations snapped; the seafloor dropped to near-black while tiles loaded | Every layer fades in 220 ms on a switch; a day step cuts. The depth-tinted seafloor stays until the new field has drawn. |
| Camera | Flights up to 2.6 s; seven durations | Flights capped at 1.8 s; nearby picks 600 ms; cluster clicks the same |
| Controls | Springs with about 11 % overshoot on press and focus | Press .965 in 90 ms, release 220 ms ease-out, no overshoot (springs only under a finger) |
| Panels | Slide 16 px; a new selection blanked the panel | Slide 12 px; a new selection fades from dimmed |
| Field pack | Text swap on save | The full progress bar gathers into a check, then the words |
| Tokens | Two parallel sets, four aliases | One set |

**Start with:**
- the opening: [opening-filmstrip-d.jpg](screens/opening-filmstrip-d.jpg) and [opening-filmstrip-p.jpg](screens/opening-filmstrip-p.jpg);
- reduced motion: [opening-filmstrip-p-reduced.jpg](screens/opening-filmstrip-p-reduced.jpg);
- the app after it: [explore-chl-d.jpg](screens/explore-chl-d.jpg) and [coast-d.jpg](screens/coast-d.jpg).

## What did not change

- Every value, date, source, caveat and colour family. Data values never animate.
- My Coast and Research never show the opening. Their content is in the first HTML and is never delayed.
- Offline behaviour: field packs and the service worker are unchanged. A saved copy never shows the opening.
- No new dependency and no second WebGL context.

## Measured

Local production build on staging data, phone profile with 4× CPU throttling, median of 3 runs.

- **Opening and pages:**
  - Explore's first visit is usable at 3.6–3.7 s (v8: 5.3 s).
  - My Coast, the briefing and Research load in the same time as v8 (±7 %).
  - Layout shift is unchanged.
- **The handoff:**
  - The mark lands on the header's mark within 0.01 px.
  - No dropped frames during the handoff (headless measurement, approximate).
- **Skip and reduced motion:** a skip is gone in ≤ 224 ms; reduced motion is gone at about 0.85 s.
- **Cost:** shared JavaScript grows by 1.6 KB gzipped.
- **Tests:** full results are in the PR.

## Decisions for the owner

1. **Length.** About 3.5 s, down from the ~5 s you asked for in v8. The critique and the "don't hide ready content" rule both pointed shorter. One value, `--nx-o` in theme.css, stretches the whole play, e.g. 1.4 gives about 5 s.
2. **The handoff.** The spec landed the chart on the map's real Monterey Canyon. It registered correctly, but at state scale it lands as about 40 px, and on the way canyon contours lay across land. The as-built version clears the chart first. A version that lands on the canyon could return if the map ever opens closer in.

## Known limits

- Safari was not tested: the loader animates `stroke-dasharray` through a CSS variable, and the mark's mask uses `@property`. Without `@property` the mark's ends are simply not faded.
- The selection outline does not draw once on arrival (MapLibre cannot transition it cleanly).
- Re-selecting while a panel is open still changes height in one step when the new content is longer.
