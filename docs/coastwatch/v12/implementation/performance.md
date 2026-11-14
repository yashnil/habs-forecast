# v12 performance against v11

**Method:** the same as `../research/E-science-and-perf.md` §4: a production build on fixture data, installed
Chrome headless, cold loads (cache disabled, service worker blocked), 9 s settle unthrottled and 22 s throttled;
LCP, CLS and long tasks from `PerformanceObserver` (TBT = long-task time over 50 ms after FCP); JS =
CDP `encodedDataLength` over the wire, everything the route loads in the settle window (MapLibre's lazy chunk
included); map-ready = first frame of `window.__cwMap.loaded()`. Throttled = CPU 4x + DevTools Slow 4G
(562.5 ms, 1.47 Mbps). Medians of 3 runs desktop, 2 throttled. Script: `scripts/perf-routes.mjs`; it also records
the largest **text** paint, which is what the E budgets name ("text LCP"), separately from the largest paint
(on v12 that is the baked chart image, an `IMG`). Same machine for both builds, 2026-11-11.

Routes: `/` is a first visit (the opening plays). The others are a returning reader at Moss Landing following
Dungeness crab, at v12's addresses: Today `/today?place=port-592` (v11: the same), Explore
`/explore?place=port-592` (v11: `/?view=map`), My Coast `/coast?place=port-592` (v11: the briefing), Research
`/research?place=port-592` (v11: `/research`).

## Per route, v11 (E report) and v12 (this pass, after the fixes below)

Times in ms. LCP columns give v12 as **largest paint / largest text**; v11's figures are E's (text set its LCP).

| Route | Build | Desktop LCP / TBT / map | 4x + Slow 4G desktop: LCP / TBT / map | Phone 390, 4x + Slow 4G: LCP / TBT / map | JS KB (files) | Fonts KB | CLS (worst) | Longest task |
|---|---|---|---|---|---|---|---|---|
| `/` | v11 (picker) | 3904 / 0 / 928 | 2766 / 144 / 14879 | 4404 / 102 / 14281 | 571 (18) | 333 | ≤ 0.04 | ≤ 220 |
| | v12 | 124 / 0 / 325 | 4560 text 4560 / 112 / 14146 | 7844 text 4566 / 129 / 14148 | 571 (17) | 245 | 0.000 | 120 |
| `/today` | v11 | 152 / 0 / 1242 | 3822 / 350 / 19435 | 3836 / 140 / 16288 | 571 (18) | 333 | ≤ 0.04 | ≤ 220 |
| | v12 | 156 / 0 / 335 | 8044 text 4484 / 244 / 14075 | 8000 text 4462 / 216 / 14042 | 570 (17) | 245 | 0.003 | 174 |
| `/explore` | v11 | 196 / 0 / 936 | 5758 / 212 / 17169 | 9242 / 265 / 14511 | 669 (19) | 333 | ≤ 0.04 | ≤ 220 |
| | v12 | 224 / 0 / 363 | 11916 text 2114 / 176 / 14450 | 11916 text 2104 / 214 / 14481 | 614 (19) | 245 | 0.009 | 164 |
| `/coast` | v11 (briefing) | 252 / 0 / 1139 | 3792 / 371 / 18756 | 3782 / 388 / 17394 | 683 (23) | 333 | ≤ 0.04 | ≤ 220 |
| | v12 | 176 / 0 / 386 | 8144 text 5210 / 162 / 14208 | 8212 text 5204 / 162 / 14226 | 605 (19) | 245 | 0.031 | 128 |
| `/research` | v11 | 172 / 0 / none | 2710 / 163 / none | 2694 / 124 / none | 619 (18); phone 293 (14) | 333 | 0.36 phone | ≤ 220 |
| | v12 | 300 / 0 / 423 | 9090 text 2454 / 240 / 14596 | 8372 text 2440 / 318 / 14680 | 597 (18); phone 597 | 245 | 0.022 | 254 |

Raw rows: `perf-v12b.json` (and the pre-fix `perf-v12a.json`) in the session's scratchpad; rerun with the script.

## Against the budgets (E §4)

| Budget | Ceiling | v12 | |
|---|---|---|---|
| Route JS | Today 571, Explore 669, briefing 683, Research 619 KB | 570, 614, 605, 597 | within; no library added |
| Research JS on a phone | 293 KB | 597 KB | **over**: see below |
| No second GL stack | none | one MapLibre, lazy | within |
| Fonts | 333 KB, target 200 | **245 KB** (was 578 before this pass) | within |
| Text LCP, phone 4x + Slow 4G | Today and briefing 3.9 s, Research 2.8 s, Explore 4.5 s | Today 4.46, My Coast 5.20, Research 2.44, Explore 2.10 | **Today and My Coast over** |
| LCP, desktop unthrottled | 300 ms | 124–300 | within (Research at the ceiling) |
| Map-ready, throttled | Today 19 s, Explore 17 s, picker 15 s | 14.0–14.7 s | within, 2–5 s faster than v11 |
| TBT, throttled | 400 ms; longest task 250 ms | 112–318; longest 254 (Research, phone) | TBT within; longest task 4 ms over on Research |
| CLS | 0.1 everywhere | ≤ 0.031 (Research phone fixed: v11 0.36) | within |

## Regressions found, and what was fixed (re-measured above)

1. **Fonts 578 KB on every route** (v11 333). The root layout preloaded v11's three faces (IBM Plex Sans, Plex
   Mono, Newsreader: 339 KB) on every page although no v12 page uses them. They are no longer preloaded (a legacy
   page fetches them when its CSS uses them), and Literata italic, used only for water names on the chart, is
   declared separately without a preload. v12 pages now load Literata upright and Public Sans first (134 KB) and
   the italic after hydration: 245 KB in all.
2. **Literata italic fetched at the highest priority** beside the page's code: the chart's water names were in the
   server HTML. They render after hydration now (they cannot be placed before the chart has a camera anyway).
   Text LCP on Today, phone throttled: 5164 → 4462 ms with (1).
3. **CLS 0.27 on Today at 390**: the baked chart images were positioned with `left/top`, so the camera settling
   counted as a layout shift. They (and Explore's baked frame) are placed by `transform` now: 0.000.
4. **CLS 0.097 on My Coast**: the gates' one-line loading state grew into three rows. The loading state holds
   their height: 0.031.
5. Explore's static tier stopped fetching the hourly currents it cannot draw (item 1 of this pass).

## Not fixed, for the owner

- **Text LCP on Today (4.46 s) and My Coast (5.20 s), phone throttled, over 3.9 s.** The cause is structural: the
  reader's fishery lives only on the device (no cookie, by design), so Today's statement and My Coast's gates
  render at hydration (Today) or after a fetch made at hydration (My Coast); v11 rendered its port page's text on
  the server. Hydration lands at about 4.2 s on Slow 4G: HTML, CSS (50 KB), the two preloaded faces (134 KB) and
  about 290 KB of first-load JS (React, Next, Motion and Radix are most of it). Options: carry the fishery in the
  address (`?f=`) so the server can render the statement; or lazy-load the Radix dialog behind the first stamp
  tap (about 15–20 KB) and inline-start My Coast's gates request before hydration. Neither is done here.
- **Research JS on a phone, 597 KB against 293.** v11's phone Research had no map; v12's one persistent chart is on
  every page, so MapLibre (281 KB, loaded lazily after first paint) is counted. Research's first-load JS before the
  map is about 316 KB. Its text LCP (2.44 s) is within budget.
- **Largest paint** under throttling is now the baked chart image (7.8–11.9 s), which v11 did not have; it starts
  at hydration and loads at low priority behind the text. The text budgets above are the ones the E report sets.
- Research's longest throttled task is 254 ms (ceiling 250).
