# 05 — Evaluation: three critique passes, accessibility and performance

**Method:** visual review of screenshots at 1440 × 900, 1280 × 720, 1024 × 768, 390 × 844 and 320 × 640, plus two scripts:
- [`scripts/audit.mjs`](prototypes/scripts/audit.mjs) runs on a production build: axe-core, cumulative layout shift, LCP, long tasks, text under 12 px, targets under 24 px, horizontal overflow, and whether the place title and an official link are inside the first screen.
- [`scripts/perf.mjs`](prototypes/scripts/perf.mjs) measures journey frame times, day-switch latency and selection latency.

Raw results are in [`evaluation/`](evaluation/).

**No stakeholder testing was done.** "First-time user" claims below are heuristic checks (is the place named and the primary action visible on the first screen?), not observed behaviour.

## Pass 1 — visual critique (desktop and phone)

Found and fixed during the build:

| Finding | Why it mattered | Fix |
|---|---|---|
| The no-value hatch showed **through** the translucent forecast | The whole field read as "no data" | Valued cells are pre-mixed with the sea and drawn opaque, so the hatch only shows where there is no value |
| The hatch covered the whole sea while the forecast was loading or had failed | A delay looked like an absence | The basemap stays calm until a field exists; failures say so in words |
| "Nearest sample 1,503 days ago" at Monterey | Mixed the last *visit* with the last *toxin* sample | Separate last-sampled and toxin-last-measured dates |
| The disputed-notice box showed for points far from Monterey County | Irrelevant prominence | Shown only within 30 km of the record's area; later moved into the official row |
| Phone: controls stacked over half the map; the one-line answer appeared twice; status line ran to three lines | Map unusable one-handed | One control row riding above the sheet, field mode in the sheet, short status line |
| The legend collided with the time bar; the purpose card covered the coast | Composition | Legend top-right; purpose card top-right |
| Provenance threads did not draw | An effect raced the map's readiness | A map-ready state feeds the thread effect |

## Pass 2 — automated audit (first full run)

[`evaluation/audit-pass2.json`](evaluation/audit-pass2.json)

| Finding | Where | Fix |
|---|---|---|
| CLS 0.10 (selected) and 0.21–0.36 (Research) | Desktop map panel | The disputed notice moved inside the official row. The one-line answer reserves three lines. Research holds its expanded ledger behind a same-size skeleton until sources settle (≤ 800 ms). The masthead count has a fixed width |
| CLS 0.12–0.22 | Station page | Header height reserved on desktop |
| `heading-order` | Map | `h2` (visually hidden) for the ledger; notice titles as `h3` |
| `landmark-one-main`; no official link on screen | Station | `main` landmark; status line with CDFW and CDPH links added |
| 16 px horizontal overflow at 320 px | States | `minmax(min(320px, 100%), 1fr)` |
| `region`, `scrollable-region-focusable`, 11 px text | Atlas direction | Landmarks, a focusable scroll column, a 12 px floor |

## Pass 3 — final

[`evaluation/audit-pass3.json`](evaluation/audit-pass3.json): every route at every viewport.

| Measure | Result |
|---|---|
| axe-core violations (map canvas and attribution excluded) | **0** on all 6 routes × 5 viewports |
| Horizontal overflow | **0 px** everywhere, including 320 × 640 |
| CLS | ≤ 0.085 (map), ≤ 0.048 (station), ≤ 0.074 (Atlas). All under 0.1 |
| LCP (local preview) | 360–420 ms |
| Place title visible on the first screen | Yes, on every map and station view |
| Official link (CDFW or CDPH) visible on the first screen | Yes on every product view. The States specimen page is a design document, not a product view |
| Text under 12 px | 0, except 1 node in the Atlas direction (a superscript note number set to 12 px renders at about 11.9 px) |
| Long tasks | 390–570 ms at map start-up (MapLibre initialisation and decoding the 2448 × 3208 relief images). None during interaction |

### Interaction performance

[`evaluation/perf-gpu.json`](evaluation/perf-gpu.json). These are Chromium figures with GPU rendering on the development Mac, not a phone.

| | Result |
|---|---|
| Coastal journey Santa Cruz → Morro Bay, frame time | p50 16.7 ms, p95 16.7 ms (60 fps); max 16.8 ms on the second run and 150 ms on the first, which coincided with tile loading |
| Day switch (click to the selected state on screen) | 9–59 ms |
| Water selection (click to ledger updated) | 5–21 ms |
| Field render (351 × 391 cells → 1404 × 1970 Mercator canvas) | about 40 ms per render, about 7 ms to encode |

**Software WebGL (headless, no GPU)** gives a 15 fps journey and about 300 ms selections ([`perf-software-gl.json`](evaluation/perf-software-gl.json)). This is a reminder that low-end devices without GPU acceleration will see choppier camera motion. The journey still ends where the reader asked, and any touch stops it.

### Bundle

Production build, gzip:

| Chunk | Size |
|---|---|
| MapLibre (already a coastwatch-web dependency) | 291 kB |
| React and app shell | 64 kB |
| Explorer | 15.6 kB |
| Ledger | 4.8 kB |
| Station | 10 kB |

**No new runtime dependency** is introduced: motion is CSS, WAAPI-free `requestAnimationFrame` and MapLibre paint transitions.

## Heuristic first-time-user check

For each persona's first screen:

| Persona and view | Place named? | Primary action visible? | Answer in ≤ 1 sentence? |
|---|---|---|---|
| Commercial, phone, after picking a point | "Off Santa Cruz" | Sheet peek: answer plus field mode; official links in the masthead | Yes |
| Shellfish harvester, desktop | Yes | CDPH and CDFW buttons in the official row and the empty state | Yes |
| Researcher, Research mode | Yes, with coordinates | Exact values and the per-lead table | n/a (Level 3) |
| General visitor, first visit | "Choose your waters" | Purpose card plus search | The four marks are explained in one sentence |

## Not tested or unverified

- Real users, with or without assistive technology. axe is necessary but not sufficient.
- Screen-reader announcement order in the sheet on iOS VoiceOver and Android TalkBack.
- Real phones: frame rate and battery during journeys; virtual-keyboard overlap with search. The sheet listens to `visualViewport`, but this was not tested on a device.
- Font-failure behaviour beyond fallback stacks, and right-to-left languages.
