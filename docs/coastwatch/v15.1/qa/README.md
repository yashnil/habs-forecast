# Neritic v15.1 QA: journey, regression, offline, accessibility, screenshots

Recorded 2026-11-13. The closing pass ran at `feat/neritic-v15.1` **0989abe77**: every agent's last fix (A
5f1af6e6d, B bdffeba48) and the lead's v12-addresses spec fix. That head was merged into `v151/e` (34aeba8c5), which
adds only the QA files. Every run used a clean production build (`rm -rf .next`) on fixture data, with the server
clock pinned to 2026-10-13T06:00Z:

```
npm run data:fixture && npm run build
CW_E2E_PORT_BASE=7500 npx playwright test --shard=N/8 --workers=3       # N = 1..8, run outside a sandbox
CW_DATA_BASE_URL=/data/fixture/v1 CW_CONDITIONS_FIXTURE=tests/conditions-fixtures.json CW_NWS_API=1 \
  CW_FIXED_NOW=2026-10-13T06:00:00Z npx next start -p 7590
CW_SHOTS_HEADED=1 node scripts/v151-shots.mjs http://localhost:7590 <out> docs/coastwatch/v15.1/qa   # headed: GPU flight
```

Everything ran in installed Chrome on macOS. Nothing has run in a Linux container or in CI yet. Other agents shared the
machine during the earlier runs (load average up to 170); those runs' load timeouts are marked below.

## Start here: the ten demo moments (DIRECTION §8)

| # | Moment | Desk (1440 × 900) | Phone (390 × 844) |
|---|---|---|---|
| 1 | Today at the port, with the Explore California Waters entry in the ticket | [1440-01](1440-01-today-port.png) | [390-01](390-01-today-port.png) |
| 2 | The flight from the port to the whole coast (mid-flight, then settled) | [1440-02a](1440-02a-transition.png), [1440-02b](1440-02b-statewide.png) | [390-02a](390-02a-transition.png), [390-02b](390-02b-statewide.png) |
| 3 | Chlorophyll with the time rail | [1440-03](1440-03-chlorophyll-time.png) | [390-03](390-03-chlorophyll-time.png) |
| 4 | C-HARM and its caveats (model, past run, not toxin) | [1440-04](1440-04-charm-caveats.png) | [390-04](390-04-charm-caveats.png) |
| 5 | Observed HF-radar currents | [1440-05](1440-05-currents.png) | [390-05](390-05-currents.png) |
| 6 | Back to Moss Landing, then zoomed into Monterey Bay | [1440-06](1440-06-coastal-zoom.png) | [390-06](390-06-coastal-zoom.png) |
| 7 | My Coast ticket dragged on its own, then Reset layout (desk only; phones use sheets) | [1440-07](1440-07-panel-moved.png), [1440-07b](1440-07b-panel-reset.png) | n/a |
| 8 | Research at Crescent City (sparse: nearest pier 77 km) | [1440-08](1440-08-research-sparse.png) | [390-08](390-08-research-sparse.png) |
| 9 | About, Privacy and Sources on the new system | [about](1440-09-about.png), [privacy](1440-09-privacy.png), [sources](1440-09-sources.png) | [about](390-09-about.png), [privacy](390-09-privacy.png), [sources](390-09-sources.png) |
| 10 | Back on the saved trip ($2,602.24, planned) | [1440-10](1440-10-saved-trip.png) | [390-10](390-10-saved-trip.png) |

The frames come from a headed Chrome window, so the GPU composites the flight (02a is mid-flight). Video at desk
size, [walkthrough-v151.webm](walkthrough-v151.webm), under a minute: Today, the entry, the whole coast,
chlorophyll, C-HARM, play, back to the port, the saved trip. The script also takes 1280 × 800, 1100 × 800 and
320 × 640 frames. Those stay in the scratch output (`S/v151/shots/final/`) and are not copied here. I looked at
every size. The script logs any moment whose control is missing as MISSING and never fakes it; this run had none.

## Acceptance: `tests/e2e/v151/`

| Spec (owner) | Covers | Result at 0989abe77 |
|---|---|---|
| `journey.spec.ts` (E), 18 tests | Plan and save a Moss Landing trip. Today's entry flies to the whole coast, through intermediate frames, and Explore's panel really fades in (opacity 1). Back to Moss Landing returns to the port. The trip reopens with its storage byte-identical. The workspace's "Explore the water" opens the coast, and Back to your trip returns. The nav opens the coast with the port kept. Six shoreline points from the Oregon border to the Mexican border must be uncovered (the chart is the topmost element) at 1440, 1280, 1100, 390 and 320. Offline reopen after a browser restart. Reduced motion: no opening, and the camera jumps. Static tier: no MapLibre, a baked statewide field and its note. axe on statewide Explore (desk and phone) and on a moved My Coast panel | 18 pass |
| `ocean.spec.ts` (A) | Entry and flight at 1440 and 390, both borders, the nav, every live layer's legend, units and dates, play, the tap readout, back to the trip | pass |
| `panels.spec.ts` (B) | My Coast, Today and the trip workspace: move, resize, collapse, reset, persistence, keep on screen from 1440 to 1100, phones at 390 and 320 | pass |
| `research.spec.ts` (C) | Crescent City (sparse), Monterey (dense), Fort Bragg (nothing near), phone order | pass |
| `routes.spec.ts` (D) | Every informational page on the shell and the ds layout. The disclosures, Privacy, Terms and Sources rows are kept. axe at 1440, 390 and 320. Dusk and night. The opening with Skip and reduced motion | pass |

## Regression: the whole e2e suite

| Head | Tests | Passed | Skipped | Failed |
|---|---|---|---|---|
| 4b0ecd875 (V15 base, baseline) | 455 | 413 | 38 | 4 (all load timeouts; each passed twice when rerun alone) |
| d2ffb6241 (first full merge) | 524 | 480 | 38 | 6 (all fixed later by A) |
| 5019c75f9 (8fe0e7197 + QA) | 526 | 491 | 38 | 3 (load timeouts; passed when rerun alone) |
| b213fc994 (fix round) | 540 | 501 | 38 | 1 (v12-addresses:26, wording; fixed by C and the lead) |
| **0989abe77 (closing)** | **541** | **503** | **38** | **0** |

The closing run had no failures and nothing to rerun (load average 33–61 during the run).

The 38 skips are the suites' own conditional skips (live and pilot paths), the same number as at the V15 base.
The V15 acceptance suite (`tests/e2e/v15`, 49 tests) passes inside the final run. The intermittent in
`v15/plan.spec.ts:240` that QA found at the base was fixed by the lead (the outgoing lifecycle stamp lingered in
AnimatePresence).

Unit tests (`npm test`) at 0989abe77: **1923 passed, 74 skipped** (140 files passed, 1 skipped). `typecheck`, `lint` (0 errors, 14 warnings) and
`lint:design` are clean.

## Offline

`npm run test:offline` (`scripts/offline-proof.mjs`, service worker `neritic-sw-7`) at 0989abe77: **exit 0, 57 PASS,
0 FAIL**.
The journey's offline test plans the trip, opens Explore, closes the browser, then reopens the same profile with no
network. The workspace opens with $2,602.24, and the notice says it is a saved copy, never live.

## Accessibility (axe-core, WCAG 2.1 A/AA, serious or critical)

| State | Result |
|---|---|
| Statewide Explore, C-HARM with currents and stations, 1440 and 390 | none |
| My Coast with a panel moved (Reset layout showing) | none |
| Every informational route at 1440, 390 and 320 (`routes.spec.ts`) | none |
| The earlier a11y suites (`a11y.spec.ts`, `v12-a11y.spec.ts`, `v15/k-a11y-perf.spec.ts`), inside the full run | none |

## Responsive

The journey checks no sideways scroll and the whole coast uncovered at 1440, 1280, 1100, 390 and 320. B's panels
spec covers 1440 to 1100 and phones at 390 and 320. D's routes spec covers 1440, 390 and 320. The shots cover all
five sizes.

## Datasets: what is truly live (from A's audit, [science/datasets.md](../science/datasets.md))

| Dataset | Status | In Explore |
|---|---|---|
| Sentinel-3 OLCI chlorophyll-a 300 m (latest view and day frames) | **live** (production) | yes: Chlorophyll |
| VIIRS chlorophyll-a 750 m | **live** | yes: second track |
| Multi-sensor chlorophyll tiles; NASA GIBS VIIRS / PACE browse | **live** (tiles only, no values) | no |
| C-HARM v3.1 particulate DA probability, leads 0–3 | **live**, but the newest run (issued 2026-11-03) is a **past run** | yes: "C-HARM", drawn lighter, labelled past run |
| C-HARM Pseudo-nitzschia and cellular DA probabilities | **live** | not drawn |
| HF-radar surface currents, hourly, and the 24 h mean | **live** (no coverage north of Cape Mendocino) | yes: Currents |
| CalHABMAP shore stations (pDA, PN) | **live** | yes: Stations |
| NDBC buoys; CO-OPS tides and water temperature; NWS marine text | **live read** at request time | buoys in Stations; the rest on Today and My Coast |
| SST (MUR), 8-day chlorophyll outlook, gap-filled estimate | **staging only** | only with `CW_DERIVED_LAYERS=1` or local fixtures; otherwise "Not published yet" |
| CDPH shellfish biotoxin sidecar | **not published** (pilot) | no |

The screenshots use the e2e fixture, which keeps the derived layers (Temperature, Outlook) so the v13 tests of
them still run. On deployed data without the flag those layers show as "Not published yet". The fixture's chlorophyll
and C-HARM grids cover Monterey Bay only, so the statewide frames show a small field.

## Known limits

1. Everything was verified with installed Chrome on macOS under heavy shared load. It has not yet run in a Linux
   Playwright container or in CI.
2. `page.clock` (setFixedTime or install) fakes `performance.now` and `requestAnimationFrame`, which freezes Motion.
   After a client-side navigation, panels can sit at opacity 0 while `toBeVisible` still passes. The journey spec and
   the shots script shift only `Date` and assert opacity 1. The V15 helpers' `newContext` still uses setFixedTime, so
   screenshots taken from those specs can show blank panels. This is a test artifact, not a product bug.
3. The whole-coast check is a point check at six shoreline places. It proves that the coast's ends and its middle
   are visible, not that every label is legible at 320 px.
4. The statewide view on phones gives the map about the top half of the screen, under the sheet, so the coast
   reads small at 390 and 320.
5. The trips page can show "Reading the trips kept on this device…" for a moment after a navigation (V15 limit 2).
6. Not verified: the staging data channel (`CW_DATA_BASE_URL=<staging>`), the protected preview, real devices,
   Safari and Firefox, and the derived layers with `CW_DERIVED_LAYERS=1` on deployed data.
