# Neritic v15 QA: acceptance, offline, accessibility, performance

Recorded 2026-11-13; the final numbers below are for `feat/neritic-v15` at f54e10d1a (the code of the preview). Every run below used the
production build on fixture data, with the server clock and the device clock both pinned to 2026-10-13T06:00Z:

```
npm run data:fixture && npm run build
CW_DATA_BASE_URL=/data/fixture/v1 CW_CONDITIONS_FIXTURE=tests/conditions-fixtures.json CW_NWS_API=1 \
  CW_FIXED_NOW=2026-10-13T06:00:00Z npx next start -p 6590
CW_E2E_PORT_BASE=6590 npx playwright test tests/e2e/v15        # run Playwright outside a sandbox
node scripts/v15-shots.mjs http://localhost:6590 <out> docs/coastwatch/v15/qa
```

Local runs use installed Chrome on macOS. They have not been run in a Linux container or in CI yet.

## Start here

| Journey | Desk (1440 × 900) | Phone (390 × 844) |
|---|---|---|
| Plan a trip, then land on its workspace | [1440-1-planned.png](1440-1-planned.png) | [390-1-planned.png](390-1-planned.png) |
| Compare two departure periods | [1440-2-departure-comparison.png](1440-2-departure-comparison.png) | [390-2-departure-comparison.png](390-2-departure-comparison.png) |
| Change feed after a recheck (NWS wind raised for the departure day) | [1440-3-change-feed.png](1440-3-change-feed.png) | [390-3-change-feed.png](390-3-change-feed.png) |
| Departure briefing opened offline after a browser restart | [1440-4-offline-briefing.png](1440-4-offline-briefing.png) | [390-4-offline-briefing.png](390-4-offline-briefing.png) |
| Planned against actual, recorded offline | [1440-5-actual-vs-planned.png](1440-5-actual-vs-planned.png) | [390-5-actual-vs-planned.png](390-5-actual-vs-planned.png) |
| History, offline | [1440-6-history.png](1440-6-history.png) | [390-6-history.png](390-6-history.png) |

Walkthrough at the desk size: [walkthrough-1-online.webm](walkthrough-1-online.webm), which covers planning, the
comparison and the recheck. [walkthrough-2-offline.webm](walkthrough-2-offline.webm) covers the same profile reopened
with no network: the briefing, the outcome and the history. The script also takes the 1280 × 800 (laptop) and 320 × 640 frames, which stay in
the scratch output and are not copied here.

## Acceptance (DIRECTION §17): `tests/e2e/v15/`

| Test | Spec | What it proves | Result |
|---|---|---|---|
| A: plan and save (plus the offline recheck staying in place) | `plan.spec.ts` (agent A) | Moss Landing, commercial Dungeness, regression economics, Save trip lands on the workspace with $2,602.24 | pass |
| B: reopen with a changed forecast | `bcd-recheck.spec.ts` | `/coast/data?part=sea` is routed to the fixture answer with Wednesday's wind raised. The feed has exactly one item ("stronger winds", planned 5–10 kt, now 20–25 kt). The stored planning snapshot is byte-identical before and after. "What Neritic showed you" opens it | pass |
| C: revised CDFW record | `bcd-recheck.spec.ts` | The season text and URL are revised and a whale measure changes. The feed shows "Agency information changed; review the original CDFW page." with the new URL as its source link, plus the whale change. The feed contains none of `safe`, `legal`, `allowed`, `all clear` or `approved` | pass |
| D: failed sources | `bcd-recheck.spec.ts` | The sea endpoint returns 503 and the gates report `testsError`. The feed is `incomplete`, names NWS, CDPH, the buoy and tides as not checked, and never says "No changes found" | pass |
| E: departure comparison | `ef-departure-outcome.spec.ts` | Wednesday night is dropped from the saved forecast and shows as "no NWS forecast". Tuesday and Wednesday compare with exactly one comparison panel, which shows the wind direction change (NE → W against NE → NW). With a 12 kt limit set in Edit plan, Tuesday night crosses it ("Wind stated up to 15 kt; your limit 12 kt") | pass |
| F: complete trip | `ef-departure-outcome.spec.ts` | Actual 450 lb at $8.50, 35 gal at $8.00, $120 bait and ice, $700 crew. The result is +$122.76 net to boat against plan, split as catch +$450.00, price −$225.00, fuel −$32.80, supplies −$20.00, crew −$49.44. "Profit" does not appear | pass |
| G: privacy | `g-privacy.spec.ts` | Trip data lives only in this context's `nx-trips-v1:` keys. No request carries its notes, area, name or money (the opaque id is in the page address). A second context sees nothing. Deleting one trip from the history removes it and its snapshots. Delete everything removes every trips key and the kept trips page | pass |
| H: offline | `h-offline.spec.ts` | The trip is planned on My Coast. The browser is closed and the persistent profile relaunched with no network. The workspace, the briefing ($2,602.24, "Planning snapshot", marked as a saved copy) and the history open. The economics are edited offline (500 lb gives $3,322.24) and the outcome is recorded offline. A second offline restart keeps every edit | pass |
| I: mobile lifecycle | `plan.spec.ts` (agent A) | The full lifecycle at 390 and 320 px: nothing overlaps or scrolls sideways | pass |
| K: known evidence survives (review blocker) | `k-known-evidence.spec.ts` | NWS adds a small craft advisory and raises the departure day to 15–25 kt, and a recheck finds both. The sea endpoint then returns 503 and the next recheck is incomplete. Offline, a third recheck compares with the saved copy. At every step the workspace's departure day and the briefing still show the advisory and 15–25 kt, and the briefing says when each was last read. The feed never says "nothing new" or "No changes found" | pass |
| J: existing pages | `j-existing.spec.ts` | Today, Explore, My Coast and its worksheet, Research, Alerts and Notices load at desk and phone sizes with their primary element and no page errors. The chart host is mounted | pass |

Final run of `tests/e2e/v15` at f54e10d1a (49 tests: A–K, a11y, perf, the not-found state and the desk/laptop layout at 1440, 1280, 1100 and 1024 px): **49 passed, 0 failed** (inside the full run below).

Routed tests (B, C, D) block the service worker in their context, because once the trips page has registered it, the
worker fetches `/coast/data` itself and bypasses `page.route`. K keeps the worker on and routes on the context instead,
which also catches the worker's own requests.

## Regression (the whole e2e suite)

The full `tests/e2e` (455 tests: v13, v14, journeys, a11y, offline, pilot, v15 and the rest) ran under the official
`playwright.config.ts` (six fixture servers, `CW_E2E_PORT_BASE=6600`) in 8 shards at 3 workers, at f54e10d1a:
**417 passed, 38 skipped, 0 failed** (455 tests, at f54e10d1a). The 38 skips are the suites' own conditional skips (live and pilot paths). 

Unit (`npm test`, vitest) at f54e10d1a: **1857 passed, 74 skipped**. `typecheck`, `lint` (0 errors) and
`lint:design` are clean.

## Offline

- `npm run test:offline` (`scripts/offline-proof.mjs`, the v12/v13 field-pack proof) passes against service worker
  `neritic-sw-7`: exit 0, 57 PASS and 0 FAIL (at f54e10d1a).
- The trips page keeps itself, its code, its fonts and the static tier's chart pictures in `neritic-trips-v1` each
  time it opens online. Offline, every `/coast/trips` address (history, `?id=`, `&view=briefing`) is answered from that
  copy, marked with the time it was kept. The offline notice then reads "Offline. Showing what this device saved …";
  the page never says "live".
- Unit: `tests/unit/v15/tripsOffline.test.ts`. It covers what is kept, that a new build replaces the old code, the
  worker version and marks, and that Delete everything clears trips and the cache.

## Accessibility (axe-core, WCAG 2.1 A/AA, serious or critical): `k-a11y-perf.spec.ts`

| State | Desk | 390 px |
|---|---|---|
| History (`/coast/trips`) | none | none |
| Workspace (`?id=`) | none | none |
| Briefing (`&view=briefing`) | none | none |

## Performance (cold load, desk, local production build): `k-a11y-perf.spec.ts`

Script bytes are counted on the wire (CDP `encodedDataLength`, compressed as served) and LCP comes from a
`PerformanceObserver`. Two isolated runs before the final merge, then the run inside the final parallel suite:

| Page | JS transferred | Script files | LCP, isolated | LCP, in the final parallel suite |
|---|---|---|---|---|
| `/coast?view=trip` (the worksheet the workspace replaces) | 780 KB | 22 | 244–256 ms | 836 ms |
| `/coast/trips` (history) | 720 KB | 42 | 208–224 ms | 716 ms |
| `/coast/trips?id=` (workspace) | 720–759 KB | 42 | 288–372 ms | 412 ms |
| `/coast/trips?…&view=briefing` | 720 KB | 42 | 252–280 ms | 296 ms |

The spec's budget allows the workspace at most 1.35 × the worksheet's script and an LCP under 2.5 s, and the
workspace passes it. LCP rises when the whole suite shares one machine, so read the figures as relative, not absolute.

## Known limits

1. Everything here was verified with installed Chrome on macOS. It has not yet run in a Linux Playwright container or CI.
2. After an offline navigation, and briefly after a recheck, the trips page can show its "Reading the trips kept on this
   device…" placeholder for a moment before the history or the workspace returns. The screenshot script waits for the
   page to settle.
3. (Resolved at 77606658c: the change feed, the trip's state and its next action are readable from their first frame;
   motion is a 0.16 s settle, none with reduced motion.)
4. Offline at phone widths, the static-tier chart picture can draw as a partial rectangle over the water. This is shell
   behaviour, not specific to trips.
5. The trip id travels in `?id=` (an opaque random key), so the site's server sees it like any address. Nothing the trip
   holds is sent (Test G).
6. The CDPH test site and the buoy are drawn only when the snapshot gives their position. Snapshots taken before D
   added positions draw the port and the zone only.
