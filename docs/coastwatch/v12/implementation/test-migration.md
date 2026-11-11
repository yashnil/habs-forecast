# v12 test migration: the old e2e specs

Before v12's integration, 385 of 520 Playwright tests failed on the main config (`playwright.config.ts`), almost
all because `/`, `/today`, `/explore`, `/coast` and `/research` are now the v12 pages. Each failing test outside
`nx-shell*.spec.ts` and `journeys/` was either **ported** (its behaviour is still required, so the assertion now
runs on the v12 pages) or **retired** (it described a layout or front door that no longer exists). Tests that still
passed on legacy pages (`/about`, `/sources`, `/fisheries`, `/ask`, `/account`, the APIs) were kept.

New specs: `v12-honesty.spec.ts` (16), `v12-a11y.spec.ts` (13, plus 2 `fixme` for the gaps below),
`v12-brand.spec.ts` (3), `v12-addresses.spec.ts` (5). Result on the fixture build: the legacy set and the new specs
run 114 passed, 0 failed, 3 skipped (the 2 `fixme` and `pilot-account-live`, which needs live credentials).

Where a behaviour was already covered by the v12 suites, the row says so: **shell** = `nx-shell*.spec.ts`,
**J1/J2/J3** = `journeys/first-visit`, `returning`, `explore`/`research`, **coast** = `journeys/coast.spec.ts`.

## Ported

| Old spec (tests) | What still holds | Now in |
|---|---|---|
| briefing-trust (11) | Official records not verified, with agency and date; a stale C-HARM run said as past; no station within 60 km said as a gap; the shellfish dispute unresolved | v12-honesty: Today stamps, Explore C-HARM, Research Shelter Cove, sources differ |
| trust (18) | Notices "not verified"; a failed update keeps its last good dates; C-HARM says what it is not; satellite dates as a range; currents "no streak is no data"; nothing current said, not a bare chart | v12-honesty: failed update, Explore chlorophyll/currents/C-HARM, no dataset |
| explore-v11 (16), explore (10), explore-v10 (5), v5-map (12) | "Algae, not toxin" and "no view" beside the ramp; C-HARM's mandatory words; a past run never forward-looking; the outlook's own kind, typical error and "no signal"; real frames only; URL keeps the view | v12-honesty Explore; J3 (frames, tap reading, URL, reduced motion) |
| m3 (11 of 18), m6 (9), research (23), sync (4), ux-final (5 of 6) | Reported 0 in its own lane; fixed axes; model history gap hatched; the window filters the log and export; station picked on the chart or list; v2 links; unavailable dataset worded | v12-honesty Research; J3 research (brush, export, links, phone); capabilities.md |
| economics (15), mycoast-v10b (7), mycoast-pass (4), fishing (8 of 9) | Nothing pre-filled; references beside fields, entered on a tap with their source; "hypothetical"; waiting words, never 0; who pays fuel; the grid on touch; season and tests quoted with source and date | v12-honesty My Coast; coast |
| today (19), since (8 of 10), pilot-journey (2) | Answers first; missing sidecars worded; since your last visit computed, never invented; save for offline; provenance from every value | J1, J2 (change line, M2 sidecars, offline copy, provenance) |
| discrepancy (4) | A release that appears to end a notice is set apart, never shown as lifted, both quoted | v12-honesty "sources differ" (Today, Monterey shellfish, FAILED fixture) |
| readiness (6), states (6) | No dataset: an explicit unavailable state, never an empty chart; a days-old run said so | v12-honesty no dataset (Explore, Research), C-HARM past run |
| a11y (6 of 9), briefing axe, fishing axe, since axe | No serious or critical axe violation | v12-a11y: Home, Today, Explore, My Coast, Research at 1440 and 390, provenance sheet |
| navigation (8 of 9) | First Tab offers to skip the bar; About and Sources reachable; v2 links open the same view | v12-a11y skip link; v12-addresses |
| layout (6 of 12), hardening L1 | Phone controls at least 44 px; no sideways scroll | v12-a11y 320 px check (layout's sideways checks still pass and stay) |
| brand (4 of 5) | No old product name; titles, metadata, manifest; old keys and links break nothing | v12-brand |
| pilot-front-door (6), today-picker (13) | Old deep links still open Explore; Explore has its own address; campaign tags keep the front door | v12-addresses; shell (redirects, Today at / for a saved port) |
| regulatory (9 of 16) | "Not verified" in words at every width; Del Norte razor clam notices listed | regulatory.spec now runs on the legacy pages that carry the pill (incl. Del Norte on /sources); v12 pages: v12-honesty stamps |

## Retired

| Old spec | Why |
|---|---|
| splash, motion | The v9 launch screen and raster fades are gone; the v12 opening and reduced motion are in shell. Their last passing tests only asserted the splash was absent. |
| briefing, ergonomics, parity, v8, explore-rework, explore-shell | The v7–v11 briefing, dock, sheet peek, place menu and tool rows; v12 has none of these surfaces. |
| currents, coastline, coastline-fallback, gridtiles | v4 raster and particle internals (`__cwMap` layers, tile fallbacks, cell outlines); v12 draws fields and streaks through its own layers, covered by J3 and the unit tests. |
| hardening | Old dock, sheet and credits chip; its last passing test (L4) matched no element on v12. |
| offline, offline-recovery, mycoast-kept-offline | The field-pack dialog and "keep this briefing" flow are not in v12 (see Missing). Old packs opening at the new addresses is shell (nx-shell-offline). |
| evidence (8 of 9), evidence-v3 (3 of 12) | Claim rows of the old briefing and harbour panel; the API checks of the same graph still pass and stay. |
| pilot-front-door, today-picker (layout parts) | The pilot's redirect front door and the v11 port directory sheet; v12's one entry is in shell and J1. |
| m3, m6, research (layout parts), ux-final M2–M7 | Old station picker, heatmap, tabs, popovers, phone jump links. |
| a11y daylight-mode test | The v10 "daylight" theme is replaced by Day, Dusk and Night (shell). |
| navigation notices control, research-preview disclosure | The masthead notices pill and preview banner are not on the v12 pages; each official figure carries "not verified" instead. |
| helpers.ts | Drove the old layouts; no remaining spec imports it. |

## Missing in v12

Behaviour the old specs required that the v12 pages do not have (not papered over; for the owner):

1. ~~**No offline notice.**~~ Resolved in the final pass: every v12 page's margin opens with the ds `Offline`
   notice when the browser is offline or the page is a saved copy (`src/nx/offline/OfflineNotice.tsx`): when it
   was saved, what this device holds, and what on this page waits for a connection.
2. ~~**No way to save a field pack**~~ Resolved: Menu, Offline saves the port's field pack (`src/nx/offline/pack.ts`,
   cache `neritic-pack-field`): Today, My Coast (gates for every fishery, the sea window; the worksheet is in
   device storage), Explore's latest frame of each layer around the port, Research's open station record, the
   baked chart in all three modes. The Menu says what is saved and when; `npm run test:offline` proves it.
3. ~~**No list of every current official notice**~~ Resolved: `/notices?place=…` lists every current notice
   at the port (statewide ones included), those whose reach there is not settled, "sources differ" records
   (both quoted, never lifted) and every other notice on the coast (the Del Norte razor clam records among
   them), each with agency, action, dates, area, source links and "not verified". Reached from Today's Notices
   and Others rows and from Alerts (`/account`). `v12-notices.spec.ts` (4).
4. **T37 for C-HARM:** no "a low value does not mean open or safe" near the probability (trust.spec "a low value is not safe").
5. **ds Stamp does not return focus** to the stamp when its provenance sheet closes (components.md promises it):
   `v12-a11y` fixme.
6. **The wordmark link is 27 px tall** at 320 px, under the 44 px floor: `v12-a11y` fixme.
7. Seen while porting: the unavailable states on Explore and Research print the server's file path from the
   error (`manifest not available (ENOENT … /public/data/…)`), and the `/` title repeats the name
   ("… for your port — Neritic").

## `npm run test:offline` (scripts/offline-proof.mjs)

Ported to v12 in the final pass. The proof opens Today at Moss Landing, saves the port's field pack from the
Menu, empties the worker's own code cache (as after a deploy), stops the app server, and reopens the same
browser profile with every request sent to a dead proxy. `/`, `/today`, `/coast`, `/explore` and `/research`
must each open from the pack at 390 and 1440 px with their own content, the baked chart (static tier, every
image loaded), Explore's baked frame and the offline notice; a tab tapped offline opens the next saved page; an
unsaved page says it is offline and links to the saved ones; no request leaves the device, no same-origin
request fails, no page error. Result on the fixture build: 41 checks, all pass ("PROVEN").
