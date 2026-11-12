# Neritic v13: the independent review (agent F)

Reviewed: `feat/neritic-v13` at `f50375db3` plus `a2742ccbb` (the first-visit test fix), production build
(`npm run build`), by hand-driven Playwright with **staging data** (real clock, 2026-11-12) and with the fixtures
(clock 2026-10-13), at 1440 × 900, 390 × 844 and 320 × 640. The reviewer wrote none of the product code.
Not in the reviewed build: four commits still on `v13/a` (the land-adjacent chlorophyll pixels, a sample that
arrives in a link, the kit's selection and legend words).

Stills: [before-after.md](before-after.md). Clips: [media/](media/README.md). Evidence for the defects below:
[evidence/](evidence/).

## Verdict

v13 is a different and much better product than v12 on Research, My Coast and Alerts, and those three would pass
the owner's gate on a desktop today. It does **not** pass as a whole: on a phone, Explore's layer switch and time
controls cannot be reached while the chart is visible (F-01), and several first-screen states at 1440 are visibly
unfinished (F-02, F-03, F-10). Nothing found is a data-correctness or arithmetic error.

## Defects, ranked

| # | P | Owner | Page / size | What | Evidence | Expected fix |
|---|---|---|---|---|---|---|
| F-01 | **P0** | shell-ds, Explore | Explore, Today, Notices at 390 and 320 | The phone sheet's scroller is taller than the part of the sheet on screen, so scrolling stops with the last controls under the destinations bar. On Explore with a reading open there are four sheet heights: the layer switch is reachable only from the third and **Play and the time rail only at full height, where the chart is covered**. At 390 the time rail is also clipped on the left ("ime", "hlorophyll"). Today's "Notices" and "Others" rows and Notices' last links rest under the bar. This is v12's "Save cannot be reached" defect, moved. | `evidence/F-01-*.webp`; `tests/e2e/v13/guards.spec.ts` G2 (5 expected failures) | The scroller ends where the visible sheet ends at every height (bottom inset = hidden part). On Explore the layer switch and Play stay above the bar at the resting height (or live on the chart). The rail fits 390 px. |
| F-02 | P1 | My Coast | trip, 1440 | After "Save as new" the desk header grows to 310 px (a third of the screen): the saved name wraps, then Rename, the saved time, two rows of buttons and the confirmation, beside an empty top-left quarter. The waterfall leaves the first screen. | `evidence/F-02-*.webp` | Name on one line (truncate, full in title), the confirmation transient, buttons in one row; header no taller than before saving. |
| F-03 | P1 | My Coast | trip, 1440 (gate state 6) | With some fields blank, the waterfall's per-bar "waiting for …" labels overprint each other and run past the plot into the pane divider. Illegible. | `evidence/F-03-*.webp` | One waiting sentence under the plot, or labels that never overlap (clip to the bar's column). |
| F-04 | P1 | Research | 1440 | "Latest sample" and Enter on a point select the sample, but the inspector opens at y ≈ 1470 of a 900 px screen; the first screen shows only a tooltip (itself clipped by the plot's bottom edge). A reader does not learn the exact value, QC and source are one scroll away. The phone does this well (a sheet). | `evidence/F-04-*.webp` | The inspector beside or directly under the plot within the first screen at 1440, or scrolled into view on selection. (`v13/a` has an unmerged commit that pins a card; re-check after it lands.) |
| F-05 | P1 | Research (layout with shell-ds) | `/research?place=…`, phone, Slow 4G | Layout shift 0.279 at first paint in 5 of 6 loads (`n12-margin` moves). Over the 0.1 "good" line; My Coast, same layout, is 0.004. | `F/final/perf-phone-cls.json` | Reserve the page's geometry before the station is resolved from `place` (as My Coast does for its gates). |
| F-06 | P1 | Explore, shell-ds | Explore, 390 and 320 | The chart's own buttons (+, −, re-centre, "Fit the data": 92 × 180 px) cover the port's name and about a quarter of the phone chart; half its width at 320. | `evidence/F-06-*.webp` | One compact control on a phone (zoom is a pinch there); "Fit the data" in the legend or the menu. |
| F-08 | P1 | alerts-personal (offline), My Coast | `/coast?…&view=trip`, offline | The field pack saves the brief's HTML and serves it for the trip view, so every offline open of the workspace raises React error #418 (hydration mismatch) and rebuilds on the client. The page then works (the saved trip reopens at $2,602.24 and takes an Update). `npm run test:offline` is NOT PROVEN on this one check (56 pass, 1 fails). | `F/final-offline.log` | `src/nx/offline/pack.ts`: save `/coast?place=…&view=trip` as its own page (or have the worker key on `view`). |
| F-09 | P1 | Today (test) | `journeys/first-visit.spec.ts:100`, 390 | Fails every run after `a2742ccbb`: the large "Dec 15–18" is at y 881, under the destinations bar (788). The one-line gate summary above it does carry the date and is in view. | `F/final-e2e.log` | Assert what the page now promises (the Whale line of `nx-gate-summary` in view), not the block below it. |
| F-10 | P1 | Explore | 1440 | "Play" is cut off at the right edge of the time rail ("Pla"). | `evidence/F-10-*.webp` | Keep the button inside the rail's width. |
| F-07 | P2 | My Coast | trip, 320 | At the end of the page "Copy numbers-only link" rests under the fixed result bar. | guards G2 (owner's trip at 320) | Bottom padding equal to the bar. |
| F-11 | P2 | My Coast | saved trips | An automatic name is not refreshed by Update: a trip updated to $10/lb is still called "… 400 lb at $9 …". | `F/final-offline.log` ("Updated “… at $9 …”") | Automatic names follow the numbers until the reader renames the trip. |
| F-12 | P2 | My Coast, ds | saved trips, all sizes | The unticked Compare box is a solid dark square: it reads as ticked. | `evidence/F-12-*.webp` | An empty box when unticked. |
| F-13 | P2 | Research | 390, and 1440 whole record | The "C-HARM threshold, 500 ng/L" label is overprinted by the data. | `evidence/F-13-*.webp` | Place the label in the clear (right end, or above the plot). |
| F-14 | P2 | My Coast | brief, all sizes; 320 | "Work out the money" touches the paragraph above it. At 320 the fishery name collides with the select's arrow, and the fixed header takes 265 of 640 px, leaving about one field of working room. | `evidence/F-14-*.webp` | A gap above the button; a shorter fixed header at 320 (status on one line). |
| F-15 | P2 | shell-ds | Research and My Coast, phone | The chart strip is labelled "Chart · Moss Landing" while it shows a station (Research). On My Coast's strip the port's name is struck through by the 37°11′ N zone line. | `F/sheets/p-research.png`, `p-coast.png` | Name what the chart shows; keep the label off the line. |
| F-16 | P2 | Explore | reading, chlorophyll | "Neritic's station list does not say which piers measure this." is our data architecture, said to the reader. | `F/final/staging/04b-*-1440.png` | Say what the reader can do: "No pier nearby is known to measure chlorophyll. Nearest pier: …". |
| F-17 | P2 | My Coast | trip, 1440, scrolled | A sliver of the waterfall's axis shows under the pinned result. | `F/final/staging/07a-*-1440.png` | Clip the scrolling column under the pinned block. |
| F-18 | P2 | tests (Today) | `journeys/returning.spec.ts:28`, `first-visit.spec.ts:36` | See "The recording time-outs" below: not a product defect. | | Record video only with `NX_MEDIA=1`, or run the recording journeys in a project with one worker. |

## The twelve gate states

The brief lists twelve states, not twelve questions; each is answered here as "does this state pass the owner's
gate at all three sizes?".

| # | State | Pass? | Why |
|---|---|---|---|
| 1 | First-time Today | **Yes** | Purpose and what a port leads to are said; three taps; three gates in three lines. |
| 2 | Returning Today | **Yes** at 1440; **No** on a phone | Advisories with source and issue time, old test worded as old. Phone: the last rows rest under the bar (F-01). |
| 3 | Explore with a layer | **No** | "Play" clipped at 1440 (F-10); phone layer switch and time controls (F-01), chart buttons (F-06). |
| 4 | Explore with a harbour | **No** on a phone, **Yes** at 1440 | The reading is tied to its cell, date and grid, with hand-offs that arrive. Phone as above. |
| 5 | My Coast start | **Yes** | The work has the width; gates and sea window with sources. Small spacing faults (F-14). |
| 6 | Economics with inputs | **No** | The overprinted waiting labels (F-03). |
| 7 | Economics with results | **Yes** | $3,600.00 / $247.20 / $347.20 / $650.56 / $2,602.24 / 39 lb on the page at every size; the result in view while typing; "not profit or take-home pay"; the delayed-season flag. |
| 8 | Research, a measured pDA sample | **Yes** | One large plot, availability by variable, model apart. Label collision (F-13). |
| 9 | Research, nothing in range | **Yes** | The best state in the product: says not collected, why, when it last was, and offers three ways on. |
| 10 | Research sample inspection | **Yes** on a phone; **No** at 1440 | Complete content; at 1440 it is below the fold (F-04). |
| 11 | Saved scenarios and alerts | **Yes**, with F-02 | No duplicates, Update / Duplicate / Rename / Compare; Alerts honest and in the shell. The header after saving (F-02) is the blemish. |
| 12 | Phone versions | **No** | F-01, F-06. My Coast and Research are good on a phone. |

## Journeys

| Journey | End to end? | Notes |
|---|---|---|
| J1 first-time | Yes | |
| J2 returning | Yes | The change line and Alerts say what changed; limitations are in the gates. |
| J3 trip | Yes | To the cent; save, update, duplicate, compare, reopen after reload and restart. F-02, F-03, F-11. |
| J4 research | Yes | Monterey Wharf pDA: latest real sample (Aug 31, 2022) in one press, source row, QC, range, model apart, export. F-04. |
| J5 exploration | Yes at 1440; **blocked on a phone** | F-01. Explore → Research honours the variable and a station within 25 km; Explore → My Coast carries the cell and date. |
| J6 offline | Yes | Field pack and a trip saved, server stopped, browser restarted with no network: both reopen, an Update made offline is what a second offline restart opens. One hydration error (F-08). |
| J7 alerts | Yes | By hand on staging and in `v13/alerts.spec.ts`: follow, a controlled change, exactly one alert with before/after, source and "Not verified"; a second due check adds none. |
| J8 phone | **No** | F-01, F-06, F-07, F-14. No sideways scroll anywhere; no page is rebuilt after hydration (G1). |

## Performance

`scripts/perf-routes.mjs`, fixture data, medians, v12 → v13. Budgets: JS ≤ v12 + 60 KB; phone TBT ≤ 450 ms;
desktop CLS ≤ 0.05.

| Condition | Route | JS KB | TBT ms | CLS | Text LCP ms | Map ready ms |
|---|---|---|---|---|---|---|
| desktop | home | 579 → 637 | 0 → 0 | 0 → 0 | 152 → 136 | 409 → 341 |
| | today | 578 → 635 | 0 → 0 | 0 → 0 | 232 → 164 | 466 → 359 |
| | explore | 622 → 650 | 0 → 0 | 0 → 0.008 | 136 → 92 | 509 → 344 |
| | coast | 612 → 653 | 0 → 0 | 0.031 → 0 | 100 → 60 | 696 → 345 |
| | research | 604 → 647 | 0 → 0 | 0.028 → 0 | 280 → 160 | 518 → 388 |
| phone 390, CPU 4x, Slow 4G | home | | 164 → 102 | 0 → 0 | 4660 → 5154 | 14242 → 14746 |
| | today | | 272 → 204 | 0 → 0 | 4586 → 4948 | 14180 → 14563 |
| | explore | | 215 → 122 | 0.008 → 0.008 | 2168 → 2210 | 14382 → 14604 |
| | coast | | 228 → 141 | 0 → 0.004 | 5366 → 2228 | 14414 → 14230 |
| | research | | 416 → 181 | 0.007 → **0.279** | 2248 → 2296 | 14925 → 15081 |

- JS: within budget on every route (+28 to +58 KB; Home and Today are at the edge).
- Phone TBT: within budget everywhere, and lower than v12 on every route.
- Desktop CLS: within budget; My Coast's and Research's v12 shifts are gone.
- Miss: phone Research CLS (F-05). Near miss: phone Home text LCP +10.6 % (the extra 58 KB of script ahead of it).
- Desktop at CPU 4x, Slow 4G behaves like the phone rows (TBT 102–201 ms).

## The recording time-outs

`journeys/returning.spec.ts:28` and `journeys/first-visit.spec.ts:36` timing out at `browserContext.close` in a
parallel run is **video finalisation under CPU contention, not a page that never goes idle**.

- Every route goes fully idle (`scripts/v13-idle-probe.mjs`, desktop and phone, 8 routes): after 8 s, over 6 s,
  0 animation frames requested, 0 frames composited, ≤ 1 ms/s of main-thread work, 0 requests, no running
  animation. Nothing here costs a phone battery at rest.
- Closing a context that recorded 25 s at 1440 × 900 takes 0.65 s alone and 0.67–0.70 s with six recording at
  once on an idle machine; with the CPUs saturated it takes 2.7–7.4 s, and it grows with the length of the
  recording and the contention.
- Both tests passed in this review's parallel run of the whole suite (271 passed; 18.8 s and 10.6 s). The lead's
  run shared the machine with other agents' builds.

## What was added to the automated proof

- `tests/e2e/v13/guards.spec.ts` (29 tests): G1 no page element is replaced after hydration on any of eight
  routes at 390 px, CPU 4x (all pass: v12's remount is gone); G2 every control of every route is reachable by
  scrolling alone at 390 and 320 (five expected failures and one filtered control, each tied to F-01 or F-07
  with `test.fail`, so the fix must delete the entry); G3 hand-offs arrive (Explore → Research on the layer's
  variable at a station ≤ 25 km with the window; Research honours a variable other than pDA; Explore → My Coast
  shows the cell and its date and carries them into the trip; Today's gate links keep the port they were opened
  from when it is not the home port).
- `scripts/offline-proof.mjs`: the new My Coast brief offline, and journey 6 (a trip saved online reopens at
  $2,602.24 with no server, takes an Update to $2,922.24, and a second offline restart opens the update).
- `scripts/v13-gate-shots.mjs`, `scripts/v13-clips.mjs`, `scripts/v13-idle-probe.mjs`.

Unit tests: 1641 on D's branch against 1624 integrated is exact: B removed `tests/unit/nx/coast/trip.test.ts`
(21 tests) and added 4 to `workspace.test.ts`. No test file was dropped by a merge.
