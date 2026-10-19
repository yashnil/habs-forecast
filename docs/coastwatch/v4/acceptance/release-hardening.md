# v4 release hardening (2026-10-19)

A focused pass: no redesign and no new features. It resolves the scientific notes #66 left open and
the nine low-severity walkthrough issues, and it investigates CI crashes and performance.

## Scientific notes from #66 (commit `ee34b99`)

| Note | Resolution | Held by |
|---|---|---|
| `flag_high` values were ranked and could be "last as high" | A value above the pipeline's plausibility ceiling (kept for review) is never a comparison point. The season's flagged samples are counted and said ("1 flagged for review … is left out"). A flagged newest value is said and not ranked. Production holds none today. | `history.test.ts` |
| Sample dates were the UTC date of the timestamp | Each sample is shown on its Pacific date; 103 of 5,688 production samples move by a day. The original UTC timestamp travels beside it: `StationLite.latest[].at`, the record's `at`, the evidence node's `time.observedAt`, and the lineage "sampled at … (UTC, as published)". The pipeline's `last_date` and its schema are unchanged. | `history.test.ts` |
| Notices merged the agency's dates, Neritic's retrieval and the review state | Each record shows them separately. Took effect (the agency's date). Agency published (each source's own date, named by its release). Neritic transcribed it, and last checked that agency's pages. Reviewed by a person ("No. Transcribed by an AI assistant; not yet checked by a person against the agency's page"). | `hardening.spec.ts` |

## Low-severity walkthrough issues (commit `b76068f`)

| | Fix |
|---|---|
| L1 | Phone search box 48 px tall. |
| L2 | Every link, button and disclosure in the evidence and the notices is at least 44 px tall. |
| L3 | Circles take no part in MapLibre's label collision, so city and station labels were drawn into harbour markers ("San Dieg●San Diego"). An invisible collider now reserves each marker's place. The phone map credit has a backing. |
| L4 | Layer subtitles wrap; they are never cut with an ellipsis. |
| L5 | Each latitude limit is named at its coast end, wrapped and running out to sea, so a side panel never hides it. |
| L6 | Offline on a phone, the banner sits under the "Notices not verified" disclosure, clear of the map controls. |
| L7 | Any date outside the current year carries its year. The field pack's dates use one format. |
| L8 | Choosing a layer from the peeking phone sheet opens the sheet to half, so its time row is in view. |
| L9 | Offline, the service worker answers from the pack first. Online it stays network-first. |

Each is held by `tests/e2e/hardening.spec.ts`; every one of those tests fails on the previous build.

One edge case remains. At 320 px, when the coast end of a limit line is at the bottom of the visible
map, the map credit's backing can cover the limit's label until the map is panned.

## Accessibility

`tests/e2e/a11y.spec.ts` runs axe-core (WCAG 2.1 A and AA) on these views:
- Explore with a harbour open;
- the notices panel;
- Research with its tools;
- the briefing.

Each runs at desktop and phone width, and none has a serious or critical violation. Previously only
the briefing and one widget were checked.

## Browser "crashes" and performance

**The "crashes" were timeouts.** "Target page, context or browser has been closed" is what Playwright
reports when a test reaches its 45 s limit while waiting for the map. These tests normally take
13–17 s.

**Main's CI timeout (`m5.spec`).** The trace showed:
- each map call taking 5–8 s, so the main thread was saturated on the runner;
- tile requests completing in under 240 ms.

The network was not the cause.

**The GPU readback hypothesis was tested and rejected.** Making the tile-clipping canvases CPU-backed
(`willReadFrequently`) changed nothing: Linux long tasks were 7.3 s without it and 7.5 s with it. The
change was reverted.

**v2 and v4 compared fairly.** Both ran in Linux (software WebGL), with 4× CPU throttling, 5 runs each
and the same camera (Monterey Bay, z 9.2–9.3):

| View | Map loaded, v2 | Map loaded, v4 | Main-thread long tasks, v2 | v4 |
|---|---|---|---|---|
| Forecast | 4.5 s | 5.1 s | 4.6 s | 4.0 s |
| Satellite, both sensors | 5.4 s | 7.0 s | 4.8 s | 5.4 s |

The forecast view is on par. The multi-sensor satellite view is a modest real regression: about
1.5 s later to load and 12 % more main-thread work in this harsh setting. Not yet fixed. An earlier
"+60 %" figure compared different zoom levels and is withdrawn.
