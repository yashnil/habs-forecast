# Neritic v11: Open Water

A design reset of Today, Explore, My Coast, Research and the shell, built as the long-term direction for the main app. The spec is [design.md](design.md). The evidence is [research.md](research.md) and [ia-audit.md](ia-audit.md). The component and motion APIs are in [kit.md](kit.md).

## The direction

**The ocean is the stage; everything else is staged on it.** The previous build gave every fact the same treatment: a caps label, a hairline, grey 14 px text, a small chart. Nothing led. Now each page has one primary thing, and most of the words sit one tap behind it.

| Page | One primary thing | What changed |
|---|---|---|
| Today | The answer, over the reader's real water | Living map hero; one sentence; four quick reads; chapters rise over the map as a sheet |
| Explore | The field | Ten chrome regions became three zones: top bar, one field dock with the time rail, bare tools |
| My Coast | The trip | One trip line, the status from Today's model, three figures, 60-hour lanes; the trip ledger's live waterfall |
| Research | The record | The chart takes the page; the station ladder is a picker; the crosshair carries the reading |
| Shell | Wayfinding | One trust pill replaces the disclosure strip and the notices cluster, on every page |

The look: chrome has no hue (the accent is ink, not cyan), so colour means data or official status. Labels are sentence case, not tracked capitals. Newsreader runs at real display scale. Mono is kept for coordinates and axis ticks. Sections are separated by space rather than rules.

## Before and after

Real staging data, the same port, 2026-11-09. Full set in [before/](before/) and [after/](after/).

| | Before | After |
|---|---|---|
| Today | [desktop](before/today-desktop.png) · [phone](before/today-phone.png) | [desktop](after/today-desktop.png) · [phone](after/today-phone.png) · [whole page](after/today-desktop-full.png) |
| Explore | [desktop](before/explore-desktop.png) · [with a place](before/explore-place-desktop.png) · [phone](before/explore-phone.png) | [desktop](after/explore-desktop.png) · [with a place](after/explore-place-desktop.png) · [phone](after/explore-place-phone.png) |
| My Coast | [landing](before/coast-desktop.png) · [briefing](before/coast-brief-desktop.png) · [phone](before/coast-brief-phone.png) | [landing](after/coast-desktop.png) · [briefing](after/coast-brief-desktop.png) · [phone](after/coast-brief-phone.png) |
| Research | [desktop](before/research-desktop.png) · [phone](before/research-phone.png) | [desktop](after/research-desktop.png) · [phone](after/research-phone.png) |

Filmstrips: [Today, the sheet rising](after/film-today-rise.png), [Explore, a place opening](after/film-explore-place.png), [Research, draw-in](after/film-research-arrive.png), [Research, a range change](after/film-research-range.png), [Today to Explore](after/film-route.png). The Today, Explore and Research strips were captured on fixture data.

Crowding, first load at 1440×900. Before is from the audit; after is counted from screenshots unless marked. Treat counts as estimates.

| | Before | After |
|---|---|---|
| Today numbers | ~58 | ~15 |
| Today caps labels | ~21 | 0 |
| Explore chrome regions | 10 | 3 zones |
| Explore unobstructed map | ~67% (estimated) | 85.2% (measured) |
| Explore on a phone | ~58% (estimated) | 75.8% (measured) |
| My Coast briefing regions | 13 + 4 on the map | ~6 |
| Research controls | ~33 | ~13 |

## What the research changed

Full notes and sources in [research.md](research.md).

| Finding | Source | Decision it drove |
|---|---|---|
| The legend is the map's title and its readout | MyOcean Light, Global Fishing Watch | Explore's field dock holds kind, date, "not toxin" and the ramp, with a marker that follows the cursor |
| Chrome keeps to at most three fixed zones | MyOcean Light, GFW, savvy navvy | Layer index, legend, rail and provenance merged into one dock |
| A written sentence first; order by what matters now | Apple Weather | Today's lede; flagged fisheries lead, quiet ones collapse to one line |
| Observed vs forecast is carried by position | Apple Weather, GFW, Zoom Earth | One time rail, past left of now; layers grouped by kind instead of badges |
| Colour belongs to the data | Linear, Ventusky | Accent became ink; no cyan chrome |
| Particles only where they are the physics | nullschool, Ventusky | Moving streaks are measured HF-radar currents, nothing else |
| High-frequency actions cut or cross-fade; UI under 300 ms | Emil Kowalski, Vercel guidelines | Layer switch stays 220 ms, a day step cuts |
| Hiding expert functions drew complaints | Windy redesign threads | Every tool stays one tap away |

Gap: nothing we read documents a visual convention for stale data. "(old)" on the value is our own design.

## Motion and interaction

All in `src/v4/motion/` and `src/v4/stage/`, reusable, no new dependency. Reduced motion keeps fades and drops all travel and loops.

- **The stage:** a lazy map hero with a pitched camera, an arrival move once per session, measured currents that pause off screen. Falls back to a poster of real Monterey Canyon isobaths when offline, on save-data or without WebGL.
- **Scroll:** the briefing rises over a pinned map, which dims and scales while the title condenses.
- **Route changes:** View Transitions cross-fade pages while the top bar holds still. The map is rebuilt on arrival, so it is a cross-fade, not a true morph.
- **Details:** `Sheet` (side on desktop, draggable bottom on phones), `Fold`, gliding segmented controls, hover linking from a row to its subject on the map.
- **Charts:** one kit; lines draw in once, marks follow, values never tween. The Research crosshair glides between real samples and cuts on keyboard steps.
- **Time rail:** playback of the frames that exist, each labelled, one frame per 900 ms, stopping on any input.
- **Loading:** nothing for 300 ms, then a flat skeleton. No shimmer.

## Honesty fixes found along the way

The audit found the old pages contradicting each other. Each is fixed and tested.

- Today joined the first wind direction to the period's top speed. It now says what NWS said.
- The advisory's end differed between pages. Both now say what NWS gave, or "no end time given".
- A C-HARM run whose days had passed was charted as an outlook. It is now a "past run" and steps back.
- The outlook read "next 8 days" for a period already under way. It now names its dates.
- My Coast told a commercial reader Neritic held no commercial season while Today led with one. Both now read one model.
- The Monterey shellfish row is "Status unclear" everywhere, awaiting a person's review.
- Today's lede quoted an unreviewed record with no marker. It now carries its own "Not verified", and the trust pill is on every page.
- "None on record" always carries "Not a promise it is open" on the same line.
- "Not toxin" and the model caveats that were hidden on phones are now visible.
- Safari writes dates differently from the server ("Nov 9 at 9:23 PM"), which made React rebuild Today and three older pages on load. Dates with times are now formatted one way, and a test blocks new ones.
- The invented curves behind Today's title and My Coast's hand-drawn coast are gone. Every contour is real geometry.

## Tradeoffs and deferred ideas

- **Not checked on a real iPhone.** All 14 pages and page-to-page navigation were run in Playwright's WebKit (phone and desktop) and Firefox with no page errors or sideways scroll, but that is the Safari engine, not iOS Safari. Firefox has no scroll-linked animation, so its hero stays static as designed.
- **Today to Explore is not a true map morph.** Keeping one live map across routes needs a shared layout; every page still renders its own shell.
- **Explore stays flat and north-up** for reading. The pitched camera is Today's only.
- **My Coast's map is the restyled `HarbourMap`,** without currents or pitch.
- **Streaks over bright chlorophyll are subtle,** even with the new dark casing.
- **`/bloom` and `/fisheries`** got the shell treatment only.
- **Terrain and true 3D** are deferred: there is no elevation source today, only pre-shaded relief.
- **Stale-data styling** is untested with fishermen.
- **The stage adds a real download** on Today (MapLibre and tiles, lazy), and Today's saved copy shows the poster rather than a map.
- **Open question:** Ask's mussel answer still leads with the quarantine and then says sources differ, because its golden tests pin that text.

## Testing

On `253a945c2`, in Chrome on macOS, with fixture data:

- Playwright full suite: 433 passed, 4 failed, 1 skipped. The Mac went to sleep during that run (`ERR_NETWORK_IO_SUSPENDED`). All 4 passed when rerun alone: `m3:157`, `m7:236`, `motion:60`, `offline-recovery:63`.
- Vitest: 1059 passed, 74 skipped, with a 30 s timeout. At the default 5 s, `evidenceCheck` and `tripEvaluation` can time out when the machine is under load.
- Typecheck is clean. ESLint has 0 errors and 12 warnings.
- Earlier full runs on heavier load had up to 18 failures. Each was then either a load timeout that passed alone or a real integration issue that was fixed.

Tests that pinned the old layout were rewritten to assert the new layout and the rule behind it. Honesty and accessibility assertions were not weakened.

Not run: a real iPhone, the full browser suite in WebKit or Firefox (they had a smoke pass of every page), the live Ask evals, cloud Supabase accounts, real SMS, and a full Linux Playwright run on this machine (GitHub CI is the Linux check).
