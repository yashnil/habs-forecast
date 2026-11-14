# Independent critic review (rounds 1 and 2)

Written by an independent reviewer against the running app (fixture data). Screenshot paths below refer to the reviewer's local working folder and are not committed. Round-2 items were fixed afterwards in 976fa42a9 and ae98255af.

# Neritic v12 "Chart": independent critic review

Date: 2026-11-11. Branch `feat/neritic-v12` at `ac5246534`. I built a read-only copy of `coastwatch-web/` in
`critic/app` (the worktree's `.next` was older than its last commits). I ran it on port 4700 with the e2e
fixture environment (`CW_DATA_BASE_URL=/data/fixture/v1`, `CW_FIXED_NOW=2026-10-13T06:00:00Z`, recorded
conditions) and drove it with installed Chrome through Playwright.

Coverage: 67 stills (7 routes × 1440×900, 390×844, 320×640 and 844×390 in Day; Day, Dusk and Night at 1440 and
390; first visit, no-port and reduced-motion variants). I also walked all three journeys plus My Coast end to
end, ran axe (WCAG 2 A/AA, 30 page/mode/size combinations), a font-size floor scan, CLS on phone, a
persistent-canvas check across routes, and `lint:design`.

Screenshots: `critic/shots/` (journeys in `shots/j1`, `shots/j3`, `shots/coast`). Scripts: `critic/*.mjs`.

R = reproduced in the running app. S = suspected (from code or data, or only partly reproduced).

## What is good (so the fixes don't break it)
- It does not look AI-generated. The type pairing (Literata statements with Public Sans), paper and ink, ruled
  margins, no cards, no shadows, no pills, and real graduated neatlines all hold up. `lint:design` is clean
  (166 files).
- Today at desktop passes the 5-second test: a magenta "Season delayed to at least Jan 1, 2027", then the next
  date, then two soundings at their places.
- Returning to `/` shows Today with no flash: the margin text is there at 150 ms. The map is one persistent
  canvas across Today, Explore, My Coast and Research.
- axe found 0 violations in any mode or size. CLS is 0.000 on phone for /research, /today, /explore and /coast.
- The Journey 3 export is excellent: provenance header, licence, citation, "not measured is never zero", and the
  window from the URL.
- The worksheet gets its states right: "waiting for …" everywhere, nothing pre-filled, "who pays fuel" is
  required, and the arithmetic checks out by hand (crew $1,126, net $1,689).

## Findings, most severe first

### 1. [Broken · R] The port finder ignores what you type, so Enter opens the wrong port
- **Where:** `/` first visit at every size; also the shell finder (same component).
- **What happens:** typing "moss" leaves 7 ports in north-to-south order with **Shelter Cove** selected, and
  Enter opens Shelter Cove. "eureka" selects Crescent City, "zzz" selects Santa Cruz, and the "No port
  matches" empty state never appears. The cause: cmdk fuzzy-matches against
  `value={`${group} ${p.name} ${p.county} ${p.code}`}`, so the region name and scattered letters match, and
  the results stay unranked inside fixed groups.
- **Why it matters:** "type moss" is step 2 of Journey 1. A stranger who types their port and presses Enter
  lands on another port's Today: wrong season, wrong tests. The e2e test hides this because it clicks the
  option by name (`tests/e2e/journeys/first-visit.spec.ts:57-60`).
- **Evidence:** `shots/j1/1440x900-finder-moss.png`, `shots/j1/1440x900-after-port.png` ("What do you fish
  from here?" under Shelter Cove), `src/nx/ds/PortFinder.tsx:46`.
- **Fix:** set `value={p.name}` and `keywords={[p.county]}`, with no group or code. Use a custom `filter`
  (prefix or substring on the name, then the county). While a query is present, show one ranked flat list with
  no region groups. Add a test that types "moss", presses Enter and expects Moss Landing, plus one for the
  empty state.

### 2. [Truth T42 · R] Today's Whale row gives commercial crabbers the recreational rule
- **Where:** `/today` at Moss Landing, Dungeness commercial; all sizes.
- **What it says:** the row reads **"Zone 4, traps prohibited"**. Expanded, it says "Commercial season delayed
  … Recreational crab traps prohibited. Fleet Advisory." My Coast's gate for the same reader says "Fleet
  Advisory. No depth limit or gear reduction on record."
- **Why it matters:** this is a wrong one-line regulatory summary for the main audience, and two pages
  contradict each other. It is exactly IA risk 4 in T42.
- **Evidence:** `src/nx/today/model.ts:398` (a regex over the whole `w.action` string picks "traps
  prohibited" from the recreational clause); the rows text is in the transcript of `rows.mjs`;
  `shots/today-row-Whale.png`.
- **Fix:** build the commercial summary from structured fields (`depth_limit_fathoms`, `gear_reduction_pct`,
  advisory) and drop clauses that name "recreational". Share one formatter between Today and My Coast so they
  can't disagree.

### 3. [Truth T8, possibly T7 · R] The rock crab test shows the average and hides the 20 ppm maximum
- **Where:** `/today`, the Tests row expanded.
- **What it says:** "Monterey Bay rock crab tested under the limit on Sep 28: avg 3.8 ppm, limit 30." The
  fixture has `max_ppm: 20.0`, and `values_raw` is `["<2.5","ND","ND","20","<2.5","2.6"]`. The 3.8 average is
  (20 + 2.6) / 6, so ND and <2.5 count as 0 (S: that is a pipeline choice, but the UI prints it as is).
- **Same problem on phone:** the 390 Today sounding drops "highest of 6" and the age. It reads "9.8 of 30 ppm,
  Dungeness test, CDPH, Sep 28", where desktop says "highest of 6 … 14 days ago".
- **Why it matters:** a crabber reads 3.8 against 30 as "fine". One crab at 20 is two-thirds of the action
  level.
- **Evidence:** `shots/today-row-Tests.png`, `public/data/fixture/v1/biotoxin/index.json` site
  `bma-g-monterey-bay-rock-crab`, `shots/p-day_.png`.
- **Fix:** always print the highest value first ("highest 20, average 3.8; 4 of 6 below detection"). Keep
  "highest of n" and the age in the phone sounding; drop the leader text before the qualifier.

### 4. [Truth T3 · R] Explore calls a past outlook an "outlook"
- **Where:** `/explore`, Outlook layer.
- **What it says:** the tab reads "Outlook / outlook" and the legend "Mean of Oct 4–11, issued Oct 8", on
  Oct 12. Every day it covers has passed. Today's margin already says "Algae outlook past its dates", and
  C-HARM on the same tab row says "past run". Explore treats the outlook as current.
- **Evidence:** `shots/j3/1440x900-layer-outlook.png`.
- **Fix:** give the outlook the same past-run treatment as C-HARM ("past run (Oct 8)", drawn lighter, a
  subtitle that isn't the word repeated).

### 5. [Truth T17/T25/T27 · R for the display, S for the stale limit] Stale currents stream over chlorophyll by default
- **Where:** `/explore`, default state at every size.
- **What happens:** Currents are "on, measured" by default. The streaks animate over the chlorophyll field,
  and the frame is **Oct 8, 5:00 AM PT**, about 4.7 days before the fixed now. Your freshness table puts HF
  radar at 1/3 days, yet no "(old)" appears. The chlorophyll readout, by contrast, does say "Oct 6 (old)".
- **Why it matters:** T25 says no streaks on a chlorophyll layer. Moving lines over a bloom picture read as
  "the bloom drifts this way" (T27). A value past its stale limit should leave the headline.
- **Evidence:** `shots/d-day_explore.png`, `shots/d-dusk_explore.png`, `shots/j3/1440x900-readout.png` ("Nearest
  measured current here: 0.33 kt … Oct 8").
- **Fix:** default Currents to off when the base layer is chlorophyll or a model, or draw them as a static,
  separate layer. Compute currents' freshness from `freshness.*` and show "(old)", or drop them past the stale
  limit.

### 6. [Truth T10 · R] Research shows C-HARM at 75 % without its mandatory words
- **Where:** `/research`, the Model strip.
- **What it says:** "Model · C-HARM P(pDA > 500 ng/L) … 75 %". "Probability of conditions in the water, not
  toxin in seafood" and the no-skill note exist only inside the collapsed "Model" row of the margin.
- **Evidence:** `shots/d-day_research.png`; the `rtext.mjs` output shows has-words false until the row is
  expanded.
- **Fix:** put the T10 sentence in the strip's caption, always visible, as Explore already does.

### 7. [Spec principle 1, visual · R] Research lays a white panel over the chart, and the neatline then lies
- **Where:** `/research` at 1440 and 844×390.
- **What happens:**
  - The series panel covers about 60 % of the sheet. The neatline's longitude labels carry on across the top of
    the charts ("121°20′W … 120°20′W" above the domoic acid strip), which is graduation for map that isn't
    there.
  - During the route change, the scale bar ("20 km. Not for na…") and "Monte…" are cut by the panel.
  - On the map, "Santa Cruz" collides with "Santa Cruz Wharf" and "Monterey" crowds "Monterey Wharf".
- **Why it matters:** this is the "map behind a panel" pattern the spec bans, and false ticks are a truth
  problem on a chart.
- **Evidence:** `shots/d-day_research.png`, `shots/j3/1440x900-research.png`, `shots/nav-Research-350.png`.
- **Fix:** on Research, make the neatline frame only the map pane, or lay the page out as two sheets: a small
  map, and a series page with its own frame. Suppress the port and town labels that duplicate station labels.

### 8. [Spec, 5-second · R] On half the pages the chart is wallpaper, and regulation never reaches it
- **Where:** `/coast`, `/notices`, every phone page.
- **What happens:** My Coast shows an empty bay with one buoy glyph while the reader scrolls about 4,000 px of
  margin. `/notices` says "Pigeon Point (37°11′N) to Point Lobos (36°31′N)", but the chart draws nothing. RAMP
  Zone 4 is named in three places and never drawn. **Magenta, "regulation", appears only as text and a
  decorative tick rule.** On phone, every page is the same top-half bay with a sheet below, so the pages look
  identical.
- **Why it matters:** "The chart is the product" and T43 ("zones are latitude bands drawn as lines") are the
  core of the Chart direction the owner chose. Right now it reads as a nice basemap beside a text app.
- **Evidence:** `shots/d-day_notices.png`, `shots/d-day_coast.png`, `shots/p-day_notices.png`,
  `shots/p-day_coast.png`.
- **Fix:** draw the RAMP zone latitude lines (magenta, labelled at the neatline) on Today, Notices and My
  Coast; draw each notice's latitude extent as a magenta bracket on the neatline; and on My Coast draw the
  NWS zone, the buoys and the tide station that feed the sea window. Then the map earns its 70 % of the screen.

### 9. [Spec consistency, T2 · R] The same mark means two things on one page
- **Today agenda:** the dots are solid for past or now and hollow for future (from Tide). So the **NWS
  forecast** "Oct 12 … NWS tonight" gets a solid dot, while the page legend says solid = measured and hollow =
  model or prediction.
- **Worksheet:** the waterfall hatches the Fuel and Crew pay bars, and hatch is the model mark.
- **Evidence:** `shots/d-day_today.png` (agenda), `shots/coast/1440x900-filled-b.png`.
- **Fix:** show time in the agenda by position against the now line, not by fill. Keep fill for kind. Use
  solid or outline for cost bars, never hatch.

### 10. [Spec consistency · R] Dusk and Night flip the chlorophyll ramp
- **Where:** `/explore` in Dusk and Night.
- **What happens:** Day runs low = cream to high = dark teal. Dusk and Night run low = dark to high = pale. The
  spec says "never inverted" and gives `--chl` "high is dark". Someone switching modes at sunset sees the
  bay's high values change from darkest to brightest.
- **Evidence:** `shots/d-day_explore.png` vs `shots/d-dusk_explore.png`, `shots/d-night_explore.png`.
- **Fix:** decide. Either keep the direction and recompute the steps, or record the inversion as an intended
  rule in the README and say "brighter = more" in the legend.

### 11. [Truth T19, visual · R] Explore draws hard lines at the data's edge, with blank water beyond
- **Where:** `/explore` with no saved port, and with reduced motion.
- **What happens:** horizontal rules cross the whole chart at about 37°12′N and 36°23′N. Water outside the
  product footprint is plain blue with no "no view" hatch, so it reads as data.
- **Evidence:** `shots/d-noport_explore.png`, `shots/crop-reduced-explore-top.png`.
- **Fix:** apply the "stop at the data's edge" camera clamp when there is no port too, and hatch everything
  outside the footprint.

### 12. [Layout at small sizes · R]
- **390, Explore:** the locator inset covers the "Monterey" label (`shots/p-day_explore.png`).
- **320, Today:** the crab sounding is dropped entirely; only the port symbol is left, though the test is
  Today's main fact (`shots/s-day_.png`).
- **844×390 first visit:** the primary "Find your port" button is cut off at the fold
  (`shots/l-first_.png`).
- **320, `/system`:** the page overflows (scrollWidth 346). The `sensTable` and a sounding block run past the
  edge. The same worksheet table may overflow `/coast` at 320 with long figures (S).
- **Phone sounding unit:** "of 30 ppm" is 12.9 px, below the 14 px floor.

### 13. [Worksheet comprehension · R] The sensitivity grid labels its columns "Catch", but they are price
- The corner label "Catch" sits over the −20 %, as-entered and +20 % columns, but those are **price** (I checked:
  $1,268 = (0.8 × $3,510 − $694) × 0.6). Nothing labels the price axis. The waterfall bars are about 70 px wide
  and $0 fees is a hairline.
- **Evidence:** `shots/coast/1440x900-filled-b.png`.
- **Fix:** put "Price →" over the columns and "Catch ↓" on the rows, or say the same in the corner.

### 14. [5-second comprehension · R] Explore's margin is an instrument panel before you get to the water
- **What's there:** the legend, a 5-line streak legend, 5 tabs (one reads "Outlook / outlook"), and a time rail
  with three bar rows (OLCI, VIIRS, Currents by the hour). Then "Bars: share of the coastal ocean with a value
  that day. This frame: values over 25 %." Then an "Also" line. A fisherman sees satellite names, not
  meaning.
- **Fix:**
  - Lead with one sentence ("Algae seen from space, Oct 1–6, latest clear view per pixel. Tap the water.").
  - Hide sensor names behind "Sources".
  - Move the time rail under the chart, as on GFW.

### 15. [Research legibility · R] Strip headers are cut off
- "ng/mL in seawater; dashes: C-HAR…" and "cells/L, larger cells; dash…" are clipped, so the explanation of
  the dashed threshold line is lost.
- **Where:** 1440 and phone, `/research`.
- **Also:** at "All", the brush frame is a sliver; the 90-day window is about 5 px wide on a 12-year strip.
- **Fix:** wrap the unit line to two lines. Give the brush a minimum handle width, or use a 12-month overview
  when the window is small.

### 16. [Visual framing · R] Default views waste the sheet on land
- No-port Today and My Coast frame the whole western US, with Nevada and Arizona taking about 60 % of the
  chart. Explore is about 45 % land at desktop and about 60 % in landscape. The Research map is mostly
  hillshade.
- **Evidence:** `shots/d-noport_today.png`, `shots/d-noport_coast.png`, `shots/l-day_explore.png`.
- **Fix:** clamp every camera to the ocean plus a thin coastal strip, the way a coastal chart sheet is cut.

### 17. [Night mode · R] Night's primary button is a bright tan block
- **Evidence:** `shots/d-night_today.png`.
- **Why it matters:** a large luminous block is what the Night table exists to avoid.
- **Fix:** in Night, draw the primary as an outline in `--ink`.

### 18. [Accessibility, minor · R]
- The sheet's More/Less button has no `aria-expanded`; detent state is only in `data-detent`.
- `/system` opens scrolled to y = 11,965 because the PortFinder demo autofocuses.
- The Water row mixes units: "63 °F" collapsed, "17.2 °C" expanded.

### 19. [Review hygiene · R] The fixture clock makes the owner's screenshots wrong
- The fixed now is Oct 12. Yet stamps say "checked Nov 8", the offline note says "Saved … Nov 11", the diesel
  reference says "week of Oct 19", and the agenda calls the Oct 30 delay "Coming" while the record was checked
  Nov 8.
- **Fix:** review the owner preview on staging data with the real clock, or re-date the pilot sidecars to
  match `CW_FIXED_NOW`.

### 20. [Attribution T42 · S] Groundfish "no restriction on record" is stamped "Official · CDPH"
- **Where:** Today's Others row.
- **Why it matters:** CDFW acts on fisheries, so this looks like the wrong agency.
- **To check:** the registry entry.

## Motion
- The opening (dithered canyon) shows the statement at frame one, has a Skip, and is gone by about 2.5 s.
- Reduced motion settles to final states, and Explore draws static streamlines.
- Route changes keep one canvas and pan.
- In the Research transition, the strips draw from a collapsed domain while the map is still panning, so
  for about 350 ms the strips are empty frames (`shots/nav-Research-350.png`). That is acceptable, but it
  exposes the panel-over-map problem (finding 7).
- No stagger, count-up or ambient effects found.

## Verdict
**Ship it to the owner as a preview after fixing 1–3** (a few hours of work). The craft is real; this is the
first version that does not read as generated. But a broken port search on the front door, and two
wrong-or-misleading fishery facts on Today, would undercut the "truth beats drama" pitch in the first minute
of review.

**The three changes that would most improve it**
1. **Fix the port finder** (finding 1), and add a typed-Enter journey test.
2. **Put regulation on the chart.** Draw RAMP zone lines and notice extents in magenta on Today, Notices and My
   Coast, and give the Research map its own frame. This turns Chart from a basemap into the product (findings 7,
   8, 16).
3. **One truth sweep over Today and Explore copy:** commercial vs recreational, max before average, past-run
   labelling for every model, stale currents, and T10 on Research (findings 2–6, 9).

---

# Round 2: `feat/neritic-v12` at `b2fca0f18`

## Method

- **Build:** I rebuilt the read-only copy (`critic/app`) from `b2fca0f18` and ran `npm run data:fixture`. That
  script reads `../pipeline/tests/fixtures/compat/m2`, so I symlinked `critic/pipeline` to the worktree's
  `pipeline/`. Read dates are now held to the fixture clock: 7 × `checked_at: 2026-10-12`.
- **Round 1 data was stale.** My round 1 `data:fixture` had failed at that step without my noticing, so round 1
  ran on stale public data. That is the source of "checked Nov 8" in #19, and it also hid the outlook state
  (see N5).
- **Server:** `next start` on port 4700, same env as round 1.
- **Scripts:** the same scripts, same sizes and modes, all 67 stills, plus the journeys, the worksheet, axe and
  CLS. Round 2 stills are in `critic/r2/shots/`. Round 1 stills stay in `critic/shots/`.

## Original findings

| # | Status | Evidence |
|---|---|---|
| 1 Port finder | **Fixed** (R) | "moss" → [Moss Landing] and Enter opens it; "eureka" → Eureka; "zzz" shows the worded empty state and Enter does nothing (`r2/shots/j1/finder-zzz.png`); "monterey" → Monterey, Moss Landing; "half moon" and the typo "mos landing" both work. Nit: "ft bragg" finds nothing (no abbreviations). |
| 2 Whale row recreational rule | **Fixed** (R) | The row now quotes "CDFW, Zone 4, for commercial crabbers: 'Commercial season delayed …; Fleet Advisory.' No depth limit or gear reduction on record", which matches My Coast. |
| 3 Max hidden by average | **Partly** (R) | Tests row: "highest 20 ppm of 30, all 6 samples under, 4 below detection". Fixed. The phone sounding now has "Highest of 6, Sep 28, 14 days ago", but it lost the species and the source (see N4). |
| 4 Past outlook called "outlook" | **Fixed in code, not reproducible** | `src/nx/explore/model.ts:226-229` gives the outlook the same past-run rule as C-HARM, which is now labelled "past run (Oct 8)". The pinned fixture's outlook is current (Oct 13–20, issued Oct 12), so I could not show the past state. |
| 5 Stale currents over chlorophyll | **Fixed** (R) | Currents are "off" by default on every size. I did not re-test the "(old)" label with Currents switched on. |
| 6 T10 hidden on Research | **Fixed** (R) | "C-HARM: probability of conditions in the water, not toxin in seafood. No demonstrated skill at the shore…" is always visible. But see N3: its strip is now pushed out of view. |
| 7 Research panel over chart | **Fixed** (R) | The map has its own framed pane and neatline, the false ticks are gone, and the duplicate "Santa Cruz" label is gone (`r2/shots/d-day_research.png`). Nit: on the Santa Cruz Wharf view the Moss Landing symbol is cut by the frame (`r2/shots/research-link.png`). |
| 8 Chart as wallpaper | **Mostly fixed** (R) | My Coast frames the whole zone with the Pigeon Point and Lopez Point lines labelled; Notices draws the 37°11′–36°31.46′ band; Today with no port draws zones 1–5 statewide; Today with a port shows "Whale zone 4". The phone versions lose their labels (N1). |
| 9 Mark clashes | **Fixed** (R) | The agenda shows the NWS item with a hatched square; the waterfall costs are solid (`r2/shots/d-day_.png`, `r2/shots/coast/1440x900-filled-b.png`). |
| 10 Dusk/Night ramp | **Accepted as decided** | Every legend says "Darker is more." or "Lighter is more (dark chart)." (`r2/shots/d-dusk_explore.png`). Reasonable on the merits; I don't re-flag it. |
| 11 Data-edge lines | **Fixed** (R) | No horizontal rules, and the hatch runs past the footprint (`r2/shots/d-noport_explore.png`, `r2/shots/d-reduced-first_explore.png`). |
| 12 Small screens | **Partly** (R) | Fixed: the 390 Explore inset moved top-right and no longer covers Monterey; the 844×390 call to action is above the fold; no text under 14 px; `/system` no longer scrolls sideways at 320. **Not fixed: 320 Today still drops the crab test.** `[data-nx-sounding=test]` stays `visibility:hidden` at 3, 6 and 10 s, at 1× and 2× (`r2/s320.mjs`, `r2/shots/s-day_.png`). |
| 13 Grid axes | **Fixed** (R) | "Price →" over the columns, "Catch ↓" on the rows, wider bars; the numbers are unchanged and correct. |
| 14 Explore margin density | **Fixed** (R) | It now opens on a plain first line ("Algae seen from space, Oct 1–6 … Not toxin."), with one rail row and "Sources and coverage". Nit: the C-HARM tab subtitle wraps to 3 lines ("past run (Oct / 8)") at 1440. |
| 15 Truncated headers, brush | **Fixed** (R), with a side effect | Headers wrap and the brush frame is visible. But the strips are now taller than their box (N3). |
| 16 Framing on land | **Fixed** (R) | My Coast, Notices and no-port Today frame water (`r2/shots/grid-r2.png`). |
| 17 Night primary button | **Fixed** (R) | It is an outline in ink (`r2/shots/d-night_today.png`). |
| 18 Accessibility minors | **Mostly fixed** (R) | `/system` opens at scrollY 0; the Water row reads "63.0 °F (17.2 °C)"; the More button has `aria-expanded`. Nit: it still says `false` after More moves the sheet to the middle detent. axe: 0 violations in 30 runs; CLS 0.000 on four routes. |
| 19 Fixture clock | **Fixed** (by `data:fixture`) | Pages say "checked Oct 12" and "Saved … Oct 12". Leftover: "Recreational season open since Nov 7" reads as current on an Oct 12 page, because agency dates are not held. This affects the fixture only. |
| 20 Groundfish attribution | **Fixed** (R) | "Official · CDFW". |

Journeys: Journey 1 works by typing and by Enter at 1440 and 390, with the tour present; the returning visit
works; Journey 3 goes Explore tap → "See the record nearby: Santa Cruz Wharf, 16 km" → Research with the URL
state → CSV export (`neritic-HABs-SantaCruzWharf-2026-08-07-2026-10-06.csv`). The map stays one canvas across
routes.

## New problems from the fixes (most severe first)

### N1. [Truth T38/T39, regulation on the chart · R] On phone, the magenta shapes lose their labels
- **Notices, 390 and 320:** the band is drawn, but its label "2 anchovy notices · not verified" is
  `visibility:hidden`. A magenta regulation shape is left with no name and without "not verified". On 1440 the
  label shows. Evidence: `r2/shots/p-day_notices.png`, `r2/shots/s-day_notices.png`, the `r2/nb.mjs` output.
- **My Coast, 390 and 320:** the Pigeon Point 37°11′ line is drawn unlabelled. At 320, "Whale zone 4" is hidden
  too, and "Lopez Point 36°00′ N" crowds the scale bar. Evidence: `r2/shots/coast/320x640-filled-b.png`.
- **Fix:** a regulation line or band may only be drawn with its label. If the label can't be placed on the
  water, set it on the neatline margin at its latitude. Ports and readings can yield to it there.
  - On phone, the label could shorten to "2 notices · not verified" and the latitude alone.

### N2. [Small screens, finding 12 regressed · R] 320 Today still hides the crab test
- The `insist` placement from `097ef3258` does not survive `b2fca0f18`: at 320 the "Whale zone 4 / Pigeon Point
  to Lopez Point" label now takes the water where the test figure would go, and the test stays hidden.
- That breaks the commit's own rule that lettering "yields to readings".
- **Evidence:** `r2/shots/s-day_.png`, `r2/s320.mjs` (test hidden, whale label visible).
- **Fix:** place the zone label after the readings and drop it before them. At 320 the zone name can live only
  in the margin.

### N3. [Research · R] At 1440×900 the C-HARM strip falls out of sight in a nested scroller
- **Where:** the default station (Monterey Wharf, which has 5 measured strips).
- **What happens:** wrapping the headers made the strips taller than their box (`stripsBox` scrollHeight 603
  vs clientHeight 502). The **Model · C-HARM strip** is now below the fold inside an inner scroll area with no
  visible scrollbar on macOS. Its T10 caption still shows under the water temperature strip, captioning a
  strip you can't see.
- **Evidence:** `r2/shots/d-day_research.png`, compared with `r2/shots/research-link.png` (the Santa Cruz
  Wharf station has 4 strips, so it fits).
- **Fix:** shorten each strip by about 20 px, or let the page scroll instead of a nested box. Keep the caption
  attached to its strip.

### N4. [Truth T1/T6 · R] The phone crab figure no longer says what it is
- **What it says at 390:** "9.8 of 30 ppm · Highest of 6, Sep 28, 14 days ago". The species ("Dungeness") and
  the source ("CDPH") are gone, so it isn't named as a CDPH lab result for a species. Desktop is fine.
- **Evidence:** `r2/shots/p-day_.png`.
- **Fix:** "Dungeness, highest of 6 · CDPH, Sep 28 (14 days)". Wrap to 3 lines rather than dropping words.

### N5. [Truth T15, existing code shown by the corrected fixture · R] The outlook is quoted to two figures
- **Where:** the Today Models row.
- **What it says:** "expected to run higher than usual for this time of year, about 1.6× the typical level".
  The same block says the typical error is about a factor of 2, and T15 asks for one significant figure for
  the outlook.
- **Code:** `src/pilot/facts.ts:902` (`ratio.toFixed(1)`).
- **Note:** this is older shared code. I missed it in round 1 because of the stale data.
- **Fix:** "about 2× the usual level", or just "higher than usual".

### N6. [Notice band shape, T43 · R] The notice band draws a western edge that doesn't exist
- **What it looks like:** the band is a closed box with a vertical western side inset from the frame. That side
  sits at about 122°53′W on desktop and about 122°25′W on phone, so it is plainly arbitrary. It reads as an area
  with an offshore limit.
- **Evidence:** `r2/shots/d-day_notices.png`, `r2/shots/p-day_notices.png`.
- **Fix:** draw the two latitude lines plus a bracket on the latitude scale of the neatline. Drop the vertical
  side in the water.

### N7. [Clarity · R] "Whale zone 4" floats mid-water on Today
- The whole view is inside Zone 4, so no line is visible. The name is anchored to nothing
  (`r2/shots/d-day_.png`).
- **Fix:** set it at the neatline edge, or say "whole view: Whale zone 4".

### N8. [Minor · R]
- The Explore readout's transect axis overprints "here" and "12 km E" when you tap near the coast
  (`r2/shots/j3/1440x900-readout.png`).
- The C-HARM tab subtitle wraps to 3 lines at 1440.
- `aria-expanded` stays `false` at the middle sheet detent.
- On the Santa Cruz Wharf Research view, the Moss Landing symbol is cut by the frame.

No new regressions in axe (0), CLS (0), first-load flash, Night or Dusk legends, the port finder's ranking or
its empty state.

## Verdict, round 2

**Not yet ready for the protected preview. It's close: four small fixes first.**

All twenty round 1 findings are fixed or decided except the 320 crab test. The desktop experience is ready.
What blocks it is phone-only truth and clarity on the new chart work:

1. **N1:** never draw a regulation line or band without its label. On Notices that label must include "not
   verified".
2. **N2 + N4:** at 390 and 320, the crab test must always be shown and named: species, CDPH, date and age.
3. **N3:** keep the C-HARM strip visible on Research at 1440×900, next to its caption.
4. **N5:** round the outlook ratio to one figure (T15).

N6 and N7 are worth doing in the same pass, but they don't block. With N1–N5 done, I would say yes.
