# Neritic v11: information-architecture audit

Read-only audit of `feat/neritic-pilot` (`coastwatch-web/`), from the code and the screenshots in `scratchpad/before/`. Paths are relative to `coastwatch-web/src/`. Counts are what is drawn at first load, counted from the screenshots, accurate to within a few items.

## 1. Inventory at first load

| Surface | Viewport | Regions | Text lines | Caps labels | Chips | Numbers | Caveats | Links | Controls |
|---|---|---|---|---|---|---|---|---|---|
| Shell | 1440 | 3–4 | 2 | 0 | 2 | 1 | 1 | 8 | 9 |
| Shell | 390 | 3 | 2 | 0 | 2 | 1 | 1 | 6 | 8 |
| Today | 1440 | 7 | ~40 | ~21 | 6 + 3 marks | ~58 | 1 | ~14 | 10 |
| Today | 390 | 5 | ~22 | 7 | 4 | ~14 | 1 | 6 | 9 |
| Explore, no place | 1440 | 10 | ~32 | 3 | 2 | ~35 | 4 | 9 | 21 |
| Explore, place open | 1440 | 11 | ~45 | 5 | 5 | ~40 | 3 | 13 | 27 |
| Explore | 390 | 7 | 7 | 0 | 2 | 6 | 1 | 6 | 10 |
| My Coast landing | 1440 | 6 | ~40 | 10 | 1 | ~20 | 1 | ~15 | 5 |
| My Coast landing | 390 | 5 | ~17 | 4 | 1 | 6 | 1 | 7 | 5 |
| My Coast briefing | 1440 | 13 + 4 on the map | ~45 | 12 | 5 | ~40 | 5–6 | ~18 | 12 |
| My Coast briefing | 390 | 8 | ~24 | 7 | 3 | ~12 | 1 | 5 | 8 |
| Research | 1440 | 12 | ~60 | 8 | 2 | ~70 | 6 | 8 | ~33 |
| Research | 390 | 10 | ~22 | 4 | 2 | ~25 | 1 | 4 | 14 |

## 2. Shell

**Diagnosis.**
- `v4/shell/TopBar.tsx:182–217` draws two plates plus a slot. The right plate holds the notices pill ("10 in CA | Not verified"), the status chip, the bell and the more menu.
- `v4/shell/Disclosure.tsx:11–48` repeats "Notices not verified" as a separate line with CDFW and CDPH links. "Not verified" is on screen twice before any content.
- On a phone the shell is three bands (bar, line, dock), about 150 of 844 px.
- Today is inconsistent with the rest. `pilot/today/TodayView.tsx:211` mounts `TopBar` with no `notices` and no `RegistryDisclosure`, so it has neither the pill nor the line, only a 13.5 px link (line 290).
- The daylight toggle exists twice: `TopBar.tsx:81–99` and the hero's `SunToggle` (`TodayView.tsx:77–95`).

**Proposal.**
- (a) Primary: wayfinding.
- (b) Visible (12 words): brand, four tabs, and one trust pill, "3 here · not verified", with the stale clock when any source is old.
- (c) Behind the pill's popover: the full disclosure, agency links, the status centre and "why not verified".
- (d) Merge the disclosure line into the pill on every page. Delete the hero `SunToggle`.
- (e) The words "not verified" stay as text in the pill on every page, Today and phones included. An icon alone is not enough.

## 3. Today

**Diagnosis** (`today-desktop.png`).
- **Two heroes.** The left block (`TodayView.tsx:232–293`) holds eyebrow, port, a 3-line lede, a sea line, five freshness items, three buttons and a link. The `Glance` aside (`:98–127`) adds six rows, each restating something within 400 px: the tally, the sea line verbatim, the tide, the test, C-HARM, and the whale rule.
- **Equal-weight rows.** `Status.tsx:62–85` gives all seven fisheries the same row: mark, label, NOT VERIFIED chip, status chip, sentence, source line. Three say "none on record". The strip is about 640 px tall for one sentence of news.
- **Always-open prose**, about 330 words: the lanes caption (`Sea.tsx:124`), the buoy sentence (`:204–226`), the tide caveat (`:249–261`), the dot key and source (`Tests.tsx:58–78`, `:101–111`), a 54-word C-HARM paragraph (`Bloom.tsx:85–91`), a 62-word outlook paragraph (`:123–127`), and the first-visit paragraph (`Changed.tsx:36–40`).
- **Repeats.** The Dungeness delay is said four times on the first screen and six on the page. "N 5–15 kt · seas 6–9 ft" is said three times. The next tide appears four times. "Was this useful?" appears five times (`ui.tsx:98–102`). Ask appears twice.
- **The grid interleaves two threads.** `today.css:59` pairs status with sea, tests with bloom, whale with changed, so the evidence ladder zigzags across columns.
- **Phone.** The hero and buttons take 58% of the first screen. The first status row starts at y≈565.

**Proposal.**
- (a) Primary: the lede, the top official fact for the reader's fisheries.
- (b) Visible (about 70 words on desktop, 45 on a phone):
  - port and stamp;
  - the lede with its own cue, "CDFW · checked Nov 8 · not verified";
  - a ribbon of seven status marks with a one-line tally;
  - one sea line with a wind and seas strip for today and tonight;
  - a freshness cue listing only what is stale.
- (c) Disclosure:
  - a status row opens the existing `StatusSheet`;
  - "none on record" rows collapse in place to one line;
  - the 5-day lanes, buoy detail and tide list expand in place;
  - chart keys and "what is C-HARM" move to an info popover;
  - Change port moves into the title; Ask becomes one fixed button.
- (d) Delete `Glance`, the hero `SunToggle`, the first-visit paragraph and four of the five feedback prompts. Merge `WhaleRules` into the crab rows' sheet, with a one-line RAMP row in the status section. Merge the Actions links into the footer.
- (e) Must stay:
  - the status word and shape per row;
  - "(old)" beside the stale value itself;
  - "probability of conditions in the water, not toxin in seafood" on the C-HARM line;
  - "chlorophyll, not toxin" in the outlook heading;
  - the caveat on the collapsed line: "None on record: rock crab, groundfish, squid. Not a promise they are open."

**Is one "Not verified" per group honest?** Yes, under four conditions:
1. The marker is in the section header that stays with the rows.
2. It is computed. With mixed records the header says "5 of 7 not verified" and the exceptions are marked per row.
3. Anything that quotes a record outside the group (the lede, an alert, an Ask answer) carries its own marker.
4. The row's sheet still states that record's review state.

Per-row chips today are less honest than they look: the no-record rows have `verified: null` and no chip (`pilot/facts.ts:535`, `ui.tsx:63`), so by contrast they read as verified.

**Section order, glance and detail.**

| # | Section | Glance (one line + one graphic) | Detail |
|---|---|---|---|
| 0 | Changed | One line, only when something changed | The list; "agencies may still have acted" moves to the footer |
| 1 | Official status, with RAMP | Tally + seven-mark ribbon; flagged fisheries as one-line rows | Sheet: cause, area, review, source, checked, verified, conflict, zone ladder |
| 2 | Toxin tests | "Dungeness 5.5 ppm avg, Sep 28 · 42 d old" + one shared 0–30 ppm dot strip | Per-site charts, key, clean-set progress, CDPH link |
| 3 | Bloom: model, then outlook | Two labelled lines, each with a mini mark | Explanation, skill, numbers, caveats |
| 4 | Sea: now, then forecast | Advisory + wind and seas + today and tonight bars + next tide | 10-period lanes, buoy provenance, tide curve |
| 5 | Keep, alert, ask | Three buttons | Readiness text inside each flow |

On desktop the left column is the evidence ladder (1–3) and the right column, sticky, is the sea. No row-pairing.

**Today / My Coast split.** About four of Today's seven sections are restated in the briefing: the sea (forecast, buoy, tide, advisory), C-HARM, "what changed" and the offline copy. Both also have a block titled "Before you go" meaning different things (`Actions.tsx:52`, `v4/coast/Briefing.tsx:413`).
- **Today owns the port now:** status and RAMP, CDPH tests, bloom, a 36-hour sea summary, alerts, Ask, changes.
- **My Coast owns one trip:** date, species, activity; 60-hour lanes against the reader's limits; forecaster's words; checklist; economics; notice geography on the map; dock value; kept copies.
- My Coast stops composing its own status line and shows Today's row for the chosen species from the same model (`buildTodayModel`).
- Today's multi-day lanes link out: "5 days and your limits → trip briefing".
- One name and one list (`/coast/kept`) for "Keep a copy".

## 4. Explore

**Chrome regions at 1440×900: 10, or 11 with a place open** (`v4/explore/Explore.tsx:1257–1331`).
1. Brand and tabs.
2. Search.
3. Notices and status cluster.
4. "Field" layer index: 6 rows of 2–3 lines, 263×425 px.
5. Legend plate: ramp, a 3-line gap note, "not toxin", a hint, a link.
6. Two tool stacks, 6 buttons.
7. Time rail: 2 captions, 11 day labels.
8. Rail provenance line.
9. Disclosure line.
10. Credits.
11. Inspector, 380×625 px.

**Diagnosis.**
- The index states kind, source and age for every layer permanently: about 60 words for five unselected layers.
- The legend holds onboarding text ("Tap a harbour…", "Go to Monterey Bay").
- The chlorophyll date appears three times: index row, rail caption, rail provenance.

**Unobstructed map now.** About 67% of the viewport by area. The largest clear rectangle is about 59% (x 280–1378, y 68–760). With a place open: about 49% and 41%.

**Proposal.**
- (a) Primary: the field.
- (b) Visible (25 words): the top bar with search, and one **field dock** at the bottom left. It is a capsule (swatch, "Chlorophyll · observed · Nov 7 · not toxin", an inline ramp) with the time rail as one 48 px track beside it. Zoom and locate stay. A 20 px margin holds credits.
- (c) Disclosure:
  - the layer index opens from the capsule as a popover (the `shortWide` path at `Explore.tsx:1279–1293` already does this);
  - gap, bay and reference notes go behind the legend's info button;
  - share, field pack and area select go behind one "more" tool, as on phones;
  - the read hint is a one-time coach mark;
  - a place opens a side sheet led by three statement lines.
- (d) Merge index, legend, rail and rail provenance into the dock. Fold the disclosure line into the shell pill.
- (e) Must stay on the map: the kind word and date in the capsule; "not toxin" (chlorophyll) or "probability, not toxin" (C-HARM); a hatch swatch labelled "no view"; stale colouring on the date; OSM credits.

**Result.** About 84% unobstructed by area and 78% clear rectangle at rest; about 66% with a place sheet open. On a phone the peek shrinks from 198 px to about 100 px, so the map gets about 72% instead of 58%.

## 5. My Coast

**Landing.** The hero is the best-composed screen in the app. Below it, `v4/coast/Landing.tsx:63–67` draws six cards with 18 readout cells. Four of the first nine say "Not reported". That is a dashboard for places the reader has not chosen.
- (a) Primary: the question and the search.
- (b) Visible (30 words): the hero, with the coast picker enlarged as the selector.
- (c) Readings show on hover or focus of a picker point.
- (d) Replace the grid with a one-line list of six names, home port first.
- (e) Any reading that stays keeps its age and instrument.

**Briefing diagnosis** (`coast-brief-desktop.png`).
- Thirteen regions stack in a 795 px column before the fold: disclosure line, eyebrow, title, three selects, subtitle, the season plate with four links (`Official.tsx:28–95`), a five-cell decision row (`Decision.tsx:74–108`), a three-line source strip (`:110`), the checklist fold, "Remembered on this device", and the lanes heading.
- The decision row gives equal weight to wind, seas, tide, a notice count and an empty economics promo.
- The source strip repeats the buoy reading, which the lanes' MEASURED row repeats again.
- The notices appear as claims in `Matters` and again as a list in `brief-official`.

**Briefing proposal.**
- (a) Primary: the trip line, never a go or no-go: "Tue Nov 10 · Dungeness (commercial): season delayed · sea within your limits until Tue night".
- (b) Visible (60 words): title; the trip as one summary line with "Change" (the phone pattern at `Briefing.tsx:304`, on every width); the species' status line; wind, seas, tide; the 60-hour lanes above the fold. The map stays as the ground.
- (c) Disclosure: season list, sea detail, evidence and unknowns stay in-place folds. Notice geography becomes a map interaction (tap a limit line). The worksheet stays a drill-in. The checklist becomes a side sheet.
- (d) Delete the source strip. Merge the notices cell into the status line and the `brief-official` list into the claims. Move "Remembered on this device" to the more menu.
- (e) Must stay: "not verified" on the status line; "Nothing here says a trip is allowed" on every width; "model · context" and "a gap, not a low value"; "hypothetical" on economics; "observed · Nov 2–7 · not toxin" on the map.

## 6. Research

**Diagnosis.** The density suits the audience and the chart is already primary. What competes with it:
- the always-open station ladder, 17 stations and 34 lines (`v4/research/CoastLadder.tsx`);
- two header readouts restating the chart's last point;
- three italic notes and five tabs below the chart (`Research.tsx:581–590`).

**Proposal.**
- (a) Primary: the two-panel record.
- (b) Visible (45 words): station, quantity tabs, range, both panels with their titles.
- (c) The ladder becomes a picker opened from the title or mini-map, as on phones.
- (d) Fold the readouts into the crosshair's default reading. Merge the notes into two info popovers.
- (e) Must stay on the chart: "measured" against "model probability" in the panel titles; "not comparable with the samples as numbers"; the "no model history kept before…" hatch; the reported-zero key.

## 7. Honesty risks in the current build

1. **The Today lede and Glance quote unverified records with no marker.** The first "Not verified" is at y≈584. Today has no shell pill or disclosure line either.
2. **"None on record" shows without its caveat.** "…is not a promise" (`Status.tsx:107–116`) is about 430 px below the desktop fold.
3. **No-record rows carry no chip** beside chipped rows.
4. **Commercial status contradicts across pages.** Today leads with a curated commercial Dungeness delay. The briefing shows the sport list ("24 on the open list, 3 closed") and tells a commercial reader "Neritic holds no commercial season" (`Official.tsx:69–74`).
5. **The Monterey shellfish notice differs.** Today: "Sport harvest closed" with a 13 px "sources differ". My Coast and Explore: "Status unclear… awaiting review".
6. **Wind.** Today prints "N 5–15 kt": `pilot/today/marine.ts:27–45` joins the first direction to the period's top speed, which NWS did not say. The briefing prints "N 5–10 kt, then…".
7. **Advisory.** Today: "in effect through this evening". The briefing lane: "no end time given" (`Lanes.tsx:201`).
8. **A past run reads as an outlook.** "Bloom outlook" charts days Nov 2–5, and the Glance prints "75%–77% (run Nov 3)" with no stale mark in place. The briefing calls the same run "a gap".
9. **A caveat is hidden on phones.** "Nothing here says a trip is allowed" is `max-sm:hidden` (`Official.tsx:76`).
10. **Explore phone peek** shows the chlorophyll ramp without "not toxin"; that line is below the rail (`Explore.tsx:1415`).

## 8. Tests that encode the current structure

Unit tests (`tests/unit/todayFacts.test.ts`, `scienceWording.test.ts`) cover the model and wording, not layout. They change only if risks 6–8 are fixed.

| Spec (`tests/e2e/`) | Asserts | Affected by |
|---|---|---|
| `today.spec.ts:29–107` | `today-glance` tally; `data-area` order exactly `status, sea, tests, bloom, whale, changed, actions`; 10 lane columns; `whale-lede`; per-row `not-verified`; full `outlook-words` text | Glance, order, whale merge, group marker, shorter prose |
| `today.spec.ts:170, 281, 318` | Feedback once per section; `changed-first`; `today-freshness` "(old)" | Feedback, changed, freshness |
| `pilot-journey.spec.ts` | `today-lede`, `sea-lanes`, `tests-lede`, `action-*`, `save-offline` | Actions, lanes disclosure |
| `explore-shell.spec.ts:31–62` | `layer-deck` always visible at x<40; no `layer-capsule` on desktop; rail over 90% wide; pairwise non-overlap with `registry-disclosure` | Field dock |
| `explore-rework.spec.ts:90–128, 227` | `rail-status`, `map-legend`, tools row under the inspector | Dock, tools |
| `explore-v10.spec.ts:79`, `navigation.spec.ts:99`, `hardening.spec.ts:60, 102`, `ergonomics.spec.ts:29, 43` | `layer-list` radios and arrow keys; `read-hint`; `zoom-monterey`; phone peek | Index popover, coach mark |
| `layout.spec.ts:82, 132`, `regulatory.spec.ts:8, 21`, `navigation.spec.ts:18, 39` | Peek under 45%; `registry-disclosure` and `notices-button` on every page | Shell pill |
| `mycoast-v10b.spec.ts:19–83` | Five decision cells on one line above 900 px; `glance-measured`; "tied here" once; phone first screen | Decision row, source strip |
| `briefing.spec.ts:12, 94, 128`, `mycoast-pass.spec.ts:12`, `fishing.spec.ts:13, 36`, `ergonomics.spec.ts:67` | Section order; notice counts agree; fold placement; "holds no commercial season" | Status unification (`fishing` encodes risk 4) |
| `v8.spec.ts:56, 73` | Six landing cards with `landing-readings`, `landing-sources` | Landing list |
| `research.spec.ts:12, 416, 445, 473` | `research-notes`, `research-stats`, ladder and map selection, phone first screen | Ladder picker, notes |
