# My Coast as a trip briefing (fishermen-first transformation)

Branch `feat/fishermen-first-ux`, based on `release/2026-10-14` (the research preview released on
2026-10-14). Owner brief: "I'm planning to fish in these waters. What do I need to know, and what
should I check or do next?"

## 1. Diagnosis (the release, on real data, 2026-10-13)

- **The trip was not the subject of the page.**
  - Activity, species and day sat in a card below the fold.
  - The page opened on a place name, four dense lines of mixed facts, and a "remembered on this device" note.
- **The same statement appeared three times.** CDFW's Dungeness season status showed in the answer line, the official card and "Before you go".
- **Distant evidence got a full card.** A station 394 km away (Bodega Bay, for Crescent City) had its own evidence card, with "too far to describe these waters" as a footnote.
- **The map opened on 2 km of blocky relief.** On desktop, the place view had no landmarks.
- **Every page carried an amber banner** ("Official notices are not verified…") as well as the "10 · Not verified" pill.
- **A dead end.** For commercial Dungeness crab, the CDFW domoic acid closure line was never offered: the rule tested the species id `dungeness` for "crab".

## 2. What changed

**My Coast, Coastal mode** (`components/trip/*`, `lib/trip.ts`). One trip, one page, five parts:

1. **Your next trip.**
   - The place name is the heading.
   - Activity, species and day are three chips. Each opens a picker: a popover on wide screens, a bottom sheet on phones.
   - The day goes in the link (`date=`).
   - Days are today plus three, the span of the tide predictions. A day NOAA's model does not reach says so.
2. **What matters right now.** At most four or five cards, most consequential first:
   - the agency's own words on the fishery (season status, quoted, with any second statement from the agency quoted inside the same card);
   - a notice an agency release appears to lift;
   - the notices that cover these waters by their own geography;
   - one HAB line;
   - one conditions line.

   Restrictive notices and closed seasons sit on an amber rule. A gap is hatched and says what is not known, never that anything is safe.
3. **HAB evidence near the waters.** One time axis: the last 90 days, then the past week and the trip day at a finer scale. Rows:
   - Pseudo-nitzschia cells and domoic acid in seawater, each with its station, distance and age;
   - the latest clear satellite view;
   - NOAA's model days.

   A row whose nearest source does not describe these waters is hatched. Its place and distance are listed under the chart. The limits are said once, under the chart.
4. **Before you leave.** Up to five actions, each with:
   - an action title and why;
   - which waters or species it concerns;
   - a large button to the agency page or phone line;
   - "What CoastWatch can't settle", which opens in place.

   Steps on the same agency page are folded into one. Ticks are kept on this device per trip, and a tick means only that the source was opened.
5. **Explore the evidence.** The existing detail in tabs: HAB detail (station timelines, satellite, model days), notices, conditions, science and sources.

**Other changes.**

- **Phones.** The map is a sheet behind a Map button. It is loaded on the first open only, and "Show on map" opens it on the item.
- **"Since you last checked"** (PR #47) follows "What matters", in the order the fishermen-first research gives: status, notices, HAB, conditions, then changes.
- **Server rendering.** The server renders the trip briefing (Coastal is the default). The evidence chart never draws wider than a phone before it is measured; a wider first frame made phones zoom out, and a test now checks the layout stays 390 px wide.
- **Masthead and notice provenance.**
  - Coastal no longer has the standing amber line. The masthead pill is neutral: "10 · Not verified", and "Not verified copy" on very wide screens.
  - Each official card carries its agency source and "CoastWatch's copy, checked {date}".
  - One sentence under "What matters" says that no person has reviewed the copy and that the agency pages are the authority.
  - The first-visit welcome now links CDFW and CDPH.
  - Research mode keeps the full status line.
  - The registry remains `pending_human_review`: nothing here marks a notice reviewed. The human review (`cwp review-official`) remains an outstanding operational requirement.
- **Map.**
  - Coastal opens the Ocean Map on the whole California coast unless a link names a region.
  - No data is a faint diagonal hatch (the design language's "not known" mark) instead of a dense dot lattice.
  - C-HARM cells are unchanged: exact edges, no smoothing, no interpolation.
- **Conditions.**
  - `api.weather.gov` stays off (robots.txt `Disallow: /`).
  - For waters within 20 km of a CDFW port or CDPH harvest site, the route reads NWS's coastal waters forecast and headlines from the plain-text product on `tgftp.nws.noaa.gov`, which has no robots.txt. Each place's inner zone was checked against the zone file's own description (`lib/nwsText.ts`).
  - The conditions card quotes the trip day's period, and an advisory becomes an alert line.
  - `CW_NWS_TEXT=0` turns it off.

**Not changed.** Research mode's My Coast, the Bloom and Fisheries pages, the pipeline, the published data and its schema (no manifest fields), notice records and their review state, C-HARM rendering.

## 3. Data sources investigated

[sources.md](sources.md) has the measured table: status, access and robots, units, flags, California nearshore coverage, age and verdict. In short:

- **Sentinel-3A OLCI 300 m: keep.** On an average day only about 17 % of pixels within 3 km of the coast have a value.
- **NOAA-21/NOAA-20 VIIRS (PR #46): adopt.** It reaches about 50 % within 1 km while flowing, but has been stalled since 2026-10-05. A stall alarm is urgent because S-NPP ends on 2026-11-02.
- **NOAA DINEOF gap-filled chlorophyll: reject.** It has no observed/filled flag and fills 96 % of the 0–1 km strip.
- **IFCB: reject.** Only raw images are published.
- **CDPH shellfish results: PDF only.** They need human curation.
- **WCOFS: later.** Its OPeNDAP endpoint failed.
- **NWS text forecasts on tgftp: adopted here.**
- **NDBC `.ocean` pier relays (water temperature, chlorophyll fluorescence): recommended next.**

Compositing: one sensor per pixel with its sensor and date is legitimate and is what CoastWatch already does. Averaging sensors or showing unflagged gap-filled values is not. Toxin probabilities are never filled.

## 4. Journeys measured (scripted, not user tests)

`coastwatch-web/scripts/trip-bench.mjs`:

- **Setup.** Eight journeys on a 390 × 844 phone, run against the committed fixture data, a fixed clock (2026-10-08) and recorded NOAA responses. The same rubric was applied to the release (`2a995992`) and to this branch.
- **Columns.** "Found" counts the rubric items a person can reach at all. "Scrolling" is how far below the first screen they are.
- **Fixture limits.**
  - The fixture's notice list predates the SN26-021 discrepancy.
  - The recorded tides begin on 2026-10-12, so the fixture day has no tides of its own.

| Journey | Found (before → after) | On the first screen | Scrolling to reach them, screens | Taps |
|---|---|---|---|---|
| J1 Crescent City, commercial Dungeness crab | 4/5 → 5/5 | 2/5 → 3/5 | 6.9 → 2.7 | 1 → 0 |
| J2 Monterey, commercial anchovy | 4/4 → 4/4 | 2/4 → 3/4 | 6.9 → 1.0 | 0 → 0 |
| J3 Santa Cruz, recreational fishing (fresh HAB data) | 4/4 → 4/4 | 2/4 → 2/4 | 5.4 → 1.7 | 0 → 0 |
| J4 Clam Beach, recreational razor clam harvest | 3/3 → 3/3 | 1/3 → 2/3 | 7.0 → 0.9 | 0 → 0 |
| J5 Shelter Cove, sparse monitoring | 3/3 → 3/3 | 1/3 → 1/3 | 4.5 → 2.1 | 0 → 0 |
| J6 San Diego, outside the toxin model's values | 2/2 → 2/2 | 1/2 → 1/2 | 1.9 → 0.9 | 0 → 0 |
| J7 Crescent City, a trip in three days | 2/2 → 2/2 | 1/2 → 2/2 | 3.6 → 0.0 | 2 → 2 |
| J8 A returning fisherman checks what changed | 1/1 → 1/1 | 1/1 → 0/1 | 0.0 → 0.5 | 0 → 0 |
| **All eight** | **23/24 → 24/24** | **11/24 → 14/24** | **36.2 → 9.8** | **3 → 2** |

**The one regression is J8.** The change list is now half a screen down, after "What matters", which follows the fishermen-first order. No real fisherman has used either version. These numbers say where things are on the page, not whether people understand them.

## 5. Known limitations and follow-ups

- **Human notice review** (`cwp review-official`) is still outstanding. It is an operational requirement, not a styling problem.
- **NWS headlines come from the latest product.** An advisory "until 9 PM this evening" stays listed until NWS issues the next product (it is quoted with its own time).
- **VIIRS:** the NOAA-21/20 chlorophyll stall needs an alarm (PR #45/#46).
- **Next sources:** NDBC pier relays (temperature, chlorophyll fluorescence) and CDPH crab/shellfish results (human-curated).
- **The Ocean Map's place chip** says "Monterey Bay" for one frame before Coastal switches it to "All California". Making the default region statewide on the server would also change Research's default.
- **California emphasis:** dimming non-California land needs a state outline; the bundled county file has borders only.
