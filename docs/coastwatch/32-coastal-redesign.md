# 32 — Coastal redesign: a briefing, not a catalogue

**Status:** draft PR for review, built on the notice-discrepancy fix (PR #40). Private preview only; nothing public. No forecast value, threshold, unit, mask, notice record or published dataset changed.

**Read with:**
- [31 — product audit](31-product-audit.md), the diagnosis;
- the research in [research/](research/):
  - [ocean sources](research/ocean-sources.md);
  - [nearshore coverage](research/nearshore-coverage.md);
  - [data health](research/data-health.md);
  - [stakeholders and interview plan](research/stakeholders-and-interviews.md);
  - [fishing-opportunity feasibility](research/fishing-opportunity-feasibility.md);
- the offline [sablefish prototype](../../research/sablefish-prototype/README.md).

## 1. Correctness first (PR #40)

**Problem:** CDPH lifted SN26-018 (Monterey County bivalves) on 2026-10-12 in SN26-021, but CoastWatch still showed it as in effect.

**Fix:**
- **Pipeline:** an unreviewed CDPH release titled as a lift, matching an active CDPH record (same county from the record's structured fields, same kind of seafood, later release number), is published as an "Agency source discrepancy" in the existing `conflicts` field. The schema is unchanged, so the public demo keeps validating.
- **Web:** such a record is never counted or listed as applying. It is shown as "Reported lifted by the agency · awaiting review" with the release link.

**Review packet:** [notice-review/2026-10-12-sn26-021](notice-review/2026-10-12-sn26-021/README.md).
- It holds the validated record change for a person to apply, evidence with raw-page hashes, and the audit.
- The audit covers: SN26-020 not yet reviewed; the changed CDFW page matches the registry; the mussel quarantine "through at least Oct 31"; C-HARM's skipped days; and NASA's 2026-11-02 end of Suomi-NPP data, which C-HARM relies on.

## 2. Information architecture: two modes, one source of truth

| | Coastal (default) | Research |
|---|---|---|
| For | Commercial and recreational fishers, shellfish harvesters, port and coastal operators, first-time visitors | Researchers and technical visitors (the welcome's "Research and science" sets it) |
| Navigation order | My Coast · Map · Blooms · Fisheries | Map · Blooms · Fisheries · My Coast |
| My Coast | Briefing: official status → next steps → map → conditions → *Scientific details and every notice* (closed) | Official status → map → conditions → the full scientific sections, measurements first → next steps |
| Ocean Map | Opens on the coast and seafloor (depth-tinted bathymetry); forecast, satellite and currents one tap away, each with its date | Opens on the forecast or satellite view, as before |
| Bloom | "What the latest samples show" in plain words, against the station's own last 12 months; *Full station analysis* (closed) | The full station analysis |
| Fisheries | "Your fishery, now": notices that name the species, the agency's statements, its rules, "plan a trip in My Coast"; *History* (closed) | Statewide landed-value history |
| Status line | Notices disclosure only (failed updates under "More" and on each layer) | Notices disclosure and failed updates |

**Rules for the modes:**
- The mode changes order, defaults and what is open, **never a value, a notice, a caveat or a verification status**.
- Every page and every notice is reachable in both modes.
- Switching is instant: in the masthead from 1280 px, at the foot of My Coast, and through the welcome's choice.
- The mode is kept on the device. A link's `mode=` wins for that view and is not saved.
- Shared links (place, species, activity) reproduce the content in either mode.
- The server renders the Coastal layout, so pages still work without JavaScript; the folds are native `<details>`.

## 3. Design system (one visual language)

- **Surfaces:** paper background; white cards with an 18 px radius and hairline borders. Official content uses the warm official tint; nothing else is tinted.
- **Type:** a serif headline per card that states the answer ("2 notices apply here for anchovy or sardine"), a small uppercase eyebrow for its context ("Official notices · these waters"), and 13–15 px body text.
- **One colour language for kinds of information**, used as dots in Next steps and Conditions:
  - official: amber;
  - conditions: teal;
  - measured: green;
  - model: violet;
  - not known: grey.
- **Freshness** is a word plus a date ("issued Oct 11", "observed 10:00 PM PDT"). A failed update reads **Not confirmed**, never "Current".
- **Caution is placed, not repeated.**
  - There is one site-wide notices disclosure.
  - Each card states the one caveat that changes its meaning: "That is not an all-clear", "not a forecast for these waters", "context for planning · not a toxin or fishing forecast".
  - The old pattern of the same sentence in every component is gone. On My Coast, "not verified" now appears 4 times instead of 7: the masthead pill, the status line, the official card and one next step.
- **Map:**
  - With no data layer, the map is a depth-tinted bathymetric basemap. It is drawn from the same NOAA ETOPO 2022 and Coastal Relief Model data as before, by a tint variant of `build_relief.py`.
  - Under any data layer the neutral relief returns, so a tint can never be read as a value.
  - The neutral images are reproduced byte for byte.

## 4. Alternatives compared (actual Monterey data, 2026-10-12)

| My Coast alternative | Strength | Weakness | Verdict |
|---|---|---|---|
| **A, Briefing** ([1440](redesign/img/alt-A-briefing-1440.jpg), [390](redesign/img/alt-A-briefing-390.jpg)): a single column of cards with the map beside it | Official status, next steps and conditions all visible without tabs; becomes one simple column on a phone | Long on a phone (mitigated by short cards and the closed science fold) | **Chosen** |
| B, Chart table ([1440](redesign/img/alt-B-chart-table-1440.jpg)): the map leads, with a tabbed briefing rail | Striking | Next steps and conditions hidden behind tabs; repeats the Ocean Map | Rejected for My Coast; its place card informs the map |
| C, Checklist ([390](redesign/img/alt-C-checklist-390.jpg)): numbered "Before you go" carries the data | Clear sequence | Nested indentation wastes phone width; the data become hard to scan | Rejected; its "Before you go" framing became the Next steps eyebrow |

| Map treatment (same data) | Monterey | Statewide | Verdict |
|---|---|---|---|
| A, today's default: forecast raster at full opacity | [img](redesign/img/map-monterey-A.jpg) | [img](redesign/img/map-statewide-A.jpg) | The bay is a flat pink field that reads as a hazard map; the geography is hidden |
| B, basemap only: neutral relief, no data | [img](redesign/img/map-monterey-B.jpg) | [img](redesign/img/map-statewide-B.jpg) | Calm but dull; the relief was built dark to sit under data |
| C, forecast at 50 % opacity | [img](redesign/img/map-monterey-C.jpg) | [img](redesign/img/map-statewide-C.jpg) | Still dominant, and translucency mixes the colour scale with the basemap: values become hard to read |
| **Chosen: depth-tinted bathymetry with no data layer; the forecast opt-in at full fidelity** | [after](redesign/img/after-map-coastal-1440.jpg) | [phone](redesign/img/after-map-coastal-390.jpg) | Attractive with no scientific layer on; exact colours and legends when one is chosen |

## 5. My Coast briefing

- **Plan bar:**
  - activity: any, commercial, recreational or shellfish;
  - species: 11 California fisheries, matched only to each notice's own species list;
  - forecast day.
  - Species and activity go in the link (`?species=razor-clam&for=shellfish`). Choosing a species re-orders and never hides: "N more for other species here" is one tap away.
- **Official status:**
  - The headline answers for the chosen species.
  - Each notice shows its agency, action, audience, title, "since" date and agency source.
  - A limit nearby is called out ("south limit 8.4 km south of here").
  - Notices that may apply but aren't settled by their stated area are listed in a line; nearby notices outside their area are counted.
  - Disputed notices appear as "Reported lifted by CDPH · awaiting review".
  - "All 11 notices in California" opens the drawer.
- **Next steps** (`lib/nextSteps.ts`, unit-tested): rules, not advice. Every step has a reason, a time where one matters, and a source to open. The kinds are:
  - **Official:**
    - the notices that apply, as one step;
    - a notice an agency appears to have lifted, first;
    - notices whose geography doesn't settle it;
    - CDFW's quoted statements for the species, matched to the activity;
    - the species' rule page;
    - the CDFW region's sport rules, or the commercial rules;
    - CDFW's domoic acid closure line for commercial crab;
    - CDPH's shellfish map and biotoxin line for shellfish;
    - MPAs.
  - **Conditions:** NWS marine alerts and the zone forecast; the nearest station's measured wind and waves.
  - **Gaps:** no toxin sample within 30 days; a marine forecast that didn't load; the notice list not being verified.
  - No step says waters are open, seafood is safe or fishing will be good. A test checks every combination of species, activity and place.
- **Conditions** ([`app/api/conditions`](../../coastwatch-web/src/app/api/conditions/route.ts)): see §6. Toxin sample, toxin forecast and satellite view are shown in the same grid, each with its own date and source. The toxin forecast reads "No toxin forecast for these waters" where the waters' own cell has none.
- **Scientific details and every notice:** the previous sections (full official list with tiers, measurements, forecast availability, observations, what is not known), unchanged, in one fold. In Research mode they are open and come first.

## 6. Ocean conditions: staged integration

| Source | What is shown | Behaviour | Verdict |
|---|---|---|---|
| NWS coastal marine zone forecast and alerts (`api.weather.gov`) | The first period's wind and seas in the forecasters' words, with the zone, issue time and link; active alerts (e.g. Small Craft Advisory) with end time | The zone is found at the waters, or by searching the nearest water in rings (2, 5, 10 km × 8 directions). The search uses the nearest zone that has a forecast; harbor-bar zones have none. | **GO (staging)** |
| NDBC station, latest observations (`ndbc.noaa.gov/data/realtime2`) | Wind, waves, water temperature, each with its own time, station and distance | The 8 nearest stations within 60 km are tried in order. Each field takes its latest valid row, never older than 6 h. Reads are ranged to 8 KB. Silent stations are skipped and the reason is stated. | **GO (staging)** |
| CO-OPS tide predictions | The next high and low water at the nearest prediction station, in feet above MLLW | Predictions, labelled as such; within 40 km | **GO (staging)** |

- **Checked live** at 26 places (22 ports and 4 harvest sites):
  - every place resolved to a marine zone with a forecast;
  - 20 had a recent station reading; 5 had only silent nearby stations or none within 60 km (Shelter Cove), and each says so;
  - one tide request (Point Arena) briefly returned no predictions; the route now falls back to the next-nearest of 3 stations.
- **Operation:** each part fails on its own with a stated reason. Every request has a 6 s budget, and responses are cached for 10 minutes.
- **Tests:** they replay 26 places' recorded responses from one 435 KB file (`tests/conditions-fixtures.json`), so CI never touches the network.
- **Labelling:** shown as "Context for planning · not a toxin or fishing forecast".

**Deferred, with reasons** ([ocean sources](research/ocean-sources.md)):
- **WCOFS:** later. NOAA's own report says its 4 km grid "does not sufficiently resolve the coastal bays and estuaries".
- **Satellite SST:** later; the next candidate.
- **CUTI/BEUTI upwelling indices:** later; they run about 15 days behind.
- **Copernicus Marine:** no; it needs an account and comes with attribution and usage-record obligations.
- **Machine-readable regulations:** no; CDFW publishes only HTML tables and PDFs.
- **MPA polygons:** only as a reference overlay. The CDFW layer is dated 2019, says it is not for legal boundaries, and the Commission's decision on 20 MPA petitions is pending.

## 7. Three critique passes (real browsers, production data)

**Pass 1 (structure):**
- The toxin value showed as a raw float with "1503 days ago"; it is now rounded with "over 4 years ago".
- Buoy names carried CDIP clutter.
- The activity chips wrapped.
- Next steps repeated each notice already shown in the official card; they are now one step.
- The header repeated coordinates.
- The mode switch crowded the phone masthead.
- The day chips wrapped.
- The marine zone for Monterey harbor stepped across the peninsula into the wrong zone; the ring search fixed it.
- Eureka hit a harbor-bar zone with no forecast; the route now uses the nearest zone that has one.
- San Pedro's nearest stations were silent; the route now tries up to 8.

**Pass 2 (consistency):**
- With no layer chosen, the map panel said "No current forecast or recent satellite view". That was false in Coastal mode; it now says "The coast and seafloor" and offers each layer with its date.
- The status line repeated failed updates.
- The drawer led with a large verification box and grouped notices by agency. It now shows one line ("Transcribed… not verified. Why?") and groups by what people harvest.
- The Fisheries empty-state panel stretched to its neighbour's height.
- The Bloom locator map was empty while loading; it now has the tinted seafloor.
- The text "open water" tripped the safety rule and became "ocean cell".

**Pass 3 (detail and width):**
- "dungeness crab" lost its capital.
- A commercial fisher saw CDFW's *recreational* Dungeness statement; statements now follow the activity.
- The day chips didn't say what they control ("Forecast day").
- At 1280 px the masthead scrolled sideways with the switch beside full labels; full labels now start at 1536 px.
- Without JavaScript, Fisheries showed an empty placeholder; the server now renders the Coastal layout.
- Station lists were bundled into the page; they are now server-only.
- **Checks at every size:**
  - **My Coast:** every example place at 1440 and 390 px; Santa Cruz also at 1280, 1024 and 320; Clam Beach at 320.
  - **Coastal map:** all five sizes.
  - **Other views:** Bloom, Fisheries, a Research view and a deep link, each at 1440 and 390 px; the first visit at 1440 and 390 px.
  - **Result:** no page error and no sideways scroll in any.

## 8. Measurements (production data, 2026-10-13; median ranges of 3 runs × 3 places)

| | Before (doc 29, Oct 12) | After |
|---|---|---|
| Official status on screen, desktop | 0.39–0.59 s (the summary) | **0.13–0.26 s** |
| Official status on screen, phone (CPU 4×) | 0.94–1.30 s | **0.45–0.58 s** |
| Live ocean conditions on screen | n/a (none) | 0.16–0.44 s desktop · 0.68–0.94 s phone |
| Layout shift (CLS) | ≤ 0.046 desktop · ≤ 0.014 phone | ≤ 0.026 desktop · 0 phone |
| Filter response | 76 ms desktop · ~130 ms phone | 42–113 ms · 84–138 ms |
| Clicks to an agency source | 1 (in the official section, below the summary) | **0: a source link is on the first screen in every run** |
| Map unobstructed, Ocean Map default | ~51% (forecast raster covering the sea) | **85% desktop · 73% phone, with the geography visible** |
| Words on the first screen (desktop): My Coast · Ocean Map · Fisheries · Bloom | 319 · 281 · 463 · 562 | 247 · 180 · 344 · 517 |
| "Not verified" on the whole My Coast page (masthead and status line included) | 7 | 4 |
| Controls on My Coast's first screen (desktop) | 33 | 38: more, because of the activity, species and day choices; the trade-off for answering by species |
| First-load JS | `/coast` 151 kB · `/` 174 kB · Bloom 144 · Fisheries 138 | 159 · 176 · 152 · 148 |
| Data transferred on My Coast load (incl. map tiles and grids) | — | about 1.3 MB |

Accessibility:
- axe finds no serious or critical issue on the Coastal briefing, desktop and phone.
- All earlier axe checks still pass.
- Keyboard paths are unchanged; mode, activity, species and day are buttons or a native select.

## 9. Tests

- **Unit:** 228 pass. New ones cover:
  - parsers for NWS, NDBC and CO-OPS on recorded responses;
  - nearest-station selection;
  - next-steps rules: sources, safety wording, activity-matched statements, disputes first, gaps;
  - species matching.
- **Browser:** all pass. `coastal.spec.ts` covers:
  - the briefing;
  - species in the link;
  - the honest gap at Shelter Cove;
  - mode switching and link precedence;
  - the basemap-first map;
  - "Not confirmed";
  - axe on desktop and phone.
- **Earlier suites** run in Research mode through the shared storage state, so their assertions stand.
- **Pipeline:** 205 pass, including PR #40's discrepancy tests.

## 10. GO / NO-GO by feature

| Feature | Recommendation |
|---|---|
| Discrepancy state for agency lifts (PR #40) | **GO now**: merge, so production stops presenting SN26-018 as in effect; a person then reviews SN26-021 |
| Coastal / Research modes, My Coast briefing, next steps | **GO for the private preview and review**; GO to merge after your review |
| Live ocean conditions (NWS, NDBC, CO-OPS) | **GO in staging (preview)**; GO to merge after review. Public use needs the attribution page updated and a week of monitoring |
| Depth-tinted bathymetry, basemap-first Coastal map | **GO** |
| "Not confirmed" freshness rule | **GO** |
| Coastal Bloom and Fisheries views | **GO** after your review of wording |
| Fishing-location or date recommendations | **NO-GO.** No public data has catch and effort at the needed resolution ([feasibility](research/fishing-opportunity-feasibility.md)) |
| Sablefish survey-density prototype in the product | **NO-GO** (no level skill, under-covering intervals, fails in the South); at most a research exhibit after a groundfish scientist's review |
| Descriptive seasonal landings by port and month | **Conditional**: needs CDFW reuse terms confirmed, which means contacting CDFW, only with your approval |
| Nearshore toxin fill (interpolation or proxy) | **NO-GO**: no validated nearshore toxin product exists ([nearshore](research/nearshore-coverage.md)) |
| Public release of the full site | **NO-GO** until the notice registry and associations are reviewed by a person, an independent scientist reviews, the VoiceOver test is done, and stakeholder sessions validate the hypotheses |

## 11. Risks and blockers

- **C-HARM may stop or change version after 2026-11-02.** It uses Suomi-NPP VIIRS. The pipeline needs a C-HARM version and metadata watch with its own alert ([data health](research/data-health.md) §improvements).
- **Silent failures.** The 00:42Z run had three source failures and raised no alert, because alerts need three failed runs in a row.
- **The live conditions route depends on NOAA endpoints.** Each part fails visibly, but the preview has no monitoring for them yet.
- **Stakeholder needs** for CPFV operators, shellfish harvesters, port managers and C-HARM users are mostly assumptions: no published California study was found. Interview them first ([plan](research/stakeholders-and-interviews.md)).
- **Wording** in the next steps and statements should be reviewed by someone who fishes California waters.

## 12. Screenshots (production data, after)

| View | Desktop | Phone |
|---|---|---|
| My Coast, Santa Cruz, commercial, Dungeness | [1440](redesign/img/after-coast-santa-cruz-1440.jpg) · [1280](redesign/img/after-coast-santa-cruz-1280.jpg) · [1024](redesign/img/after-coast-santa-cruz-1024.jpg) · [full page](redesign/img/after-coast-santa-cruz-1440-full.jpg) | [390](redesign/img/after-coast-santa-cruz-390.jpg) · [320](redesign/img/after-coast-santa-cruz-320.jpg) |
| Monterey, recreational, anchovy | [1440](redesign/img/after-coast-monterey-1440.jpg) · [full page](redesign/img/after-coast-monterey-1440-full.jpg) | [390](redesign/img/after-coast-monterey-390.jpg) |
| Crescent City, commercial, Dungeness | [1440](redesign/img/after-coast-crescent-city-1440.jpg) | [390](redesign/img/after-coast-crescent-city-390.jpg) |
| Eureka, shellfish, razor clams | [1440](redesign/img/after-coast-eureka-1440.jpg) | [390](redesign/img/after-coast-eureka-390.jpg) |
| Morro Bay, commercial, rock crab | [1440](redesign/img/after-coast-morro-bay-1440.jpg) | [390](redesign/img/after-coast-morro-bay-390.jpg) |
| San Pedro, recreational, rockfish | [1440](redesign/img/after-coast-san-pedro-1440.jpg) | [390](redesign/img/after-coast-san-pedro-390.jpg) |
| San Diego, recreational, spiny lobster | [1440](redesign/img/after-coast-san-diego-1440.jpg) | [390](redesign/img/after-coast-san-diego-390.jpg) |
| Clam Beach (harvest site), shellfish | [1440](redesign/img/after-coast-clam-beach-1440.jpg) | [390](redesign/img/after-coast-clam-beach-390.jpg) · [320](redesign/img/after-coast-clam-beach-320.jpg) · [full page](redesign/img/after-coast-clam-beach-390-full.jpg) |
| Ocean Map, Coastal | [1440](redesign/img/after-map-coastal-1440.jpg) · [1280](redesign/img/after-map-coastal-1280.jpg) · [1024](redesign/img/after-map-coastal-1024.jpg) | [390](redesign/img/after-map-coastal-390.jpg) · [320](redesign/img/after-map-coastal-320.jpg) |
| Ocean Map, forecast chosen | [1440](redesign/img/after-map-forecast-1440.jpg) | [390](redesign/img/after-map-forecast-390.jpg) |
| Bloom (Coastal) | [1440](redesign/img/after-bloom-1440.jpg) | [390](redesign/img/after-bloom-390.jpg) |
| Fisheries (Coastal) | [1440](redesign/img/after-fisheries-1440.jpg) | [390](redesign/img/after-fisheries-390.jpg) |
| My Coast in Research mode | [1440](redesign/img/after-coast-research-1440.jpg) | [390](redesign/img/after-coast-research-390.jpg) |
| Deep link to a port on the map | [1440](redesign/img/after-deep-link-map-port-1440.jpg) | [390](redesign/img/after-deep-link-map-port-390.jpg) |
| First visit (welcome) | [1440](redesign/img/after-first-visit-1440.jpg) | [390](redesign/img/after-first-visit-390.jpg) |
| Before | [map](redesign/img/before-d-map.jpg) · [My Coast](redesign/img/before-d-coast-monterey.jpg) · [drawer](redesign/img/before-d-drawer.jpg) · [Bloom](redesign/img/before-d-bloom.jpg) · [Fisheries](redesign/img/before-d-fisheries.jpg) | [map](redesign/img/before-m-map.jpg) · [My Coast](redesign/img/before-m-coast-monterey.jpg) |
