# 29 — My Coast MVP

**Status:** draft PR #38 for review; not merged, not deployed publicly. Web only: no pipeline, dataset, forecast value or notice record changed, and nothing new is predicted.

**Design brief:** [28](28-my-coast-brief.md) (PR #37). **Evidence for all 26 locations, production data:** [my-coast/applicability-evidence.json](my-coast/applicability-evidence.json).

## 1. What My Coast does

`/coast` answers, for waters someone chooses, eight questions without asking them to read an offshore raster:

| Question | Where | Source |
|---|---|---|
| What official information is relevant? | **Official information**: three groups, by the notices' own geography: *apply here*; *may apply, not settled* (shown open); *nearby, outside the official area* (folded). Why each is listed; a note when agency pages have changed since the last review; agency links, which are always shown and described as the complete record | CDFW/CDPH registry (not verified); CDFW region pages; CDPH map and results PDFs |
| What was measured nearby, and how recently? | **Measured nearby**: stations within 40 km, value, unit, sample date, age, "no sample in the last year" | CalHABMAP via SCCOOS ERDDAP |
| Is a regional C-HARM forecast available? How far offshore are the nearest values? | **Forecast availability**: value only if the waters' own 3 km cell has one. Otherwise it leads with "No C-HARM forecast for these waters" (or names the one variable that is missing), then sets the nearest real cell apart as "not a forecast for these waters", with its distance, direction, centre coordinates, valid date and limits; 15 km summary; run dates; stale flag | NOAA C-HARM v3.1 |
| Recent satellite and current observations? | **Recent observations** | NOAA CoastWatch OLCI/VIIRS; HF radar |
| What does CoastWatch not know? | **What CoastWatch does not know here** (computed for the place) | — |
| Where to check the rules? | Agency list: CDFW advisories, CDFW regulations for the region (and the neighbour near a limit), commercial fishing, MPAs, CDPH advisories, map and test results, OEHHA, phone lines | Agency pages (checked by title) |

**Places:**
- **Ports:** the 22 CDFW ports.
- **Named harvest sites:** 4 CDPH biotoxin monitoring sites, using CDPH's own coordinates. These are Crescent Beach, Clam Beach, Tomales Bay and Goleta Pier.
- **Points:** any point chosen on the map.

**Home port** is remembered on the device and kept separate from the waters chosen. No one is assumed to fish near their home port.

**Links:** `/coast?place=port-593&home=port-550&for=commercial` reproduces a view. Back and Forward move between waters, and an unknown place in a link says so instead of guessing.

**First screen:**
- The place.
- A four-line summary of what exists (official, measured, forecast, satellite), which is never a rating.
- On wide screens, a map showing where each fact came from: the waters, the stations, the C-HARM cell used and a dashed line to it.

**Elsewhere:**
- The Ocean Map's port panel and point inspector link to the same waters ("View in My Coast").
- My Coast is in the masthead and is the sixth phone tab.

## 2. Workflows (production data, 2026-10-12)

| Stakeholder | Workflow | What they get | Assumption basis |
|---|---|---|---|
| **Commercial fisher** | Saves Santa Cruz as home port; opens Monterey Bay waters by clicking the map; filter "Commercial fishing" | CDFW anchovy take restriction (geography: inside 37°11′–36°31′ N) and CDPH anchovy advisory; "1 more for other kinds" folded, not hidden; CDFW commercial fishing link; nearest toxin sample Santa Cruz Wharf 8.8 km, 12 days ago | Documented: notices by species and fishery type, protocol links (appendix B, doc 27). Hypothesis (interviews): that fishers pick waters on a map rather than by port |
| **Recreational angler** | Opens Monterey; filter "Recreational fishing" | CDFW Central-region regulations link (37°11′–34°27′ N); anchovy notices with "south limit 8.4 km south of here; check which side you are on"; Monterey Wharf "no sample in the last year" (last pDA Aug 31, 2022) | Documented: recreational and commercial status can differ; regulations by region (CDFW pages). Hypothesis: anglers start from a port name |
| **Shellfish harvester** | Opens Clam Beach (CDPH site); filter "Shellfish harvest" | CDFW Humboldt razor clam closure (inside the agency's Humboldt outline); CDPH test-results PDFs (razor clams); "PSP results are not shown here"; the CDPH advisory SN24-003 appears once the next data refresh publishes it | Documented: CDPH advisories by county; PSP decisive and absent from CoastWatch. Hypothesis: beach names as the entry point |
| **Port manager** | Opens Crescent City | Del Norte razor clam closure and advisory, North Coast rock crab advisory; "No monitoring station within 40 km; nearest Trinidad Pier, 77 km"; nearest C-HARM toxin cell 2.4 km WSW; status line shows stale or failed updates site-wide | Hypothesis: port managers want a single port view with source ages |
| **HAB researcher** | Opens a point in Monterey Bay, then Moss Landing | Point: C-HARM has a value in its own cell; Moss Landing: the port's own cell has a value (74%) while Santa Cruz's nearest is 7.2 km; station history and source CSV links; the map draws the cell used | Documented: provenance and downloads (appendix B §5) |

All five were exercised in a browser at 1440 × 900 and 390 × 844. Interview validation is still needed for every row marked "hypothesis" (doc 27 §4 lists the plan).

## 3. Scientific and geographic correctness

- **Notices are matched by their own geography only, never by text.**
  - The matching rules are statewide scope, official latitude limits, and the agencies' area outlines (point-in-polygon, with a 1 km shore tolerance for generalized outlines).
  - A unit test changes a notice's description to "Santa Cruz County" while its outline stays Monterey County, and the notice must not apply to Santa Cruz.
  - The Northern Channel Islands advisory names Santa Cruz Island and is never matched to Santa Cruz port.
- **C-HARM is never interpolated.**
  - The value "at these waters" is the value of the 3 km cell containing them, if any.
  - Otherwise the section and the summary lead with "No C-HARM forecast for these waters". The nearest real cell is set apart as "not a forecast for these waters", with its distance and direction to the cell centre, the centre's coordinates, the valid date and its limits.
  - The toxin and bloom grids are masked differently. At 7 ports (Trinidad, Shelter Cove, Fort Bragg, Bodega Bay, Monterey, Port Hueneme, Oceanside) the cell has a bloom probability but no toxin value, and the lead says exactly that.
  - Checked in a browser on production data at all 22 ports and 4 harvest sites for all four forecast days: only Moss Landing's cell has a toxin value; every other place leads with the missing forecast, and the notice groups match the evidence file.
  - The 15 km summary counts real cells only.
  - Unit tests use a synthetic grid with an empty cell, a valued cell, and no cell within the radius.
- **Agreement with the existing app.** For all 22 ports, on production data and in a unit test on the fixtures, the geographic tier matches exactly (0 mismatches) the pipeline's own port-to-notice relations (`port_intel`). Beyond those relations, the outside tier lists Monterey County's advisory 23 km from Santa Cruz, and the unresolved tier lists the islands part of the rock crab advisory near Ventura, Port Hueneme, Santa Barbara and Goleta.
- **Stations** come from the validated observations dataset, with real sample dates. A sample over a year old reads "over N years ago (date)" and "no sample in the last year".
- **Freshness:** the stale-run test uses the failed-update fixture and shows "Stale" in the forecast section, in the summary and in what is not known.

## 4. Applicability model and the association audit

**Three tiers (lib/applicability.ts).** Being near a notice's area is never shown as the notice applying.

| Tier | Meaning | Shown as |
|---|---|---|
| `geographic` | The notice's official geography includes the location: statewide; inside its official latitudes; inside, or within 1 km of, the agency's outline of its area | "Apply here by their official geography" |
| `unresolved` | The official geography does not settle it: a point on the water off a county- or area-defined notice (no notice states its offshore reach); a shore place within 5 km of a generalized outline's edge; a part of a notice with no official boundary | "May apply: not settled by their official geography", shown open |
| `outside` | A documented boundary excludes the location: beyond an official latitude limit, or a shore place clearly outside the agency's outline. Listed only because it is within 30 km | "N nearby notices outside their official area", folded |

Ports and harvest sites are shore locations; a point chosen on the map is on the water, so a county notice near it is unresolved, never outside.

**`noticeParts`:** one reviewable candidate. The rock crab viscera advisory also covers "the waters off Santa Rosa Island and the Northern Channel Islands", which no official boundary defines. Its outline is borrowed from CDPH's islands advisory, so associations from it are only ever unresolved. **`reviews`:** empty. No association is confirmed or rejected, and no reviewer name exists by default.

**Audit of the associations raised for review** (official wording checked on 2026-10-12 against the CDFW Health Advisories page and the CDPH releases; production data):

| Location | Notice | Official geography (verbatim) | Tier | Finding |
|---|---|---|---|---|
| Santa Cruz | CDPH SN26-018, Monterey County bivalves | "gathered from Monterey County" | **outside** (23 km) | Santa Cruz is in Santa Cruz County. The advisory does not cover it; it stays visible for harvesters who also go to Monterey County. **CDPH lifted SN26-018 on 2026-10-12 (SN26-021)**; the registry still lists it until a person records the lift |
| Santa Barbara, Ventura, Port Hueneme, Goleta Pier | CDPH Northern Channel Islands bivalve special advisory | CDPH map polygon of Anacapa, Santa Cruz, Santa Rosa and San Miguel islands | **outside** (8–29 km) | The polygon is official and excludes the mainland. Boats from these ports reach the islands, so it stays listed |
| same four | Rock crab viscera advisory, islands part | "from the waters off Santa Rosa Island and the Northern Channel Islands" | **unresolved** | "Waters off" the islands has no stated boundary. CoastWatch cannot say whether it reaches the mainland side of the channel. The North Coast part (CA/OR border to the Mendocino/Sonoma line) is far away |
| Monterey | CDPH SN26-019 and CDFW anchovy restriction | "between Pigeon Point (37° 11.00′ N) and Point Lobos (36° 31.46′ N)"; CDFW: "in Monterey Bay" | **applies**, with "south limit 8.4 km south; check which side you are on" and "offshore extent not stated" | Inside the official latitudes and in Monterey Bay. Fishing south of Point Lobos is outside |
| Shelter Cove | CDFW commercial rock crab closure | "from the Mendocino/Humboldt county line (40° 00.00′ N) to Cape Mendocino (40° 30.00′ N)" | **applies**, with "south limit 2.8 km south" | The port is at 40.026° N. Waters south of 40° 00′ N are outside |
| Crescent Beach | CDFW Humboldt razor clam closure (and CDPH SN24-003, once published) | "remains closed in Humboldt County" | **outside** (28 km) | Crescent Beach is in Del Norte County. The Del Norte closure and CDPH SN26-020 apply there instead |
| Any point on the water | County and area notices | No offshore limit stated | **unresolved** | Unchanged: unresolved up to 30 km |

**Counts across the 26 locations:** 51 apply, 4 unresolved, 6 outside. The geographic tier still matches the pipeline's own port relations for all 22 ports (0 mismatches), and its "nearby" relations are exactly the outside tier.

**Omission safeguard:** the agency box always lists CDFW's and CDPH's own pages and says that automatic matching may miss a notice. It links to all active notices in CoastWatch's list. When agency pages have changed since the last review, a note at the top of the section says a notice may have been issued or lifted since.

## 5. Screenshots and the three review passes

**Pass 1 (layout):**
- The header was crowded and the summary half off the phone's first screen ([before](my-coast/img/pass1-port-593-390.jpg)).
- The satellite line disagreed with the satellite section.
- "Not reviewed" was repeated on every notice.

**Pass 2 (interaction and science):**
- The filter moved into Official information.
- The summary now uses the observation section's own reading.
- "1503 days ago" became "over 4 years ago (date)".
- The small map's credits start folded.
- The copy-link icon is no longer the pin.

**Pass 3 (consistency and width):**
- Six tabs fit at 320 px ([320](my-coast/img/port-593-320.jpg)).
- The masthead fits at 1024 px with "My Coast".
- A point no longer repeats its coordinates.
- Search ranks names that start with the text first.

**After (production data):**

| Place | Desktop | Phone |
|---|---|---|
| Santa Cruz | [1440](my-coast/img/port-593-1440.jpg) · [1280](my-coast/img/port-593-1280.jpg) · [1024](my-coast/img/port-593-1024.jpg) · [full page](my-coast/img/port-593-1440-full.jpg) | [390](my-coast/img/port-593-390.jpg) · [320](my-coast/img/port-593-320.jpg) |
| Monterey (pier no longer measures toxin) | [1440](my-coast/img/port-550-1440.jpg) | [390](my-coast/img/port-550-390.jpg) |
| Crescent City (closure; no station within 40 km) | [1440](my-coast/img/port-201-1440.jpg) | [390](my-coast/img/port-201-390.jpg) |
| Eureka | [1440](my-coast/img/port-220-1440.jpg) | [390](my-coast/img/port-220-390.jpg) |
| San Pedro (Southern California) | [1440](my-coast/img/port-770-1440.jpg) | [390](my-coast/img/port-770-390.jpg) |
| Port Hueneme (ambiguous associations) | [1440](my-coast/img/port-605-1440.jpg) | [390](my-coast/img/port-605-390.jpg) |
| Clam Beach (harvest site) | [1440](my-coast/img/site-clam-beach-1440.jpg) | [390](my-coast/img/site-clam-beach-390.jpg) · [full page](my-coast/img/site-clam-beach-390-full.jpg) |
| Point in Monterey Bay (model value in its own cell) | [1440](my-coast/img/point-36.900_-121.950-1440.jpg) | [390](my-coast/img/point-36.900_-121.950-390.jpg) |
| Stale forecast (test data) | [1440](my-coast/img/stale-port-593-1440.jpg) | — |
| Shelter Cove (closure limit 2.8 km away) | [1440](my-coast/img/port-215-1440.jpg) | [390](my-coast/img/port-215-390.jpg) · [official section](my-coast/img/port-215-390-coast-official.jpg) |
| Crescent Beach (Humboldt closure outside its area) | [1440](my-coast/img/site-crescent-beach-1440.jpg) | [390](my-coast/img/site-crescent-beach-390.jpg) |
| Moss Landing (the only port with a toxin value in its own cell) | [1440](my-coast/img/port-592-1440.jpg) | [390](my-coast/img/port-592-390.jpg) |

**After the association audit (2026-10-12, production data):** the three notice groups at Port Hueneme ([official section, 390](my-coast/img/port-605-390-coast-official.jpg)). Forecast availability where only the bloom grid has a value: Monterey ([1440](my-coast/img/port-550-1440-coast-model.jpg)). Where neither has a value: Santa Cruz ([390](my-coast/img/port-593-390-coast-model.jpg)). The Santa Cruz, Monterey, Port Hueneme, Crescent City and Clam Beach images above were retaken after the audit.

Every location was also captured at 1280, 1024 and 320 px. No page error occurred and nothing scrolls sideways at any size.

**Contrasting cases covered:**

| Case | Location |
|---|---|
| Relevant closure | Crescent City, Clam Beach |
| No recent toxin measurement | Crescent City, Port Hueneme, Monterey |
| Model nearby but no value for the waters | every port except Moss Landing |
| Missing satellite | most ports, this week |
| Stale forecast | test data |
| Ambiguous association | Port Hueneme, Santa Cruz |

## 6. Tests, CI and performance

**Tests:**
- **Unit:** 207 pass, 17 of them new for applicability, availability, places and links. One checks the geographic tier against the pipeline's own port relations for every fixture port.
- **Browser:** 201 pass, 14 new. They cover:
  - deep links and Back;
  - keyboard and map picking;
  - the filter;
  - home port;
  - unknown links;
  - a stale run;
  - navigation;
  - the phone first screen;
  - 320 px;
  - axe on desktop and phone.
- **Lint and types:** clean.

**Existing tests that changed:**
- Two asserted that My Coast did not exist. They now assert it exists and that every navigation entry has a page.
- One asserted five phone tabs; it now expects six.

**Performance** (median of 3; production data; phone run with the CPU slowed 4×):

| | Useful information on screen | LCP | CLS | Filter response | Map share of first screen |
|---|---|---|---|---|---|
| Desktop 1440 | 0.39–0.59 s | 68–76 ms | 0.029–0.046 | 76 ms | 51% |
| Phone 390 | 0.94–1.30 s | 140–144 ms | 0.007–0.014 | 130–137 ms | 0% (directly below the summary) |

First-load JavaScript: `/coast` 145 kB; the map page grows from 164 kB to 166 kB (navigation entry and port-panel link).

## 7. Source-coverage limits

- C-HARM has toxin values for 12.6% of coastal water within 1 km of shore (doc 27). At 21 of 22 ports the waters' own cell has no value.
- Toxin samples within 40 km: none for 9 of the 26 locations. Several stations stopped measuring toxin (Monterey Wharf since 2022) or measure *Pseudo-nitzschia* only.
- PSP and shellfish-tissue toxin results are not in CoastWatch. CDPH's PDFs are linked, not ingested.
- Satellite coverage at the coast is often missing (fog and cloud).
- Notice offshore extent is unstated for most notices.

## 8. Needs human review

1. The notices themselves: the registry is still unverified (`cwp review-official`, by the owner). **Urgent:** CDPH lifted SN26-018 (Monterey County bivalves) on 2026-10-12 in SN26-021; CoastWatch shows it as active until the lift is recorded. The next data refresh will flag SN26-021 as an unreviewed item.
2. The associations in §4: none is reviewed. The audit above is CoastWatch's reading of the official wording, not a review.
3. The species tiers and wording: independent scientist review (outreach drafts are ready and not sent).
4. Stakeholder interviews for the hypotheses in §2.

## 9. Recommendation

**GO to merge into `main` after your review, and NO-GO for public release.** It is safe to merge because:
- it changes no data;
- it never shows a value that is not real, and never states that waters are safe or open;
- every association is labelled unreviewed.

The public-release blockers in doc 26 still apply, and the §4 associations should be reviewed before a public launch.
