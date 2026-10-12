# 27 — Coastal intelligence: nearshore gaps, fishermen-first design, and what can be solved

**Status (2026-10-12):** research and design for review. **No forecast value, mask, dataset or production data changed. Nothing new is predicted.**
- Two low-risk improvements are in separate PRs: the guided tour (#33) and the county layer (#34).
- Everything else here is measurement, research or proposal.

**Bottom line:**
- C-HARM, the only gridded nearshore toxin product, has **toxin values for 12.6% of California's coastal water within 1 km of shore, and 31.5% within 3 km**.
- Satellite chlorophyll near shore is mostly lost to fog in autumn.
- Only **1 of 22 ports** has a toxin measurement from the last 30 days within 25 km.
- **No public, validated source** fills the nearshore toxin gap, and nothing in the literature supports stretching offshore C-HARM values to the shore.
- The useful near-term product is therefore not a smoother map. It is a **port-centred summary** with four parts:
  - the official notices that apply, by species and by who they apply to;
  - the age of the nearest measurement;
  - the distance to the nearest real model value;
  - an honest account of what is unknown.

How the evidence was gathered:
- **Measured by CoastWatch, with scripts and data in [coastal/](coastal/):** §1 and §8.
- **Research appendices:** [A, sources and literature](coastal/appendix-a-sources-and-literature.md), [B, user needs](coastal/appendix-b-user-needs.md) and [C, county boundaries](coastal/appendix-c-county-boundaries.md). These were compiled with an AI research assistant. Every claim carries a URL, and claims seen only in secondary sources are marked UNVERIFIED. I re-checked the key numbers quoted below against the primary sources, for example the C-HARM v1 AUCs in Anderson et al. 2016.

---

## 1. Nearshore coverage and missingness audit (measured)

**Method** ([audit.py](coastal/analysis/audit.py)):
- Reference geography is ETOPO 2022 at 15″ (about 460 m).
- Coastal water means sea, bays and estuaries connected to the open ocean, from 32.53° N to 42.0° N. The Delta and Suisun Bay are excluded.
- Each water pixel is assigned to the C-HARM 0.03° cell that contains it, using the pipeline's own rule, and measured as distance to the nearest land.
- Results are percentages of water **area**.
- Run: the production nowcast valid 2026-10-10, plus 40 more ERDDAP nowcasts (the 1st and 15th of each month, Nov 2024 to Sep 2026).

### 1.1 How much water has a C-HARM value (H1, H2)

| Distance from shore | Water area | With a bloom (*Pseudo-nitzschia*) value | With a toxin (pDA) value |
|---|---|---|---|
| 0–1 km | 2,204 km² | 48.0% | **12.6%** |
| 1–3 km | 4,613 km² | 72.2% | 40.5% |
| 3–5 km | 4,384 km² | 92.1% | 82.2% |
| 5–10 km | 11,100 km² | 99.2% | 98.8% |
| 10–25 km | 33,364 km² | 100% | 100% |
| **Cumulative, within 1 / 3 / 5 / 10 km** | | 47.9 / 64.3 / 75.2 / 87.1% | **12.6 / 31.5 / 51.3 / 75.0%** |

**Stable through time.** Across the 41 nowcasts from Nov 2024 to Oct 2026, toxin coverage within 1 km never moved outside 12.3–12.6%, and within 10 km outside 73.8–75.0%. The gaps are structural, not weather or outages.

### 1.2 By region (0–10 km; toxin value / bloom value / breakdown of the rest) (H3)

| Region | Toxin value | Bloom value | Toxin mask | Shoreline cell | Ocean cell without any value |
|---|---|---|---|---|---|
| North Coast (40–42° N) | 75.6% | 94.2% | 18.6% | 2.0% | 3.8% |
| Mendocino/Sonoma | 76.3% | 89.2% | 13.0% | 2.3% | 8.5% |
| SF Bay area, outer coast | 81.4% | 89.5% | 8.0% | 2.7% | 7.8% |
| **Inside San Francisco Bay** | **3.7%** | 8.8% | 5.1% | 12.1% | **79.1%** |
| Monterey Bay | 73.3% | 87.2% | 14.0% | 2.0% | 10.8% |
| Central Coast | 75.2% | 88.2% | 13.0% | 2.5% | 9.3% |
| Southern California (incl. islands) | 81.0% | 92.4% | 11.4% | 1.7% | 5.9% |

**Within 1 km**, toxin coverage is 7.5% on the North Coast and the Central Coast, 9.0% in Monterey Bay, 9.2% in Mendocino/Sonoma, and about 21% around San Francisco and in Southern California. San Francisco Bay itself is essentially outside the model.

Maps:
- [whole coast](coastal/img/coverage_map.jpg)
- [Monterey Bay](coastal/img/coverage_monterey.jpg)
- [San Francisco Bay](coastal/img/coverage_sfbay.jpg)
- [Humboldt](coastal/img/coverage_humboldt.jpg)
- [Southern California](coastal/img/coverage_socal.jpg)

Colours: blue = toxin value; amber = toxin mask; grey = shoreline cell; red = ocean cell without any value.

### 1.3 The ten kinds of gap, kept apart

| # | Kind | Measured where | Inherent, avoidable, or CoastWatch? |
|---|---|---|---|
| 1 | **Land vs ocean at 3 km**: water inside a C-HARM cell that is mostly land | 22% of water within 1 km; about 0 beyond 3 km | Inherent to a 3 km grid. CoastWatch already clips drawing to the shore (doc 24) and never extends values onto land or water |
| 2 | **Model domain limits** | 0.0% of water 25–200 km offshore lacks a value | None along California |
| 3 | **Documented toxin mask**: bloom value but no toxin value | 35% of water within 1 km, 32% at 1–3 km, 10% at 3–5 km | Inherent to C-HARM. The producer does not document why (appendix A §1.1). It is not avoidable with any other product |
| 4 | **Ocean cells with no prediction at all**: the model's own land and bay mask | 30% within 1 km; 79% inside SF Bay | Inherent (WCOFS/C-HARM grid). Not avoidable |
| 5 | **Satellite cloud, fog and missed overpasses** | OLCI, Oct 5–10: median **5% of water within 1 km valid on a given day**; 55% never valid in 6 days ([sat_audit.py](coastal/analysis/sat_audit.py)). Appendix A's transects, Jul–Oct: about 10% of days within 1 km | Inherent: weather, plus coastal adjacency and quality flags. Not avoidable; never interpolated |
| 6 | **Upstream outages** | C-HARM: **286 of 1,440 days** since 2022-11-01 have no nowcast (about 20%; 121 in 2024, 42 in 2025, 44 so far in 2026) | Inherent to the producer. The pipeline shows the last real run with its age |
| 7 | **Old but valid observations** | Station recency, §8 | Inherent to sampling and lab latency (1 week to more than 2 months). Shown with dates |
| 8 | **Tile and rendering failures** | Doc 24: 0 forecast pixels over land, 0 hidden gaps, at most 0.005% ocean values not drawn (18 views); bundled fallback when the basemap fails | CoastWatch. Already fixed and tested |
| 9 | **Gaps in observing-system coverage** | 17 CalHABMAP stations; none within 40 km of Crescent City; Monterey Wharf pDA ended 2022-08; 7 stations report *Pseudo-nitzschia* but no toxin | Inherent to the monitoring network |
| 10 | **No validated nearshore estimate exists** | All of the above inside about 3 km | Requires new science (§3, Track 3) |

**Pipeline parity:**
- The production grid's no-value cells are **identical to ERDDAP's for the same day**, with 0 differing cells across all three variables. CoastWatch's pipeline creates no gaps.
- Satellite parity is checked on every run by `cwp verify-satellite`.

---

## 2. Data sources: what can actually help near shore

Full table and citations: [appendix A §1](coastal/appendix-a-sources-and-literature.md). In summary:

| Source | Variable | Resolution / cadence / latency | Nearshore verdict |
|---|---|---|---|
| **C-HARM v3.1** (NOAA) | Modelled probabilities: PN > 10⁴ cells/L, pDA > 500 ng/L, cDA > 10 pg/cell | 3 km, daily, 0–3 day leads, archive from Nov 2022 with about 20% of days missing | **Toxin probability offshore only.** Only v1 has a published skill test (Anderson et al. 2016): pDA AUC 0.77 at Santa Cruz, **0.04 at Stearns**; PN at or below chance at those piers |
| **CalHABMAP piers** (SCCOOS ERDDAP) | Measured pDA, dDA, tDA, PN by size group, chlorophyll | 17 points, about weekly, lab latency 1 week to more than 2 months | **Nearshore toxin, at points only** |
| CalHABMAP SPATT (3 piers) | Time-integrated dissolved DA | Points | Nearshore toxin, integrated |
| CDPH biotoxin monitoring | Shellfish toxins (DA, PSP) | Not published as data; StoryMaps run about 5–6 months behind; advisories by county | **Decisions, not data.** PSP is missing from CoastWatch entirely |
| CA IFCB network | Imaged plankton (PN counts) | 9 sites, hourly; dashboard only, no documented feed or licence | Bloom proxy; worth asking SCCOOS for a feed |
| CDPH volunteer phytoplankton | Qualitative PN | Rolling 6 weeks, machine-readable | Bloom proxy; needs archiving |
| **Sentinel-3 OLCI** | Chlorophyll (biomass) | 300 m, daily, often clouded near shore | Bloom proxy, intermittent |
| **Sentinel-2 MSI** | Reflectance; chlorophyll only via research algorithms | 10–60 m, 5-day revisit | **No operational California product.** The surf zone is resolvable but uncalibrated. Not a toxin signal |
| VIIRS, MUR/GOES SST, WCOFS, HF radar, upwelling indices | Physical context | Various | Context only |
| CDFW, CDPH, OEHHA notices | Regulatory decisions | Event-driven, mostly PDF and HTML | The authority. Needs human curation |

**Key conclusions:**
- **No other product supplies nearshore toxin information as a field.** The literature says chlorophyll and multispectral sensors carry no Pseudo-nitzschia or domoic acid signal (Smith et al. 2018, appendix A §1.3).
- **Interpolating offshore C-HARM to the shore has no validation, and three sources argue against it:**
  - Frolov et al. 2013: nearshore biomass is decoupled from biomass 2–4 km offshore.
  - Anderson et al. 2016: nearest-pixel matchups at piers are rough approximations.
  - SCCOOS bulletins: pier data and open-coast C-HARM disagree.
- **Higher-resolution satellite imagery is not higher-resolution toxin information.**

---

## 3. Three tracks for better coastal information

### Track 1: integrate what exists (preferred, near term)

Build trustworthy coastal-location summaries from:
- official notices, mapped to ports by a reviewed area table;
- pier measurements with their age;
- regional C-HARM, shown with the distance to the nearest real cell;
- the last clear satellite view;
- HF-radar currents as context.

This needs **no new data and no prediction**. §4 sets out the design.

### Track 2: additional validated sources (no new predictions)

| Candidate | Value | Effort / risk | Recommendation |
|---|---|---|---|
| CalHABMAP **SPATT** (ERDDAP) | Integrated toxin at 3 piers | Low (same server) | Prototype after the review |
| **IFCB** classified PN counts | Hourly bloom proxy at 9 sites | Needs a public feed and a licence from SCCOOS | Ask SCCOOS (outreach) |
| CDPH volunteer phytoplankton | Coast-wide qualitative PN | Low; must be archived ourselves | Prototype as "observations", clearly qualitative |
| **CDPH biotoxin results (DA and PSP in shellfish)** | The quantity that drives decisions | Not public as data | Ask CDPH; essential for any shellfish view |
| CDFW BIOS Dungeness zones (ds3120), 3 nm line | Correct marine geometry for notices | Low | Use instead of county polygons for marine notices |
| Sentinel-2 chlorophyll | 10–60 m bloom proxy | High (atmospheric correction, no validation) | Research only |

### Track 3: novel nearshore prediction (research proposal; not built)

**Is there enough labelled data?** It's marginal ([appendix A §2.4](coastal/appendix-a-sources-and-literature.md), ERDDAP counts):
- CalHABMAP has **4,268 pDA samples at 10 stations** since 2014, but only **155 at or above 0.5 ng/mL**, clustered in 2014–17 and 2025.
- Monterey Wharf stopped reporting pDA in 2022.
- Drivers differ by region (Sandoval-Belmar 2023) and by event (Trapp 2025).

**The study design, if pursued:**
- **Targets:**
  - P(pDA at the pier ≥ 0.5 ng/mL) one and two weeks ahead;
  - a secondary target of PN bloom ≥ 10⁴ cells/L.
- **Features** (only as available at issue time):
  - lagged pier values;
  - C-HARM at the nearest valued cell;
  - OLCI/VIIRS chlorophyll and SST;
  - upwelling indices (CUTI/BEUTI);
  - HF-radar alongshore flow;
  - river discharge;
  - distance offshore and bathymetry.
- **Baselines** that must be beaten:
  - persistence (last value);
  - climatology by month and station;
  - C-HARM nearest-valued-cell probability;
  - logistic regression on lagged values.
- **Validation:**
  - leave whole years out (and 2015 as an event);
  - leave whole stations out, to test geographic transfer;
  - as-of joins for lab latency, so there is no leakage;
  - PR-AUC, Brier score and reliability diagrams;
  - calibration with an isotonic or Platt method, fitted on training years only;
  - uncertainty from bootstrap over years.
- **Sampling bias:** sampling is not missing at random (sampling increases during events). Model the sampling indicator and report skill only on scheduled samples.
- **Evaluation limit:** skill can be measured **only at the 10 toxin-measuring piers**, not in the water between them. Any output is a pier-point outlook, never a map.
- **Go/no-go:** proceed beyond research only if it beats persistence on PR-AUC in held-out years at most stations and is reliably calibrated; and then only after independent HAB-scientist review.
  - It would be labelled "experimental" and kept apart from agency forecasts and notices.
  - It would never be used in a safety or harvest context.
- **Assessment:** feasible as a research study at one to two weeks for well-sampled piers (Santa Cruz, Cal Poly, Stearns, Santa Monica, Scripps, Newport). **Not** feasible as a coast-wide nearshore field.

---

## 4. Fishermen-first product: "My Coast"

**Who needs what** ([appendix B](coastal/appendix-b-user-needs.md); [D] = documented in a published source, [H] = hypothesis needing interviews):
- **Commercial crabbers** [D]:
  - Their decisions hinge on CDFW/OEHHA closures, delays and evisceration orders, by biotoxin management area.
  - The 2015–16 event cost up to 72% of a season, concentrated in Crescent City, Fort Bragg and Moss Landing.
  - They asked for clear protocols and notifications, not predictions.
- **Recreational fishers** [D/H]: recreational and commercial status diverge. Example: a commercial closure alongside a recreational "don't eat the viscera" advisory.
- **Shellfish harvesters** [D]:
  - The mussel quarantine and county advisories apply to recreational harvest.
  - Certified commercial growers are tested separately.
  - **PSP is decisive and not in CoastWatch.**
- **Port managers** [H]: fleet-wide status and history.
- **Researchers** [D]: provenance, downloads and methods. The source CSV link is in #32.
- **Documented harms:**
  - Advisories depress demand.
  - A third-party "risk" map near an open fishery can cause harm.
  - Confusing advisories with closures is common.

**What CoastWatch can reliably provide today:**
- notices per port, by species and fishery type, linked and dated, with verification status;
- pier measurements with their age;
- distance to the nearest real model value, plus a regional C-HARM summary labelled as a model;
- the last clear satellite view;
- next steps with agency contacts.

**What needs regulatory expertise or new data:**
- marine geometry for notices;
- federal versus state waters;
- what an evisceration order means for a landing;
- PSP;
- CDPH test results.

**Smallest useful set** (for a port page reached from the map's existing port selection):
1. **Notices that may apply here**, grouped by **who** (commercial, recreational, sport-harvest shellfish, consumption) and **species**, each with the agency, date and source, and "Not verified" until reviewed. Statewide agency statements (crab, lobster) are folded below. "No notice found for X" is shown as **not** an all-clear.
2. **Measured nearby**: the nearest pier or piers with distance, last toxin value and date, and age in days. When there is none: "No station within 40 km; nearest is … last sampled …".
3. **Model context**: "nearest cell with a toxin value is *N* km away", plus the regional median and range within 15 km, labelled as a model probability and never compared with the pier value.
4. **Satellite**: last clear view and chlorophyll, labelled as biomass, not toxin; or "no clear view in 6 days".
5. **Before you fish or harvest**: the agency pages and phone lines.

What the design leaves out on purpose:
- green or "safe" states;
- composite scores;
- "low risk" wording;
- any notice-to-port matching by string. The prototype showed why: a county-name match put the Northern Channel Islands advisory on Santa Cruz, because Santa Cruz Island is named in its area. Matching must use a reviewed table.

**High-fidelity mockups (production data, 2026-10-12):**
- **Crescent City** ([desktop](coastal/img/mycoast-crescent-city-1280.jpg), [phone](coastal/img/mycoast-crescent-city-390.jpg)):
  - razor clam closure and do-not-eat advisory;
  - **no station within 40 km**;
  - C-HARM median pDA probability **1%**;
  - no clear satellite view.

  This is the case that shows why a low model value must never read as safety.
- **Santa Cruz** ([desktop](coastal/img/mycoast-santa-cruz-1280.jpg), [phone](coastal/img/mycoast-santa-cruz-390.jpg)):
  - anchovy restriction and advisory;
  - pier pDA 0.20 ng/mL, 12 days old;
  - model median **75%**, with the nearest valued cell 7.2 km out.

  The measurement and the model disagree, and both are shown, uncompared.
- **Morro Bay** ([desktop](coastal/img/mycoast-morro-bay-1280.jpg), [phone](coastal/img/mycoast-morro-bay-390.jpg)): the piers measure PN only, with no toxin.

**Journeys:**
- A commercial crabber at Crescent City opens the port and sees, in order:
  1. CDFW razor clam closure (recreational);
  2. statewide Dungeness statements (no toxin closures; season closed);
  3. "no station within 40 km";
  4. links to the CDFW line.
- A recreational clammer sees the CDPH do-not-eat advisory first.

**Measuring success with fishermen** (appendix B §8). Interview 15–20 people across the North, Central and South Coast: harbour districts, Sea Grant extension, fishermen's associations and shellfish growers. Targets:

| Measure | Target |
|---|---|
| Testers who understand CoastWatch does not declare an area safe | ≥ 90% |
| "Safe" inferences in think-aloud tests | 0 |
| Median lag from an official publication to its display | < 24 h |
| Time to find a species' status for a port, compared with agency sites | ≥ 50% faster |
| Repeat use during an event season | (to be measured) |

---

## 5. Guided onboarding: implemented in #33

The tour has nine steps on the real controls:
1. what CoastWatch shows;
2. choosing a place;
3. the C-HARM forecast and its limits;
4. satellite observations;
5. what empty water means;
6. a port and the measurements near it;
7. the official notices;
8. Bloom Intelligence and Fisheries;
9. "not a safety clearance".

How it behaves:
- It is not modal. The notices drawer opens above it, focus is never trapped, and Escape, X, Skip or Finish close it.
- It shows progress, works on desktop, tablet and phone, and respects reduced motion.
- It is offered once per device, and can be reopened from the rail, the phone help button or `/?tour=1`.
- It builds on the existing first-visit card rather than competing with it.

Tests: 8 e2e, including axe on desktop and phone; 3 unit tests on the copy rules; the full suite passes.

## 6. County geography: implemented in #34

- **Source:** all 58 counties from California Department of Technology / CDTFA boundaries (public domain, edited 2026-09-21).
- **Accuracy:** coverage-simplified to within 205 m; 72 kB gzip.
- **What's drawn:** only the borders between counties, so there is never a second coastline. Coastal counties are emphasised, and lines and names are optional.
- **Wording:** an explicit "not a marine or regulatory boundary" note.
- **Marine boundaries:** none are drawn.
  - CDPH advisories are worded by county but never state an offshore extent.
  - CDFW closures follow latitude lines; "40°00′ N" is drawn as a latitude, not the county edge, which is 160 m away.
  - Crab zones (BIOS ds3120) and the 3 nm line are documented for later use (appendix C).

## 7. Missing-data presentation: alternatives (screenshots; not shipped)

| Variant | What it does | Screenshots |
|---|---|---|
| **v0, current** | Fine dots on every water area without a value | [Monterey](coastal/img/gap-v0_current-monterey.jpg) · [SF Bay](coastal/img/gap-v0_current-sfbay.jpg) · [phone](coastal/img/gap-v0_current-m390.jpg) |
| **v1, quieter dots and a coverage edge** | Dots at 35%; a thin line where toxin values end | [Monterey](coastal/img/gap-v1_quiet_edge-monterey.jpg) · [SF Bay](coastal/img/gap-v1_quiet_edge-sfbay.jpg) · [phone](coastal/img/gap-v1_quiet_edge-m390.jpg) |
| **v2, classified** | Toxin-mask cells hatched amber ("bloom value, no toxin value"); other no-value water a light wash; coverage edge | [Monterey](coastal/img/gap-v2_classified-monterey.jpg) · [SF Bay](coastal/img/gap-v2_classified-sfbay.jpg) · [phone](coastal/img/gap-v2_classified-m390.jpg) |

**Recommendation: v2 (classified), with a legend.** It is the only variant that distinguishes the model's toxin mask from water outside the model, and it keeps every gap visible.
- **Prototype issue:** v2's hatch is drawn from whole 3 km cells, so it crosses the shore (visible at Point Reyes). A shipped version must use the same water clip as the data (doc 24).
- **Satellite:** keep "no observation" transparent with a cloud/fog legend entry, a different state from "no model value".
- **Never:** a low probability in place of missing data, or interpolation.
- The inspector's "why is this empty" readout already exists and should name the class.

## 8. Ports with nearby measurements (H5)

From [ports_audit.py](coastal/analysis/ports_audit.py), data as of 2026-10-12, for 22 ports:

| Within 25 km of the port | Ports |
|---|---|
| Toxin (pDA) measurement ≤ 30 days old | **1** (Santa Cruz) |
| Toxin measurement ≤ 90 days old | 6 |
| PN count ≤ 30 days old | 5 |
| Any sample ≤ 30 days old | 7 |
| Nearest C-HARM toxin value ≤ 3 km | 5 |
| Nearest C-HARM toxin value ≤ 5 km | 15 |

The North Coast piers last sampled in March and April 2026. Southern California toxin results lag sampling by about two months.

## 9. Scientific risk assessment

| Risk | Severity | Mitigation |
|---|---|---|
| A low C-HARM value read as "safe" next to an active closure (Crescent City) | **High** | Notices first; no green states; "model, not seafood"; tour step 9; tests on safety wording |
| Interpolating or extending offshore values to the shore | High | Not done; prohibited in design; nearest-cell distance shown instead |
| Notice matched to the wrong port | High | Reviewed area-to-port table; marine geometry from CDFW sources; human review (`cwp review-official`) |
| Shellfish view without PSP | High | No shellfish "status"; say PSP is not included; link CDPH |
| Stale measurements read as current | Medium | Ages in days on every value; freshness classes |
| An experimental model confused with agency forecasts | Medium | Track 3 only after validation and review; separate namespace and visual kind |
| A third-party map harming open fisheries' markets | Medium | No hazard scores; frame as "where to check" |

## 10. Ready to ship, needs review, research

| Ready to ship (after your review) | Needs review or decisions | Longer-term research |
|---|---|---|
| #32: 320 px masthead; source CSV link | v2 missing-data presentation (needs the water clip) | Track 3 pier-point DA outlook study |
| #33: guided tour | "My Coast" port page (reviewed notice-to-port table; scientist review) | Sentinel-2 nearshore chlorophyll |
| #34: county layer | Track 2: SPATT, volunteer phytoplankton, IFCB and CDPH data requests | PSP integration (needs CDPH data) |
| | Marine geometry for notices (BIOS crab zones, 3 nm) | |

## 11. Tests and performance (this round)

| PR | Tests | Performance | CI |
|---|---|---|---|
| **#32** | unit 183, e2e 178 (incl. 10 release tests), lint | — | Green |
| **#33** | unit 182, e2e 176, lint | Map first-load JS 161 → 164 kB | Green |
| **#34** | unit 183, e2e 171, lint | +72 kB gzip data, fetched once; map JS unchanged | Running at the time of writing |

**Overlap:** #33 and #34 both append to `content/copy.ts`, so the second to merge needs a trivial conflict resolution.

## 12. Recommended roadmap

1. **Now:**
   - Human notice review (B1).
   - Merge #32–#34 after your review.
   - Send the scientist outreach.
2. **Next (Track 1, about two PRs):**
   - A reviewed notice-to-port and species table.
   - A port page ("My Coast") with the five blocks above.
   - The v2 missing-data presentation with the water clip.
   - Test with 5–10 fishers and harvesters before any public release.
3. **Then (Track 2):**
   - Add SPATT and volunteer phytoplankton as labelled observations.
   - Request IFCB and CDPH biotoxin data.
   - Add CDFW crab zones for notice geometry.
4. **Research (Track 3):** a pre-registered pier-point DA outlook study with held-out years and stations, reported whether or not it succeeds, and reviewed before anything is shown.

**What CoastWatch can responsibly show near shore today (H8):**
- official notices;
- dated pier measurements;
- the distance to the nearest real model value, with a labelled regional summary;
- the last clear satellite view;
- explicit unknowns.

**What needs new science (H9):** any nearshore toxin field, any forecast at unmonitored ports, and PSP.

**Additions that only make the map look more complete (H7):**
- smoothing or interpolating C-HARM;
- satellite gap-filling;
- coloured risk zones by county;
- a single coastal hazard score.

None of these is proposed.

**Additions that answer a real fishing or regulatory question (H6):**
- notices by species and fishery type;
- measurement age;
- distance to real model data;
- agency next steps;
- the tour's limits step.
