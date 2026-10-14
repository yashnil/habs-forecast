# 28 — My Coast: MVP design brief and notice-applicability model

**Status:** design and feasibility (approved scope, 2026-10-12). Nothing here is implemented.

**What the MVP will not include:**
- nearshore interpolation;
- new probabilities;
- composite or safety scores;
- any statement that an area is open or seafood is safe.

**Evidence:** [doc 27](27-coastal-intelligence.md) (coverage audit, sources, user needs), its [mockups](coastal/img/) and this brief's [example data](coastal/data/mycoast.json).

## 1. What the MVP is

A **port-centred summary**, reached by choosing a port on the Ocean Map or from the place list. It has five blocks, always in this order:

| # | Block | Content | Never |
|---|---|---|---|
| 1 | **Official notices that may apply here** | Notices related to the port by the reviewed applicability table (§3), grouped by **who** (commercial / recreational / sport-harvest shellfish / anyone eating) and **species**. Each shows agency, action, dates, area wording, source link, "Not verified" status and **why it is listed**. Statewide agency statements (crab, lobster) are folded below. **Nearby areas with separate notices** and **notice limits near the port** get their own sub-list. | "Open", "safe", "clear"; hiding a notice because no species filter matches |
| 2 | **Measured nearby** | The nearest CalHABMAP stations within 40 km. For each: distance, latest pDA (with qualifier) and PN counts, each with date and **age in days**. A value more than a year old is marked "no measurement in the last year"; if a station no longer measures toxin, it says so. Otherwise: "No station within 40 km; nearest is … last sampled …". | Treating the pier as the port's water or seafood; a combined index |
| 3 | **Model availability (C-HARM)** | "The nearest cell with a toxin value is **N km** from the port" first. Then, labelled as a model, the regional median and range within 15 km for the chosen day, with issue and valid dates. If no valued cell lies within 15 km, no summary. | Interpolating toward the port; comparing with the pier value; colour scales implying safety |
| 4 | **Recent observations** | Last clear OLCI view within 5 km: date, age and median chlorophyll ("biomass, not toxin"). Otherwise "no clear view in N days (cloud, fog or no overpass)". Optionally, HF-radar current direction as context. | Filling cloud gaps; implying toxin |
| 5 | **Before you fish or harvest** | Agency pages and phone lines (CDFW closure line, CDPH shellfish line). | Recommendations |

**Freshness labels:**

| Measurement | Labels |
|---|---|
| Pier toxin | ≤ 14 days "recent"; 15–90 days "older"; > 90 days "old" |
| Satellite | ≤ 2 days "recent" |
| C-HARM | Uses the run's existing freshness rule |

Ages are always shown as numbers, never as colours alone.

## 2. Realistic examples (production data, 2026-10-12)

| | **Crescent City** (Del Norte) | **Monterey** (Monterey) | **Santa Cruz** (Santa Cruz) |
|---|---|---|---|
| Notices that apply (current rules) | Mussel quarantine (statewide); CDFW razor clam closure and CDPH razor clam advisory (Del Norte); CDPH rock crab viscera advisory (port within 38°46′–42°00′ N) | Mussel quarantine; CDPH bivalve advisory (Monterey County); CDPH anchovy advisory and CDFW anchovy restriction (port within 36°31′–37°11′ N) | Mussel quarantine; CDPH anchovy advisory and CDFW anchovy restriction |
| **Nearby, separate notices** (proposed) | Humboldt County razor clam closure and advisory (area begins 26 km south) | — | **Monterey County bivalve advisory**: the area begins 22 km away across the bay, and Santa Cruz boats harvest there |
| **Notice limits near the port** (proposed) | — | **Point Lobos (36°31.46′ N), the anchovy notices' south limit, is 8 km south of the port** | — |
| Measured nearby | No station within 40 km; nearest Trinidad Pier, 77 km, last sampled Apr 17 (178 days) | Monterey Wharf, 0.6 km: **pDA last measured Aug 31, 2022**; PN last Aug 27, 2025; "no longer measured here" | Santa Cruz Wharf, 2.2 km: pDA 0.20 ng/mL on Sep 30 (12 days); PN 4,900 cells/L |
| Model availability | Nearest toxin cell 2.4 km; median 1% within 15 km for Oct 11 | Nearest toxin cell 6.6 km; median 73% | Nearest toxin cell 7.2 km; median 75% |
| Recent observations | No clear OLCI view within 5 km in 6 days | Oct 5 (7 days): 0.79 mg/m³ | Oct 10 (2 days): 1.41 mg/m³ |

**What the examples show:**
- **Crescent City:** a **1%** model value sits beside an active razor clam closure, so block 1 must come first and the model block must never read as reassurance.
- **Monterey:** the nearest pier no longer measures toxin.
- **Santa Cruz:** the measurement and the model disagree, and both are shown, uncompared.

Mockups: [Crescent City](coastal/img/mycoast-crescent-city-1280.jpg) · [Monterey](coastal/img/mycoast-monterey-1280.jpg) · [Santa Cruz](coastal/img/mycoast-santa-cruz-1280.jpg) · [Eureka](coastal/img/mycoast-eureka-1280.jpg) · [Morro Bay](coastal/img/mycoast-morro-bay-1280.jpg). Each has a phone version (`-390.jpg`).

## 3. Notice applicability: the problem and a reviewed model

**Today:**
- The pipeline relates notices to ports by four rules in `pipeline/coastwatch_pipeline/sources/port_intel.py`:
  - statewide;
  - the port's county is in the notice's `area.counties`;
  - the port's latitude lies within a notice's official limits;
  - a notice's official polygon lies within 60 km.
- This is structured, not text matching. A first draft of the doc 27 mockups did use text matching, and got three ports wrong.

**What the current rules get wrong or leave out:**
1. **Ports are not fishing grounds.** A Santa Cruz boat works Monterey County waters, but "same county" hides the Monterey bivalve advisory.
2. **Notices with two parts.** The rock crab viscera advisory covers the North Coast band *and* the Northern Channel Islands. Only the band is encoded, so Santa Barbara and Ventura are not linked to its islands part.
3. **Limits close to a port.** Monterey is 8 km from the anchovy notices' south limit. "Inside" is true but needs a warning that the limit is near.
4. **CDPH's areas are CDPH's own polygons**, not CDT counties. The Channel Islands are separate areas.
5. **Offshore extent** is unstated for most notices, and some CDFW actions apply to state waters only.
6. **No person has approved any port–notice link.**

### Proposed model: `data/curated/notice_applicability.json` (reviewed)

```jsonc
{
  "schema_version": 1,
  "areas": [
    { "area_id": "cdph:humboldt", "kind": "cdph_advisory_area",
      "definition": { "service": "https://services2.arcgis.com/wi1yEacfYjH5viqb/arcgis/rest/services/California_Coastal_Counties/FeatureServer/3", "where": "TITLE = 'Humboldt County'" },
      "offshore_extent": "not stated", "source_version": "layer last edited 2020-08-26; read 2026-10-12" },
    { "area_id": "band:pigeon-point_point-lobos", "kind": "latitude_band",
      "definition": { "lat_south": 36.524333, "lat_north": 37.183333, "named_limits": ["Pigeon Point", "Point Lobos"] },
      "offshore_extent": "not stated" },
    { "area_id": "cdph:northern-channel-islands", "kind": "cdph_advisory_area", "definition": { "where": "TITLE = 'Northern Channel Islands'" } },
    { "area_id": "statewide-coast", "kind": "statewide" }
  ],
  "notice_areas": [
    { "notice_id": "cdph-rock-crab-viscera-north-coast-nci", "area_ids": ["band:or-border_mendocino-sonoma", "cdph:northern-channel-islands"] }
  ],
  "port_areas": [
    { "port_code": 593, "area_id": "cdph:monterey", "relation": "harvest_waters",
      "method": "reviewer", "evidence": "Santa Cruz fleet fishes Monterey Bay south of the county line (to be confirmed with the harbor district)",
      "status": "proposed", "reviewed_by": null, "reviewed_at": null },
    { "port_code": 550, "area_id": "band:pigeon-point_point-lobos", "relation": "contains",
      "method": "latitude", "metric": { "km_to_nearest_limit": 8.4 }, "status": "proposed" }
  ],
  "notice_facets": [
    { "notice_id": "cdph-2024-sn24-003-razor-clam-humboldt", "audience": ["sport_harvest"], "species_group": ["shellfish"], "species": ["razor clam"] }
  ]
}
```

**Relations a port can have to an area:**

| Relation | Meaning | Shown as |
|---|---|---|
| `contains` | The port is inside the area | "Applies here" |
| `harvest_waters` | Reviewer judgment: the port's fleet commonly harvests there | "Applies to waters this port's boats use" |
| `near_limit` | Inside, within 10 km of a stated limit | "Applies here — a limit is N km away" |
| `nearby` | Not inside, within 30 km | "Nearby, separate notice" |

**Rules:**
- **Proposals are made by geometry only.** CDPH polygons use point-in-polygon and distance; latitude bands compare latitudes. Text is never matched.
- **Review:** each port–area link is `proposed` until a person approves it. A command such as `cwp review-applicability`, mirroring `cwp review-official`, would record the reviewer and date. **Never filled in automatically and never with your name by default.**
- **Unreviewed links:** until reviewed, the app shows proposed links with "Listed by location; not yet reviewed", alongside the notice's own "Not verified".
- **Tests:**
  - every active notice has at least one area;
  - every port links to `statewide-coast`;
  - no free-text matching anywhere;
  - the three examples above produce exactly the lists in §2.
- **Facets** (audience, species group, species) are curated per notice, so the filter chips (Crab / Finfish / Shellfish; Commercial / Recreational) never hide a notice the reader needs. A notice outside the chosen filter is still counted: "2 more notices for other species".

## 4. Decisions before implementation

1. **Harvest waters:** should a port's notices include areas its fleet uses (reviewer-defined), or only the area that contains it? Recommendation: include them, labelled, after confirming with harbour districts.
2. **Unreviewed proposals:** show them labelled, or hide them until reviewed? Recommendation: show them labelled, since hiding a possibly applicable notice is the worse error.
3. **Nearby radius** (30 km proposed) and **near-limit distance** (10 km proposed).
4. **Shellfish:** PSP is not in CoastWatch. Should the page carry a fixed line, "Shellfish toxins (PSP) are tested by CDPH and not shown here"? Recommendation: yes. Showing CDPH's 30-day results PDF would be a separate decision after the data request (see the private research note).
5. **Who reviews the applicability table:** you; and should harbour districts or Sea Grant be asked to confirm the harvest-waters links?
6. **Ports in scope:** the 22 CDFW ports. Or should named harvest sites (for example Clam Beach, Crescent Beach) be added for recreational clammers?

## 5. Sequencing (when approved)

1. **Applicability table:**
   - generate proposals;
   - you review them;
   - add tests.

   This is pipeline and data only, with no UI. Effort: small.
2. **Port page:** the five blocks in the existing port inspector (desktop) and sheet (phone), using the current data plus the reviewed table. Effort: medium.
3. **Usability test** with 5–10 fishers and harvesters (doc 27 §4 metrics) before any public release.

**Out of scope (deferred research):**
- the nearshore toxin model (doc 27 Track 3);
- new data sources in production;
- federal-waters rules.
