# Final scripted walkthrough (2026-10-17)

**A scripted expert walkthrough, not a usability study. No fishermen or other real users took part.**

A Playwright script followed known paths through the visible UI. It:
- counted actions by keystroke-level rules;
- recorded what was on screen at each answer.

The results show:
- the shortest path;
- what breaks.

They do not show whether a person would find a path, understand the answer or trust it.

## Setup

- **Data:** the live dataset, pipeline run 39002736197. The C-HARM run is from Oct 14 (marked failed / not current, as the app says). Satellite data runs to Oct 15; HF radar to Oct 16, 7 PM PDT.
- **Builds:** v4 at commit `bf9f46638`, and the public v2 build.
- **Viewports:** desktop 1440×900, phone 390×844, small phone 320×640.
- **Harbours:**
  - Monterey 550;
  - Crescent City 201;
  - Eureka 220;
  - San Diego 880;
  - Shelter Cove 215, the sparse one (nearest shore station 78.7 km, no buoy within 60 km).
- **Scale:** 10 tasks, 174 runs, 2 per cell.
- **Comparison with v2:** run for tasks 1–7 only. v2 has no Research sampling tools, field pack or offline mode, so tasks 8–10 have no v2 equivalent.

## Results (median actions; ranges cover the harbours)

| Task | v4 desktop | v4 390 px | v4 320 px | v2 desktop | v2 phone |
|---|---|---|---|---|---|
| 1 Latest observations | 4–5 | 5 | 5–6 | 4–6 (water temperature has no time) | 5 |
| 2 Waters from the map | 3 | 22 | 33 | 3–5 | not possible (no map) |
| 3 Home port, save and return | 6 | 6 | 6–7 | 24 | 28 |
| 4 Currents vs chlorophyll | 7 | 10 | 12 | 8 | 12 |
| 5 Satellite coverage | 4 (wrong date, H2) | 5 | 5 | 7 | 9 |
| 6 Read a notice's wording | 6 | 8 (overlapped, H3) | **failed (H3)** | 6 | 7–8 |
| 7 Trip checklist | 8–11 | 7–12 | 12–16 | 7–9 | 11–13 |
| 8 Research satellite + radar point | 10–12 | 18–21 | 23–26 | – | – |
| 9 Download field pack | 2 (23 MB, verified) | 2 | 2 | – | – |
| 10 Reopen with networking off | 9 | 12 | 17 | – | – |

**Task 10, the offline reopen:** 6 of 6 runs passed.
- Method: browser restarted, offline, dead proxy, HTTP cache emptied first, so only the pack could answer.
- 0 requests answered by the network.
- 104–120 requests answered from the pack.
- 66–71 refused (the service worker tries the network first; nothing leaves the device).

## Defects and their status

| | Defect | Status |
|---|---|---|
| H1 | Radar-hour chips dated by UTC day ("7 PM PDT Oct 17" for Oct 16) | Fixed `fd50e0347`. Instants are dated by Pacific day everywhere (unit + e2e) |
| H2 | Multi-sensor coverage said "overpass of Oct 15" for pixels from Oct 10–15 | Fixed. Names the span of days its pixels come from (unit + e2e) |
| H3 | Phone notices panel opened under the harbour sheet; failed at 320 px | Fixed. The header stacks above the sheet; e2e at 390 and 320 px checks the panel is top-most and its wording opens |
| H4 | Choosing waters on My Coast's map: map 5,300+ px down; the new briefing stayed scrolled to the bottom | Fixed. "Choose waters on the map" at the top on phones; a new place opens at the top of its briefing (e2e) |
| M1 | "N notices cover these waters" named 3 and dropped the rest silently | Fixed: "· and N more, all in Notices" (unit) |
| M3 | One observation "2 days ago" in one row, "3 days ago" in another | Fixed. Ages over 36 h count Pacific calendar days (unit) |
| — | (found while fixing H3) "Back to the briefing" sat over the disclosure row | Fixed. On phones it sits in the search stack (e2e, 1440 and 320 px) |
| — | (from the screenshots) Hour chips too narrow for "10:00 AM PDT"; the labels ran together | Fixed `5ffe70868`: a chip is at least as wide as its label (e2e; fails on the old build) |
| — | (from the screenshots) The closure line printed "(831) 649-2883: +18316492883" | Fixed `62abdc980`: the number is shown once (e2e) |
| M2 | Research: picking another station leaves the grid map where it was | Fixed `9b60c80`: a new station moves the map to it and clears the old point, line and area (e2e) |
| M4 | The latest shore sample is dated three ways (Research list, briefing, Explore) | Fixed `9b60c80`: one date everywhere (the map label's variables), with its year when not this year (e2e) |
| M5 | Currents/chlorophyll stamps are below the fold in the layer panel | Fixed `9b60c80`: the stamps and coverage come before the legend; both times above the fold at 1440×900 (e2e) |
| M6 | On phones, reaching layers means closing the harbour; the count falls from "N here" to the statewide count | Fixed `9b60c80`: layer chips at the top of a place's sheet; a count with no place says "in CA" (e2e) |
| M7 | Research on phones: tools 2,600–3,150 px down (18–26 actions) | Fixed `9b60c80`: "Record · Statistics · Map tools" jump links at the top; the tools are one tap away (e2e) |
| M8 | A map-chosen point's first statement has no date | Fixed `9b60c80`: each listed notice carries "(since <date>)" (e2e) |
| L1–L9 | Small tap targets (32–39 px), label collisions, a truncated subtitle, dates without a year, offline banner over a pill at 320 px, extra taps past sheets, ~70 refused requests per offline reopen | Open (v2 has more small targets: 10–14 per screen) |

The scripts, every screenshot and the per-run results are kept outside the repository with the
acceptance evidence. The PR links to the report.
