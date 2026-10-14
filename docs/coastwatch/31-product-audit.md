# 31 — Product audit: why CoastWatch is not yet useful or delightful

**Date:** 2026-10-12/13. **Method:** the full site on `main` (`ce68a541`) with production data, viewed as a first-time visitor at 1440 × 900 and 390 × 844, plus measured counts of text, caution phrases and controls. Screenshots: [redesign/img](redesign/img). This is deliberately critical: it judges the product by what a fisherman, harvester or port manager can do with it, not by how much engineering it contains.

## The verdict

CoastWatch is accurate, careful and well tested, but it is built like a **data catalogue with warnings**, not a **coastal briefing**. Every page leads with what CoastWatch is *not* (not verified, not a forecast, not safe, not open) before it says anything useful. The map leads with a model raster that, on most days, colours the entire bay the same pink. Nothing on any page answers the question a stakeholder arrives with: *what applies to my waters, what is known there, and what should I check before I go?*

## Measured (production data, 2026-10-12, Monterey)

| Page | Words on the first screen (desktop / phone) | Words on the page | "not verified" | "does not mean … safe/open" | Controls on the first screen (desktop) |
|---|---|---|---|---|---|
| Ocean Map | 281 / 169 | 281 | 3 | 1 | 34 |
| My Coast (Monterey) | 319 / 163 | 1,549 | **7** | 2 | 33 |
| Bloom Intelligence | **562** / 288 | 2,109 | 3 | 1 | **41** |
| Fisheries | 463 / 302 | **2,350** | 3 | 1 | 23 |
| About | 332 / 168 | 857 | 2 | 1 | 16 |

The redesign's target for a stakeholder's first screen is about 120 words and 15 controls or fewer. Every CoastWatch page is two to five times denser than that.

## The first 5, 10 and 30 seconds

| | Ocean Map | My Coast | Bloom | Fisheries |
|---|---|---|---|---|
| **5 s** | A pink rectangle over the sea; a cream warning bar; a forecast panel. Read: "the bay is bad?" | A port name and a search box; four status lines | A notices banner, then a station list, a mini-map and a headline | A notices banner, then a headline about statewide revenue |
| **10 s** | The legend says 0–100 % probability of particulate domoic acid above 500 ng/L. The panel says both "Current · issued 1 day ago" and "Last update failed … showing the last successful run" | The summary reads like a status report ("3 notices tied to these waters, 1 statewide · not verified") | Five equal metric tiles; a log-scale chart | Three large dollar figures and a stacked bar chart |
| **30 s** | Still no answer to "can I fish/harvest here, and what should I check?" | The notices (good) under seven repetitions of unverified status, then four scientific sections of equal weight | Requires knowing what pDA, seriata/delicatissima groups and log scales mean | Requires understanding that statewide revenue says nothing about this week |

## What is wrong, by severity

1. **Correctness first.** A known lifted advisory (CDPH SN26-018, lifted 2026-10-12) was presented as in effect. *Fixed separately in PR #40 (discrepancy state; no record edited).*
2. **No answer to the core question.** No page combines *my waters + my species + my date* into "what applies, what is known, what to check next".
3. **The map is dominated by a model layer.** C-HARM probabilities are uniformly high offshore on most days, so the default view is a flat pink field that hides the geography and reads as a hazard map, which it is not. The basemap is attractive but is shown only through gaps in the forecast.
4. **Caution is repeated instead of placed.** The site-wide status line, the masthead pill, each card, each section and each footer restate the same caveats. Repetition teaches readers to skip all of them, including the one that matters.
5. **Five equally prominent scientific categories.** My Coast gives official, measured, forecast, satellite and currents equal weight. For a fisherman, the official notices and the next steps matter most; the rest is context.
6. **Research vocabulary in stakeholder views.** C-HARM, cells, thresholds, pDA, seriata/delicatissima groups and log scales appear on first screens.
7. **The notices drawer is a document, not a list.** Eleven notices are shown as cards of equal weight, sorted by agency, not by relevance to the reader, and preceded by a verification box.
8. **Fisheries economics don't connect to a decision.** Statewide landed value 2015–2024 is valuable research context but answers no question someone planning a trip has.
9. **Inconsistent freshness language.** "Current", "issued 1 day ago" and "Last update failed" appear together for the same layer.
10. **Sparse and stale nearshore data with no ocean conditions.** No wind, waves, tides, water temperature or marine weather, the first things a fisherman checks, although free, reliable, official sources exist (NWS, NDBC, CO-OPS).
11. **Visual inconsistency.** Cream warning boxes, dashed cards, uppercase eyebrows, pills, chips and badges coexist without a clear hierarchy; information density varies widely between pages.

## What to keep

- The data pipeline, provenance and freshness model; the verification and discrepancy logic.
- My Coast's place model (port ≠ fishing grounds; points on the water; notice applicability from official geography only).
- The honest "no forecast here / nearest cell" treatment, the station history, the CSV links.
- Accessibility work (axe-clean pages, keyboard paths, 320 px support).

## What to remove, collapse, move or visualise

| Today | Proposal |
|---|---|
| Caveat repeated in every component | One status line site-wide, plus one precise caveat where it changes meaning (e.g. "not a forecast for these waters") |
| Forecast raster on by default | Basemap-first map; forecast an opt-in layer at a readable opacity with exact values on hover/tap |
| Five equal sections in My Coast | A briefing: place and freshness → official status → next steps → conditions → details (collapsed in Coastal mode, open in Research mode) |
| Notices sorted by agency | Sorted by relevance to the chosen place and species; one-line rows; detail on demand |
| Statewide revenue as a top-level page for everyone | Research mode; Coastal mode shows season context and notices for the reader's species |
| Bloom metric wall | Coastal mode: "latest samples near you" in plain language with age; Research mode: the full station analysis |
| No ocean conditions | Marine forecast, nearest buoy, tide predictions, each with its issue/observation time and source |

## Why it is not delightful (honest summary)

It asks the reader to do the synthesis. A product people return to does the synthesis for them, carefully, and shows its working on request. CoastWatch shows its working first and the synthesis never.
