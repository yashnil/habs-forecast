# M7: Fisheries & Economic Exposure — redesign brief

**Status:** brief only, for review. Nothing is implemented.

**Baseline:**
- the approved design-reset spec §3.3 ([design-spec.md](https://github.com/yashnil/habs-forecast/blob/archive/pr7-design-reset/docs/coastwatch/design-reset/design-spec.md#33-fisheries--economic-exposure));
- the doc 17 audit (§4.3);
- the M5 and M6 design language: the Ocean Map, and Bloom Intelligence as "one station, one timeline".

**Before** (production data, 2026-10-11): [1440](m7/before/1440-fisheries-full.jpg), [phone](m7/before/390-fisheries-full.jpg).

## 1. What the page must do

Show what California's toxin-affected commercial fisheries **have landed in past years**, by species group, so that a reader understands what was at stake in earlier seasons. It must do so without suggesting that landings are losses, a forecast of losses, or anything about the current season. The published terminology is the lede:

> Historical fisheries exposure: the reported value of past commercial landings of species that marine toxins can affect. It is not a prediction of losses, not an estimate of harm, and says nothing about any current or future season.

## 2. Where the page is today

The page is still the M3 structure:
- two caveat boxes before any data;
- three KPI cards;
- one single-colour bar series;
- three small per-group bar charts;
- a values table;
- two method cards.

It is the weakest page on the site:
- it reads as a dashboard, not an editorial story;
- the 2015–16 domoic-acid season, the most important moment in the data, is not marked;
- the absence of port-level data takes more space than the data;
- on a phone it is 4,071 px of cards.

## 3. The data (published, `fisheries-*.json`)

| | |
|---|---|
| **Years** | 2015–2024 (10). 2025 is not yet published by NOAA |
| **Scope** | California commercial landings, statewide, by species group and year (all ports and gears) |
| **Groups** | **Tier 1** (toxin closures): Dungeness crab, rock crabs, northern anchovy. **Tier 2** (advisories): California spiny lobster, Pacific sardine, bivalve shellfish (oysters, clams, mussels, scallops) |
| **Values** | Per group and year: pounds, nominal dollars, and 2024 dollars (CPI-U) |
| **State total** | NOAA's statewide total per year, with the count of rows without a value |
| **Withheld** | Confidential landings, aggregated across species by NOAA. Never attributed to any group |
| **Port level** | **Unavailable**, with published reasons: CDFW's MFDE (2021+) requires permission that has not been obtained; the CC0 CALFISH compilation ends in 2019 |

**Example, the latest year in 2024 dollars:**

| Group | Value |
|---|---|
| Dungeness crab | $49.7M |
| Spiny lobster | $20.9M |
| Bivalves | $10.9M |
| Rock crabs | $2.4M |
| Pacific sardine | $0.67M |
| Northern anchovy | $0.63M |

## 4. Direction: editorial data storytelling

The page should read like a well-made data article. M7 keeps the spec §3.3 structure and applies the M6 lessons:
- one hero visual;
- no card chrome;
- caveats next to what they qualify;
- defaults that follow the data;
- performance on a phone.

1. **Official strip.** The notices referenced by the species groups (rock crab, anchovy, bivalves), using the M6 `OfficialStrip` pattern.
2. **Hero.**
   - A serif headline: "What California's toxin-affected fisheries have landed".
   - The terminology sentence as the lede.
   - "Annual data through 2024; 2025 not yet published by NOAA".
3. **Three typographic figures, no cards.**
   - The latest year.
   - The 10-year average, with its low and high years.
   - The share of NOAA's state total, stating the withheld amount.

   Directly under them: "Past landings show what was at stake in earlier seasons. They are not losses, not a forecast of losses, and say nothing about the current season."
4. **One primary chart.**
   - Stacked annual bars by group, tier by hue family (blues for Tier 1, sage for Tier 2), totals on top.
   - A "share of state total" row under the years.
   - The **2015–16 bracket** ("Dungeness season delayed by domoic acid", from the group's published `tier_basis`).
   - The legend doubles as a filter.
   - Controls: Tier 1 or Tiers 1 + 2; 2024 dollars, nominal dollars or pounds.
5. **One table by species group.**
   - Columns: tier, latest value, 10-year average, 10-year sparkbars, share.
   - Groups outside the selected set are dimmed, not hidden.
   - Selecting a row highlights the group in the chart and opens a detail panel with the tier basis, linked official notices, pounds (meat weight for bivalves), NOAA market categories, and the aquaculture caveat for bivalves.
6. **Geography, only where the data allow** (see decision F2). An optional small map of CDFW port areas, showing official geometry only: *where landings come from*, never per-port values that are not published. The absence of port data is **one quiet line** that expands to the published reasons.
7. **How these numbers are made.**
   - Four short columns: source with the NOAA courtesy line, inflation, confidentiality, duplicates.
   - Disclosures for every caveat, the full values table including the withheld row, and the processing steps.
   - The licence.
8. **Phone.** Narrative first; full-width segmented controls; stacked figures; an edge-to-edge chart with two-digit years; a two-column table (value, share); the detail opens on selection; the port note.

**Rules carried over:**
- History is never presented as a forecast. The `Historical` chip and the terminology sentence lead the page.
- Withheld value is never assigned to a group.
- No port split is ever invented.
- The dollar basis is named in every chart subtitle.
- No "loss" wording, except in negated sentences (`assertNoUnsafeClaims`).
- No suggestion that CoastWatch is endorsed by NOAA or CDFW.

## 5. Interpretive limits to state plainly

- Landed value is ex-vessel revenue. It excludes processing, retail, tourism and other economic activity.
- Groups are lower bounds, because confidential landings sit in the withheld row.
- A low year is not evidence of toxin effects, and a high year is not evidence of safety. Markets, regulations, ocean conditions and effort all move landings.
- Bivalve values include aquaculture.
- NOAA revises recent years, and annual summaries are preliminary.

## 6. Decisions for you

The recommended option is in bold.

- **F1. Pounds.** Offer pounds alongside dollars, as a third unit control? **Yes**; pounds avoid inflation questions. Alternatively, dollars only.
- **F2. Geography.** **A small CDFW port-area map with official geometry and no values**, or no map until port-level permission exists.
- **F3. Permission.** Should I draft a request to CDFW for MFDE port-level republication? It would unlock real regional context later. **Draft it as a document only.**
- **F4. Years before 2015.** Extend the series back (FOSS has older years) for context before the 2015–16 event? **Not in M7**; it would be a pipeline change needing its own review.

## 7. Work plan and review standard (as M6)

- About 6–8 commits:
  - lib helpers with tests;
  - the page;
  - the chart and table;
  - phone;
  - browser tests;
  - review passes and report.
- Squash merge.
- Three documented review passes at 1440, 1280, 1024 and 390, on production data.
- A real-browser interaction audit.
- Measured performance against the current page on a throttled phone.
- An axe scan with no serious violations.

**Not in M7:**
- no new data sources or pipeline changes (beyond F3's document);
- no forecasts or loss estimates;
- no deployment;
- no change to the Research Preview demo.
