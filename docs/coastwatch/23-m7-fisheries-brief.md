# M7: Fisheries & Economic Exposure — brief and implementation record

**Status:** implemented on this branch (§8–§14); awaiting the owner's review before merge. §1–§7 are the original brief, kept as written; where the implementation departs from it, §9 says why.

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

## 6. Decisions (asked in the brief; answered 2026-10-11, see §8)

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

## 8. Decisions as approved (2026-10-11)

- **F1 — approved.** Pounds alongside revenue. The page names the unit everywhere and never adjusts pounds or implies price adjustment beyond the CPI-U rescaling it states.
- **F2 — approved with geographic verification.** CDFW publishes its landing ports as **points**, not port-area boundaries. The source is "Marine Landing Data System Ports" (BIOS ds3081, CDFW Marine Region, CC BY 4.0): 112 ports, each assigned to one of 11 port areas, geocoded with a note that "the point may not represent the exact port location". The page therefore draws points, labelled by port area, and **no areas and no values** ([where-landings-are-reported](m7/after/where-landings-are-reported.jpg)).
- **F3 — approved.** The draft request is [m7/cdfw-data-request-draft.md](m7/cdfw-data-request-draft.md). It has not been sent.
- **F4 — deferred.** The series stays 2015–2024, with no pipeline change.

## 9. Pre-build audit: the old page and the data

**The old page** ([before](m7/before/1440-fisheries-full.jpg)):
- two caveat boxes before any data;
- KPI cards;
- one single-colour series with no annotation;
- three small charts, each on its own scale;
- no pounds;
- the absence of port data took more space than the data.

**The data, checked against each proposed encoding:**

| Data property | Consequence for the design |
|---|---|
| 6 groups × 10 calendar years × 3 measures (2024 $, nominal $, pounds) | One stacked annual bar chart, one measure at a time; never two axes |
| Dungeness crab is 80–96% of Tier 1 revenue; the smallest groups are under 1% | A stack cannot compare small groups: choosing a group shows it **alone on its own scale** (the brief's "focused group at the base" was tried and dropped in pass 1, [evidence](m7/review/pass1-rock-crab-sliver.jpg)) |
| In pounds, anchovy and sardine dominate; in dollars, crab and lobster do | Pounds is a first-class unit; the group detail shows revenue and pounds side by side, each on its own scale, with the sentence "separate measures on separate scales" |
| Annual values are **calendar years**; crab seasons span two | The 2015–16 events are dated brackets placed by day within the year slots, and the chart notes say so |
| Bivalve pounds are meat weight and mostly aquaculture | Flagged in the legend, detail and notes when pounds are shown |
| The withheld amount is in NOAA's state total but in no group | Shares are labelled lower bounds; the withheld row stays in the full table |
| Rows without a published value (bivalves, some years) | Left out and counted, never zero; listed in the detail and starred in the table |
| No port-level values | One quiet line plus a disclosure with the published reasons; no split |

**Documented events.** The page marks only the 2015–16 domoic acid closures, from an official state record: the California Legislature's Joint Committee on Fisheries and Aquaculture, *2015–16 Commercial Crab Fishing Timeline* (July 2016).
- **Dungeness crab:** delayed 6 Nov 2015, reopened in stages, statewide 26 May 2016.
- **Rock crab:** closed 6 Nov 2015; on 21 July 2016 it was still closed north of Pigeon Point, so its bracket has an open, dashed end.

Later season delays, such as those for whale entanglement risk, are not HAB events and are not marked. Nothing on the page attributes a change in landings to an event. Unit tests forbid causal wording in event text, and the browser tests check every sentence with all disclosures open.

**Two errors in the published text, now corrected** (approved 2026-10-11). Only the text changed; the numbers are identical, as the regenerated fixtures confirm.
1. **The `withheld[].note`** said confidential landings are "not included in group or statewide values". NOAA's FOSS caveats say "Total landings by state include confidential data and will be accurate", and the pipeline's totals already include them. The note now reads: included in the statewide total, not attributable to any species or group.
2. **The rock crab `tier_basis`** said "north of 40° N". CDFW's health-advisories page (checked 2026-10-11) gives the commercial closure as "from the Mendocino/Humboldt county line (40° 00.00' N. lat.) to Cape Mendocino, Humboldt County (40° 30.00' N. lat.)". The basis now quotes those limits.

Pipeline tests tie both texts to how the totals are built and to the notice limits. A staging run from this branch checks the result on real data. The production dataset changes only at the first scheduled refresh after the merge.

## 10. What was built

1. **Official notices first.** The five notices the species groups reference, collapsed to one line with the "Not verified" badge. This is the same strip as Bloom, now shared.
2. **Opening.**
   - The serif headline "What California's toxin‑affected fisheries have landed".
   - The published terminology as the lede.
   - Freshness, the data window ("2025 not yet published by NOAA") and "tiers await independent scientific review".
   - Beside it, three typographic figures, without cards:
     - the selected year's total;
     - the 10-year average, with its lowest and highest years;
     - the share of all California commercial landings, labelled a lower bound.
3. **The primary figure.**
   - Stacked annual bars, in Tier 1 blues and Tier 2 stone. Teal, violet and amber stay reserved for measured, model and official.
   - Totals on the bars, and the share of the state total under each year.
   - The dated 2015–16 brackets.
   - Year buttons that work by pointer, touch and keyboard (arrows, Home, End).
   - The legend doubles as the group chooser.
   - Controls: Tier 1 or Tiers 1 + 2; 2024 $, nominal $ or pounds.
   - **The year beside the chart:** each group, the total, the state total, the share and the withheld amount. When 2015 or 2016 is selected, it adds the dated record and its source.
   - "Reading this chart" explains calendar years and that the brackets do not say how much of any change the closures explain, with a unit-specific note.
4. **By species group.**
   - One sortable table (`aria-sort`): the selected year, its share, the 10-year average, and a 10-year strip on the group's own scale.
   - Tier 2 rows are dimmed, not hidden.
   - A row opens in place:
     - the published basis;
     - the referenced notices (not verified), linked to the drawer;
     - revenue and pounds side by side;
     - NOAA's market categories;
     - rows without values.
5. **What these figures can and cannot tell you.**
   - **They do not show:** harm (a change is not a loss, and overlap is not cause), profit or income, fishing effort, or anything about a port, county or the current season.
   - **They do show:** what was landed, how large the fisheries are, and when documented events fell.
6. **Where landings are reported.** CDFW's 112 ports by port area on a 14 kB static outline (Natural Earth, public domain). It needs no map library, and it loads only when the section nears the screen. Port-level values are explained in one line plus a disclosure.
7. **How these numbers are made.**
   - Four short columns: source, dollars, confidentiality, species groups.
   - Disclosures:
     - every value, including the state total and the withheld row (built only when opened);
     - the inflation index;
     - the duplicate rows;
     - all caveats;
     - the processing steps.
   - Sources with retrieval times, the 2015–16 source, and the citation and licence.
8. **Link state:** `unit`, `tiers`, `group`, `year`. Unknown values fall back to defaults, and a Tier 2 group in the link brings Tier 2 into the chart.

**Shared changes, safe for all pages:**
- `OfficialStrip` takes record ids and a heading, so Bloom and Fisheries share it.
- `useWidth` moved to `lib/`.
- The unused `BarChart` was removed.
- The header no longer overflows at 768 px.
- At tablet widths, the "not verified" banner had truncated away its "See the list" button; it now has a short tablet line.

## 11. Review passes (production data, run 38140958238)

Captures: [m7/after/](m7/after/) at 1440×900, 1280×720, 1024×768 and 390×844, including filter and table states.

1. **Composition and story** ([pass 1](m7/review/pass1-1440-full.jpg)).
   - The chart was below the first screen, so the figures moved beside the opening and the lede and the "at stake" sentence merged.
   - The geography text column had collapsed; the grid was fixed.
   - Focusing a small group drew a sliver, so a chosen group is now shown alone on its own scale.
   - The rock crab closure was missing from the bracket, so the event became two dated records.
2. **Charts, interaction, accessibility and science.**
   - Phone axis labels and the last total were clipped; fixed.
   - Pale group colours fell under 3:1 in single-group strips, so those use a strong tier colour.
   - The "lower bound" wording now names both reasons: withheld landings and generic categories.
   - Axe flagged nested landmarks and an unfocusable scrolling table; both fixed.
   - The 768 px overflow and the hidden banner button were fixed.
3. **Final desktop and phone review, and regressions.**
   - The notices strip is compact (one line plus a disclosure).
   - A non-breaking hyphen keeps "toxin‑affected" on one line.
   - The share caption is shorter.
   - Bloom's 20-check interaction audit still passes.
- **CI found a real bug, now fixed.** The first CI run caught a phone overflow in the moments before the chart measured its width (the year row was placed in pixels from a 900 px guess). The chart is now sized to its container from the first paint and the years are placed in percentages. A new test loads the page with JavaScript off, which is that unmeasured state; it fails without the fix.
- On a phone, the selected year's total is lifted clear of the neighbouring bars.

**User journeys walked in a real browser:**
- a first-time visitor reading top to bottom;
- one species (rock crab: legend, then its own scale, its row and its notices);
- a researcher comparing years (keyboard through the years, nominal versus 2024 $, pounds, the full table);
- a phone visitor (tap a bar, open a row, no overflow).

**Interaction audit:** 30/30: years, hover, keyboard, legend, Tier 2 rows, sorting, units, the drawer, deep links, garbage parameters, phone tap, reduced motion, and no overflow at 1440, 1280, 1024 and 768 px.

## 12. Tests and performance

| Check | Result |
|---|---|
| Web unit (vitest) | 157 passed (21 new in `m7.test.ts`; the M3 copy test now also checks the new list copy) |
| Browser (Playwright, local Chrome and CI mode) | 144 passed (15 new in `m7.spec.ts`; the M3 Fisheries test ported to the new structure with the same or stricter checks) |
| Pipeline (pytest) | 196 passed; `build_fisheries_geo.py` passes ruff |
| Typecheck, lint, production build | Clean; 0 build warnings |
| Axe (1440, 1024, 390; detail open; pounds) | No violations of any severity |

**Performance:** Fisheries page, production data, local production servers, median of 3; the phone has its CPU slowed 4×.

| | Old page (main) | M7 |
|---|---|---|
| Phone: blocking time | 48 ms | 53 ms |
| Phone: unit switch to visible change | 176 ms | **110 ms** |
| Phone: transfer | 337 KB | 354 KB |
| Phone: page elements | 604 | 613 |
| Desktop: unit switch | 59 ms | 57 ms |

The map and its CDFW points load only near the screen, and the 80-cell values table is built only when opened. No chart or map library is added.

## 13. Final review (2026-10-11)

- **The phone opening is compact, with nothing dropped** ([before](m7/review/phone-first-screen-before.jpg), [after](m7/review/phone-first-screen-after.jpg)).
  - The three figures sit in one row with short captions; their full sentences are one tap away.
  - The notices heading leads with the count, and the "Not verified" badge stays on one line.
  - The meta line still says the species tiers await review.
  - The legend is one row that scrolls sideways.
  - The figures and the chart title are on the first screen. The chart starts at 995 px instead of 1,273 and is fully visible after one scroll.
- **The Bloom audit failure** (one run of 18/20 right after a server restart) did not recur in 14 runs, 10 of them from a cold start.
  - The audit now saves a screenshot, the URL, console errors, failed requests and timing for any failed check.
  - The browser test suite keeps a trace and screenshot for every failure, and CI uploads them.

## 14. Remaining limitations

- **Port-area values** wait on CDFW's answer to the F3 request.
- **Tiers** await independent HAB scientific review.
- **NOAA revises** recent years.
