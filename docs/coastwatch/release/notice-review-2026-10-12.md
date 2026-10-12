# Official notices: review worksheet (2026-10-12)

**For the owner to work through personally.** Nothing in the registry has been changed, verified, lifted or expired in preparing this sheet, and the app still shows every notice as **not verified**. This sheet supersedes the [2026-10-10 checklist](../p3/regulatory-review-2026-10-10.md). Since then, the Del Norte razor clam closure (CDFW) and warning (CDPH SN26-020) have been added to the registry as unverified records (commit ef7beec2).

**Registry:** [`data/curated/official_notices.json`](../../../data/curated/official_notices.json), 10 records and 4 "no closure" statements.
**Last review:** none by a person. It was transcribed by an AI assistant on 2026-10-08 19:32 UTC.

## How to use this sheet

1. Open each source link yourself. The automated results below only say whether the quoted text is still on the page; they are not a review.
2. For each record, tick the boxes, or write what has to change.
3. Make any edits to the registry yourself (or ask for them in a PR), then run the pipeline tests (`uv run pytest tests/test_official.py`).
4. When every record has been checked, run from `pipeline/`:
   ```
   uv run cwp review-official --reviewer "<your name>" --confirm
   ```
   This records your name, the time and new page fingerprints. The app then shows the notices as reviewed by you, with the date. Use the name you want to appear publicly.
5. If you stop partway, don't run the command. A partial review must not be recorded as a review.

**Never** mark a record lifted or expired because a date passed or a page changed. Do so only when the agency says so.

## 1. Watched pages (watcher run 2026-10-12 06:50 UTC, read-only)

| Page | Compared with the 2026-10-08 transcription |
|---|---|
| [CDPH shellfish advisories (release list)](https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx) | Changed: release **SN26-020** is new since transcription; it is now in the registry as an unverified record (#7). No other new release. |
| [CDFW health advisories and closures](https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories) | Changed: the razor clam paragraph adds Del Norte County; it is in the registry (#6). |
| [CDFW Whale Safe Fisheries](https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries) | Unchanged. |

## 2. Records

"Quote check" compares each record's `official_text` with the page as fetched on 2026-10-12 06:55 UTC, ignoring spacing and punctuation.

### 1. `cdph-2026-annual-mussel-quarantine`: annual quarantine of sport-harvested mussels (CDPH)
- **Says:** mussels from the whole coast, bays and estuaries are for bait only, from 2026-05-01, "through at least October 31".
- **Sources:** [SN26-008](https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-008.aspx) (quote check: **verbatim**); [SN26-018](https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-018.aspx) and SN26-020 restate the end as "at least October 31".
- [ ] The quarantine is still in effect and "at least October 31" is still the latest statement.
- [ ] Note: after 2026-10-31 this record needs your decision, based on the next CDPH release. It must not expire on its own.

### 2. `cdph-2026-sn26-018-monterey-bivalves`: do not eat sport-harvested bivalves, Monterey County (CDPH)
- **Sources:** [SN26-018](https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-018.aspx) (**verbatim**); [CDPH advisory map](https://experience.arcgis.com/experience/394836318cfe4f7494e1c09097a43559) (an interactive map; it can't be read automatically).
- [ ] The map still shows Monterey County under advisory.
- [ ] No later CDPH release lifts or extends it.

### 3. `cdph-2026-sn26-019-anchovy-central-coast`: do not eat northern anchovies, Pigeon Point to Point Lobos (CDPH)
- **Sources:** [SN26-019](https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-019.aspx) (**74 % of quoted phrases found**; the release prints the degree sign as "⁰" and breaks lines differently).
- [ ] The latitudes 37° 11.00′ N to 36° 31.46′ N match the release.
- [ ] **Sardines?** The CDFW page says "The CDPH has also issued a recreational health advisory, recommending that the public not consume **sardines and anchovies** taken from the area." SN26-019's title names northern anchovies only. Check whether CDPH issued a sardine advisory. If so, it needs its own record, and the Fisheries page's Pacific sardine tier basis ("no current commercial toxin closure is listed") should mention it.

### 4. `cdfw-2026-anchovy-take-restriction-monterey-bay`: no take of northern anchovy for human consumption, Monterey Bay (CDFW)
- **Sources:** [CDFW status page](https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories) (**verbatim**); [news release](https://wildlife.ca.gov/News/Archive/commercial-and-recreational-take-restrictions-in-place-for-northern-anchovy-in-monterey-bay-due-to-public-health-hazard) (worded differently); [memo, PDF](https://nrm.dfg.ca.gov/FileHandler.ashx?DocumentID=247331) (not read automatically).
- [ ] Effective 2026-09-11 and the area match the memo.
- [ ] Still in effect on the status page.

### 5. `cdfw-2024-razor-clam-humboldt`: recreational razor clam fishery closed, Humboldt County (CDFW)
- **Sources:** [CDFW status page](https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories) (**verbatim**, current wording); [OEHHA memo, PDF](https://oehha.ca.gov/sites/default/files/media/2024-10/razorclamrecreationalfisherymemo050224.pdf) (not read automatically).
- [ ] Closed since 2024-05-02; still closed.
- [ ] SN26-020 mentions a CDPH "razor clam advisory for Humboldt County". The registry has the CDFW *closure* but no CDPH Humboldt *advisory*. Decide whether that advisory needs its own record.

### 6. `cdfw-2026-razor-clam-del-norte`: recreational razor clam fishery closed, Del Norte County (CDFW) (new, unverified)
- **Sources:** [CDFW status page](https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories) (**verbatim**).
- [ ] Closed 2026-10-09. Confirm it against a primary document if one exists (CDFW news release or OEHHA memo), and add it as a source.
- [ ] Don't confuse it with the page's older headline "Del Norte County Razor Clam Fishery Opens After One Month Closure" (a previous event).

### 7. `cdph-2026-sn26-020-razor-clam-del-norte`: do not eat sport-harvested razor clams, Del Norte County (CDPH) (new, unverified)
- **Sources:** [SN26-020](https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-020.aspx), dated October 9, 2026 (**92 % of quoted phrases found**). The quote joins two sentences that are not adjacent in the release.
- [ ] Area, species (razor clams only, sport-harvested) and toxin (domoic acid) match.
- [ ] Optional: mark the gap in `official_text` with " … " so it isn't read as one passage.

### 8. `cdfw-rock-crab-commercial-40n`: commercial rock crab closure, 40° 00′ N to 40° 30′ N (CDFW)
- **Sources:** [CDFW status page](https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories) (**verbatim**); start date not published (CDFW line (831) 649-2883).
- [ ] Still listed. If you call the line, record the start date and the source in `effective_date_note`.

### 9. `cdph-rock-crab-viscera-north-coast-nci`: do not eat rock crab internal organs, CA/OR border to the Mendocino/Sonoma line and the Northern Channel Islands (CDPH, quoted by CDFW)
- **Sources:** [CDFW status page](https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories) (**verbatim**, including the agency's spelling "interal"); the CDPH release list does not carry this wording.
- [ ] Find the originating CDPH release, add it as a source, and confirm the advisory is still in effect.

### 10. `cdph-nci-bivalve-special-advisory`: bivalve special advisory, Northern Channel Islands (CDPH)
- **Sources:** [CDPH advisory map](https://experience.arcgis.com/experience/394836318cfe4f7494e1c09097a43559) (not read automatically).
- [ ] The map still shows the special advisory.

## 3. "No closure" statements (shown as agency statements, not as safety)

| Statement | Source | Quote check |
|---|---|---|
| No marine-toxin closures, recreational Dungeness crab | [CDFW](https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories) | present |
| No marine-toxin closures, commercial Dungeness crab | same | present |
| No marine-toxin closures, spiny lobster | same | present |
| Dungeness crab 2026-27 season closed (all zones, not a toxin closure) | [Whale Safe Fisheries](https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries) | present |

- [ ] Each statement still reads as quoted.

## 4. Anything missing?

- [ ] Read the CDPH release list from SN26-020 back to the start of the year; every release still in effect has a record.
- [ ] Read the whole CDFW status page; every closure or take restriction listed there has a record.
- [ ] Call or check the CDPH biotoxin line, 1 (800) 553-4133, if anything is unclear.

When everything above is ticked, run the command in "How to use this sheet".
