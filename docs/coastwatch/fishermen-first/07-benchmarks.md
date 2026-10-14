# Before-and-after: six scripted task walkthroughs

**These are scripted simulations in a browser, not tests with fishermen.** They measure whether the information needed for each task is reachable, how many steps it takes, and what a person would read first. They cannot measure comprehension, trust or real-world usefulness: that is what the pilot in [03](03-testing-kit.md) is for.

## Method

| | |
|---|---|
| **Before** | `d414dd63` (#41 as of the checkpoint), unmodified |
| **After** | `599d2d18` (#41's push: season status first, the HAB briefing, no chapters) **plus** this branch's "Since you last checked" block, mounted under the answer line in Coastal mode (integration contract §1) |
| Data | The committed fixture dataset (frozen production structure; CoastWatch's agency check dated Oct 8). Ocean conditions replayed from the recorded 2026-10-13 NOAA responses. **No live third-party request.** |
| Device | Chromium as a phone: 390 × 844, touch, first visit unless the task says otherwise |
| Script | [`scripts/fishermen-bench.mjs`](../../../coastwatch-web/scripts/fishermen-bench.mjs). Three runs per version; median time. Taps, typing, pages and scrolls were identical in all three runs |
| Success | A written rule per task (in the script), applied identically to both versions |
| Time | Scripted speed, no reading time: useful only to compare the two versions |

## Results

| Task | | Success | Taps | Typed | Scrolls to see the answer | Pages | Sources outside CoastWatch still needed | Time (median, scripted) |
|---|---|---|---|---|---|---|---|---|
| **T1** Monterey commercial: tomorrow's conditions, cold first visit through the front door | before | ✓ | 5 | 8 | 0 | 2 | 0 | 4.9 s |
| | after | ✓ | **4** | 8 | 1 | 2 | 0 | 4.7 s |
| **T2** Crescent City commercial Dungeness: notices and restrictions | before | ✓ *but first line misleading* | 3 | 0 | 0 | 1 | 0 | 3.0 s |
| | after | ✓ | **2** | 0 | 0 | 1 | 0 | 2.9 s |
| **T3** Clam Beach: latest official razor-clam information | before | ✓ | 3 | 0 | 0 | 1 | 0 | 4.1 s |
| | after | ✓ | **2** | 0 | 1 | 1 | 0 | 3.8 s |
| **T4** Santa Cruz: recent HAB activity | before | ✓ | 0 | 0 | 0 | 1 | 0 | 0.2 s |
| | after | ✓ | 0 | 0 | 0 | 1 | 0 | 0.4 s |
| **T5** Returning after several days: what changed? | before | **✗ not possible** | — | — | — | 1 | **3** (CDFW health advisories, CDPH releases, Whale Safe page, compared from memory) | — |
| | after | **✓** | **1** ("Continue with Commercial fishing · Dungeness crab") | 0 | 0 | 1 | **0** to find the changes (1 to confirm, linked) | 2.1 s |
| **T6** Researcher: the same observations and their provenance (Research mode) | before | ✓ | 0 | 0 | 1 | 1 | 0 | 1.5 s |
| | after | ✓ | 0 | 0 | 1 | 1 | 0 | 1.7 s |

**Tasks completed: 5 of 6 before, 6 of 6 after.** Taps fall in T1, T2 and T3 (one each) and T5 becomes possible. There is no regression in success. The cost is one scroll in T1 and T3 (below).

## What a person reads first, and where it could mislead (reviewer's judgement on the captured screens)

| Task | Before | After |
|---|---|---|
| T2 | The first line: *"No notice in CoastWatch's list covers these waters for Dungeness crab."* CDFW's "Season is closed" was one tap away. **Could be read as "open"**. [screenshot](img/bench-before-1-T2.jpg) | The first line: *CDFW: "Commercial Fishery: Season is closed. Recreational Fishery: Season is closed."* (#41's I1). The "no notice" line follows it. [screenshot](img/bench-after-1-T2.jpg) |
| T5 | Nothing tells a returning crabber that anything changed. [screenshot](img/bench-before-1-T5.jpg) | *"Since you last checked · Oct 1, 9:00 AM PDT"*, then *"CDFW now says: … Season is closed"* with the earlier wording quoted. It says the page gives no date for the change and when CoastWatch saw it, links the CDFW page, then *"CoastWatch has not checked the agencies' pages recently"* (true for the fixture). [screenshot](img/bench-after-1-T5.jpg) |
| T3 | Both razor-clam notices, with "Since May 2, 2024" and agency links | The same, after the species choice. [screenshot](img/bench-after-1-T3.jpg) |
| T4 | "Toxin last measured at Santa Cruz Wharf (2.2 km), 13 days ago" | "*Pseudo-nitzschia* 4,900 cells/L at Santa Cruz Wharf (2.2 km), 13 days ago" (#41's newest-evidence rule). Each line has a date. "Cells" is not toxin; #41's card says so below. |

**Staleness and coverage were visible in both versions.** Every answer line carries an age or date, and gaps are hatched. **After** adds "CoastWatch last checked the agencies' pages <time>" to the first screen and, on return, a "has not checked recently" item when that check is more than 2 days old.

**Next action.** Before, it came in a folded "Before you go" chapter. After, it is #41's next steps, plus the change list's source link on each item.

**Useful with no HAB measurements (Crescent City, where no station is within 60 km).** Yes in both. After, the agency status leads, the gap lines are hatched, and the change list still works, because it compares agency information, not only samples.

## Costs and remaining problems (honest list)

1. **The first-visit line costs about 110 px on a phone.** It moves "Kind of fishing" from y = 487 to y = 597 and the Dungeness chip from y = 615 to y = 724. Both stay above the tab bar, which starts at y = 780.
   - This is the main cause of the extra scroll in T1 and T3. #41's layout (the choices below the answer) is the other cause: after tapping a choice, the answer above it scrolls off.
   - Already reduced from a 380 px card to one line plus actions.
   - **To test in the pilot:** whether the line is noticed at all on a first visit, or should appear only from the second visit.
2. **Time is scripted.** A person reads, so real times are longer. Only the differences between versions are meaningful, and they are small except in T5.
3. **T5 depends on the device.** A different phone, a cleared browser or private browsing starts as a first visit, and the block then says nothing.
4. **"What changed" is limited to what CoastWatch records.** An agency change that CoastWatch has not yet transcribed appears only as "page changed" (when the page hash moved). Nothing appears if CoastWatch could not read the page; that case is listed as an outage.
5. **None of this shows that fishermen would use it.** Pilot tasks T-status, T-changed and T-gap (03 §3) test that, against their usual tools.
