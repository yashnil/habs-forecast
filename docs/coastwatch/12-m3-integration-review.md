# 12 — M3 integration review

Date: 2026-10-08 (UTC 2026-10-09). Branch `feat/coastwatch-m3-bloom-economics` → `main`. The Next.js website is **not deployed**. Official notices remain **Not verified** (pending human review); no scientific sign-off is claimed; M4 not started.

## 1. Redistribution terms and attribution

**CalHABMAP (all 17 datasets checked individually).** Each dataset's `NC_GLOBAL` attributes were read from `https://erddap.sccoos.org/erddap/info/HABs-<Station>/index.csv`: Trinidad Pier, Humboldt, Humboldt South Bay, Bodega Marine Lab, Bodega Marine Lab Buoy, Tomales Bay Mouth, Tomales Bay Mid-Channel Buoy, Inner Tomales Bay, Santa Cruz Wharf, Monterey Wharf, Morro Bay Front Bay, Morro Bay Back Bay, Cal Poly Pier, Stearns Wharf, Santa Monica Pier, Newport Beach Pier, Scripps Pier.

| Attribute | Value (identical in all 17) |
|---|---|
| `license` | "The data may be used and redistributed for free but is not intended for legal use, since it may contain inaccuracies. Neither the data Contributor, ERD, NOAA, nor the United States Government … makes any warranty … or assumes any legal liability …" |
| `creator_name` / `institution` | CalHABMAP |
| `creator_url` / `infoUrl` | https://calhabmap.org |
| citation / acknowledgement attributes | none published |

CalHABMAP's own data page (https://calhabmap.org/datasites) links each of its stations to exactly these SCCOOS ERDDAP datasets, so this is the program's designated distribution. No separate data-use or citation policy was found on calhabmap.org (about, data, FAQ, resources) or on SCCOOS's HAB and data-access pages. The licence text is ERDDAP's standard wording, applied uniformly by the publisher.

**Conclusion:** free redistribution is permitted. The restriction "not intended for legal use" matches CoastWatch's rule that official agencies decide closures. CoastWatch attributes "California Harmful Algal Bloom Monitoring and Alert Program (CalHABMAP), via SCCOOS ERDDAP" and quotes the licence on the page and in the artifact. No dataset has different terms.

**NOAA Fisheries FOSS landings.**
- NOAA Fisheries copyright policy (https://www.fisheries.noaa.gov/national/about-us/website-policies-and-disclaimers): "Information created by the U.S. government … is not subject to copyright in the United States … we also request that NOAA Fisheries be given appropriate acknowledgement ('Courtesy: National Oceanic and Atmospheric Administration')." **Added** to the artifact's citation (it appears on the page).
- FOSS caveats (https://apps-st.fisheries.noaa.gov/foss/f?p=215:240): only non-confidential statistics are released; confidential landings are "combined with other landings and usually reported as 'Withheld for Confidentiality'"; "Total landings by state include confidential data and will be accurate, but landings reported by individual species may, in some instances, be misleading"; bivalves are meat weight; "Landings do not include aquaculture products except for clams, mussels and oysters"; dollars are nominal ex-vessel ("Users can use the Consumer Price Index … to convert"); data are updated weekly. All are now stated on the page.

**BLS CPI-U** (`CUUR0000SA0`): U.S. Government work, public domain.

**Not used (unchanged):** CDFW MFDE (permission uncertain), CALFISH (download challenge/token), PacFIN direct (terms unverified), CDPH phytoplankton layer (no licence).

## 2. NOAA landings methodology audit

**Duplicate oyster listing: confirmed upstream.** FOSS's own source-species table (`/ods/foss/source_species/`) maps PacFIN (`source_id` 3) code **KSTR "KUMAMOTO OYSTER"** to both TSN 79868 ("OYSTER, PACIFIC") and TSN 79869 ("OYSTER, KUMAMOTO"). It is the **only** one of 455 PacFIN codes mapped to two names. The same landings therefore appear twice in every year: California has the identical pair in all 10 years, and Washington shows the same pair (2023: $1,305,870.82 in both rows). CoastWatch counts the row once and lists the excluded row on the page.

**Statewide totals vs NOAA's published summaries** (Fisheries of the United States 2022, Table 4: [S3 PDF](https://s3.amazonaws.com/media.fisheries.noaa.gov/2025-01/FUS-2022-final3.pdf); FUS 2023, Table 4: [S3 PDF](https://s3.amazonaws.com/media.fisheries.noaa.gov/2026-02/FUS-2023-web.pdf)), California, thousands of dollars:

| Year | NOAA FUS | FOSS, all rows summed | FOSS, duplicate counted once (**CoastWatch**) | Difference vs FUS |
|---|---|---|---|---|
| 2021 | 209,783 (FUS 2022) | 210,456 | 209,713 | −0.03% |
| 2022 | 207,919 (FUS 2022) / 207,849 (FUS 2023) | 211,803 | 207,870 | −0.02% / +0.01% |
| 2023 | 170,330 (FUS 2023) | 175,009 | 170,610 | +0.16% |

NOAA's published state totals equal FOSS with the KSTR duplicate counted once. A plain sum of all rows overshoots by 0.3–2.7%. The residual differences are consistent with FOSS's weekly revisions after each FUS edition, which FUS itself calls "preliminary and subject to revision".

**Change made:** the earlier statewide total also excluded the withheld row. NOAA's state totals include it and NOAA states they are accurate, so the CoastWatch total now **includes the withheld amount**. The withheld amount is still shown separately and never attributed to any species. Before the change, 2024 read $200.0M excluding $10.8k withheld; it now reads $200.0M including it.

**Inflation adjustment.** BLS's series page shows half-year averages (HALF1/HALF2), and the mean of each pair reproduces every annual index the pipeline computes from 12 monthly values (2015 237.017 … 2024 313.689). October 2025 is printed "-(X)" with the footnote "Data unavailable due to the 2025 lapse in appropriations", which confirms 2024 as the base year. 2024-dollar check: Dungeness 2015 = $17,002,331 × 313.689 / 237.017 = $22.50M (tested to the cent).

**Species categorization audit** (every California name 2015–2024 matching crab, anchovy, lobster, sardine, bivalve and related terms):
- Assigned as intended: CRAB, DUNGENESS; CRAB, RED ROCK; ANCHOVY, NORTHERN; LOBSTER, CALIFORNIA SPINY; SARDINE, PACIFIC; OYSTER (Pacific, Eastern, Edible, Olympia); CLAMS **.
- **Bug fixed:** `SCALLOPS **` was not matched because FOSS's generic suffix " **" was allowed only for clams. All bivalve patterns now accept it ($0 in these years, so no published value changed).
- Deliberately unassigned and disclosed as making groups lower bounds: CRABS, DECAPODA (ORDER) ** ($2.9M over 10 years; may include yellow and brown rock crab), MOLLUSKS ** ($0.9M), ANCHOVIES ($0), CRAB, KING **, market squid and sea urchin (not toxin-closure fisheries).
- The bivalve group is mostly **aquaculture** in FOSS and is now labelled so.

**Confidentiality suppression.** Only FOSS's public, already-aggregated withheld row is used. Nothing is estimated or redistributed to species, and port-level data are not used.

**Pipeline fix found during review:** the weekly fisheries refresh would have reused an artifact built by older code. It now rebuilds whenever the pipeline version changes (tested).

## 3. Tests, C-HARM verification and staging checks (final head)

| Check | Result |
|---|---|
| Pipeline offline (`uv run pytest`) | **126 passed**, 1 deselected (live) — new: NOAA FUS reproduction, generic-category matching, attribution/caveats, version-triggered rebuild |
| Live upstream test (`pytest -m live`) | 1 passed |
| Full live pipeline run (local) | all 7 sources updated; `check-published` 30/30 |
| **C-HARM verification** (`cwp verify-charm`, live ERDDAP) | **156/156** comparisons, 0 failures (13 points × 12 layers) |
| Web unit | **68 passed** (new: region bounds equal curated regions) |
| Lint, typecheck, production build | clean |
| End-to-end (4 servers incl. published M2 data; axe; mobile) | **42 passed** |
| Staging data run [37873331347](https://github.com/yashnil/habs-forecast/actions/runs/37873331347) | green; `check-published` 30/30 with run id; fisheries rebuilt by the new code (2022 total $207.87M, NOAA courtesy line present) |
| CI on head | green |

## 4. Frontend against published data (production build)

| Dataset | Result |
|---|---|
| GitHub Pages (M2 data, run 37846066753) | **3/3**: forecast, official notices, port panel; `/bloom` and `/fisheries` show "published before … were added" states |
| Staging (M3 data) | **3/3** |

Schema compatibility: M3 adds only optional manifest fields. The M2 Pages files are committed as fixtures, validated by the pipeline models and the browser schemas, and used by an e2e server.

An earlier staging run passed only on retry. The cause is the documented server fetch cache (stale-while-revalidate, 5 minutes): `.next/cache` still held the staging manifest from before the republish. With the cache cleared, both datasets passed 3/3 with retries disabled.

## 5. Screenshot review (13 images in `m3/`)

| Screenshot | Finding | Action |
|---|---|---|
| 11 (Trinidad) | Station map framed all of California, so the selected station sat on the edge | **Fixed:** the map frames the station's official region and re-frames when the region changes (13 shows Scripps) |
| 04 header | Page-level "Current" badge could be read as describing the selected station | **Fixed:** labelled "newest sample in the network; each station shows its own" |
| 06 | Share tile said the total excluded withheld landings | **Fixed** with the new NOAA-consistent total and wording |
| 01–03, 05, 07–10, 12 | Official first, Not verified visible, reported zeros in a separate lane, not-measured states explicit, model on its own axis, legend on map, mobile layouts without overflow | No change |

Accessibility: axe reports no serious or critical violations on map, bloom, fisheries and sources pages. Charts are keyboard-readable and have data tables; station selection works by keyboard and select.

## 6. Publication exposure check

The data branch receives only `v1/` pipeline artifacts (`guard-publish` OK). The M3 additions:
- `observations-<sha>.json`: public CalHABMAP values under their redistribution licence.
- `fisheries-<sha>.json`: NOAA public, non-confidential statewide statistics, with NOAA's withheld aggregate shown as published. No port-level fields, no synthetic test data, no contact details, no estimates of confidential values.

The official registry is published exactly as in M2: `pending_human_review`, shown as Not verified.

## 7. Merge and production verification

| Step | Result |
|---|---|
| PR [#5](https://github.com/yashnil/habs-forecast/pull/5) checks (pipeline, web) | green; `CLEAN` / `MERGEABLE` |
| Merged with a merge commit | `8d1087a` |
| First production refresh on `main` with the M3 pipeline ([37880564301](https://github.com/yashnil/habs-forecast/actions/runs/37880564301)) | build, publish, check **green**; review issue skipped (official pages unchanged) |
| GitHub Pages deploy, triggered by the refresh ([37880874831](https://github.com/yashnil/habs-forecast/actions/runs/37880874831)) | deploy and check **green**; 0 Node 20 annotations |
| Independent `check-published` on `https://yashnil.github.io/habs-forecast/v1` | **30/30 files**, run id 37880564301, `access-control-allow-origin: *`, `cache-control: max-age=600` |
| Sources in the production manifest | C-HARM unchanged (issued 2026-10-08, verified 156/156 earlier this review); GIBS, ports, official, port intel, **CalHABMAP (17/17 stations updated)**, **FOSS** updated |
| Published content | fisheries 2022 total $207,870,047 (NOAA FUS: $207.9M), 10 duplicate rows excluded, port level `unavailable`; official registry still `pending_human_review` |
| Production build against Pages (now M3 data), retries disabled | **3/3** |

The C-HARM verification step in the production run was skipped by design (`--only-if-updated`; the C-HARM run was unchanged). The same run's values were verified live against ERDDAP earlier in this review: 156/156.

**Status:** M3 is integrated into `main`. The production pipeline and GitHub Pages now publish CalHABMAP observations and statewide historical fisheries exposure as public data. The Next.js website is **not deployed**. Official notices are **Not verified**, and no scientific sign-off is claimed. M4 is not started.

## 8. Remaining limitations

- Official registry awaits human review (`cwp review-official --reviewer NAME --confirm`). HAB scientist review of `/bloom`, the species tiers and the C-HARM context is pending (checklist 08).
- Port-level landings are unavailable until CDFW responds; a draft request is in [`drafts/cdfw-landings-data-request.md`](drafts/cdfw-landings-data-request.md) (not sent).
- CalHABMAP publishes no detection limits or analytical methods. Lab results lag sampling by days to weeks. Northern stations have been inactive since spring 2026, and Monterey Wharf has had no pDA since 2022.
- FOSS is revised weekly and FUS is preliminary, so small differences (≤0.2%) are expected. Group values are lower bounds (generic categories unassigned), and bivalves are mostly aquaculture.
- The BLS v1 API is rate-limited (about 25 requests/day/IP); a refused request keeps the previous fisheries artifact. Fisheries rebuilds at most weekly or on a new pipeline version.
- After a publish, the app can show the previous complete dataset for up to about 15 minutes (5-minute server cache plus 10-minute Pages cache).

## 9. Recommendations for M4

1. **Human review cadence** for official notices: a named reviewer and a crab-season rota, so the app can show Verified.
2. **Scientific review session** with a HAB scientist on `/bloom`, the tiers and the C-HARM wording, then record the sign-off in the repo.
3. **Send the CDFW request** (after you review the draft). If permission is granted, add the port-area adapter through the existing disclosure-safe aggregation.
4. **Staging website deployment** for user testing with Monterey Bay fishermen and port staff (needs your approval and a host).
5. **Alerting** when a source fails or goes stale, beyond GitHub issues.
6. **Accessibility depth:** screen-reader testing on real devices, Spanish copy, and a keyboard-operable map inspector.
