# Official notices: human verification checklist (issue #12)

**Prepared 2026-10-10 03:18 UTC for a person to complete.** Nothing in the registry (`data/curated/official_notices.json`) has been changed, verified, added or lifted. The app keeps showing every record as **not verified**.

**Why issue #12 opened.** The watcher saw official pages differ from the snapshot taken when the registry was transcribed, on 2026-10-08 19:32 UTC. That transcription was made by an AI assistant from the official pages and **has never been reviewed by a person**. So every record below needs a human check, not only the changed ones.

## 1. What changed

| Watched page | At last review | Now (watcher, 2026-10-10 03:18 UTC) | What changed |
|---|---|---|---|
| [CDPH shellfish and seafood advisories (release list)](https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx) | releases SN26-001 … SN26-019 | content differs; **new release SN26-020** | New CDPH advisory (2.1) |
| [CDFW health advisories and closures](https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories) | matched at 2026-10-09 22:56 UTC | content differs | **Razor clam paragraph now includes a Del Norte County closure** (2.2) |
| [CDFW Whale Safe Fisheries](https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries) | — | unchanged | — |

The registry stores only a fingerprint (hash) of each page at review time, not its text. So "previous wording" below is the wording transcribed into the registry on 2026-10-08, and "current wording" is what the page says now.

## 2. New official notices **not** in the registry (the app does not show them)

### 2.1 CDPH SN26-020: razor clams, Del Norte County (consumption advisory)
- **Source:** <https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-020.aspx>, dated **October 9, 2026**.
- **Current wording (verbatim):**
  - "CDPH is advising the public not to consume sport-harvested razor clams gathered from Del Norte County due to high levels of Domoic Acid."
  - "For Del Norte County, this warning only applies to razor clams …"
  - "This shellfish safety notification is in addition to the razor clam advisory for Humboldt County and to the annual mussel quarantine."
- **Previous wording:** none (new release).
- **Checklist:**
  - [ ] Open the release; confirm date, area (Del Norte County), species (sport-harvested razor clams only) and toxin (domoic acid).
  - [ ] Decide the record: suggested id `cdph-2026-sn26-020-razor-clam-del-norte`, action `consumption_advisory`, area county Del Norte, effective 2026-10-09, no stated end.
  - [ ] Note that it refers to a **"razor clam advisory for Humboldt County"** by CDPH. The registry has the CDFW Humboldt *fishery closure* but no CDPH Humboldt razor-clam *advisory*. Check whether one is active and needs its own record.

### 2.2 CDFW: recreational razor clam fishery closed in Del Norte County
- **Source:** <https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories> (razor clam section).
- **Current wording (verbatim):** "The recreational razor clam fishery closed in Del Norte County on October 9, 2026, and remains closed in Humboldt County due to elevated levels of domoic acid in razor clams. The Humboldt County closure has been in effect since May 2, 2024."
- **Previous wording (registry, record `cdfw-2024-razor-clam-humboldt`):** "The recreational razor clam fishery remains closed in Humboldt County due to elevated levels of domoic acid in razor clams. The Humboldt County closure has been in effect since May 2, 2024."
- **Checklist:**
  - [ ] Confirm the Del Norte closure, preferably against the OEHHA recommendation memo or the CDFW news release. The page also lists an older headline, "Del Norte County Razor Clam Fishery Opens After One Month Closure", from a previous event; don't confuse the two.
  - [ ] Decide whether to add a new record (suggested `cdfw-2026-razor-clam-del-norte`, fishery closure, effective 2026-10-09) or to widen the existing Humboldt record. A separate record keeps each closure's own dates.
  - [ ] Update `official_text` of `cdfw-2024-razor-clam-humboldt` to the current sentence. The Humboldt closure itself is unchanged.

## 3. Existing records

**How each record was checked.** The transcribed `official_text` was searched for, verbatim, on each source page as fetched on 2026-10-10. "Verbatim" means the exact wording is still there. "Not verbatim" shows the closest current passage; usually the source is a secondary page that words it differently.

| # | Record | Agency | Area | Result | Human check |
|---|---|---|---|---|---|
| 1 | `cdph-2026-annual-mussel-quarantine` (annual quarantine of sport-harvested mussels, 2026-05-01 to at least 10-31) | CDPH | Whole coast, bays and estuaries | [SN26-008](https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-008.aspx): **verbatim**. [SN26-018](https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-018.aspx) and SN26-020 restate it: "will continue through at least October 31." | [ ] Confirm "at least October 31" is still the latest statement. |
| 2 | `cdph-2026-sn26-018-monterey-bivalves` (do not eat sport-harvested bivalves, Monterey County, from 2026-09-14) | CDPH | Monterey County | [SN26-018](https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-018.aspx): **verbatim**. The [advisory map](https://experience.arcgis.com/experience/394836318cfe4f7494e1c09097a43559) can't be read automatically. | [ ] Confirm in the map that Monterey County is still shown as advisory. |
| 3 | `cdph-2026-sn26-019-anchovy-central-coast` (do not eat northern anchovies, Pigeon Point to Point Lobos, from 2026-09-14) | CDPH | 37° 11.00′ N to 36° 31.46′ N | [SN26-019](https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-019.aspx): **not verbatim** (similarity 0.83). The page now prints "37⁰ 11.00’ N." with a different degree character and line breaks. Wording appears otherwise the same. | [ ] Read the release and confirm the latitudes. |
| 4 | `cdfw-2026-anchovy-take-restriction-monterey-bay` (no take for human consumption, Monterey Bay, from 2026-09-11) | CDFW | Pigeon Point to Point Lobos | [CDFW status page](https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories): **verbatim**. The [news release](https://wildlife.ca.gov/News/Archive/commercial-and-recreational-take-restrictions-in-place-for-northern-anchovy-in-monterey-bay-due-to-public-health-hazard) words it differently; the [memo (PDF)](https://nrm.dfg.ca.gov/FileHandler.ashx?DocumentID=247331) needs reading by hand. | [ ] Confirm against the PDF memo. |
| 5 | `cdfw-2024-razor-clam-humboldt` (recreational razor clam fishery closed, Humboldt, since 2024-05-02) | CDFW | Humboldt County | **Changed sentence**, see 2.2. [OEHHA memo (PDF)](https://oehha.ca.gov/sites/default/files/media/2024-10/razorclamrecreationalfisherymemo050224.pdf) needs reading by hand. | [ ] Update the wording; Humboldt status unchanged. |
| 6 | `cdfw-rock-crab-commercial-40n` (commercial rock crab closure, 40° 00′ N to 40° 30′ N) | CDFW | Mendocino/Humboldt line to Cape Mendocino | [CDFW status page](https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories): **verbatim**. The start date is still not published. | [ ] Ask the CDFW line (831) 649-2883 for the start date if needed. |
| 7 | `cdph-rock-crab-viscera-north-coast-nci` (do not eat rock crab viscera, CA/OR border to Mendocino/Sonoma line, and Northern Channel Islands) | CDPH, quoted by CDFW | North Coast and Northern Channel Islands | [CDFW status page](https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories): **verbatim** (including the agency's spelling "interal"). The CDPH release list doesn't carry the wording. | [ ] Find the originating CDPH release and confirm it is still in effect. |
| 8 | `cdph-nci-bivalve-special-advisory` (bivalve special advisory, Northern Channel Islands) | CDPH | Northern Channel Islands | [Advisory map](https://experience.arcgis.com/experience/394836318cfe4f7494e1c09097a43559): can't be read automatically. | [ ] Confirm in the map that the special advisory is still shown. |

The CDFW page also states, unchanged: "There are currently no closures of the recreational or commercial California spiny lobster fisheries …" and "There are currently no closures in the recreational [and commercial] Dungeness crab fishery due to naturally occurring marine toxins." The registry has no records for these, which is correct.

## 4. After the check

When every box is ticked and the registry is edited by a person:

```
uv run cwp review-official --reviewer "NAME" --confirm
```

This records the review and the new page fingerprints. The app then shows the notices as reviewed, with the reviewer and date. Until then it says "Not verified" and links to the agency pages.

**Do not** use a detected page change as grounds to mark a record verified, lifted or expired. The watcher only says "something changed"; the agencies' pages are the authority.

**Urgency.** Items 2.1 and 2.2 are **active advisories the app does not list.** The app tells users that "no notice listed here does not mean an area is open or that seafood is safe", but Crescent City and Del Norte users would not see the razor clam warning until a person adds it.
