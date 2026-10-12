# Official notices: review checklist (2026-10-12)

**For the owner to complete personally.** Preparing this checklist changed nothing in the registry ([`data/curated/official_notices.json`](../../../data/curated/official_notices.json)). No record has been verified, added, lifted or expired, and the app shows every notice as **not verified**. The registry was transcribed by an AI assistant on 2026-10-08 and has never been reviewed by a person. This checklist replaces the [2026-10-10 checklist](../p3/regulatory-review-2026-10-10.md).

All source text below was fetched from the agency pages on **2026-10-12** and is quoted verbatim, except that line breaks are removed. "Stored" is the wording in the registry today.

## How to work through it (about 45–60 minutes)

1. Do the **three priority items** first (§1–§3). They contain the open questions.
2. Then check the remaining records (§4), the "no closure" statements (§5) and completeness (§6).
3. Tick each box only after opening the agency link yourself. Anything you would change, note it in the box. You can edit the registry yourself or ask for a PR. Then run `uv run pytest tests/test_official.py` in `pipeline/`.
4. **Only when every box is ticked**, run this from `pipeline/`:
   ```
   uv run cwp review-official --reviewer "<the name you want shown publicly>" --confirm
   ```
   The app will then show the notices as reviewed by that name on that date. If you stop partway, don't run it.
5. Never mark a record lifted or expired because a date passed or a page changed; only when the agency says so.

**Agency contacts:** CDPH shellfish and biotoxin line, 1 (800) 553-4133. CDFW domoic acid fishery closure line, (831) 649-2883.

### CDPH 2026 shellfish and seafood releases (for reference)

| Release | Date | Title (as published) | In the registry? |
|---|---|---|---|
| [SN26-008](https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-008.aspx) | Apr 24 | Annual Quarantine of Sport-Harvested Mussels Begins May 1 | yes (record M) |
| [SN26-013](https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-013.aspx) | Jun 26 (updated Jul 6) | CDPH Warns the Public Not to Consume Pacific Sardines and Northern Anchovies Harvested from Central California Waters | no; lifted by SN26-016 |
| [SN26-015](https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-015.aspx) | Jul 31 | CDPH Warns Consumers Not to Eat Razor Clams from Del Norte County | no; lifted by SN26-017 |
| [SN26-016](https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-016.aspx) | Aug 3 | CDPH Lifts Warnings about Pacific Sardines and Northern Anchovies Harvested in Central California Waters | n/a (lift) |
| [SN26-017](https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-017.aspx) | Aug 31 | CDPH Lifts Domoic Acid Warning for Del Norte County Razor Clams | n/a (lift) |
| [SN26-018](https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-018.aspx) | Sep 14 | CDPH Warns Consumers Not to Eat Sport-Harvested Bivalve Shellfish from Monterey County | yes (record B) |
| [SN26-019](https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-019.aspx) | Sep 14 | CDPH Warns the Public Not to Eat Northern Anchovies Harvested from Central California Waters | yes (record A1) |
| [SN26-020](https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-020.aspx) | Oct 9 | CDPH Warns Against Consuming Razor Clams from Del Norte County | yes (record R4) |

The other 2026 releases were all lifted by later releases:
- Dungeness crab viscera, North Coast: SN26-001 and -003, lifted area by area in SN26-002, -004 and -005 (the last on Jan 26: Reading Rock to Cape Mendocino).
- Sport-harvested bivalves: Marin and San Mateo (SN26-006, lifted by -010 and -011); Sonoma (-007, lifted by -012); Del Norte (-009, lifted by -014).

Confirm this in §6.

---

## 1. PRIORITY — Anchovies and sardines, Monterey Bay

**What was found.** The CDFW page says CDPH "has also issued a recreational health advisory, recommending that the public not consume **sardines and anchovies**". But its link points to **SN26-013**, the **June 26** advisory for *commercially* harvested sardines and anchovies, which CDPH **lifted on August 3** (SN26-016). The current CDPH advisory, **SN26-019** (Sept 14), names **northern anchovies only**. So the CDFW sentence appears out of date or mislabelled. No current CDPH sardine advisory was found. That is a finding to confirm, not a conclusion.

### Record A1 — `cdph-2026-sn26-019-anchovy-central-coast` (CDPH consumption advisory)
- **Agency source:** [SN26-019](https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-019.aspx), September 14, 2026.
- **Source text:** "The California Department of Public Health (CDPH) advises the public not to consume northern anchovies harvested from waters between: Pigeon Point, San Mateo County (37⁰ 11.00’ N. lat.) and Point Lobos, Monterey County (36⁰ 31.46’ N. Lat.) … The public is advised to not consume northern anchovies harvested from the affected area until further notice. Please note that this advisory does not apply to northern anchovies directly off-loaded or caught recreationally for use as bait (i.e. not intended for human or pet consumption)."
- **Stored:** "The California Department of Public Health (CDPH) advises the public not to consume northern anchovies harvested from waters between: Pigeon Point, San Mateo County (37° 11.00’ N. lat.) and Point Lobos, Monterey County (36° 31.46’ N. Lat.)". This matches apart from the degree character (the source uses "⁰").
- **Region:** 37° 11.00′ N to 36° 31.46′ N. The offshore extent is not stated (the registry notes this).
- **Effective:** 2026-09-14. **End:** "until further notice".
- [ ] Still listed by CDPH and not lifted.
- [ ] Latitudes match.
- [ ] Species is northern anchovy only. Commercial *and* recreational consumption is covered (the release says "the public"); bait is excepted.

### Record A2 — `cdfw-2026-anchovy-take-restriction-monterey-bay` (CDFW take restriction)
- **Agency sources:** [CDFW status page](https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories); [CDFW news release](https://wildlife.ca.gov/News/Archive/commercial-and-recreational-take-restrictions-in-place-for-northern-anchovy-in-monterey-bay-due-to-public-health-hazard); [CDFW memo (PDF)](https://nrm.dfg.ca.gov/FileHandler.ashx?DocumentID=247331).
- **Source text (status page):** "Northern anchovies may not be harvested recreationally or commercially for human consumption in Monterey Bay (Pigeon Point, San Mateo County to Point Lobos, Monterey County) due to elevated levels of domoic acid toxin in the fish. The Monterey Bay closure went into effect on September 11, 2026. Commercially taken anchovies from the area may only be sold as dead bait clearly labeled as not for human consumption or use in pet food. Recreationally taken anchovies may only be used as bait."
- **Stored:** the first two sentences, verbatim.
- **Region:** Pigeon Point to Point Lobos. **Effective:** 2026-09-11. **End:** not stated.
- [ ] The memo (PDF) confirms the date, area and the commercial and recreational scope.
- [ ] Still listed on the status page.

### Open questions
- [ ] **Is there any current CDPH advisory for Pacific sardines?** Check the CDPH release list and, if unclear, call 1 (800) 553-4133.
  - *If none:* no new record. Optionally tell CDFW that its page still links the lifted SN26-013 as a "recreational health advisory".
  - *If one exists:* add it as a record. The Fisheries page's sardine basis ("no current commercial toxin closure is listed") then needs a mention.
- [ ] SN26-013 also advised recreational catchers to remove viscera. It was lifted, so CoastWatch should **not** show it. Confirm that.

## 2. PRIORITY — Razor clams, Humboldt and Del Norte

**Timeline from the agency pages:**
- **Humboldt:** CDPH warning [SN24-003](https://www.cdph.ca.gov/Programs/OPA/Pages/SN24-003.aspx) (May 2, 2024), still in effect per [SN25-024](https://www.cdph.ca.gov/Programs/OPA/Pages/SN25-024.aspx) (Oct 6, 2025), [SN25-025](https://www.cdph.ca.gov/Programs/OPA/Pages/SN25-025.aspx) (Oct 24, 2025) and SN26-020. The CDFW fishery has been closed since May 2, 2024.
- **Del Norte:** CDPH warned on Jul 31, 2026 (SN26-015) and lifted on Aug 31 (SN26-017). CDPH warned again on **Oct 9 (SN26-020)**, and CDFW closed the fishery on **Oct 9**.

### Record R1 — `cdfw-2024-razor-clam-humboldt` (CDFW recreational fishery closure)
- **Agency sources:** [CDFW status page](https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories); [OEHHA memo, May 2, 2024 (PDF)](https://oehha.ca.gov/sites/default/files/media/2024-10/razorclamrecreationalfisherymemo050224.pdf).
- **Source text:** "The recreational razor clam fishery closed in Del Norte County on October 9, 2026, and remains closed in Humboldt County due to elevated levels of domoic acid in razor clams. The Humboldt County closure has been in effect since May 2, 2024."
- **Stored:** identical.
- **Region:** Humboldt County. **Effective:** 2024-05-02. **End:** none stated.
- [ ] Still closed in Humboldt.
- [ ] The OEHHA memo matches the date.

### Record R2 — **missing:** CDPH razor clam advisory, Humboldt County (not in the registry)
- **Agency source:** [SN24-003](https://www.cdph.ca.gov/Programs/OPA/Pages/SN24-003.aspx), May 2, 2024: "CDPH is advising the public not to consume sport-harvested razor clams gathered from Humboldt County due to high levels of Domoic Acid."
- **Still in effect per CDPH:** SN25-024 (Oct 6, 2025): "A warning against eating sport-harvested razor clams from Humboldt County remains in effect". SN26-020 (Oct 9, 2026): "in addition to the razor clam advisory for Humboldt County".
- [ ] Confirm it is still in effect. It is a CDPH **consumption** advisory, distinct from the CDFW **fishery** closure (R1).
- [ ] Decide whether to add it. Suggested record: id `cdph-2024-sn24-003-razor-clam-humboldt`, consumption advisory, sport harvest, razor clams, Humboldt County, effective 2024-05-02, sources SN24-003 plus SN26-020.

### Record R3 — `cdfw-2026-razor-clam-del-norte` (CDFW recreational fishery closure)
- **Agency source:** [CDFW status page](https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories). The same sentence as R1, which is shared by both records.
- **Region:** Del Norte County. **Effective:** 2026-10-09. **End:** none stated.
- [ ] Find a primary document (CDFW news release or OEHHA memo for the Oct 9 closure) and add it as a source.
- [ ] Don't confuse it with the page's older headline "Del Norte County Razor Clam Fishery Opens After One Month Closure" (an earlier event).

### Record R4 — `cdph-2026-sn26-020-razor-clam-del-norte` (CDPH consumption advisory)
- **Agency source:** [SN26-020](https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-020.aspx), October 9, 2026.
- **Source text:** "CDPH is advising the public not to consume sport-harvested razor clams gathered from Del Norte County due to high levels of Domoic Acid. … For Del Norte County, this warning only applies to razor clams as they are known to concentrate and retain domoic acid toxin in their meat and tissues at higher levels and for much longer than other species of bivalve shellfish."
- **Stored:** the same two sentences, joined. They are not adjacent in the release.
- **Region:** Del Norte County. **Effective:** 2026-10-09. **End:** none stated.
- [ ] Area, species (sport-harvested razor clams only) and toxin (domoic acid) match.
- [ ] Optional: insert " … " between the two sentences in `official_text`.

## 3. PRIORITY — Annual mussel quarantine ending

### Record M — `cdph-2026-annual-mussel-quarantine` (CDPH quarantine)
- **Agency sources:** [SN26-008](https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-008.aspx), April 24, 2026; restated in SN26-018 and SN26-020.
- **Source text (SN26-008):** "The quarantine begins May 1 and applies to all species of mussels that are recreationally harvested for human consumption along the California coast, including all bays and estuaries. During the quarantine, mussels may only be harvested for use as bait."
- **Latest end statement (SN26-020, Oct 9):** "…will continue through at least October 31."
- **Stored:** the SN26-008 sentences, verbatim. Expected end 2026-10-31, with a note that it "can be extended; an end date is not confirmed until CDPH announces it".
- **Region:** the whole California coast, including bays and estuaries (statewide; not drawn on the map).
- **Precedent:** in 2025, CDPH announced the end a week ahead ([SN25-025](https://www.cdph.ca.gov/Programs/OPA/Pages/SN25-025.aspx), Oct 24: "ends at midnight Friday, October 31, 2025"). It stated that county warnings remained in effect.
- [ ] Still in effect today.
- [ ] **Watch for CDPH's end-of-quarantine release (expected late October).** When it is published, mark the record ended *as of the date CDPH gives*, citing that release. If CDPH extends the quarantine, update the expected end. If nothing is announced by Oct 31, leave it active and call CDPH. The app will not end it on its own.
- [ ] Ending the quarantine does **not** end the county advisories (Monterey bivalves; razor clams in Humboldt and Del Norte). Check that they remain.

---

## 4. Other records

### B — `cdph-2026-sn26-018-monterey-bivalves` (CDPH consumption advisory)
- **Sources:** [SN26-018](https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-018.aspx), Sept 14, 2026; [CDPH advisory map](https://experience.arcgis.com/experience/394836318cfe4f7494e1c09097a43559).
- **Source text = stored:** "CDPH is advising the public not to eat sport-harvested mussels, clams, scallops, or oysters, gathered from Monterey County due to dangerous levels of domoic acid toxin detected in mussels from Monterey County."
- **Region:** Monterey County. **Effective:** 2026-09-14. **End:** none stated.
- [ ] The map still shows Monterey County under advisory, and no lifting release exists.

### C1 — `cdfw-rock-crab-commercial-40n` (CDFW commercial fishery closure)
- **Source:** [CDFW status page](https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories), rock crab section.
- **Source text = stored:** "A Domoic Acid Fishery Closure is in effect for the commercial rock crab fishery from the Mendocino/Humboldt county line (40° 00.00’ N. lat.) to Cape Mendocino, Humboldt County (40° 30.00’ N. lat.)."
- **Region:** 40° 00′ N to 40° 30′ N. **Effective:** not published. **End:** none stated.
- [ ] Still listed. Optional: ask the CDFW line for the start date and record it with its source.

### C2 — `cdph-rock-crab-viscera-north-coast-nci` (CDPH advisory, quoted by CDFW)
- **Source:** [CDFW status page](https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories). No 2026 CDPH release carries this wording.
- **Source text = stored:** "CDPH currently recommends not eating the interal organs of rock crab caught from the California/Oregon border to the Mendocino/Sonoma county line, and from the waters off Santa Rosa Island and the Northern Channel Islands." The spelling "interal" is the agency's.
- **Region:** CA/OR border to the Mendocino/Sonoma line, plus the Northern Channel Islands. **Effective:** unknown.
- [ ] Find the originating CDPH release (likely before 2026). If none can be found, keep the CDFW page as the source and say so.

### N — `cdph-nci-bivalve-special-advisory` (CDPH special advisory)
- **Source:** [CDPH advisory map](https://experience.arcgis.com/experience/394836318cfe4f7494e1c09097a43559) only. **Stored:** "Special Advisory Area - Northern Channel Islands".
- **Region:** Anacapa, Santa Cruz, Santa Rosa and San Miguel islands. **Effective:** not stated.
- [ ] The map still shows the special advisory.

## 5. "No closure" statements (shown as agency statements, never as safety)

All four were found verbatim on the agency pages on 2026-10-12.

| Statement (stored = source) | Source |
|---|---|
| "There are currently no closures in the recreational Dungeness crab fishery due to naturally occurring marine toxins." | [CDFW](https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories) |
| "There are currently no closures in the commercial Dungeness crab fishery due to naturally occurring marine toxins." | same |
| "There are currently no closures of the recreational or commercial California spiny lobster fisheries due to naturally occurring marine toxins." | same |
| "Commercial Fishery: Season is closed. Recreational Fishery: Season is closed." (Dungeness 2026-27; season status, not a toxin closure) | [Whale Safe Fisheries](https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries) |

- [ ] Each still reads as quoted.

## 6. Anything missing?

- [ ] Every 2026 CDPH warning in the table above is either in the registry or lifted by a later release.
- [ ] The CDFW status page lists no closure or take restriction that is missing here.
- [ ] Decide on R2 (CDPH Humboldt razor clam advisory) and on the sardine question (§1).

When every box is ticked, run the command in "How to work through it".
