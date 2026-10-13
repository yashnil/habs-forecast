# Fishermen-first checkpoint (for the #41 session before any prototype)

**Date:** 2026-10-13. **Branch:** `research/fishermen-first` (from `feat/coastal-redesign` at `d414dd63`). **Status:** research checkpoint. No app files changed and nothing integrated.

**Method.**
- Peer-reviewed and agency evidence already cited in [research/stakeholders-and-interviews.md](../research/stakeholders-and-interviews.md); source IDs `S#` refer to that file.
- A real-data audit of `d414dd63`, run on my own server on port 3360 against the production data branch, with the manifest generated 2026-10-13 22:47 UTC.
  - Five scripted phone and desktop walkthroughs.
  - The published `official` and `observations` files.
- A read-only look at the uncommitted work in `wt-redesign` (`HabBriefing.tsx`, `hab.ts`, `nextSteps.ts`), so these recommendations don't duplicate it.
- Deeper literature, product and data-source research is still running and will extend this. Nothing here comes from an interview: none has taken place.

**Evidence tags:**
- **[E]** a published study or agency record;
- **[O]** observed today in the app or its data;
- **[H]** a hypothesis that needs interviews or usability tests.

---

## 1. The five most consequential problems CoastWatch could realistically help with

| # | Problem | Why it is consequential | Evidence |
|---|---|---|---|
| 1 | **"Is my fishery open here, and what is gating it?"** The status of a fishery in an area depends on several separate regimes. Each one is published in a different place. | **Season status:** the opener, delays and fair-start rules. **Biotoxin closures:** by Biotoxin Management Area (BMA) and evisceration order. **Whale rules:** RAMP depth and zone restrictions. **Other:** meat-fill testing, health advisories. Getting one wrong costs a season, because Dungeness catch is concentrated in the first ~6 weeks after an opener. | [E] S8 Pomeroy & Culver 2026: advisories are scattered across "various websites", email sign-ups and a hotline. [E] S13: several regimes apply at once. [E] S5 Holland & Leonard 2020: the first six weeks decide the season. [E] S3 Ekstrom et al. 2020: recreational crab stayed open on an advisory while commercial was closed, which "generated confusion… that likely led to distrust" |
| 2 | **Learning that something changed, soon and from the source.** Status reverses within weeks, so an old belief becomes a wrong one. | **Reversals:** Monterey anchovy was restricted in June, lifted July 31 and closed again Sept 11, 2026. Del Norte razor clams reopened in summer and closed on 2026-10-09. **Notice:** closures can be "sometimes more advanced notice, sometimes sudden". | [E] S8: fishermen want a "timely, accessible, easily understood notification system". [E] S3, S12, S24, S23. [E] S2 Moore et al. 2020: CA non-fishing respondents asked for "improved information about the bloom that is easy to understand" (38.2%) |
| 3 | **Missing or stale data read as "clean"** | California stops sampling a site after one clean pre-season test, and sampling depends on volunteers. "No recent result" must never read as "safe". | [E] S7, S3, S8. [O] Every North Coast HABMAP station has its last sample in **Jan–Apr 2026**: Trinidad 04-17, Humboldt Bay 04-07, Bodega 04-13, Tomales 03-03. **Monterey Wharf's last toxin (pDA) value is 2022-08-31**; only its temperature is current |
| 4 | **Preparing for an opener:** how far testing has got, how many clean rounds are still needed, and the earliest date the protocol allows | Gear, crew, buyer and port-move decisions are taken weeks ahead. Fishermen describe the process as opaque: "They don't give us a 30-day notice." | [E] S7: all 6 crabs must pass, and a closure lifts only after 2 clean rounds ≥7 days apart. [E] S8, S11 (advance notice is a management goal). [O] CoastWatch does not ingest CDPH crab results; they are published as PDFs |
| 5 | **One quick, trustworthy pre-trip check on a phone,** often on weak signal: official status plus tomorrow's sea conditions, each with its time | Fishers combine many sources and rely on real-time data. Connectivity and on-the-water use are plausible constraints but undocumented for California. | [E-adj] S10 Kuonen et al. 2019 (Oregon). [E] S2. [H] that connectivity and offline use matter enough to build for |

**Deliberately not in the top five:**
- **Disaster-assistance navigation.** [E] It is a real hardship (S2: financial support was the top request; S5: aid took more than 3 years). CoastWatch's realistic role is a direct link to official resources during a major closure, which is small; it is a follow-up item.
- **"Where are the fish".** NO-GO; see #41's [fishing-opportunity feasibility](../research/fishing-opportunity-feasibility.md).

---

## 2. What the current implementation (#41, `d414dd63`) does and does not do

Phone walkthroughs, 390 × 844, production data. The times run from navigation start to the official headline being visible on a local production build (my machine, not a phone).

| Task | Answer on the first screen? | Measured | Verdict |
|---|---|---|---|
| T1 Monterey commercial, cold first visit via the front door | Official notices, yes. "Conditions at sea" is folded: the summary line shows "Tonight: NW wind 10 to 15 kt…", and tomorrow needs another tap | **4 taps plus typing** to the briefing (Plan a trip → Commercial → search → Monterey); 5 to open conditions. Official headline at 0.25 s (deep link). 171 words on the first screen | Mostly works |
| T2 Crescent City, commercial Dungeness | **Misleading.** The first line reads *"No notice in CoastWatch's list covers these waters for Dungeness crab"*. CDFW's own statement, *"Dungeness crab 2026-27… Commercial Fishery: Season is closed"*, is in the registry but appears only inside the folded "Before you go" chapter. Even there the first item shown is "no closures… due to naturally occurring marine toxins" | 0.17 s; 162 words. The season status needs 1 more tap and scrolling. No whale/RAMP explanation anywhere; the species link goes to the Whale Safe page labelled "Check the rules" | **Fails the core question (problem 1)**, and could be read as "open". See [screenshot](img/before-crescent-dungeness-390.jpg) |
| T3 Clam Beach, razor clam | Yes: two notices (CDFW closure and CDPH health advisory) with "Since May 2, 2024" and sources. "Toxin last measured at Trinidad Pier, 179 days ago" | 0.17 s | Works. The good pattern to copy |
| T4 Santa Cruz, any activity | The answer lines, yes. The official chapter waits until "What are you after?" is answered: the species grid fills the first screen | Official headline not shown within 15 s without a choice | Acceptable, but a returning user has to choose again every time: only the place is remembered (`cw-home-port`); activity and species are not |
| T5 Return after several days: "what changed?" | **Not supported.** No memory of the last visit and no change list. When the official pages were last checked appears only in the notices drawer's detail, not on My Coast | n/a | **Missing (problem 2).** The pipeline already detects changes: [O] the `watch` block flagged CDPH **SN26-021** (the Monterey bivalve lift) as new, and My Coast shows it as "Reported lifted · awaiting review". That is good, but a returning user can't tell it is new *to them* |
| T6 Researcher provenance | Research mode and Scientific details exist, with sources | not timed yet | Next pass |

**Other findings:**
- [O] **Nearest, not newest.** Monterey's toxin line names Monterey Wharf, 0.6 km away, last value "over 4 years ago". Santa Cruz Wharf, about 33 km away and inside the 40 km radius, has a 2026-09-30 toxin value. *Terminal 1's uncommitted `hab.ts` (newest sample per kind, `EVIDENCE_RADIUS_KM = 60`) appears to address this. I am not touching it.*
- [O] **No offline support.** There is no service worker or web manifest, so the briefing is gone without signal. There is no dated copy to keep, print or share.
- [O] **Working well and worth keeping:**
  - every notice line carries its agency, date and link;
  - gap lines are hatched, never green;
  - the disputed-lift handling;
  - conditions are labelled "context for planning, not a toxin or fishing forecast";
  - nothing implies "safe" or "open".

---

## 3. The three highest-impact improvements, with acceptance criteria

All three change existing My Coast behaviour; none is a new product. They avoid the HAB-evidence work under way in `hab.ts` and `HabBriefing.tsx`.

### I1. Season and fishery status first, for the chosen species
*Problem 1. Evidence: strong [E]. Data already published (registry `statements`). Cost: small.*

When the registry holds an agency statement about the season or fishery status for the chosen species, quote it in the answer lines **before** any "no notice found" line. For Dungeness, add a sourced row naming the separate gates, as links and protocol facts only, never CoastWatch conclusions:
- the CDFW season and whale-rule (RAMP) status page;
- the CDPH crab test results;
- the closure line.

**Acceptance criteria:**
1. Crescent City, commercial, Dungeness, on a 390 px phone: the **first screen** shows CDFW's season statement verbatim, with "CDFW · transcribed <date> · not verified" and a link to the source page. It appears above "No notice in CoastWatch's list…".
2. The "no toxin closure" statement is never the first or only Dungeness line while a "Season is closed" statement exists.
3. Species without a status statement render exactly as today, verified by the existing `coastal.spec.ts` with no changes to those expectations.
4. No new words "open", "safe" or "clear". Nothing is derived; the text is the agency's own.
5. A unit test on the selection rule and an e2e test with the fixture dataset.

### I2. "Since you last checked": a local change list for saved waters, activity and species
*Problem 2. Evidence: strong for wanting timely change notification [E] S8; [H] for whether a since-last-visit list is the form people want. Data: computed in the browser, no schema change (see the manifest-compat rule). Cost: medium.*

**Behaviour:**
- Remember place, activity and species on the device, with no account, and the set of official items and data times seen at the last visit.
- On return, show at most 5 dated lines **above** the answer. Each line is one of:
  - **Agency action:** a new or lifted notice whose agency date falls after your last visit.
  - **New to CoastWatch, older agency date:** "first listed by CoastWatch after your last visit; CDPH dated it Sep 14". *Never called "new".*
  - **Release awaiting review:** for example "CDPH SN26-021 appears to lift…".
  - **New measurement:** a new sample at the nearby station, with its date.
  - **Model update:** a newer forecast run.
  - **No longer available:** a source failed, or data aged past its freshness rule.
- Add one line, "Agency pages last checked by CoastWatch: <time>", on the first screen.

**Acceptance criteria:**
1. A first visit shows no change list and no claims.
2. Two fixture snapshots (before and after: one notice added with an old effective date, one lifted, one new sample, one stale source) produce exactly the right category and dates. The test asserts that the old notice is **not** labelled newly issued.
3. With `localStorage` blocked or cleared, the page renders as today, with no errors and no "nothing changed" claim.
4. Every line has its agency or source, a date, the place, and a link.
5. No mention of alerts, notifications or push, because none exist.
6. **Benchmark:** the return task T5 ("what changed since Tuesday?") can be answered from the first screen in at most 1 tap, compared with impossible today.

### I3. A dated briefing you can keep (save, share or print) that never claims to be current
*Problem 5. Evidence: weaker: [H] connectivity; [E] S8 asks for accessible information; harbor-office posting is [H] (S#3.4). Build only the minimal version, then test it. Cost: small, because it reuses I2's local snapshot.*

**Behaviour:**
- "Keep this briefing" stores the rendered official lines, sources, phone numbers and conditions, each with its observation or issue time.
- When reopened offline, or after 12 hours, it shows "Saved <date time>. This may be out of date; check CDFW (831) 649-2883 / CDPH (800) 553-4133."
- A print stylesheet produces one page with the source URLs written out.

**Acceptance criteria:**
1. Offline in Playwright (`context.setOffline(true)` after a save), the saved briefing renders with the saved timestamp. Nothing on it reads "current", "now" or "today".
2. The printed page fits on one A4/Letter sheet for T1–T3 and shows full URLs.
3. Tap targets are at least 44 px.
4. No service worker unless measured to be needed; prefer the simplest mechanism that passes criterion 1.

**Not recommended now:**
- **Pre-season crab testing progress** (problem 4). It needs a new pipeline source, CDPH crab PDFs, with human transcription. The data audit (in progress) will judge whether it can be done reliably. Meanwhile, I1's protocol row covers the explanation part.
- **Push notifications.** Unvalidated; I2 tests demand first.

---

## 4. Documented vs. needs interviews

| Claim | Status |
|---|---|
| Fragmented official channels; fishermen want timely, easily understood notification | **Documented** [E] S8, S3 |
| Opener timing dominates seasonal revenue; aid is slow | **Documented** [E] S5, S2 |
| Reversals within weeks in 2026 (anchovy; Del Norte razor clams) | **Documented** [E] agency records S12, S23, S24 |
| Missing samples ≠ clean; sampling is volunteer-dependent and stops after a clean pre-season test | **Documented** [E] S7, S3, S8 |
| North Coast HABMAP samples are 6+ months old; Monterey Wharf toxin was last measured in 2022 | **Observed** [O] in published data, 2026-10-13 |
| The Crescent City Dungeness first screen could be read as "open" | **Observed** [O] that the wording is ambiguous. **[H]** that users misread it, which is the first thing to test |
| Users want a "since you last checked" list rather than push alerts | **[H]**: interview question and usability task |
| Weak signal and offline use matter enough to build for | **[H]** |
| Harbor offices would post a printed briefing | **[H]** |
| CPFV operators', harvesters' and port managers' information habits | **Unstudied** (no published CA study): interviews |

**Proposed primary pilot:** commercial **Dungeness crab fishermen on the North Coast** (Crescent City, Trinidad, Eureka). The workflow is the pre-opener status check: "Is my area open, what is gating it, and what changed since I last looked?"
- The opener is Dec 1, with pre-season testing about a month before, so the timing is now.
- This is the community most exposed to HAB closures [E] S6: Crescent City lost 72% of fishing opportunity in 2015–16.
- It is the walkthrough where the current app fails.

---

## 5. Coordination

- **No frontend files edited.** I read `wt-redesign` without changing it, and my server uses port 3360.
- **Overlap to avoid.** I1 and I2 touch `Briefing.tsx` (the answer lines) and `nextSteps.ts`, both of which Terminal 1 is editing. So I propose:
  - **I1:** Terminal 1 implements it in its own files; it is small. I supply the selection rule as a pure function plus tests.
  - **I2:** I build the pure change-detection module (`lib/changes.ts`) with fixtures and tests, plus a standalone component. Terminal 1 mounts it once `Briefing.tsx` settles.
  - **I3:** I build it after I2, in its own files.
- **Next for me:** finish the literature, existing-product and data-source audits; the user-testing plan; and the before/after benchmarks for T1–T6.
