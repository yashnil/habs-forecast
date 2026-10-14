# Fishermen-first report: decisions, ranked problems, a minimal experience and the handoff

**Date:** 2026-10-13. **Branch:** `research/fishermen-first` (PR #47, based on #41). **Read first:** the [checkpoint](00-checkpoint.md). This report consolidates the four research documents:

| Doc | What it holds |
|---|---|
| [01 Literature and testimony](01-literature.md) | 41 new sources (L1–L41) added to #41's S1–S36. Covers impacts, participation, small versus large vessels, RAMP, disaster aid and regional differences. Each claim is tagged [R] research, [T] testimony, [H] hypothesis or [U] unsupported. |
| [02 Existing products](02-existing-products.md) | How fishermen find each kind of information today: the hops per task, what to link rather than rebuild, and patterns from Oregon, Washington and Florida. |
| [03 Testing kit](03-testing-kit.md) | The 15-minute pilot, recruitment drafts (not sent), observation sheet, friction log, and the novice-versus-experienced method. |
| [04 Source feasibility](04-source-feasibility.md) | A live audit of 14 sources (lag, cadence, failure behaviour, licence, authority) and a respectful change-monitor design. |

No interview or usability session has taken place. Every user walkthrough below is a scripted simulation on real published data, not stakeholder evidence.

---

## 1. Verified findings, assumptions and open uncertainties

### Verified (research or agency record [R], or observed in data [O])
1. **Timing is where the value lies.**
   - Dungeness revenue is concentrated in the first ~6 weeks after the opener [R, S5].
   - Commercial openers were delayed in every season from 2022–23 to 2025–26 [R, 01 L11], with whale-entanglement (RAMP) rules a major cause (01 §Q5).
   - Delays are announced days ahead. The 2025–26 Reading Rock–Cape Mendocino delay was announced on 8 Jan 2026 [R, verified].
2. **Small vessels bear more of the loss.**
   - Boats under 40 ft lost roughly 30% of their relative share of crab revenue, and more than twice that in Crescent City [R, Jardine 2020, 01 L7].
   - Larger boats moved to open areas; small ones waited or switched fishery [R].
3. **Temporary inactivity is common; permanent exit because of HABs is not established.**
   - 56.4% (north) and 72.5% (central) of crab vessels landed nothing in California during the early 2016 closures. Participation "mostly returned to their predisturbance state" [R, Fisher 2021, verified pp. 3–4].
   - **No study attributes retirement or vessel sales to HABs** [U if claimed].
4. **Official information is fragmented and lags.**
   - Fishermen describe notices scattered across websites, e-mail lists and a hotline, and ask for a timely, understandable notification system [R/T, S8].
   - Agency pages disagree for hours to days. CDPH lifted the Monterey bivalve warning on 10/12, and CDFW's list showed the lift only later on 10/13 [O, 02 S1, rechecked].
   - The CDPH advisory map updated about 25 h after the release [O, 04].
   - A Fish and Game Commission staff summary dated after 9 Oct still reported the 31 Aug Del Norte reopening [R, 01].
5. **Scientific evidence is sparse and late.**
   - North Coast HABMAP stations: last samples Jan–Apr 2026 [O].
   - Toxin (pDA) results trail cell counts by about 2 months at Avila, Stearns and Santa Monica; Santa Cruz is the exception [O, verified].
   - The CA HAB Bulletin's latest issue is May–June 2026 [O].
   - CDPH crab and seafood PDFs post about 10–14 days after catch [O, 04].
6. **Disaster aid is slow and small.**
   - GAO: 1.3–4.8 years from request to payment nationally.
   - 2015–16 Dungeness: requested Feb 2016 → allocated June 2018 → checks around late 2019.
   - 58% of salmon fishermen got $739.50 for 2023 [R/T, 01 L22].
7. **Already solved well elsewhere (link to them, don't rebuild):** NWS marine forecasts, NDBC, CO-OPS tides, the CDFW in-season changes page, CDFW e-mail/text alert topics [O, 02].
8. **Current CoastWatch (d414dd63) fails two of the six tasks:**
   - Crescent City Dungeness status. The #41 session has since fixed it, uncommitted, to be re-benchmarked.
   - "What changed since I last looked", which is not supported [O, 00 §2].

### Hypotheses that need interviews or tests [H]
- Fishermen would check a "since you last looked" list, and would prefer it to push alerts.
- The Crescent City wording before the fix was actually misread as "open". The wording was ambiguous [O]; whether anyone misreads it is untested.
- Weak signal or offline use matters enough to justify a dated, kept briefing.
- Harbor offices would post a printed briefing.
- Older owner-operators prefer the phone line and colleagues to any website. The fleet is older (owners averaged 58 in 2017 [R]), but there is **no California data on phone or app use**.

### Open uncertainties
- How Southern California livelihoods respond to HAB closures (unstudied).
- California fuel and insurance costs after 2004 (no source found).
- The RAMP regulation's minimum notice period (text blocked during research).
- Whether CDFW's GovDelivery alerts reach fishermen faster than dock talk.

---

## 2. Stakeholder decision inventory and priority

Scored 1–3 on each factor. Evidence: 3 = [R]/[T], 2 = [R-adjacent], 1 = [H]. Feasibility = CoastWatch can help today with reliable, available data.

| Stakeholder | Decision | When | Consequence | Information needed | Current sources and friction | CoastWatch can help? | Frequency | Measure | Sev | Freq | Frust | Evid | Feas | **Σ** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Dungeness crab, commercial** | Set gear, crew and buyer for the opener, or move to an open zone | Weeks to days before Nov 15 / Dec 1, and at each delay | A season's revenue; fair-start lock-in | Season status per zone, the biotoxin (BMA) state, RAMP action, testing progress | CDFW pages, e-mail, hotline, CDPH PDFs, dock talk; 8+ hops (02 Task A) | **Yes:** status quoted with sources (I1, done by #41); changes (I2); RAMP link. Testing progress needs PDF transcription (later) | Weekly in season; daily near decisions | Time to the agency source; correct status; can explain the change | 3 | 2 | 3 | 3 | 3 | **14** |
| **Small commercial vessel (any fishery)** | Go or no-go tomorrow | Night before, morning | Safety, fuel, a day's income | Marine forecast and alerts, buoy, official restrictions | NWS, NDBC, CDFW (well served individually; 3 hops) | Partly: conditions are already integrated; add only freshness clarity | Daily | One check covers status plus weather, with times | 3 | 3 | 2 | 2 | 3 | **13** |
| **Recreational shellfish harvester** | Dig or pick this weekend; eat it | Days ahead around low tides | Poisoning risk; wasted trip | Closure and advisory by beach and county, DA and PSP, date | CDFW and CDPH; ~4 hops; the newest notice is not always found first | **Yes:** already good (T3); I2 catches reversals | Weekly in season | No stale-notice errors | 3 | 2 | 2 | 3 | 3 | **13** |
| **Recreational angler / CPFV** | Which species to keep and eat; passengers' questions | Day before, at the dock | Health; trust with passengers | Advisories by species and area; rules | Fish Rules app (regulations), CDFW; HAB notices separate | Yes for HAB notices; regulations → link to Fish Rules or CDFW | Per trip | Correctly separates HAB from contaminant advisories | 2 | 3 | 2 | 1 | 3 | **11** |
| **Commercial groundfish** | Where and at what depth to fish | Per trip, in-season | Legal compliance, revenue | Management-area depth rules, in-season changes, MPAs | CDFW in-season page, PFMC, federal rules; mapping the town to the management area is manual (02 Task D) | Link only. CoastWatch must not judge legality | Per trip | n/a (out of scope) | 3 | 3 | 2 | 2 | 1 | **11** |
| **Port / harbor office** | What to tell tenants and the public; fee relief | Around closures | Revenue, public trust | Official status for the port's waters; history | Forwards CDFW releases [H] | A dated, printable briefing (I3) [H] | Monthly; spikes around closures | Would they post it (test) | 2 | 1 | 2 | 1 | 3 | **9** |
| **HAB scientist / fisheries researcher** | Compare model and observation; assemble closure history | Bulletin cycles, studies | Scientific accuracy | Timestamps, provenance, versioned notice history | ERDDAP, CDPH PDFs; closure records assembled by hand (S6) | Research mode already strong; a versioned registry is a later item | Weekly | Can cite provenance in under 2 min | 2 | 2 | 2 | 3 | 2 | **11** |

**Primary pilot:** North Coast commercial Dungeness crab fishermen. **Workflow:** the pre-opener check, "Is my zone open, what is holding it, and what changed since I last looked?"
- It is the highest-scoring decision.
- It is the most HAB-exposed community (S6: Crescent City lost 72% of fishing opportunity in 2015–16).
- Small vessels there lose most (L7).
- The opener is weeks away.

The **secondary arm** is shellfish harvesters (the same "what changed" question, a different agency and toxin).

---

## 3. Ranked user problems worth solving

| Rank | Problem | Why ranked here | CoastWatch status |
|---|---|---|---|
| 1 | Knowing the current status of my fishery in my waters, and which separate gates apply (season, biotoxin, RAMP) | Highest consequence × frequency; strong evidence | **Fixed by #41 (I1), uncommitted.** RAMP link row recommended (link only; no computable status exists, 04 §2.12) |
| 2 | Learning what changed since I last looked, without mistaking an old notice for a new one | Reversals within weeks; agencies lag one another; fishermen ask for timely notification [R/T] | **Missing.** I2 proposed |
| 3 | Not mistaking missing or stale evidence for "clean" | Sampling is sparse and late [O]; a documented misreading risk [R] | **Good:** hatched gap lines. Newest-not-nearest fixed by #41's `hab.ts` |
| 4 | Preparing for the opener: testing progress and the protocol | High value [R], but the data are PDFs posted 10–14 days after catch | **Not started.** Needs a reviewed transcription pipeline: later, not now |
| 5 | A quick, dated pre-trip check that survives weak signal | Plausible, weakly evidenced [H] | No offline support. I3 (minimal) to test |
| — | Finding fish; legal judgements (inside an MPA, depth limits); disaster-aid advice | Out of lane or unsupported | Link only; never advise |

---

## 4. A simple initial experience (What changed? What does it mean? What next?)

My Coast's Coastal mode, after #41's current work, already follows "answer → waters and species → HAB evidence → official + conditions → next steps → details". The smallest change that answers the three questions adds only one block, and no new controls:

```
[ Crescent City · Commercial · Dungeness crab ]       saved on this phone
─ Since you last checked (Tue 6 Oct, 7 days ago) ───────────────────────────
  │ CDFW · Season status page changed · checked Oct 13 3:47 PM      [Read ↗]
  │ Other species here: CDPH · Del Norte razor clams closed · dated Oct 9   [Read ↗]
  ▨ North Coast shore stations: still no sample since April
─ What it means here (answer lines, from #41) ──────────────────────────────
  │ CDFW: "Commercial Fishery: Season is closed." · transcribed, not verified
  ▨ No toxin sample within 40 km · no model value for these waters
─ What to check next (next steps, from #41 + RAMP link row) ────────────────
  1 CDFW season and RAMP actions for your zone ↗   2 CDPH crab results ↗   3 (831) 649-2883
```

- **First visit:** the "Since you last checked" block is absent. Nothing is claimed.
- **localStorage blocked:** the block is absent, and the page is unchanged.
- The scientific workspace stays in Research mode and in Details.

---

## 5. "What changed near your waters?": the case for building it, and its rules

**Justified for a small, local build, not for a notification service.**
- **For:** the evidence on reversals and fragmentation [R/T], and the watcher already detects agency changes [O].
- **Against a push service:** demand for the *form* is [H]. The testing kit's T-changed task measures it, and the decision rules in 03 §8 say when an opt-in alert would be justified.

**Rules** (from 04 §4.3, adapted to the browser so the manifest needs no schema change):

| Category | Shown as | Required fields |
|---|---|---|
| Agency issued or lifted, dated after the last visit | "CDPH · dated Oct 9" | Agency, agency date, place, link |
| Newly listed by CoastWatch, agency date before the last visit | "First listed by CoastWatch since your last visit · CDPH dated it Sep 14" | Both dates. **Never "new".** |
| Release that appears to lift, awaiting review | "CDPH SN26-021 appears to lift … · not reviewed" | Release id and link |
| Agency page changed (hash only, no structured item) | "CDFW season page changed · checked <time>" | Page and check time. No claim about what changed. |
| New measurement at a station near these waters | "Santa Cruz Wharf · new sample Sep 30" | Station, sample date, km |
| New model run | "Forecast updated · issued Oct 12" | Issue date |
| Became unavailable or stale | "Shore station has not reported since …" / "update failed" | Source, last success |

---

## 6. Before-and-after benchmarks

Full results, method and limitations: [07](07-benchmarks.md). These are scripted walkthroughs on fixture data, not stakeholder tests.

| Task | Before (`d414dd63`) | After (`599d2d18` + "Since you last checked") |
|---|---|---|
| T1 Monterey, tomorrow's conditions, cold | ✓ 5 taps + typing | ✓ 4 taps + typing (1 scroll) |
| T2 Crescent City Dungeness | ✓ 3 taps, **first line "No notice…"** | ✓ 2 taps, **first line: CDFW "Season is closed"** |
| T3 Clam Beach razor clams | ✓ 3 taps | ✓ 2 taps (1 scroll) |
| T4 Santa Cruz recent HAB activity | ✓ 0 taps | ✓ 0 taps |
| T5 What changed since the last visit | **✗ not possible: 3 agency pages to compare by hand** | **✓ 1 tap, every change with its source and date** |
| T6 Researcher provenance | ✓ 0 taps, 1 scroll | ✓ 0 taps, 1 scroll |

## 7. Integration handoff (with the #41 session)

The contract is in [06](06-integration-contract.md).

| Item | Owner | State |
|---|---|---|
| Season status first (I1) | #41 | **Pushed** in `a948908c`/`32449b76` |
| RAMP / whale-rule gate | #41 | Link only: no machine-readable status exists (04 §2.12) |
| "Since you last checked" (I2) | This branch | **Built, mounted under the answer line in Coastal mode** (`4d840ec8`, the agreed one-line mount on top of #41's head) |
| Remembered activity and species | This branch | Built: a "Continue with …" link (1 tap), never an automatic redirect |
| Dated kept briefing (I3) | This branch | Built: print view `/coast/kept` and a single offline HTML file |
| Honest watcher User-Agent | This session, PR #48 against `main` | Draft PR; approved in principle by the owner |

**Integrate:** the change block, the remembered activity and species, and the kept briefing as one secondary link.
**Do not integrate:** push, e-mail or SMS alerts (not built, demand unvalidated); weather in kept copies or any new NOAA call from these features (robots.txt question); pre-season crab-testing progress (needs a reviewed PDF transcription pipeline); disaster-assistance links (below).

### Disaster-assistance resources: verified, not integrated

These are direct official sources, live on 2026-10-13 ([02 §9](02-existing-products.md)):
- NOAA Fisheries Fishery Resource Disaster Assistance, and its list of determinations;
- CDFW Federal Fishery Disaster Relief;
- PSMFC Federal Fishery Disaster Relief;
- SBA disaster assistance;
- California Sea Grant.

**Rule if they are ever shown:**
- only while a Commerce disaster determination or request covers the chosen species and region, linked by its number;
- worded as "official information about fishery disaster relief", **never** as eligibility, availability, amounts or timing;
- with no tax, loan or financial advice.

None of the determinations seen in the review (02 §9) concerns the 2026–27 Dungeness season, but the list was not checked exhaustively for pending requests. Until someone checks it, nothing is shown.

## 8. Decisions that need the project owner

**Decided on 2026-10-13:**
- the pilot (North Coast commercial Dungeness; shellfish harvesters second);
- the ownership split;
- the honest User-Agent (PR #48, conditional on provider rules, which were checked);
- no e-mail subscription, outreach or incentives yet;
- a documented API is not exempt from robots.txt.

**Still open:**
1. **`api.weather.gov` and robots.txt.** It is `Disallow: /` for every agent, and #41's `/api/conditions` uses it. Production use needs your decision. Options: ask NWS, use only their web pages as links, or accept the documented API. My features make no NWS calls.
2. **ERDDAP robots.txt (SCCOOS/CeNCOOS).** The pipeline's shore-station and model data come from it. Same question, from the source audit.
3. **Outreach package** (03 §2, §9): intermediaries, incentive amount and payment method, IRB route, and who holds contacts. Drafts only.
4. **Merging #48** (the watcher User-Agent) into `main`.
5. **Integration of #47 into #41** (or after #41 merges): your review of the change block and the kept briefing ([07](07-benchmarks.md) §Costs lists the trade-offs).
