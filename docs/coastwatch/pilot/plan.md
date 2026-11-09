# Neritic: plan for the Thursday 2026-11-12 pitch pass

Written 2026-11-08. Sources: the repo (origin/main 20fd5406e, live demo v1, data feed, draft PRs #49/#72/#74/#89), four research passes (fishermen needs and official data, forecasts and data sources, Stanford and grants, production stack and compliance), and the owner's answers.

---

## 1. What the research changed

1. **Bloom risk hits fishermen every day through the official pipeline, not the map.** What decides a working day in Monterey Bay right now is mostly official status, not chlorophyll:
   - **Commercial Dungeness: delayed until at least 2027-01-01** in Zones 1–5, Oregon to Point Conception, **because of humpback entanglements, not domoic acid (DA).** The next risk assessments are Dec 15–18 and Jan 13–15. Recreational crab opened Nov 7 with traps banned.
   - **Crab toxin tests in Monterey and Half Moon Bay are clean:**
     - BMA G Dungeness averaged 5.5 ppm on 9/28.
     - Pillar Point, Pigeon Point and Duxbury averaged 3–4 ppm.
     - Usal on the north coast averaged 38.5 ppm (max 99), with 50% over the action level.
   - **Monterey Bay anchovy and sardine restriction**, Pigeon Point to Point Lobos, since Sep 11. Anchovy viscera hit 1,200 ppm in Block 509 in August.
   - The 2026 mussel quarantine order ran to Oct 31. **No lift notice found yet.**
   - Commercial salmon has reopened, with trip limits.
2. **The highest-value data is weekly, official and buried in PDFs.**
   - CDPH's rolling `CrabDAResults.pdf` has six individual ppm values per sampling set, by BMA and port. It parses cleanly with `pdftotext -layout`.
   - CDFW whale-entanglement (RAMP) decisions come as PDFs. Zone polygons exist as BIOS ds3120.
   - Fishermen ask for exactly this (Pomeroy & Culver 2026, 20 CA crab fishermen): *"a timely, accessible, and easily understood notification system"* and clarity about how decisions are made.
3. **Whale rules must sit beside bloom risk.** Whale entanglements, not toxin, are delaying this season. There is no whale-sightings data for Monterey (Whale Safe covers only the Santa Barbara Channel and SF, has no API, and is built for ship strikes). The defensible layer is **RAMP zone polygons plus the current CDFW action, transcribed from CDFW.**
4. **Fishermen pay for tools that help them find fish, not avoid risk.**
   - FishTrack costs about $8–15 per month for SST and chlorophyll charts.
   - PredictWind costs $29–499 per year for wind.
   - The habit that brings someone back every day is **weather, sea state, tides and water temperature**. Bloom risk and compliance ride along.
5. **An honest 8-day outlook already exists.** It is #74's `chl_outlook_8d` (climatology + AR(1)):
   - Held-out RMSE (ln): 0.760, against 0.857 for climatology and 0.892 for persistence (n = 185k).
   - It forecasts **chlorophyll, not toxin**.
   - C-HARM gives toxin-risk probability only out to day 3, because WCOFS stops at 72 hours.
   - The PINN audit (#49) found no network beats climatology + AR(1), so the app should use that baseline, and saying so is the honest science story.
6. **The data feed is not actually broken upstream.**
   - All six "404" NOAA satellite datasets exist. coastwatch.noaa.gov returns a generic 404 for *any* ERDDAP error, including asking for a date that has no granule. A real gap happened on Oct 31–Nov 1.
   - Fix: read each dataset's time axis first and request only times that exist.
   - C-HARM itself is about 6 days stale, failing its own staleness test.
7. **Fact-check of the pasted research:** use these corrected numbers in the pitch.

   | Claim | Use instead |
   |---|---|
   | "$5–15k per trip" | Wrong for most of the fleet. A 42-ft crab boat uses about $135 of fuel per trip, and operating costs are about 45% of gross. Drop the figure, or quote ranges by fishery. |
   | ">$100M DA damage" | "About $48M estimated for CA Dungeness in 2015–16 (the state disaster request)". Later analysis found actual losses smaller. |
   | "$50–100 per month willingness to pay" | No evidence. Comparable tools charge $8–15 per month. Don't claim a price; it's a free public-good tool. |
   | "Weekly pre-season lab testing" | Partly wrong. Dungeness is tested about a month before the opener and retested only when over the limit. Sites are sampled every 1–4 weeks. |
   | "15–40% gear reductions" | Real, but the range is 15–50%, and reductions are one RAMP tool alongside depth limits, closures and delays. |

8. **The pitch to Fringer is a natural fit.**
   - He already leads a Woods "Big Ideas for Oceans" HAB forecasting project with SFEI (David Senn), motivated by the 2022 Bay *Heterosigma* bloom. It uses machine learning plus model fusion.
   - His student Sarah Chang works on ML for HABs. That she is the student at the meeting is an inference, not confirmed.
   - Big Ideas for Oceans (up to $150k) is "expected to reopen in November", and it requires a faculty PI.
   - Almost every grant needs a faculty PI and an institution, or a fiscal sponsor. That is the structural reason to partner.

## 2. Product thesis for the MVP

> **Neritic tells a California commercial fisherman, in under a minute every morning, what is open, what the water and weather are doing, what is changing, and what that means for their next trip. It sends the change to their phone before they have to go looking.**

Strict hierarchy (unchanged): official status → measured tests and observations → forecasts → advice.

**Status colours, recommended:** red means an official closure or restriction; amber means a rising signal to watch (tests going up, a high C-HARM probability, a RAMP assessment due). The third state is **"No restriction on record" in neutral blue-grey, never a green "safe"**. Toxin is invisible and notices can lag. A green light would be the one claim that could hurt someone and damage credibility with scientists. This is a decision for you; see §6.

## 3. What gets built (Thursday scope)

### A. The front door: **Today at <port>**, mobile-first. This is the centerpiece and the ambitious visual transformation.
A single scrolling page for each port. Moss Landing, Monterey, Santa Cruz and Half Moon Bay come first; the rest of the state uses the same template.
1. **Status strip:** one line per fishery the user selected (Dungeness, rock crab, salmon, groundfish, anchovy/sardine, squid, shellfish), each with a red, amber or neutral chip, its source and the date checked. Example: "Dungeness: commercial delayed to ≥ Jan 1 (whale entanglement). Next CDFW assessment Dec 15–18."
2. **Sea and weather go/no-go context:** NWS zone forecast for 5 days (PZZ535 for Monterey), plus buoy 46042, tides from CO-OPS station 9413450, and water temperature. This is the daily-habit layer.
3. **Crab toxin tests:** a sparkline for each BMA site of the six individual values against the 20/30 ppm action levels. "Two consecutive clean sets" is shown as progress, with the date of the next expected result. Plain language goes first, for example "Monterey Bay crab tested clean on Sep 28 (avg 5.5 ppm, limit 30)."
4. **Bloom outlook:**
   - C-HARM probabilities for days 0–3, followed by the 8-day chlorophyll outlook, presented as one visual timeline.
   - Days 4–8 are labelled "conditions outlook (chlorophyll), not a toxin forecast".
   - It falls back gracefully when C-HARM is stale.
5. **Whale rules:** the RAMP zone the port sits in, the current action (delay, depth limit, gear reduction or advisory), and the next assessment date.
6. **What changed since your last visit:** reuse SinceLastCheck (#47).
7. **Actions:** Save offline (the existing field pack), Set an alert, Ask Neritic, and a pre-trip checklist.

The existing Explore map, Research and Sources stay, one tap away. The map becomes a supporting view, not the front door.

### B. Data pipeline
- **Fix the refresh** (issue #95): request only times on each ERDDAP time axis, treat a 404 for a missing granule as "no new data", and keep the 90-day OLCI window in mind. The aim is a green refresh before anything else.
- **New source: CDPH crab and seafood DA results.**
  - Fetch the PDF (the CDPH certificate chain needs an explicit trusted bundle, not `-k`), parse it, validate it, and test it against recorded fixtures.
  - Output goes to a **sidecar file outside the manifest**, like `derived/index.json`. The deployed web app rejects unknown manifest fields, so the manifest is never changed.
- **New curated source: RAMP and season status.** This is reviewed YAML, the same way notices work: zone polygons come from BIOS ds3120, the current action and next assessment date are transcribed, and records stay `pending_human_review` until you run the review.
- **Bring #72 (SF Bay scope) and #74 (8-day outlook, SST) forward.** Rebase them, rerun staging, and use them in the demo build. Publishing to production needs your approval.
- **Risk to check:** the 8-day outlook is driven by the MODIS-Aqua NRT 8-day composite. MODIS-Aqua is late in its mission, so confirm the feed's continuity and plan a VIIRS or OLCI replacement.

### C. Accounts, personalisation and alerts (Supabase)
- New project in **us-west-1**. Use **Pro ($25 per month)**, because the free tier pauses after a week idle, which is fatal for a pilot.
- **Sign-in by email one-time code.** Phone sign-in waits for SMS registration.
- **Tables, all with row-level security:**
  - `profiles`: role, fisheries, home port, vessel size class; age 13+ attestation.
  - `saved_ports`
  - `alert_subscriptions`: channel and trigger type.
  - `notifications_sent`: used as the idempotency key.
  - `feedback`: "was this useful?" on every card.
  - `trip_logs`: deliberately minimal (date, port, fishery, outcome band, observed conditions, useful yes/no).
- **Alert fan-out:**
  1. A scheduled Supabase job (pg_cron + pg_net) calls an Edge Function after each data refresh.
  2. The function compares the new data with the old for each subscription: a status change, a new toxin result over or under the limit, a new RAMP action, or a sharp jump in C-HARM risk.
  3. It sends a web push. On iOS that only works once the app is installed to the home screen, so add an install prompt.
- **Legal pages, live on day 1** (the SMS registration needs their URLs):
  - Privacy policy: CalOPPA-compliant, with data categories, retention, deletion and Do-Not-Track.
  - Terms of use: as-is, not for navigation, not an official source, the user stays responsible, AI may be wrong.
  - SMS terms.
  - Start from the CC-licensed Automattic or Basecamp templates. Have a person read them; this is not legal advice.
- Product analytics should be privacy-preserving and cookieless, with first-party events.

### D. Ask Neritic: grounded assistant (web chat now, SMS soon)
- Build on #49's evidence-grounded prototype, which has a deterministic guard and passes 22/22 golden and 14/14 adversarial questions.
- Use the Claude API (Sonnet 5.5 for answers, Haiku 5.5 for routing):
  - It retrieves only Neritic's own published JSON for the port and date in question.
  - Every answer must carry citations (record id, source, timestamp) that resolve. Answers whose citations don't resolve are rejected.
  - It always states how old the data is.
  - Fixed refusal templates cover "is it safe to eat / safe to go out / bar crossing" and point to CDPH, CDFW, NWS and USCG.
- Logging is consented, with 90-day retention. Add rate limits per user and per number, and a daily cost cap.
- **SMS:** register a Twilio **sole-proprietor 10DLC** campaign on day 1 (about $21 up front plus $2 per month, approved in about 1–3 weeks). For Thursday, demo SMS on verified team phones in trial mode. Present it honestly as "in carrier review".

### E. Visual transformation
- Day 1 starts with a short design brief and a **choice of 2–3 directions** for the Today page, mocked as phone and desktop screenshots. Building starts only after you pick one.
- The goal: a calm marine-instrument look with large type, one idea per card, and "what it means" before "the number". It must read one-handed on a rocking deck and in bright sun (high contrast, a dark mode for 4 a.m.).
- The current Neritic mark stays.
- Gates: axe, phone and desktop screenshots, real iPhone and Android checks, and the 700 KB bundle budget.

### F. Pitch package for Thursday
- A **10-minute deck**:
  1. Problem, with a fisherman's quote from Pomeroy 2026.
  2. Live demo: "Monday 5 a.m. at Moss Landing".
  3. The honest science slide: the PINN matches but does not beat ConvLSTM; climatology + AR(1) is the baseline; chlorophyll is not toxin.
  4. Roadmap.
  5. The asks.
- A **one-page leave-behind** and a **demo script with a fallback** (screenshots in case of bad wifi).
- **Asks for Fringer:**
  1. A methods critique and a shared baseline with the Bay HAB ML project.
  2. Faculty PI or mentor on a Big Ideas for Oceans proposal; offer to draft it.
  3. Data and model access (SUNTANS / SFEI Bay hindcasts), and showing their validated Bay HAB likelihood maps in Neritic.
  4. Introductions: David Senn (SFEI), CeNCOOS, the C-HARM team (Kudela, UCSC), and California Sea Grant extension for fishermen interviews.
  5. A defined student role (directed study or RA) and a venue for the corrected PINN re-analysis.
- Also ask:
  - Would Stanford host the notice and science review (closing B1 and B2)?
  - Can Hopkins Marine Station or Moss Landing Marine Labs connect you to fishermen for 10 interviews?

## 4. Not by Thursday (said out loud in the pitch)
- Public SMS (in carrier review).
- Verified notices, unless you run `cwp review-official` after checking the sources.
- Independent scientific review.
- A whale-sightings layer for Monterey (no data exists).
- Processor or wholesaler features, payments, a native app.
- Ex-vessel prices (no timely official feed).
- Neritic's own toxin forecast.

## 5. Schedule

| Day | Claude | You |
|---|---|---|
| **Sun night / Mon** | Fix the refresh (#95); CDPH parser; RAMP/season YAML; Supabase schema with row-level security; legal pages; 2–3 design directions for Today | Pick a design direction; approve spend; create the Supabase project (or let Claude script it); **submit Twilio 10DLC**; review notices with the packet Claude prepares |
| **Tue** | Build the Today page; rebase #72/#74 into a demo branch; sign-in, saved ports, alerts and web push; feedback and trip log | Try it on your phone; send notes |
| **Wed** | Ask Neritic (web) plus SMS trial; polish; real-device and accessibility pass; Linux CI; **private preview** deploy (needs your OK) | Real-phone walkthrough; rehearse the demo |
| **Thu AM** | Deck, one-pager, demo script and screenshots; code freeze | Pitch |

Engineering rules: one worktree per track, a draft PR per coherent milestone, Linux Playwright before every push, nothing merged or deployed to production without your explicit OK.

## 6. Decisions you need to make (short answers are fine)
1. **Spend:** Supabase Pro at $25/mo, Twilio about $25, the Claude API with a cap (suggest **$30 for the week**), and possibly a domain. OK?
2. **Status colours:** red / amber / "no restriction on record" with no green "safe", as recommended. Or do you want a literal traffic light?
3. **What Fringer sees:** a private preview link (recommended), or update the public demo (that needs a separate GO)?
4. **Data:** may the demo build read **staging** data so it includes #72/#74 and the CDPH and RAMP layers? Publishing to production waits until later.
5. **Notices:** can you spend about 45 minutes on Monday or Tuesday checking the review packet and running `cwp review-official`? Otherwise everything stays "Not verified" on Thursday.
6. **Redesign scope:** transform the Today page and app shell, and restyle the other pages to match only where time allows (recommended). Or a full redesign of every page?

---

## 7. The implementation prompt (paste into the next session or reply "go")

> Implement the Neritic Thursday pass described in `neritic-thursday-plan.md` §3 (A–F), in the order of §5, following CLAUDE.md (one worktree per track; draft PRs; Linux Playwright before every push; no merge, production data publish or deploy without my explicit OK; never mark notices verified; never add manifest fields; never present chlorophyll as toxin or observations as forecasts; only same-pipeline research numbers).
> Decisions: [answers to §6].
> Start with: (1) fix the data refresh (#95) and get one green staging run; (2) put 2–3 design directions for "Today at <port>" in front of me as screenshots before building. Then build the CDPH crab-DA and RAMP/season sources (sidecar files, staging), the Today page, Supabase accounts/saved ports/alerts/web push with privacy policy, terms and SMS terms, Ask Neritic (grounded, cited, refusal templates, daily cost cap), and the pitch deck, one-pager and demo script. Report progress at each milestone with preview links and screenshots; stop and ask before anything outward-facing.
