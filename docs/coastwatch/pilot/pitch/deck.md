# Neritic: 10-minute deck (10 slides)

Number sources: plan = [`../plan.md`](../plan.md); #74 = draft PR #74 metrics table; AUDIT = `research/pinn_audit/AUDIT.md` on PR #49; README = repo paper §4.1. Visuals name app screens; screenshots captured by the integrator (see demo-script.md).

---
### 1. Every morning, a fisherman decides
**On-slide:** "A timely, accessible, and easily understood notification system." — 20 California crab fishermen (Pomeroy & Culver 2026)
**Notes:** Introduce yourself in one sentence. Bloom risk reaches fishermen through official decisions, not maps. They asked for timely, plain notification and clarity on how decisions are made (plan §1.2).
**Visual:** the quote over a dock photo or the Neritic mark.

### 2. This season: status beats chlorophyll
**On-slide:** Commercial Dungeness delayed to at least Jan 1, 2027. Cause: whale entanglements, not toxin. Monterey crab tested clean.
**Notes:** Zones 1–5 delayed by humpback entanglement; next assessments Dec 15–18 (plan §1.1). Monterey BMA crab averaged 5.5 ppm on 9/28, limit 30 (CDPH via plan §1.1). Northern anchovy in Monterey Bay: no take for human consumption since Sep 11 (domoic acid). What decides the day is status. Stakes: about $48M estimated CA Dungeness loss in 2015–16 (state disaster request; later analyses found smaller actual losses).
**Visual:** Today status strip, Dungeness red chip.

### 3. The information exists, scattered
**On-slide:** CDPH PDFs. CDFW whale decisions. NWS. NOAA C-HARM. Five sites, no alerts, no phone-first view.
**Notes:** CDPH crab results are a rolling PDF; RAMP decisions are PDFs; RAMP tools include gear reductions of 15–50% alongside depth limits, closures and delays (plan §1.7). Paid tools (FishTrack ~$8–15/mo) help find fish, not track rules (plan §1.4). Gap: one place, in order of authority, pushed to the phone.
**Visual:** collage of source pages vs one Today screen.

### 4. Live demo: Monday 5 a.m. at Moss Landing
**On-slide:** Today at Your Port. Status, tests, sea and weather, bloom outlook, whale rules, what changed.
**Notes:** Run demo-script.md, 4 minutes. Say "Pilot demo" up front.
**Visual:** live app, /today?place=port-592.

### 5. How it stays honest
**On-slide:** Official status, then measurements, then forecasts, then advice. No green "safe". Notices marked "Not verified".
**Notes:** Fixed hierarchy (plan §2). Third state is "No restriction on record", neutral, because toxin is invisible and notices lag. Ask Neritic cites every answer and refuses "is it safe to eat / go out". Notices stay "Not verified" until a person checks the sources.
**Visual:** status chip legend; a "Not verified" badge.

### 6. The science, honestly
**On-slide:** 0–3 days: NOAA C-HARM. Next 8-day period: chlorophyll outlook, higher or lower than usual; RMSE 0.760 vs climatology 0.857. Chlorophyll, not toxin.
**Notes:** Held out 2019-01 to 2021-06, n = 185,364, RMSE in ln: clim+AR(1) 0.760, climatology 0.857, persistence 0.892; climatology minus model 95% CI [0.076, 0.121] (#74). The outlook is one mean over the next 8-day period, not a day-by-day forecast; a typical miss is a factor of two, so the app only says higher or lower than usual (at least 1.5× either way). Does not cover the bays. C-HARM stops at 72 h because WCOFS does (plan §1.5).
**Visual:** the Bloom outlook section: the C-HARM chart with its run date, and the outlook's scale against usual.

### 7. What my own audit found
**On-slide:** My PINN matches, does not beat, the ConvLSTM. A simple baseline matches both. Needs independent reproduction.
**Notes:** Published test RMSE 0.801 for both (README §4.1). My audit (AUDIT, 2026-10-14) reports *possible* issues: physics term contributes ~3e-5 of the loss; physics on/off identical to 5 decimals across 3 seeds; clim+AR(1) on that pipeline scores 0.762, both networks within its bootstrap interval. These require independent reproduction; the paper is unchanged until then. This is why I want your critique.
**Visual:** simple three-bar chart (persistence 0.889 / ConvLSTM 0.801 published / clim+AR1 0.762) from AUDIT §1, captioned "single audit, unreviewed".

### 8. Pilot: Monterey Bay ports
**On-slide:** Moss Landing, Monterey, Santa Cruz, Half Moon Bay. Ten dockside interviews. Alerts, offline, feedback on every card.
**Notes:** Interviews first (interview-script.md); measure: do they return daily, are alerts timely, what they distrust. No paid tier; public-good tool. SMS is in carrier review.
**Visual:** port picker / saved ports.

### 9. Roadmap
**On-slide:** Now: Monterey pilot. Next: Bay coverage with partner models. Later: independent science review, statewide.
**Notes:** Not yet: public SMS, verified notices, independent review, whale sightings for Monterey (no data exists), Neritic's own toxin forecast (plan §4). Bay layer would show partner-validated maps, attributed.
**Visual:** three-column timeline.

### 10. Asks
**On-slide:** Methods critique. Faculty PI on a Woods proposal (Big Ideas for Oceans or EVP). Data access. Introductions. A student role.
**Notes:** Walk the ranked list in fringer-brief.md; offer to draft the proposal; ask which one he'd say yes to today.
**Visual:** contact + private preview link.
