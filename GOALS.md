# Product goals — California fisheries decision platform

This document consolidates goals from **`GOALS.pdf`** (desktop). It defines what “success” looks like beyond a research chlorophyll map: a **compliance-first, operational** tool for California fishermen and coastal stakeholders.

---

## 1. What the product should fundamentally be

The right goal is **not** “a pretty HAB map.” It is a **compliance-first fisheries decision platform** that helps users decide **whether** to fish, **where** to fish, **what to avoid**, **what risk** they are taking, and **how confident** the system is.

California already has official layers that affect fishing—commercial regulations, closures, MPAs, health advisories, fish-ticket/reporting systems, and HAB forecast products—but they are **fragmented**. A strong product **unifies** those sources into one operational interface.

**Core promise:**

> Show me legal, lower-risk, economically viable fishing zones, explain why, and alert me when conditions change.

That is stronger than only showing chlorophyll, because HABs affect fisheries through **toxin risk**, **closures**, and **operational uncertainty**. NOAA notes that domoic-acid-producing blooms can restrict access to important commercial and subsistence fisheries; California’s official advisory pages are intended as information sources for fishermen and the industry.

---

## 2. The most important product rule: strict hierarchy

The app needs a **strict hierarchy**:

1. **Official rule / compliance engine**  
2. **Observational data**  
3. **ML predictions**  
4. **User-specific recommendations**

If a zone is **closed**, restricted by an **MPA**, under a **health advisory**, or otherwise **noncompliant**, the recommendation engine must **never** surface it as “good,” no matter what the chlorophyll or catch model says. Official regulations, health advisories/closures, and MPA maps **override** every predictive layer.

---

## 3. Primary user workflows

### A. Pre-trip planning

Before departure, users need:

- Today’s HAB and toxin risk  
- Legal/open zones  
- Likely productive areas  
- Expected species conditions  
- Weather/ocean context  
- Travel/fuel implications  
- **“What changed since yesterday”**

### B. On-water decision support

Underway, users need:

- Current position relative to bloom/risk/closure zones  
- Fast visual cues  
- Alternate candidate zones  
- Simple compliance checks  
- **Push alerts** if conditions change  

### C. Post-trip logging and learning

After the trip:

- Target species, catch outcome, observed water conditions, bloom observations  
- Trip cost / effort band  
- Whether the recommendation was useful  

This supports **personalization** and **model improvement**, aligned with how California uses logbooks, landings, fish tickets, and e-logs.

---

## 4. Complete feature set (target)

### A. Map and geospatial experience

- Immersive California coast map  
- Default **coastal analysis ribbon** (~first 10 miles from shore); optional offshore expansion  
- Harbor/port entry points  
- Fast layer toggles; click/tap zone inspection  
- **Timeline scrubber**: yesterday / now / next 72 hours  
- **Split-screen** compare: past vs present  

**Map layers (minimum ambition):**

- HAB risk layer  
- Chlorophyll-a layer  
- Bloom extent / footprint polygons  
- Toxin-risk proxy layer  
- MPA boundaries  
- Active closure / advisory zones  
- Legal fishery zones by species/gear/season  
- Species opportunity layer  
- Weather/ocean hazards  
- Recent user observations  
- Vessel route / saved zones  

Rationale: C-HARM and the California HAB Bulletin separate bloom likelihood, toxin risk, and forecast timing; California publishes MPAs and advisories separately—the UI must **combine** them into one decision surface.

### B. “Today’s Coastal Brief” panel

First screen content (besides the map):

- Overall risk today for selected region  
- Current advisories/closures  
- Top 3 candidate zones  
- Highest-risk areas to avoid  
- Bloom direction/expansion  
- Species at risk  
- Confidence score  
- Last update time and **source badges**  

### C. Zone detail panel (on click)

- Legal status: open / caution / closed  
- HAB risk; toxin-related concern level  
- Bloom trend over 24–72h  
- Likely target species; species at risk  
- Confidence / uncertainty  
- Estimated trip score; estimated economic score  
- **Why** the zone was ranked this way  
- Which **rules or advisories** are active  

### D. Recommendation engine UI

- **Ranked shortlist**, not a single “best” location  
- Per candidate: legality, HAB safety score, catch opportunity, transit/fuel score, uncertainty penalty, overall trip score, key reasons, key warnings  

### E. Alerts and notifications

Subscriptions for: port/county/segment, species, bloom growth, closure/advisory changes, saved zone risk tier changes, weather/hazards, lower-risk alternatives. C-HARM (daily nowcasts/short forecasts) and California health-advisory updates motivate this.

### F. Accounts and personalization

Profile: vessel, home port, trip radius, gear, permit type, targets, schedule, risk tolerance, notification channels.  
Saved: favorite ports/zones, species watchlist, default layers, recommendation style (conservative / balanced / aggressive).

### G. Private fishery logbook

Fields: date/time, ports, area fished, target/actual species, catch, bycatch, duration, fuel/effort, bloom/water observations, photos, notes on dead fish/foam/discoloration/smell/mammal distress, whether app advice matched reality.

### H. Compliance and regulation module

Dedicated module: opening/closure calendar, area restrictions, MPA overlay, species/gear checks, active health advisories, “why legal / not legal,” downloadable compliance summary, regulation change log, **official linkouts**.

### I. Community / coastal support

Public advisory view, seafood safety education, bloom explainers, species health notes, community observations, harbor/co-op dashboards, local trends, impact history (aligned with OEHHA-style guidance).

---

## 5. How to use ML (stacked, not one black box)

| Layer | Role |
|--------|------|
| **HAB risk nowcast/forecast** | Bloom probability, toxin-risk proxy, nearshore HAB score, uncertainty, 1–3 day horizon (C-HARM-style inputs: circulation, ocean color, statistics) |
| **Catch opportunity** | Given legal + lower-risk zones, where catch likelihood is stronger (species, season, habitat, fronts, landings patterns, logs, range) |
| **Economic-risk** | Trip utility, downside risk, conservative alternatives, “not worth going today” under safety/compliance constraints |
| **Species-at-risk / impact** | No overclaim on exact toxic burden; species commonly affected, active concerns, intersection with blooms, confidence notes |
| **Personalization** | Secondary to rule engine; learns from logs |

**Avoid early overclaim:** exact catch quantity, exact toxin by species everywhere, weak rules parsers as “legality,” precise offshore routing without full hazard integration.

---

## 6. Data to ingest (online)

- **Official:** commercial regulations, openings/closures, health advisories, MPAs, permit/reporting context (CDFW, related state pages).  
- **HAB/ocean:** C-HARM or comparable, California HAB Bulletin, HABMAP, satellite chl/SST, currents, wind/upwelling, SCCOOS/CeNCOOS, gliders/buoys where available.  
- **Weather:** NWS marine zone forecasts, small craft advisories, wind/wave, hazardous alerts.  
- **Fishery intelligence:** CDFW Marine Fisheries Data Explorer, landings summaries, seasonal/species trends.  
- **User-generated:** private logs, observations, catch outcomes, photos, location (with consent).

---

## 7. UI non-negotiables

- Mobile-first, sunlight-readable contrast  
- **Low-data mode**, fast maps  
- Clear legends, large tap targets, minimal jargon  
- **Explicit source badges and timestamps**  
- **Offline cache** of latest maps/briefings  

**Screen hierarchy (in order):**

1. Can I fish here legally?  
2. Is this area risky for bloom/toxin?  
3. Is it worth going economically?  
4. What species are implicated?  
5. How confident is the system?  

**Suggested layout:** left rail (Today’s Brief, alerts, top zones, bloom change); main map; right drawer (zone details); bottom strip (timeline, layers, compare).

---

## 8. Trust, privacy, governance

With accounts, logs, and location: privacy notice, consent, export/delete, role-based access, encryption, aggregation/anonymization for shared analytics, **no public display of private hotspots**, audit logs for inputs/sources. **User trip data private by default.** (Note: California privacy rules evolve; treat compliance as a product requirement.)

---

## 9. Stakeholder value

- **Fishermen:** fewer fragmented sources, faster go/no-go, fewer wasted trips, clearer compliance, personalized zones.  
- **Harbors/co-ops:** aggregate bloom risk, operations, regional trends.  
- **Scientists/agencies:** auditable explanations, community observations, model feedback.  
- **Coastal communities:** public info, safety guidance, impact dashboards.

---

## 10. Phased roadmap (from goals doc)

| Phase | Focus |
|--------|--------|
| **1 — Operational MVP** | Immersive map, coastal ribbon, official closures/advisories/MPAs, HAB risk + bloom extent, zone detail, weather/hazard layer, basic alerts; light or no accounts |
| **2 — Personalized platform** | Accounts, private logs, saved ports/species, recommendation engine, simple economic scoring, compare/forecast mode |
| **3 — Advanced** | Personalized ML, fuel/trip utility, co-op dashboards, community observations, model feedback, reporting helpers |

---

## 11. Bottom line: four integrated systems

To be optimally functional, the product needs:

1. **Compliance engine** — official rules, MPAs, closures, advisories, reporting context  
2. **Geospatial intelligence engine** — HAB risk, bloom movement, chlorophyll, weather/ocean  
3. **Recommendation engine** — legal + lower-risk + species-aware + economically sensible zones  
4. **Fisherman operations layer** — accounts, logs, alerts, personalization, trip history, explanation  

That combination turns a **science demo** into a **real, useful platform**.

---

## 12. High-impact features for *this* repository (grounded in the goals above)

The following are **concrete** upgrades to `habs-forecast` / `coastwatch-web` / `dashboard` that **directly implement** language and priorities from the goals document. They are ordered roughly by leverage vs. effort.

### 1. **Compliance-first layer ordering in the UI (Goal §2, §4H)**

- **What:** Enforce the hierarchy *in the product*: render **official** layers (MPA, closure/advisory polygons or links) **above** ML/chlorophyll; label any model output as “non-regulatory.”  
- **Why the doc says so:** *“The recommendation engine must never surface [noncompliant zones] as ‘good,’ no matter what the chlorophyll… says.”*  
- **How here:** Ingest GeoJSON or WMS for **CDFW MPA** boundaries and **CDPH advisory** summaries (even as link-out panels + static weekly snapshots at first); add a **“Legal / advisory status”** strip that gates copy in the Coastal Brief.

### 2. **“Today’s Coastal Brief” as the home panel (Goal §4B, §7)**

- **What:** A fixed **first panel** with: overall risk for selected region, **top 3 candidate zones** vs **areas to avoid**, **last update + source badges**, and **“what changed since yesterday”** (diff of last two `snapshot.json` or two GIBS dates).  
- **Why:** *“Users will not want to decode raw satellite layers every time.”*  
- **How here:** New React component in `coastwatch-web` fed by manifest + optional second snapshot; show CDPH/NOAA links as badges.

### 3. **Timeline + forecast strip (72 h) (Goal §3A, §4A, §5A)**

- **What:** Scrubber for **yesterday / now / next 72h** using **C-HARM** or **California HAB Bulletin** URLs + your **PINN/ConvLSTM** multi-step export if available.  
- **Why:** *“Timeline scrubber for yesterday / now / next 72 hours”* and HAB nowcast/forecast model outputs.  
- **How here:** Backend job writes **time-stamped** `snapshot.json` + PNG per step; front-end swaps `ImageSource` URL or uses a small manifest array of `{time, overlayUrl}`.

### 4. **Zone detail + explainability drawer (Goal §4C, §4D)**

- **What:** Click map → drawer with **legal placeholder**, **HAB tier from `regional_algae`**, **species-at-risk text**, **confidence** (from your **uncertainty** branch when trained), and **“why ranked this way”** bullet list (rules: tier + distance to port + data age).  
- **Why:** *“If the app says ‘avoid,’ the user should see whether that came from an official closure, a bloom trend…”*  
- **How here:** Mapbox `onClick` + query rendered features; bind to `regional_algae` + `fisheries_context.json` + static compliance links.

### 5. **Ranked candidate zones (not one “best” pin) (Goal §4D)**

- **What:** Shortlist **3–5 polygons or grid cells** with columns: legality (TBD), HAB safety, catch opportunity (heuristic), transit/fuel proxy (distance from chosen port), uncertainty penalty.  
- **Why:** *“Do not output a single ‘best’ location. Output a ranked shortlist.”*  
- **How here:** Python script scores ocean cells from NetCDF + harbor distance; writes `candidates.json` for the web app.

### 6. **Ingest C-HARM / HAB Bulletin / HABMAP (Goal §6)**

- **What:** Treat **C-HARM**, **California HAB Bulletin**, and **HABMAP** as **authoritative HAB layers** alongside your model.  
- **Why:** *“C-HARM and the California HAB Bulletin… your UI has to combine those into a single decision surface.”*  
- **How here:** Add optional Mapbox **raster or WMS** source for C-HARM outputs where public tiles/API exist; sidebar deep links to bulletin + HABMAP station pages by region.

### 7. **Weather / hazard layer (Goal §3A, §4A, §6)**

- **What:** NWS marine zone forecast link + optional **GeoJSON** or raster for small-craft/hazard (or embed NWS mobile-friendly page).  
- **Why:** Pre-trip planning and *“weather/hazard layer”* in Phase 1 MVP.  
- **How here:** Region-based `forecast.weather.gov` links in Brief + one optional overlay if you add a simple API route.

### 8. **Catch-opportunity + economic-risk scores (v1 heuristics) (Goal §5B, §5C)**

- **What:** **Zone-level catch-opportunity score** from public landings summaries (CDFW Data Explorer aggregates) + chlorophyll front proxy; **economic score** = opportunity minus distance/fuel penalty minus uncertainty penalty — **no** claim of exact revenue.  
- **Why:** *“Maximize expected value under safety and compliance constraints,”* not raw catch max.  
- **How here:** Offline CSV → JSON lookup by region; merge in `export_map_snapshot.py` or a new `scripts/build_scores.py`.

### 9. **Alerts MVP (Goal §4E)**

- **What:** Email or **web push** stub: “advisory page changed” (diff hash weekly) + “your saved region tier changed” (compare two snapshots).  
- **Why:** *“Notification workflows are central to real usability.”*  
- **How here:** GitHub Action or cron compares CDPH page hash + local `snapshot.json` tier; sends SendGrid/Resend (optional).

### 10. **Accounts + private logbook (Phase 2) (Goal §4F, §4G)**

- **What:** Auth + Postgres/SQLite for trip logs (fields from §4G).  
- **Why:** *“This creates the dataset needed for personalization and model improvement.”*  
- **How here:** Next.js route handlers + Prisma; keep **private by default** (Goal §8).

### 11. **Mobile-first + low-data + offline cache (Goal §7)**

- **What:** Service worker caching `snapshot.json`, `overlay.png`, and last Brief; reduce Mapbox style weight; **“low data”** toggle turns off GIBS/rasters.  
- **Why:** *“Mobile-first… low-data mode… offline cache of most recent maps/briefings.”*  
- **How here:** `next-pwa` or Workbox; already partially aligned with large tap targets and legends in Coastwatch.

### 12. **Uncertainty on the map (Goal §5A, `pinn` uncertainty code)**

- **What:** Train/deploy **MC dropout / ensemble** from `pinn_model_uncertainty.py`, export **p10/p50/p90** or variance grid; optional second map layer “confidence.”  
- **Why:** *“Uncertainty/confidence”* as first-class output.  
- **How here:** Extend `export_map_snapshot.py` to write `overlay_uncertainty.png` + legend.

### 13. **Compliance module v0 (Goal §4H)**

- **What:** Dedicated page: calendar placeholders, MPA map link, health advisory link, **downloadable PDF** “trip area checklist” (links only, no auto-legal opinion).  
- **Why:** *“Dedicated module, not a footnote.”*  
- **How here:** Static Next.js page + `jspdf` like WellWatch.

### 14. **Audit / provenance footer (Goal §8, scientist stakeholders)**

- **What:** Every recommendation lists **data sources and timestamps** used for that view (model ckpt id, GIBS date, bulletin fetch time).  
- **Why:** *“Auditable explanation layer… audit logs for recommendation inputs.”*  
- **How here:** Extend `snapshot.json` with `provenance: {}` written at export time.

---

*End of goals consolidation. Update this file when `GOALS.pdf` changes.*
