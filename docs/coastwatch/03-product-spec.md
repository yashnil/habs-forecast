# 03 — Product specification

## 1. Who it is for and what it must do

**Primary users**

| User | Typical question | Device / context |
|---|---|---|
| Commercial crab and trap fishermen (Dungeness, rock crab, lobster) | "Is my season delayed or my area closed for domoic acid? Is it getting better or worse?" | Phone at the dock, early morning, bright sun or dark cabin, weak signal |
| Commercial / charter finfish operators | "What are conditions near my port this week? Anything official I need to know?" | Phone or laptop the night before |
| Shellfish growers and recreational harvesters | "Is there a quarantine or advisory here?" | Phone |
| Port managers, harbor districts, Sea Grant extension, fishermen's associations | "How exposed is our port to bloom closures? What's happening up and down the coast?" | Laptop; sharing screenshots and links |
| Researchers / journalists | "Where does the data come from? How good is the model?" | Laptop |

**Core promise (MVP)**

> For any California port: the official status first, the official bloom forecast second, what satellites see third, and how much the port's fishing economy has historically depended on bloom-sensitive species — each with its source and time, in under ten seconds on a phone.

**Not the promise:** where to fish, what to catch, whether seafood is safe to eat, or expected revenue. See `05-science-and-safety.md`.

---

## 2. Information architecture

```
/                       Live Ocean Map  (default; port drawer opens over it)
/ports                  Port index (search, nearest-to-me)
/ports/[portId]         My Coast — port page (shareable URL)
/bloom                  Bloom Intelligence            (phase 2; MVP has a stub explainer)
/fisheries              Fisheries & Economic Exposure (phase 2; MVP has port-level card only)
/models                 Research models & model cards (phase 3)
/data                   Sources & status — live freshness of every feed
/about                  What this is / isn't, disclaimers, contact, citation
```

Global elements on every page:

- **Status bar** (top): worst current official status for the selected port or "Select a port", plus a data-freshness dot (green = all current, amber = some stale, grey = failed) linking to `/data`.
- **Port picker**: search + "nearest to me" (geolocation is opt-in; nothing leaves the device).
- **Footer**: standing disclaimer (short form), "Official sources" links.

---

## 3. The four experiences

Release tags: **MVP** = first public release (v1.0) · **P2** = next phase · **P3** = later · **NO** = not planned without new data/partners.

### 3.1 Live Ocean Map

| Feature | Release | Notes |
|---|---|---|
| Basemap tuned for coast reading (muted land, bathymetry hint, port labels) | MVP | Light + dark styles |
| Ports layer (official port list) | MVP | Tap → port drawer |
| **Official status** overlay: MPAs | MVP | Polygons, name, designation, regulation link |
| **Official status** overlay: HAB closures/advisories where geometry is published or derivable (e.g., latitude-defined crab zones) | MVP if source verified, else P2 | Never inferred geometry; curated with `last_verified` |
| **Official forecast**: C-HARM probabilities (bloom, particulate DA, cellular DA) | MVP | Lead selector (nowcast / +1 / +2 / +3 days as published) |
| **Observation**: VIIRS chlorophyll, absolute log scale, one product at a time | MVP | Cloud gaps hatched; % valid in tooltip |
| PACE chlorophyll as an alternate (not blended) | MVP | Same legend treatment, its own scale |
| Point inspector: tap water → value(s), product, valid time, resolution | MVP | Reads precomputed grids, not tiles' pixels |
| Layer legend with product-class badge (Official / Observation / Experimental) | MVP | Badge colors reserved per class |
| Timeline scrubber (last 14 days of observations and forecasts) | P2 | Requires artifact retention |
| Chlorophyll anomaly vs fixed climatology | P2 | Needs a VIIRS climatology build |
| HABMAP shore stations (cell counts, DA) | P2 | Points with sparklines |
| NWS marine zones + active hazards | MVP (in port drawer) / P2 (map layer) | |
| Split-screen compare | P3 | |
| Experimental research forecast layer | P3 | Gated by `05` R12–R16 |
| Species opportunity / "where to fish" layer | NO | No defensible public data |

### 3.2 Bloom Intelligence (`/bloom`)

| Feature | Release |
|---|---|
| Plain-language explainer: blooms vs toxins vs closures; what C-HARM predicts | MVP (static page) |
| Statewide coastal strip chart: C-HARM probability by latitude over the last 30–90 days ("Hovmöller" strip along the coast) | P2 |
| Per-region trend (rising / steady / falling) from C-HARM, with method shown | P2 |
| HABMAP observations vs C-HARM forecasts at stations | P2 |
| Past events archive (2015–16, 2023, …) with official closure timelines | P2 |
| Research model hindcast showcase (2003–2021) with model card | P3 |

### 3.3 Fisheries & Economic Exposure (`/fisheries`)

| Feature | Release |
|---|---|
| Port card: landings value by species group, share from HAB-sensitive species (multi-year average, source, years, inflation basis, suppression flags) | MVP (on port page) |
| Statewide port comparison (sortable, map + bar chart) | P2 |
| Species sensitivity reference (which species have had HAB closures/advisories in CA, with citations) | MVP (static table) |
| Closure history timeline per fishery/zone (official records) | P2 |
| Modeled expected revenue at risk | P3, only with a peer-reviewable method, uncertainty, and separate presentation (`05` R17) |
| Individual vessel economics, price forecasts | NO |

### 3.4 My Coast (`/ports/[portId]`)

Card order follows the hierarchy and never changes:

1. **Official status** — every active closure/advisory/quarantine relevant to the port's area, with effective dates and source; or "No active notices found in [sources], checked [time]" — and if any regulatory source is stale/failed, "Status not verified" (R2).
2. **Official bloom forecast** — C-HARM summary within the port's nearshore area (defined polygon, documented): e.g. "Share of nearshore area with ≥ 50% probability of *Pseudo-nitzschia* bloom: 12% (nowcast), 7-day trend ↑". Threshold choices documented.
3. **What satellites see** — VIIRS chlorophyll median and % valid pixels in the same area, last valid date; P2 adds anomaly.
4. **Marine weather** — NWS marine zone forecast headline + any active marine hazards.
5. **Historical exposure** — economic card (3.3).
6. **Sources & times** — compact provenance table for every number on the page.

**Port model.** Ports come from CDFW's port and port-area reference tables (MFDE Port Reference Table), with coordinates reviewed by hand. Each port carries its CDFW port area (economics: 9 marine port areas; note Crescent City is inside "Eureka"), PacFIN port group (10 CA groups, Crescent City separate), RAMP zone, county (CDPH advisory unit), and NWS marine zone. Economic figures on a port page are explicitly labeled as **port-area** figures.

| Feature | Release |
|---|---|
| Port page with cards 1–6 | MVP |
| Favorite port remembered on device (localStorage) | MVP |
| Share link / printable one-page brief | MVP (print CSS) |
| Email alerts on official status change for a port | P2 (no accounts: double-opt-in email list) |
| SMS alerts, push notifications | P3 |
| Accounts, saved zones, trip logbook | P3 (privacy design first) |
| Spanish (and other community languages) | P2 — copy is externalized from MVP |

---

## 4. MVP definition (v1.0)

**In:** Live Ocean Map (ports, MPAs, C-HARM, VIIRS/PACE chlorophyll, inspector, legends, badges) · Port pages for all official ports · Historical exposure card · Official status card with curated/verified records · `/data` freshness page · `/bloom` explainer · `/about` · light/dark · mobile-first · print brief.

**Out (explicitly):** accounts, alerts, logbook, experimental ML layer, anomaly layer, timeline scrubber, modeled losses, recommendations of any kind.

**Success criteria for v1.0**

- Every number on screen is traceable to a source record with timestamps (automated test).
- Page usable on a mid-range phone over 3G: map interactive < 5 s, port page < 2.5 s LCP.
- Zero safety-rule violations in the copy/logic review checklist (`06` §5).
- At least 5 fishermen / port staff in a usability session can answer "what's the official status at my port and how current is it?" without help.

---

## 5. Visual design direction

The aesthetic goal is "nautical chart meets scientific instrument": calm, precise, high-contrast, and trustworthy rather than flashy.

**Principles**

1. Hierarchy is visual: official status is always the top, largest, and only element allowed to use the "closure" color family.
2. The map is the hero on desktop; on phones, the port card stack is the hero and the map is one tap away.
3. Every visual encoding has a legend, a unit, and a time.
4. Design for sunlight and night: a light high-contrast theme and a dark theme with equal care.

**Color system (tokens; final values validated with the `dataviz` method)**

| Role | Guidance |
|---|---|
| Official regulatory status | Reserved semantic set: closed / restricted / advisory / no notices found / not verified. Icon + label + color, never color alone. No green "safe" state — "no notices found" uses neutral |
| C-HARM probability | Sequential, perceptually uniform, single-hue-to-warm ramp, 0–100%, fixed (never rescaled per map) |
| Chlorophyll | Log-scaled sequential "algae" ramp (cmocean-style), fixed range per product; cloud gap hatch |
| Experimental model | Distinct accent + dashed legend frame + "Experimental" badge |
| Product-class badges | Official (solid), Observation (outline), Experimental (dashed) |

**Typography:** keep Geist Sans for UI and Geist Mono for data values; tabular figures for all numbers; minimum 16 px body on mobile.

**Components:** status chips, product-class badges, legend (continuous + categorical), source pill with time-ago, freshness dot, coastal strip chart, sparkline, bar chart for species mix, bottom sheet (mobile) / side panel (desktop), skeleton loaders, empty/stale/failed states designed explicitly.

**Accessibility:** WCAG 2.2 AA contrast, keyboard map controls, screen-reader summaries for every map layer ("C-HARM nowcast, issued …, highest probability near …"), reduced-motion support.
