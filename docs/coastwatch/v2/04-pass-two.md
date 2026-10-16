# Coastal intelligence v2 · pass two: measured acceptance

Scope of the pass (owner's brief after the pass-one audit): one evidence backbone, the real My Coast redesign on
the phone, corrected harbour data, the harbour-scale map elements from the approved prototype, and before/after
measurements. Measured on the 15 October 2026 production data with Playwright (installed Chrome), 390 × 844 for
the phone and 1440 × 900 for the desktop; the scripts are `scratchpad/measure2.mjs`, `sections.mjs` and
`small.mjs` of the session (reproducible from `docs/coastwatch/v2/review/`).

## 1. One evidence backbone

- `coastwatch-web/src/lib/harborEvidence.ts` now produces **every** briefing claim: the season status, a disputed
  notice, the notices here, the HAB line (measured, model or gap), the model, the satellite and the sea
  conditions. Each claim carries `headline`, `detail`, `quote`, `source`, `focus`, `text` (the one-sentence
  form for the API and a future assistant) and `supports` (node ids).
- Deleted: `whatMatters`, `habMatter`, `conditionsMatter`, `modelStaleWords` from `lib/trip.ts`;
  `measuredClause`, `modelClause` and the `habSentence` from `MyCoast.tsx`. My Coast's "Right now" lines, its
  answer line (Research mode), the evidence list and the map's thread all read the same graph; the map's cell
  thread now takes its cell from the same neighbourhood the claims use (`outDay.pda`), never a second search.
- `GET /api/evidence` returns the same graph, now with the agency statements, the live conditions (through
  the server's own `/api/conditions`) and the trip day; it also loads the bundled water mask so its satellite
  reading sets shore-adjacent pixels aside exactly as the page does.
- **Consistency tests.** `tests/e2e/evidence.spec.ts` renders My Coast for Monterey, Santa Cruz, Crescent City
  and Shelter Cove and asserts, against `/api/evidence` on the same server and clock: every "Right now" line's
  id, kind, headline and supporting node ids equal the API's claim; every evidence row's value text equals the
  API's node; the map thread's distance equals the API's C-HARM node distance; every supporting id exists.
  The spec found one real contradiction while being written (the API read the satellite without the water
  mask: 0.94 vs 0.88 mg/m³ at Monterey) and now guards it. Unit: `harborEvidence.test.ts` (15 tests, the
  briefing's claim tests moved there from `trip.test.ts`).

## 2. My Coast on the phone, before → after (Coastal mode, five harbours)

| Measure | Monterey | Crescent City | San Francisco | San Diego | Shelter Cove |
|---|---|---|---|---|---|
| Page length, phone screens | 7.4 → **4.9** | 6.8 → **4.9** | 6.7 → **4.9** | 6.7 → **4.9** | 6.7 → **4.9** |
| Words on the page | 1,300 → **810** | 1,228 → **852** | 1,167 → **823** | 1,181 → **844** | 1,186 → **833** |
| Repeated sentences (≥ 5 words) | 3 → **0** | 2 → 1 | 1 → 0 | 2 → 0 | 2 → 1 |
| "not verified" said | 2 → 2 | 2 → 2 | 2 → 2 | 2 → 2 | 2 → 2 |
| First conclusion on the first screen | yes (y 340) → **yes (y 340)** | yes → yes | yes → yes | yes → yes | yes → yes |
| Taps to the first conclusion | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| Controls under 44 px (of all) | 31 of 65 → **8 of 76** | 29 of 60 → 8 of 73 | 28 of 60 → 8 of 73 | 29 of 60 → 8 of 73 | 29 of 60 → 8 of 73 |
| Median body text size | 13 px → **14 px** (claims 17 px, details 15 px) | same | same | same | same |
| Fixed buttons over content | "Map" pill → **none** (docked bar) | same | same | same | same |

The eight remaining small controls are all in the shared masthead and status line (skip link, brand, the
notices pill at 32 px, the menu at 36 px, the CDFW/CDPH links and "More" in the banner, the footer "About"
link's width), which the release tests pin at narrow widths; they are a separate, deliberate change.

Section heights on the Monterey phone page after: trip bar 161 px · Right now 1,148 px (4 claims, 269 words) ·
since-last-check 98 px · evidence (gaps, the readings behind one disclosure, the timeline) 753 px · checklist
880 px (5 actions, reasons behind "Why") · detail disclosure 54 px · footer 232 px. Total 4,012 px.

Load timing is not comparable between the two runs (the after run shared the machine with the full Playwright
suite); the server render (DOMContentLoaded) is unchanged in kind, 0.2–0.6 s locally.

What the page now is, top to bottom: the trip bar (place, activity, species, day) · **Right now at <place>**:
the graph's claims, each with its source and "Show on map" · since your last visit (returning visitors) ·
**The evidence**: the gaps said first, every reading behind one disclosure ("Every reading (n)"), the 90-day
timeline · **Before you leave**: five actions, each a source button, the reason behind "Why" · **More detail and
every source** (one disclosure: HAB detail, notices, conditions, science and sources) · home port, mode,
purpose, sources. On the phone the map is a docked bar above the tab bar that opens a full-height sheet; on
the desktop it is the sticky right column, unchanged.

## 3. Harbour data corrected (CO-OPS)

`docs/coastwatch/v2/03-coops-audit.md` has the detail. In short: the `f` flags are X (max exceeded), N (min
exceeded), R (rate of change) and any non-zero flag rejects a reading; sensor elevations relative to MLLW are
recorded per station (Monterey −6.8 ft, Crescent City −3.0 ft, San Diego −18.2 ft), no depth below the
surface is stated by NOAA; Los Angeles 9410660 has no water-temperature sensor at all and San Francisco,
Redwood City and Martinez are flagged "suspect" by NOAA since late 2025. The rule is now: the single nearest
station, only within 6 km (the harbour or bay), "in the harbour" within 2 km, no fallback to a second station.
Per port: in-harbour Avila, Monterey, Crescent City, Point Arena; nearby-station Eureka (5.7 km), San Diego
(4.0 km); "sensor not reporting" San Francisco and San Pedro; none within 6 km for the other 14. Readings
older than 1 h are marked stale; the route distinguishes "no data", "no readings in 3 h", "every reading
failed NOAA's checks" and an HTTP failure. 51 unit tests (`conditions.test.ts`).

## 4. Harbour-scale map

In the Ocean Map (`LiveOceanMap`, `MapCanvas`): from z 10.5 a labelled note at a no-value centre ("No C-HARM
value here: nearshore cells not modelled · nearest cell N km") with a dashed thread to the nearest valued
cell and its value; from z 9.5 the HAB shore stations, filled when sampled in the last 30 days, hollow
otherwise, labelled "<name> · last sample <date>"; latitude limit lines clipped to the sea with the bundled
water geometry; day chips that never truncate ("2 d ago"); product and native resolution in every hover and
inspector readout. Tests: `m5.spec.ts` (the harbour view), `m2.spec.ts` (limits over the sea only),
`map.spec.ts` (no clipped chip at 1440 and 390 px), units for `gapAt`, `clipLineToWater`, `stationFeatures`.

## 5. What is production-ready, what is experimental

- Production-ready in this branch: the evidence backbone and its consistency tests; the phone layout and
  targets listed above; the CO-OPS rule and flags; the harbour-zoom map elements; the Neritic name across the
  README, site copy, pipeline strings and workflows (NOAA CoastWatch credits and technical identifiers kept).
- Experimental or pending: the eight masthead controls under 44 px; the light map; the evidence list's
  per-row map focus for buoys (the sea row focuses the conditions as one); anything predictive.
