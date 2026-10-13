# 04 — Interaction specifications

Each interaction lists its purpose, its behaviour, motion, accessibility, the scientific guardrail it must keep, its prototype, and its integration status. Source paths are relative to `prototypes/src/`.

## 1. Provenance threads ★ signature

- **Purpose:** show *where each statement about a point comes from*. When the reader selects a water point, thin lines draw from it to:
  - the sampled station;
  - the model cell that holds the value, or the nearest valued cell if the point's own cell has none;
  - the satellite pixel, or the nearest clear pixel;
  - the nearest official latitude limit, drawn straight north or south as a ruler.
- **Behaviour:**
  - Threads draw out once: 520 ms each, 90 ms stagger, ease-out.
  - Each thread ends in a dot in its source's mark colour.
  - Hovering or focusing a ledger row brightens its thread and dims the others through feature-state, with no React re-render.
  - A station more than 25 km away still gets a thread, at half opacity, and its row says "too far to describe these waters".
- **Motion:** `line-gradient` progress on a `lineMetrics` GeoJSON source. Under reduced motion the threads are drawn instantly.
- **Accessibility:** threads are decorative. Every fact they show is in the ledger text: distance, bearing, date.
- **Guardrail:**
  - A thread is a connector, not a value: it is never coloured by data and never animated in a loop.
  - It always ends at the source's real location.
  - The point never inherits the source's value. The text says "it describes that cell, not this one".
- **Prototype:** `map/threads.ts`, `pages/Explorer.tsx` (threads effect), `ui/Ledger.tsx` (`onThread`).
- **Status:** integrate now. It needs a `lineMetrics` GeoJSON source (`line-gradient` is long supported in MapLibre) and the redesign's `associationsAt` for notice tiers.

## 2. Coastal journey

- **Purpose:** keep geographic context when changing place.
- **Behaviour:**
  - Under 40 km: `easeTo`.
  - 40–450 km: the camera follows a Catmull-Rom curve through offshore waypoints (each port pushed 10 km west), lifting out by distance and settling in.
  - Over 450 km: `flyTo` (curve 1.5, maxDuration 3.2 s).
  - Duration is 900 + 4.2 ms/km, capped at 2.6 s.
  - Any pointer, wheel, touch or key input stops the journey where it is.
  - A status line ("Moving along the coast to …, touch the map to stop") is shown and announced.
- **Reduced motion:** `jumpTo`.
- **Measured:** 60 fps median and p95 with GPU (Santa Cruz → Morro Bay; `evaluation/perf-gpu.json`).
- **Prototype:** `map/journey.ts`.
- **Status:** integrate now (MapPlace / PlacePicker).

## 3. Time-aware exploration

- **Purpose:** never confuse *the day a forecast describes* with *when it was made*, or either with an observation's date.
- **Behaviour:**
  - The **For** row has one radio per lead day: "nowcast", "yesterday", "today", "tomorrow".
  - The **Issued** row is a single bracket under all four days ("One NOAA C-HARM run, Sun, Oct 11 · Not confirmed"). Changing the day never changes this row.
  - Satellite mode replaces both rows with its window and says that each pixel keeps its own date.
- **Data swap:** **dip through the basemap**. The old field fades out over 100 ms, then the new field fades in over 140 ms (delayed 100 ms). No pixel ever shows a blend of two days. A cross-fade would invent intermediate colours.
- **Accessibility:** radiogroup with ←/→ keys.
- **Prototype:** `ui/TimeBar.tsx`, `map/MapStage.tsx` (field effect).
- **Status:** integrate now (LeadStrip / LayerDock).

## 4. Interactive uncertainty: "why is this blank?"

- **Purpose:** make gaps spatial and specific.
- **Behaviour:**
  - The selected cell is outlined with its exact edges. A hatched cell means no value.
  - The model row explains why: nearshore mask, outside the model area, or source unavailable.
  - A thread goes to the nearest valued cell with its value and distance, labelled "it describes that cell, not this one".
  - For the satellite, the 300 m pixel is outlined and the thread goes to the nearest clear pixel with *its* date.
  - The "Not known" row always lists toxin in seafood: "only CDPH testing tells you that, and CoastWatch does not have those results".
- **States covered:**
  - outside the domain;
  - nearshore mask;
  - cloud, fog or no overpass;
  - no sampling within 25 km;
  - last measurement too old (toxin last measured 2022 at Monterey Wharf);
  - source unavailable.
- **Prototype:** `lib/provenance.ts`, `ui/Ledger.tsx` (`GapExplainer`).
- **Status:** integrate now. The gap reasons match the redesign's copy (`content/copy.ts`).

## 5. Forecast compositing (veil, lens, class lines)

See [03 — Design language](03-design-language.md#data).
- The canvas is rendered in Web Mercator at 4 px per cell, so the edges are exact. It takes about 40 ms per render and about 7 ms to encode (`evaluation/`).
- Isobaths, coastline and labels are drawn above every field.
- **Status:**
  - **Veil:** integrate now. It is a pre-mix in the pipeline or client renderer.
  - **Lens:** integrate with care. It needs client rendering; ship it only with the dimmed outside kept, never hidden.
  - **Class lines:** integrate now for Research.

## 6. The four-marks ledger (progressive disclosure)

- **Level 1:** one sentence under the place name, each clause led by its mark. For example: "— 3 official notices cover these waters. ● Nearest shore sample 12 days ago, 18 km away. ▪ Forecast 60–70 % chance."
- **Level 2:** one row per kind of knowing, in a fixed order: official, measured, satellite, modelled, not known. Each row has a kind label, a one-line answer and a date/source line. Rows expand in place (`grid-template-rows: 0fr → 1fr`).
- **Level 3:** Research mode opens every row and adds:
  - exact values with quantization error;
  - cell centre and edges;
  - layer ids;
  - the inferred issue-date method;
  - the station id and source link;
  - per-lead exact values for the same cell.

  To avoid layout shift, it waits up to 800 ms for its sources behind a same-size skeleton.
- **Critical information never hides:**
  - The masthead status line and official links are always visible.
  - Disputed notices sit in the official row ("1 reported lifted, awaiting review"), never shown as in effect.
- **Status:** the redesign's Briefing is in flux. Adopt the *row anatomy and order* when it settles; do not rewrite it now.

## 7. Purpose onboarding where the map answers

- **Behaviour:** a small card (not a modal). Picking a purpose frames the map and states what changed in one line, which auto-dismisses after 9 s or with OK:
  - **Commercial:** offshore point, forecast lens.
  - **Shellfish:** the shoreline at zoom 11, basemap and notice limits; the forecast is off because it does not cover most nearshore water.
  - **Research:** statewide, class lines, Research mode.
  - **Exploring:** a three-stop coastal-journey tour that explains the marks with real places.
- **Guardrail:** "It sets what the map shows first, never what is true."
- **Status:** depends on the redesign. Its purpose welcome exists (#39); adopt the "map answers" framing and the tour stops.

## 8. Fishing workflow progression

Select waters, then activity/species, then what applies, then check official sources, then plan.

- The species select narrows the official row and the map's emphasis together: applicable notice geometry brightens and the rest dims via feature-state.
- The official actions (CDPH, CDFW, phone) are a button row inside the official row and in the empty state.
- **Status:** depends on the redesign's PlanBar/NextSteps (in flux). The *coupling* (one selection updates panel and map emphasis together) is the transferable idea.

## 9. Mobile bottom sheet

- **Snap points:** peek (answer plus one row), half (55 %) and full.
- **Non-modal:** the map stays interactive above the sheet.
- **Rendering:** transform-only, driven by Pointer Events with capture. Snapping is velocity-aware (projected 180 ms ahead), with a rubber band at the ends and the `linear()` settle easing.
- **Handle:** a real button with a state label. Tap cycles; ↑/↓ and Esc keys work.
- **Layout:** it sits above the 5-item tab bar. Map controls ride above the sheet at peek and hide otherwise. The camera pads by the sheet height.
- **Measured:** no horizontal overflow at 390 and 320.
- **Library decision:** the research found vaul unmaintained, react-modal-sheet without accessibility, and Base UI Drawer's non-modal map interaction unverified. The roughly 120-line custom sheet is the recommended path until Base UI's `modal={false}` is verified against MapLibre.
- **Prototype:** `ui/Sheet.tsx`.
- **Status:** integrate now (replaces MobileSheet).

## 10. Chart transitions and the season story

- Samples are keyed by time.
- **Window change:** the x-domain tweens for 460 ms and marks are placed from it every frame.
- **Quantity change:** samples move vertically at the same date. Samples with no value fade where they are, never to zero.
- Lines join only samples ≤ 21 days apart; longer gaps are hatched.
- Reported zeros sit in their own floor lane, never at log(0).
- Notices are amber rules at their effective date.
- The model band is on its own 0–100 % axis, labelled "A model probability, not a measurement".
- **Story:** five data-derived steps (year peak, algae peak, agencies act, latest sample, what isn't known). Each step moves the chart and dims everything else. ←/→ keys.
- **Accessibility:** keyboard inspection, a summary sentence, and "Show as table".
- **Prototype:** `charts/*`, `pages/Station.tsx`.
- **Status:** integrate now in Bloom's station analysis. Use d3-scale/d3-shape (about 14 kB) instead of the prototype's local scales.

## 11. Thoughtful feedback

- **Press:** scale 0.97 over 110 ms.
- **Selection marker:** ring settles from 1.9× with the spring.
- **Search:** combobox with recents first, kind marks, live result count.
- **Loading words:** appear only after 400 ms.
- **Errors:** one sentence plus "Try again". Nothing is drawn in place of failed data.
- **Saved copy:** "saved copy on this device" in official ink.
- **Status:** integrate now (tokens and small components).

## 12. Layer comparison (studied, deliberately limited)

**Compatible pairs only:**
- two lead days of the same run and variable (same units, same issue);
- two OLCI days (same sensor and units).

Forecast vs satellite is *not* comparable (probability vs mg/m³), and no swipe between them should exist.

**Recommended form:** the existing time bar is already the comparison for leads. A swipe needs two synchronised map instances, roughly 2× GPU memory, which is not justified for lead days whose difference is better read as the per-lead cell table (Research).

**Status:** exploratory; not prototyped beyond the per-lead table.
