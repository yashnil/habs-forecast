# Readiness of the combined experience (2026-10-13, production data)

The coastal redesign (#41) with the strongest Signature Experience ideas (#43) integrated. What was taken and left: [integration.md](integration.md).

## What was checked

- **Two design passes in real browsers** on production data: My Coast, the Ocean Map (default, forecast on, a chosen point), Bloom, Fisheries and the notices drawer, at 1440 × 900, 390 × 844 and 320 × 640, in Coastal and Research mode. No page error, no sideways scroll.
  - Pass 1 found: the place picker and nine controls before the answer on phones; next steps repeating the conditions card; a "Not verified" status that wrapped on phones; leftover amber tints; a step saying "apply here" beside "cover these waters"; a depth label hiding values; a journey bug (NaN camera). All fixed.
  - Pass 2: answer on the first screen at 320 px; keyboard order skip link → navigation → mode → notices → CDFW/CDPH → page; "Change place" moves focus into the search.
- **Reduced motion:** threads drawn at once, journeys jump, the sheet and loading words do not animate.
- **Performance** with and without a GPU: [integration.md § Performance](integration.md#performance-production-builds-production-data-1440--900-before--41-at-9349aefa).
- **Tests:** 233 unit and 227 browser tests pass, including axe on every main page. New tests cover the hatch-flash fix, provenance threads, the camera options and the masthead menu.

## Remaining weaknesses (candid, ranked)

1. **Software-WebGL start-up still blocks the main thread for about 1.1 s** (single tasks up to ~0.7 s). It is MapLibre itself, not our images; a device without GPU acceleration will feel a pause before the map is interactive. The page's words appear first (LCP ~0.13–0.4 s).
2. **The Ocean Map still has a left icon rail that duplicates the dock's "add a layer" buttons.** Rebuilding the controls into two zones (PR #43's top/bottom layout) was deferred.
3. **Each page has its own map.** Moving between Map, My Coast and Bloom loses the camera; a shared map is an architecture change.
4. **"Not verified" appears three times on My Coast's first screen** (masthead pill, status line, official card). Each is a different control, but together they are loud. Folding the pill's word into the status line on phones would remove one.
5. **The forecast field, when switched on, still covers the bay in near-uniform pink.** That is the data (most days are 60–90 %); the veil was rejected to keep one set of colours. Isobaths now draw over it.
6. **Threads cover the model cell, the station and the limit, not the satellite pixel** (the satellite reader does not yet expose its pixel's position).
7. **Technical labels remain in Research views** ("pDA last measured…", "native 3 km"). Intended for researchers, but Coastal readers reaching Bloom's analysis see them.
8. **The live conditions route is not monitored.** NOAA outages read honestly but alert nobody.
9. **No stakeholder has used it yet,** and no physical phone or screen reader was tested. Every first-screen claim here is a heuristic check.

## Recommended next steps

1. Owner review of the private preview (link in PR #41) and of the accepted and rejected list.
2. Stakeholder sessions with the interview plan (`docs/coastwatch/research`).
3. Then, if the direction holds: the two-zone map controls, the conditions route in the health checks, and a decision on a shared map.
