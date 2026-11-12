# Neritic v12: the product, from the user's side

Evidence: `../research/A-fishing-workflow.md` (no interviews; desk research), `C-ia-usability.md`. Each journey
is built as a working flow and proven by a Playwright journey test that records video and screenshots
(`tests/e2e/journeys/*.spec.ts`).

## Architecture

| URL | Destination | Its one job | First thing on screen |
|---|---|---|---|
| `/` | Home → Today | First visit: the opening, then the picker. Returning: Today for the saved port, no redirect flash | The statement and Find your port / Today at the saved port |
| `/today` | **Today** | What changed at my port, what needs attention, the next date | Status and the next date on the chart of your port |
| `/explore` | **Explore** | See the water: observations and models in place and time | The chart, full sheet, with the active layer's legend-as-title |
| `/coast` | **My Coast** | Plan a trip: gates, the sea window, trip economics | Gates for your fishery, then the window, then the worksheet |
| `/research` | **Research** | Check the record: series, QC, provenance, export | A station's record as small multiples with one brush |

The port (`nx-port-v1`) and the chart's camera travel across all four. Old URLs redirect (`/?view=map` → `/explore`, `/bloom` → `/research`, PWA start_url → `/`). Saved offline packs migrate with the URLs.

## Journey 1. A fisherman evaluating Neritic for the first time
- **Goal:** decide in under a minute whether this is worth bookmarking.
- **Minimum information:** what it does, for my port, with something true and current.
- **Flow:** `/` → opening (≤ 5 s; statement legible at frame one; skip at any moment) → it settles into the chart with **Find your port** → PortFinder (type "moss" or pick north-to-south) → choose fisheries (default: none; suggested from the port's landings, labelled) → the chart pans to the port and Today assembles: status, next date, two soundings, margin notes.
- **Principal action:** pick your port and fishery.
- **Outcome Neritic reliably delivers:** season status with source and verification, the next decision date, the latest crab test with its age, sea conditions with their source.
- **Acceptance:** a stranger can name the product's purpose from the first frame; no statewide numbers before a port is chosen; three taps from `/` to Today; works offline after the first visit.

## Journey 2. A returning fisherman checking the home port
- **Goal:** know in ten seconds whether anything changed and what's next.
- **Flow:** `/` → Today at Moss Landing immediately (no opening, no redirect flash) → "Since Nov 8: no new season notice; NWS now forecasts 11–15 ft seas Saturday" → next date → tap a sounding or row to open its provenance and source → Alerts to subscribe → My Coast to plan.
- **Principal action:** read what changed and open its source.
- **Acceptance:** the change line is computed from the last visit (`nx-port-v1.lastSeen`), never invented; if nothing changed it says so in one line; every value opens its provenance.

## Journey 3. A researcher investigating an observation
- **Goal:** check a chlorophyll patch or a toxin record and export the evidence.
- **Flow:** Explore → tap the water → Readout (value, kind, date, pixel age, transect) → "See the record nearby" → Research with the nearest HABMAP station preselected and the time window linked → brush → QC marks and provenance → Export CSV of the window.
- **Principal action:** pick a place and time, read the record, export.
- **Acceptance:** the Explore selection survives into Research via URL state; the brush filters every strip; export contains exactly the visible window with source columns.

## Planning a trip (My Coast)
- **Flow:** gates (season, whale measures, tests) with dates → sea window (NWS periods, buoy, tide; the window is the reader's choice, never a recommendation) → worksheet: the reader types catch, price, fuel, crew share; nothing pre-filled; sourced reference values enter only on tap → live waterfall gross → net, break-evens, a ±20/30 % sensitivity table that animates the waterfall on hover → save the trip locally.
- **Acceptance:** every output says "hypothetical, from your numbers"; blank inputs read "waiting for …", never 0; who pays fuel is an explicit choice.

## Help without a tutorial
- First visit: one optional "Show me around" in the margin that highlights, in order, the status, a sounding, the time rail and the provenance stamp, each with one sentence; dismiss forever with one tap.
- Every page's margin starts with one sentence saying what this view is.
- Empty, stale, offline and sources-differ states are written for the reader: what's missing, why, and what to do.
- Uploads: none in this pass. Research found no evidence fishermen want document upload, landing records or buyer prices in this tool; revisit after interviews.
