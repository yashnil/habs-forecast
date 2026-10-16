# Neritic v4: the product transformation

Branch `feat/neritic-v4`, from the v2 release on `main` (`4d2e3e1c`). It does not include Harbor
Intelligence v3 (draft PR #66), which will be merged separately once it is approved.

## What changed

| | v2 | v4 |
|---|---|---|
| Map | A page beside a report column; masthead, dock and drawer stacked over the map | **Explore**: the map fills the screen; one layer selector with its legend and provenance; a timebar only for layers with time; an inspection surface for the one selected thing |
| Selection | Port inspector or point inspector, each with its own wording | One selection model (harbour, station, model cell, point, area), in the URL (`sel=`), with Back to the previous view; the map, panel, charts and briefing all follow it |
| Briefing | A long report with a permanent map column | **My Coast**: answer first (what matters now, what changed, evidence near you, farther or older, official restrictions not verified, what is not known, before you go). A small locator map; every statement has "Show on map", which opens Explore at the exact node and offers a way back |
| Research | A mode that made My Coast more verbose | **Research**: a separate workspace with station records on fixed axes, ranges the station really covers, the C-HARM history beside the samples (never compared numerically), provenance and QC, a CSV export with a provenance header, the native C-HARM grid and transects |
| Loading | Per-component spinners and words | **Readiness**: every part reports its own state (loading, ready, partly available, older than usual, saved copy, unavailable, downloading, incomplete); a status centre; a start card instead of a blank map; failures are per part, with Try again |
| Offline | None | **Field pack** for Monterey Bay: a self-hosted OpenStreetMap extract (PMTiles, ODbL), the relief, the C-HARM run, the latest satellite view, currents, and the briefing and evidence snapshot; a service worker serves it when the network fails. Proven with the network gone after a browser restart (`npm run test:offline`) |
| Design | Navy masthead, cream paper, Plex Sans | Two environments: dark instrument (map) and light chart (reading); Atkinson Hyperlegible Next, Source Serif 4, IBM Plex Mono for values; 56 px field controls on phones; WCAG AA contrast on every text token |
| Brand | Installed v2 mark | A **provisional** mark in `src/v4/brand/Brand.tsx` (one file to swap); three bathymetric concepts in `brand/` await the owner's choice |

## What did not change

- The data contracts: no pipeline, schema or manifest change. `lib/` (evidence graph, grid
  sampling, representativeness, freshness, official-notice associations) is reused as it was.
- The scientific wording: provenance lines (`lib/stamp`), the evidence graph's claims
  (`lib/harborEvidence`), the hover readout and the official-notice texts are the tested v2 code.
- v2 links: `region`, `port`, `var`, `lead`, `layer`, `inspect`, `place`, `for`, `species`, `date`
  open the same view; `mode=research` on My Coast opens the Research workspace.

## Architecture

- `src/v4/state/selection.ts`: the selection union and its URL encoding.
- `src/v4/readiness/`: the readiness store (no dependencies; `useSyncExternalStore`) and the status centre.
- `src/v4/explore/`: the workspace, the layer model (`layers.ts`), panels, timebar, search, inspection views, data hooks (cancelled on change).
- `src/v4/coast/`: `useBriefing` (v2 My Coast's assembly, moved, logic unchanged) and the briefing view.
- `src/v4/research/`: the workspace and the transect sampler.
- `src/v4/offline/` and `public/sw.js`: field packs. `public/offline/monterey/` holds the extract and its `pack.json` (source run, OSM date, checksum, licence, command).
- `components/map/MapCanvas.tsx`: extended with opt-in props (`chrome`, `highlight`, `onStationPick`, `clusterStations`, `onStatus`, `basemapPmtiles`); v2 behaviour unchanged when they are unset.

## Decisions and why

- **Promote cells' honesty over smoothness.** The forecast stays nearest-sampled 3 km cells. A
  strong zoom-dependent fade was tried and reverted: the no-value hatch showed through valued
  cells and read as "no value". The harbour view uses cell outlines and printed values instead.
- **No particle flow.** Currents are arrows, one per observed 2 km cell.
- **Statements settle together.** The briefing shows its statements once the model and the
  harbour conditions have answered (or after 6 s), so the answer does not reshuffle.
- **Offline basemap.** A browser cannot send the identifying User-Agent OpenFreeMap requires for
  offline downloads, and its tile URLs change weekly; so the pack carries a regional extract cut
  from OpenFreeMap's planet file with `pmtiles extract` (7.0 MB, OSM data of 2026-10-05).
- **Slashed zero.** Atkinson Hyperlegible Next always slashes its zero (no alternate); kept as a
  legibility feature (0 versus O in glare).

## Known limitations

- One field pack (Monterey Bay). More regions need their own extracts (about 5–10 MB each).
- Area statistics are computed for C-HARM cells only (not satellite or currents).
- The pack's extract is committed to the repository (7 MB); a larger set belongs on object storage.
- Browsers may evict stored packs; the panel says so and asks the browser to persist.
- Final identity, the light/dark question for the phone briefing, and Research's depth are owner decisions.
