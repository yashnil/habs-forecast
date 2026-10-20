# Final product-excellence pass (2026-10-19)

**One bounded pass before the owner's review. A scripted, expert pass: no real users or physical phones took part.**

## Method

- **Base:** `feat/neritic-v4` at `f141843`.
- **Audit:** five parallel, read-only audits ran against the baseline build on live published data:
  - A: visual design;
  - B: UX workflows;
  - C: geospatial and scientific integrity;
  - D: mobile, offline and accessibility;
  - E: performance and reliability.
- **Consolidation:** their findings were merged into one prioritised list with a single owner per file.
- **Implementation:** three tracks worked in parallel, each in its own worktree:
  - science wording;
  - the Explore shell and map;
  - My Coast, offline, charts and design tokens.
- **Integration and review:** the lead integrated the tracks, and an independent reviewer then walked the before and after builds side by side.

## What changed

| Area | Before | After |
|---|---|---|
| Disputed notice | Harbour panel and briefing led with "CDPH appears to have lifted …" | "Status unclear: CDPH notice …; release SN26-021 appears to lift it; no person has reviewed that release". The count reads "3 notices cover these waters, plus 1 whose status is unclear" |
| Saved copies | Offline, a 3-day-old pack still said "What matters now", "buoy now", "tonight" | Headings are dated ("What mattered when saved · …"). Our own sentences carry absolute times. NWS's relative period names are quoted and dated. The banner says how old the copy is |
| Offline, other places | Asking for Crescent City offline silently showed Monterey | Says the place is not in a saved pack and links to Monterey Bay |
| Failed layers | Failed satellite or C-HARM tiles hatched the whole sea as "no value", and the status said Ready | The layer says it could not be loaded, nothing is drawn in its place, Try again; the status centre shows Unavailable (e2e) |
| Currents | Opened on the newest hour even when it held 29 of 220,320 cells | Opens on the newest usable hour. Near-empty hours stay selectable and are labelled "incomplete (N cells)" |
| C-HARM after a failed update | "No newer run has been published" | "Neritic's newest copy is the run of …; it could not reach NOAA at …, so a newer run may exist" |
| Satellite medians | Labelled with the newest day only | The real span ("Oct 13–16, most Oct 16") and "nearest clear pixel 849 m away" |
| Desktop Explore with a place open | Layer deck (340 px) and panel (404 px) both permanent | Deck folds to a one-line capsule (layer · variable · day); map tools in two grouped stacks |
| Phone sheet | Credits paragraph, then layer chips, then a full-width briefing button; no evidence at half height; cream surface | Credits are a strip on the map edge; one row of compact chips; "Briefing ›" on the title row; the first evidence line sits at half height; cool v4 surface |
| Search | "Monterey" + Enter opened Moss Landing; "Pillar Point" returned nothing, silently | Ranked matches (exact, prefix, word, alias, county), other harbour names, a next step on no match, home port first |
| Home port | Set, then mostly ignored | Remembers activity and species; the My Coast tab and Explore's start follow it |
| My Coast | 7,966 px on a phone; one notice repeated 4 times; three stacked pills; giant chevrons in the picker | Evidence below "Before you go" in counted, folded sections (nothing removed); "Monterey ▾" place menu; sized icons |
| Research | "All" x-axis: about 70 overlapping month labels | Ticks chosen from the pixel width (years beyond 3 years); the grid map mounts when it nears the screen |
| Field pack dialog | Focus leaked behind it; Escape closed the panel behind; a backdrop tap hid a running download | One accessible dialog on Explore and My Coast: focus trapped, Escape closes only it, a running download survives closing and reopening; removal asks to confirm |
| Status centre on phones | Ran off the left edge | Fixed under the top bar |
| Performance | Intl date formatters built on every call (about 70% of hover script time at 4× CPU) | Formatters cached; the hover readout re-renders alone; offline basemap range reads use `Blob.slice`; the service worker's code cache is named per build and pruned |

## What did not change

- No pipeline, schema or manifest change.
- No new dependency.
- No logo change: C2 is not installed.
- No capability removed. Rarely used controls moved behind disclosure. The removed-capability list in `feature-parity.md` was rechecked (home port, waters from the map, currents/chlorophyll compare, coverage, multi-sensor agreement, per-place notices, harbour summaries, checklist ticks, field packs, research sampling and exports, provenance/QC/timestamps).

## Verification (exact head)

| Check | Result |
|---|---|
| Linux CI container: gen-types, lint, typecheck, unit, e2e | see the PR (exact counts) |
| Pipeline (pytest, schema, fixtures) | 246 passed; schema and fixtures unchanged |
| Strict offline proof (`npm run test:offline`) | PROVEN: 0 requests leave the device |
| Live evidence graphs (8 harbours × 4 profiles, `checkGraph` + banned-phrase scan) | 32 graphs, 0 violations |

## Known limitations

- **Statewide framing:** at exactly 1440×900 the map's longitude bounds (shared with the bundled water fallback) stop it zooming out far enough. The statewide start shows the north coast, and San Diego sits about 35 px below the edge. Widening the bounds needs a regenerated water-fallback asset.
- **Not done this pass:**
  - Choosing waters still means a round trip from Explore back to My Coast; a full "pick mode" was left out.
  - The no-data hatch is calmer, but it is still drawn over ocean outside a product's coverage footprint.
  - `viewport-fit=cover`, which needs a real device.
- **Untested here:** real phones and real people (see `real-device-checklist.md`).
