# Redesign readiness (2026-10-13, production data)

## What was checked

- **Both modes, every main page:** My Coast, Ocean Map, Bloom and Fisheries in Coastal and in Research mode, at 1440 × 900 and 390 × 844.
  - No page error and no sideways scroll.
  - The earlier full set at 1440, 1280, 1024, 390 and 320 px is in [32 §12](../32-coastal-redesign.md#12-screenshots-production-data-after).
- **Ocean conditions:**
  - live at all 22 ports and 4 harvest sites (32 §6);
  - an outage now reads in plain words ("NOAA's forecast service did not answer just now"), with the technical reason kept in the tile's title, and a test covers it.
- **Mode changes never re-mount content:** a regression test holds the Bloom chart and the Fisheries history while the device's mode is read. This was the cause of the CI failure on `c638f27`.
- **Official sources:**
  - an agency source link is on the first screen of My Coast in every measured run (desktop and phone);
  - the notices drawer is one tap away in both modes;
  - a disputed notice (agency reports it lifted) is shown as such everywhere.
- **Tests:** 228 unit and 226 browser tests pass, including axe on the Coastal briefing on desktop and phone.

## Matched screenshots (same pages and sizes; before = `main` at `ce68a541`, after = this branch)

| View | Before | After (Coastal) | After (Research) |
|---|---|---|---|
| Ocean Map, desktop | [before](img/before-d-map.jpg) | [after](img/matched-d-map.jpg) | [research](img/matched-d-map-research.jpg) |
| Ocean Map, first visit | [before](img/before-d-map-first-visit.jpg) | [after](img/matched-d-map-first-visit.jpg) | — |
| My Coast, Monterey, desktop | [before](img/before-d-coast-monterey.jpg) | [after](img/matched-d-coast-monterey.jpg) | [research](img/matched-d-coast-monterey-research.jpg) |
| Bloom, desktop | [before](img/before-d-bloom.jpg) | [after](img/matched-d-bloom.jpg) | [research](img/matched-d-bloom-research.jpg) |
| Fisheries, desktop | [before](img/before-d-fisheries.jpg) | [after](img/matched-d-fisheries.jpg) | [research](img/matched-d-fisheries-research.jpg) |
| Notices drawer, desktop | [before](img/before-d-drawer.jpg) | [after](img/matched-d-drawer.jpg) | same |
| Ocean Map, phone | [before](img/before-m-map.jpg) | [after](img/matched-m-map.jpg) | — |
| My Coast, Monterey, phone | [before](img/before-m-coast-monterey.jpg) | [after](img/matched-m-coast-monterey.jpg) | [research](img/matched-m-coast-monterey-research.jpg) |
| Notices drawer, phone | [before](img/before-m-drawer.jpg) | [after](img/matched-m-drawer.jpg) | same |
| NOAA conditions outage, phone | — | [first screen](img/matched-m-coast-conditions-outage.jpg) · [full page](img/matched-m-coast-conditions-outage-full.jpg) | — |

## Remaining weaknesses (ranked)

1. **The no-data hatch flashes over the whole sea while the forecast raster loads**, and shows through a translucent raster. Found by the Signature Experience session (PR #43), which proposes the fix: draw valued cells opaque and keep the hatch hidden until the field exists. Not yet fixed here.
2. **The plan bar adds controls.** My Coast's first screen has 38 controls against 33 before. Activity and species could become one control ("What are you after?"), with the forecast day moved into the conditions card.
3. **On a phone, the header block pushes the briefing down.** The place name, search and home-port link sit above the official headline: the headline is on the first screen, but the next steps are not.
4. **Caution still appears 4 times on My Coast:** the masthead pill, the status line, the official card and one next step. The Signature session proposes folding the cream status line into the masthead (a decision for you).
5. **Uppercase eyebrows are heavy.** The Signature session proposes sentence-case kind labels (a decision for you).
6. **Technical labels in Bloom's station rail** ("pDA last measured…") remain in Coastal mode.
7. **Fisheries' 11 species chips wrap to two lines** on desktop. A select or a two-row grouping would be calmer.
8. **The Coastal map's default forecast treatment** (none, veil or lens) is still open. The Signature session proposes options (a decision for you).
9. **The live conditions route has no monitoring in the preview.** NOAA outages are shown honestly, but nobody is alerted. If it ships, add it to the health checks.
10. **Research mode keeps the plan bar and the official card above the science.** That is intended (official information leads in both modes), but researchers may want measurements on the first screen.

## Recommended sequence for the final combined experience

1. **Merge #42 (CI)**, so every branch gets one run per push. Then open the monitoring PR, which is held back to avoid a duplicate run on the old workflow.
2. **Merge the monitoring PR**, so silent outages alert before the end of Suomi-NPP data on 2026-11-02.
3. **Suomi-NPP migration:** implement the NOAA-21/NOAA-20 switch on a branch, using the overlap validation, and test it on the staging channel. Merge it by about 2026-10-26.
4. **Decide the three Signature proposals** (status line in the masthead, sentence-case labels, Coastal forecast treatment). Then adopt the Signature tokens and components into #41, fixing weakness 1 first.
5. **Rework the My Coast header and plan bar** (weaknesses 2 and 3) on the adopted components. Then do one more critique pass on production data.
6. **Merge #41**, refresh the private preview, run stakeholder sessions with the interview plan, then decide on public release.
