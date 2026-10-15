# Research preview release, 2026-10-15

The public research preview at the BASES URL now runs main as of 2026-10-15. This release brings:
- environmental data restored after the NOAA West Coast ERDDAP outages;
- consistent chlorophyll quality control;
- stale-data labels where people read them;
- the Next 16 / React 19.3 upgrade that stops the intermittent hydration error.

It remains a **research preview**. No independent scientist or agency has reviewed it, and it is not a regulatory, fishing or seafood-safety decision service.

| | |
|---|---|
| Production URL | https://coastwatch-demo.vercel.app (public, no login) |
| Vercel project | `yashnils-projects/coastwatch-demo` (`prj_QOAYjTDc2W5U81WwwJ288zEijewY`), Node.js 24.x, Git integration disconnected |
| Deployment | **`dpl_HfUScMViHiKPs2aMC9TAsQWrTHjA`** (coastwatch-demo-97q83019i), target production |
| Commit | **`ade20099`** on `main`, tag **`demo-release/2026-10-15`** |
| Released | 2026-10-15 (UTC), with the owner's approval |
| Data | the published dataset, https://yashnil.github.io/habs-forecast/v1. Its first run with the new pipeline was 38682849155 (2026-10-15 04:42Z) |
| Rollback | `demo-release/2026-10-14` = **`dpl_jxGvioj1FRFnwf6AM577bSczWFue`** (coastwatch-demo-cef13qpf8). Before it: `demo-release/2026-10-10` = `dpl_GpFfbFP42QGBM3XwvT6FfikWw4jm`, then `demo-release/2026-10-09-submitted` = `dpl_JCxjvXS1qAjmPimD3AnmjjwgEEjs` |

## What is included
Everything is merged into main.

| PR | What it does |
|---|---|
| #56 | HF radar falls back to NOAA NDBC's copy when CoastWatch West Coast ERDDAP refuses or reloads. Satellite alerts are per product and platform. A host that refuses the runners is not asked again that run. |
| #46 | VIIRS 750 m and port chlorophyll come from NOAA-21 (NOAA-20 filling gaps). The newest valid lineage wins over older days, with a 7-day window. |
| #58 | One chlorophyll reading per place, shared by the briefing, the Science tab and the map: water pixels only, shoreline and isolated extremes set aside and named, no fixed cap. Stale C-HARM, HF radar and satellite readings are labelled where they are read. The phone map sheet keeps at least 55 % of the map visible. |
| #54 | Next.js 16.4 and React 19.3 (stops React #418). |
| #55 | Restores doc 28 (historical). |
| #59 | Unique ids for the two HAB evidence headings (the briefing's region had the wrong accessible name). The masthead axe check runs after the entrance fade. |
| #60 | Direct CDFW and CDPH links in the masthead on every page in Coastal mode too, as in the previous release. |

Not included: the PINN/LLM research (#49, `research/coastwatch-intelligence`).

## How it was built and promoted
1. Clean export: `git archive <commit> coastwatch-web`, linked to `coastwatch-demo` (`vercel link --project coastwatch-demo`). The `.env.local` it writes was deleted.
2. Production build without the public domain:
   `vercel deploy --prod --skip-domain --build-env CW_DATA_BASE_URL=https://yashnil.github.io/habs-forecast/v1 --env CW_DATA_BASE_URL=…`
3. Smoke test on that deployment through the protection-bypass **cookie** (a bypass header breaks CORS preflights).
4. `vercel promote <dpl id>`, which re-aliases the prebuilt deployment without rebuilding it.
5. The same checks were run on the public URL without login.

**A first candidate was promoted and then superseded.** It was `dpl_cQrdczKyPfxzWkDsKLtrj4sZxXjv` (main 3500e1a4).
- The post-promotion check found that Coastal mode (#51) had dropped the direct CDFW/CDPH links on pages such as About and Bloom.
- That was fixed in #60 and released as above.
- Its rollback was never needed.

## Checks
- **CI:** GitHub CI is green on main at every merge, `ade20099` included.
- **Linux container matching CI** (Playwright 1.64 noble, Node 22, CI=1):
  - lint 0 errors (4 warnings);
  - typecheck clean;
  - unit 310/310;
  - e2e **244/244**, including the hydration regression spec and the axe checks.
- **Candidate and public URL:**
  - All 22 ports' My Coast pages: HTTP 200, 0 console errors, 0 #418.
  - Cold loads: 30 interleaved, 0 hydration errors.
  - 16 page loads (landing, My Coast, a port briefing, Ocean Map, Bloom, Fisheries, About, Sources; desktop and phone): all 200, with 0 hydration and 0 console errors.
  - Saved briefing: dated and labelled, with its links.
  - Map layers on desktop, 390 px and 320 px: correct freshness words. At peek the map stays 73–78 % visible at 390 px and 62–65 % at 320 px, with no horizontal scroll.
  - Duplicate ids on My Coast: none.
  - Notices: shown as **Not verified** in the pill, the drawer and My Coast. SN26-021 reads "Reported lifted by the agency · awaiting review".
  - CDFW and CDPH links are in the masthead on every page.
- **External links:** 37 of 38 answer 200. `https://www.fisheries.noaa.gov/foss` answers 403 to automated clients, a headless browser included (NOAA bot protection), so it could not be checked from here.

## Data at release (published 2026-10-15 04:42Z)

| Source | State |
|---|---|
| C-HARM | issued 2026-10-14, current |
| HF radar | current (hourly via NOAA NDBC while `ucsdHfrW2` is unknown on coastwatch.pfeg) |
| Sentinel-3A OLCI and VIIRS (NOAA-21/20) | newest pixels 2026-10-12 |
| Port chlorophyll | 21 of 22 ports |

- **Sentinel-3B** has had no scene since 2026-10-01. Health has an open alert for it, and coverage is reduced, not filled.
- **San Diego:** its 60-day NOAA-21/20 history did not finish within the run's time budget. Its port panel says **"Satellite values unavailable"** until a successful observation is published; nothing is filled in.

## Known limitations
- **Notices:** official notices are an unreviewed transcription (`pending_human_review`). A person must run `cwp review-official` after checking the sources.
- **Chlorophyll:** chlorophyll is algae biomass, not toxin. Readings close to shore are less reliable. The cross-sensor and cloud-edge flags (Q4, Q5) and pipeline-side flags are not implemented.
- **C-HARM v3:** only coastwatch.pfeg.noaa.gov serves it. When that server refuses or reloads, the last run stays with its own date and is labelled stale. NOAA ERD has been asked about the runner refusals.
- **Phone label:** on phones the map page keeps the "Research preview" label in the menu and footer, not the masthead (as in the previous release).
- **Lint:** one Next 16 lint warning (`no-location-assign-relative-destination`, PurposeSwitch.tsx).

## Rollback procedure
```
cd <any directory linked to coastwatch-demo>
vercel rollback dpl_jxGvioj1FRFnwf6AM577bSczWFue --scope yashnils-projects   # demo-release/2026-10-14
vercel inspect coastwatch-demo.vercel.app --scope yashnils-projects         # confirm the id
```
Then open the public URL without login and check the landing page, My Coast and the masthead links. Rolling back does not change the published data. The previous release reads the same dataset and validates it against the same schema.
