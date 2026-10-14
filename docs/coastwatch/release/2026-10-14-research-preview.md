# Research preview release, 2026-10-14

CoastWatch's full site replaces the map-only Demo Release at the URL submitted to BASES. It is a
**research preview**: no independent scientist or agency has reviewed it, and it is not a regulatory
or seafood-safety service.

| | |
|---|---|
| Production URL | https://coastwatch-demo.vercel.app (public, no login) |
| Vercel project | `yashnils-projects/coastwatch-demo` (`prj_QOAYjTDc2W5U81WwwJ288zEijewY`), Git integration disconnected |
| Deployment | `dpl_jxGvioj1FRFnwf6AM577bSczWFue` (coastwatch-demo-cef13qpf8), target production |
| Commit | `2a995992` on `release/2026-10-14`, tag **`demo-release/2026-10-14`** |
| Released | 2026-10-14 (UTC), with the owner's approval |
| Data | the published dataset, https://yashnil.github.io/habs-forecast/v1 (unchanged by this release) |
| Rollback | `demo-release/2026-10-10` = `dpl_GpFfbFP42QGBM3XwvT6FfikWw4jm` (map-only Demo Release, 372a80c); before it, `demo-release/2026-10-09-submitted` = `dpl_JCxjvXS1qAjmPimD3AnmjjwgEEjs` |

## How it was built and promoted
1. Clean export: `git archive 2a995992 coastwatch-web`, linked to `coastwatch-demo`
   (`.vercel/project.json`), recorded test fixtures removed.
2. Staged production build, not yet public:
   `vercel deploy --prod --skip-domain --build-env CW_DATA_BASE_URL=… --env CW_DATA_BASE_URL=…`.
   - `NEXT_PUBLIC_CW_DEMO` is **not** set: the old map-only switch no longer exists, and this is the full site.
   - `CW_NWS_API` is **not** set: NWS's API is linked, not queried (see Limitations).
   - The project stores no environment variables.
3. The candidate was checked through the protection-bypass cookie, with the smoke test described below.
4. `vercel promote dpl_jxGvioj1FRFnwf6AM577bSczWFue`, which re-aliases the prebuilt production
   deployment without rebuilding it.
5. The same smoke test was run on the public URL without login.

`--skip-domain` also moved the team-scoped alias `coastwatch-demo-yashnils-projects.vercel.app` to
the candidate. That alias is behind Vercel login, like every deployment URL.

## What is included
The release branch combines:
- **#41**, Coastal redesign:
  - Coastal and Research modes;
  - the welcome;
  - the map and motion system;
  - geographic provenance;
  - the HAB-focused My Coast briefing (shore evidence, the C-HARM neighbourhood with the nearshore gap kept apart, satellite chlorophyll);
  - species- and activity-driven next steps;
  - CDFW season status shown first;
  - the disputed-notice handling;
  - the conditions map focus.
- **#47**, fishermen-first: "Since you last checked", remembered waters, activity and species, and kept (dated, printable) briefings.
- **Release safeguards:**
  - "Research preview" on every page (masthead on wide screens, footer, welcome, About, page title);
  - the footer says CoastWatch does not replace CDFW or CDPH;
  - no request to `api.weather.gov`.

Not included:
- **#45**, data monitoring: a pipeline change the website does not need.
- **#46**, VIIRS NOAA-21/20: an unfinished scientific change, with a documented coverage shortfall and open decisions.
- **#48**, honest user agent: a pipeline change, which only takes effect from main.

None of these affects the website. They need the owner's merge into main.

## Checks before promotion
- **Linux container matching CI** (Node 22, CI=1, 2 workers):
  - full run: 239 of 240 browser tests passed. The failure was in the new research-preview test, whose title assertion checked /about instead of the site title on "/"; it was corrected;
  - the release, coastal and coast specs repeated twice: 72/72.
- **Unit tests:** 270. They include the NWS gate: no request reaches api.weather.gov unless `CW_NWS_API=1`.
- **Pipeline:** 205 passed. The pipeline is unchanged from main except one relief-build script.
- **Lint and typecheck:** 0 errors.
- **Smoke test on the candidate and again on the public URL** (published data):
  - first visit at 1440, 390 and 320 px;
  - the Ocean Map forecast layer;
  - My Coast for Monterey commercial, Crescent City Dungeness and Clam Beach shellfish, on desktop and phone;
  - SN26-018 shown as "Reported lifted by CDPH · awaiting review", with "Not verified";
  - the model-grid map focus;
  - species selection;
  - switching between Coastal and Research;
  - remembered waters and the change briefing;
  - the Bloom, Fisheries, About, Sources and Kept pages;
  - `/api/conditions` (NDBC and CO-OPS live, NWS linked) and `/api/briefing`;
  - CDFW and CDPH links in the masthead;
  - open-water unavailable states.
  
  All passed.

## Known limitations (public)
- Most ports sit in C-HARM's nearshore gap: CoastWatch shows nearby offshore cells, kept apart, and no nearshore toxin estimate.
- Shore sampling is sparse, and fog often hides the satellite view.
- The official notices are transcribed and not verified by a person. SN26-021 (and SN26-020) await review.
- The marine forecast and warnings are a link to NWS. CoastWatch does not query NWS's API, because its robots.txt disallows automated clients.
- HF radar currents have been stale since 2026-10-12, and the app labels them so.

## Known issue
- **An intermittent React hydration mismatch (#418).**
  - It occurs on about 1 in 30 page loads, on any page, in production builds only.
  - It has been present since #41 and is absent on main.
  - React recovers by re-rendering that part of the page in the browser, so the content is correct.
  - It was not reproducible in development builds. Tracked for a follow-up.

## Rollback
```
vercel rollback dpl_GpFfbFP42QGBM3XwvT6FfikWw4jm --scope yashnils-projects   # or: vercel promote dpl_GpFfbFP42QGBM3XwvT6FfikWw4jm
```
Then check that https://coastwatch-demo.vercel.app serves the title "CoastWatch — California
coastal bloom forecasts".
