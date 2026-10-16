# Research preview release, 2026-10-16: Neritic v2

The public research preview at the BASES URL now runs Neritic v2 (main as of 2026-10-16). This release brings:
- the product name Neritic (#62);
- the design record (#63);
- Coastal Intelligence v2 (#65): trust fixes, one evidence backbone for My Coast, in-harbour water temperature from NOAA CO-OPS, the harbour-zoom map, and the 44 px phone masthead.

It remains a **research preview**. No independent scientist or agency has reviewed it, and it is not a regulatory, fishing or seafood-safety decision service. The Neritic name is accepted for this independent research preview only. It has not been cleared as a trademark (see `docs/coastwatch/brand/neritic.md`).

| | |
|---|---|
| Production URL | https://coastwatch-demo.vercel.app (public, no login) |
| Vercel project | `yashnils-projects/coastwatch-demo` (`prj_QOAYjTDc2W5U81WwwJ288zEijewY`), Node.js 24.x, Git integration disconnected; no setting changed |
| Deployment | **`dpl_HyAetdpnw23aJ1oKu3CFFGjDoxt5`** (coastwatch-demo-hah1zh51r), target production, promoted 2026-10-16 ~16:25 UTC with the owner's approval |
| Deployed code | `git archive 3a9c9d89:coastwatch-web`, web tree **`3032bb89`**: 1,445 files checked against the git blobs, 0 differences. This is the tree the private preview and the test runs used. |
| Commit and tag | **`4d2e3e1c`** on `main` (full tree `a7788565`, the same as `3a9c9d89`), tag **`demo-release/2026-10-16`** |
| Data | https://yashnil.github.io/habs-forecast/v1 (`CW_DATA_BASE_URL` at build and run time). At promotion: run 38893860718 (pipeline `fc448079`, 11:27Z). First run on the final pipeline: **38937129623** (pipeline `4d2e3e1c5da2`, generated 2026-10-16 17:06:51Z), dispatched by hand after the merge. The schema is unchanged. |
| Rollback | **`vercel rollback dpl_HfUScMViHiKPs2aMC9TAsQWrTHjA --scope yashnils-projects`** (`demo-release/2026-10-15`, verified Ready before promotion). Before it: `dpl_jxGvioj1FRFnwf6AM577bSczWFue` (`demo-release/2026-10-14`). |

## Release order: promote first, merge after

Scheduled data runs use `main`'s pipeline, and the stack's display-text changes start with #62. So the new UI was promoted before anything merged, and the old CoastWatch UI never received Neritic-branded data text.

1. **Candidate:** `vercel deploy --prod --skip-domain` on `coastwatch-demo`, not aliased to the BASES URL. Vercel moved the project's own team alias (`coastwatch-demo-yashnils-projects.vercel.app`) to it, as it does for every production deployment. That alias stays behind Vercel login (302).
2. **Candidate checks, through the bypass cookie:**
   - the production dataset is served (run 38893860718), and `/api/evidence` is identical to the private preview for four harbours;
   - 51 page loads (17 pages × 1440/390/320): 0 errors, no sideways scroll;
   - 30 cold loads with 0 hydration errors;
   - saved briefing, Copy link and 7 shared links work;
   - axe WCAG 2.1 AA: 0 violations; reduced motion: no running animation;
   - 33 external links (only NOAA FOSS refuses scripts; it opens in a browser);
   - mixed wording: only the Fisheries tier note from the then-current data.
3. **Promoted** `dpl_HyAetdpnw23aJ1oKu3CFFGjDoxt5`, then the same checks on the public URL without login: all passed.
4. **Squash merges:**
   - #62 → `5e2db159`: main's tree = #62's tree, CI green.
   - #63 → `de16e486`: main's tree = #63's tree, docs only.
   - #65 → `4d2e3e1c`: main's tree = #65's tree, web tree `3032bb89`, CI green.
5. **First data on the final pipeline** (run 38937129623): every job succeeded. Health issue #44 was reused, with no duplicate. The registry is still `pending_human_review`, and the SN26-021 conflict is still published.

### What the merge needed (for the next stacked release)

- Deleting #62's branch **closed** #63 instead of retargeting it. The branch was restored at the same commit `86ac4c1a`, #63 was reopened and retargeted to `main`, and the branch was deleted again. #63 was merged without deleting its branch, #65 was retargeted, and then the branch was deleted.
- After the squash merges, #65 conflicted textually with `main` (its later edits touched lines the squashed #62 had changed). `main`'s tree equalled the tree of `38376407`, an ancestor of #65, so `main` had nothing #65 lacked. One merge commit (`71c394e8`, `-s ours`) recorded `main` as a parent and kept #65's tree `a7788565` unchanged. CI was green on it before the squash.
- `feat/coastal-intelligence-v2` is kept because draft PR #66 (Terminal 2's v3, not part of this release) is based on it. Delete it after #66 is retargeted.

## Verification on the public site after the final data run

- 51 page loads (17 pages × 1440/390/320): 200, 0 console or page errors, no sideways scroll, "Not verified" on every page, Neritic on every page.
- 30 cold loads: 0 hydration errors. axe: 0 violations. Reduced motion: no running animations.
- LCP 0.52–1.21 s, unthrottled.
- No product-name "CoastWatch" text in either mode, including expanded details.
- `/api/evidence` on all 22 ports:
  - no offshore C-HARM value presented as a harbour's own;
  - every C-HARM node carries its run date (Oct 14) and valid day (Oct 16, "2 days old");
  - every notice is flagged "not verified";
  - no safety, closure or catch wording; no dangling citation.
- CO-OPS harbour water at Monterey, Crescent City, San Diego, Point Arena, Avila and Eureka: station, distance (≤ 6 km), sensor depth and a 6-minute time. Santa Cruz: "no station within 6 km".
- Before the merge: 262/262 Playwright tests (macOS, one complete run), 385/385 unit tests, 246/246 pipeline tests; GitHub CI green on every merged head.

## Known and open

- **Official notices:** not verified (`pending_human_review`). CDPH SN26-021 appears to lift SN26-018; SN26-020 and the CDFW page changed. See `2026-10-16-notice-review-guide.md`. `cwp review-official` was not run; only a person may run it.
- **Legacy provenance text in carried-forward data:** five daily OLCI layers (Oct 9–14), built before the rename, still say "Rendered by CoastWatch" in their provenance. No page shows it, and they leave the 7-day window by about Oct 21.
- **Nonblocking display issues:**
  - the masthead's CDFW/CDPH links sit slightly below the text baseline;
  - on phones the layer sheet covers MapLibre's 29 px zoom buttons (pinch zoom works);
  - the independence statement appears twice on About and Sources.
- **Evidence behind deployment protection:** `/api/evidence` fetches the water mask and `/api/conditions` from its own origin, so on a protected preview it falls back (VIIRS 750 m, no NWS line). The public URL has the full evidence.
- **Rollback limit:** a UI rollback now also needs a data revert to undo the Neritic text in the published data. Revert the squash commits on `main` and run "CoastWatch data refresh"; `main` is never rewritten.

The logo is the mark from #62. The logo studies in #65 are documentation only, and nothing from them is installed. No v3 work (#66), experimental forecasting or new AI feature is included.
