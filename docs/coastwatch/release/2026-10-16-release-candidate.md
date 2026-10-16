# Release candidate: Coastal intelligence v2 (prepared 2026-10-15, evening; verified on the private preview 2026-10-16)

Prepared for the owner's decision. Nothing here has been merged, promoted or deployed to the public
URL. The public research preview at the BASES URL is unchanged: `dpl_HfUScMViHiKPs2aMC9TAsQWrTHjA`
(main `ade20099`, tag `demo-release/2026-10-15`).

| | |
|---|---|
| Candidate | the `coastwatch-web` tree **`3032bb89`** (unchanged since `9a650960`; the head of draft PR #65 adds docs, a test-timing fix, the approved pruning and two restored identifiers, none of which is in the web bundle) |
| Equivalent after merge | `main` once #62, #63 and #65 are squash-merged in that order |
| Public URL (unchanged) | https://coastwatch-demo.vercel.app → `dpl_HfUScMViHiKPs2aMC9TAsQWrTHjA` |
| Rollback target (unchanged) | the same deployment; the one before it is `dpl_jxGvioj1FRFnwf6AM577bSczWFue` (`demo-release/2026-10-14`) |
| Private preview | `dpl_APuqAoSi48UygLivnTG3Zegxy7VM`, https://coastwatch-preview-knqgz1ln7-yashnils-projects.vercel.app (project `coastwatch-preview`, target preview, behind Vercel Authentication), deployed 2026-10-16 07:27 PDT from `git archive d8d5b712:coastwatch-web` (tree `3032bb89`) once the upload window had reset; one attempt, no project setting changed. |

## 1. The PR stack, reviewed

| PR | Base | Commits | Files | What it is | Verdict |
|---|---|---|---|---|---|
| #62 `feat/neritic-signal-rebrand` | main | 6 | 84 (+675 −186) | The product rename CoastWatch → Neritic: site copy, README, brand components, `brand.spec`. NOAA CoastWatch credits and technical identifiers kept. | Product change. Merge first. The name is not legally cleared (owner's note); nothing in the code depends on that. |
| #63 `design/neritic-excellence` | #62 | 1 | 91 (+20,046 −0) | `docs/coastwatch/design/2026-10-neritic-excellence/`: the design audit, reference study, logo directions and two interface prototypes (46 webp, 12 svg, 9 html, 7 mjs; 5.3 MB committed). | Documentation only, no file under `coastwatch-web/src` or `pipeline/`. Merge second as the design record that #65's decisions cite, or keep it as a branch if main should stay lean. |
| #65 `feat/coastal-intelligence-v2` | #63 | 17 + this pass | 233 (+8,650 −841) | The product work: trust fixes, the evidence graph and `/api/evidence`, My Coast on the phone, CO-OPS harbour water, harbour-zoom map, the pipeline's OLCI 404 handling, the last name references, the masthead targets. Plus `docs/coastwatch/v2/` (145 files, 14.2 MB committed). | Product change. Merge last. See §2 for what should not come with it. |

**Safest integration order: bottom-up, one squash merge per PR**, and only after the new UI is promoted (§4).
1. Squash-merge #62 into `main`; wait for CI on `main`.
2. GitHub retargets #63 to `main` when #62's branch is deleted; squash-merge it (docs only, CI is a no-op for it).
3. GitHub retargets #65 to `main`; squash-merge it; wait for CI on `main`.

Not recommended: merging #65 alone (it contains #62 and #63) would flatten three milestones into one commit and leave #62 and #63 to be closed by hand; merging with merge commits would put 24 branch commits on `main` against the repository's squash rule.

Before step 3 the owner decides on §2. Each pruning is a normal commit on #65 (no history rewrite).

## 2. Files that should not enter production as they are

Nothing under `docs/` ships: a release is a `git archive` of `coastwatch-web/` alone (§4), so none of the
following is in the deployed bundle. The question is what `main` carries.

| Path | Size (committed) | What it is | Recommendation |
|---|---|---|---|
| `docs/coastwatch/v2/carto/shots/*.png` | 9.4 MB, 24 files | Full-size captures of the cartography prototype (T0–T4). The review page already carries JPEG versions of the ones that matter. | **Removed (owner-approved, 2026-10-16)**; `shots/*.png` is now ignored and `shoot.mjs` regenerates them. `shots/perf.json`, which the README quotes, stays. The prototype itself (`index.html`, `ramp.mjs`, `serve.mjs`, `shoot.mjs`) stays: it is the record of the approved palette. |
| `docs/coastwatch/v2/carto/data/water-fallback.json` | 0.4 MB | Byte-identical copy of `coastwatch-web/public/basemap/water-fallback.json`. | **Removed (owner-approved, 2026-10-16)**; `serve.mjs` serves the app's file at the same path, so the prototype is unchanged. |
| `docs/coastwatch/v2/carto/data/*.u16.gz`, `relief-*.webp` | 0.5 MB | The 15 Oct C-HARM grids and relief tiles the prototype renders. | Keep (small, and the prototype is not reproducible without them). |
| `docs/coastwatch/v2/logo/out/`, `logo/shots/` | 1.3 MB, 75 files | Generated logo candidates and board captures. The decision is still open (2b recommended). | Keep until the logo decision; then keep only the chosen direction's sources. `logo/fonts/` is gitignored (OFL files fetched at run time). |
| `docs/coastwatch/v2/review/` | 2.3 MB | The private review page and its JPEGs. | Keep: it is the acceptance record. |
| `docs/coastwatch/design/…` (#63) | 5.3 MB | The design review's prototypes and references. | Owner's call (see §1). |
| `coastwatch-web/tests/fixture-data-failed/`, `tests/conditions-fixtures.json` | 0.8 MB | Recorded fixtures CI needs (failed-update dataset, replayed NOAA conditions). | Keep; required by the e2e suite. |
| `coastwatch-web/scripts/fishermen-bench.mjs` | small | Pre-existing (from #50), only renamed here. | Keep. |
| `.claude-private/` | untracked | Session progress notes. | Never committed; stays untracked. |

**Experimental code in the product, left in deliberately and labelled:**
- `GET /api/evidence`: the evidence graph as JSON. It is what the page renders and what the consistency tests read; it is not documented as a public API and may change. No key, no write.
- The light map variant exists only in the cartography prototype, not in the app.
- No experimental model, no prediction, no new HAB measurement is in the branch.

## 3. What this pass verified

- **One source of claims.** Every sentence that states evidence on My Coast comes from `lib/harborEvidence.ts`: the "Right now" lines, the Research-mode answer line, the evidence rows, the map's thread, and (since this pass) the HAB detail's summary paragraph, which used to build its own sentences and quoted an offshore median the graph deliberately avoids. `tests/e2e/evidence.spec.ts` now asserts, for four harbours against `/api/evidence` on the same clock: each line's id, kind, headline and supporting node ids; each evidence row's value; the summary paragraph's text; the map thread's distance.
- **Scientific semantics intact** (unit tests in `harborEvidence.test.ts`): a model value is "a probability, not a measurement"; offshore cells are context, never the harbour's own; an old run is named by its date and age; chlorophyll is "algae of every kind, not toxin"; no claim uses safe, clear, clean, risk or catch words; every claim cites nodes that exist. The pipeline's published numbers are untouched.
- **Fewer repeated facts.** On harbours without a shore sample (Crescent City, Shelter Cove, San Diego) the HAB line restated the model and the model line said it again; the HAB line now keeps to the shore stations.
- **Masthead targets.** Every control in the masthead and the status line is a 44 × 44 px target at 320 and 390 px with no two hit boxes overlapping, while the drawn pill stays 32 px and the masthead row 52 px; the phone header grew by 2 px (86 → 88). The skip link is visually hidden until focused, and is 44 px when focused. Pinned by `release.spec.ts`.
- **Harbour zoom.** The distance label at the nearest model cell is anchored on the side away from the harbour, so it no longer covers "Crescent City" (2.4 km away).
- **320 px.** No sideways scroll on any inspected page; the docked map bar's label never wraps; the status line wraps to two lines by design.

## 4. Release sequence: promote first, merge after (every gate needs the owner's go; nothing below has been run)

The owner approved the release direction on 2026-10-16 and asked for an order in which the scheduled data run can never publish Neritic-branded text to the old public CoastWatch UI. Data runs use `main`'s pipeline (every 6 h at minute 17 UTC). The stack's pipeline text changes start with #62, so in this order no merge happens until the new UI is public. The new UI on the current data is exactly what the private preview verified (§7).

The Neritic name is accepted for this independent research preview only. It is not trademark clearance (see `docs/coastwatch/brand/neritic.md`).

1. **Preflight.** `origin/main` is `fc448079`. Let `H` be #65's final head (`git rev-parse origin/feat/coastal-intelligence-v2`), with CI green. `git rev-parse H:coastwatch-web` is `3032bb89` (the preview's tree; commits after `211ead29` are docs only). Terminal 2's `feat/harbor-intelligence-v3` is not in any of the three branches; it stays local and unpushed until v2 is on `main`.
2. **Candidate on the demo project, not aliased.** `git archive H:coastwatch-web | tar -x -C <empty dir>` (the project root is `.`, so the app goes at the archive root), `vercel link --project coastwatch-demo --scope yashnils-projects`, delete the `.env.local` it writes, then
   `vercel deploy --prod --skip-domain --build-env CW_DATA_BASE_URL=https://yashnil.github.io/habs-forecast/v1 --env CW_DATA_BASE_URL=https://yashnil.github.io/habs-forecast/v1`
   The BASES URL keeps serving `dpl_HfUScMViHiKPs2aMC9TAsQWrTHjA`.
3. **Smoke test the candidate** through the protection-bypass cookie (a bypass header breaks CORS preflights), with the same checks as §7:
   - 17 pages × 1440/390/320 px: 200, 0 console or page errors, no sideways scroll, "Not verified" present, no product-name CoastWatch except the Fisheries tier note from the live data;
   - 30 cold loads with 0 hydration errors;
   - `/api/evidence` invariants on all 22 ports;
   - CO-OPS lines for Monterey, Crescent City and San Diego;
   - a kept briefing, Copy link and shared links;
   - axe on six pages.

   Stop on any failure.
4. **Owner GO → `vercel promote <candidate dpl> --scope yashnils-projects`.** This re-aliases the prebuilt deployment without rebuilding. Repeat the smoke test on https://coastwatch-demo.vercel.app without login, and check that `vercel inspect coastwatch-demo.vercel.app` names the candidate. **Rollback in this window:** `vercel rollback dpl_HfUScMViHiKPs2aMC9TAsQWrTHjA`. It is complete, because no data has changed.
5. **Owner GO → merge, bottom-up, one squash merge per PR:**
   1. #62 into `main`, then wait for CI on `main`.
   2. #63 (GitHub retargets it to `main` when #62's branch is deleted).
   3. #65 (retargeted to `main`), then wait for CI on `main`.

   Then `git rev-parse origin/main^{tree}` must equal `git rev-parse H^{tree}` (record `H` before #65's branch is deleted), so `main` carries exactly the deployed web tree and the final pipeline and workflow fixes. If `main` moved in between, compare `origin/main:coastwatch-web` with `3032bb89` instead, and rebuild if it differs.
6. **First data run after the merge** (scheduled, ≤ 6 h; or a manual `workflow_dispatch` of "CoastWatch data refresh" if the owner wants it sooner):
   - the Fisheries tier note reads "Neritic's editorial grouping";
   - the run commented on issue #44 or opened nothing new (no second "pipeline health" issue);
   - `/api/evidence` still validates.
7. **Record:**
   - tag `demo-release/2026-10-16` on `H`, the commit whose tree was deployed (it stays reachable after the branch is deleted);
   - write `docs/coastwatch/release/2026-10-16-research-preview.md` in a docs PR naming the promoted deployment and `main`'s squash commits;
   - Terminal 2 then rebases v3 onto `main`.

Official notices are a separate track: see `2026-10-16-notice-review-guide.md`. They stay `pending_human_review` until a person reviews them. That doesn't block the steps above, because every notice is labelled Not verified.

## 5. Rollback (one command, no rebuild, data untouched)

- `vercel rollback dpl_HfUScMViHiKPs2aMC9TAsQWrTHjA` returns the public URL to today's release (`demo-release/2026-10-15`). Behind it: `dpl_jxGvioj1FRFnwf6AM577bSczWFue` (`demo-release/2026-10-14`).
- The web app reads the published dataset at `https://yashnil.github.io/habs-forecast/v1`. Its schema does not change, and both UIs read it (the preview is the new UI on today's data; `tests/unit/compat` checks the released schema against new pipeline output), so a UI rollback never needs a data rollback to keep working.
- Between promotion (step 4) and the merge (step 5), a rollback is complete: the data has not changed.
- After the merge and the next data run, a UI rollback does **not** undo the pipeline's text changes: the rolled-back UI would show provenance strings such as "computed by Neritic" beside the CoastWatch name. To undo those as well, revert the three squash commits on `main` (revert commits; `main` is never rewritten) and run "CoastWatch data refresh" (`workflow_dispatch`), or wait for the next scheduled run.
- The official-notice registry stays `pending_human_review`; nothing in this release marks anything verified, lifted or expired.
- `main` is never rewritten; if a merged PR must come out, it is a revert commit.

## 6. Checks on the candidate (re-run 2026-10-16 on the pushed head)

| Check | Result |
|---|---|
| Web unit tests (vitest, Linux container) | 385 of 385, 34 files |
| Lint / typecheck / generated types in sync with `schemas/v1` | 0 errors (4 old warnings) / clean / in sync |
| Pipeline (`pytest -n 4` on the full tree) | 246 of 246, after restoring the User-Agent; `cwp schema` leaves `schemas/` unchanged; both fixture datasets regenerate byte-identically |
| Playwright, one complete run, macOS, bundled Chromium, no retries | **262 of 262** in 4.8 min |
| Playwright in the arm64 Linux container (2 complete runs) | 235/262 and 246/262. Not a pass: almost every failure is "Target crashed" (the renderer dies with software WebGL on arm64 under Docker's memory cap), and the failing set changes between runs. The one failure without a crash, `coast.spec.ts:124`, changes a select before hydration, so on that slow runner the change is lost. GitHub CI (x86-64 Linux) and the macOS run pass all of them. |
| GitHub CI | green on `d8d5b712` (same web tree); the run on the pushed head is linked from PR #65 |

## 7. Production-equivalent verification on the private preview (2026-10-16)

`dpl_APuqAoSi48UygLivnTG3Zegxy7VM`, read through the protection-bypass cookie, on the live dataset generated 2026-10-16 11:27 UTC (pipeline `fc448079`, run 38893860718).

- **17 pages × 1440 / 390 / 320 px (51 loads):** every page 200; 0 console errors, 0 page errors, 0 React #418; no sideways scroll; Neritic on every page; "not verified" on every page; no product-name "CoastWatch" text except the Fisheries tier note from live data (§4 step 2). Pages: landing, Ocean Map (forecast, satellite, currents), My Coast for Monterey, Crescent City, San Diego, Santa Cruz, Shelter Cove and Point Arena (low coverage), a shared crab link, Monterey in Research mode, Blooms in both modes, Fisheries, About, Sources.
- **Against the public release (same matrix):** on phones My Coast, About and Sources have 0 controls under 44 px (public: 4–9 per page); Research-mode repeated sentences fell from 10 to 3 (the notice titles, listed in the notices and again in the evidence table); visible "CoastWatch" product text fell from up to 13 strings per page to 0. Not new and not fixed: MapLibre's 29 px zoom buttons, the 24 px layer-sheet handle and the 40 px Fisheries chips on phones; the sheet covers the zoom buttons on phones (pinch zoom works).
- **30 cold loads** across ten pages, alternating desktop and phone: 0 hydration errors, 0 other console errors.
- **Evidence graph** (`/api/evidence`, all 22 ports): no offshore C-HARM node is the harbour's own value where the harbour has no cell (21 ports state the nearshore gap and the distance, 2.2–11.8 km; Moss Landing is the one port inside a valued cell); every C-HARM node carries the run date (Oct 14) and the valid day (Oct 16, "2 days old"); every notice node is flagged "Neritic's transcription, not verified"; every claim cites nodes that exist; chlorophyll is always "algae of every kind, not toxin"; no claim is a safety, closure or catch statement.
- **Harbour water (CO-OPS):** Monterey 9413450 (1.0 km, thermistor 2.1 m below MLLW), Crescent City 9419750 (0.1 km, 0.9 m), San Diego 9410170 (0.3 km, 5.6 m): 6-minute readings, NOAA's three QC flags passed, station link. Santa Cruz: "No NOAA water-temperature station within 6 km" (no second-station fallback).
- **Flows:** a kept briefing (Crescent City, commercial, Dungeness) reads back on `/coast/kept` with its time and the "saved copy, not current information" warning; Copy link copies the current URL and it opens the same waters in a fresh browser; seven shared links open cold with their place, layer and day.
- **Accessibility:** axe (WCAG 2.1 A/AA) on six pages at 1440 and 390 px: 0 violations. Reduced motion: no running animation on the landing, the currents map or My Coast; the currents flow stays off.
- **Links:** 33 external links on About, Sources, My Coast, Blooms, Fisheries and the map; 32 return 200 to a script, and every CDFW, CDPH, SCCOOS and NOAA ERDDAP link opens its intended page. The NOAA Fisheries FOSS link refuses scripts (403) and opens in a browser (200).
- **Timing** (one cold load, no throttling): LCP 0.41–1.52 s on desktop, 0.49–0.75 s on a 390 px phone; CLS ≤ 0.035; 204–596 KB of JavaScript per page.
- **Official notices:** the registry is `pending_human_review` (AI transcription of 2026-10-08, not checked by a person). The watcher reports CDPH SN26-021 and SN26-020 as unreviewed and one conflict (SN26-021 appears to lift SN26-018); the pages say exactly that. Only a person running `cwp review-official` after checking the sources may change it.

### Two identifiers restored on #65

The last rename pass (`c588656e`) had changed two strings that the brand notes keep as identifiers. Both are back: the data workflow's "CoastWatch pipeline health" issue title (the open issue #44 is found by exact title; the new title would have opened a duplicate and left #44 open), and the pipeline User-Agent `CoastWatch-pipeline/0.1 (+repo)` (unchanged while NOAA ERD investigates the runner 403s).
