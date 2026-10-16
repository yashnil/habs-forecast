# Release candidate: Coastal intelligence v2 (prepared 2026-10-15, evening)

Prepared for the owner's decision. Nothing here has been merged, promoted or deployed to the public
URL. The public research preview at the BASES URL is unchanged: `dpl_HfUScMViHiKPs2aMC9TAsQWrTHjA`
(main `ade20099`, tag `demo-release/2026-10-15`).

| | |
|---|---|
| Candidate commit | **`9a650960`**, head of `feat/coastal-intelligence-v2` (draft PR #65) |
| Equivalent after merge | `main` once #62, #63 and #65 are squash-merged in that order |
| Public URL (unchanged) | https://coastwatch-demo.vercel.app → `dpl_HfUScMViHiKPs2aMC9TAsQWrTHjA` |
| Rollback target (unchanged) | the same deployment; the one before it is `dpl_jxGvioj1FRFnwf6AM577bSczWFue` (`demo-release/2026-10-14`) |
| Private preview | Vercel project `coastwatch-preview` (deployment-protected). The upload is blocked by the free tier's 5,000-file window until about 17:15 PDT on 2026-10-16; one retry is prepared, nothing else in Vercel is touched. |

## 1. The PR stack, reviewed

| PR | Base | Commits | Files | What it is | Verdict |
|---|---|---|---|---|---|
| #62 `feat/neritic-signal-rebrand` | main | 6 | 84 (+675 −186) | The product rename CoastWatch → Neritic: site copy, README, brand components, `brand.spec`. NOAA CoastWatch credits and technical identifiers kept. | Product change. Merge first. The name is not legally cleared (owner's note); nothing in the code depends on that. |
| #63 `design/neritic-excellence` | #62 | 1 | 91 (+20,046 −0) | `docs/coastwatch/design/2026-10-neritic-excellence/`: the design audit, reference study, logo directions and two interface prototypes (46 webp, 12 svg, 9 html, 7 mjs; 5.3 MB committed). | Documentation only, no file under `coastwatch-web/src` or `pipeline/`. Merge second as the design record that #65's decisions cite, or keep it as a branch if main should stay lean. |
| #65 `feat/coastal-intelligence-v2` | #63 | 17 + this pass | 233 (+8,650 −841) | The product work: trust fixes, the evidence graph and `/api/evidence`, My Coast on the phone, CO-OPS harbour water, harbour-zoom map, the pipeline's OLCI 404 handling, the last name references, the masthead targets. Plus `docs/coastwatch/v2/` (145 files, 14.2 MB committed). | Product change. Merge last. See §2 for what should not come with it. |

**Safest integration order: bottom-up, one squash merge per PR.**
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
| `docs/coastwatch/v2/carto/shots/*.png` | 9.4 MB, 24 files | Full-size captures of the cartography prototype (T0–T4). The review page already carries JPEG versions of the ones that matter. | Remove before merge, or re-encode as webp (about 1 MB). The prototype itself (`index.html`, `ramp.mjs`, `serve.mjs`, `shoot.mjs`) stays: it is the record of the approved palette. |
| `docs/coastwatch/v2/carto/data/water-fallback.json` | 0.4 MB | Byte-identical copy of `coastwatch-web/public/basemap/water-fallback.json`. | Remove; point the prototype at the app's file. |
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

## 4. Release sequence (every step needs the owner's go; nothing below has been run)

1. **Approve** the merge order (§1) and the pruning (§2). I merge #62, #63, #65 as described and wait for CI on `main`.
2. **Data:** nothing to publish. The manifest schema is unchanged; the pipeline fixes in #65 (OLCI time listing, user agent) take effect on the next scheduled data run.
3. **Build the candidate without touching the public URL:** `git archive main coastwatch-web` into an empty directory, `vercel link --project coastwatch-demo`, delete the `.env.local` it writes, then
   `vercel deploy --prod --skip-domain --build-env CW_DATA_BASE_URL=https://yashnil.github.io/habs-forecast/v1 --env CW_DATA_BASE_URL=https://yashnil.github.io/habs-forecast/v1`
   This produces a production-target deployment that is not aliased: the BASES URL keeps serving `dpl_HfUScMViHiKPs2aMC9TAsQWrTHjA`.
4. **Smoke test the candidate** through the protection-bypass cookie (a bypass header breaks CORS preflights): all 22 ports' My Coast pages 200 with 0 console errors and 0 React #418 over 30 cold loads; page vs `/api/evidence` on Monterey, Crescent City, San Diego, Shelter Cove; the masthead at 320 and 390 px; "Not verified" on the pill, the drawer and My Coast; CDFW and CDPH links on every page; the harbour-zoom map; the saved briefing link.
5. **Owner GO → `vercel promote <candidate dpl>`**, which re-aliases the prebuilt deployment without rebuilding. Repeat the checks on the public URL without login.
6. **Record:** tag `demo-release/2026-10-16` on the `main` commit, write `docs/coastwatch/release/2026-10-16-research-preview.md`, update the memory note.

## 5. Rollback (one command, no rebuild, data untouched)

- `vercel rollback dpl_HfUScMViHiKPs2aMC9TAsQWrTHjA` returns the public URL to today's release (`demo-release/2026-10-15`). Behind it: `dpl_jxGvioj1FRFnwf6AM577bSczWFue` (`demo-release/2026-10-14`).
- The web app reads the published dataset at `https://yashnil.github.io/habs-forecast/v1`, which this release does not change, so there is no data rollback.
- The official-notice registry stays `pending_human_review`; nothing in this release marks anything verified, lifted or expired.
- `main` is never rewritten; if a merged PR must come out, it is a revert commit.

## 6. Checks on the candidate commit

- Web: 385 unit tests; lint 0 errors (4 pre-existing warnings); typecheck clean.
- Playwright, one complete run with the fixture servers (`npm run test:e2e`, installed Chrome, no retries): **262 of 262 passed** in 6.0 min.
- Pipeline: unchanged since CI's green run on `ff02d31f`.
- GitHub CI on the pushed head: linked from the PR.
