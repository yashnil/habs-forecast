# 10 — M2 integration and staging review

Date: 2026-10-08. PR: [#3](https://github.com/yashnil/habs-forecast/pull/3) (`feat/coastwatch-m2-ocean-intelligence` → `main`). The public website remains undeployed. M3 has not been started.

## 1. Inputs

- GitHub Pages enabled (Source: GitHub Actions). The maintainer's manual run of "CoastWatch data hosting (GitHub Pages)" ([37840212743](https://github.com/yashnil/habs-forecast/actions/runs/37840212743)) deployed and verified the M1-era dataset at `https://yashnil.github.io/habs-forecast/v1/`.
- Independent check from a workstation: manifest served with `access-control-allow-origin: *`, `content-type: application/json`, `cache-control: max-age=600`.

## 2. Findings and fixes

| # | Finding | Severity | Fix |
|---|---|---|---|
| 1 | **Schema v1 break.** M2 made `ports.features[].properties.county/region` required without bumping `schema_version`. The M2 `check-published` rejected the M1-era dataset on Pages (44 validation errors), so M2 readers would reject data written by the M1 pipeline during the switch-over. | High (integration) | Fields made optional (additive change). Compatibility tests in the pipeline and the web app now validate the actual Pages-hosted M1 files (`pipeline/tests/fixtures/compat/m1/`). M2 app on M1 data: official notices and port summaries show **Unavailable** states; forecast and map work (screenshot `m2-staging/04-transition-m1-data.png`). |
| 2 | **Node.js 20 deprecation warnings** on `actions/checkout@v4`, `setup-node@v4`, `upload-artifact@v4`, `download-artifact@v4`, `deploy-pages@v4`, `astral-sh/setup-uv@v5`. | Low | Each moved to its oldest Node 24 major, confirmed from each release's `action.yml`: checkout v5, setup-node v5, upload-artifact v6, download-artifact v7, deploy-pages v5, upload-pages-artifact v5 (its composite steps use upload-artifact v7), setup-uv v7. New CI and data runs report **0** Node 20 annotations. |
| 3 | No way to run the M2 pipeline on GitHub without changing the production dataset. | Medium | Data workflow gains a manual `channel` input. `staging` publishes to `coastwatch-data-staging`, which GitHub Pages never serves and which opens no review issues. |
| 4 | Data & sources page: port summaries were badged "Historical"; the official row labelled the review date "Latest valid". | Low (wording) | New additive product class `derived_summary` ("Derived summary"); label "Last reviewed or transcribed". |

## 3. Staging run on GitHub (real M2 pipeline)

Run [37840887573](https://github.com/yashnil/habs-forecast/actions/runs/37840887573) (`workflow_dispatch`, ref `feat/coastwatch-m2-ocean-intelligence`, channel `staging`): build → publish → public-URL check all green; review notification skipped (staging); production `coastwatch-data` branch untouched.

| Check | Result |
|---|---|
| `check-published` against `https://raw.githubusercontent.com/yashnil/habs-forecast/coastwatch-data-staging/v1` | ✅ 28/28 files (manifest, 12 images, 12 grids, ports, official, port intel), CORS ok, run id matches |
| All five sources | ✅ `updated` (C-HARM, GIBS, CDFW ports, official, port intel); no port notes |
| CDPH page from a Linux GitHub runner | ✅ readable (the shipped TLS intermediate works there too) |
| Watcher fingerprints from a different network | ✅ all three match the transcription (fingerprints are stable) |
| Official geometry | ✅ no errors |
| C-HARM verification inside the run | ✅ 156/156 vs ERDDAP |

## 4. Production-equivalent frontend (website not deployed)

A local production build (`next build` + `next start`) reading the staging URL (`CW_DATA_BASE_URL`):

- `npm run test:published` (Playwright): **2/2 passed**. Forecast image, grids and values load cross-origin and match the pre-publish ERDDAP verification. Official notices load and show **Not verified** (transcription pending review). The Monterey port panel lists the related notices, its forecast median matches the published port summary, and the official overlay loads. No failed cross-origin requests.
- The same build against the current Pages data (M1-era): 1 passed, M2 test skipped by design. The app shows explicit **Unavailable** states rather than errors.
- Screenshots: [`m2-staging/01-staging-desktop-port.png`](m2-staging/01-staging-desktop-port.png), [`02-staging-mobile-port.png`](m2-staging/02-staging-mobile-port.png), [`03-staging-sources.png`](m2-staging/03-staging-sources.png), [`04-transition-m1-data.png`](m2-staging/04-transition-m1-data.png).

## 5. Tests at the end of the review

| Suite | Result |
|---|---|
| Pipeline offline | 90 passed, 1 deselected (live) |
| Web unit | 53 passed |
| Lint, typecheck, production build | clean |
| End-to-end (3 fixture scenarios, desktop + mobile) | 25 passed |
| CI on PR #3 (push + pull_request) | green; 0 Node 20 annotations |

## 6. Integration decision

All integration gates pass. Merging PR #3 switches the six-hourly production refresh to the M2 pipeline. GitHub Pages will then publish the official-notices registry (marked `pending_human_review`, shown as **Not verified**) and the port summaries as public data. The Next.js website stays undeployed.

**Before any public website launch:** a person reviews the official registry (`cwp review-official --reviewer NAME --confirm`), and a HAB scientist signs off (checklist 08).
