# 09 — Milestone 2: official advisories, port intelligence, Live Ocean Map

Branch `feat/coastwatch-m2-ocean-intelligence` (pushed; **not merged, website not deployed**). Built and verified 2026-10-08.

## 0. M1 handoff (Part A)

| Step | Result |
|---|---|
| PR #1 mergeable, checks green | ✅ CLEAN / MERGEABLE; CI pipeline + web passed on push and PR |
| Merged into `main` with a merge commit (history preserved) | ✅ `4cf841b` "Merge PR #1 …" |
| Six-hourly data workflow on `main` | ✅ `CoastWatch data refresh` registered with `cron: 17 */6 * * *` and manual dispatch |
| First run on `main` | ✅ manual dispatch [37831194499](https://github.com/yashnil/habs-forecast/actions/runs/37831194499): build → publish → public-URL check all green; `check-published` independently re-run from a workstation: 26/26 files, run id matches |
| GitHub Pages hosting workflow | ✅ `coastwatch-pages.yml` merged to `main` via PR #2 (`d625feb`). Deploys only `v1/` artifacts from the data branch after each successful refresh, then runs `check-published` against the Pages URL. Dispatch run [37831358212](https://github.com/yashnil/habs-forecast/actions/runs/37831358212) skipped with a warning because Pages is not enabled |
| Hosted data verified | ⏸ **Needs one setting:** Settings → Pages → Build and deployment → Source: **GitHub Actions**. Then re-run "CoastWatch data hosting (GitHub Pages)"; its `check` job validates every file at `https://yashnil.github.io/habs-forecast/v1/`. Until then data is served (and verified) from `raw.githubusercontent.com/.../coastwatch-data/v1` |

## 1. What works

**Official closures and advisories (highest priority).**
- `data/curated/official_notices.json`: 8 active notices (CDPH quarantine and advisories, CDFW closure and take restriction) and 4 quoted official statements, each with the agency's verbatim wording, dates (or an explicit note when a date is not published), species, toxin, area as worded, how it is drawn, documented uncertainties and source links.
- Human-reviewed workflow: people edit the registry by pull request; `cwp review-official --reviewer NAME --confirm` records a review and fingerprints the watched official pages. The pipeline watcher (`cwp watch-official`, also in the data workflow) re-fetches CDPH's release list and the two CDFW status pages each run and reports changes or new releases. **It never creates, edits or lifts a record.** When something changes, the data workflow opens or updates a GitHub issue asking for review.
- Verification is decided in the browser: **Verified** only if a person reviewed within 3 days, the watched pages are unchanged and readable, the watcher ran within 2 days and no records conflict; **Review ageing** at 4–7 days; otherwise **Not verified** with every reason listed. The committed registry is an AI transcription and is marked `pending_human_review`, so the app currently shows **Not verified** everywhere, by design.
- Official notices appear first in the rail, the port panel and the point inspector. "No notice listed here does not mean an area is open or that seafood is safe" appears wherever notices do. An active notice past its expected end date is flagged as unconfirmed, never shown as lifted.
- Geometry: CDPH county polygons (Monterey, Humboldt) and the Northern Channel Islands special-advisory polygon from CDPH's ArcGIS layers; latitude-defined notices are drawn as their official north/south limits from the coast out 30 km (coastline from C-HARM's land/sea mask), labelled with the coordinate. Statewide notices are not drawn.

**Port intelligence (22 CDFW ports).** Selecting a port (map marker, region list, or `?port=`) shows, in order: identity and county; official notices whose stated area may include nearby waters, with the reason (statewide, same county, port within the official latitudes, official area within 60 km); C-HARM statistics within 15 km for each lead (median, range, cell count, nearest cell distance, nearshore-masking note); the last 30 days of nowcasts (real ERDDAP history, missing runs left blank); NOAA VIIRS 8-day chlorophyll (latest composite, clear-pixel share, 60-day chart); methods, caveats and sources.

**Live Ocean Map.** Region navigation (All California, North Coast, Mendocino–Sonoma, San Francisco & Farallones, Monterey Bay default, Central Coast, Southern California) with a per-region port list showing each port's near-port forecast median and notice count; agency forecast and satellite chlorophyll layers (one raster at a time) with lead selection and fixed legends; official notice overlay with legend; click inspector (notices at the point first, then forecast values); URL state for sharing; freshness and verification indicators; Data & sources page shows verification for the official row instead of a generic "current" badge.

**Design.** Solid navy/charcoal panels (no glassmorphism), restrained cyan accent for navigation, a reserved amber family for official notices, magenta probability ramp for C-HARM, teal for chlorophyll; single-series charts with axes, units, dates, gaps and hover; compact disclosure cards; mobile slide-up sheet (peek / half / full, tap or drag); map dominates the desktop layout with a 372 px rail and a 424 px detail panel.

## 2. Regulatory sources and the review process

| Source | Used for | Access (verified 2026-10-08) |
|---|---|---|
| CDPH Shellfish & Seafood Advisories list | release IDs/titles (watcher); SN26-008, -018, -019 text | `https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx` — server omits its TLS intermediate; the pipeline ships the public Sectigo OV R36 intermediate (`pipeline/coastwatch_pipeline/certs/`, SHA-256 documented) and keeps verification on |
| CDFW Health Advisories and Closures | closures/restrictions; quoted statements | `https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories` (HTML comments stripped before hashing) |
| CDFW Whale Safe Fisheries | Dungeness 2026-27 season status statement | `https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries` |
| CDPH ArcGIS (county polygons, NCI special advisory) | official geometry | `services2.arcgis.com/wi1yEacfYjH5viqb/...` |
| OEHHA memo PDF, CDFW declaration PDF | provenance links | linked, not scraped |

**Review workflow for a person:**
1. When the data workflow opens "Official notices: sources changed since last review", or at least every 3 days during active events: open each source linked from `data/curated/official_notices.json`.
2. Edit records by pull request: add, update, or set `status: "lifted"` with `lifted_date` only when an agency notice says so. Quote the agency verbatim in `official_text`.
3. Run `cd pipeline && uv run cwp review-official --reviewer "Your Name" --confirm`; commit the updated `review` block in the same PR. CI validates the registry (adversarial rules) on every PR.

## 3. Files changed and commits

| Commit | Content |
|---|---|
| `67f2072` | Pipeline: official registry, validation, watcher, review CLI, geometry, port intelligence, schemas, recorded fixtures, data-workflow review issue |
| `32213e9` | Web: official notices UI and verification, port panel, charts, region navigation, overlays, inspector, mobile sheet, design system, react-map-gl 8.1.3 |
| `408b7cc` | Web tests: unit + end-to-end for M2 |
| `b155bee` | Latitude-defined notices drawn as official limits instead of shaded areas |
| (this commit) | M2 report, screenshots, small UI fix |

Merged to `main` separately as part of the M1 handoff: PR #1 (`4cf841b`) and PR #2 Pages workflow (`d625feb`).

## 4. Data coverage (live run 2026-10-08T19:56Z) and limitations

- C-HARM run issued 2026-10-08 (inferred), all four leads; all 22 ports have forecast cells within 15 km; nowcast history 27 of the last 30 days (3 days without a published run upstream).
- VIIRS 8-day chlorophyll available for all 22 ports (latest composite centred 2026-10-01).
- Official: 8 active records; geometry for 2 counties, the Channel Islands area and 3 latitude-limit pairs; 0 geometry errors; watcher fingerprints match the transcription.

**Limitations.** Official records are an unreviewed transcription until a person runs the review. The watcher detects page changes, not their meaning. County polygons are land outlines (as on CDPH's map); latitude-defined notices state no offshore limit. Port statistics describe forecast cells within 15 km, not the dock or fishing grounds; C-HARM masks toxin probabilities near shore. VIIRS composites overlap in time (daily steps of 8-day windows) and cloud gaps are common. No historical time slider for map imagery (only the current run and port histories). GitHub Pages hosting awaits one setting.

## 5. Tests and validation

| Suite | Result |
|---|---|
| Pipeline offline (`uv run pytest`) | **89 passed**, 1 deselected (live) — 36 new: registry adversarial cases (contradictory, lifted without date, missing dates, ambiguous/invalid bands, unknown county, unsafe wording, duplicates), conflicts, watcher changed/new release/blocked/short page, review CLI guard, limit-line geometry, missing geometry, port relations, statistics recomputed independently, history cross-checked against the recorded ERDDAP file, no-data ports |
| Live C-HARM verification | 156/156 against ERDDAP; `check-published` 28/28 files (incl. official and port intel) |
| Web unit (`npm test`) | **52 passed** — 12 new for verification states, flags and notices-at-point |
| Lint, typecheck, production build | clean |
| End-to-end (`npm run test:e2e`, desktop + mobile, 3 data scenarios) | **25 passed** — 10 new: notices listed and flagged, verbatim text, geometry kinds, stale-dataset reasons, default region, port panel order and values vs pipeline, unavailable states, region navigation, inspector, mobile sheet; every test scans rendered text for un-negated "open"/"safe" |

## 6. Screenshots (production build, live data)

[`m2/01-desktop-monterey-bay.png`](m2/01-desktop-monterey-bay.png) · [`02-desktop-port-monterey.png`](m2/02-desktop-port-monterey.png) · [`03-desktop-port-charts.png`](m2/03-desktop-port-charts.png) · [`04-desktop-statewide.png`](m2/04-desktop-statewide.png) · [`05-desktop-point-inspector.png`](m2/05-desktop-point-inspector.png) · [`06-desktop-crescent-city.png`](m2/06-desktop-crescent-city.png) · [`07-mobile-default.png`](m2/07-mobile-default.png) · [`08-mobile-port.png`](m2/08-mobile-port.png) · [`09-data-and-sources.png`](m2/09-data-and-sources.png)

Design changes made from screenshot review: replaced shaded latitude bands (spilled over land at 3 km) with official limit lines; compact notice rows with critical flags always visible; brief verification note in the port panel instead of repeating the rail; SVG disclosure chevrons; mobile camera padding for the sheet; map kept mounted across layout changes; scale bar moved out from under the rail; removed a double frame around the inspector; forecast opacity lowered so the coastline stays dominant.

## 7. Remaining scientific and design issues

- **Human review of the official registry** (blocks showing "Verified").
- Scientific reviewer sign-off on the C-HARM copy (checklist 08) and on the port statistics (15 km radius, medians).
- The rail is long on laptops: official notices (by design first) push the region ports and layers below the fold.
- The C-HARM raster's 3 km cells look blocky at Monterey Bay zoom; resampling would misrepresent resolution, so it is left as is.
- NASA GIBS legend SVGs are large (~600 KB) and drawn for light backgrounds.
- No automated accessibility audit yet; map inspection is pointer-only (lists provide keyboard access to ports).

## 8. Run and test

```bash
cd pipeline && uv sync
uv run cwp run                         # live data -> ../coastwatch-web/public/data/v1
uv run cwp verify-charm                # C-HARM vs ERDDAP at 13 points
uv run cwp watch-official              # official pages changed since last review? (read-only)
uv run cwp review-official --reviewer "Name" --confirm   # after checking every record
uv run pytest
cd ../coastwatch-web && npm install
npm run dev                            # http://localhost:3000  (?region=monterey_bay&port=550)
npm run lint && npm run typecheck && npm test && npm run test:e2e
```

## 9. Recommendations for M3

1. **Human review cadence** for official notices (named reviewers, rota during crab season) and a reviewer notification channel beyond GitHub issues.
2. Enable GitHub Pages hosting and point the app at it; then deploy the web app to a staging URL for user testing with Monterey Bay fishermen and port staff.
3. Add CalHABMAP shore-station observations (pDA, Pseudo-nitzschia counts) to port panels — the only measured toxin data near ports.
4. Historical economic exposure by CDFW port area (MFDE), per the M3 plan, kept separate from forecasts.
5. Map time slider for the last 14 C-HARM nowcasts (requires retaining images).
6. Accessibility pass (axe, keyboard map inspection, screen-reader summaries) and Spanish copy.
