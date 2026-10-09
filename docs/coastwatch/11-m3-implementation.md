# 11 — Milestone 3: Bloom Intelligence, Fisheries Economics & Product Design

> Updated by the integration review ([12](12-m3-integration-review.md)): licensing evidence, NOAA reconciliation, statewide-total definition, fixes and merge results are recorded there.

Branch `feat/coastwatch-m3-bloom-economics` (pushed; **not merged, website not deployed, nothing published to production**). Built and verified 2026-10-08. Data published only to the staging channel (`coastwatch-data-staging`, never served by GitHub Pages).

**Not claimed:** the official notices remain an unreviewed AI transcription (`pending_human_review`, shown as **Not verified**); no HAB scientist has reviewed the new pages (shown as **Scientific review pending**); species tiers are an editorial grouping awaiting that review.

## 1. Source feasibility audit (verified 2026-10-08)

| Source | Content | Access | Decision |
|---|---|---|---|
| **CalHABMAP via SCCOOS ERDDAP** (`erddap.sccoos.org/erddap/tabledap/HABs-*`) | 17 shore stations; pDA/dDA/tDA (ng/mL), *Pseudo-nitzschia* seriata/delicatissima groups (cells/L), extracted chlorophyll (mg/m³), temperature; 2011→present, mostly weekly | Public tabledap CSV; licence "may be used and redistributed for free but is not intended for legal use" | **Ingested.** No detection limit or analytical method is published; 0.0 values occur; server intermittently answers 502 / "unknown datasetID" while reloading (retried). `time_coverage_end` metadata lags the data, so dates come from rows. |
| CeNCOOS / IOOS HAB portals | Same CalHABMAP stations re-served | — | Not duplicated (SCCOOS is the publishing server). |
| CDPH Marine Biotoxin monitoring (shellfish tissue, phytoplankton) | Weekly reports, ArcGIS phytoplankton layer | No stated licence or machine-readable tissue data | **Not ingested**; linked as an official source. |
| **NOAA Fisheries FOSS** (`apps-st.fisheries.noaa.gov/ods/foss/landings`) | California commercial landings by species and year (PacFIN-derived), statewide only; one "WITHHELD FOR CONFIDENTIALITY" row per year | Public JSON API, U.S. Government work | **Ingested, 2015–2024.** 2025 returns no rows yet (reported as unavailable). One listing ("OYSTER, KUMAMOTO **") duplicates "OYSTER, PACIFIC" to the cent in every year → counted once and disclosed. |
| **BLS CPI-U** (`CUUR0000SA0`) | Monthly index | Public API v1 (POST for year ranges, ≤10 years/request, no annual average; ~25 requests/day/IP) | **Ingested.** Annual averages computed from 12 monthly values; they equal BLS's published averages (e.g. 2023 304.702, 2024 313.689). October 2025 was never published ("lapse in appropriations"), so 2025 has 11 months and **2024 is the base year**. The flat-file server (`download.bls.gov`) refuses non-browser clients unless an email is sent in the User-Agent; not used. |
| **CDFW MFDE** (port-area landings, 2021+) | The only source of 2021–2025 port-level landings | Interactive dashboards only; CDFW asks users to consult before using the data; the former landings-report page now redirects to MFDE | **Not extracted.** Permission for automated extraction/republication is uncertain → port-level adapter disabled. |
| CALFISH (Dryad, CC0, doi:10.25349/D9M907) | Port × species landings 1941–2019 | Download behind a proof-of-work browser challenge; API download needs a bearer token | **Not downloaded** (not circumventing the challenge; a token needs an external account). Ends 2019 anyway. |
| PacFIN | Port-level fish tickets | Data-use terms not verified for republication | Not used. |

Evidence: the request/response checks above were run from a workstation; the recorded responses used by tests are in `pipeline/tests/fixtures/recorded/` (`scripts/record_m3_fixtures.py`).

## 2. Measured observations (pipeline)

`pipeline/coastwatch_pipeline/sources/observations.py` → artifact `observations-<sha>.json` (`ObservationDataset`, schema `schemas/v1/observations.schema.json`).

- **Schema.** Each station keeps its sample times once; each variable is a column aligned to them. Variables carry upstream name and long name, units verbatim, kind (toxin / cell abundance / pigment / physical), matrix (`seawater`), fraction (particulate / dissolved / total), method text, `detection_limit` (null — not published), detection-limit note, zero policy and a plausibility ceiling. Fractions, size groups and variables are never combined; tDA is never derived from pDA + dDA.
- **Missing vs zero vs censored.** NaN upstream → `null` = *not measured* (never 0). A reported 0 stays 0 with qualifier `reported_zero` and is always displayed as "reported 0 (not quantified)". Negatives → `null` + `rejected_negative`. Values above the ceiling are kept with `flag_high`.
- **QC per station.** Units row checked against expected units on every request (a change fails the station rather than mislabelling it); exact duplicates and future-dated rows dropped and counted; sampling position checked; negatives and flags counted.
- **Summaries.** Per variable: measured count, reported zeros, rejections, first/last date, last value and qualifier, days since last, measurements and median interval in the last 365 days, 365-day maximum. Recomputed independently in tests.
- **Station context.** Region by the official region bounds (nearest-port fallback), nearest CDFW port and distance, C-HARM nowcast median within 15 km for the last 180 days (separate field; never combined with measurements).
- **Failure isolation.** Each station is fetched with retries; a failing station keeps its previous data with original dates (`carried_forward`), a station never retrieved is `failed`; the source is `partial` and other stations publish normally.

Staging run, 2026-10-08: all 17 stations updated and C-HARM context for all 17. Weekly toxin sampling continues at Santa Cruz Wharf (pDA 49 times in the last year), Cal Poly Pier, Stearns Wharf, Santa Monica, Newport and Scripps. **Monterey Wharf has no pDA since 2022-08-31** and no *Pseudo-nitzschia* counts since 2025-08-27 (temperature only). Northern stations (Trinidad, Humboldt, Bodega, Tomales) have no samples since January–April 2026 and show as **Historical — not current**. Bodega, Tomales and Morro Bay report no domoic acid at all in this dataset.

## 3. Bloom Intelligence (`/bloom`)

- Official notices come first: verification badge (**Not verified**), the notices whose stated area may include waters near the station's port, "no notice ≠ open/safe", link to all notices.
- Station map (dots = sampling piers, shaded by recency of the last sample) and a station list grouped by region with **Monterey Bay first (flagship, default Santa Cruz Wharf)**; a native select on mobile.
- Station header with freshness and explicit notices for carried-forward or historical stations; a fixed sentence separating measurements from the model and seawater toxin from seafood.
- Latest value of each quantity with its own date and freshness, or "Not measured at this station since 2014".
- Charts on one shared time window (12 months / 3 years / since 2014) and a shared hover: each dot is one laboratory value; reported zeros and rejections sit in a separate "0*" lane under the axis; a strip of ticks shows every sampling visit (faint = sampled but not measured) — the **measurement-frequency and missing-data indicator**; a coverage line ("measured in 49 of 49 visits · typical interval 7 days · 10 reported 0"); "Not measured in this period. No measurement is not the same as no toxin" when empty; data table; keyboard reading with arrow keys. *Pseudo-nitzschia* panels share an axis; log scales are labelled.
- C-HARM section below a dashed divider: its own badge (Agency forecast), its own 0–100% axis, the same time axis (hatched before the 180-day history starts), variable switch, nearest-cell distance, "not comparable with the measurements above", "not a closure decision; a low value does not mean an area is safe".
- Methods: caveats, variable definition table (units, matrix/fraction, method, detection limit "not published", zero policy), per-station QC, processing steps, source links, licence.

## 4. Fisheries & Economic Exposure (`/fisheries`)

`sources/fisheries.py` → `fisheries-<sha>.json` (`FisheriesDataset`).

- **Terminology.** "Historical fisheries exposure: the reported value of past commercial landings of species that marine toxins can affect. It is not a prediction of losses…" Tests fail if "loss" appears un-negated anywhere in the artifact or rendered page.
- **Groups (tiers await scientific review).** Tier 1 — Dungeness crab, rock crabs, northern anchovy (commercial closures or take restrictions for domoic acid, current or past; rock crab and anchovy link the active official records). Tier 2 — California spiny lobster, Pacific sardine, bivalve shellfish. Each upstream row maps to at most one group by exact name; an ambiguous name raises.
- **Inflation.** Real = nominal × CPI(2024)/CPI(year), CPI-U annual averages from 12 published months; base year rule and the missing October 2025 value are shown on the page.
- **Suppression.** FOSS's withheld row is shown separately every year and never attributed to a species or group; it is included in the statewide total, as in NOAA's own state totals (changed in the integration review, see 12). Rows without a value are counted, never treated as $0.
- **No double counting.** Group sums reconcile exactly to the assigned source rows and the statewide total to all kept rows in every year (tested against the raw responses); the duplicate oyster listing (PacFIN code KSTR mapped to two FOSS names, confirmed upstream in 12) is excluded and listed. The statewide total reproduces NOAA's published California totals within 0.17%.
- **Port level.** `PortLevelStatus = unavailable` with reasons. A disclosure-safe aggregation (`aggregate_cells`) exists for when a port-level source is authorised: suppressed components are never estimated, aggregates containing them are labelled "at least", and complementary suppression blocks a total that would reveal a single withheld component. Tested only with clearly labelled synthetic cells; no real port data exist in the repository.
- **UI.** Definition box, port-level unavailable notice, tier and real/nominal filters, three summary tiles, total bar chart, per-group small multiples (own scales, tier basis, official links, upstream category names), values table with statewide total and withheld row, duplicate disclosure, deflator box, methods and provenance.

Staging values (2024, nominal): Dungeness $49.7M, spiny lobster $20.9M, bivalves $10.9M, rock crab $2.4M, sardine $0.67M, anchovy $0.63M; all California commercial landings $200.0M (NOAA state total, including the $10.8k withheld).

## 5. UI and design changes

- **Live Ocean Map rail:** the regulatory summary is now compact — brief "Not verified" line (reasons one click away), three one-line notice rows that expand to the agency's wording, "Show all 8 notices". On a 1280×720 laptop the port list now starts above the fold (screenshot 02); previously the rail was entirely advisories.
- **Legend on the map:** a floating card next to the rail with the layer switch (Forecast / Chlorophyll / Off), the current layer, lead and valid date, and the colour key; mobile shows the same card as a collapsible key under the region bar. The forecast panel keeps threshold text, run line and caveats.
- **Navigation:** Bloom Intelligence and Fisheries are links (short labels below 1280 px; scrollable on phones); My Coast stays **Upcoming**; no accounts or notifications.
- Reading pages scroll the document with a sticky header; charts resize to narrow screens; navy palette unchanged; new data colours: lavender for toxin measurements, light blue for cell counts, teal for chlorophyll (as before), magenta reserved for C-HARM, amber reserved for official notices.

Changes made from screenshot review: rounded chart coordinates (server/browser `Math.log10` differences caused a hydration mismatch); compact note instead of an empty chart for quantities a station never reports; charts no longer force horizontal scrolling on phones (grid items needed `min-width: 0`); region assignment by official region bounds (Inner Tomales Bay had been filed under Mendocino–Sonoma via its nearest port); fisheries freshness policy relaxed to the real publication lag (was "Stale" for normal 2024 data) and annual data labelled "data through 2024"; chart y-axis steps refined.

## 6. Architecture and compatibility

- Two new sources in the same pipeline (`calhabmap`, `foss_landings`); `ALL_SOURCES` now has seven. Each is isolated; fisheries refreshes at most every 7 days (annual data; keeps the BLS request count low) and otherwise reports `unchanged`.
- Two new content-addressed artifacts, referenced by two **optional** manifest fields (`observations_url`, `fisheries_url`); schema version stays 1 because the change is additive. Pruning and `check-published` cover the new files. Observations are written as compact JSON (≈650 kB, ≈100 kB gzipped).
- `RunContext.poster` added for the BLS POST API; fixtures record POSTs by URL + body hash.
- Web: `loadBloomData` / `loadFisheriesData` validate only the artifacts each page needs (Ajv, generated schemas); the bloom page sends station summaries plus one station's series to the browser and switches stations through the URL (`?station=`).
- **Compatibility tested with the data actually published by M2** (Pages run 37846066753, committed in `pipeline/tests/fixtures/compat/m2/`): M3 models validate it; an M3 pipeline run on top of an M2 output directory adds the new artifacts and keeps the M2 files for one run; the M3 web app on M2 data shows "published before measured observations / fisheries data were added" and the map works.

## 7. Tests

| Suite | Result |
|---|---|
| Pipeline offline (`uv run pytest`) | **123 passed**, 1 deselected (live). New: 17 observation tests (null vs zero against the recorded CSV, reported-zero qualifiers, independent summary recomputation, Monterey Wharf absence, fractions not combined, model context only where recorded, region assignment, adversarial CSV with negative/duplicate/future/implausible rows, unit change, missing column, carry-forward with original dates, retries, all-stations-failed with and without a previous artifact, caveat wording), 14 fisheries tests (CPI averages and base year, deflation, reconciliation/no double counting, duplicate listing, withheld separation, null ≠ 0, ambiguous names, port level unavailable, terminology, FOSS and CPI failures, weekly refresh, synthetic suppression/complementary suppression), 2 M2-compatibility tests |
| Web unit (`npm test`) | **67 passed** (14 new: schema validation incl. a corrupted artifact and the M2 manifest, observation semantics, fisheries totals, copy) |
| End-to-end (`npm run test:e2e`, 4 servers incl. published M2 data) | **42 passed** (17 new: bloom default/order/values, absence states, shared range, model unavailable, historical station, keyboard station selection and chart reading, path-like station id, fisheries tiles vs data and filters, M2 data unavailable states, failed-update dates, **axe: no serious/critical violations on map, bloom, fisheries, sources**, three mobile checks) |
| Lint, typecheck, production build | clean |
| CI on the branch | green (run 37870345283) |

## 8. Staging and production-equivalent checks

| Check | Result |
|---|---|
| Data workflow, channel `staging` ([37870352669](https://github.com/yashnil/habs-forecast/actions/runs/37870352669)) | build → publish → public-URL check green; production branch untouched; review issue skipped |
| `check-published` on `coastwatch-data-staging/v1` | **30/30 files**, CORS ok, run id matches; sources: C-HARM unchanged (same issue date), GIBS / ports / official / port intel / **CalHABMAP updated (17/17 stations)** / **FOSS updated** |
| Production build vs **GitHub Pages (M2 data)** — `npm run test:published` | **3/3 passed** (M3 pages show the unavailable states) |
| Production build vs **staging (M3 data)** | **3/3 passed**; on the first invocation one test passed only on its retry (the first request after server start was slow); two further runs with retries disabled passed 3/3 |

## 9. Screenshots (production build, staging data unless noted)

[`m3/01-map-desktop.png`](m3/01-map-desktop.png) · [`02-map-laptop-1280.png`](m3/02-map-laptop-1280.png) · [`03-map-port-monterey.png`](m3/03-map-port-monterey.png) · [`04-bloom-santa-cruz.png`](m3/04-bloom-santa-cruz.png) · [`05-bloom-monterey-wharf.png`](m3/05-bloom-monterey-wharf.png) · [`06-fisheries.png`](m3/06-fisheries.png) · [`07-bloom-mobile.png`](m3/07-bloom-mobile.png) · [`08-fisheries-mobile.png`](m3/08-fisheries-mobile.png) · [`09-map-mobile.png`](m3/09-map-mobile.png) · [`10-bloom-on-published-m2-data.png`](m3/10-bloom-on-published-m2-data.png) (Pages data) · [`11-bloom-historical-station.png`](m3/11-bloom-historical-station.png) (fixture, stale scenario) · [`12-sources.png`](m3/12-sources.png)

## 10. Decisions that need you, limitations, and how to reproduce

**Needs explicit approval (not done):**
1. **Merge** `feat/coastwatch-m3-bloom-economics` into `main`. Merging switches the six-hourly production refresh to publish observations and statewide landings on GitHub Pages (public data). The website stays undeployed either way.
2. **CDFW MFDE permission** for port-level 2021–2025 landings (contact CDFW Marine Region before any extraction). Until then port-level stays unavailable.
3. Optional **CALFISH file** (manual browser download of `CDFW_1941_2019_landings_by_port_species.xlsx`, SHA-256 `ead63e8b…0831`) for pre-2020 port context; an adapter would be written against the real columns.
4. **Human review** of the official registry (`cwp review-official --reviewer NAME --confirm`) and **HAB scientist review** of `/bloom`, the tiers and the C-HARM context (extend checklist 08).

**Limitations.** No detection limits or analytical methods are published for CalHABMAP, so "reported 0" cannot be turned into a numeric limit. Stations are points; their values describe one pier. Laboratory results arrive days to weeks late. C-HARM station history covers 180 days. Landings are statewide ex-vessel revenue only (no ports, processing, tourism). BLS v1 allows about 25 requests per day per IP; a refused request fails the fisheries source and keeps the previous artifact. Tier assignments are editorial.

**Reproduce:**
```bash
cd pipeline && uv sync
uv run cwp run --only calhabmap,foss_landings,cdfw_ports   # live, into ../coastwatch-web/public/data/v1
uv run pytest                                              # offline, recorded fixtures
uv run python scripts/record_m3_fixtures.py                # re-record M3 fixtures (network)
uv run cwp check-published --base https://raw.githubusercontent.com/yashnil/habs-forecast/coastwatch-data-staging/v1
gh workflow run coastwatch-data.yml --ref feat/coastwatch-m3-bloom-economics -f channel=staging
cd ../coastwatch-web && npm ci
npm run lint && npm run typecheck && npm test && npm run test:e2e
npm run build && PUBLISHED_DATA_URL=https://yashnil.github.io/habs-forecast/v1 npm run test:published
PUBLISHED_DATA_URL=https://raw.githubusercontent.com/yashnil/habs-forecast/coastwatch-data-staging/v1 npm run test:published
npm run dev   # http://localhost:3000/bloom?station=HABs-SantaCruzWharf · /fisheries
```
