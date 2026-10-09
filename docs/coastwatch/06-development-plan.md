# 06 — Development plan

> **Status (2026-10-08).** "Milestone 1: Honest Baseline + One Live C-HARM Forecast" — which combines M0 below with the MapLibre migration, design foundation and scheduled refresh from M1/M3 — is implemented on `feat/coastwatch-m1-baseline`. See [`07-m1-implementation.md`](07-m1-implementation.md) for what is built, test evidence and open issues. Later status: M2 (official notices, port intelligence, map redesign) is merged — [`09`](09-m2-implementation.md), [`10`](10-m2-integration-review.md); the delivered "M3: Bloom Intelligence, Fisheries Economics & Product Design" (measured CalHABMAP observations, statewide historical fisheries exposure, UI refinement) is on `feat/coastwatch-m3-bloom-economics` awaiting approval — [`11`](11-m3-implementation.md). Not yet built from M1/M2: VIIRS ERDDAP ingestion, MPA/RAMP geometry, NWS, curated official records, port pages.

## 1. Prioritized MVP feature list

Priority order = build order. P0 items block public launch.

| # | Feature | Pri | Depends on |
|---|---|---|---|
| 1 | Remove synthetic/unsourced content from the public build; fix dead CDPH link, GIBS newest-date bug, dead legend URLs | P0 | — |
| 2 | Schemas (Pydantic → JSON Schema → TS) and provenance envelope | P0 | — |
| 3 | Pipeline: C-HARM v3.1 (4 leads, 3 variables) → reprojected images + value grids + validation | P0 | 2 |
| 4 | Pipeline: VIIRS N20/N21 primary with S-NPP fallback | P0 | 2 |
| 5 | Pipeline: MPA (ds582), RAMP zones (ds3120), CDPH county polygons | P0 | 2 |
| 6 | Curated `ports.yaml` from CDFW port / port-area tables with reviewed coordinates, PacFIN group, NWS zone | P0 | — |
| 7 | Curated `advisories.yaml` / `closures.yaml` + regulatory watchers that open GitHub issues | P0 | 2 |
| 8 | Official-status engine (hierarchy rules R1–R4) + Official Status card | P0 | 6, 7 |
| 9 | Manifest, `status.json`, object storage publish, last-known-good | P0 | 3–5 |
| 10 | Map rebuild on layer registry (MapLibre), legends, product-class badges, inspector | P0 | 9 |
| 11 | Port pages (`/ports/[portId]`) with the six cards | P0 | 8, 9, 13 |
| 12 | `/data` freshness page; status bar freshness dot | P0 | 9 |
| 13 | Historical exposure metric (MFDE, tiers, CPI, suppression handling) | P0 | 6 |
| 14 | NWS marine zone forecast + hazards on port page | P1 | 6 |
| 15 | `/bloom` explainer, `/about`, disclaimers, hotlines | P0 | — |
| 16 | Design system (tokens, light/dark, typography, components) | P0 | — |
| 17 | Print-friendly port brief; favorite port | P1 | 11 |
| 18 | Accessibility pass (WCAG 2.2 AA) and performance budgets | P0 | 10, 11 |

## 2. Milestones

Durations assume one developer working part-time with AI assistance; treat them as relative sizing.

### M0 — Truthful baseline + C-HARM vertical slice (≈ 1–1.5 weeks) — **first implementation milestone**

Goal: nothing false is shown, and one official forecast flows end-to-end through the new architecture.

Scope:
1. Remove `snapshot.json`, `overlay.png`, "Bundled notes", `fisheries_context.json` copy, `recommendations.ts` tier functions, and `sync-data.mjs` dependency on `dashboard/`. Remove the README claim that the app renders the research overlay.
2. Hotfixes: CDPH link → `…/EMB/Shellfish/Marine-Biotoxin-Monitoring-Program.aspx`; GIBS date = `Default − 1` with tile probe; legend URLs from capabilities; show one chlorophyll product at a time.
3. Scaffold `pipeline/` (uv, ruff, pytest), `models.py` with `Provenance`, `LayerArtifact`, `SourceStatus`, `Manifest`; JSON Schema export; TS type generation into `coastwatch-web/src/generated/`.
4. `sources/charm.py`: probe latest time for all four leads, fetch CA subset, validate, reproject to EPSG:3857, render fixed-palette PNG + uint16 value grid, write `LayerArtifact`s and a local `latest.json` (storage upload can be local filesystem/`public/` in M0).
5. Web: layer registry with one official-forecast layer group (C-HARM PN / pDA / cDA, lead selector), legend with threshold text, product-class badge, provenance pill with valid time and computed freshness.
6. Tests: pipeline unit tests on a recorded ERDDAP fixture; reprojection accuracy test; schema contract test; vitest for freshness rules; one Playwright smoke test.
7. Decisions recorded: Next 16 upgrade (yes/no), MapLibre basemap choice (spike), storage provider.

Exit criteria: app builds with zero synthetic artifacts; C-HARM layer shows correct valid date/lead and matches ERDDAP values at 5 spot-check pixels; stale state demonstrably appears when the fixture is aged; Streamlit CI removed or disabled.

### M1 — Data spine (≈ 2 weeks)

VIIRS primary/fallback, MPA/RAMP/county geometry, NWS zones, `status.json`, scheduled `pipeline-daily` in Actions, upload to object storage, last-known-good pointer, per-source error isolation, GitHub-issue alerts on failure.

Exit: 7 consecutive days of unattended runs with correct freshness reporting, including at least one observed upstream failure handled gracefully (or simulated).

### M2 — Official status (≈ 1.5 weeks)

`ports.yaml` (all CDFW marine port areas and principal ports, reviewed coordinates), `advisories.yaml`/`closures.yaml` seeded from the 2026-10-08 status in `02-data-sources.md` and re-verified by a human, watchers (`watch-regulatory`) opening issues on diffs, hierarchy engine + Official Status card + hotlines + "not verified" state. Curator runbook in `docs/coastwatch/runbooks/curation.md`.

Exit: a simulated new CDPH release produces an issue within 6 h; merging a record updates every affected port page on the next run; tests prove R1–R4.

### M3 — Map and port experience (≈ 2–3 weeks)

Design system, MapLibre migration, map rebuild (registry, official overlays on top, inspector reading value grids, cloud-gap hatch), port index + port pages (cards 1–4, 6), `/data`, `/about`, `/bloom` explainer, mobile bottom sheet, light/dark, print CSS.

Exit: usability sessions (≥ 5 people from fishing/port community or Sea Grant extension) can answer the core question unaided; Lighthouse ≥ 90 performance/accessibility on port pages.

### M4 — Economics, hardening, launch (≈ 2 weeks)

Exposure metric and card (+ species sensitivity table with citations), CDFW acceptable-use confirmation for MFDE, outreach to SCCOOS/CoastWatch about C-HARM attribution and skill statement, accessibility and copy-safety review, uptime monitor on `status.json` age, launch checklist.

Exit: all §5 criteria green; written sign-off from at least one domain reviewer (HAB scientist or extension agent) on copy and hierarchy.

### Post-MVP

| Phase | Scope |
|---|---|
| P2 | Timeline (14–90 days); C-HARM coastal strip chart and regional trends; CalHABMAP + CDPH phytoplankton stations; VIIRS anomaly vs fixed 2015–2024 climatology; email alerts on official status change per port; Spanish; service-worker offline cache; statewide port comparison; closure history timeline |
| P3 | Experimental model: retrain on VIIRS-era inputs (drop/replace `nflh`), persist norm stats/channel order, ensembles/MC-dropout, held-out ≥ 2022 evaluation vs persistence and vs C-HARM chl fields, model card, weekly batch inference to `experimental/`; research hindcast showcase page; accounts/logbook only after privacy design |

## 3. Component and directory architecture

See `04-architecture.md` §3 (repository layout) and §4 (schemas).

## 4. Dependency / data-source map

See `02-data-sources.md` §3.

## 5. Testing and validation criteria

| Area | Test | Gate |
|---|---|---|
| Schema contracts | Every published artifact validates against JSON Schema in the pipeline *and* on load in the app; fixtures for each `LayerArtifact` kind | CI |
| Pipeline correctness | Recorded upstream fixtures per source; value round-trip (grid decode = source value within quantization error) | CI |
| Reprojection | Known coastline/harbor points land within 1 pixel of expected location after warp; regression test against the ~18 km corner-stretch bug | CI |
| Spot-check vs upstream | Daily job compares 5 fixed pixels per C-HARM lead and VIIRS against direct ERDDAP queries | Alert on mismatch |
| Freshness | Unit tests for SLA states per source; e2e with aged fixtures shows `stale`/`failed` UI | CI |
| Safety rules R1–R4 | Unit tests: a port with an active official record never renders a forecast/observation headline above it; missing/stale regulatory data renders "Status not verified" | CI |
| R5–R8 | Copy lint: forbidden terms in UI strings (`safe`, `clear to fish`, `go fish`, `all clear`, `no risk`); every chlorophyll legend includes the biomass disclaimer; no code path computes percentiles of the rendered field for color/tier (code review + grep check) | CI + review |
| R12–R16 | Experimental layers absent unless a model card artifact exists; experimental layers off by default and visually distinct (visual test) | CI |
| R17–R19 | Exposure card never displays suppressed values; shows "lower bound" when any suppression; no field named or labeled "loss" | CI |
| Provenance | Every number on port pages has a matching `Provenance` (DOM test walks data attributes) | CI |
| Links | Link checker over all official URLs in content and curated records (weekly + on PR) | CI / scheduled |
| Accessibility | axe-core on all routes; keyboard navigation of map controls; contrast tokens validated | CI |
| Performance | Lighthouse CI budgets (LCP < 2.5 s 4G on port page, JS < 250 KB gz) | CI |
| Visual | Playwright screenshots of map + port page, light/dark, mobile/desktop | CI (manual approve) |
| Scientific review | Domain reviewer signs off copy, thresholds, area-fraction method | Pre-launch |
| Usability | ≥ 5 target users, task success on core questions | Pre-launch |

## 6. Deployment strategy

1. **Environments:** `preview` (every PR, fixture data), `staging` (main branch, live pipeline writing to `/staging/v1`), `production` (tagged release, `/v1`).
2. **Pipeline:** GitHub Actions scheduled workflows; concurrency group per job; artifacts uploaded with content hashes; `latest.json` swapped last (atomic pointer).
3. **Web:** Vercel (or Cloudflare Pages) builds from `coastwatch-web/`; ISR reads the production manifest URL from env.
4. **Secrets:** storage write credentials only (MVP); Copernicus credentials only in P3.
5. **Monitoring:** scheduled check that `status.json` is < 12 h old and that each source is within SLA → GitHub issue/email; uptime monitor on the site.
6. **Rollback:** repoint `latest.json` to the previous manifest; web rollback via host's instant rollback.
7. **Launch:** soft launch to a pilot port community and Sea Grant extension contacts, gather feedback 2–4 weeks, then public announcement.

## 7. Biggest risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Regulatory status curation lapses → stale or wrong status shown | Medium | **Severe** (trust, safety) | "Not verified" default; SLA on `last_verified_at`; watchers; ≥ 2 curators; hotlines always visible |
| Users read C-HARM probability or chlorophyll as "safe to harvest/eat" | Medium | Severe | R3–R5 copy rules, tests, domain review, usability testing of wording |
| C-HARM gaps/outage (multi-week gaps seen in 2024) | High | Medium | Freshness states; show last valid run clearly; no interpolation |
| C-HARM v3.1 skill undocumented; known false positives | High | Medium | Display producer caveats; cite v1 skill paper honestly; contact producers |
| Upstream schema/URL changes (ERDDAP IDs, ArcGIS services, MFDE backend) | High over a year | Medium | Probes + validation gates; raw snapshots; fallback datasets; source register re-verification per release |
| MFDE undocumented API changes or use not sanctioned | Medium | Medium | Snapshot annually; confirm with CDFW; PacFIN CSV fallback |
| Exposure metric misread as loss forecast | Medium | Medium | Separate panel, wording, lower-bound flag, no multiplication with forecasts |
| Research model fails to beat persistence on operational inputs (MODIS → VIIRS shift, `nflh` gap) | High | Low for MVP (not shipped) | Gate P3 on held-out skill; publish negative results honestly |
| Research README's 8.3% claim propagates into public copy | Medium | Medium (credibility) | Use only same-pipeline numbers; correct the README in a separate research PR |
| Basemap/map library migration costs more than expected | Low | Low | Spike in M0; Mapbox remains a fallback |
| Liability perception | Low–Medium | High | Clear disclaimers, no recommendations, official-first hierarchy, legal review if partnering with an agency |

## 8. Work that should NOT be done yet

- **No "where to fish" or "what to catch" recommendations, species-opportunity layers, or trip scores.** No defensible public data; violates the core promise's integrity.
- **No modeled expected revenue loss** until a closure-to-revenue model is developed and reviewed (R17).
- **No experimental ML layer on the live map** until the P3 preconditions (§2) are met; do not deploy current checkpoints.
- **No accounts, logbooks, or personal data collection** before a privacy design.
- **No push/SMS alerts** before official-status curation has proven reliable for a full season.
- **No automated parsing of closure/advisory status into the UI** without human review.
- **No scraping of OEHHA** (bot-protected) or of PacFIN APEX sessions.
- **No database, tile server, or Kubernetes**; no paid APIs.
- **No changes to research code or paper results** as part of this product work (README corrections go in a separate research PR).
- **No MODIS-Aqua-dependent features**; no blending of different sensors under one legend.
- **No custom HAB "risk index"** combining chlorophyll, C-HARM, and closures into one score.
