# Coastwatch — planning documents

Planning and architecture for turning `coastwatch-web/` and the HABs research into a public coastal-intelligence platform for California fishing communities. Planning written 2026-10-08; Milestone 1 implemented the same day (see 07).

| Doc | Contents |
|---|---|
| [01 — Codebase audit](01-codebase-audit.md) | Reusable assets, every synthetic/unsourced dataset, misleading logic, model artifacts and inference requirements, tech debt |
| [02 — Data sources](02-data-sources.md) | Verified source register (endpoints, resolution, coverage, variables, cadence, license, constraints, last observation), ingest mode and freshness SLA per source, dependency map |
| [03 — Product spec](03-product-spec.md) | Users, information architecture, the four experiences, MVP vs P2/P3, visual design direction |
| [04 — Architecture](04-architecture.md) | Design decisions, system diagram, repository layout, normalized schemas, pipeline, storage, failure handling, model-inference integration, cost |
| [05 — Science & safety](05-science-and-safety.md) | Non-negotiable rules R1–R26 (hierarchy, chlorophyll, C-HARM, experimental models, economics, provenance) |
| [06 — Development plan](06-development-plan.md) | Prioritized MVP list, milestones M0–M4 + P2/P3, testing & validation, deployment, risks, work not to do yet |
| [07 — Milestone 1 implementation](07-m1-implementation.md) | What is built, verification results, tests, known issues, how to run |
| [08 — Scientific review checklist](08-scientific-review-checklist.md) | 20-minute checklist for a HAB scientist reviewing the C-HARM layer |
| [09 — Milestone 2 implementation](09-m2-implementation.md) | Official notices (human-reviewed), port intelligence, redesigned map; tests, screenshots, limits |
| [10 — M2 integration review](10-m2-integration-review.md) | Schema-compatibility fix, Node 24 actions, staging run, production-equivalent checks, merge decision |
| [11 — Milestone 3 implementation](11-m3-implementation.md) | Source audit, CalHABMAP observations, Bloom Intelligence, statewide fisheries exposure (FOSS + CPI-U), UI refinement, compatibility with published M2 data, staging results, decisions awaiting approval |
| [12 — M3 integration review](12-m3-integration-review.md) | Licence evidence per dataset, NOAA reconciliation (KSTR duplicate, FUS Table 4), deflator check, fixes, test and publication evidence, merge results |
| [13 — Redesign P0: foundations](13-redesign-p0.md) | Design tokens (paper + dark sea), self-hosted type, navy shell with the official pill and drawer, mobile tab bar with Notices, status primitives; tests and screenshots |
| [14 — P1: Ocean Map](14-p1-ocean-map.md) | Sentinel-3 OLCI 300 m satellite chlorophyll (VIIRS fallback), banded C-HARM, layer groups, inspector, currents contract and WCOFS PoC; live verification, staging, screenshots |
| [drafts/cdfw-landings-data-request.md](drafts/cdfw-landings-data-request.md) | **Draft, not sent:** request to CDFW for disclosure-safe port-area landings and display permission |
| [redesign-p0/](redesign-p0/) | Screenshots of redesign phase P0 (published data, 2026-10-09) |
| [p1/](p1/) | P1 screenshots (staging data, 2026-10-09), currents contract and evidence |
| [15 — P2: currents, multi-sensor, health](15-p2-currents-multisensor.md) | P1 production publication; HF-radar observed currents (hourly, 24 h mean, arrows/flow); multi-sensor satellite view with coverage and agreement; WCOFS evaluation; pipeline health alerts; staging, screenshots |
| [p2/](p2/) | P2 screenshots (staging data, 2026-10-10), WCOFS evaluation, evidence |
| [17 — Status, design audit, roadmap](17-roadmap-and-design-audit.md) | Demo protection, regulatory integration (#16), P3 review (#15), design-reset gap audit (Ocean Map, Bloom, Fisheries), prioritized plan M4–M9, next milestone M5 |
| [m3/](m3/) | Screenshots of the M3 app (staging data, 2026-10-08) |
| [m2/](m2/) | Screenshots of the M2 app (live data, 2026-10-08) |
| [m1/](m1/) | Screenshots of the running app (live data, 2026-10-08) |
| [evidence/](evidence/) | Raw verification logs from live requests (satellite, C-HARM, regulatory, economics) and the M1 C-HARM point verification |

## Key decisions

1. **Official first, structurally.** Official regulatory → official forecast → observation → experimental → history. Enforced by separate namespaces, schemas, UI badges, and tests — not just disclaimers.
2. **C-HARM v3.1 is the bloom/toxin-risk layer** (`wvcharmV3_*day`, live today). Older v1/v2 datasets are frozen; producer pages still link to them.
3. **Regulatory status is human-curated.** No agency publishes closure/advisory status as data. Scrapers raise alarms; curators update PR-reviewed YAML; stale curation shows "Status not verified."
4. **Static-first pipeline.** Scheduled Python jobs precompute validated, reprojected artifacts to object storage; the Next.js app reads a manifest; freshness is computed in the browser. No database, tile server, or paid APIs in the MVP.
5. **Economics = historical exposure only** of HAB-sensitive species (tiered), inflation-adjusted, suppression-aware. No modeled loss. *As built in M3:* statewide NOAA FOSS landings with BLS CPI-U; port-level values (CDFW MFDE) stay unavailable until CDFW permits extraction (see 11).
6. **The research model is not deployable yet** (data freeze absent, normalization stats not persisted, ambiguous checkpoints, MODIS-Aqua/`nflh` input dependency, ~10% skill vs persistence). It enters later as an explicitly experimental layer after retraining and held-out validation.

## First implementation milestone

**M0 — Truthful baseline + C-HARM vertical slice** (see `06` §2): remove synthetic content and broken links/tiles from the live app, scaffold the pipeline and shared schemas, and ship C-HARM through the new architecture end-to-end with provenance and freshness.
