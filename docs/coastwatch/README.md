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
| [m2/](m2/) | Screenshots of the M2 app (live data, 2026-10-08) |
| [m1/](m1/) | Screenshots of the running app (live data, 2026-10-08) |
| [evidence/](evidence/) | Raw verification logs from live requests (satellite, C-HARM, regulatory, economics) and the M1 C-HARM point verification |

## Key decisions

1. **Official first, structurally.** Official regulatory → official forecast → observation → experimental → history. Enforced by separate namespaces, schemas, UI badges, and tests — not just disclaimers.
2. **C-HARM v3.1 is the bloom/toxin-risk layer** (`wvcharmV3_*day`, live today). Older v1/v2 datasets are frozen; producer pages still link to them.
3. **Regulatory status is human-curated.** No agency publishes closure/advisory status as data. Scrapers raise alarms; curators update PR-reviewed YAML; stale curation shows "Status not verified."
4. **Static-first pipeline.** Scheduled Python jobs precompute validated, reprojected artifacts to object storage; the Next.js app reads a manifest; freshness is computed in the browser. No database, tile server, or paid APIs in the MVP.
5. **Economics = historical exposure only**, from CDFW MFDE port-area landings of HAB-sensitive species (tiered), inflation-adjusted, suppression-aware, labeled a lower bound. No modeled loss.
6. **The research model is not deployable yet** (data freeze absent, normalization stats not persisted, ambiguous checkpoints, MODIS-Aqua/`nflh` input dependency, ~10% skill vs persistence). It enters later as an explicitly experimental layer after retraining and held-out validation.

## First implementation milestone

**M0 — Truthful baseline + C-HARM vertical slice** (see `06` §2): remove synthetic content and broken links/tiles from the live app, scaffold the pipeline and shared schemas, and ship C-HARM through the new architecture end-to-end with provenance and freshness.
