# Neritic private pilot (Thursday 2026-11-12)

The integration branch is `feat/neritic-pilot`. Each track branches from it, works in its own worktree and merges back. The integrator owns `coastwatch-web/src/pilot/contracts.ts`, `readiness.ts`, the nav and the final integration.

## Product
**Today at your port** (`/today?place=port-592`) is the pilot's front door. Within a minute it should tell a commercial fisherman:
- what is open, restricted or delayed for their fisheries, with official status first;
- what the latest toxin tests say and where they are heading;
- what sea, weather and tide are doing;
- what bloom risk looks like over the next 0–3 days (C-HARM), plus an 8-day chlorophyll outlook;
- what the whale-entanglement (RAMP) rules are;
- what changed since the last visit.

It ends with a few actions: save offline, set an alert, Ask Neritic.

Explore, My Coast, Research and Sources stay as supporting views. Public v1 (`coastwatch-demo`) is never touched.

## As integrated
- **One fact source.** `loadPortTodayFacts(placeId)` (`coastwatch-web/src/pilot/today/portToday.ts`) is the port's facts for every fishery: Today draws them, alerts compare them, Ask Neritic answers from them, with the same `PortFact.id`s. Each has an address on Today (`/today?place=…#fact-<id>`).
- **One shell.** Today, `/ask`, `/account` and the legal pages share the v4 top bar (with the alerts bell), the phone dock and the Almanac styling (`src/pilot/shell/`, `today.css`, `pilot.css`).
- **Front door.** With `NERITIC_PILOT=1` a bare `/` opens `/today`; every deep link (`?region`, `?port`, `?layer`, `?sel`, …) still opens Explore, whose own address is `/?view=map`. Without the flag `/` is Explore, as on the public release.
- **The whole journey** is `tests/e2e/pilot-journey.spec.ts` (390 and 1440 px, axe clean). The demo's fallback screenshots are in [`pitch/shots/`](pitch/shots/); recapture with `scripts/pilot-shots.mjs`.

## Tracks
| Track | Branch | Owns |
|---|---|---|
| D: data | `pilot/data` | `pipeline/**`, `schemas/v1/biotoxin*` + `fisheries-status*`, `data/curated/fisheries-status/**`, `coastwatch-web/src/pilot/data/**` (loaders), web fixtures under `tests/fixture-data/v1/{biotoxin,fisheries-status,derived}` |
| T: Today | `pilot/today` | `coastwatch-web/src/pilot/today/**`, `src/pilot/facts.ts`, `src/app/today/**`, Today e2e/unit tests, design notes `docs/coastwatch/pilot/design/**` |
| A: accounts + alerts | `pilot/accounts` | `supabase/**`, `src/pilot/account/**`, `src/pilot/alerts/**`, `src/app/{account,privacy,terms,sms-terms}/**`, `src/app/api/alerts/**` |
| Q: Ask + SMS | `pilot/ask` | `src/pilot/ask/**`, `src/app/api/{ask,sms}/**`, `src/app/ask/**`, golden/adversarial evals |
| P: pitch | `pilot/pitch` | `docs/coastwatch/pilot/pitch/**` |

Shared files that a track needs to touch go in its PR notes for the integrator: `package.json`, `layout.tsx`, `TopBar.tsx`, `globals.css`, `next.config.ts`, `sw.js`.

## Rules
- Every science, CLAUDE.md and repo rule applies:
  - never present chlorophyll as toxin, or observations as forecasts;
  - never mark notices verified;
  - never add manifest fields; new data goes in sidecars, like `derived/index.json`;
  - quote only same-pipeline research numbers.
- Data goes to the **staging** channel only. No public deploy, no purchases. Missing credentials give honest readiness labels (`src/pilot/readiness.ts`).
- Integration tests are required: each track adds unit tests, and e2e tests run on fixture data. The integrated branch must pass lint, typecheck, vitest, pipeline pytest and Playwright in the Linux container.
