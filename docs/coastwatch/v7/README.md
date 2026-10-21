# Neritic v7: design reset — results

- **Spec:** [design.md](design.md). It records the design owner's decisions, drawn from four parallel audits (visual, fisherman workflow, scientific content, frontend).
- **Branch:** `feat/neritic-design-reset`, stacked on `feat/neritic-v6` (#75). Nothing was merged; production and production data are unchanged.

## How it was built

1. Four audits ran concurrently. One design owner then fixed the composition, built the shell and the primitives, and wrote the spec.
2. Three page tracks ran in parallel, each in its own worktree with file ownership: Explore, My Coast and Research. Integration stayed central.
3. The design owner reviewed the integrated screens, then a fresh reviewer, who had not built the UI, ran the rejection pass. Both rounds of findings were fixed before delivery.

## Screens

The before and after captures are in [screens/](screens). Before is the v6 head `e17d30d`; after is this branch. Both use real staging data at 1440×900, 390 and 320.

| State | Before | After |
|---|---|---|
| My Coast, first visit | `before-01-coast-first-*` | `after-01-coast-first-*` |
| Returning Monterey, commercial rock crab | `before-02-…` | `after-02-…` |
| Trip economics | `before-03-coast-econ-*` | `after-03-…`, `after-09-coast-econ-filled-d1440`, `after-10-…-breakdown`, `after-11-coast-phone-bar-p390` |
| Explore, nothing selected | `before-04-explore-*` | `after-04-…`, `after-08-explore-layers-d1440` |
| Explore, harbour selected | `before-05-…` | `after-05-…` |
| Research, starting state | `before-06-research-*` | `after-06-…` |
| Offline | `before-07-offline-*` | `after-07-…` |

## Verification

The final head was checked in Playwright's Linux image (v1.64.0-noble, Node 22, CI=1, software GL, 4 CPUs, 2 workers):

- lint: 0 errors;
- typecheck: clean;
- unit tests: 567/567;
- e2e: 291/291.

The full macOS suite also passed 291/291 three times during integration. Splash timing was run 10×: 70/70.

Other checks:

- Pipeline: 246 passed.
- Offline field-pack proof: PROVEN. No request left the device, and every resource asked for inside the pack's box was saved.
- Private preview: protected, reading the staging data. All 9 routes return 200, the disclosure is on every page, and there were 0 page errors.
