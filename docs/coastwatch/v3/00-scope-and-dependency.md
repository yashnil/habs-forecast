# Harbor intelligence v3 · scope, isolation and the dependency on the Neritic v2 release

Started 2026-10-16 by the session that owns longer-term scientific capability ("Terminal 2"). The
release session ("Terminal 1") owns the #62 → #63 → #65 stack, the Vercel preview and production.

## Base and dependency

- Branch `feat/harbor-intelligence-v3`, created from `feat/coastal-intelligence-v2` at **d8d5b7128**
  (PR #65's head, CI green: pipeline, scope, web, web-e2e ×2, web-static).
- The draft PR targets `feat/coastal-intelligence-v2`, so its diff is only v3's work. It is **not
  mergeable on its own**: it depends on #62 → #63 → #65 reaching `main` first.
- Integration, once the owner has merged the v2 stack (bottom-up squash, per
  `docs/coastwatch/release/2026-10-16-release-candidate.md`):
  1. rebase this branch onto the new `main` (the v2 commits are then already there, squashed);
  2. re-run the unit suite, the full Playwright suite and `scripts/check` against fixtures;
  3. retarget the PR to `main`; the owner decides on merging and on any release.
- If #65 changes after d8d5b7128, this branch merges or rebases onto it only when a conflict or a
  needed change requires it (CLAUDE.md), never the other way round.

## What this branch never touches

- `main`, the #62/#63/#65 branches, Vercel projects and configuration, production or staging data,
  the pipeline's output schema (the manifest is frozen: everything here is computed in the web
  from data already published, or read live through web routes with recorded fixtures).
- Official notices' review state. Agency facts are quoted, with their page and read date.

## Scope, in order

1. **Historical change**: a shore measurement against its own station's record
   (`src/lib/history.ts`). See `01-history-species-consistency.md`.
2. **Species and activity**: which toxins the agencies' pages name for the chosen species, in their own
   words (Neritic's unreviewed copy, each sentence under its own agency), and what Neritic's readings can and cannot say about them (`src/content/speciesEvidence.ts`).
3. **Evidence consistency**: mechanical rules every claim must keep against its nodes
   (`src/lib/evidenceCheck.ts`), run over every harbour, activity, species and source state.
4. **Verified nearshore observations**: CeNCOOS shore stations (QARTOD-flagged) through
   `/api/conditions` (`02-nearshore-sources.md`).
5. **Research, isolated**: AI evaluation and forecasting baselines under `research/`, not wired into
   the app until independently validated.
