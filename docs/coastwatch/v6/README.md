# Neritic v6: coastal decision intelligence (working notes)

## Base and integration strategy (verified 2026-10-20)

| PR | Branch | Base | Head |
|---|---|---|---|
| #68 | `feat/neritic-v4` | `main` (`b78c69f`) | `260ad2e` |
| #73 | `feat/neritic-v5` | `feat/neritic-v4` | `47e5506` (all CI green) |
| #72 | `fix/coastal-scope-bays` | `main` | `050551b` |
| #74 | `feat/coastal-derived-layers` | `fix/coastal-scope-bays` | `7bf3c45` |

- **v6** is branch `feat/neritic-v6`, created from `47e5506`: the integrated app, which already contains #68.
  - It becomes a draft PR stacked on #73.
  - Existing PRs and their history are left as they are.
- **Data:** the derived layers and bay coverage stay in #72/#74. The app reads `derived/index.json` only when present.
  - Private previews read the **staging** dataset (`coastwatch-data-staging`).
  - Production data is not changed.
- **Parallel implementation tracks** run in their own worktrees from the v6 base. One owner per shared file (navigation, tokens, global styles, map). The lead integrates.
- **Not done here:** merging, public deploys, data publishing, paid services.
