# CLAUDE.md

## Git / commit hygiene

Do not commit every small edit, test fix, formatting change, or intermediate step.

For substantial work, group related changes into meaningful logical commits. A large day of work should
usually result in roughly 5–15 meaningful commits, and often fewer, rather than dozens of microcommits.

Good commit boundaries are things like:
- complete one feature or page;
- complete one data/pipeline change;
- complete one meaningful refactor;
- add a coherent test suite;
- fix one distinct bug;
- finish one documentation/provenance milestone.

Bad reasons to create a new commit:
- tiny CSS/layout tweak;
- typo;
- rerunning formatting;
- fixing a test caused by the immediately preceding change;
- adding one missing import;
- every individual file edit;
- intermediate WIP state that has no independent value.

Before pushing:
1. inspect `git log --oneline`;
2. identify obvious WIP/fixup/microcommits;
3. squash those into the logical commit they belong to;
4. preserve genuinely distinct milestones rather than flattening everything into one giant commit.

Prefer a clean, understandable history over maximizing commit count.

If the branch has NOT been pushed yet, freely use interactive rebase to clean it up.

If the branch has already been pushed, only rewrite history if it is an isolated feature branch and doing
so is safe. Use `git push --force-with-lease`, never plain `--force`.

Never rewrite `main` or another shared branch.

For PRs, leave the branch history readable. We do not need one commit per tiny action.

## GitHub workflow

Claude may manage the normal Git/GitHub workflow for this repository. On a feature branch, handle the
full workflow without asking, unless an action is destructive or needs a product/design decision.

Allowed without asking:
- inspect git status/history; create and switch feature branches; stage; make meaningful commits;
- clean up obvious WIP/microcommits before pushing (see commit hygiene above);
- push feature branches;
- use the authenticated `gh` CLI: `gh pr create`, `gh pr edit`, `gh pr view`, `gh pr diff`,
  `gh pr checks`, `gh pr comment`;
- create draft PRs and write/update their titles and descriptions as the work changes.

Do not ask the user to run routine Git or `gh` commands that Claude can safely run itself.

### PR workflow

- When a coherent feature/design milestone is ready, push the branch and create or update a **draft** PR.
- The PR is the review hub. Include: concise summary; what changed; what did not change; testing/status;
  important screenshots/docs; decisions the user needs to make; known limitations/follow-ups.
- Link directly to relevant files (clickable GitHub links), not bare paths.
- If screenshots/design docs exist, point reviewers to the best starting page.
- Keep the PR in draft until the user explicitly approves making it ready.

### Ask first

- merging into `main`;
- deleting important remote branches;
- force-pushing a branch other people may be using;
- changing production infrastructure;
- deploying;
- irreversible or destructive changes.

## Project-specific rules (CoastWatch)

- **One session per checkout.** Do development in a dedicated `git worktree` per branch. Never switch
  the branch of a checkout another session may be using, and stage explicit paths, never `git add -A`
  in a shared tree.
- **BASES demo:** <https://coastwatch-demo.vercel.app>.
  - It is a separately managed release in the Vercel project `coastwatch-demo`.
  - Git auto-deploys are disconnected; it deploys only by a manual `vercel deploy --prod`, and **only
    with the user's explicit approval each time**.
  - Releases are recorded as protected tags `demo-release/YYYY-MM-DD`; the previous tag is the rollback.
  - Never merge demo branches into `main`.
- **Data:** never publish to the production data branch (`coastwatch-data`) outside the scheduled or
  approved workflow; test new data on the `staging` channel.
- **Official notices:** never mark the registry or a record human-verified, lifted or expired.
  Only a person running `cwp review-official` after checking the sources may do that.
- **Science:** quote only same-pipeline research numbers (the PINN matches, does not beat, the
  ConvLSTM). Never present observations as forecasts or chlorophyll as toxin.
