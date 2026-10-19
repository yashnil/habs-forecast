# Neritic v4: merge, deployment, smoke test and rollback plan

**Nothing here runs without the owner's explicit approval.** Each of these needs its own approval:
- merging #68 (step B);
- the production deploy (step D);
- promoting it to the public URL (step F).

The current public site stays untouched until step F, and step I restores it.

## Starting point (verified 2026-10-19)

| | |
|---|---|
| Public research preview | `coastwatch-demo.vercel.app` = deployment `dpl_HyAetdpnw23aJ1oKu3CFFGjDoxt5` (Neritic v2, `main` `4d2e3e1c`, tag `demo-release/2026-10-16`) |
| `main` | `b78c69f` (v2 + #66 squash); never deployed |
| Candidate | draft PR #68, `feat/neritic-v4`, rebased on `b78c69f`; head as listed in the PR |
| Rollback references | Tags `demo-release/2026-10-16` (= public), `archive/main-before-66` (= `4d2e3e1c`), `archive/neritic-v4-before-rebase-on-66`. Branches `feat/harbor-intelligence-v3` and `feat/neritic-v4`, both kept |
| Vercel project | `yashnils-projects/coastwatch-demo`: Git disconnected (manual deploys only), no stored env vars |

## A. Go criteria (all must hold)

1. The real-device checklist (`real-device-checklist.md`) has passed on one iPhone and one Android.
2. Owner decisions are recorded:
   - logo: C2 installed, or the provisional mark kept;
   - name clearance;
   - release with notices unverified, or a person has run `cwp review-official`.
3. #68's head is the exact tested head: one complete Linux run and GitHub CI, both green.
4. `main` has not moved since `b78c69f`. If it has, rebase #68, rerun every suite, and start again here.

## B. Merge #68 (owner approval)

```sh
gh pr view 68 --json headRefOid,mergeStateStatus        # head = tested head; CLEAN
gh pr checks 68                                         # every job pass
gh pr ready 68
gh pr merge 68 --squash --match-head-commit <tested head sha> \
  --subject "Neritic v4: Explore, My Coast and Research … (#68)"
git fetch origin
git rev-parse <new main>^{tree}; git rev-parse <tested head>^{tree}   # must be equal
```

- Watch main's CI on the merge commit until every job passes. A single map-test timeout on the runner
  is a known mode (see `release-hardening.md`). Rerun only the failed job, and quote that it was rerun.
- Keep the `feat/neritic-v4` and `feat/harbor-intelligence-v3` branches until step H is done.

## C. Freeze references

```sh
git tag -a archive/main-before-68 b78c69f -m "main before the v4 squash merge"
git push origin refs/tags/archive/main-before-68
```

## D. Build the production candidate without touching the public URL (owner approval)

```sh
D=$(mktemp -d)/v4-release
git archive <new main sha>:coastwatch-web | tar -x -C "$D"     # a clean export, never a working tree
cd "$D" && vercel link --yes --project coastwatch-demo --scope yashnils-projects
vercel deploy --prod --skip-domain --archive=tgz --yes \
  --build-env CW_DATA_BASE_URL=https://yashnil.github.io/habs-forecast/v1 \
  --env CW_DATA_BASE_URL=https://yashnil.github.io/habs-forecast/v1 \
  --meta gitCommit=<new main sha> --meta tree=$(git rev-parse <new main sha>:coastwatch-web)
```

- `--skip-domain` keeps `coastwatch-demo.vercel.app` on v2.
- It does move the team alias `coastwatch-demo-yashnils-projects.vercel.app`, which is protected (302).
- Never use `vercel promote` of a preview: it can rebuild without these flags.

## E. Smoke-test the candidate (before anyone sees it)

Use the deployment's own URL with the protection-bypass **cookie**
(`/?x-vercel-protection-bypass=<token>&x-vercel-set-bypass-cookie=true`). A bypass header breaks the
CORS preflight to the data host.

| Check | Pass |
|---|---|
| The 12 routes (`/`, `/coast`, two briefings, `/research`, a station, `/about`, `/?layer=multi`, `/?layer=currents&chl=1`, `/api/conditions`, `/sw.js`, `/offline/monterey/pack.json`) | all 200 |
| Status centre | "Ready", or each degraded source said by name |
| `/api/evidence` for 8 harbours × 4 activity/species through `checkGraph` | 0 violations |
| Remote offline proof (`v4shots/remote-offline.cjs`) | PROVEN |
| Visual at 1440, 390 and 320 px: Explore with a harbour, the notices panel, a briefing, Research | warnings ("Notices not verified", "not reviewed by a person") visible; nothing overlapped |
| Official notices | unverified; the SN26-018/SN26-021 dispute shown as unresolved |

**Any failure: stop. The public site has not changed.**

## F. Promote (owner approval)

```sh
vercel promote <candidate dpl id> --scope yashnils-projects
vercel inspect coastwatch-demo.vercel.app --scope yashnils-projects   # id = the candidate
```

## G. Smoke-test the public URL (no token needed)

1. Run every check from E against `https://coastwatch-demo.vercel.app`.
2. Open it on a phone over mobile data.
3. Re-check after 30 minutes: data age, the status centre, one notice.

## H. Record

```sh
git tag -a demo-release/<YYYY-MM-DD> <new main sha> -m "Neritic v4 on the research preview: dpl …"
git push origin refs/tags/demo-release/<YYYY-MM-DD>
```

Add a release record under `docs/coastwatch/release/`, in a PR: deployment id, commit, smoke results,
rollback. Then retire the archive branches.

## I. Rollback (any time after F)

```sh
vercel rollback dpl_HyAetdpnw23aJ1oKu3CFFGjDoxt5 --scope yashnils-projects      # Neritic v2, demo-release/2026-10-16
vercel inspect coastwatch-demo.vercel.app --scope yashnils-projects            # id = dpl_HyAetdpnw…
```

After rolling back:
- Smoke-test v2 on its own routes: `/`, `/coast`, `/bloom`, `/about`.
- A reader who saved a v4 field pack keeps a service worker. Online it is network-first, so they get
  v2's pages; their pack still opens offline until they remove it. v2 has no `/sw.js`, so the worker
  is never updated.
- `main` keeps the v4 commit. To take it off `main`, revert the squash commit in a PR (owner
  approval). The data pipeline and the published data are unaffected either way: v4 changes no
  pipeline output or manifest field.

**Roll back if any of these happens:**
- a hard error on any route;
- an official notice shown as verified, or missing;
- chlorophyll or a model value presented as a measurement or as toxin;
- the offline pack failing to open;
- the status centre claiming "Ready" while a source is down.
