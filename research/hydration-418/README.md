# React #418 on the public site: cause, evidence and options (2026-10-14)

**Status:** the cause is identified. Two fixes were tried and both were rejected, for the reasons
below. **Option C (framework update) is now the candidate fix, see "Candidate fix" at the end.**
Nothing has been deployed to production (`demo-release/2026-10-14`).

## Symptom
On some cold page loads the browser logs React error #418 ("the server-rendered HTML didn't match
the client"). React recovers by re-rendering the affected part in the browser, so the visible
content ends up correct.
- **Production:** about 1 in 20 loads on Vercel, on any page.
- **Locally:** about 1 in 140 loads.

## Cause
1. Next starts hydrating while the HTML, and the server-components (RSC, Flight) payload inlined
   at the end of `<body>` as `self.__next_f.push(…)` scripts, are still streaming.
2. React hydrates as far as the payload allows. When parts of the page tree are still pending, it
   suspends inside the app shell, at `OfficialProvider`'s context provider, the shell `<div>`'s
   children, and a fragment under Next's layout router. (Seen by instrumenting `throwException`
   in a local production react-dom.)
3. There is no Suspense boundary between those points and the root, so the root suspends and
   React retries.
4. Occasionally the retry leaves the shell `<div>` with its server children unclaimed.
   `popHydrationState` finds `nextHydratableInstance = <header>` and throws #418. (Seen by
   instrumenting `throwOnHydrationMismatch` / `popHydrationState`.)
5. When the error fires, `document.readyState` is `"loading"` and the parsed DOM is already
   complete. Only the payload scripts are still arriving.

## Evidence
| Condition | #418 |
|---|---|
| Release build, HTML drip-fed (4 KB every 25 ms, `dripproxy.mjs`) | 24 errors in 34 cold loads |
| Old main `8bf182d6`, drip-fed | 5 of 22 |
| Next 15.5.27 (latest 15.5 patch), drip-fed | 10 of 36 (not fixed) |
| Release build, direct local loads | about 1 in 140 |

The redesign's larger pages (more payload) make the error more frequent, but the weakness predates
it.

## Fixes tried
1. **`<Suspense>` around the shell's content in `AppShell`.**
   - #418 fell to 0 in 225 loads, both drip-fed and direct, and `hydration.spec.ts.txt` (the drip
     regression test) failed without it and passed with it.
   - **But the full Linux suite failed 114 tests with duplicated DOM.** Fizz outlines that boundary
     (`<!--$?--><template id="B:0">` plus hidden `S:` segments). When an update reaches the boundary
     before its streamed content is revealed, React client-renders it and the streamed content is
     inserted as well.
   - Rejected.
2. **A root `app/loading.tsx` (Next's segment boundary).**
   - #418 fell to 0 in 31 drip-fed loads, with no duplicated DOM.
   - **But without JavaScript the page content stays hidden.** The streamed segment needs the
     reveal script, which breaks the deliberate server-rendered design (m7: "the server-rendered
     page fits a phone").
   - Rejected.

## Options for the owner
- **A. Leave it.** React recovers on its own and the content is correct. Keep this reproduction.
- **B. Shrink what the shell waits on.** Pass smaller props to the shell's client components and
  move large data (station timelines, the full official dataset) into client fetches. That makes
  the error rarer, not impossible.
- **C. Upgrade to React 19.2 / Next 16,** then re-measure with `dripproxy.mjs` and
  `hydration.spec.ts.txt`. That is a framework upgrade needing its own review and full testing.
- **D. A narrow boundary that Fizz does not outline,** for example only around client components
  whose props are already serialised. That needs more investigation, and the risk is the same as
  fix 1.

## Candidate fix: Next 16.4.0 + React 19.3.0 (option C, 2026-10-14)
- **Why a React-only bump does nothing:** Next 15.5.x bundles its own React for the App Router
  (the browser ran `19.2.0-canary-0bdb9206-20250818`, never the `react` in package.json). Bumping
  react/react-dom on 15.5.14 or 15.5.27 left the app's client chunks byte-identical.
- Next 16.4.0 ships React `19.3.0-canary-278794d7-20261002`. It includes
  [facebook/react#35494](https://github.com/facebook/react/pull/35494) ("Correctly handle replaying
  when hydrating"), which matches step 4 of the cause above. That link is inferred from the changelog
  and the measurements, not instrumented.

| Build | Drip-fed #418 | Direct #418 |
|---|---|---|
| next 15.5.14 / React 19.1.0 (baseline and controls) | 10/54, 17/90 | 2/60, 0/120 |
| react/react-dom 19.2.8 or 19.3.0 on next 15.5.x | 6–7 of 30 each (no change) | – |
| **next 16.4.0 / React 19.3.0** | **0/185** | **0/120** |

The full Linux suite passes on the upgrade: 289 unit tests, 243 e2e including `tests/e2e/hydration.spec.ts`, and
102/102 repeats. Without JavaScript the content is visible, and the text matches the baseline. There is no
duplicated DOM. On Vercel the rate still has to be measured on a preview.

**Correction:** reproduce with the fixture data (`CW_DATA_BASE_URL=/data/fixture/v1`, as the Playwright
config does). Without it, pages are about 35 KB instead of 400–650 KB, and the error almost never fires.

## Reproduce
```
node research/hydration-418/dripproxy.mjs 3439 <next-start-port> 4096 25
SAMPLE=0.15 node research/hydration-418/coldloads.mjs http://localhost:3439 3 out.json
```
`hydration.spec.ts.txt` is the Playwright regression test. Put it in `coastwatch-web/tests/e2e/`
together with a fix.
