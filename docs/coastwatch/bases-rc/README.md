# BASES demo release candidate (`demo/bases-rc`)

Release candidate for **coastwatch-demo.vercel.app**: a map-only preview build (`NEXT_PUBLIC_CW_DEMO=1`) reading the published GitHub Pages dataset. **Production is unchanged.** It still serves deployment `dpl_JCxjvXS1qAjmPimD3AnmjjwgEEjs` (`coastwatch-demo-fonrbircq`, 2026-10-09 22:59 PDT), which is also the rollback target.

## What the candidate contains

| Source | What |
|---|---|
| PR #15 (P3, @4b45ffb) | speed-class arrows, on-map timestamp chip, region fit, port values only with the forecast, California-day freshness ages |
| PR #14 (preview mode) | map-only build, first-visit introduction, `/bloom` and `/fisheries` redirect, currents hidden |
| PR #16 (@8061383) | Del Norte razor clam closure (CDFW) and CDPH SN26-020 records, unverified; `RegistryDisclosure` for full builds |
| this branch | items below |

This branch adds:
1. **One notice disclosure, driven by the data.** In previews, one strip replaces the red preview banner and `RegistryDisclosure`. It names the Oct 9 Del Norte closure and warning, and says whether the published list contains them yet: it reads the registry and checks the two record IDs. It links to CDFW and SN26-020. HF-radar outages no longer raise a banner in previews, because the layer is hidden there.
2. **Introduction.** Shorter copy that covers the problem, who it affects and the three kinds of layer. Its actions sit in a pinned footer, so the call to action is in view at 1280×720 and 390×844.
3. **Phone layout.** In the collapsed dock, the picker sits beside the layer tabs, the forecast days read "nowcast / day 1–3", and longer notes move to the expanded view, while the freshness badge stays visible. The duplicate phone notice card is removed in previews. Map height on a 390×844 forecast view went from ~210 px to ~380 px.
4. **Provenance fix (all builds).** The on-map satellite stamp said "Sentinel-3 300 m" for VIIRS 750 m. It now names the sensor actually drawn, and both sensors for multi-sensor (`12544ee`, for cherry-picking into P3).
5. **Copy fix (all builds).** "CoastWatch lists only notices a person has transcribed" contradicted the registry, which is an AI transcription not reviewed by a person. Corrected, with a safety test (`a5dbc72`).
6. Limit labels sit above their lines; the preview drawer title reads "N notices listed".

Not changed: the pipeline, datasets, raster colours and opacity, C-HARM resolution, data values, `main`, and the production deployment.

## Visual review passes

**Pass 1** (RC = main + P3 + preview, live data, 1440 / 1280 / 390):
- *Problems:*
  - On a phone the map got ~210 of 844 px, behind two banners (~250 px), a notice card and a ~55 vh dock.
  - A currents outage banner appeared for a layer previews hide.
  - The notice warning appeared four times.
  - The intro's call to action was below the fold at 1440×900.
- *Changes:* items 1–3 above.
- *Result:* improved. The phone map roughly doubled; one disclosure strip.
- *Left over:* the phone picker truncated to "Particulate D", "2 days ago" wrapped, and the badge row wrapped. All three were fixed in the same pass.

**Pass 2** (deployed preview, live data):
- *Problems:*
  - The VIIRS stamp mislabelled the sensor (item 4).
  - The sources page claimed person-transcribed notices (item 5).
  - The axe check flagged `aria-required-children`: the picker sat inside `role=tablist`. Fixed.
  - Limit labels overlapped their lines.
- *Changes:* items 4–6.
- *Result:* improved, re-verified on the final preview.
- *Left over:* see Known issues.

## Verification (final commit, 2026-10-10)

- `tsc`, `eslint`: clean.
- Unit (vitest): 101/101.
- e2e, full build (fixtures, ports 3420–3423): 94 passed, 6 skipped. 1 failure, a map-load timeout in `p2-currents` under parallel load; it passed 3/3 alone. Two tile-source tests flaked once under 5 workers and passed alone and on the next run.
- e2e, preview build (`tests/e2e/demo.spec.ts`, `CW_E2E_DEMO=1`): 6/6, including axe on the introduction.
- Deployed preview, live data, 30 page loads at 1440/1280/390:
  - map, Monterey, Santa Cruz port, C-HARM day 3, Sentinel-3, VIIRS, multi-sensor, notices drawer, statewide, sources;
  - no horizontal overflow, no failed requests;
  - `/bloom` and `/fisheries` return 307 to `/`.
- External links (CDFW, CDPH SN26-020, CDPH shellfish advisories, GitHub): all HTTP 200.

## Known issues (not fixed)

- **Satellite tile 404s in the console.** The pipeline skips empty tiles by design, and the manifest has no tile index. Fixing this needs a pipeline change. No visible effect.
- **Intermittent React hydration warning #418.** Seen on 4 of 30 loads under parallel load and 1 of 16 sequentially, on the current production deployment as well. React recovers by rendering on the client, and every affected screenshot renders correctly. I couldn't reproduce it in a UTC dev server, and the root cause is not found.
- **C-HARM looks like a uniform pink sheet around Monterey Bay.** That is the data: the particulate DA probability is ~75–78 % across the bay. Colours and opacity were left as published so the map matches its legend.
- **The notice registry is still unverified** (issue #12) and needs human review. The live dataset doesn't contain the Del Norte records until PR #16 is merged and published. Until then the strip says they are "not yet listed".
- **HF-radar currents stay hidden.** The source was failing on 2026-10-10 (ERDDAP 404 since 05:55Z).

## Release and rollback (needs explicit approval)

**Release.** Build the reviewed commit for production with the same flags as the preview, from a clean export, so nothing outside the commit is uploaded. Then verify the public URL at 1440 and 390.

```
git archive <reviewed commit> coastwatch-web | tar -x -C /tmp/cw-release
cd /tmp/cw-release/coastwatch-web && mkdir .vercel && cp <linked project.json> .vercel/
npx vercel deploy --prod --yes \
  --build-env NEXT_PUBLIC_CW_DEMO=1 \
  --build-env CW_DATA_BASE_URL=https://yashnil.github.io/habs-forecast/v1 \
  --env CW_DATA_BASE_URL=https://yashnil.github.io/habs-forecast/v1
```

This is preferred over `vercel promote <preview>` because promoting a preview can trigger a rebuild, and the flags above were passed on the command line, not stored in the project.

**Rollback** (instant; reuses the submitted build):

```
npx vercel rollback dpl_JCxjvXS1qAjmPimD3AnmjjwgEEjs --scope yashnils-projects
```
