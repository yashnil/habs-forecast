# Neritic v8: market pass

Branch `feat/neritic-v8-market`, stacked on `feat/neritic-design-reset` (#77). Research and the gap analysis: [research.md](research.md). Before and after screens: [screens/](screens/).

## The brief

> The current app is not visually appealing … something like this first and foremost has to have a good UI/UX in order to be production ready and presentable. Research current competitors and similar geophysical apps, find the gaps, implement the missing functionality and frontend fixes. The opening animation is way too short: extend the play.

## What changed

| Area | Before (v7) | After (v8) |
|---|---|---|
| **Opening** | Resolved in ~1 s and stepped aside: a flash | A ~5 s play: the isobaths draw, the mark resolves, a sounder ping, the lockup and a line, then a dive into the map. Skippable at once; first visit only; a still frame under reduced motion. |
| **Explore basemap** | Bright blue-grey land; visible seams where the relief image ends; loud lakes, borders and labels | Graphite land matched to the relief (no seams); quieter lakes, borders, roads and labels; county lines come up with zoom |
| **Data rasters** | Nearest-sampled everywhere: stair-steps at state scale | Smoothed below zoom 8; true cells from harbour scale |
| **Overlays** | Station counts under harbour dots; heavy official lines and model edge; full-strength no-data hatch over the whole ocean | Clusters sit just offshore; fine official lines at state scale; a quiet model edge; the hatch half-strength at state scale, full near shore |
| **Time** | Steps only | Play / pause (Space), stops at any input; a break and hatched steps for forecast frames |
| **Dock** | The read hint inside the dock made it a slab | The hint floats in a pill above; the deck gets a bezel highlight and layered shadow |
| **My Coast, first visit** | A question, a search box and six text links | A dark hero (question, search, a coast drawing with the places) and six cards, each with water, waves and wind **measured now**, dated and sourced |
| **Briefing** | Four small soundings; tides as a text line in a fold | Soundings and a **tide strip** in one instrument card: the day's NOAA-predicted curve, turns labelled, a now line with rising or falling |
| **Craft floor** | — | touch-action, palette selection and caret, instrument scrollbars, measured line lengths, a halo on chart reference labels |

Start with [after-coast-landing-d.jpg](screens/after-coast-landing-d.jpg), [after-explore-chl-monterey-d.jpg](screens/after-explore-chl-monterey-d.jpg), [after-coast-briefing-p.jpg](screens/after-coast-briefing-p.jpg) and [after-opening-filmstrip.jpg](screens/after-opening-filmstrip.jpg).

## What did not change

- What any layer means, its coverage words, sources, dates and model boundaries.
- Notices are still unverified; "a missing notice is not open".
- Trip economics: models, scenarios, nothing pre-filled.
- Offline behaviour and accessibility semantics.
- The front door's readings come from the existing `/api/conditions` route (NDBC and CO-OPS). They are observations, never a forecast, and never a verdict on the water.
- The tide strip draws CO-OPS predictions only. It draws nothing for a day the predictions do not reach.

## Decisions for the owner

1. **Opening length.** About 5 s by request, then a 0.7 s dive. It is skippable by any input and shown once per device. The CSS cap is 6 s.
2. **Smoothing at state scale.** v7 nearest-sampled every zoom (design reset rev. 2 §5.1: "a pixel is a real source cell"). v8 keeps that from zoom 8 and smooths below it, where cells are a few pixels wide. Revert by setting `SHARP_ZOOM = 0` in MapCanvas.
3. **Front-door readings.** Six small `/api/conditions` requests when the cards come into view, each cached 10 minutes server-side. Acceptable for a preview. At public scale, consider one batched route.
4. **Follow-ups found by the research, not built** (they need backends, data or moderation):
   - saved places with alerts (push or email);
   - anomaly against climatology;
   - an impact-based legend from agency effect bins;
   - A/B compare (swipe);
   - structured fleet reports.

## Testing

- Unit (vitest): 573, including the new `tests/unit/tideCurve.test.ts`.
- New e2e: `tests/e2e/v8.spec.ts` covers playback and its stops, the forecast break, zoom-dependent smoothing, the front door's cards and readings, and the tide strip. `tests/e2e/splash.spec.ts` is updated for the full play.
- Full Linux CI-equivalent e2e run before push: see the PR.
- impeccable detector on `src/` and the live pages; the Vercel guidelines checklist.
