# Coastal intelligence v2 · the plan, and what this pass did

Owner's brief: a harbour-first, scientifically rigorous, beautifully drawn coastal intelligence platform, built
in bounded stages, never pretending to know what sparse data cannot tell us. This document is the concise plan;
`00-baseline.md` is the state before it, `01-sources.md` the data feasibility matrix, `review/index.html` the
private review page with previews and screenshots.

## Principles that decide every call below

1. Harbour first: the question is "what do we know about *these* waters", never "what does the dataset show".
2. Every value carries its class (measurement · satellite retrieval · published forecast · reconstruction ·
   Neritic's experiment), its support, its distance from the place, its time and its freshness.
3. Representativeness is per variable (piers 2/15/60 km; satellite pixels 0.5/5/15 km; C-HARM own cell vs
   15 km context; synoptic wind 30 km; harbour water temperature 2 km), never one radius.
4. Rendering never creates data: nearest-neighbour cells, exact edges, no smoothing across land, no-value
   cells stay visible and explained.
5. Nothing experimental reaches the public map until validated against independent in-situ data.

## Stages

| Stage | Scope | State after this pass |
|---|---|---|
| 0 Baseline and research | audits, coverage at 22 ports, source matrix | **done**: `00-baseline.md`, `01-sources.md` |
| 1 Correctness and trust | day labels, latency tone, distance first, offshore demoted, duplicates, regression tests | **done** (commit "Trust fixes…"); map limit lines over land remain (Stage 3) |
| 2 Data and harbour evidence | evidence graph + `/api/evidence`; one or two verified in-situ improvements | **evidence service done**; in-situ improvement per `01-sources.md` |
| 3 Cartography | translucent uniform ramp, zoom-adaptive outlines and labels, harbour-zoom gap treatment, light map | **prototype on real data** in `carto/`; product integration next |
| 4 My Coast and mobile | answer-first briefing from the evidence graph, one evidence section, checklist, progressive disclosure | briefing line and outlook changed; full layout next |
| 5 Brand and polish | contour logo (owner's choice), type scale, controls, motion | **logo board** in `logo/`; nothing installed |
| 6 Experimental prediction and AI | pier-point outlook study; tool-oriented evidence access | design below; nothing built into the product |
| 7 Proof | full test matrix, before/after measurements | unit + affected e2e green; full CI on push |

## Stage 2 · the data model (built)

`coastwatch-web/src/lib/harborEvidence.ts` is the structured contract. Each **node**: `class`, `variable`,
`units`, `value`, `text`, `source {id, name, url, product, version}`, `time {observed, valid, issued, ageDays}`,
`support {kind: point|cell|cells|pixel|pixels|zone, nativeM, radiusKm, n}`, `where {lat, lon, name, km, bearing,
representative: at|near|context|far, setting}`, `freshness`, `quality.flags`, `lineage[]`, `caveats[]`.
**Gaps** say what is not known and the nearest alternative. **Claims** are deterministic sentences, each with
`supports: nodeId[]`, so a page, an API answer and a future assistant rest on the same evidence.

`GET /api/evidence?place=port-550&for=commercial&species=any&date=YYYY-MM-DD` returns the graph from Neritic's
own published data. Live conditions join on the page (from `/api/conditions`) and go into the same graph.

The manifest stays unchanged (frozen schema); every new in-situ source goes through a web route with recorded
fixtures, exactly as NDBC, CO-OPS and NWS do today, so the pipeline and the demo are never destabilised.

## Stage 3 · cartography (prototype → product)

The prototype in `carto/` keeps the exact-cell renderer and changes only what the reader sees: a perceptually
uniform ramp on a translucent raster so the relief shows through; hairline cell outlines from z 8.5 and per-cell
values from z 11; at harbour zoom the no-value water is a labelled hatch, the nearest valued cell is joined to
the harbour by a dashed line with its distance, stations are drawn with their last sample date, and the 15 km
context ring is dashed. Integration into `MapCanvas` is a display-only palette (the readout keeps the published
values and the legend states the classes).

## Stage 4 · My Coast (next)

Answer-first: the claims of the evidence graph, in order official → measured → model → satellite → conditions,
then one compact evidence section (the timeline), then the checklist. Target: the first useful answer within one
phone screen; detailed science behind one disclosure.

## Stage 6 · a harbour-level prediction research track (design only, not built)

What the labels allow: CalHABMAP particulate domoic acid, 4,267 samples at 10 piers since 2014, 155 at or above
0.5 ng/mL, clustered in 2014–17 and 2025 (doc 27). Monterey Wharf has no toxin since 2022-08; seven stations
count cells only. CDPH shellfish results are not published as data; IFCB counts are dashboard-only.

Therefore the only defensible target is a **pier-point outlook** (Pseudo-nitzschia abundance class or pDA ≥
0.5 ng/mL at a named pier, 1–2 weeks ahead), evaluated as a point series, never a map:

- predictors: C-HARM cells within 15 km, satellite chlorophyll within 5 km with age, HF-radar advection toward
  the pier, NWS/NDBC wind and upwelling index, water temperature, day of year;
- baselines: climatology, persistence, AR(1) on anomalies, C-HARM's own cell probability;
- splits: chronological (train to 2022, test 2023–26) **and** leave-one-pier-out; no interpolated labels; skill
  by pier and by distance to the nearest C-HARM cell; Brier score and reliability; failure cases per event;
- promotion rule: beats every baseline on held-out piers across at least two seasons, with calibration within
  5 points, before anything reaches a page, and then only at the piers it was tested at.

The PINN/ConvLSTM chlorophyll work (PR #49 audit: climatology + AR(1) matches both networks; the physics term is
inert) is not a toxin model and will not be advertised as one. A validated nearshore chlorophyll or current
forecast is preferable to an unvalidated toxin heatmap; if the labels cannot support pier-level toxin skill,
the study says so and stops.

## Stage 6 · AI architecture (design only)

Tools first, model later. The evidence graph is the tool surface: `evidence(place, activity, species, date)`,
`coverage(place)`, `history(station, variable, range)`, `forecast(place, date)`, `notices(place, species)`,
`sources()`. Response contract: `{ answer, evidence: nodeId[], sources: url[], dates: {observed, valid},
applicability: {place, km, representative}, uncertainty, missing: Gap[], next_steps: official links }`.
The research prototype in PR #49 (`research/intelligence/cwi`) already enforces citation per sentence, official-
first ordering and forbidden-phrase guards with no network calls; it is the evaluation harness for golden and
adversarial questions. No paid model call is made without approval.

## Release controls

Private previews only. No merge, deploy, domain change or purchase. PR #62 (rebrand) and PR #63 (design
proposal) stay as they are; this work is stacked on #63 so the review page can link the design study, and the
Stage 1 commits are independent of the brand and can be cherry-picked to `main` if the rebrand waits.
