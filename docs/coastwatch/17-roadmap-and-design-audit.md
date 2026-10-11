# 17 — Status, design audit and prioritized roadmap (2026-10-10, updated after the #16/#15 merges)

## 1. Status

| Item | State |
|---|---|
| BASES demo, <https://coastwatch-demo.vercel.app> | **Released 2026-10-10 with the user's approval:** PR #18's release candidate (`372a80c`), deployment `dpl_GpFfbFP42QGBM3XwvT6FfikWw4jm`. Its notice strip reads the published registry, so it stays true before and after the Del Norte records are published. The Vercel project's Git auto-deploys are disconnected (manual deploys only). Release refs are protected tags: `demo-release/2026-10-09-submitted` (the build submitted to BASES, the rollback) and `demo-release/2026-10-10`. |
| Production data, GitHub Pages | P0–P3 on `main`, publishing every 6 h with all verification gates. NOAA dataset reloads ("Currently unknown datasetID") recur. The previous data stay up with their own dates. |
| Full website | Not deployed. |
| PR #16, regulatory | **Merged** (`4b603fd`). The Del Norte CDFW closure and CDPH SN26-020 warning are in the registry, **unverified**, with the site-wide disclosure and the corrected "not checked by a person" wording. |
| PR #15, P3 map refinement | **Merged** (`88d6aef`) after #16. Includes Pacific-calendar freshness ages, the sensor-name fix for the on-map timestamp, and de-flaked tests. |
| Parallel sessions | Ended. One development session; worktree and branch ownership consolidated. |
| Issue #12 | Open. It needs a person: [checklist](p3/regulatory-review-2026-10-10.md). |

## 2. Regulatory omissions (PR #16)

**Two new records, transcribed verbatim and checked on the agency pages on 2026-10-10:**

| Record | Agency, action | Scope | Effective | Text |
|---|---|---|---|---|
| `cdfw-2026-razor-clam-del-norte` | CDFW, recreational fishery closure | Del Norte County, razor clam | 2026-10-09 | "The recreational razor clam fishery closed in Del Norte County on October 9, 2026, and remains closed in Humboldt County due to elevated levels of domoic acid in razor clams." |
| `cdph-2026-sn26-020-razor-clam-del-norte` | CDPH, consumption advisory, sport harvest | Del Norte County, razor clams only | 2026-10-09 | "CDPH is advising the public not to consume sport-harvested razor clams gathered from Del Norte County due to high levels of Domoic Acid." |

- **Humboldt wording:** that record now carries the CDFW page's current sentence.
- **Open questions** are listed in each record's `uncertainties`: the OEHHA memo, and the "razor clam advisory for Humboldt County" that SN26-020 refers to.

**Safe integration:**
- **Nothing is marked verified.** The registry stays `pending_human_review`, and SN26-020 stays unreviewed, so the watcher keeps flagging it.
- **Disclosure on every page.** A strip appears whenever the registry is not human-verified. It says *why*: agency pages changed since the last review (naming the items), or the list was never checked by a person. It says the list may be incomplete, links to CDFW and CDPH, and opens the list. On phones it is one line.

  It is driven by the verification state, so a future omission is disclosed automatically as soon as the watcher sees a page change, not only this one.
- **Order:** merge #16 **before** #15, so the full site never shows the list without the disclosure.
- **Still needed:** a person completes the checklist and runs `cwp review-official`.

## 3. P3 review (PR #15)

| Area | Finding | Action |
|---|---|---|
| Current arrows | Five speed classes with a matching legend, outlined glyphs centred on the cell; readable over the dark sea and over chlorophyll | Keep |
| Combined layers | Opt-in. The panel and the map chip state that chlorophyll is not toxin, that currents are observed and not a forecast, the time gap (a median of about 3 days), and that arrows don't show where a bloom will travel. Evidence: only 53 % of radar cells have chlorophyll; arrows turn 25–29° per day | Keep opt-in; never the default |
| Map composition | The region fit uses the largest free area. But at 1440 the Monterey fit can cut off the south of the bay, and with a port selected at 1280 three panels squeeze the bay | Redesign milestone (§5) |
| Mobile | **On phones the map gets only about 300 px of 844.** The disclosure strip, the failure banner, the official card and a 55 vh dock stack over it. This is the most important visual problem | Redesign milestone |
| Timestamps and provenance | **Fixed:** freshness ages counted UTC days while day labels used Pacific days ("Oct 9 · today" next to "issued 2 days ago"). Ages now use the Pacific calendar (`83a79cb`) | Done |
| Tests | Two satellite e2e tests were flaky under load (tile source read before update) | Done: they wait for the expected URL (`4b45ffb`) |
| Regulatory | P3 doesn't touch the official shell, so it is independent of #16 | Merge #16 first |
| Production data | P3 screenshots and performance were measured on production data. No schema change | Compatible |
| Also seen | Latitude-limit labels ("36°31.46′ N · south limit") collide with the coast and town labels. A failure banner shows for a layer group the visitor isn't using | Redesign milestone |
| Minor data bug | When Sentinel-3 is carried over, the satellite source's `latest_valid_date` reports VIIRS (Oct 4) instead of Sentinel-3 (Oct 8). Conservative, but wrong | Small pipeline fix, M4.1 |

**Verdict:** P3 is ready to merge after #16. The remaining issues are layout and visual problems that belong to the redesign, not regressions.

## 4. Design audit against the design reset

The design-reset spec ([design-reset/design-spec.md](https://github.com/yashnil/habs-forecast/blob/archive/pr7-design-reset/docs/coastwatch/design-reset/design-spec.md)) is the baseline. The bar you set today is higher: an immersive, portfolio-grade product. Screenshots: [map 390](roadmap/audit-map-390.png), [Bloom 1440](roadmap/audit-bloom-1440.png), [Fisheries 1440](roadmap/audit-fisheries-1440.png).

### 4.1 Ocean Map

**Done to spec:**
- three-state navigation;
- the official component family (pill, nav row, inspector section, drawer);
- the stepped forecast ramp with no-value hatching;
- coastline above the data;
- dock with day steps;
- inspector order (official first);
- URL state;
- satellite, multi-sensor and currents layers.

**Differences from spec:**
- **Hover readouts:** hover-to-read exact values (§3.1) only exists as a tap or click in the inspector.
- **Inspector content:** the model section lacks the 4-day min–max strip and the 30-day nowcast sparkline. "Measured nearby" (the nearest CalHABMAP station) is not in the map inspector.
- **Region views:** only Monterey Bay has a hand-set view; the other regions use their published bounds.
- **Missing map layers:** the graticule and the CalHABMAP station layer.
- **Ports:** no keyboard navigation of ports.

**Below the new bar:**
- **The map does not dominate.** Floating cards take about 45 % of a 1440 screen and about 65 % of a phone. Banners stack above the map instead of being one calm status line.
- **Panels are generic.** Same-weight rounded white cards; the dock mixes controls, legend, provenance and notes at one visual weight.
- **Motion and transitions:** no designed motion. Panel open and close and layer changes are instant cuts.
- **Mobile:** a long scrolling dock instead of designed sheets with snap points (peek, half, full).
- **Basemap:** the default OpenFreeMap style, lightly tinted. Coastline and terrain are not designed for a marine product: no bathymetry tint, no refined label hierarchy.

### 4.2 Bloom Intelligence: still largely the M3 structure

| Spec (§3.2) | Implemented |
|---|---|
| One-line official strip | **Full official card block** (three notice cards at the top) |
| Sticky station rail, freshness glyph and status line | Rail plus a small map, roughly to spec |
| Freshness card with a 120-day sampling strip as the most prominent element | **Missing.** Only a freshness badge |
| Five selector cards with in-period previews; stale values never at headline size | Five cards without previews. The stale-value rule is partly met ("Not measured since…") |
| **One primary chart** (about 340 px; range buttons; "highest in view"; crosshair) | **Five medium charts stacked** (pDA, two Pseudo-nitzschia, chlorophyll, temperature) |
| Season heatmap since 2014 | **Missing** |
| Violet forecast band on the same time axis | Present (separate axis) |
| Methods as three columns plus disclosures | List plus disclosures, roughly to spec |

### 4.3 Fisheries: still largely the M3 structure

| Spec (§3.3) | Implemented |
|---|---|
| Hero (serif H1, lede, data-through note) | Small title, then two caveat boxes **before** the data |
| Three typographic figures, no card chrome | Three **KPI cards** |
| Stacked annual bars by group, share row, a 2015–16 bracket, legend as filter | **Single-colour bars**, one series |
| One table with sparkbars and a row detail panel | **Three small per-group charts** plus a plain values table |
| Port-level absence as one quiet line | **A full amber box at the top** |
| Methods as four columns at the end | Two cards |

### 4.4 System-wide

**Type system:** used inconsistently. The serif display size (44 px) appears only on the map's region title and the port name. Page H1s are 28–32 px sans.

**Not yet designed:**
- an icon set beyond the shield and map glyphs;
- an empty-state illustration style;
- loading skeletons;
- an error-state pattern.

**Colour:** semantic colours are applied consistently (amber official, violet model, teal measured).

## 5. Prioritized development plan

Each milestone gets its own branch and draft PR. Nothing merges or deploys without your approval.

| # | Milestone | Branch | Contents | Exit criteria |
|---|---|---|---|---|
| **M4.0** | **Regulatory correctness** | `fix/official-del-norte-razor-clam` (#16) | The records and disclosure (done). Then a person reviews; then the satellite `latest_valid_date` fix | #16 merged. Review recorded, so the strip changes to "reviewed on …" |
| **M4.1** | **P3 integration** | #15 | Merge after #16. Fold in the demo session's responsive fixes when it hands back the five map files (about 24 h) | Merged; production unchanged |
| **M5** | **Ocean Map, visual centrepiece** | `feat/coastwatch-p5-map-experience` | See §6 | Two documented visual review passes, at 1440, 1280 and 390, against the brief |
| **M6** | **Bloom Intelligence redesign** | `feat/coastwatch-p6-bloom` | Spec §3.2 in full: the freshness hero with a 120-day sampling strip; selector cards with previews; **one large primary chart** (range presets, gaps, the `0*` lane, sampling ticks, crosshair, highest in view); the season heatmap; the violet model band on the same axis; progressive disclosure of methods; a station picker sheet on mobile | Same review standard; every control works on real data |
| **M7** | **Fisheries redesign** | `feat/coastwatch-p7-fisheries` | Spec §3.3 in full, editorial: hero; typographic figures; stacked bars by group with the 2015–16 bracket and legend filter; table with sparkbars and a detail panel; the port-level absence as one line; methods at the end. A geographic panel of CDFW port areas with the official geometry, where data allow, with no invented port splits | Same |
| **M8** | **External review and user testing** | — | An independent HAB scientist reviews the [scientific checklist](08-scientific-review-checklist.md). The regulatory review is done by a person. Five-person moderated tests with fishers and harvesters (task-based: "Can I dig razor clams in Del Norte?") | Findings triaged and fixed |
| **M9** | **Full-site deployment** | — | A separate Vercel project (never `coastwatch-demo`), a production domain, monitoring, rollback plan | Your explicit approval |

**Reliability (parallel, small):** send the [NOAA outreach email](p3/noaa-outreach-draft.md). If NOAA recommends one, add the second HF-radar host.

## 6. Next milestone: M5, the Ocean Map as the centrepiece

**Brief:** the map fills the screen. Controls float, compact and quiet. Information arrives progressively.

**1. Layout**
- **Desktop:** full-bleed map. A slim left **control rail** for layer group, product and time, about 64 px collapsed and 320 px expanded. A **floating legend** in the lower left, sized to its content (about 280 × 90). The inspector is a right **sheet** that pushes the map's fit area, never covering the selected place.
- **Status:** one calm status line under the masthead. The official disclosure and data-failure notices merge into a single expandable line, never two banners.

**2. Mobile**
- **Designed sheets:** a bottom sheet with three snap points (peek with the legend and time; half with controls; full with the inspector), drag handles, and a map that is always at least 55 % visible at peek.
- **Navigation:** places as a full-screen search sheet. Layer group as a segmented control in the sheet's header.

**3. Basemap**
- a custom MapLibre style: bathymetric shading from public GEBCO or NOAA relief tiles, if licence and size allow, for a sense of the shelf and canyons such as Monterey Canyon;
- refined land tones and a coastline weight scale by zoom;
- a label hierarchy (ports, then towns, then roads);
- latitude-limit labels placed offshore with collision rules.

**4. Legends**
- **Compact:** a single-row ramp with three to five ticks, and a units tooltip.
- **Expandable:** class detail and caveats.
- **Readout:** pointing at the map shows the exact value and date on desktop (spec §3.1).

**5. Motion**
- **Panels:** 150–200 ms ease-out transitions for panels and sheets.
- **Layers:** a raster crossfade only when the layer changes, never between time steps of observed data, which would imply interpolation. Off under reduced motion.

**6. Inspector**
- **Model:** the model section with the 4-day min–max strip and the 30-day nowcast sparkline.
- **Measured nearby:** the nearest CalHABMAP station with its last sample and freshness, linking to Bloom.
- **Satellite and currents:** compact rows with their own dates.

**7. Visual QA**
- **Pass 1:** implement, screenshot at 1440, 1280 and 390 against production data, write down every shortcoming, fix.
- **Pass 2:** repeat with fresh eyes, comparing against the design-reset prototypes, and document where the result improves on them.
- **Report:** both passes go in the M5 report with before and after images.

**Unchanged:**
- every value comes from published data;
- no fabricated or decorative metrics;
- observation and model are distinguished everywhere;
- freshness on every layer;
- official notices first, with the unverified disclosure.

## 7. Unresolved blockers

1. **Official notices need a person.** No automated step can replace it, and the disclosure stays until it happens.
2. **NOAA reliability.** Dataset reloads ("unknown datasetID") and 403s to runners recur. The email draft is ready to send.
3. **Concurrent sessions in this repo.** A demo session works in parallel and owns five map files until about 24 h from now. M5 starts after that handoff, or works first on the basemap style and sheet components, which touch other files.
4. **Basemap data licence.** Bathymetry tiles need a licence and size check before M5's basemap work.
