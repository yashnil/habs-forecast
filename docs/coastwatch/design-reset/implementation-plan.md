# CoastWatch design reset: implementation plan

Status: **approved 2026-10-09 (PR #7 review). P0 authorized; P1–P4 follow after the P0 report.**
Design revision 2 incorporates the review refinements. The pipeline palette change (P1.1) still needs
its own PR and staging run.

## Principles for the rebuild

- **Presentation only.** `lib/` helpers (data loading, Ajv validation, freshness, `observations.ts`,
  `fisheries.ts`) and the published schema stay as they are. Components consume the same computed values.
- **Keep the safety net.** Every existing `data-testid` used by vitest, Playwright and the published-data
  tests is carried over or renamed in the same PR as its test. The `assertNoUnsafeClaims`, reading-order,
  overflow and axe checks must pass at every step.
- **One page per PR**, each releasable on its own behind the shared shell, so the validated M1–M3
  system is never broken between steps.

## Phases (priority order)

### P0 — Foundations (shared by all pages)

1. **Tokens:** port `prototype/tokens.css` into the Tailwind 4 `@theme` (colours, type scale, radii,
   shadows), replacing the current navy-everywhere palette. Add a unit test that the forecast class hexes
   match the pipeline palette.
2. **Fonts:** Newsreader, IBM Plex Sans and IBM Plex Mono via `next/font/google` (self-hosted at build, no
   runtime Google request). Turn on `tnum`/`lnum` globally.
3. **Shell:** new `Masthead` (nav without "My Coast — UPCOMING"), `OfficialPill`, `DataStatusLink`,
   mobile `TabBar` with Notices, and `AppShell` light and dark (map) variants. P0 restyles the existing
   pages inside the new shell. Their page layouts change in P1–P3.
4. **Official family:** `OfficialDrawer` (from the existing `Official.tsx` content), `OfficialStrip`,
   `OfficialBlock`, `OfficialSection`, `NoticeRow`, `VerificationLine`, using the existing registry
   types, relations and verification logic unchanged.
5. **Status primitives:** `FreshnessChip` (word + glyph + colour), `ProductChip`, `AgencyChip`,
   `Segmented`, `TabList`.

Exit: every current page renders inside the new shell; all existing tests green; axe clean.

### P1 — Ocean Map

1. **Forecast palette (one small, gated pipeline change).** Add `cw-probability-classes-v1` (10 stepped
   classes, hexes in `render_rasters.py`) to `pipeline/coastwatch_pipeline/process/palette.py` and make it
   the C-HARM palette. Only PNG colouring changes: value grids, port statistics and the schema are
   untouched, and `palette.id` already travels in the manifest so the legend follows. The palette
   docstring keeps the safety-rule comment (fixed domain, low class ≥ 2:1 against the sea). Verify with a
   staging run. *Alternative without a pipeline change:* colour the published u16 grid on the client in a
   canvas source. Possible, but heavier and duplicative; not recommended.
2. `NavCard` (official summary row, `PlaceSearch`, region list → port list → breadcrumb), `ForecastDock`
   (quantity tabs, day steps, stepped legend with "display steps, not risk levels" and the hatched
   no-value swatch), `ValueReadout` (pointer/tap, built on the existing `lib/grid.ts` `sample()`), the
   `PortInspector` order (official → model → measured nearby → satellite) shown only with a port, and
   measured panel padding for `fitBounds` with hand-set region views.
3. Basemap style updates in `lib/basemap.ts`: lighter land, coastline above the forecast, no-value
   hatching under an **opaque** raster, graticule, layer order, CalHABMAP station dots.
4. Mobile `BottomSheet` replacing `MobileSheet`: floating place button, quantity select beside the day
   steps, compact legend, condensed inspector.

Exit: the map e2e tests pass (official status, region ports, mobile legend), plus new tests for the
official-first order in the inspector, the inspector being absent until a port is selected, the dock
never overlapping the inspector at 1280 px, and legend swatches matching the palette.

### P2 — Bloom Intelligence

1. `StationRail` (grouped north to south, freshness glyph, pDA sub-line) and the mobile `StationPicker`.
2. `StationHeader` and `FreshnessCard` with `SamplingStrip`; historical banner.
3. `MeasurementSelector` with `SelectorCard` previews and the **old-value demotion rule** (new unit test:
   a value older than the stale limit never renders in the figure style); a swipeable row on mobile.
4. `ObservationChart` (taller, peak annotation, zero lane, visit ticks). Keyboard tooltip behaviour from
   M3 carries over. No small multiples.
5. `SeasonHeatmap` (weeks on desktop, months on mobile, collapsed by default on mobile).
6. `ModelBand` with a shared time axis, a hatched pre-history span and partial-coverage markers (a pure
   helper in `lib/observations.ts` with unit tests).
7. `AboutData` disclosures.

Exit: all `tests/e2e/m3.spec.ts` Bloom cases pass (selectors re-pointed where markup moved), and the
M2-compat and failed-update scenarios still render their explicit states.

### P3 — Fisheries & Economic Exposure

1. Hero, `ControlsBar`, `FigureRow` (the same computed totals the tiles use today).
2. `StackedBars` with a share-of-state row and the 2015–16 annotation sourced from `tier_basis`.
3. `BreakdownTable` with `SparkBars`, row selection and `GroupDetail` (official links, bivalve caveat).
4. `PortLevelNote`: one expandable line (reasons and port areas from the published data), plus the
   `NotLossNote` under the figures.
5. `MethodsGrid` and the `ValuesTable` disclosure (keeps `withheld-row`, `deflator`, `data-through`).

Exit: the fisheries e2e test (tiles match data; tier and real/nominal switches; port level unavailable)
passes unchanged in intent.

### P4 — Polish and verification

- Screen-reader pass (VoiceOver iOS, NVDA), focus order, reduced motion (no `fitBounds` animation).
- Screenshot set at 1440 / 1280 / 390 against fixture, M2-compat and published data. Compare with this
  design-reset set.
- Run the published-data Playwright suite against Pages. No deploy without separate approval.

## Rough effort

| Phase | Size |
|---|---|
| P0 foundations | 2–3 days |
| P1 map (incl. palette PR + staging run) | 3–4 days |
| P2 bloom | 3–4 days |
| P3 fisheries | 2 days |
| P4 polish | 2 days |

## Decisions (recorded 2026-10-09)

1. Hybrid theme (light editorial pages, dark map): **approved**.
2. Banded forecast colours: **approved**, as display intervals rather than risk categories, with exact
   values and source thresholds preserved.
3. Newsreader / IBM Plex Sans / IBM Plex Mono, self-hosted: **approved**.
4. Remove "My Coast" from navigation: **approved**.
5. "Monitoring stations" map toggle: **deferred** until after the primary redesign.

## Separate finding (not part of this design work)

While building the data snapshot I noticed that the published fisheries artifact's `withheld[].note` still
says the withheld row is "not included in group or statewide values". Since the M3 integration fix, the
statewide total **does** include it (the `method.withheld` and `method.statewide_total` text is correct,
and so is the UI). The stale sentence is at `pipeline/coastwatch_pipeline/sources/fisheries.py:269`. The
web app does not display it, so no user-facing number is wrong, but anyone reading the JSON gets
contradictory text. It is a one-line fix. I left it alone because this task excludes pipeline changes.
