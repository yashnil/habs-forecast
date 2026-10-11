# M6: Bloom Intelligence redesign — implementation plan

**Status:** plan only, on `feat/coastwatch-p6-bloom` (draft PR). No implementation has started.

**Baseline:** the approved design-reset specification, revision 2 ([design-spec.md §3.2](https://github.com/yashnil/habs-forecast/blob/archive/pr7-design-reset/docs/coastwatch/design-reset/design-spec.md#32-bloom-intelligence)), with:
- its conventions (§2 type and colour, §4 layout, §5.2–5.6 charts, §6 rules, §7 mobile);
- the reference prototype (`prototype/bloom.html`, `js/bloom.js`).

Where the spec and the prototype differ, the spec wins.

**Scope.** Spec §3.2 in full, as set in the roadmap ([doc 17 §5](17-roadmap-and-design-audit.md), row M6). This is a presentation redesign of the existing page:
- no new scientific model;
- no new data source or pipeline output;
- no Fisheries work (M7);
- no deployment.

## 1. Where the page is today

The current page is still the M3 structure ([audit, doc 17 §4.2](17-roadmap-and-design-audit.md); [screenshot](roadmap/audit-bloom-1440.png)):

| Spec §3.2 | Today | M6 work |
|---|---|---|
| 1. One-line official strip (nearest port's notices, verification, "All N notices") | Full card block of up to three notices | Replace it with `OfficialStrip`, reusing the M5 notice family |
| 2. Sticky 296 px station rail, grouped north to south; freshness glyph; one status line ("pDA 0.20 ng/mL", "pDA last measured Aug 2022") | Rail plus a small map, roughly to spec | Restyle to spec; keep the small map below the rail as a secondary aid |
| 3. Station header (eyebrow, serif name, mono coordinates, port distance, "Open on map") | Partly | Complete; link to `/?port=…&region=…` |
| 4. **Freshness card** with a **120-day sampling strip**; historical banner | **Missing** (badge only) | New `FreshnessCard` and `SamplingStrip` |
| 5. **Five selector cards with previews** on the chart's period and axis; stale values never at headline size | Five tiles without previews; stale rule partly met | New `MeasurementSelector`, `SelectorCard` and `MiniPreview` |
| 6. **One primary chart** (~340 px): range presets, fixed decade axes, breaks over 21 days, `0*` lane, visit ticks, "Highest in view", crosshair, summary line | Five medium charts stacked; **data-fitted log axes** | One `ObservationChart` rebuilt to §5.2 |
| 8. **Season heatmap** since 2014, decade bins, "not risk levels" | **Missing** | New `SeasonHeatmap` |
| 9. **Violet model band** on the same time axis, pre-history hatched, partial runs open | Present, on a separate axis | `ModelBand` aligned to the primary chart's x-scale |
| 10. About these data: three columns plus disclosures; legend qualifiers always visible | List plus disclosures | Restructure |
| Mobile (§7): station picker sheet, swipeable selector row, edge-to-edge chart, heatmap collapsed, model band open | Desktop structure, narrowed | Mobile-specific structure |

**One scientific correction is part of this work.** Today the charts fit their log axes to the data (`logDomain`), so a quiet year looks as busy as a bloom year. Spec §5.2 requires **fixed whole-decade axes**:

| Quantity | Axis |
|---|---|
| pDA | 10⁻⁴–10² ng/mL |
| Pseudo-nitzschia | 1–10⁷ cells/L |
| Chlorophyll | 10⁻²–10³ mg/m³ |
| Temperature | Linear, 5–30 °C |

M6 adopts the fixed axes.

## 2. Data: everything needed is already published

From the production observations file of run 38119843030:
- **Stations:** 17 CalHABMAP shore stations.
- **Quantities:** 7 (pDA, dDA, tDA, *Pseudo-nitzschia* seriata and delicatissima groups, extracted chlorophyll, temperature).
- **Per-sample values:** qualifiers (`reported_zero`, rejected, flagged), sampling visit times, and per-quantity summaries (latest value and date, 12-month count, cadence).
- **History length:** seven stations reach back to January 2014; the others start 2017–2023 (Morro Bay 2023).
- **Model history:** each station carries the C-HARM nowcast median within 15 km, up to 180 days (146 held before the 2026-10-11 outage).
  - That run published none, because NOAA was down.
  - PR #24 keeps the recent history through such outages.
  - Until it merges, the band must handle "history unavailable" well. It already does, and that state will be designed and tested.

No pipeline or schema change is planned. Every new visual is computed in `lib/` from these fields (spec §4.3: charts take already-computed series), so presentation cannot change scientific handling.

## 3. Components and files

New in `src/components/bloom/` unless noted:

| Component | Job | Notes |
|---|---|---|
| `OfficialStrip` (official/) | One line: notices related to the station's nearest port, "Not verified", "All N notices" | Reuses the M5 `NoticeCard` and drawer; never truncates titles (M5 rule) |
| `StationRail`, `StationRow` | Grouped list with glyph and status line; keyboard navigable | `toxinStatus()` already produces the lines |
| `StationHeader` | Eyebrow, serif name, mono coordinates, nearest port and distance, "Open on map" | |
| `FreshnessCard`, `SamplingStrip` | Latest sample in display type, freshness word and age, 120-day strip (one tick per visit, gap since last shaded, today marked), cadence and lab-latency sentence; top rule in the freshness colour; historical banner | Pacific calendar days (rule 6) |
| `MeasurementSelector`, `SelectorCard`, `MiniPreview` | Five cards (pDA, *P-n* seriata, *P-n* delicatissima, chlorophyll, temperature); preview on the chart's period and fixed axis; reported zeros as open dots on the baseline; "Not measured since …" with the value in small text when stale; "Never measured here" | Selection lives in the URL (`?var=`) |
| `ObservationChart` (charts/, rebuilt) | §5.2: fixed axes, 21-day breaks, `0*` lane, visit ticks, flagged-high open markers, "Highest in view", crosshair tooltip, range presets 3 mo / 1 yr / 3 yr / Since 2014, summary line | Keyboard: arrows and Escape (M3 behaviour) |
| `SeasonHeatmap` | Years × weeks (desktop) or × months (mobile); highest value per period; teal decade bins; separate swatches for reported-0-only and unsampled | Caption: "Decade bins, not risk levels" |
| `ModelBand` | Violet band, `MODEL, NOT A MEASUREMENT`, C-HARM v3.1 · NOAA chip, the same x-scale as the primary chart (capped at 12 months), pre-history hatched, partial runs open, "never compared numerically" | Variable switch for the three C-HARM quantities |
| `AboutData` | Three columns (what a value means, blanks and zeros, seawater is not seafood), then disclosures, source, licence, review-pending note | |
| `StationPickerSheet` (mobile) | Full-screen station list from a picker button | M5 sheet patterns |

**dDA and tDA** are not among the spec's five cards. They stay reachable as rows in the "What each quantity is" disclosure, with their latest values and dates, and as the pDA tooltip's companions. See decision D2.

**New `lib/` helpers**, all pure and unit-tested:
- `fixedDomain(variable)`;
- `samplingStrip(times, now, days = 120)`;
- `segments(samples, gapDays = 21)`;
- `zeroLane(samples)`;
- `highestInView`;
- `heatmapBins(samples, period)`;
- `previewSeries(samples, x0, x1)`;
- `modelOnAxis(history, x0, x1)`;
- `staleHeadline(summary, policy, now)`.

## 4. Order of work (commits)

1. **`lib/` helpers and fixed axes**, with unit tests. Nothing visual yet.
2. **Page skeleton:** the official strip, rail and header restyled to spec; `/bloom?station=&var=&range=` URL state.
3. **Freshness card and sampling strip.**
4. **Selector cards with previews**, replacing the five medium charts.
5. **The primary chart** rebuilt to §5.2, with range presets, crosshair and keyboard.
6. **Season heatmap.**
7. **Model band** on the shared axis.
8. **About these data**, and the mobile structure: picker sheet, swipeable selector, collapsed heatmap.
9. **Visual review passes 1 and 2**, with fixes and the report (`docs/coastwatch/22-m6-bloom-intelligence.md`).

Each step leaves the page working. The commits are squashed into logical units before the PR is ready.

## 5. Tests

**Unit (vitest).** Every helper in §3, including:
- the 21-day break;
- zeros never on the log axis;
- a stale value never returned as a headline;
- Pacific-day freshness at the date line;
- heatmap bins at decade edges;
- the model band never interpolating across missing days.

**Browser (Playwright, fixture data).** Updates `m3.spec.ts` and adds an `m6.spec.ts`:
- **Rule 1 order:** official before model and measurement (`compareDocumentPosition`).
- **Freshness:** the strip tick count equals the visits in 120 days.
- **Stale value:** Monterey Wharf pDA reads "Not measured since …".
- **Charts:** selector previews share the chart's axis; ranges change the x-domain only, never the y-domain.
- **Lanes and summary:** the `0*` lane is present when zeros exist; "Highest in view" matches the data; the summary line counts are correct.
- **Heatmap:** shows "not risk levels".
- **Model band:** shares the chart's x-scale and says "Model history unavailable" when empty.
- **Mobile:** the station picker sheet; no horizontal scroll at 390.
- **Accessibility:** keyboard chart navigation, and axe on desktop and phone.
- **Safety:** `assertNoUnsafeClaims` passes.

**Live data.** A run against the production dataset, with every control exercised on real stations:
- Santa Cruz Wharf (630 samples since 2014);
- Monterey Wharf (last pDA Aug 2022);
- Morro Bay (history from 2023, pDA never measured);
- Trinidad Pier (north coast);
- a station whose model history is unavailable.

## 6. Visual review standard (as M5)

**Passes:** two documented passes at 1440 × 900, 1280 × 720 and 390 × 844, on production data.

**Before/after pairs:**
- the station page at the top;
- the selector and chart;
- the heatmap;
- the model band;
- the phone page and the picker.

**Measurements:**
- **Phone page height.** Spec revision 2: about 2,770 px at 390, against 4,490 for M3. I'll record the actual figure.
- **Chart size:** the primary chart is at least 340 px tall on desktop.
- **Text size:** no text below 13 px apart from axes.

## 7. Scientific and regulatory acceptance (spec §6, unchanged)

- **Official notices:**
  - official first, one interaction from anywhere;
  - "Not verified" wherever notices appear;
  - registry titles quoted, never truncated.
- **Safety wording:**
  - no "safe", "open" or "all clear";
  - seawater measurements are not seafood safety, and no action levels are drawn on seawater data.
- **Measured versus model:** teal against violet, separate containers, never on one y-axis, never compared numerically.
- **Missing values:** missing, reported 0, rejected, not sampled and no model value each have their own treatment and legend entry.
- **Old values:** an old value never looks current. Freshness is computed in Pacific days from the dataset's dates.
- **Display bins:** heatmap bins are labelled as display intervals, not risk levels.
- **Review notice:** "Not yet reviewed by an independent HAB scientist" stays.
- **Attribution:** CalHABMAP via SCCOOS ERDDAP with the licence; C-HARM v3.1 · NOAA on the band.

## 8. Decisions for you

The recommended default for each is in bold.

- **D1. Station map.** The spec's rail has no map; today's page has a small one under the rail. **Keep it under the rail on desktop, hide it on phone.** The alternative is to remove it.
- **D2. dDA and tDA.** Not in the spec's five cards, and rarely measured. **Keep them in the quantities disclosure with their latest values.** The alternative is a sixth "More" card.
- **D3. Heatmap default quantity.** It follows the selected quantity (spec). **Follow the selection**, rather than fixing it to pDA.
- **D4. Model band window.** The spec caps it at 12 months, but only 180 days of history are fetched. **Show what exists and hatch the rest of the visible range.** Extending the history fetch to 365 days would be a small pipeline change; it is not in scope unless you ask.

## 9. Not in M6

- New models or forecasts.
- New data sources, or station markers on the Ocean Map (an M5 follow-up).
- Fisheries (M7).
- Web deployment (M9).
- Changes to the Research Preview demo.
