# M6: Bloom Intelligence redesign — implementation plan

**Status:** approved with defaults D1–D4 (2026-10-11). The design review in §0 was done before coding and refines the spec where it falls short. Implementation is in progress on `feat/coastwatch-p6-bloom`.

**Baseline:** the approved design-reset specification, revision 2 ([design-spec.md §3.2](https://github.com/yashnil/habs-forecast/blob/archive/pr7-design-reset/docs/coastwatch/design-reset/design-spec.md#32-bloom-intelligence)), with:
- its conventions (§2 type and colour, §4 layout, §5.2–5.6 charts, §6 rules, §7 mobile);
- the reference prototype (`prototype/bloom.html`, `js/bloom.js`).

Where the spec and the prototype differ, the spec wins.

**Scope.** Spec §3.2 in full, as set in the roadmap ([doc 17 §5](17-roadmap-and-design-audit.md), row M6). This is a presentation redesign of the existing page:
- no new scientific model;
- no new data source or pipeline output;
- no Fisheries work (M7);
- no deployment.

## 0. Design review before coding

The owner asked for an exceptional redesign, not a literal implementation of the spec. Before coding, I reviewed the current page and the approved prototype on production data (run 38119843030), against two standards:
- the M5 Ocean Map's design language;
- practice in professional scientific visualisation.

Screenshots of today's page are in [m6/before/](m6/before/).

### 0.1 What today's page gets wrong

| Area | Problem | Evidence |
|---|---|---|
| **Hierarchy** | Nothing leads. The heaviest elements are, in order: a 270 px official block, a badge row, five equal cards, then five equal charts. The station name is the same size as a chart title. | [1440](m6/before/1440-santa-cruz-full.jpg) |
| **Repetition** | Four charts repeat the same legend, the same "Measured in 49 of 49 sampling visits" line and the same "Data table" control. "Current" appears 6 times above the fold. The rail repeats "Historical — not current" 13 times. | same |
| **Composition** | The model sits about 1,500 px below the measurements it is meant to be read beside, on its own axis, in a grey box. P-n delicatissima takes half a row to say it was never measured. | same |
| **Unreliable map** | On first load at Santa Cruz Wharf the station map rendered almost entirely black; at Monterey Wharf it rendered normally. The cause was not pinned down. | [1440, Santa Cruz](m6/before/1440-santa-cruz-full.jpg) |
| **Axes** | Log axes are fitted to the data in view. A quiet year and a bloom year look the same, and the axis jumps when the range changes. | `logDomain()` |
| **Zeros** | Most results are "reported 0": 341 of 621 at Santa Cruz, 617 of 657 at Scripps Pier. On a log axis they shrink to a faint row of rings, so a mostly-zero record reads as "almost no data". | data |
| **Copy** | Technical and defensive before it is useful: "newest sample in the network; each station shows its own", bullet walls of caveats, raw error strings. | — |
| **Phone** | 4,720 px tall. The official block and the station map come before anything about the station. | [390](m6/before/390-santa-cruz-full.jpg) |

### 0.2 Where the approved prototype still falls short

The prototype fixes most of 0.1. Four weaknesses remain, and M6 improves on them:

1. **Measured and model are far apart.** The spec puts the heatmap between the chart and the model band, so "same time axis" is never visible at once. *M6:* the model band sits **directly under the primary chart, on the same x-scale**, with a shared crosshair. It stays a separate violet panel with its own 0–100 % axis, labelled "Model, not a measurement", so the two are never mixed (rule 5).
2. **Dashboard cards.** Five bordered cards with big numbers still read as a KPI dashboard. *M6:* the selector becomes a **quantity index**: one hairline-ruled strip, a column per quantity (name, latest value, date with its freshness dot, preview on the chart's axis). The selection shows as a teal underline.
3. **A separate freshness card.** It competes with the station name. *M6:* the latest sample date and the 120-day sampling strip sit **in the title block**. The station name, when it was last sampled and how regularly then read as one statement.
4. **The heatmap is a picture, not a tool.** *M6:* it becomes the page's **navigation through time**:
   - hovering a week shows its highest value and date;
   - selecting a year sets the primary chart to that year (range "2015", for example);
   - the current week and year are marked.

### 0.3 Direction: one station, one timeline

The page reads top to bottom as one exploration of one pier:
- where it is;
- what was last measured, and how fresh that is;
- the record of the chosen quantity;
- what the model estimated over the same days;
- how this season compares with every year since 2014;
- what the numbers mean.

**Principles:**
- **One hero chart** at about 360 px. Everything else supports it, and range, quantity and year all drive it.
- **Fixed whole-decade axes** (§5.2).
- **The zero lane is first-class:** it is labelled with its count ("341 reported 0"). The summary line splits visits into measured values, reported 0 and not measured.
- **No decorative chrome:**
  - no badge row;
  - freshness is a dot and a word, once per item;
  - one data table, behind a disclosure, for the primary chart;
  - borders only where a container means something: official amber, model violet.
- **Plain copy, written for a harvester or a coastal manager.** Caveats sit next to what they qualify ("Toxin in seawater, not in seafood" under pDA). Methods come last, in three short columns.
- **Real data only, with provenance:** each panel credits its source (CalHABMAP via SCCOOS ERDDAP; C-HARM v3.1, NOAA), and the page shows when the data were retrieved.
- **Uncertainty is stated, not invented.**
  - The laboratories publish no measurement uncertainty or detection limit, and the page says so.
  - The model band shows a median over N cells, and days with partial coverage are open markers.
  - Nothing is smoothed, interpolated or averaged across samples.
- **Motion:** M5 timings, 150–200 ms for panel changes and none for data. Off under reduced motion.

**The station map stays (D1)** as a small locator above the rail, with the canvas resize fixed. It is hidden on the phone, where the station picker sheet replaces it.

**Data facts that shape the design**, from the production dataset:
- **Santa Cruz Wharf:** 630 samples since January 2014. The highest pDA in the last 12 months was 11.2 ng/mL on May 27, 2026; the highest since 2014 was 16.2 ng/mL on June 4, 2014.
- **Monterey Wharf:** pDA last measured Aug 2022.
- **Morro Bay:** pDA never measured.
- **Model history:** today's dataset carries none (NOAA outage; issue #26). The empty state is therefore a first-class design case, not an afterthought.

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
| `StationHeader` + `SamplingStrip` (title block, §0.2 item 3) | Latest sample in display type, freshness word and age, 120-day strip (one tick per visit, gap since last shaded, today marked), cadence and lab-latency sentence; top rule in the freshness colour; historical banner | Pacific calendar days (rule 6) |
| `QuantityIndex` (§0.2 item 2): a hairline strip, a column per quantity, with a preview | Five cards (pDA, *P-n* seriata, *P-n* delicatissima, chlorophyll, temperature); preview on the chart's period and fixed axis; reported zeros as open dots on the baseline; "Not measured since …" with the value in small text when stale; "Never measured here" | Selection lives in the URL (`?var=`) |
| `ObservationChart` (charts/, rebuilt) | §5.2: fixed axes, 21-day breaks, `0*` lane, visit ticks, flagged-high open markers, "Highest in view", crosshair tooltip, range presets 3 mo / 1 yr / 3 yr / Since 2014, summary line | Keyboard: arrows and Escape (M3 behaviour) |
| `SeasonHeatmap` (interactive, §0.2 item 4) | Years × weeks (desktop) or × months (mobile); highest value per period; teal decade bins; separate swatches for reported-0-only and unsampled | Caption: "Decade bins, not risk levels" |
| `ModelBand` (directly under the chart, shared crosshair, §0.2 item 1) | Violet band, `MODEL, NOT A MEASUREMENT`, C-HARM v3.1 · NOAA chip, the same x-scale as the primary chart (capped at 12 months), pre-history hatched, partial runs open, "never compared numerically" | Variable switch for the three C-HARM quantities |
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

## 4. Order of work

The whole milestone should come to about 7 substantive commits. Each is a tested unit; intermediate iterations are not committed.

1. The plan and design review (this document).
2. `lib/bloom.ts`: fixed axes, sampling strip, headlines, previews, segments, summary counts, heatmap, model band, year ranges, with unit tests.
3. The page rebuilt, as one coherent unit:
   - the title block with sampling strip;
   - the rail and the fixed locator map;
   - the quantity index;
   - the primary chart with the zero lane and crosshair;
   - the model band underneath.
4. The interactive season heatmap, "About these data", and the phone structure (picker sheet, swipeable index).
5. Browser and accessibility tests for the new page, with the M3 Bloom tests updated.
6. Visual review passes 1 and 2 with their fixes, measurements, and the report (`22-m6-bloom-intelligence.md`) with before/after images.

The PR is squash-merged unless you want the steps kept.

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

## 8. Decisions (approved 2026-10-11 with the defaults)

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
