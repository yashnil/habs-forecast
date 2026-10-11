# M6: Bloom Intelligence

**Status:** built on `feat/coastwatch-p6-bloom` (PR #25), then refined in a finishing pass (§7). The web is not deployed. The Research Preview demo is unchanged; it does not include this page.

**Plan and design review:** [doc 21](21-m6-bloom-intelligence-plan.md); §0 is the critique written before coding.

## 1. Result

The page now reads as **one station, one timeline**, built on the approved design-reset spec (§3.2) and improving on it where the review found it weak.

**Before and after**, on production data:
- [1440 full page](m6/review/1440-santa-cruz-full.jpg)
- [1440 fold](m6/review/1440-santa-cruz-fold.jpg)
- [1280](m6/review/1280-santa-cruz-fold.jpg)
- [phone](m6/review/390-santa-cruz-full.jpg)
- [Monterey Wharf](m6/review/1440-monterey-wharf-fold.jpg) (old pDA)
- [Morro Bay](m6/review/1440-morro-bay-fold.jpg) (pDA never measured)

**After only:** [chart with the model band](m6/review/chart-and-model-band-staging.jpg), drawn on the staging dataset, which still holds 146 days of model history.

From top to bottom:

1. **Official notices strip.** The notices for the station's port, with registry titles in full, the shared verification badge and "All N notices". The text below it says that a missing notice does not mean an area is open or seafood is safe.
2. **Station rail** (desktop). North to south, a freshness glyph and one plain line per station, under a locator map that stays in place while the list scrolls. On a phone it is a full-screen station picker.
3. **Title block.** The station name in display serif, the latest sample date, freshness as a dot and a word, the **120-day sampling strip** (a tick per visit, the gap shaded, today marked), and the sampling cadence.
4. **Quantity index.** One hairline strip, not dashboard cards. Each quantity shows its latest value, date and freshness, plus a preview on the chart's own axis. A value older than the stale limit is never shown at headline size ("Not measured since Aug 2022" at Monterey Wharf).
5. **One primary chart** on **fixed whole-decade axes**. Reported zeros sit in their own lane and are counted in the summary ("49 sampling visits in view: 39 measured values, 10 reported 0"). Every visit is ticked. Lines break at gaps over 21 days. "Highest in view" is labelled with its date. There is a crosshair, keyboard reading, and one table of the values. If nothing was measured in the period, the chart says when the quantity was last measured and offers the whole record.
6. **The C-HARM model band directly beneath**, on the same time axis and crosshair, with its own 0–100 % axis. It is labelled "Model, not a measurement". Gaps and partial-coverage days are shown, and the empty state is explicit.
7. **Season heatmap.** The highest value per week (per month on a phone) since 2014, in decade bins that are labelled "not risk levels". Zero-only and unsampled periods have their own swatches, and the current week is marked. **Choosing a year charts that year.**
8. **About these data.** Three plain columns:
   - what a value is;
   - blanks, zeros and uncertainty (the laboratories publish no uncertainty or detection limit);
   - seawater is not seafood.

   Disclosures follow, including every quantity's latest value here (dDA and tDA, decision D2), quality checks, processing, caveats, and the citation and licence.

**Defaults follow the data; the link always wins:**
- The page opens on particulate domoic acid unless the station never measured it; Morro Bay opens on *Pseudo-nitzschia*, seriata group.
- It opens on the last 12 months, unless the quantity has nothing in them; Monterey Wharf opens on its whole pDA record, which ends in 2022.
- Quantity, range and year are kept in the link.

## 2. Measurements

| | M3 page | M6 |
|---|---|---|
| Phone page height (390 × 844, Santa Cruz) | 4,720 px | **2,889 px** (−39 %) |
| Desktop page height (1440, Santa Cruz) | 2,385 px | 2,492 px. The primary chart is 300 px of plot and the season heatmap is new; the four secondary charts are gone. |
| Charts drawn | 5 measurement charts and 1 model chart | **1** primary chart with its model band, plus a heatmap |
| Repeated "Current" pills above the fold | 6, and 13 "Historical" pills in the rail | 0 (glyphs, with the word once in the title block) |
| Text below 13 px (outside chart axes) | many | **Only** 12 px uppercase eyebrows and agency codes (allowed by spec §2.1), the shared 11 px verification badge (an M5 component), and MapLibre's attribution |
| Axes | fitted to the data in view | fixed whole decades (§5.2); the axis never changes with the range |

**Accessibility:**
- **axe:** no serious or critical violations on desktop or phone.
- **Contrast fix:** the freshness colours were darkened so that every pair reaches at least 5.2:1 on paper, white and the selected-column tint. The spec's original "current" green was 4.5:1 on paper.
- **Keyboard:** stations are buttons, the chart is read with the arrow keys, Home, End and Escape, and heatmap years are buttons.

## 3. Tests

| Suite | Result |
|---|---|
| Web unit (vitest) | 135 passed, including 17 new for `lib/bloom.ts` |
| Browser (Playwright, fixture data) | 128 passed. M3 Bloom tests moved to the new structure with no assertion dropped; 8 new in `m6.spec.ts` |
| Browser, CI mode (bundled Chromium), Bloom and M6 | 25 passed |
| Typecheck, lint, production build | Clean |

**What the new tests check against the fixture data:**
- sampling-strip ticks;
- the zero lane, with the count and every zero drawn;
- "Highest in view" value and date;
- the model band's alignment with the chart and its crosshair;
- the link state surviving a reload;
- heatmap year navigation;
- zero-only and unsampled swatches;
- data-led defaults, with the link overriding them;
- the phone structure, overflow and axe.

## 4. Scientific and regulatory rules (spec §6)

| Rule | How M6 meets it |
|---|---|
| 1. Official first | The strip precedes the station in reading order (tested), and all notices are one click away |
| 2. Verification never upgraded | The shared badge reads "Not verified"; nothing marks a record verified |
| 3. Agency wording quoted | Registry titles in full, never truncated |
| 4. No safety implication | "View on the Ocean Map" replaces "Open on…" (caught by the unsafe-claims check). The missing-notice and seafood sentences are kept on every width |
| 5. Observation vs model visual | Teal against violet, separate panels, separate axes, "never compared"; the time axis is shared, the y-axis never is |
| 6. Freshness in Pacific days | Computed with `ageInDays`, the M5 helper; the strip and heatmap use California dates (tested at 11:30 pm PT) |
| 7. Missing is not zero | Blank, reported 0, rejected and not sampled each have their own mark and legend entry |
| 8. Old values never look current | Selector headlines (tested on Monterey Wharf) |
| 10. Attribution | CalHABMAP via SCCOOS ERDDAP, with retrieval time, citation and licence; C-HARM v3.1, NOAA on the band |
| 11. Review state | "Not yet reviewed by an independent HAB scientist" |
| 12. Display bins are not risk levels | Heatmap legend and caption; no action levels drawn on seawater |

## 5. Limitations and notes

- **No model history in production today.** The 2026-10-11 refresh ran during a NOAA outage. PR #24 keeps the recent history through such outages; issue #26 tracks confirmation in production. The band's populated state was checked on the staging dataset.
- **Station map:** in the review the old map rendered almost black on first load at Santa Cruz Wharf but normally at Monterey Wharf. The cause was not pinned down. The rebuilt rail's map rendered correctly in all 10 review captures (5 views, 2 passes). It needs a watch, not a claim of a fix.
- **Heatmap weeks** count 7-day blocks from January 1 in California time. Week 53 is short.
- **The model band** shows the C-HARM median within 15 km and its cell count in the tooltip. No model uncertainty is published.
- **The web is not deployed** (M9). The Research Preview does not include this page.

## 6. Merge recommendation

See §7 for the finishing pass; the recommendation stands.


**Squash merge** PR #25 after your review. The branch has five commits:
- plan;
- design review;
- helpers;
- page and tests;
- this report.

Squashing gives `main` one well-described commit. Nothing in it changes the data, the pipeline or the demo.

## 7. Finishing pass (2026-10-11)

This was an independent, skeptical re-review of the page and of the whole site, run on production and staging data in a real browser. There were three documented passes: composition, detail, and final verification.

**Images:** [m6/finishing/](m6/finishing/). The before side is PR #25 as reviewed; the after side is this pass.
- [first screen at 1440](m6/finishing/1440-first-screen.jpg)
- [tablet](m6/finishing/1024-first-screen.jpg)
- [phone](m6/finishing/390-first-two-screens.jpg)
- [heatmap](m6/finishing/heatmap-santa-cruz.jpg) and [sparse heatmap](m6/finishing/heatmap-sparse-tomales.jpg)
- [many zeros](m6/finishing/chart-many-zeros-scripps.jpg)
- [values below the axis](m6/finishing/chart-below-axis-2016.jpg)
- [model band with crosshair (staging)](m6/finishing/model-band-staging-crosshair.jpg) and [model band in production](m6/finishing/model-band-production.jpg)
- [never measured](m6/finishing/chart-never-measured-morro.jpg)
- [official drawer](m6/finishing/official-drawer.jpg)
- [Sources](m6/finishing/sources-top.jpg)

### 7.1 Scientific and data-integrity findings

**Values below the axis were misplaced (fixed).**
- Five published pDA results lie below the spec's 10⁻⁴ ng/mL floor (Santa Cruz Wharf, 2016–2018, as low as 1.4 × 10⁻⁵).
- The chart clamped them silently onto the 0.0001 gridline, misplacing them by up to a decade.
- They are now open edge markers. The exact value is in the readout, a legend line counts them, and lines break at them.
- No published value exceeds an axis top; the highest is 16.2 ng/mL pDA in 2014. The same markers would cover that case. Tests use the real values.

**The model band's wording was not the producer's (fixed).** The band said "exceeds the model threshold" and called the Pseudo-nitzschia layer a "bloom". It now quotes each threshold verbatim from the manifest, as the Ocean Map does:
- "…exceeds 500 ng per litre";
- "…10,000 cells per litre";
- "…10 pg per cell".

**The empty-history message blamed NOAA in every case (fixed).** It now names the real reason: no model cells near the station, a failed retrieval, or nothing published.

**Below-limit values:** CalHABMAP publishes no detection or reporting limit and no measurement uncertainty. A reported 0 is shown as "not quantified, not absent", never as a nondetect at a known limit, and the page says this.

**Data checks against the production record:**
- reported zeros are 3,454 of 4,267 pDA results (81 %);
- no rejected or flagged values exist today, though both are handled;
- dDA was never measured at any station;
- no missing value is ever shown as zero.

**Issue #26.** A staging refresh from `main` ([38129067967](https://github.com/yashnil/habs-forecast/actions/runs/38129067967)) restored history at 17/17 stations and 22/22 ports. The outage path is covered by fixture tests. Production confirmation is still pending: the next scheduled production run had not happened, and production still carries the empty history from run 38119843030. The issue stays open.

### 7.2 Design changes

- **The chart is on the first screen at 1440 × 900.** The official strip is shorter: same rules, count, verification badge, full titles and the missing-notice sentence, but the text the global banner already says is removed. The title block and index are tighter.
- **A first-time visitor knows what the page is.** A one-sentence lede is built from the pier's own record, so Morro Bay says "Domoic acid itself is not measured here".
- **The quantity index has one baseline.** A fixed two-line name block aligns every value. The Pseudo-nitzschia and chlorophyll caveats move to the chart note for those quantities.
- **The heatmap no longer lets missing data look like low values.**
  - "Not sampled" is blank. It used to be a near-white fill almost identical to the lowest teal bin.
  - Zero-only weeks are a quiet warm tone with a small dot.
  - The ramp's lightest step is distinct from the paper, and log bins read 1k–10k.
  - A screen-reader table gives each year's highest value and date, and its counts of sampled and zero-only weeks.
- **Tablet (1024) uses the station picker,** as phones do. The rail appears from 1280, so the content keeps its width.
- **Shared masthead:** the official pill and "Data status" no longer wrap at mid widths. Screen readers still hear "official notices".
- **Detail fixes:**
  - "Pseudo-nitzschia" no longer breaks at its hyphen;
  - the coordinates line wraps cleanly;
  - the chart summary never breaks mid-phrase;
  - the readout sits below the "Highest in view" label;
  - zero rings shrink in long views;
  - "Model history starts" replaces "No model history kept before".

### 7.3 Interaction audit (real browser, production data)

20/20 checks pass:
- Ocean Map "Measured nearby" → the station page;
- station switching, Back/Forward, and deep links restoring station, quantity and range;
- keyboard-only quantity, range, chart reading and heatmap years, with visible focus;
- the drawer opening from the strip, with CDFW and CDPH links and Escape to close;
- phone tap to read, the picker opening at the selected station, picking a station;
- reduced motion: no animations;
- garbage URL parameters fall back cleanly.

**Bug found and fixed:** a tap on the chart did nothing on touch screens, because a lifted finger cleared the readout. It now has a regression test.

### 7.4 Performance (production data, local server, median of 3)

| | M3 page (`main`) | M6 as reviewed | **M6 now** |
|---|---|---|---|
| Phone, CPU slowed 4×: blocking time | 78 ms | 232 ms | **64 ms** |
| Phone: quantity switch | 183 ms (range switch) | 214 ms | **134 ms** |
| Phone: transfer | 1,037 KB | 1,052 KB | **373 KB** |
| DOM nodes | 1,237 | 2,234 | **1,133** |
| Desktop: quantity switch | 118 ms (range switch) | 82 ms | **67 ms** |
| Desktop: first contentful paint | 108–132 ms | 104–148 ms | **104 ms** |

**Causes, found by CPU profiling:**
- **The hidden desktop rail started MapLibre on phones** (about 340 ms; 680 KB downloaded). The map now mounts only where the rail is shown.
- **`pacificToday` built a new `Intl.DateTimeFormat` on every call,** once per sample when binning. It is now built once; this also benefits the Ocean Map.
- **The heatmap was about 690 elements with their own handlers.** It is now one SVG with a path per colour and a single pointer handler.

### 7.5 Tests

| Suite | Result |
|---|---|
| Pipeline (pytest) | 186 passed, on this branch's base. CI tests the merge with `main` |
| Web unit | 136 passed. New: values beyond the axis break the line |
| Browser, local Chrome | 129 passed |
| Browser, CI mode (bundled Chromium) | 129 passed |
| Typecheck, lint, production build | Clean, 0 build warnings |

**New regression coverage:**
- below-axis markers on the real 2016 record;
- verbatim C-HARM thresholds when switching model quantity;
- heatmap hover descriptions and the zero-only and binned cell counts;
- tap to read on phones.

No test was removed or weakened. Two assertions were updated to the new intended behaviour, each with a comment:
- a 300 ng/mL synthetic point is now an edge marker;
- heatmap cells are counted on the SVG.

### 7.6 Site-wide assessment

**Ocean Map (M5)** is the strongest page: immersive, calm, and clear about model, observation and official notices. The map and Bloom now link both ways: "Measured nearby" to a station, and "View on the Ocean Map" to the port.

**Bloom Intelligence (M6)** reads as one exploration: station, freshness, one chart, the model on the same timeline, the season since 2014.

**Fisheries** is still the M3 structure: two caveat boxes before the data, KPI cards, and repeated small bar charts. It is the weakest page (M7).

**Sources** now leads with the official notices, then the research, then source status. Raw upstream errors sit behind "Technical detail" under a plain sentence that matches the outcome ("could not reach" or "part … did not complete").

**User journeys:**

| Journey | Assessment |
|---|---|
| 1. HAB conditions near Monterey | Map → Moss Landing inspector → "Measured nearby" → station. Works. Friction: the C-HARM failure banner and three amber surfaces compete on first load. |
| 2. Fisher checking notices | The masthead pill shows the count and verification state; one tap opens the drawer with agency links and hotlines. Clear. |
| 3. Researcher | Fixed axes, per-visit values table, heatmap, source links and citation. Missing: a data download (only links to ERDDAP). |
| 4. First-time visitor and the research | Improved: Sources shows the paper second. Still missing: a short "About CoastWatch" in the main navigation. |
| 5. Phone user | The map's sheet tabs and place button, and Bloom's picker, swipeable index and collapsed heatmap, all work. The Bloom chart is about 1.5 screens down on a phone. |

**Remaining UX and UI weaknesses, highest impact first:**

1. **Fisheries editorial redesign (M7).**
2. **One status line instead of stacked amber.** The global banner, the page strip and inspector notices repeat "not verified" across the site. This needs one shared pattern, designed with the M5 shell.
3. **"About CoastWatch" in the navigation:** what it is, who made it, the research, the limits, in about 150 words, with Sources linked.
4. **A phone chart summary near the top of Bloom:** the latest value and the highest of the year as a sentence above the index, so the answer comes before scrolling.
5. **Data download** of the values in view (CSV), with provenance in the file.
6. **A single 11 px exception:** the shared "Not verified" badge is still 11 px, set by the M5 component.
7. **The Ocean Map's C-HARM "failed update" line** shows on every load while NOAA is down. It is accurate, but it could be quieter once the data is more than a day old and clearly dated.

