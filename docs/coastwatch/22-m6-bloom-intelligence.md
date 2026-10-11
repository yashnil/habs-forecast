# M6: Bloom Intelligence

**Status:** built on `feat/coastwatch-p6-bloom` (draft PR #25). Not merged or deployed. The Research Preview demo is unchanged; it does not include this page.

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

**Squash merge** PR #25 after your review. The branch has five commits:
- plan;
- design review;
- helpers;
- page and tests;
- this report.

Squashing gives `main` one well-described commit. Nothing in it changes the data, the pipeline or the demo.
