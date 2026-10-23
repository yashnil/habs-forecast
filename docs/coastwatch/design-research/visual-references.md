# v10 visual references

Status: research for the v10 redesign concepts (see [concepts.md](concepts.md)), 2026-10-22.
Scope: visual language only. Copernicus Marine, Global Fishing Watch, Observable and NN/g are covered
by [decision-ux-references.md](decision-ux-references.md) and are not repeated here.

**How this was checked.** Each source below was fetched on 2026-10-22. Several are JavaScript or canvas
apps that do not render to a text fetch. Where that happened it says so, and the claims rest on the
product's own about/help pages, its terms, a design award citation or (marked) community or press
sources. Nothing below describes behaviour we could not source. The "before" screenshots are the
v10 build on staging data, captured 2026-10-22 at 1440×900 and 390×844, in
`scratchpad/v10-before/` (file names are given in each section).

---

## 1. Earth Nullschool — the field is the interface

Sources: <https://earth.nullschool.net/> (canvas app; the fetch returned menu labels only, so the
visuals could not be inspected), <https://earth.nullschool.net/about.html> (rendered),
<https://github.com/cambecc/earth> (rendered; MIT).

1. **Exceptional:** one full-bleed globe, with every control in a single "Show/Hide Menu". There are
   modes (Air, Ocean, Chemistry, Particulates, Space Weather, Biology), animated ocean currents, 9
   projections and a choice of bilinear or nearest interpolation. Keyboard: j/k steps time, n jumps to
   latest, p toggles animation (about page).
2. **Why it feels polished:** the chrome-to-content ratio is close to zero. The repo describes the
   structure: an SVG map with two canvases above it, one for the animated overlay and one for the
   coloured scalar field. One field, one flow and one hidden menu. The about page names each layer's
   provider and update cadence and says plainly that it "implies no guarantee of accuracy".
3. **Where Neritic is worse:** `explore-chl-d.png`. The field is strong, but a 1,000 px bottom dock
   holds three boxed panels (layer card, legend card, time card) of 11–13 px text. Together with the
   top pill bar, the search pill and the right-hand control column, the chrome takes about 30% of the
   viewport, and the eye lands on chrome before water.
4. **Principle to adapt:** *let the field be the interface.* One scalar field at a time, the layer
   index as quiet text, and the time control as a single line. Readings appear only where the user
   asks (a probe), never in permanent panels. (Concept A; B's plate.)
5. **Do not copy:** the particle-globe look, the projection gallery or bilinear advection. Bilinear
   advection violates our no-interpolation rule (integrity-guardrails §1.6). The code is MIT, but the
   live site is © Nullschool Technologies with licensing by inquiry, so its look and trade dress are
   not ours.

## 2. Windy — one map, one time scrubber, one point probe

Sources: <https://www.windy.com/> (did not render: only the title, logo and attribution came back).
No official help page for the timeline or picker came up, so the behaviour below is from Windy's own
community forum (<https://community.windy.com/topic/25149/how-to-display-the-timeline>,
<https://community.windy.com/topic/27232/windy-39-is-here-learn-how-to-masterclass-new-features/?page=1>).
Terms: <https://account.windy.com/agreements/windy-terms-of-use>.

1. **Exceptional (per forum posts):** a play button runs the forecast on the current layer along a
   bottom timeline. A draggable picker opens a detailed forecast for that point and updates as it
   moves.
2. **Why it feels polished:** three objects carry the whole product: the map, the time bar and the
   probe. Users report that the timeline "disappears randomly" (same thread), and the lesson we take
   is to keep time permanently visible.
3. **Where Neritic is worse:** `explore-chl-d.png` and `explore-currents-d.png`. Time is a box of
   day buttons ("Wed 14 / Thu 15 / Fri 16 / Latest clear"). It does not say how products relate in
   time: observed days, model leads, an outlook window and "now" are never drawn on one axis. There is
   no point probe on the map until a click opens another panel.
4. **Principle to adapt:** *a permanent time rail with a hard "now" boundary, and a probe that turns
   the map into numbers in place.* Each product sits on the rail as what it is: observed days as
   cells, model leads as cells from one run, the outlook as a single bracket rather than days.
   (Concepts A, B and C all carry this.)
5. **Do not copy:** Windy's colour ramps, logo, layout or widgets. The Terms give no brand rights
   (§2.5), prohibit copying the Services and automated collection (§7.6), require the Windy logo on
   visualisations (§9.4) and bar weather apps from using the widgets (§9.6). No embedding and no
   scraping. Its smooth interpolated playback is also against our rules: our steps cut.

## 3. Apple Weather (iOS) — modules that reorder by relevance

Sources: Apple Newsroom, iOS 15
(<https://www.apple.com/newsroom/2021/06/ios-15-brings-powerful-new-features-to-stay-connected-focus-explore-and-more/>,
rendered). Support guide: table of contents only. iOS 16 details come from press
(<https://mjtsai.com/blog/?p=32802>, <https://macrumors.com/2022/06/08/ios-16-weather-app-update>).

1. **Exceptional:** "dynamic layouts that change based on conditions", "full-screen maps" and "more
   graphical displays of weather data" (Apple). Press reports that rain moves the radar and hourly
   modules to the top, and that tapping a module opens a graph view.
2. **Why it feels polished:** each module carries one number at display size, with its label and
   context in a second, quieter voice. Hierarchy comes from size and order, not from boxes and badges.
3. **Where Neritic is worse:** `coast-sc-salmon-d.png`. The page reads as a form: three dropdowns of
   the same weight, a conditions card in which "Waves · not measured nearby —" takes the same space as
   the forecast wind, and the salmon season sentence at the same size as the tide caption. The trip
   economics are fields with steppers before the reader knows why. The most important line (the
   season) is not visually the most important.
4. **Principle to adapt:** *order by consequence, and size by importance.* The season and the
   advisory come first, the measured or forecast numbers are big, provenance is small but always
   present, and missing things take less space than present ones. (Concept C's header row; A's
   instrument row.)
5. **Do not copy:** Apple trademarks, SF fonts or the iOS visual language on the web, animated sky
   backgrounds, or any good-day/bad-day iconography. NWS text is never turned into icons
   (integrity-guardrails §1.8).

## 4. Flighty — trusted domain vocabulary and status first

Sources: <https://flighty.com/> (rendered); Apple Design Award 2023, Interaction
(<https://developer.apple.com/design/awards/2023/>, rendered).

1. **Exceptional:** Apple cites "key information where you need it most" and "a look that mirrors
   time-honored airport design conventions". The site leads with short status phrases ("Gate
   changed", "Delayed 45m") and has gate, times and booking code "all at a glance".
2. **Why it feels polished:** it borrows a vocabulary people already trust (departure boards) and puts
   status before detail. Numerals are tabular and large, and labels are terse.
3. **Where Neritic is worse:** `coast-sc-salmon-d.png` and `research-d.png`. Our domain already has
   trusted vocabularies (the NWS coastal waters forecast, the tide table, the chart margin, the
   lab sheet), and the UI uses none of them. It speaks a generic web-form language: grey cards,
   dropdowns, a pill-shaped status bar.
4. **Principle to adapt:** *speak the sea's own documents.* A tide table set as a tide table, NWS text
   quoted in the forecaster's words, a chart with neatline and graticule, and a ledger for money.
   (Concept B is built on this; C's lanes use the time-table form.)
5. **Do not copy:** Flighty's branding or card designs. Neither source describes a "boarding pass"
   design, so we do not rely on that idea.

## 5. FT Visual Vocabulary — choose the chart by the question

Source: <https://github.com/Financial-Times/chart-doctor/tree/main/visual-vocabulary> (rendered;
© The Financial Times Ltd, all rights reserved).

1. **Exceptional:** it chooses the chart by purpose (Deviation, Correlation, Ranking, Distribution,
   Change over time, Magnitude, Spatial, Flow), with rules such as "columns must always start at 0" and
   a warning that readers will read relationships as causal.
2. **Why it feels polished:** the editorial graphics it supports use direct labels instead of legends,
   annotations in the plot, a single highlight colour and generous white space. (These traits are
   general FT and NYT practice; the repo documents the method rather than the house style.)
3. **Where Neritic is worse:** `research-d.png`. The chart is good science, with fixed decades, a
   threshold and a separate model panel. But it sits in a rounded card beside a 300 px station
   column. It has no title that says what it shows, no annotation of the year's events, and its key
   caveats are a grey paragraph under the plot.
4. **Principle to adapt:** *the chart is the page.* A headline that states the subject, annotations on
   the data, provenance as a footnote band, and the station list as an index rather than a sidebar.
   (Concept B's "Figure 1"; C's small multiples on a shared axis.)
5. **Do not copy:** the poster or any FT assets, and the FT salmon background colour. We reuse the
   method only.

## 6. PredictWind — forecasts turned into a departure decision

Source: <https://www.predictwind.com/features> (text rendered; images came back as alt text only, so
we make no claims about its colours or layout).

1. **Exceptional:** "Graphs and Tables to compare forecast data", switchable map models, and a
   Departure Planning tool that compares leaving on days 1–4 and marks an "Optimal Departure".
2. **Why it matters here:** it connects conditions to the trip decision on one time axis.
3. **Where Neritic is worse:** `coast-sc-salmon-d.png` (full page `full/coast-sc-salmon-d.png`). The
   marine forecast, tides, notices and trip economics are separate stacked sections, so nothing shows
   the trip against the forecast periods or the tide.
4. **Principle to adapt:** *put the user's trip window on the same axis as the forecast and the tide.*
   (Concept C's lanes and the worksheet strip.)
5. **Do not copy:** "Optimal Departure" or any score. Neritic must not rank days, score fishing or
   imply a go decision (integrity-guardrails §5 A10, C19). We keep only the alignment; the judgement
   stays with the skipper. We also do not copy proprietary model names or branding.

---

## Cross-cutting findings (what the before screenshots share)

- **Chrome over content.** Pills, cards and boxed panels at similar weights (`explore-*-d.png`), so
  nothing is clearly first.
- **No typographic hierarchy below the place name.** Most text is 11–14 px grey sans, and value
  numbers are barely larger than captions (`coast-sc-salmon-d.png`).
- **Time is fragmented.** Day buttons in Explore, a 24 h tide card in My Coast, and separate charts in
  Research, with no shared axis or "now" line anywhere.
- **Missing things take as much room as present ones.** "Waves · not measured nearby —" gets a full
  column.
- **The science guardrails are present but read as fine print.** They are correct, but given no
  typographic role. In all three concepts they become a designed voice: italic marginalia in B,
  coloured class labels in A, tags and the ledger in C.

## Sources that did not render

- earth.nullschool.net (canvas; menu labels only)
- windy.com (title, logo and attribution only); no official Windy help pages found
- Apple Weather support guide (table of contents only)
- PredictWind images (alt text only)
