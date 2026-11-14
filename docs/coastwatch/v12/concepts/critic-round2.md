# Neritic concepts: critic round 2

Method: I re-rendered all three with `shoot.mjs` and viewed every page at desktop and 390 px, plus 320 px and landscape samples. In Chrome with WebGL I also:
- captured opener frames at 0.3 to 6.5 s;
- tested each Explore with a click, a mouse-wheel zoom, a drag and a phone tap;
- changed catch to 400 lb in each ledger;
- dragged each Research brush.

All three changed substantially. The first-frame problem is fixed everywhere: every opener shows its statement by about 0.3 to 0.9 s, and the animation plays behind it. No page threw a JS error.

---

## A. Chart

**Verdict: READY TO SHOW for selection, after the two phone fixes listed under blockers.** Best: Today. Weakest: Explore on phone.

**What improved**
- **Opener:** the headline is up at 0.3 s and the contours draw in beneath it. It is calm, fast and says what the product is.
- **Today** is now the best single screen in the round. The status sits directly on open water in large purple serif, with the next assessment below it. Two data labels only. Secondary facts (sea, tide, algae, models) moved to a quiet expandable rail. The card is gone and the chart reads as a chart.
- **Home** has no box. The headline is set in open water. The data reads "Crab toxin 9.8 of 30 ppm, Monterey Bay test area", with no false pin, and "Waves 6.6 ft at 14 s".
- **Neatline** is thin, and the magenta band is labelled in place ("Whale zone 4: commercial Dungeness delayed, not verified").
- **My Coast:**
  - the waterfall has connectors;
  - the sensitivity grid is now a plain table;
  - editing catch to 400 lb updated every figure and the grid instantly.
- **Research** adds a dated log beside the chart, which is the right pairing. The brush re-scaled the chart and the log (to 4 entries in June 2019) with no lag.
- **Explore:**
  - pan and zoom stay registered;
  - layers are now a list (Nov 9 9 PM, 24 h mean, C-HARM Nov 5 and Nov 2);
  - the probe opens in the rail instead of over the map.

**What still fails**
- **Explore, phone (blocking).** Tapping the water draws the transect, but the readout renders below the legend and layer list, off screen. On a phone nothing appears to happen. Fix: open the readout as a bottom sheet over the timeline, at least 40% of screen height.
- **Label clipping (blocking).**
  - Today at 320 px: the crab label is cut at the map's left edge ("ungeness", "9.8").
  - Home at 390 px: "Buoy 46236, 7:00 AM PT" loses "PT" at the bottom edge.
  - Research on phone: "Monterey" is cut at the bottom.
  - Fix: keep labels inside the frame with a collision pass, or anchor them inward on narrow frames.
- **Explore currents** are now too sparse at rest: two dozen short dark dashes, so the bay-wide flow and the eddy no longer read. Fix: roughly double the density and lengthen the trails slightly. Keep the colour; it is right.
- **Home, phone:** about 80 px of dead space between the map and the bottom bar. Fix: let the map fill it.
- **Research, phone:** the four tabs wrap onto two uneven lines. Fix: make them a horizontally scrollable segment control.
- **Today:** the purple dashed rule under the status ("T T T") is an unexplained symbol. Fix: label it as the zone-boundary symbol, or use a plain rule.

**Truthfulness**
- **Explore** gives the same current time twice in different clocks: "05:00 UTC Nov 10" in the legend and "Nov 9, 9 PM PT" in the list. A reader sees two times. Fix: Pacific time everywhere. UTC only in Research and exports.
- **Today's tide row** ("High 5.8 ft, 9:54 AM, prediction") does not name the station. Fix: "NOAA Monterey, 24 km", as C does.
- **Home** shows "7:00 AM PT" with no date on a page that may be seen days later. Fix: "Tue Nov 10, 7:00 AM".

---

## B. Canyon

**Verdict: REVISE (showable as the immersive direction once the blockers are fixed).** Best: Home. Weakest: My Coast.

**What improved**
- **Palette** is transformed: sand land, pale sky, bright shelf water. It no longer reads dark. Home is now the most striking first screen in the set: statement on clear sky above, terrain rising below, "Looking up Monterey Canyon toward Moss Landing" labelled.
- **Explore:**
  - registration is fixed (drag moves the chlorophyll, streaks and probe together);
  - chlorophyll is clean and saturated;
  - the debug-box outline is gone;
  - the probe gives value, kind, date, source and a transect.
- **Phone** gets a proper bottom bar, and the depth rail is gone on phone.
- **Research** is a new idea: a vertical "sediment core" of years, which you drag to filter four linked strips. Dragging the top handle moved the window to 2020 to now and updated the counts (330 samples, highest 15 ng/mL). It is genuinely different and fits the concept.
- **Ledger:** sliders and number fields stay in sync, and the waterfall updates live.

**What still fails**
- **My Coast (blocking).** Season, Whale measures and Crab test are stepped right like a staircase (left edges at 152, 262 and 372 px) over terrain, with "2 of 2 official notices above: not verified" floating loose. It reads as a broken layout. Fix: one aligned table on a single panel.
- **Explore, zoom (blocking).** One wheel notch jumped from the whole bay to a featureless blue pixel field: no coastline, no labels, no scale. You cannot tell where you are. Fix: about 1.5x per step, a scale bar, and a small locator inset.
- **Explore, phone (blocking).** The availability timeline drops its row labels: unnamed yellow bars, a lone yellow square, and "Nov 2" and "Nov 5" boxes. Fix: keep a compact label column, as C does.
- **Terrain:** steep canyon walls still show vertical smearing and curtain streaks (opener end state, Home lower left). Fix: triplanar texturing on steep faces, or darken and soften slope faces.
- **Research, phone:** the five core columns are about 70 px wide each and illegible. On desktop the domoic and Pseudo-nitzschia strips are dominated by grey stems back to the axis. Fix: on phone show one variable at a time with a switcher; on desktop drop the stems to dots.
- **320 px:** the header truncates to "Commercial Dunger". Fix: drop the fishery line under 360 px.
- **Robustness:** still WebGL-only for its identity. Before the owner falls for it, make sure the static fallback is as good as these frames.

**Truthfulness:** none blocking. Dates, kinds and sources are present in the probe and legends, and the time is stated once in both clocks ("Nov 9, 9 PM PT (05:00 UTC Nov 10)"). The camera flight is labelled as a view with exaggerated depth, so it implies nothing measured.

---

## C. Tide

**Verdict: READY TO SHOW for selection, after the tide fix, but warn the owner it no longer meets "immersive".** Best: Today. Weakest: Home.

**What improved**
- **Today** is the clearest page in the set:
  - status first ("Delayed to at least Jan 1, 2027");
  - the event timeline;
  - a dated list of what changes next, every row carrying its source and a mark (hatched for predicted, solid for measured, open circle for official and unverified).
  - This is the single mark system I asked for, and it works.
- **Research** is the best Research: four aligned small multiples sharing one brush. Dragging the window to 2015 to 2023 moved every strip and the count updated (371 samples).
- **Explore:**
  - the six-row availability timeline (chlorophyll, currents, C-HARM, outlook, seas, tide) is the most honest picture of what is known when;
  - the phone tap opens a readable inline sheet with value, date and source;
  - pan works.
- **Nav** is no longer pinned to the tide curve. The Nov 10 date hero is gone, the halftone map is labelled, and the giant-integer numerals are fixed.

**What still fails**
- **Identity.** Moving from cream and orange to light grey and one blue removed the generic-poster look, and most of the character with it. Outside Explore there is no colour of the sea anywhere. It now reads as a well-made government dashboard. Fix: bring the Explore ramp into Today and Home (a tinted water band on the halftone map, a sea-coloured timeline "now" block).
- **Home headline** leads with "Dec 15–18: the next whale assessment…" while the status is in the second paragraph. A stranger learns the date before the answer. Fix: "Crab season delayed to at least Jan 1. Next assessment Dec 15–18."
- **Explore, labels and readability:**
  - "Monterey" is clipped at the map's bottom edge and "Santa Cruz" collides with the top edge;
  - the chlorophyll row's "13% 17% 47% 23%" (statewide share by day) is unexplained;
  - the timeline takes a third of the screen.
  - Fix: inset the labels, say "share of the coast seen clear", and let the timeline collapse to three rows.
- **Landscape (844×390):** the tide strip flattens to almost zero amplitude, so the tide looks flat. Fix: a minimum strip height, or hide the strip below 450 px tall.
- **Home, phone:** halftone map labels are about 8 pt. Fix: scale the label type up on small screens.

**Truthfulness (blocking)**
- C's own key says solid means measured and hatched means predicted. The tide strip draws the past portion of the tide in solid black, but the tide is labelled "a prediction" on every page (Today, My Coast, every strip). The landscape My Coast also shows a step at "now" where the solid block and the curve disagree. Fix: draw the whole predicted tide hatched, and show "now" with only a line. If observed water level is wanted, plot it from the gauge and label it as measured.

---

## Cross-concept

**Different structures, or reskins?** Clearly more different than round 1, but still one product spec.
- They now diverge where it matters:
  - A: status set on the chart, with layers as a list;
  - B: terrain as a stage with a core-sample Research;
  - C: an event timeline and dated list, an availability matrix, small-multiple Research.
- They still share the same six pages, the same "Find your port / See the water / Plan a trip / Check the record" hero, the same probe-with-transect, and an identical ledger (inputs, waterfall, sensitivity grid) in all three.
- That is fine for selection. The owner will now be choosing a character, not a costume.

**Highest long-term ceiling: still A.**
- The chart scales to every port, works without WebGL, and is the only concept where the place itself carries the status.
- It now also has the calmest, most legible Today.
- C is the best source of components: Today's dated list, the mark system, the availability timeline and the Research small multiples should be grafted into A whichever concept is chosen.
- B has the most emotional pull, but the lowest floor (WebGL, phones at sea) and depends on a canyon most ports don't have.

**Blocking defects before the owner sees them**
1. C: past tide drawn solid (= measured under C's own key) while labelled a prediction, on every page with the tide strip.
2. A: tapping the water in Explore on a phone shows its readout off screen.
3. B: My Coast's staggered cards look broken.
4. B: one zoom notch in Explore loses the coast, labels and scale.
5. B: the phone Explore timeline has no row labels.
6. Label clipping:
   - A Today at 320 px (crab label);
   - A Home phone (buoy time);
   - A Research phone ("Monterey");
   - C Explore ("Monterey", "Santa Cruz");
   - B header at 320 px.
7. A: Explore shows the same current time in UTC and PT side by side; pick Pacific time.
8. C: tide strip flattens in landscape.
