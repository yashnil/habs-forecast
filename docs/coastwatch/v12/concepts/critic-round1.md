# Neritic concepts: critic round 1

Reviewed: every page at desktop and 390 px, a sample at 320 px and landscape. All three openers watched frame by frame in real Chrome with WebGL. All three Explores exercised (click, wheel, drag), plus each port picker. No brief or planning docs read.

**Nothing here is ready to show.** A is closest. B has the strongest single moment and the worst defects. C has the best information hierarchy and the worst front door.

**Applies to all three: the opener spends the whole five seconds.** At 3.5 s no concept has told a stranger what the product is. A shows a blank grid at 0.5 s, then contours, then fills, and its headline arrives between 3.5 and 5 s. B flies a camera past unlabelled terrain and its statement lands at about 5 s. C draws a tide, then squares, then a stray red bar, and its sentence lands at about 5 s. Fix for all: the one-sentence statement is on screen from the first frame and the animation plays behind or beside it. Run the opener once per visitor.

**Also all three: "Find your port" appears while a port is already chosen** (A and B display Moss Landing on the same screen). Fix: "You're looking at Moss Landing. Change port." Or show no port until one is picked.

---

## A. Chart

**Verdict: REVISE.** Best screen: Home. Worst screen: Today (desktop).

### What actually works
- The idea is right for this audience. A fisherman already reads charts, so the map needs no explanation, and the page stays bright and calm.
- Home's final state is the only hero of the nine I'd call beautiful. Layered blues against buff land, confident contours, curved "Monterey Canyon" lettering, a real scale bar.
- Numbers placed on the water ("9.8 ppm", "6.6 ft at 14 s") are the one idea across the three concepts that fuses data with place.
- Explore's zoom and drag work and everything stays registered. The probe with its transect line is a good, specific interaction.
- Phone Home puts the map first and the statement second, with a clear button.

### Specific failures
- **Today, composition and clutter.** 258 words on one 900 px screen with no scroll. Six separate captions sit on the chart at equal weight, and the map stops being a picture and becomes a label dump. Fix: the chart carries three marks at most (crab test, buoy, tide). Forecast and models move to a single quiet drawer.
- **Today, copy.** The "Models add nothing new today" block is about 60 words of small haloed type on land, describing an absence. Fix: delete it from Today. Keep a single line in Explore's timeline, which already says it.
- **Today, ring.** The 15 km tick-marked ring is the most graphically dominant object on the page and is the least important. Its caption is top right, on land, far from it. Fix: thin the ring to a hairline and put "0.69 mg/m³" on the ring itself, or drop it from Today.
- **Today, hierarchy.** Three competing display sizes in the card (place name, purple status, date). Purple serif status reads as a link or a brand tint, not a status. Fix: status largest, date second, "Moss Landing, today" reduced to a label.
- **Home, card.** The white card hides the top-left 45% of the chart, including most of the shelf break. It is a boxed hero laid over a picture. Fix: set the headline directly in open water on the pale offshore tint, with no box. The Pacific is empty on this chart and is asking for it.
- **Home, headline.** 18 words, a list of four nouns plus "charted on the water off your port". Fix: under 10 words, e.g. "Can you fish from your port? Status, tests and the next date."
- **Home, data.** "9.8 ppm" has no action level, so a stranger cannot tell if it is alarming. The word "toxin" is absent from the label. The pin also fixes a Monterey Bay sampling result at one spot in the canyon head, implying a location it doesn't have. "6.6 ft at 14 s" never says waves. Fix: "Crab toxin 9.8 of 30 ppm limit", placed as a bay-wide label with no point marker. "Waves 6.6 ft, 14 s".
- **Neatline.** The 8 px black-and-white zebra border vibrates at every size, dominates the page edge and wastes phone width. The unexplained magenta bar on the right edge reads as a rendering fault until you find the tiny tick and caption. Fix: thin the neatline to 2 to 3 px, and label the magenta band directly in the band.
- **Explore, currents.** The black lines look like hair or scratches on the map. They kink into angular polylines, and the eddy renders as a square spiral (clearest at phone and landscape). Fix: smooth the paths, taper them, and use dark teal rather than black at lower opacity.
- **Explore, artefacts.** White-and-hatched squares break the coast at Davenport and Santa Cruz. "mg/m³" is clipped at the end of the colour ramp on desktop, phone and landscape. Fix: dilate or mask the coast edge, and put the unit outside the ramp.
- **Explore, load state.** A probe is already open on an arbitrary point when the page loads, and its transect runs under the card. A stranger reads this as someone else's selection. Fix: start clean, with a "tap the water" hint that dismisses itself.
- **Explore, timeline.** The rows are not individually selectable (a single "Time stop" control). "05:00 UTC" is a fishing-unfriendly time. Fix: separate targets for each row, and Pacific time throughout.
- **My Coast.** It collapses into a form. Hatched cost bars float at unaligned positions and read as noise. The contour-plot of money is clever but costs legibility: haloed dollars on a gradient where a 3x3 table would be read in a second. Fix: real waterfall with connectors, as in C. Keep contours only if the figures sit on plain ground.
- **Research.** The strongest of the three Research charts, on a log scale with zeros kept in a separate strip. But "AUC 0.30, BSS −11.5" sits in the intro, above the thing you came to do. Fix: move it into the provenance block.
- **Type.** The serif headline with neutral sans is tasteful and anonymous. The haloed labels are thick and smeary at phone size ("Highest of 6 Dungeness, viscera" on contours). Fix: reduce to one short line per label on phone. No right-aligned ragged four-line captions.
- **Phone.** Today at 320 ends the fold on a tan rectangle and the top of a clipped "9.8". My Coast puts a large decorative map between status and the tool you came for, and at 320 it clips to a sliver. Fix: on phone use a short landscape crop of the water with the port centred, or drop the map from My Coast.
- **First five seconds (end state): passes.** It is the only one of the three where a stranger gets place, topic and action together.

### AI tells
Boxed hero card over image. Serif plus neutral sans "heritage" default. Disclaimer tics ("Algae, not toxin", "Gaps are no data, not calm"). Aphoristic section titles ("The record behind every number"). Less obvious than the other two, because the chart itself is specific.

### Five highest-value changes
1. Strip Today's chart to three labels and remove the ring and caption paragraphs.
2. Put Home's headline in open water, no card, under 10 words, and show it from the first frame.
3. Label the Home numbers in plain language with the action level, and no false point for the crab test.
4. Redraw Explore's currents: smooth, tapered, not black, with a clean load state.
5. Thin the neatline, label the magenta band, and rework phone map crops so a map is never a sliver.

---

## B. Canyon

**Verdict: REVISE, and do not show until Explore is fixed.** Best screen: Today (first screen, with its scroll descent). Worst screen: Explore.

### What actually works
- Scrolling Today really does move the camera down the canyon as the cards pass. It is the only truly immersive moment in the set, and the one thing the owner would remember.
- Today's first screen has clear hierarchy: status, then "Next assessment", then the date, on the calmest terrain in the concept.
- The opener's plunge is memorable, and the card text stays highly legible over the scene.
- My Coast's trip ledger with sliders is solid and readable.

### Specific failures
- **Explore, broken.** Wheel and drag move the streaks and the probe but **not** the chlorophyll or the land. After settling for 6 s the streaks run across the land, and the Moss Landing label has detached and floats inland. Fix before anything else: one transform for all layers.
- **Explore, colour.** The chlorophyll is overlaid and greyed by relief shading, giving a muddy slate-violet with no vitality. It is the least attractive rendering of the same data in the three concepts. Fix: drop the relief from water and use a clean, saturated ramp as in C.
- **Explore, outline.** The radar-coverage outline is a white, stair-stepped polygon that looks like a debug bounding box. Fix: remove it, or show coverage as a soft mask with no line.
- **3D rendering, across Home, Today, My Coast, opener.** Sawtooth and comb moiré on steep slopes. White contours break into scribbles and dashes. On phone Home the chlorophyll band renders as huge stair-step blocks. At close range (opener at 1.6 s) the surface looks like a low-poly mesh. Fix: higher mesh density near the camera, filtered normals, drape contours as anti-aliased texture. Limit camera closeness until the mesh can take it.
- **Palette.** Petrol teal plus putty-grey land plus near-black canyon is murky, and My Coast and Research are dominated by dark columns. This is the concept closest to breaching "no overwhelming dark background". Grey land reads as unfinished clay and the horizon dissolves into a flat teal block. Fix: brighter sunlit shelf water, warmer land, a real horizon or crop it out, and shift the camera to favour the shallows.
- **Home, composition.** Standard formula: full-bleed image, off-white card lower right, yellow button. The card hides the head of the canyon, the most interesting thing in the picture. Fix: rotate the camera so the canyon sits left and clear of the text, and lose the card in favour of type on calm water.
- **Cards.** Every panel is the same off-white box with the same grey-green bottom bar, on every page. On scroll the concept becomes floating cards over wallpaper, and the terrain carries no information. Fix: anchor content to places on the terrain, or admit it is a background and quiet it down.
- **Depth-gauge nav.** Charming once, then false: Research is not at "3,300 m" and the depth means nothing about the content. The "Research" label collides with the teal wedge on desktop. The rail costs about 17% of the width on phone, with rotated labels nobody wants to read while on a boat. Fix: keep the gauge as a scroll indicator on desktop only, use a plain bottom bar on phone.
- **Research.** Linear scale crushes nearly all the data into the baseline. The caption "…NOAA CRM, abo" is cut off on desktop. On phone the chart is tiny with illegible axis labels, and the tabs stack unevenly. Fix: log scale with a zero strip as in A, chart full width on phone, and remove the terrain strip here. It is decoration on the most analytic page.
- **Copy.** "What stands between you and the water. The call stays yours." is copywriting performing sincerity. "Depth exaggerated 4×. Not for navigation." appears on every page, even on phone above the art. Fix: delete the former, and put the disclaimer once in the footer or an info control.
- **Port picker.** Latitude and longitude next to each port name are noise, and "Avila/port San Luis" is miscapitalised. Fix: names only, north to south, as in C.
- **Performance risk.** Every page depends on WebGL for its identity. A phone with poor signal on a boat is the actual use case. There is a no-WebGL still in the shots folder, but it is not the experience being sold.
- **First five seconds: fails on the opener, partly passes after.** A stranger sees handsome terrain and could be looking at a game or a geology site until the card lands.

### AI tells
Full-bleed dark teal hero with card. Expanded grotesk headline with yellow hard-edged button (the stock neo-brutalist move). Uniform card with offset bar. Terrain as atmosphere. Condensed captions for small print on every surface.

### Five highest-value changes
1. Fix Explore's layer registration, and rebuild its colour without relief on the water.
2. Remove the sawtooth and scribble artefacts, or pull the camera back until they vanish.
3. Lighten the world: brighter shelf water, warmer land, far less near-black.
4. Replace the phone rail with a bottom bar, and show the statement from the first frame.
5. Break the uniform card, anchor at least Today's key facts to their places in the terrain.

---

## C. Tide

**Verdict: REVISE (reject the Home outright; keep Today and My Coast).** Best screen: My Coast. Worst screen: Home.

### What actually works
- My Coast is the most useful screen in the whole set: status in three tagged rows, tides and seas side by side, a true waterfall, break-evens, and a sensitivity grid that can be read at a glance.
- Today has the most decisive type in the set. The event timeline (Crab test, Delay notice, Last checked, Next assessment, Earliest season start) is the single best explanatory graphic across all three, because it shows what a working fisherman is waiting for.
- Explore has the best colour: a clear, saturated ramp on quiet land, with the coast glowing. Short comets register direction at a glance.
- The port drawer is the best of the three, north to south, as people on this coast think.
- Measured-versus-predicted as solid-versus-hatch is a real system, used consistently.

### Specific failures
- **Home, concept.** The largest thing on the page is today's date, which everyone knows and which says nothing about fishing. Fix: replace it with the answer. "Dungeness: delayed" or the next decision date, at this size and in this colour.
- **Home, image.** The halftone relief is an inkblot. No stranger will read it as Monterey Canyon, and there is no water, no colour, no labels. Half the page is a grunge poster. Fix: label port and canyon, tint the water, and use a recognisable coast outline.
- **Home, nav.** Today and the port selector are absent from the Home nav, so the one page you most want is missing.
- **Tide-strip nav, false encoding.** Nav items are pinned to points on a real tide curve, so "Explore" appears to mean Wednesday morning and "Research" Thursday evening. A person who reads tide charts for a living will try to interpret them. Hit areas are also uneven (Today is 242 px wide, Explore 78 px). Fix: separate the nav from the data. Keep the tide strip as an instrument, and label highs and lows only.
- **Tide strip, cost.** About 20% of the phone viewport goes to header, tide and tabs on every page, including Research, where tide is irrelevant. It also says Monterey while the page says Moss Landing, without explanation. Fix: show it on Today and My Coast only, collapse on scroll.
- **Today, hierarchy.** "Dec 15–18" at 230 px is an answer without a question. You have to read downward to learn it is a whale assessment, and "Delayed", the thing that determines your week, is second. Fix: lead with "Delayed to at least Jan 1", keep the date as the immense second element, with its label attached above.
- **Today, length.** 767 words. 2,892 px on desktop and 5,146 px on phone. Halftone blocks mid-page repeat the same inkblot as filler, and one of them is captioned as not being a reading. C-HARM and outlook read as an appendix. Fix: stop at the seven-period forecast and crab test, and move the rest to Explore.
- **Numerals.** The giant-integer, small-decimal treatment makes "6.6" read as "6" and tide "5.8" read as "5". On Explore, "0.31" puts all its weight on the zero. For values people act on, this is a legibility hazard. Fix: one size for every digit.
- **"194 samples" on Research.** The biggest thing on an analytical page is a row count. Fix: remove it and give that space to the chart.
- **Grid lines.** Faint full-height column rules run behind everything, and on phone they cut through body text. Fix: drop them.
- **Explore, artefacts.** A green scribble and a black hairline float on land near Moss Landing (Elkhorn Slough pixels and a stray coast fragment, also on phone). Salt-and-pepper noise fills the offshore corner. Zooming pushes the Santa Cruz marker under the tide strip. The fourth timeline row sits hard against the bottom edge. Fix: mask inland pixels, smooth or fade noisy low-signal offshore pixels, clip markers, add bottom padding.
- **Explore, phone.** At 320 "C-HARM Nov 2" buttons wrap and touch their borders, and the "Moss Landing" halo label sits on the data. Fix: one "Model runs" control that opens the list.
- **Colour.** Cream, black and one orange is striking and has nothing to do with the sea. Outside Explore there is almost no blue or green. It is expressive but not immersive, which is half the ask. Fix: bring the Explore ramp into the system as the second colour.
- **Tone.** The aggression suits a concert poster. A fisherman looking at a delay on his income may read shouting. Fix: reserve the huge type for one element per page.
- **First five seconds: fails on Home, passes on Today only once you read past the date.**

### AI tells
The strongest of the three. Warm cream, single orange accent, condensed grotesk, giant numerals with tiny captions, ruled grid and hatch fills add up to the current default "editorial brutalist" output. Halftone used as texture rather than meaning. Black pill "Delayed" badge. The arrow-in-a-square button.

### Five highest-value changes
1. Rebuild Home around the answer and a legible, coloured coast. Remove "Nov 10".
2. Separate nav from the tide curve, and limit the strip to the pages where tide matters.
3. Lead Today with status, cut it to about a third of its length, and remove the filler halftones.
4. Set all numerals at one size, and remove "194".
5. Bring the sea's colour into the whole system, then clean the Explore artefacts.

---

## Cross-concept

**Genuinely different ideas, or reskins?** Three real visual worlds on one wireframe. The Home screens and nav metaphors differ. Beneath them, the products are nearly the same: identical page set and titles word for word ("The water, as last measured", "Plan a trip from Moss Landing", "The record behind every number"), the same four-link hero (Find your port, See the water, Plan a trip, Check the record), the same Explore layout (legend card, probe with transect, four-row availability strip), the same Research layout (small map, chart with brush, provenance list), and the same ledger. Only C's Today actually structures information differently. If the owner compares past the first screen, they will feel they are looking at one design in three costumes. A true second round should vary the structure too: for example, a port-as-a-single-page briefing, or a season calendar as the primary view.

**The whole set over-explains.** "Algae, not toxin", "A picture, not a measurement", "Not for navigation", "not verified" and source strings recur on nearly every surface. The underlying claims are necessary and must stay. The treatment should change: one consistent mark for measured, predicted, unverified and model, one legend per page, and the plain caveat stated once. As it stands, the honesty reads as caption clutter and is a large part of why all three still feel generated.

**Highest ceiling: A, the chart.**
- It's the only concept whose central idea scales to every port. Crescent City or Santa Barbara has no Monterey Canyon, so B has no story there and C's halftone turns to a featureless field. A flat shelf is still a good chart, and a series of port charts is a natural product: recognisable, printable, ownable.
- It is the only one that puts facts in their geographic place, so its metaphor does real work, and it can grow (zones, gear limits, closures as chart overlays) without a new visual language.
- It meets "no overwhelming dark background" with no effort and needs no WebGL on a boat.
- It is currently the least "captivating" of the three, and that is the risk. It needs motion and colour courage, not a change of idea.

**Recommended direction:** take A's chart as the base. Take C's status-first hierarchy, event timeline, waterfall and Explore colour ramp. Allow B's descent at most as a single, well-rendered, skippable moment. Do not carry forward A's zebra border, B's card-over-terrain structure, or C's date hero and nav-on-tide.
