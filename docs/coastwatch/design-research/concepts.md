# v10 design concepts: three directions

Status: concept stage, 2026-10-22. No app code changed. These are static HTML/CSS/SVG pages rendered
with Playwright. Every map is a real render from the running v10 app (staging data, manifest
`0bb19831d0be`), with the app's labels and chrome hidden and its paint properties adjusted for B. Every
number is real staging or live data captured 2026-10-22 at about 4:04 PM PDT, except the trip worksheet
inputs, which are a labelled **hypothetical** scenario.

- Renders (36): `scratchpad/concepts/renders/{a,b,c}-{explore,coast,econ-missing,econ-complete,research,loading}-{d,p}.png`
- Contact sheets: `scratchpad/concepts/renders/contact-{a,b,c}.png`
- Before vs. concepts (My Coast): `scratchpad/concepts/renders/comparison-my-coast.png`
- Source pages: `scratchpad/concepts/{a-instrument,b-atlas,c-mission}/`. Shared real data and chart
  builders: `scratchpad/concepts/common/common.js`. Map captures: `scratchpad/concepts/mapcap.mjs`.
- References: [visual-references.md](visual-references.md). Guardrails:
  [integrity-guardrails.md](integrity-guardrails.md).

### Real data used

- **Explore probe** at 36.900°N 122.060°W, read from the app:
  - C-HARM pDA > 500 ng/L: 82%, valid Oct 22 (day 3), run Oct 20 (inferred)
  - OLCI chlorophyll: 1.3 mg/m³, observed Oct 14
  - Outlook: 5.3 mg/m³, mean for Oct 20–27 (typically 2.3–12)
  - MUR SST: 18.6 °C, analysis for Oct 19
  - HF-radar 24 h mean: 0.10 m/s toward WNW 285°, to Oct 20 7 PM
- **Santa Cruz** (`/api/conditions`):
  - NWS PZZ535 periods, verbatim
  - Small craft advisory, 9 PM tonight through late tonight
  - Buoy 46236 (23 km): waves 4.3 ft at 11 s, water 18.3 °C at 3:30 PM
  - CO-OPS 9413745 tide predictions, Oct 22–25
- **Notices:** 3 tied to these waters, all from the staging registry: CDPH mussel quarantine; CDPH and
  CDFW anchovy, Pigeon Pt – Pt Lobos. None names salmon.
- **Evidence:**
  - Wharf *Pseudo-nitzschia*: 4,900 cells/L, Sep 30
  - Wharf pDA: 0.20 ng/mL, Sep 30
  - Satellite chlorophyll near the harbour: 2.6 mg/m³, Oct 13–14
- **Research series:** the real Santa Cruz Wharf pDA series, Oct 2025 – Sep 2026, 51 samples. The
  model panel is the real C-HARM daily history kept since Apr 23.
- **Economics:** EIA diesel at $8.242 is the real reference. Everything else in the worksheet is
  hypothetical: 36 nm, 8 kt, 4.5/2.5 gph, 9 h, ice $60, bait $90, insurance $45, 180 lb at $9.50,
  crew 30% after expenses. It is computed exactly as `src/lib/tripEconomics.ts` model A:
  - fuel: 42.75 gal → $352
  - gross: $1,710
  - crew: $362
  - net to boat: $800
  - break-even: 59.6 lb, or $3.15/lb

The **semantic colour families are kept in every concept**: official = amber, model = violet,
measured = teal, conditions = sky. They are re-tuned per background for contrast but never reassigned
(guardrails §1). Concepts differ in type, composition, ground and motion, not in what colours mean.

---

## A. Ocean Instrument

**Thesis.** Neritic as a scientific instrument pointed at the sea. The water fills the screen and the
UI is a thin layer of readouts, like a sonar or a lab console in a dark room. The field, the
bathymetry, the currents and time dominate, and words appear only where the reader points.

**Composition rules**
- Full-bleed field on every map screen. There are no panels around the map, only text on the field
  with a darkening gradient. Readings sit on one solid plate (rgba 0.86, no blur).
- A left "Field" index as plain text rows (name plus product class plus source/date); the active one
  is lit.
- A permanent time rail across the bottom:
  - teal cells for observed satellite days, the selected latest-clear day ringed
  - violet cells for C-HARM leads from one run
  - a dashed violet bracket for the outlook ("one 8-day mean")
  - a white "now" line with the future faintly hatched
  - steps cut, never fade
- The probe: a crosshair on the water with a leader line to a reading plate over land, so no data is
  covered. Every reading has a class label in its family colour, a 30 px mono value, and source and
  date.
- My Coast uses the real bathymetry and HF-radar arrows as a backdrop on the right, with the briefing
  set on the dark left 60%.

**Type and colour**
- Type: IBM Plex Sans (300–600), IBM Plex Sans Condensed for small caps labels, and IBM Plex Mono
  for every value. All are SIL OFL 1.1.
- Grounds: abyss #05090d and deep #0a1219. Ink #e6eef2 / #a3b3bd / #6e808b.
- Families on dark: amber #f0a94a, violet #b39cff, teal #3fd6bd, sky #74c7ff.

**Motion idea**
- The field never animates between days; steps cut. The only motion is instrument-like: the probe's
  crosshair settles in 120 ms, the time rail's selected cell is ringed, and loading draws the coast and
  relief first.
- If animated currents are ever reconsidered (v4 removed particles), they must follow §1.6: the
  nearest cell, death in gaps, 1–2 s lives, none under reduced motion, and a pause control. These
  concepts keep the observed arrows.

**The six screens**
1. **Explore:** the chlorophyll field fills the screen.
   - Desktop: the index on the left, the legend bottom-left (with "No observation… not low
     chlorophyll", "Estimated, not observed" and "Algae… not toxin"), and the probe plate on the right
     over land. The plate holds five readings, each with class and date, plus "A low probability
     does not mean safe". The rail runs along the bottom.
   - Phone: the map on top, the rail as a band, and the probe as a bottom sheet with three readings
     (chlorophyll, model, SST).
2. **My Coast:** "Santa Cruz" at 76 px light, then the context as text selectors.
   - An amber-ruled season statement comes first: sport list closed, commercial on CDFW's pages,
     "Nothing here says a trip is allowed".
   - An instrument row: wind and seas (NWS forecast), then waves and water labelled "observed
     offshore · buoy 46236, 23 km · not the harbour". The NWS advisory follows.
   - A full-width tide line: "times are the facts, the curve is drawn".
   - Two columns: bloom evidence (measured / model context) and official notices ("Not verified · may
     be incomplete · no notice listed does not mean open or safe").
   - The worksheet entry on the bottom rule.
   - The map on the right shows the harbour, the wharf station and the buoy over the HF-radar 24 h
     mean.
3. **Economics, missing:** a signal chain (Fuel → Other costs → Catch → Crew → Net to boat). Known
   nodes show values (42.8 gal, $195); missing nodes show "—" and an amber "needs …". The fuel price
   field is a dashed amber line with the EIA reference beside it and a "Use $8.242" button. Salmon
   shows "no species-level price reference" and no preset. The partial row shows what can be known.
   On phone the chain becomes rows, and the fuel-price field and its reference sit in the first
   screen.
4. **Economics, complete:**
   - A dashed sky "HYPOTHETICAL SCENARIO" banner under the season line.
   - The full chain ($352 → $195 → $1,710 → $362 → $800).
   - A break-even line chart: net vs. lb at your price, crossing zero at 60 lb, with your 180 lb
     marked.
   - A 5×5 sensitivity grid: price ±30%, catch −75…+25%. Loss cells are amber-hatched; your cell is
     outlined.
   - Model A's arithmetic in one line, and a crew-pay segmented control ("Share after expenses ·
     30%").
5. **Research:**
   - A coast "latitude strip" for the 17 stations, placed by latitude with collision-avoided labels.
   - The station title, then two key values: measured 0.20 ng/mL (Sep 30) and model 78% (Oct 19),
     which are never compared.
   - The chart as the canvas: fixed decades from 0.0001 to 100, reported zeros as floor marks, joins
     only under 21 days, the C-HARM threshold, and a separate model panel hatched before Apr 23.
   - Three caveat notes, including AUC 0.30 (n = 1,058).
6. **Loading:**
   - The relief and coast are drawn first at reduced saturation, with the copy "an empty sea here
     means not loaded yet, never low".
   - A plate "Reading the published layers" ticks through each source with its real date: notices
     (not verified), the C-HARM run Oct 20 (out of date), satellite tiles 14/36, radar, SST.
   - The rail is drawn from the manifest as dashed outlines before any pixel.

**Strengths:** the most immersive and "premium-scientific" of the three. The water is the hero. The
time rail and probe make product semantics visible, and it photographs beautifully.

**Risks:**
- Dark-only can fail in sun on a deck. It needs a daylight theme, which roughly doubles the design
  work.
- Long briefings are heavy in light-on-dark mono.
- The cinematic register can feel like a toy to agency readers.
- Leader lines and plates need careful collision handling at every camera.

**Phone:** the map over a solid bottom sheet, with the time rail as a band between them. My Coast
is a short header map, then the season and the instrument grid.

---

## B. Maritime Atlas

**Thesis.** Neritic as a beautifully set coastal atlas and almanac. It uses the vocabulary
fishermen and scientists already trust (chart neatlines and graticules, tide tables, quoted forecasts,
ledgers, figure captions) and is typeset with editorial authority. The guardrails become the
publication's own voice: italic margin notes, small-caps product classes and footnote bands.

**Composition rules**
- Maps are **plates**: a double neatline with graticule ticks and labels (36°54′ N, 122°00′ W)
  computed from the real camera, a title cartouche, an inset legend cartouche, and cartographic italic
  place names. The real render sits inside on cream land with a pale sea (the OLCI field or a recoloured
  relief).
- Selection is a **margin note**: the point is numbered on the plate and its reading is set in the
  right margin as a typeset entry. Values are 32 px Newsreader lining figures with small-caps class and
  italic source/date.
- Time is an **almanac strip**: one column per day.
  - ● marks observed satellite days, ◆ C-HARM valid days, ┅ the outlook bracket.
  - Today is a solid ink cell, and the selected day has a teal outline.
  - A legend line underneath says "no layer is observed today".
- My Coast is a **front page**: a dateline, a 76 px place name, an italic dek, a double rule, and three
  columns separated by rules (Regulation | Sea and weather | Plate and bloom evidence).
- Money is a **ledger**: dotted leaders, a source mark per line ("you" / "basis: EIA" / "shared" /
  "boat-paid"), blanks as amber underlines marked "needed", and marginal notes beside the point of
  need.

**Type and colour**
- Type: Newsreader (variable optical size, 300–600, italics) for display, values and notes;
  Instrument Sans for UI and small caps. Both are SIL OFL 1.1.
- Paper #f4f0e6 with a faint dot grain. Ink #18222b / #4e5961 / #7d858a.
- Families on paper: amber #a35f00, violet #6247c4, teal #0b7a6c, sky #1d6aa1. All are dark enough
  for 4.5:1 on paper.
- Cartographic paint: land #ece6d8 with grey relief, sea recoloured from the NOAA relief to a pale
  blue-grey ramp, and an ink coastline.

**Motion idea:** the motion of print. Plates fade in once, after their values are read; nothing tweens
between days. Selecting a point draws the numbered ring and slides its margin note in by 8 px. Under
reduced motion everything is static. The page should feel like turning to the right plate, not like a
dashboard updating.

**The six screens**
1. **Explore:** the "Monterey Bay" plate (960×690) with graticule, cartouche ("Observed · Satellite
   chlorophyll-a, Sentinel-3 OLCI 300 m · latest clear view; each pixel keeps its own date, Oct
   13–16") and legend cartouche.
   - Margin note 1, "Off Santa Cruz": five entries, then the italic caveat ("A low probability is not
     a sign of safety…") and "Santa Cruz briefing →".
   - The almanac strip below.
   - Phone: the plate fills the width, with the almanac and margin notes below.
2. **My Coast:**
   - Dek: "A small craft advisory from 9 tonight… Ocean salmon is closed on CDFW's sport list; the
     commercial season is CDFW's to state."
   - Regulation column: an amber-boxed season statement ("Nothing on this page says a trip is
     allowed"), three notices with agency small caps and italic area/dates, and "Not verified · may be
     incomplete. A notice missing here does not mean the water is open or seafood is safe. CDPH
     shellfish line (800) 553-4133".
   - Sea and weather column: NWS text quoted in italics with its attribution ("a forecast for the
     whole zone"), the buoy marked "observed offshore… not the harbour", a two-day tide mini curve and
     a typeset tide table.
   - Third column: a Santa Cruz plate, then the bloom evidence ledger.
   - Footer: the worksheet entry.
3. **Economics, missing:** a two-page spread.
   - The ledger on the left, with blanks as "needed", the EIA reference in a sky-ruled marginal note
     with "Use $8.242", and the salmon note "No price reference… Ask your buyer".
   - "Result" on the right is partial: fuel 42.8 gal, other costs $195, net "—". Under it, "Still
     needed, in the order the arithmetic uses them".
   - Phone puts the result above the ledger, so the gap is seen first.
4. **Economics, complete:**
   - An ink-ruled "HYPOTHETICAL" box: "not a forecast of catch, price or profit, and not advice to
     fish".
   - Big figures: $800 net, $547 trip cost, 60 lb break-even.
   - Fig. 1: the break-even line. Fig. 2: the sensitivity table, with losses as amber italic
     (parentheses).
   - The crew-model arithmetic as a caption, and the segmented crew-pay control in the ledger.
5. **Research:**
   - "Figure 1 · Particulate domoic acid at the wharf, a year of weekly samples", with a dek.
   - Measured and model values top-right, apart from each other.
   - The chart with two editorial annotations: the May 27 11.2 ng/mL peak (real), and "late 2025:
     traces near 0.001 ng/mL; squares on the floor are reported 0".
   - The station index as a typographic north-to-south list with latitudes, and a three-part footnote
     band.
6. **Loading:**
   - The plate frame and graticule appear first, over the pale relief with "uncoloured means not yet
     loaded, never low".
   - Margin: "Plate in preparation · Gathering the sources", ticking through each source with its
     date and a thin progress rule.
   - The almanac line gives the dates from the manifest.

**Strengths:**
- The most distinctive of the three, and nothing like the current dashboard.
- The strongest typographic hierarchy, and highly legible in daylight.
- The caveats read as authorship rather than disclaimers.
- It suits the regulatory content: notices *are* documents.
- Plates are inherently shareable and printable, and it ages well.

**Risks:**
- Can drift into nostalgia or "museum" if the plates are over-ornamented. Keep the neatline at one
  weight and use no faux-engraving.
- A serif for values needs lining tabular figures everywhere (Newsreader has them).
- Explore on desktop gives up some map area to the margin column.
- Night use needs a dark "chart-room" variant.
- Relief recolouring must be done in the style, not by post-processing, and needs a fix for the
  vertical seam between the regional and Monterey relief tiles (visible in one capture).

**Phone:** a single column of plate, almanac, margin notes; then the front page as dateline, name,
dek, season box and quoted forecast. The ledger reflows to label, value and needed.

---

## C. Coastal Mission Control

**Thesis.** Neritic as the skipper's operations board. Everything that matters for a trip (season,
notices, forecast periods, tide, model days, latest observations and the trip window itself) is laid
out on **one shared time axis**, with a decision ledger and the economics beside it. It links
briefing → conditions → trip → money without a wall of cards: rows, rules and tags, not tiles.

**Composition rules**
- A navy **situation bar** holding the context (Place / For / Species) and the notice status ("3
  notices · Not verified · may be incomplete") on every screen.
- A **header row** of up to four headline values, each with a source tag (Forecast · NWS, Prediction
  · CO-OPS, Advisory · NWS).
- **Lanes** over "the next 60 hours", with a 3 px family-colour rail per lane and a single "now" line
  through all of them:
  - Season · notices
  - Marine forecast: the full NWS text per period, and the advisory with "no end time given"
  - Tide: H/L facts with a drawn curve
  - Bloom model: Thu 82% day 3, then "No published run covers these days · a gap, not a low value"
    hatched, and the outlook as a dashed one-mean bar
  - Latest observations: earlier items pinned left with ages, and the buoy at now
  - Your trip
- A right **decision ledger**: Season "Not known to Neritic", Notices, Weather "Your limits are not
  set, so nothing is compared", Bloom "Context only", and Unknown 5. Then compact economics.
- Explore is split: the map on top, and below it "What exists when" (product lanes, Oct 12–27) next
  to "This point" (an inspector table).

**Type and colour**
- Type: Geist (300–700) and Geist Mono (tabular values), both SIL OFL 1.1.
- Ground #fbfcfd with a navy #0b1b2b structure. Ink #0d1722 / #4a5867 / #8592a0, rules #e1e6eb.
- Families as text plus tint pairs: amber #b46400/#fff5e6, violet #6a4bd6/#f2eefe, teal
  #0a8070/#e6f5f2, sky #1b6fb0/#e8f2fa.

**Motion idea:** a live board. The "now" line advances in real time (one step a minute, no
animation). When a source refreshes, its lane flashes its rail once (200 ms, not under reduced
motion) and the lane's timestamp updates. Dragging the trip window snaps to whole hours, and the
forecast periods it overlaps get an outline. Nothing in the data tweens.

**The six screens**
1. **Explore:** the map takes the top 520 px:
   - a segmented layer control with the class above each name (Model / Observed / Outlook / Analysis
     / Observed / Reference)
   - a crosshair with a coordinate tag, the legend, and credits
   - Below: product lanes over Oct 12–27, with the selected point's values in the cells (1.3 on Oct
     14, 82% on Oct 22, the outlook bracket, 18.6 on Oct 19, the 0.10 radar mean) and the inspector
     table.
   - Phone: the map and the inspector.
2. **My Coast:** as described above. Phone becomes a **time-ordered list** ("4:04 PM Now… 8:32 PM
   High tide 4.3 ft · 9 PM Small craft advisory begins…") under an amber season band. This is the
   strongest phone translation of the three.
3. **Economics, missing:**
   - A season band, then a conditions strip with "Trip window not set".
   - Three columns: an inputs table (value / basis; "needed" in amber with the EIA reference and
     "Use"), a partial result with a numbered "still needed" list, and an empty, hatched sensitivity
     box: "Appears once net to boat can be computed. Nothing is filled with a typical catch or price".
4. **Economics, complete:**
   - "HYPOTHETICAL · not advice to fish" in the header.
   - The conditions strip with "Your trip window Fri 5 AM – 3 PM · your plan, not an authorization"
     aligned under the NWS Fri period.
   - The inputs table with basis per line, result numbers, a "where the $1,710 gross goes" flow bar,
     the break-even line, and a large sensitivity heatmap (navy intensity, amber for losses, your cell
     outlined) with the crew-model formula.
5. **Research:**
   - A station combobox plus a latitude dot rail, variable tabs and range.
   - The main chart, plus small multiples on the same time axis: *Pseudo-nitzschia* (log) and pier
     temperature.
   - A right column of measured, model and record facts, with the caveats.
6. **Loading:** the **same My Coast layout loading in place**.
   - A progress hairline at the top.
   - Header skeletons with "fetching NWS PZZ535…".
   - Lanes that have arrived show data (season, tide, model). Lanes still loading say what they are
     fetching and why ("the zone is assigned from the port, never guessed"; "nothing older than 6 h is
     shown from the buoy").
   - The ledger is dimmed.

**Strengths:**
- The most useful for the commercial job: it is the only concept that answers "what happens, in what
  order, during my trip?", and it ties economics to conditions.
- Dense but scannable, with no cards.
- The best loading model (progressive, in place).
- The best phone briefing.

**Risks:**
- Closest of the three to "software". In weaker hands it slides back into a dashboard.
- Lanes truncate text at 1440 px, so the full text needs hover or expansion.
- The trip-window lane must never read as a go or no-go or an authorization. The copy carries this,
  and it needs review with the owner.
- The least emotionally engaging, and the map is secondary on My Coast.

**Phone:** the time-ordered list for My Coast; a stacked inputs → result list for economics, with the
fuel price and reference in the first screen; map plus inspector for Explore.

---

## Guardrail review (integrity-guardrails §5)

Applied to the concept renders. A full axe and performance check (items 28–37) belongs to the build,
not to static concepts.

| Item | A | B | C | Notes |
|---|---|---|---|---|
| A1 class in words | Pass | Pass | Pass | every value has Observed/Model/Outlook/Analysis/Measured/Forecast/Prediction |
| A2 own time per value | Pass | Pass | Pass | per-pixel observed date, valid + inferred issue, analysis day, 24 h mean window, sample date, NWS update |
| A3/A4 no interpolation or tween | Pass | Pass | Pass | rails and almanac are discrete cells; "steps cut, never fade"; sample joins < 21 d |
| A5 gaps ≠ low | Pass | Pass | Pass | hatch legend; C's "a gap, not a low value"; loading "not loaded, never low" |
| A6 estimate/outlook distinct | Pass | Pass | Pass | outlook as one dashed mean bracket, never daily. Estimate legend in A. **Owner decision:** the R12 "off by default" tension for the estimate is unchanged |
| A7 chl ≠ toxin / fish | Pass | Pass | Pass | stated beside every chlorophyll value |
| A9 one product per legend | Pass | Pass | Pass | |
| A10 no safety verdicts | Pass | Pass | Pass | no green, no "safe"; "a low probability does not mean safe" visible |
| A12 same-pipeline numbers | Pass | Pass | Pass | outlook named climatology + AR(1); AUC 0.30, n = 1,058 from the derived index |
| B13 Not verified everywhere | Pass | Pass | Pass | including phone mastheads (B and C added a phone indicator) |
| B15 no notice ≠ open/safe | Pass | Pass | Partial | C states it in the ledger; the full sentence must also sit beside the notice lane |
| B17/B18 commercial vs sport; defer to CDFW | Pass | Pass | Pass | salmon: sport list closed; commercial season on CDFW pages; "Nothing here says a trip is allowed" |
| C19 hypothetical; season above economics | Pass | Pass | Pass | |
| C20/C21 no invented defaults; references distinct | Pass | Pass | Pass | salmon has no preset or price reference; EIA copied only on "Use", then "basis: EIA" |
| C22 missing = "—" plus need | Pass | Pass | Pass | |
| D24 OSM credit on map at every width | Pass | Pass | Pass | relief credit and "Not for navigation" too |
| D27 fonts OFL, self-hosted | Concept only | Concept only | Concept only | concepts load Google Fonts; the build must self-host and ship OFL.txt |

---

## Recommendation

**Build B, the Maritime Atlas, as the product's system. Borrow two things, on B's terms: A's
interaction model for Explore, and C's shared time axis for My Coast and the worksheet.**

Why B:
1. **It is the only direction that is unmistakably not a dashboard.** The comparison sheet shows it:
   A is a beautiful dark map app, C is excellent software, and B is a publication with a point of view.
2. **The science guardrails become the voice, not fine print.** Margin notes, small-caps classes,
   quoted forecasts and footnote bands give every caveat a typographic role. This answers the
   "walls of small text" failure of v4–v9 directly.
3. **Its vocabulary is the sea's own** (plates, almanac, tide table, quoted NWS text, ledger). That is
   trustworthy to fishermen and to agencies alike, and notices *are* documents.
4. **Daylight legibility.** Dark ink on paper at 4.5:1 or better for every family colour suits a deck
   in sun. A night "chart-room" variant can reuse A's palette as B's dark theme (one brand, two
   grounds).

What to borrow, without fragmenting (one type system, one colour system, one set of components):
- **From A, for Explore:** the full-bleed field on desktop. B's plate frame becomes the share/print
  and phone presentation, and the desktop field runs to the margin column. Also from A: the
  probe-with-leader into the margin note, and the permanent time rail with a hard "now" boundary, set
  as B's almanac strip. A's dark palette becomes B's night theme, not a second brand.
- **From C, for My Coast and the worksheet:** "The next 60 hours" as a lanes figure inside B's
  centre column (same rails, same now line, typeset in B), the phone time-ordered list, the trip-window
  strip on the worksheet, the in-place progressive loading model, and the decision ledger's honest
  "Not known to Neritic / Your limits are not set" rows, set as a B ledger.
- **Do not borrow:** A's mono-for-everything; C's navy bar and tinted tags (B uses small caps and
  rules); any particle motion.

Decisions for the owner:
1. Approve B as the base, or choose A or C.
2. Whether a night theme ships in the first release.
3. The estimate-default tension (R12), unchanged by these concepts.
4. Whether the trip-window lane is acceptable for a closed sport species, given that the commercial
   season is not held by Neritic.
5. Brand and logo stay with the separate brand pass. The concepts use placeholder wordmarks.

## What could not be verified

- Windy and Earth Nullschool visuals could not be inspected: both are canvas/JS apps. Their sections
  in [visual-references.md](visual-references.md) rest on their about pages, terms and forum posts.
- Live HF-radar coverage for the current hour in Monterey Bay was empty ("no radar observations this
  hour"). The concepts therefore show the real 24-hour mean, labelled as such.
- The C-HARM "78%" for the wharf is the Oct 19 median in the observations file. The research page in
  the app showed 70% for the Sep 30 sample day. Both are real; the concepts use the newest.
- Salmon commercial season status is not in Neritic's data. The concepts state that and defer to CDFW.
