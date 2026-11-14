# Neritic v12: art-direction brief for the three concepts

Owner of this brief: the lead creative director. Concept builders execute it; they do not change the
shared sections, and they do not borrow from each other.

The failure we are escaping, in the owner's words: "crowded, generic, AI-generated". Read
`research/B-visual-system.md` section 4 for its specific tells. A concept that shows one of those tells
has failed, however well it is built.

## Shared and fixed

### What every screen answers
1. What am I looking at? 2. Why does it matter to me? 3. What can I do next?
Answer through composition and one short sentence, not through a labelled box.

### The product's answer to "why use this tomorrow?"
PredictWind and FishTrack answer *where and how windy*. Agency sites answer *one question each, on eight
pages*. Neritic puts season status, toxin tests, whale measures and the next decision date for your port
and fishery in one place, next to what the water is actually doing. The documented complaint from
fishermen is about **when** ("We don't know when we're going"), so lead with status and the next date.
Never claim to find fish, and never give a go/no-go.

### Information architecture
One entry point: `/`. The opening plays once on a first visit (4 to 5 s, skip at any moment) and
**becomes** the home screen, same surface, with no route change. Returning visitors land on Today.

| Destination | One job | Its sentence |
|---|---|---|
| Today | What changed at my port, and what's the next date | "Moss Landing, today" |
| Explore | See the water: observations and forecasts, in place and time | "The water, as last measured" |
| My Coast | Plan a trip: gates, sea window, trip economics | "Plan a trip from Moss Landing" |
| Research | Check the record: series, QC, provenance, export | "The record behind every number" |

The port travels with the reader across all four. The word "Ask" and alerts live in the shell.

### Seven deliverables per concept (files in your concept folder)
| File | Screen | Must show |
|---|---|---|
| `opener.html` | Hero/opening | Plays live, 4–5 s, skip control, ends on the same composition as `home.html`. `?capture=1` may hold at the final frame |
| `home.html` | First visit | Understood in five seconds: one sentence of specific value, one hero visual, **one** primary action (Find your port), at most three small paths. No statewide numbers |
| `today.html` | Returning user | Moss Landing + Commercial Dungeness: what changed, what needs attention, what's next |
| `explore.html` | Interactive map | The map is the page. Real chlorophyll + real measured currents animating, time control, legend-as-title, click a point for its real value and a linked chart, minimal chrome |
| `coast.html` | Trip planning and economics | Gates, the sea window, and a working ledger: editable inputs, live waterfall, sensitivity that animates meaningfully |
| `research.html` | Research workspace | Real HABMAP series, linked map selection, temporal brush, QC marks, provenance panel, export action |
| Mobile | Every page above | Each page must be deliberately composed at 390 px, and survive 320 px and 844×390 landscape |

Interaction must be real where stated: sliders recalculate, the map responds, the brush filters, skip skips.

### Real data only
`concepts/kit/kit-data.js` sets `window.KIT` (snapshot, contours, tide, habmap, ports, `KIT.project`). Python
helpers are in `concepts/kit/kit.py` (DEM, hillshade, native 300 m chlorophyll, ramps). Bake into your own folder.
Every number a reader sees comes from `KIT` and carries the kind, source and date required by the truth rules.
"Today" is Tue Nov 10, 2026, Pacific. Do not invent any measurement, status, name, testimonial or user count.

Facts to use as written:
- **Commercial Dungeness:** "Season delayed to at least Jan 1, 2027 (humpback whale entanglements)." CDFW, checked Nov 8, **not verified**.
  Next assessment Dec 15–18, then Jan 13–15, "subject to change". RAMP Zone 4, Pigeon Point to Lopez Point.
- **Crab test, Monterey Bay (Dungeness, viscera, CDPH):** Sep 28, six crab, average 5.5 ppm, highest 9.8 ppm, action level 30 ppm. 43 days old. Tests inform CDPH and CDFW; they don't open a season.
- **Sea (NWS PZZ535, issued 2:30 AM):** "N wind 5 to 10 kt, backing to NW this afternoon. Seas 4 to 6 ft. Wave Detail: NW 5 ft at 14 seconds." Print exactly.
- **Buoy 46236, Monterey Canyon Outer (15 km):** 6.6 ft at 14 s, 7:00 AM PT. **Tide** comes from `KIT.tide` (Monterey, prediction).
- **Chlorophyll:** 0.69 mg/m³ median within 15 km, Sentinel-3 OLCI, latest clear view Nov 3–8. "Algae, not toxin." Gaps are "no view".
- **C-HARM:** run of Nov 3, valid Nov 2–5, every lead is past. Label "past run". Never a forward-looking hero and never styled as an alarm. Must say "probability of conditions in the water, not toxin in seafood". The 75 % figure is not news: it has read 70–81 % on all 17 run days in 30, with no demonstrated skill at the shore.
- **8-day outlook at Moss Landing:** pure climatology this run. Say so ("no signal: equals the seasonal average"). Don't show its value beside the observed figure.
- **Currents:** HF radar, one observed hour, 05:00 UTC Nov 10, about 2 km. Gaps are no data, not calm.
- **Economics:** the product pre-fills nothing. For the mock only, typed example inputs, visibly labelled "example numbers": 800 lb at $4.75/lb, one day, 120 gal at $4.60, two crew at 15 % each of net after trip costs, 5 % fees, $180 ice and bait. Read `coastwatch-web/src/v4/coast/economics/` for the true formulas and mirror them. Output labelled "hypothetical".

### Truth rules (binding)
`research/E-science-and-perf.md` section 2, T1–T47. Forbidden words (T36) are forbidden in copy and alt text.
Hierarchy: official status, then measured tests and observations, then models. "Not verified" is one marker per group, in words, computed.
Motion may not depict what wasn't measured: no ambient particles, no flow in gaps, no animating chlorophyll or model leads as movement. Any exaggerated depth is stated on screen.

### Copy and craft floor
- Every number has one home on a screen. A percent carries its basis. Unknown is a hollow mark, not green.
- Body text ≥ 16 px desktop and phone. Nothing a person must read below 14 px. Sentence case, always. No eyebrow labels above every section. No mono caps.
- Tabular numerals in columns. Touch targets ≥ 48 px on phone. Text contrast ≥ 7:1 for anything read on deck.
- No: KPI rows; badges and icon pills; card grids; identical section templates; row-plus-chevron lists; frosted dark pills; mesh gradients or glows; text scramble; stagger on everything; decorative microcopy; emoji.
- Honor `prefers-reduced-motion` (final frame, same information). Provide a static fallback when WebGL is missing.
- Technique: plain HTML, CSS and JS in your folder. Fonts by Google Fonts `<link>`. Libraries only from cdn.jsdelivr.net with exact versions. All other assets local and relative. No build step required to open the file. Use a shared `style.css`, so each concept is one coherent system with named tokens.

## Concept A: Chart

**Ancestor.** The NOAA nautical chart of Monterey Bay (see `refs/src/noaa-enc.png`) and Admiralty engraving. The most familiar object in a fisherman's life, redrawn with contemporary precision. Daylight, sunlight-first.

**Idea.** The product is a chart, and your briefing is annotated on it. The numbers are soundings: "9.8 ppm" sits where that crab was caught; "6.6 ft at 14 s" sits at the buoy. Reading the page is reading the water.

**Colour.** Palest water offshore, most saturated at the coast, buff land, from the ENC (research B section 6.A). One foreign overprint hue, magenta, and it means *regulation* and nothing else (whale zone lines, season delay, official notices). Rust only for official closures. Interaction is ink. Design a separate light-ground chlorophyll ramp in which high is dark.

**Type.** Letter style carries class, as on a chart: italic Literata for water names and only water names; upright Public Sans for land and interface; large Literata for statements. Tabular Public Sans numerals, sized like soundings when on the map.

**Structure.** A graduated neatline with real latitude and longitude ticks frames every page. The chart cartouche, a title block in a corner, answers the three questions and holds the primary action. Square corners. No shadows: layering is by knocked-out white halos, as cartographers do. Rules at 1.5 px where they matter.

**Motion.** Drafting. Isobaths ink themselves outward from the canyon head, depth tints flood in, soundings register in place. Transitions pan the same sheet: moving between pages is sliding across one chart.

**Opener.** White sheet; the real canyon contours draw; tints flood; the neatline closes; the cartouche sets its type and is the home screen.

## Concept B: Canyon

**Ancestor.** Tanaka illuminated contours and deep-sea survey imagery. The Monterey Canyon is one of the great underwater landforms on earth, and it sits at Moss Landing's front door. The canyon is Neritic's emblem and its stage.

**Idea.** You arrive in the water. A full-bleed, lit, real relief of the canyon (CRM, depth exaggeration stated) is the world, and solid light instrument plates carry all reading. The relief is a mid-tone petrol, never near-black, and text never sits directly on it.

**Colour.** Relief runs from deep petrol `#12323E` to sea-glass `#7FA8A6` with white lit edges and dark shadow edges. Plates are bone `#EEF1EC` with ink `#10242B`. One accent, signal yellow `#FFD24A`, is interaction only, and on a plate it appears as an ink-on-yellow fill. Official status is ink weight and words. Chlorophyll has its own ramp and appears only on water.

**Type.** Archivo at maximum width for very large statements, tight and confident; Archivo at normal width for reading; Archivo Narrow for map labels. One family, three voices.

**Structure.** A depth gauge on the left edge, 0 to 3,300 m, is both the page's progress and the navigation; each destination sits at a real depth. One large plate per view, placed asymmetrically; plates have real thickness (one hard edge, no blur). Big negative space of landform.

**Motion.** Descent and surfacing. Pages are depths, and moving between them travels vertically through the relief. Light angle responds slowly to pointer, in the relief only (real geometry, relit). Currents are bright measured streaks on the surface layer.

**Opener.** Real WebGL or baked flight, starting low in the canyon, rising along the axis to the surface as the live chlorophyll coastline resolves, then the first plate lands. Start from `spikes-D/` (heightfield GL, DEM). State "depth exaggerated 4×".

**Paper Shaders.** Not here: fluid shaders read as measurements.

## Concept C: Tide

**Ancestor.** The pocket tide table, the Swiss poster, and Teenage Engineering's instrument labelling. A California harbour-office noticeboard made by someone with very high standards.

**Idea.** Time is the page. The fisherman's complaint is *when*, so one real tide curve runs across the top of every page and everything hangs from it at its true moment: past is solid, the present is a line, the future is hatched. The giant thing on Today is a date.

**Colour.** Bone paper `#F4F0E6`, ink `#15140F`, and one violent accent, international orange `#E8481C`, used for *now* and the primary action only. Status is carried by ink: filled block, outlined block, hatched block, always with words. Chlorophyll colour appears on the map and nowhere else.

**Type.** Barlow Condensed, enormous, at weights 500–700, sets dates and statements in sentence case. Barlow for reading (California plate and highway sign lineage). Numbers are the imagery: tide-table columns with subscript decimals.

**Structure.** A strict, visible 12-column poster grid with deliberately unequal blocks and big type crossing them. Square. Ruled like a table, no containers. The tide line is the navigation: pages are stops along it.

**Imagery.** Halftone. Real relief and real chlorophyll rendered through Paper Shaders (`@paper-design/shaders` halftone or dithering, exact version, Apache-2.0), **baked at build time from real images**, used large and cropped like poster photography. Label it for what it is. The halftone never carries a data value: the Explore data layer is plain colour.

**Motion.** Mechanical and exact: the now-line ticks, figures flip like a split-flap board when time changes, the curve redraws on the selected day. Transitions run along the tide line, left to right.

**Opener.** The tide curve draws across a bare page, the now-line drops, the halftone coast resolves dot by dot beneath it, the date lands in giant type. That composition is home.
