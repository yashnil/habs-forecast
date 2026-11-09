# Neritic v11: reference research

Researched 2026-11-09. Every claim carries the URL it was read from. **(PK)** marks prior knowledge that could not be re-read today (page was JS-only, rate-limited or blocked). Sources that failed: Material 3 and Apple HIG motion pages (JS-rendered, empty), Observable Plot docs (429), Surfline support (403). No first-party Stripe or NYT motion spec was found; nothing is attributed to them below except where marked.

## 1. Product observations

**earth.nullschool.net.** The map is the whole product. The menu is closed by default and opens with one control (`e`; Escape closes); stepping time, changing height and toggling animation are single keys (J/K, I/M, P), and `n` jumps to "most recent data". Each overlay names its model and cadence in the About text (weather every 3 h, currents daily, waves every 3 h, aurora every 30 min) and SST is flagged "preliminary if less than two weeks old". https://earth.nullschool.net/about.html. On screen the only persistent chrome is the word "earth" bottom-left; clicking the globe prints coordinates plus the value at that point (PK).

**Copernicus MyOcean Light.** Four fixed zones and nothing else: variables (12) in a left panel; **one legend at top centre that is also the title block** (variable name, unit, displayed date, product summary, colour bar); time axis along the bottom (drag, arrow keys, wheel to zoom the axis); depth axis bottom-right that **only appears for variables that have depth**. The viewer picks the product from zoom and date; a link above the legend says which product is showing. Style, colour map and min/max sit behind a gear that appears on legend hover. A map click drops a point and opens a time-series graph. https://help.marine.copernicus.eu/en/articles/9418874-introduction-to-myocean-light-viewer

**Global Fishing Watch.** Three zones: left layer sidebar grouped by kind (Activity, Detections, Vessels/Events, Reference), bottom timebar, a small tool column on the right. The colour scale lives **under the layer's own title in the sidebar**, and "the point on the scale reflects where your cursor is on the map", so the legend doubles as the readout. Timebar aggregation changes with range (daily under three months, monthly beyond, yearly above six years); playback has a speed toggle and loop. Latency is stated as a hard boundary, not a badge: data runs "to up to 72 hours ago" and the animation requires the end bar to respect that. https://globalfishingwatch.org/user-guide/ and https://globalfishingwatch.org/our-map/

**Zoom Earth.** The layer menu is split by epistemic status: "Live Maps" (geostationary satellite, radar) versus "Forecast Maps" (ICON 13 km, GFS 22 km). Non-live layers say so in words: heat spots are "not live", delayed "many hours". Time control is user-selectable (Timeline or Clock; local or UTC); loop length is 3/6/12/24 h with Fast or Smooth animation. https://zoom.earth/about/

**Ventusky.** Motion is chosen per phenomenon: streamlines for wind, animated arcs for waves and swell. Colour is semantic (blue cool to dark red hot; precipitation reserves orange/red for dangerous totals). One layer at a time from a list of 14. https://www.ventusky.com/about

**Windy.** Layers set what, the timeline sets when (wording from the Windy.app guide: https://media.windy.app/windyapp_guide_new). Model comparison is a per-location action ("Compare" at bottom centre of the location forecast), not a permanent panel: https://community.windy.com/topic/9427/how-to-use-the-compare-feature. A cautionary note: the redesign that hid features drew long complaint threads from sailors who lost fast model comparison: https://community.windy.com/topic/32400/i-do-not-like-the-new-interface/27. Layout from memory: bottom timeline with play, layer list on the right edge, one colour-ramp strip bottom-right, particles over a filled field, and a draggable picker that prints the value (PK).

**Apple Weather (iOS 15+).** The layout reorders itself by conditions: if rain is coming, hourly forecast, next-hour precipitation and radar move to the top. The top of the screen is a written summary of the rest of the day. The map is a small module that opens full screen; a stack control switches temperature, precipitation and air quality. The precipitation scrubber spans past and future on one bar (reported as 3 h back, 1 h ahead in the hourly view) and can be paused or dragged. https://www.macrumors.com/guide/ios-15-weather-app and https://9to5mac.com/2021/09/24/ios-15-weather-app-hands-on/. Hero is one large thin numeral, one condition word, H/L; no boxes around it (PK).

**savvy navvy.** Chart fills the screen. One hamburger top-left for account and settings; five icon buttons on the right (search, 3D, north lock, chart style and overlays, measure); four bottom tabs (Weather, Tides, Routes, Logbook). All overlays (wind, weather, tidal streams, AIS) sit behind the single layers button. https://help.savvy-navvy.com/en/article/getting-started-navigating-the-savvy-navvy-interface-9rjxtf/

**PredictWind.** Model disagreement is a designed feature: split-screen for two models on the same variable; an observations tool that "directly compares the forecast and weather station observations"; one "Forecast Options" icon top-right holds variable, model, resolution and display style. https://help.predictwind.com/en/articles/2884529-how-to-use-maps and https://predictwind.com/features/models

**NASA Worldview.** Latency is in the product description: many layers "within three hours of observation"; geostationary in ten-minute steps for 90 days; an Events tab jumps the map to current hazards. https://github.com/nasa-gibs/worldview and https://www.earthdata.nasa.gov/data/tools/worldview. Bottom timeline with coverage bars per layer and an A/B swipe compare (PK).

**Mapbox Standard.** The basemap is a configurable instrument: `lightPreset` dawn/day/dusk/night; `theme` default/faded/monochrome; switches to hide place, POI, road and transit labels; `bottom`/`middle`/`top` slots for data. https://docs.mapbox.com/map-styles/standard/guides/. Slots exist so data sits beneath place labels and above water and land (PK). MapLibre equivalent: insert layers before the first symbol layer.

**Felt.** Stroke colour is derived from the fill ("similar colors for outlines and fills helps maintain visual continuity"); on dark basemaps labels are light and the halo takes the data-derived colour; point labels try several positions to reduce collisions. https://felt.com/blog/cartography-tips-for-designing-web-maps

**CARROT Weather.** The screen is a stack of sections the user adds, removes and reorders (Interface Maker; presets), with five font choices and text size as a setting. https://www.macrumors.com/2021/01/28/carrot-weather-5-0 and https://www.macstories.net/reviews/carrot-weather-5-2-revamps-layout-customization-and-adds-new-sections-and-data/

**Linear.** Themes went from 98 variables to three (base, accent, contrast), generated in LCH so equal lightness looks equal; blue was deliberately taken out of the chrome for a neutral look; Inter Display for headings, Inter for the rest; sidebar and headers reworked to "reduce visual noise" and align icons and labels. https://linear.app/now/how-we-redesigned-the-linear-ui

**The Pudding.** Keep scroll-driven steps only when "the transitions are truly meaningful"; otherwise stack static charts on phones. Do not size the sticky graphic in `vh` (mobile bars resize); compute px from `window.innerHeight`. Remove hover on touch and replace with fixed annotation. "Short and sweet." https://pudding.cool/process/responsive-scrollytelling/. Scrollama uses IntersectionObserver plus `position: sticky`: https://pudding.cool/process/introducing-scrollama/. Mapbox's storytelling template maps chapters to camera views (flyTo by default, easeTo/jumpTo available): https://labs.mapbox.com/education/impact-tools/interactive-storytelling

**Not sourced today, low weight:** Surfline (one condition word on a colour scale plus a human-written report, PK); FT charts (warm paper background `#FFF1E5`, title states the finding, source line under every chart, PK; chart selection guide at https://github.com/Financial-Times/chart-doctor/blob/main/visual-vocabulary/README.md); Observable Plot defaults (640 px wide, system-ui 10 px, `currentColor` axes, no gridlines unless asked, PK); (Not Boring) Weather (one 3D object as the entire hero, PK; studio profile https://apps.apple.com/us/story/id1591100111). Nothing useful was verified for Navionics, Windfinder, Tomorrow.io, Strava, AllTrails, onX or Reuters.

## 2. Patterns worth taking

1. **The legend is the title.** One block carries variable, unit, valid date, source and colour bar. MyOcean Light (URL above). Replaces a separate header, legend card, timestamp chip and source footnote.
2. **The legend is the readout.** A marker on the colour scale tracks the cursor or selected point. Global Fishing Watch (user guide). Removes the need for a tooltip card and a second stats panel.
3. **Fixed geography for chrome, at most three zones.** What on one edge, when along the bottom, tools as bare icons. MyOcean Light, GFW, savvy navvy. Users learn positions; nothing floats.
4. **One layers button.** All overlays behind a single control; one field layer visible at a time. savvy navvy, Ventusky. Ends chip rows over the map.
5. **Controls appear only when they apply.** Depth axis only for depth variables; style settings on legend hover. MyOcean Light. The default screen carries no dead controls.
6. **Reorder by conditions, not by schema.** When something matters now it moves to the top. Apple Weather (MacRumors). For Neritic: a closure or active bloom leads; a quiet day collapses to one line.
7. **Sentence first, numbers second.** A written summary of the rest of the day above any chart. Apple Weather. One sentence replaces four stat tiles.
8. **Small map that opens full screen**, or the reverse: full map with one bottom sheet. Apple Weather; savvy navvy bottom tabs. Never map and dense panel at equal weight.
9. **One scrubber across past and future**, with the present marked. Apple Weather precipitation; GFW timebar. Forecast vs observed becomes a position on a line, not a label on every card.
10. **Motion encodes the physics, one idiom per phenomenon.** Streamlines for wind, arcs for swell. Ventusky; nullschool. For Neritic: moving particles only for currents and wind, never as decoration on chlorophyll.
11. **Disagreement and verification on demand at a place.** Compare models or forecast against stations from the location view. Windy Compare, PredictWind. Keeps uncertainty honest without a permanent panel.
12. **Basemap as a muted, configurable instrument.** Faded or monochrome theme, POI and road labels off, data under place labels. Mapbox Standard; Felt label and halo rules. Most "crowding" on a map is basemap noise.
13. **Three-variable colour system in a perceptual space, chrome without hue.** Linear. Hue is then reserved for data and for status.
14. **Scrollytelling only where the transition means something; stack on phones.** The Pudding. Suits an explainer or opening, not the working map.

## 3. Anti-patterns that read as generic or AI-generated

| Tell | What the references do instead |
|---|---|
| Near-black plus cyan or teal accent, with glow | Neutral chrome with the blue removed (Linear); colour belongs to the data ramp (Ventusky, MyOcean). No glow anywhere in the set. |
| Hairline-bordered rounded cards for every fact | Unboxed hero numeral and sentence (Apple Weather, PK); zones defined by position (MyOcean, GFW). |
| Uppercase letter-spaced micro-label above every block | Labels in sentence case inside the content: the legend names the variable and unit once (MyOcean Light). |
| Monospace numerals everywhere | Proportional type with `font-variant-numeric: tabular-nums` only where digits must align (https://vercel.com/design/guidelines). |
| Uniform 13 to 14 px grey text | Two families or cuts with real contrast: display for headings, text face for the rest (Linear); one very large number (Apple Weather, PK). |
| Grid of equal KPI tiles | Condition-ordered sections of unequal size (Apple Weather); user-ordered sections (CARROT). |
| Several legends, chips and badges floating over the map | One legend, one layers button, one time axis (MyOcean Light, savvy navvy). |
| Status pill on every element ("LIVE", "FORECAST", "MODEL") | Status carried by menu grouping (Zoom Earth) and by position on the time axis (Apple, GFW). |
| Everything fades and slides in on load, staggered | "Only animate when it clarifies cause & effect or when it adds deliberate delight" (Vercel guidelines). |
| Decorative particle or gradient background | Particles only where they are the data (nullschool, Ventusky). |
| Hiding expert functions to look clean | Windy's backlash: keep the fast path one tap away (community thread above). |

## 4. Motion rules observed

- UI transitions under 300 ms; a 180 ms dropdown reads as faster than a 400 ms one. https://emilkowal.ski/ui/7-practical-animation-tips
- Enter and exit with ease-out; avoid ease-in; built-in CSS curves are too weak, use custom ones. Same source. Sonner uses plain `ease` with a slightly longer duration for a calmer feel: https://emilkowal.ski/ui/great-animations
- Never scale from 0; start at 0.9 or higher. Pressed state `scale(0.97)`. Tooltips 125 ms ease-out, delayed on first open, instant (0 ms) for neighbours. https://emilkowal.ski/ui/7-practical-animation-tips
- Do not animate keyboard-initiated or high-frequency actions ("hundreds of times a day ... never animate them"). https://emilkowal.ski/ui/you-dont-need-animations. For Neritic: layer switch, time step and port switch cut or cross-fade briefly; they do not slide.
- Animate only `transform` and `opacity`; never `transition: all`; prefer CSS over WAAPI over JS; animations must be interruptible; set transform-origin where the motion physically starts. https://vercel.com/design/guidelines
- Autoplaying motion longer than 5 s needs pause, stop or hide. Same source. Applies to any looping outlook animation.
- Loading: show-delay about 150 to 300 ms, then minimum visible time about 300 to 500 ms; skeletons mirror final layout exactly. Same source.
- Honour `prefers-reduced-motion` with a real variant (a 0.2 s fade in place of movement). https://emilkowal.ski/ui/great-animations
- Material 3 tokens (PK, page unreadable today, https://m3.material.io/styles/motion/easing-and-duration/tokens-specs): short 50 to 200 ms, medium 250 to 400 ms, long 450 to 600 ms; emphasized-decelerate `cubic-bezier(0.05, 0.7, 0.1, 1)` for entering, emphasized-accelerate `cubic-bezier(0.3, 0, 0.8, 0.15)` for leaving, standard `cubic-bezier(0.2, 0, 0, 1)`; exits shorter than entrances; larger travel takes longer.
- Apple HIG (PK, https://developer.apple.com/design/human-interface-guidelines/motion): motion is optional, brief, and avoided on frequent interactions; Reduce Motion replaces slides and zooms with cross-fades.
- Time playback gives the user speed and loop control (GFW 1x toggle and loop; Zoom Earth Slow/Medium/Fast, Fast vs Smooth).
- Camera moves between chapters use flyTo, with easeTo or jumpTo as declared alternatives (Mapbox storytelling template). Using jumpTo under reduced motion is my inference, not a sourced rule.
- Sticky scroll graphics sized in px from `innerHeight`, not `vh`. The Pudding.

## 5. Honesty patterns

1. **Group by epistemic status in the menu.** "Live Maps" vs "Forecast Maps" (Zoom Earth). No per-item badge needed.
2. **Say the valid time and product in the legend**, once (MyOcean Light).
3. **State latency as a boundary on the time axis.** Data ends 72 h ago and the control cannot go past it (GFW). For Neritic: the scrubber ends at the last satellite pass, then the outlook begins.
4. **Past and future on one axis** with the present marked (Apple Weather precipitation).
5. **Plain-word caveats on the specific layer**: "not live", "preliminary if less than two weeks old" (Zoom Earth, nullschool).
6. **Name the model and its cadence** per layer (nullschool About; Ventusky model labels; Zoom Earth resolutions).
7. **Offer forecast against observation at a place** (PredictWind observations tool; Windy Compare).
8. **Jump-to-latest control** (`n` on nullschool) so a stale view is one action from current.
9. **Disclaim accuracy once, in About, not on every card** (nullschool "implies no guarantee of accuracy").

Gap: none of the sources read today documents a visual convention for stale data (greyed, hatched). Any such treatment in Neritic would be our own design and should be tested, not cited.
