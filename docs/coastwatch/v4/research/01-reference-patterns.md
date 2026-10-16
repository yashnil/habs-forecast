# Neritic v4: reference interaction patterns

Research date: 2026-10-16. Sources were read as documentation, help pages, source HTML and design write-ups. The live WebGL apps (GFW map, MyOcean Pro, nullschool) are JavaScript-only and could not be driven directly, so anything learned only from secondary sources or memory is marked **unverified**.

---

## 1. Patterns per reference

### 1.1 Global Fishing Watch map
Source: https://globalfishingwatch.org/user-guide/ (GFW-UG); platform-update archive https://globalfishingwatch.org/platform-update/view-archived-platform-and-data-updates (GFW-PU).

1. **One left sidebar holds the layers, and each row has the same five actions.** A layer is on when its toggle shows the layer's colour and off when it is grey. Every row carries the same small icons: `i` (methodology), filter, paint (colour), compass (centre the map on this layer) and delete. A drag handle changes the stacking order. *Takeaway:* the on/off state is the layer's own colour, so the toggle doubles as a legend swatch. Metadata sits one click away and never stays expanded. (GFW-UG)
2. **The legend lives inside the layer row and tracks the cursor.** When a layer is on, a colour scale appears under its title. The scale updates with zoom, and a marker on it shows the value under the cursor. *Takeaway:* the legend belongs to the active layer and works as a live readout. (GFW-UG)
3. **The timebar is a bottom band with handles, presets and automatic granularity.** You drag the start and end handles, or open a calendar with presets such as "Last 30 days". Aggregation follows the range: daily under 3 months, monthly up to 6 years, yearly beyond that. Buttons switch the zoom between years, months, days and hours. Clicking a year or month drills down into it. The timebar shows **where each dataset begins and ends**. If the range includes the latest data, it stays pinned to "latest". Otherwise it snaps to the middle of the previous range. (GFW-UG; GFW-PU Dec 2022)
4. **The app explains why something is empty and offers the fix.** Playback will not run when the end handle is at today, because data lags by 72 hours. If a selected vessel has no activity in the range, a popup offers to change the range. *Takeaway:* show the data's latency limit in the UI and offer the fix in place. (GFW-UG)
5. **Inspection works in three steps: hover, click, pin.** Hovering a grid cell gives a quick hours-of-activity readout. Clicking opens a popup listing the top vessels in that cell. "See vessel" pins the vessel to the sidebar and draws its track. Cursor latitude and longitude always show in the lower right. (GFW-UG)
6. **A workspace is a saveable, shareable URL.** The share icon copies a link that restores the same context, and the save icon names and stores the workspace. Reports, opened from an area's graph icon, can also be saved, printed, downloaded or shared. The share-link URL format is **unverified**. (GFW-UG)
7. **Change is coloured as a diverging pair.** Before/after and period-comparison views colour cells blue for a decrease and red for an increase. (GFW-UG; GFW-PU Dec 2021)
8. **Mobile behaviour is not documented (unverified).** The guide says nothing about mobile layouts, loading states or no-data states.

### 1.2 Copernicus MyOcean Pro
Source: https://help.marine.copernicus.eu/en/articles/6482737-introduction-to-myocean-pro-viewer (CMEMS).

1. **Adding a layer goes through a catalogue sheet, and provenance is built in.** "Add layer…" (upper left) opens an embedded Data Store with search and filters. You expand a product, pick a variable and click "Add to map". The product sheet shows the Product ID, **DOI**, User Manual and Quality Information Document. A **Notifications tab** lists maintenance and incidents. *Takeaway:* every layer carries its citation and a feed of known issues. (CMEMS)
2. **Each layer has a gear menu for colour map, min/max and opacity, plus a variable switcher.** A layers icon inside the config swaps variables within the same product, so the layer stays where it is. The `i` button opens the product page. (CMEMS)
3. **The time slider shows each layer's coverage and offers a fix.** The greyed part of the bar is the layer's coverage. If the chosen date falls outside it, an orange **"Fix"** button appears and jumps to valid data. Left and right arrow keys step through dates, and the mouse wheel over the bar zooms the time range. Temporal resolution follows the dataset, so there is no resolution picker. (CMEMS)
4. **The depth slider appears only when the data has depth.** It sits bottom right. Up and down arrows move in 100 m steps, and the mouse wheel gives finer steps. *Takeaway:* a dimension control appears only when the active product has that dimension. (CMEMS)
5. **Point, line and area tools each produce graphs.** A point gives a time series, a depth profile and a time-by-depth plot. A line gives a transect. An area gives an area average and a histogram. Graphs can be grouped for comparison by dragging one pin box onto another, and exported as netCDF, CSV or SVG. (CMEMS)
6. **One share link captures the whole state.** The link holds layers, configuration and geometries, so it doubles as "save and resume". Share also offers a PNG, a video or GIF and an embed. Pressing **Z** (Zen mode) hides the tools for clean screenshots. (CMEMS)
7. **Download reuses the view.** "Define on map" takes an existing geometry, and "As in map" copies the current dates and depths. "Automate" generates the equivalent CLI or Python command, which serves as the reproducibility path. (CMEMS)
8. **Colour-blind mode is a global setting.** It sits under General settings → Map. Loading and no-data visuals are **unverified**.

### 1.3 earth.nullschool.net
Sources: https://earth.nullschool.net/about.html (NS-About); the open-source predecessor's DOM https://raw.githubusercontent.com/cambecc/earth/master/public/index.html (NS-HTML); https://github.com/cambecc/earth (NS-README).

1. **One word opens the whole UI.** The only persistent chrome is the word **"earth"** at the bottom left (`#show-menu`, title "menu"). It opens a compact text menu, and the **e** key toggles it. Everything else is the field. (NS-HTML, NS-About)
2. **The menu is a stack of labelled text rows, not a panel of widgets.** The rows are **Date** (with a local/UTC toggle), **Data**, **Scale** (a gradient bar), **Source**, **Control** (`Now`, `«` `‹` `›` `»`, current position, grid), **Mode** (Air/Ocean), **Height** (surface, then pressure levels in hPa), **Overlay** and **Projection**. Short text buttons such as "TPW" and "MSLP" replace icons. *Takeaway:* the source and the valid time are labelled rows in the same menu as the controls, so they are never hidden. (NS-HTML)
3. **A status line and a progress line are the only loading UI.** `#status` and `#progress` sit in the details block. There are no spinners over the map. Exact wording such as "Downloading…" is **unverified**. (NS-HTML)
4. **Clicking a point gives one readout with a close button.** `#location` shows the coordinates, the wind value and the overlay value. Each unit is a toggle (`#location-value-units`), and a ✕ clears it. There is a single inspected point, not a stack of popups. (NS-HTML; behaviour **unverified** live)
5. **The keyboard covers the time model.** **n** jumps to the latest data, **j/k** step backward and forward, **Shift-C** opens a calendar, **i/m** move up or down in height, **p** toggles animation, **g** toggles the grid and **Esc** closes. (NS-About)
6. **The hash URL holds all state.** Configuration lives in the URL hash, which gives deep links and back-button history. Model and hash update each other two ways. (NS-README)
7. **The about page states the caveats plainly.** It gives the cadence of each source (weather and waves every 3 h, currents and SST daily). It notes that OI SST is "preliminary if less than two weeks old" and that there is "no guarantee of accuracy". It also admits a CO₂ offset that is "not scientifically valid". (NS-About)
8. **Particle flow gives an impression of precision.** Particles are re-seeded after reinterpolation (NS-README). The motion looks continuous and exact even though the 0.25° to 1° model data is not. Section 4 covers this risk.

### 1.4 Linear
Sources: https://linear.app/enablement/guide/navigating-linear (LIN-Nav); https://linear.app/now/how-we-redesigned-the-linear-ui (LIN-Redesign); https://linear.app/blog/behind-the-latest-design-refresh (LIN-Refresh).

1. **Cmd/Ctrl+K opens a command menu that runs any action or jumps to any page by name.** **/** searches, and typing an identifier (`LIN-123`) jumps straight to it. (LIN-Nav)
2. **Two-key "G then X" navigation:** G then I goes to Inbox, G then M to My Issues. **?** opens the full shortcut sheet. Shortcuts are mnemonic. The mnemonic claim comes from a third-party review and is **unverified**. (LIN-Nav)
3. **An "inverted L" frame (sidebar plus top bar) controls the content.** Headers hold the filters and display options, and side panels hold meta properties. Share, copy-link and similar actions sit in a **consistent header position**. (LIN-Redesign, LIN-Refresh)
4. **Navigation recedes.** The sidebar is "a few steps dimmer", inactive labels are muted, and the principle is "Don't compete for attention you haven't earned." Separators without a purpose were removed, and the rest were softened ("structure should be felt rather than seen"). There are fewer and smaller icons, with no coloured icon backgrounds. (LIN-Refresh)
5. **The colour system is generated in LCH from three inputs:** base, accent and contrast. Surface elevation is derived from those, and the contrast input yields high-contrast themes for accessibility. The palette moved from cool blue-grey to a warmer, less saturated grey. (LIN-Redesign, LIN-Refresh)
6. **Type:** Inter Display for headings, Inter for everything else. No published size or spacing tokens were found (sizes **unverified**).
7. **Shortcut hints in tooltips highlight modifiers as they are pressed.** This comes from a secondary source and is **unverified**.

### 1.5 Emil Kowalski (skills repo plus essays)
Sources: https://github.com/emilkowalski/skills, specifically `skills/emil-design-eng/SKILL.md` (EK-Eng), `skills/animate/SKILL.md` (EK-Anim) and `skills/animation-performance/SKILL.md` (EK-Perf); https://emilkowal.ski/ui/great-animations (EK-Great); https://emilkowal.ski/ui/building-a-drawer-component (EK-Vaul); https://emilkowal.ski/ui/building-a-toast-component (EK-Sonner); https://vaul.emilkowal.ski/snap-points (Vaul-Snap).

1. **How often an action happens decides whether it animates.** At 100+ times a day (shortcuts, command palette): no animation. At tens of times a day (hover, list navigation): nearly imperceptible or none. Occasional actions (drawers, toasts): standard. Rare ones (first-run): some delight. **Keyboard-initiated actions never animate.** "Raycast has no open/close animation." (EK-Anim, EK-Great)
2. **Choose easing by intent, never ease-in.** Entering or exiting uses ease-out. Moving on screen uses ease-in-out. Hover and colour use `ease`, and constant motion uses linear. The curves: `--ease-out: cubic-bezier(0.23,1,0.32,1)`, `--ease-in-out: cubic-bezier(0.77,0,0.175,1)`, `--ease-drawer: cubic-bezier(0.32,0.72,0,1)`. (EK-Eng, EK-Anim)
3. **Duration bands:** press feedback 100–160 ms, tooltips and small popovers 125–200 ms, dropdowns 150–250 ms, modals and drawers 200–500 ms. UI motion stays under 300 ms. Exit is faster than enter. (EK-Eng)
4. **Interruptibility:** rapidly triggered UI uses CSS transitions, which retarget mid-motion, not keyframes, which restart from zero. Gestures use springs (`duration 0.5, bounce 0.2`, bounce at most 0.1–0.3 and only on drag-dismiss). (EK-Eng, EK-Great)
5. **Origin and scale:** never animate from `scale(0)`; start at 0.95–0.97 with opacity 0. Popovers scale from the trigger (`transform-origin: var(--transform-origin)`), while modals scale from the centre. The first tooltip waits for a delay, and adjacent tooltips then open instantly at 0 ms. Pressed state is `scale(0.97)`. (EK-Eng)
6. **Vaul drawer:** `transform` over **500 ms** with `cubic-bezier(0.32,0.72,0,1)` (iOS sheet feel). It dismisses on momentum: velocity above **0.11 px/ms** dismisses whatever the distance (EK-Eng). Dragging past the top is damped. Drag only starts when the content is scrolled to the top, then waits a **100 ms** lockout. Extra touches are ignored. Snap points are fractions or pixels, and `snapToSequentialPoint` stops a flick from skipping one. `modal={false}` keeps the background interactive, and `fadeFromIndex` sets the point where the overlay starts fading. The mobile keyboard is handled through the `visualViewport` resize event. (EK-Vaul, Vaul-Snap)
7. **Sonner toast:** `transform 400ms ease` from `translateY(100%)`, 4 s by default. The timer pauses on hover and when `document.hidden`. A swipe dismisses at velocity above 0.11. Stacked toasts sit about 14 px apart and scale down by 0.05 per index. A toast exits the way it entered. (EK-Sonner, EK-Anim)
8. **Performance:** animate only `transform` and `opacity`. `clip-path` is acceptable, and `height` only for accordions. Never use `transition: all`. Blur stays at or under 20 px, and 2–5 px is enough. Never `setState` per frame. Avoid motion shorthands that animate CSS variables, and never animate a CSS variable on a parent. Use `will-change` only as a last resort. CSS and WAAPI stay smooth while the main thread is busy, for example while MapLibre is parsing tiles. (EK-Perf)
9. **Reduced motion means fewer animations, not zero.** Keep opacity and colour, and drop translate and scale (example: `fade 0.2s ease`). Gate hover effects with `@media (hover:hover) and (pointer:fine)`. (EK-Eng, EK-Anim)

---

## 2. Neritic v4 motion spec (from Kowalski)

Tokens: `--ease-out: cubic-bezier(0.23,1,0.32,1)`, `--ease-in-out: cubic-bezier(0.77,0,0.175,1)`, `--ease-drawer: cubic-bezier(0.32,0.72,0,1)`. Animate only `transform` and `opacity`, never `transition: all`.

| Surface | Enter | Exit | Notes |
|---|---|---|---|
| Mobile bottom sheet, detent change or open | 500 ms `--ease-drawer`, `translateY` | 300 ms `--ease-drawer` | Driven by the finger during drag, with no transition while dragging. Snaps on release. Dismisses on velocity above 0.11 px/ms. `snapToSequentialPoint` on. `modal={false}` at the peek and half detents. |
| Desktop side panel (inspector, Research rail) | 240 ms `--ease-out`, translateX 16 px plus opacity | 160 ms `--ease-out` | Panel slides; the map never resizes mid-animation. Apply the map resize once, at the end. |
| Popover or menu (layer picker, share) | 150 ms `--ease-out`, scale 0.97→1 plus opacity, origin at trigger | 100 ms | **0 ms when opened by keyboard.** |
| Tooltip | 125 ms opacity plus scale 0.97 after a 400 ms first delay (delay value is a Neritic choice; Kowalski gives none) | 80 ms | Adjacent tooltips open at 0 ms. Hover-capable pointers only. |
| Selection highlight (map feature, list row) | **Instant** paint change. An optional 120 ms `ease` on the outline opacity only | instant | Selection never flies, pulses or zooms by itself. |
| Layer toggle (raster opacity on the map) | 150 ms linear `raster-fade-duration`, or 0 | 0 | Swapping the legend and inspector content is instant. |
| Timebar step (j/k, arrows, playback) | **Instant** frame swap. The playhead moves without a transition | — | Keyboard-driven and frequent, so no animation. Playback is the only "motion" there is. |
| Toast | 400 ms `ease`, `translateY(100%)`→0 | 200 ms, same edge | 4 s default, longer for errors. Pauses on hover and while the tab is hidden. Stale or offline status is a persistent banner, never a toast. |
| Button press | `scale(0.97)` 120 ms `--ease-out` | — | Pointer only. |
| Map camera fly-to (on explicit "zoom to harbour") | 600–900 ms MapLibre `easeTo` | — | Only when the user asked for the move. Under reduced motion, use `jumpTo`. |

**Reduced motion (`prefers-reduced-motion: reduce`):** drop every translate and scale. Sheets and panels cross-fade in 150 ms of opacity, and the sheet detent jumps. Camera moves become `jumpTo`. Playback auto-advance is off by default and needs an explicit press. Raster fade becomes 0. Opacity and colour transitions stay at 150–200 ms or less.

**Always instant:** anything triggered from the keyboard (shortcuts, the ⌘K palette open and close, timebar stepping, layer hotkeys), selection state, hover state on map features, legend and value readouts, text in the inspector, Esc closing anything, tab switches between Explore, My Coast and Research, and route changes.

---

## 3. Neritic v4 interaction contract

1. **Map first, chrome second.** The map fills the Explore viewport, and on desktop every control floats over it. Persistent chrome is limited to the top bar, one layer control, the active-layer legend and the timebar when it applies (nullschool restraint, Linear's receding navigation).
2. **Floating surfaces handle quick, contextual tasks; full panels handle reading.** Popovers and the layer menu are floating, under 320 px wide, dismissed by Esc or a click outside. The inspector and Research views are docked panels that **push** the map, never float over it, so the readout and the selected feature are never covered.
3. **One layer selector.** There is a single "Layer" control, a radio-style list grouped by kind (observation, model nowcast, forecast, official notice). Exactly one *primary* scalar field is visible at a time. Context overlays (stations, notices, closures, harbours) are separate checkboxes in the same menu. No stacked rasters.
4. **Each layer row's swatch is its on-state** (GFW). The `i` icon opens the provenance sheet (product ID, DOI or URL, cadence, latency, known incidents), as in the MyOcean product sheet.
5. **The legend belongs to the active layer.** One legend, always the active primary layer's: units, scale, a "no data" swatch and a cursor-value marker on the scale (GFW). Switching the layer swaps the legend instantly.
6. **Provenance always shows next to the legend:** source name, **valid time** (with a local/UTC toggle), product kind (Observed, Model, Forecast) and age, as in nullschool's Date and Source rows. "Observed 2 d ago" is never styled like a forecast.
7. **The timebar appears only for temporal products.** It shows only when the active layer has a time dimension. Like MyOcean's depth slider, it is absent otherwise, never shown disabled.
8. **The timebar shows coverage, latency and "latest".** Greyed spans mark where the active product has no data (GFW and MyOcean). A **"Latest"** button pins the view to the newest valid frame and stays pinned as new data arrives (GFW). Selecting a date outside coverage shows an inline **"Jump to nearest data"** fix (MyOcean's "Fix").
9. **Timebar granularity follows the product's cadence:** a daily product steps by day and a monthly climatology by month. There is no free resolution picker.
10. **Forecast frames are visually marked on the timebar,** for example with a hatched span after "now" and a "Forecast" label in the readout. Anything past the last observation is never shown without that marking.
11. **Inspection runs hover, then click, then pin.** Hover (fine pointer only) shows a one-line value chip at the cursor. Click opens the inspector panel for that point or feature. "Pin" adds it to a compare list, capped at about 4 items. There is a single active inspection at a time, with no popup stacks (nullschool `#location`).
12. **The inspector answers first:** value and units, valid time, source, then the evidence (time series where the product has one, nearest stations, official notices with their own dates). It states "no data at this point for this date" with the reason (cloud, land, outside coverage) and never shows a blank.
13. **The selected-feature highlight is a 2 px outline plus a 4 px halo** in the accent colour, painted as a MapLibre layer, so it survives zoom and theme changes. The same state appears in any list (Research table, My Coast harbours). Selection is instant and never animated.
14. **Cursor coordinates** show in the bottom-right corner on fine pointers (GFW), and tap coordinates appear in the inspector on touch.
15. **URL state is the save model.** The view, layer, time (or `latest`), selection and open panel serialise to the query or hash, the way nullschool, GFW and MyOcean all do it. Browser back undoes navigation-level changes such as layer, selection and panel, but not every pan. Map pans use `replaceState`, everything else `pushState`.
16. **Share copies the URL.** It sits in one consistent header position (Linear). "Copy link" confirms with a toast, and "Copy link at this time" freezes `latest` to an absolute date. Accounts and saved workspaces are out of scope for v4.
17. **Mobile Explore uses a bottom sheet with three detents.** Peek is 96 px (layer name, valid time, provenance chip). Half is 50% (legend, layer list or inspector summary). Full is 92% (complete inspector). The sheet is non-modal at peek and half (`modal={false}`), so the map stays live. Detents follow the Vaul physics in §2.
18. **Map taps on mobile go to the sheet,** which moves to the half detent with the inspector. There are no map popups on touch.
19. **Every interactive control in field and mobile contexts has a target of at least 56 × 56 px** (gloves, a moving deck), including timebar step buttons, sheet handles and layer rows. Desktop dense controls are at least 32 px, with 44 px hit areas.
20. **My Coast is answer-first and has no timebar.** The first screen states the status for the harbour (verdict, valid time, source, official-notice state) in text. The map is a supporting thumbnail that opens Explore with the same URL state.
21. **Research is a separate workspace with its own frame:** table and chart first, map secondary, and the same selection model and URL state. Its numbers are same-pipeline only.
22. **Keyboard set (no animation on any of it).** ⌘/Ctrl+K opens the command menu (go to harbour, switch layer, jump to date). **/** searches places. **L** opens the layer menu. **j/k** or ←/→ step time. **n** jumps to latest (nullschool). **[ / ]** cycle layers. **i** toggles the inspector. **Esc** closes the topmost surface. **G then E/C/R** goes to Explore, My Coast or Research. **?** opens the shortcut sheet. **Z** gives clean-map mode (MyOcean Zen).
23. **Loading:** a thin progress line under the top bar plus a status line in the provenance block ("Loading SST, 2026-10-15"), as in nullschool's `#status` and `#progress`. Never put a spinner or skeleton over the WebGL canvas. The previous frame stays visible until the new one is ready.
24. **Partial data:** coverage gaps render as an explicit no-data hatch, not transparency. The legend's "no data" swatch explains them. The inspector gives the reason.
25. **Stale data** (older than the product's expected cadence plus a grace period) shows a persistent amber chip: "Stale: last update 3 d ago (expected daily)". It lives on the provenance block and on My Coast's answer, never in a toast.
26. **Offline data** shows a persistent banner with the timestamp of the cached data ("Offline, showing data from 06:10"). Interactive controls that need the network are disabled with a reason. A cached briefing still renders.
27. **Errors are specific and recoverable,** with the source named and a retry action ("NOAA ERDDAP did not respond, retry"). A failed layer never blanks the map; the last good frame stays and is labelled with its time.

---

## 4. Avoid

- **Animated particle flow (nullschool style) for currents or HAB fields.** Continuous motion implies a resolution and certainty the 1–4 km daily or model data does not have, and it competes with the evidence. Use static arrows or streamlines at the product's own resolution, or nothing.
- **Smooth temporal interpolation or "tweening" between frames.** It invents states between observations. Frames swap discretely.
- **Glassmorphism (backdrop-blur) over the WebGL map.** It forces expensive recompositing of a moving canvas every frame (EK-Perf: blur is costly, worst in Safari). It also lowers legibility and contrast over variable imagery. Use opaque or near-opaque surfaces with a hairline border and a soft shadow.
- **Animated or count-up numbers.** Values are evidence. Ticking digits suggest live precision and delay reading, so values render final and at once.
- **Animating keyboard-driven actions** (palette, time stepping, layer cycling). These are high-frequency, and Kowalski says never animate them.
- **Auto-fly camera moves** on load, selection or data refresh. They disorient and break the URL-state mental model. Move the camera only on an explicit request.
- **Stacked popups or multiple open inspectors.** There is one inspection surface.
- **Showing a disabled timebar or depth control on products without that dimension.** Hide them (MyOcean).
- **Toasts for stale, offline or official-notice state.** These are conditions, not events, and they must persist.
- **A rainbow or jet colour map on the primary field,** and any colour that suggests toxicity for chlorophyll. Chlorophyll is not toxin. Use perceptually uniform ramps with a colour-blind-safe option (MyOcean setting).
- **Bounce or overshoot springs on panels.** Bounce is reserved for drag-dismiss. These are functional surfaces.
- **`transition: all`, animating `width`/`height`/`top`, or animating a CSS variable on a parent.** They cause layout thrash next to a busy MapLibre main thread.
- **Spinners or skeletons over the map canvas.** Keep the last good frame and show progress in the chrome.
- **Persistent decorative icons or separators** (Linear: "structure should be felt rather than seen").
