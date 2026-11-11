# v11 kit: what the page tracks build from

## System and shell (`src/v4/ui/`, `src/v4/shell/`)
- `Tag {level: closed|delayed|restricted|watch|none, children?, testId?, className?}`: word + shape on a filled tint. `level="none"` defaults to "None on record"; the page must still add "not a promise it is open".
- `Unverified {count?: "7 of 7", explanation?, className?, testId="not-verified"}`: shield-slash + words, opens a popover with CDFW and CDPH links. Today's own `Unverified` in `today/ui` should be replaced by this.
- `Button` / `LinkButton {kind: primary|secondary|tertiary, size: md|lg, icon?, testId?}`; `LinkButton` also takes `href`, `external?`. 44 px default; primary 56 px on phones.
- `Figure {label?, value, unit?, age?, stale?, size: hero|value, testId?, className?}`: `stale` puts "(old)" on the value in words.
- `SectionHead {title, marker?, action?, as?: h1|h2|h3, id?, testId?, className?}`.
- `Freshness {items: {label, when, stale?, missing?}[], testId="freshness", className?}`: names only what is old; otherwise "Sources".
- `Pop {trigger, label, children, triggerClass?, testId?}`: native popover, a bottom sheet on phones. Words stay in the DOM.
- Classes: `.nx-display-hero` (place-name scale), `.nx-display`, `.nx-title`, `.nx-lede`, `.nx-figure`, `.nx-value`, `.nx-caps` (sentence case), `.nx-margin`, `.nx-sheet`, `.nx-plate`, `.nx-deck`, `.nx-tag[data-level]`, `.nx-unverified`, `.nx-btn` + `-primary|-secondary|-tertiary|-lg`, `.nx-link`, `.nx-pop-panel`, `.nx-dock`.
- Tokens: `--nx-r-8/12/16/24`, `--nx-section`, `--nx-sheet-pad`, `--nx-gutter`, `--nx-deck`, `--nx-link`, `--nx-ground-top`, `--nx-page`.
- `<TopBar active now slot? trust? over? floating?>`: default sticky solid; `over` transparent with a scrim, solid after 24 px, takes no height so a stage starts at y=0; `floating` absolute over a non-scrolling canvas.
- `trust` = `{official, status, fixture?, local?, generatedAt?}` from the server loader `loadTrust()` in `lib/data.ts`. Today adds `local` from `localNotices`. A page without a map calls `useReportTrust(trust)`.
- `Dock {active, absolute?}` is the one phone dock (60 px).
- Trust pill test ids: `notices-button`, `registry-disclosure`, `status-center` (readiness only, not clickable), `notices-count`, `notices-here` / `notices-statewide`, `fixture-banner`, `preview-disclosure` / `-short`; panel `notices-panel`, `status-panel`.
- Stopgap to remove when you own the file: a CSS override neutralises `.uppercase` on SVG text in `LinkedSeries.tsx` and `HoverReadout.tsx`; `--nx-font-caps` is an alias to the UI font because `LinkedSeries` still reads it.

## Motion (`@/v4/motion`; CSS is global)
- `Reveal {as?, group?, ...html}`: chapter blocks below the fold only. `group` staggers direct children 50 ms, first 5. `useInView(ref, {once=true, rootMargin, threshold})`.
- `useDrawOnce(ref, {ms=700, threshold=.25})` returns `"idle"|"wait"|"draw"|"done"`. `DrawOn {as?, ms?, threshold?}` scopes `.nx-draw-line` (needs `pathLength={1}`), `.nx-draw-wipe` (dashed), `.nx-draw-area`, `.nx-draw-bar`, `.nx-draw-marks`.
- `Reveal` and `DrawOn` hide nothing under `navigator.webdriver`. Set localStorage `nx-motion-force-v1=1` to test the real behaviour.
- `Sheet {open, onClose, title, children, head?, modal=true, closeLabel?, testid?, className?}`: native `<dialog>`. `modal={false}` keeps the map usable (Explore's place sheet).
- `Fold {title, children, open?, defaultOpen?, onOpenChange?, summary?, heading="h3", id?, className?, triggerClassName?, bodyClassName?, testid?}`: body stays in the DOM, inert while closed.
- `Segmented {value, onChange, options:[{value,label,disabled?,testid?}], label, mode="radio"|"tabs", panelIdFor?, className?, testid?}`. `useGlide(boxRef, key, {thumb, on, storageKey})`.
- Route transitions are mounted in `MotionRoot`; nothing to do per link. `data-no-transition` makes a link cut. `useTransitionRouter()` gives `{...router, push, replace}`. Only pathname changes transition. Put class `nx-vt-stage` on Explore's map container so it morphs (`OceanStage` already has it).
- Scroll-linked hero: `.nx-scroll-stage`, `.nx-scroll-title`, range `--nx-scroll-range` (default 60svh). `OceanStage scrollLinked` applies the stage class.
- `usePlayback(frames, {intervalMs=900, loop=false, autoStart=false, initial=0})` returns `{index, frame, playing, play, pause, toggle, seek}`. Put `data-playback-control` on the Play button.
- `useFocusLink()` returns `{focus, setFocus, bind}`. `<li {...bind(recordId)}>`; `OceanStage` reads the store unless given `focus`.
- `useLate(loading, {delay=300, min=300})`, `<Skeleton className? style? lines?/>`, `<Late loading fallback>children</Late>`.
- `.nx-hover-tint`, `.nx-hover-lift`, `.nx-press`.

## Stage (`@/v4/stage/OceanStage`, server loader `loadStageData({data?, region?, now?})` from `@/v4/stage/data`, returns `StageData|null`)
Camera `center, zoom, phoneZoom?, pitch(0-50), bearing`; content `field, currents, data, port?, marks?:[{id,lon,lat,kind:"port"|"site"|"buoy",label?}], lines?, focus?`; behaviour `interactive=false, caption? (object|false), arrival?, alt?, scrollLinked?`; plumbing `className, style, testid, onPort?, onPoint?, onMap?, children`. The page sets the height. Children sit at the bottom. Mark arrival beats with `data-beat="1|2|3"`.

```tsx
const data = await loadStageData({ data: loaded, now }); // server
<OceanStage center={[-121.79,36.8]} zoom={9.4} pitch={45} bearing={28} field="chlorophyll" currents
  data={data} port={592} arrival="port-592" scrollLinked style={{height:"68svh"}}>
  <div><h1 data-beat="1">Moss Landing</h1><p data-beat="2">…</p></div>
</OceanStage>
```
Known: streaks are striking over `field="none"`, subtle over chlorophyll. The caption shows only with the map; words are always sr-only. Not yet verified: the stage in daylight mode, Today's offline pack with the stage mounted, Safari/Firefox. On phones, keep the title and caption from colliding with map labels (size the title or zoom).

## Chart kit (`@/v4/charts/kit`)
Helpers `scaleLinear`, `timeTicks(domain,max)`, `placeLabels(ys,gap,[lo,hi])`, `FAMILY`. `ChartFrame {width,height,title,desc,draw?,className?,testid?}`, `TimeAxis {x,domain,y,ticks?,format?,gridTo?,max?}`, `NowLine {x,y0,y1,label?}`, `Band {x0,x1,y0,y1,family?,label?,line?,labelX?,labelY?,anchor?}`, `Readout {x,y0,y1,width,lines:[{text,family?}],boxWidth?}`, `ChartLine {d,family,dash?,width?}`, `ChartArea {d,family,opacity?}`, `ChartBars {bars,family,baseline,width}`, `ChartMarks`, `DirectLabel {x,y,family?,anchor?}`.
