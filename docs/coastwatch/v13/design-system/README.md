# Neritic v13 design system: what changed over v12

This is a delta. Everything in [`../../v12/design-system/README.md`](../../v12/design-system/README.md) still holds
(identity, colour tables, the truth rules, the lint) unless this page says otherwise. Code lives in
`coastwatch-web/src/nx/ds` (components, tokens, motion), `src/nx/shell` (the bar, the workspace) and
`src/nx/chart` (the persistent chart). Import components from `@/nx/ds`.

One principle is amended. v12 said "the chart is the product" on every page. v13 says: **the page's work takes the
space it needs, and the chart is context for it.** On Today, Explore and Notices the chart is the work. On My Coast
and Research the work is a form, a ledger and plots, and the chart shows where that work is.

## 1. Layout: Desk and Chart

A route declares its mode by rendering `<PageLayout>`. It renders a hidden marker that the stylesheet reads
(`.n12-app:has([data-nx-layout="desk"])`), so the server's HTML is already laid out correctly and nothing moves at
hydration.

| Mode | Routes | Wide (≥1100 × ≥520) | Narrow or short (721–1099 wide, or under 520 tall) | Phone (≤720) |
|---|---|---|---|---|
| `chart` (default) | `/`, Today, Explore, Notices | A margin column (340–372 px) beside the chart. An optional `ChartDock` along the chart pane's bottom | The same | The chart, with the margin as a bottom sheet (rest, half, reading) |
| `desk` | My Coast, Research | The desk takes the space; the chart is a pane on the right, sized by the reader | The desk has the full width; the chart opens over it from `ChartToggle` (Escape or its close button shuts it) | The desk is the page and scrolls. The chart is a strip at the top (32svh) that scrolls away to a 48 px bar; "Larger" grows it to 62svh. No sheet |

Pane sizes beside the desk: `rail` (48 px, the chart out of sight), `context` (clamp(320px, 28vw, 440px), the
default), `split` (50%), `chart` (the desk shrinks to 420 px). The reader drags the divider or uses the keyboard
(Left and Right step sizes, Enter or Space collapses to the rail and back, Home and End go to the ends). The size
is kept per route on the device (`localStorage["nx-pane-v1"]`).

Rules the implementation keeps:

- **One tree.** `Workspace` (`src/nx/shell/Margin.tsx`) renders the same elements at every width, before and after
  hydration. The phone sheet's handle and "More" control are always in the tree and shown by the stylesheet. A page
  is never remounted when the phone media query resolves.
- **One map.** The chart host is sized by its pane and keeps its single MapLibre instance through every mode and
  size, including `rail`.
- **No per-frame map resize.** While the divider is dragged the chart is clipped at its old size. On release the
  pane takes the nearest size, the map resizes once, and the camera refits with `pan`.
- On a phone desk page the map does not take touches until the strip is enlarged, so a swipe on it scrolls the page.

```tsx
import { PageLayout, Desk, DeskHeader, DeskBody, DeskMain, DeskAside, DeskFoot, DeskSticky, ChartDock, ChartToggle, useChartPane } from "@/nx/ds";

<PageLayout mode="desk" paneKey="coast" defaultPane="context" />
<Desk>
  <DeskHeader context="Moss Landing · Dungeness crab · Hypothetical" title="A trip from Moss Landing"
              titleProps={{ "data-testid": "coast-title" }} status="…" actions={<ChartToggle />} />
  <DeskBody layout="main-aside" aside="end">
    <DeskMain>…the primary visual and its result…</DeskMain>
    <DeskAside label="Trip inputs" sticky>…inputs…</DeskAside>
  </DeskBody>
  <DeskFoot>…provenance, method, limits…</DeskFoot>
  <DeskSticky>…the one result or action that must stay in reach on a phone…</DeskSticky>
</Desk>
```

| Component | Props | Notes |
|---|---|---|
| `PageLayout` | `mode: "chart" \| "desk"`, `paneKey?: string`, `defaultPane?: PaneDetent` | Renders only the marker. Without it a page is `chart` |
| `Desk` | div props | Container; a size container named `nx-desk` |
| `DeskHeader` | `context?`, `title`, `titleProps?` (data attributes for the h1), `status?`, `actions?`, `sticky = true`, `condense = true`, `pinned?`, `children?` | The page's one `<h1>`. Sticky, and after 64 px of scroll it condenses to one line of context and title, so port, fishery, "hypothetical" and official status never scroll away. Put those in `context` or `pinned` (kept when condensed); `children` fold away with the status |
| `DeskBody` | `layout: "main-aside" \| "main-full"`, `aside: "start" \| "end"` | Two columns once the desk is ≥880 px wide (aside 320–380 px). Write the aside first in the page for `"start"`: reading order is page order |
| `DeskMain`, `DeskFoot` | div / footer props | |
| `DeskAside` | `label` (required), `sticky?` | A labelled section |
| `DeskSticky` | div props | Sticks above the destinations on a phone; an ordinary block elsewhere |
| `ChartDock` | `label`, `height = 300`, `resizable?`, `onClose?`, `children` | Chart mode. Portals into the chart pane's bottom; the chart above refits. On a phone it reads in place in the sheet |
| `ChartToggle` | `className?` | Shown only where the chart is an overlay |
| `useChartPane()` | returns `{ mode, detent, setDetent, collapsed, open, setOpen, toggle }` | `toggle` does the right thing for where the pane is |

The chart host also exposes `setChartHighlight({ lon, lat, label? } | [lon, lat] | null)` from
`@/nx/chart/highlight`: an ink ring at a place, for linking a plot's sample or a list row to the chart.

## 2. Type roles

Components name a role, never a raw size. Values are `desktop / phone`.

| Token | Value | Use |
|---|---|---|
| `--type-display` | 64 / 40 | Home statement, the big date |
| `--type-title` | 40 / 32 | A page or desk title (Literata 500) |
| `--type-head` | 22 / 20 | Section heads |
| `--type-lead` | 18 | Status sentence, lead copy |
| `--type-body` | 16 | Body, controls |
| `--type-small` | 14 | Provenance, axis ticks, footnotes only |
| `--figure-xl` | 64 / 44 | The one result of a view (Public Sans 600, tabular) |
| `--figure-l` | 36 / 28 | Secondary results |
| `--figure-m` | 22 | Ledger values, inputs |
| `--lh-tight`, `--lh-snug`, `--lh-body` | 1.1, 1.3, 1.5 | |
| `--measure` | 62ch | Line length |

New size steps `--t-20` and `--t-44`. New colour tokens in every mode table: `--wash` (a hover or surface ground, a
shade off paper) and `--sel` (a selection ground: ink at low alpha, never a hue).

## 3. Controls

| Component | Props | Behaviour |
|---|---|---|
| `Button` | `variant: "primary" \| "secondary" \| "quiet"` (`"text"` is an alias of quiet), `size: "m" (48) \| "l" (56)`, `icon`, `iconEnd`, `busy`, `busyLabel`, `disabled`, `full`, `href`, `external` | Secondary hover is a wash and a 3 px foot rule; it no longer inverts to an ink block. One primary per view |
| `ButtonGroup` | `align: "start" \| "end" \| "stretch"`, `wrap = true` | One height for all; stacks full width at ≤360 px. Save / Update / Duplicate go here with one primary |
| `IconButton` | `icon`, `label` (required), `size: 44 \| 48`, `direction?`, `ruled?`, `disabled?`, `onClick` | The label is the accessible name and the title |
| `Segmented` | `label`, `options: { value, label, hint? }[]`, `value: T \| null`, `onChange`, `hideLabel?`, `fit: "content" \| "equal"`, `size: "m" \| "s"`, `unsetWords = "Choose one"`, `disabled?` | The ink block slides to the new choice. **A null value is visibly unset**: dashed frame, no segment chosen, and "· Choose one" beside the label |
| `RangeControl` | `label`, `presets: { value, label }[]`, `value: T \| null`, `onChange`, `customLabel = "Custom"`, `onJumpLatest?`, `latestLabel?`, `action?` | Quiet words with a sliding underline; a null value shows "Custom". `onJumpLatest` is the "latest sample" action; its label should say the date |
| `ScrubNumber` | `label`, `value: string`, `onChange`, `step = 1`, `min = 0`, `max`, `prefix?`, `unit?`, `decimals?`, `hint?`, `error?`, `disabled?`, `size = 6` | A native text input you can also step (Up/Down, Shift ×10) or drag sideways on its label (mouse and pen). Empty stays empty |
| `Disclosure` | `summary`, `value?`, `defaultOpen?`, `open?`, `onOpenChange?`, `id?`, `children` | A row that opens in place; replaces the "+" rows |
| `ToggleRow` | `label`, `hint?`, `checked`, `onChange`, `onWord = "On"`, `offWord = "Off"`, `disabled?` | A 44 px setting row: a switch that says its state in words |
| `Hint` | `children`, `tone: "note" \| "caution"`, `onDismiss?` | A line of help where it is needed; dismissible for first-use guidance |
| `Figure` | `value: number \| null`, `label`, `size: "xl" \| "l" \| "m"`, `format: "money" \| "int" \| "dec1" \| "dec2" \| fn`, `cents?`, `unit?`, `kind?`, `waiting?`, `note?`, `layout: "stacked" \| "inline"` | A result figure. It moves from its old value to the new one and never counts up from zero. A null value shows `waiting`, never 0. The exact value is always in the accessible text |
| `Field`, `Stepper`, `Toggle` | unchanged from v12 | |

## 4. States

One vocabulary for every state a reading can be in. Each has one mark (shape first), one word and one legend
sentence.

```tsx
<DataState kind="not-collected" variant="block" what="pDA" reason="0 of 12 samples in this window measured pDA"
           date={iso} now={now} source="CalHABMAP" action={<Button onClick={jump}>Latest pDA sample: Mar 3, 2024</Button>} />
```

`DataStateKind`: `measured`, `reported-zero`, `below-detection`, `estimated`, `forecast`, `historical`, `stale`,
`missing`, `not-collected`, `unavailable`, `outside-coverage`, `unverified`, `offline`, `loading`.

| Component | Props |
|---|---|
| `DataState` | `kind`, `what?`, `date?`, `now?`, `source?`, `reason?`, `action?`, `variant: "inline" \| "block" \| "plot"`, `children?` |
| `DataStateMark` | `kind`, `size = 14` |
| `DataStateLegend` | `kinds: DataStateKind[]` |
| `Availability` | `label`, `variables: { id, label, spans: { from, to, kind, n? }[], latest?: string \| null, summary? }[]`, `domain: [ms, ms]`, `window?`, `now?`, `selected?`, `onPick?(id)`, `onJumpLatest?(id, iso)` |
| `Skeleton` | `label` (required words), `lines = 3`, `figure?`, `plot?: px` |
| `Recovered` | `at: iso`, `now?`, `children?` |

`DATA_STATE_WORDS[kind]` gives `{ word, legend }`. v12's `Unavailable`, `Stale`, `Offline`, `SourcesDiffer` and
`Loading` remain.

## 5. The plot kit (`@/nx/ds`, source in `ds/plot`)

d3 scales, SVG marks, an HTML tooltip. No new dependency. Data never tweens or smooths: a new window is a cut.

```tsx
<PlotSync onPlace={(lonlat) => setChartHighlight(lonlat)}>
  <Plot label="Particulate domoic acid at Monterey Wharf" data={pts} x={{ domain: win }} y={{ type: "log", unit: "ng/mL" }}
        height={{ desktop: 400, phone: 260 }} onSelect={setSample}>
    <GridY /> <AxisX /> <AxisY title="pDA, ng/mL" />
    <NoData spans={gaps} words="not collected" />
    <Threshold value={500} label="C-HARM threshold 500 ng/L" />
    <Line data={pts} gapDays={45} /> <Points data={pts} />
    <NowRule t={now} /> <Crosshair /> <Selection /> <Tip render={(p) => <SampleTip p={p} />} />
  </Plot>
  <Plot label="Water temperature" data={temp} x={{ domain: win }} height={96}>…</Plot>
</PlotSync>
<Brush extent={extent} value={win} onChange={setWin} handle={brushRef} label="every plot above" />
```

`PlotPoint`: `{ t: number; v: number | null; kind?: DataStateKind; raw?: string; place?: [lon, lat]; meta?: unknown; id?: string }`.
`t` is ms on a time axis.

| Part | Props | Notes |
|---|---|---|
| `PlotSync` | `value?: { hoverT, pinned }`, `onChange?`, `onPlace?(lonlat \| null)` | Plots inside share one crosshair and one pinned sample |
| `Plot` | `label`, `data: PlotPoint[]` (sorted), `x: { type?: "time" \| "linear", domain }`, `y?: { type?: "linear" \| "log", domain?, unit?, format? }`, `height: number \| { desktop, phone }`, `margin?`, `describe?`, `describePoint?`, `onSelect?(p \| null)` | One tab stop, `role="group"`, `aria-roledescription="chart"` |
| `AxisX` | `title?`, `ticks?`, `format?` | Labels never collide or run off the edge |
| `AxisY` | `title?`, `ticks?` | The title sits above the axis, never rotated |
| `GridY` | `ticks?` | |
| `Line` | `data`, `gapDays = 45`, `kind = "measured"` | Straight segments; breaks at a gap, a reported zero or a censored value. Computed kinds are dashed |
| `Points` | `data`, `r?` | A mark per state: measured ● solid; reported zero □ hollow on the baseline; below detection ▽ hollow; estimated ◇; forecast and unverified ○; stale dashed ○; missing, not collected, unavailable, outside coverage × on the baseline |
| `NoData` | `spans: [from, to][]`, `words?` | Hatched and worded |
| `Threshold` | `value`, `label` (required) | Named above the line at the plot's left. Not drawn off-scale |
| `NowRule` | `t`, `label = "Now"` | |
| `Crosshair` | `horizontal?` | Follows at once, no easing |
| `Selection` | none | The pinned sample's ring |
| `Tip` | `render?(p)` | Stands beside the sample on the side with room |
| `PlotState` | `kind`, `what`, `reason?`, `action?`, `height = 280` | Stands where a plot would be when there is nothing to draw |
| `PlotTable` | `caption`, `columns: { id, head, cell(row), numeric? }[]`, `rows`, `rowKey`, `selected?`, `onSelect?`, `empty?` | The same samples as a real table |
| `Brush` (v12, extended) | adds `inset?: { left, right }`, `handle?: Ref<BrushHandle>` | `handle.current.jumpTo(t, width?)` moves the window to hold a moment. `windowAround(extent, window, t, width?)` is the pure form |
| `WaterfallChart` | `steps: WaterfallStep[]` (each may carry `formula`), `label`, `preview?`, `domain?`, `height`, `cents?`, `children?` | Tall bars on a money axis. Hover, focus or Left/Right on a bar shows its amount and formula. `preview` draws a second scenario as outlines |
| `BreakEvenMarker` | `label = "Break-even"` | A child of `WaterfallChart`: the named rule at zero |
| `SensitivityGrid` | `caption`, `rows`, `cols`, `rowAxis`, `colAxis`, `value(r, c)`, `format?`, `selected?`, `onSelect?`, `onPreview?`, `base?` | A keyboard grid of figures with the break-even line drawn as a contour between cells. No colour ramp |
| `usePlot()` | returns the plot's scales and state | For a custom mark |

Interaction contract (tested in the kit, so pages get it for free):

- **Mouse:** the crosshair snaps to the nearest sample within 24 px; a click pins it; a click on empty space clears.
- **Touch:** a tap pins the nearest sample (48 px reach). A long press (250 ms) then a drag scrubs. A vertical swipe
  scrolls the page.
- **Keyboard:** Left and Right step sample by sample; PageUp and PageDown a month; Home and End the first and last
  in view; Enter or Space pins; Escape clears. Each step is read by a polite live region as
  "Jun 3, 2025 · 6.2 ng/mL · measured".
- **Zoom** is only the brush and the range presets, so it is predictable.

Pure helpers for tests and pages: `describePoint`, `nearest`, `stepIndex`, `runs`, `onAxis`, `yDomainOf`,
`linearTicks`, `windowAround`, `breakEvenSteps`, `plain`.

v12's `SeriesChart`, `SmallMultiples`, `Waterfall` (the ledger) and `SensitivityTable` are still exported and
unchanged; new work should use the kit.

## 6. Motion additions (`ds/motion.ts`)

| Preset | Value | Use |
|---|---|---|
| `reflow` | spring 420 / 42 | Motion's `layout` prop: a desk reflowing, a list reordering |
| `pane` | 0.42 s, chart ease | The chart pane opening or closing |
| `shared` | 0.36 s | `layoutId` shared elements. In use: the nav underline (`nx-dest`, `nx-tab`) |
| `select` | 0.2 s | The segmented block, the range underline, a switch's knob |
| `reveal` | 0.22 s | A disclosure's body |
| `value` | spring 260 / 34 (critically damped) | `useSettledNumber(v)` and `Figure`: old value to new, no overshoot, never from zero |
| `swap` (`swapOut` 0.09 s, `swapIn` 0.18 s after it) | | A route's page changing, in sequence: the old page out, then the new one in. Applied to the page's view transition in `shell.css` |
| `scrollCondense` | 64 px | `useScrollCondense()`: a sticky header condensing |

In use: a change of the pane's size beside the desk moves the columns with `pane` (the map is never resized per
frame: growing, it takes its new size at once and the column uncovers it; shrinking, it is clipped and resizes
once on arrival). The collapsed rail says what it holds in words ("Chart · Moss Landing"); it has no thumbnail.
A page change is sequenced, and a test reads the view transition's own timings to hold it there
(`tests/e2e/v13-layout.spec.ts`).

Reduced motion still means no animation: `MotionPrefs` completes every Motion animation at once, and the view
transition rules are switched off under `prefers-reduced-motion`.

## 7. Shell

- The bar's Alerts link goes to `/alerts?place=…` (`alertsHref` in `@/nx/shell/unread`). `/alerts` currently
  redirects to the alerts section of `/account`; the alerts page replaces that redirect.
- The unread count beside Alerts is the alerts feature's own: the shell reads `useUnread()` from
  `@/nx/alerts/useInbox` (undefined until hydrated, shown only above zero). `<AlertWatch />` is mounted once in the
  route group's layout.
- The bar's port finder calls `choosePort` (the first port a reader picks becomes their home port).
- The menu's "Account" is "Your data" (same `/account`). Ask and Alerts are in the menu at 380 px and under.
- A page joins the shell by living in `src/app/(neritic)/` and being added to `NERITIC_PATHS`
  (`src/nx/shell/LegacyChrome.tsx`). `/ask` and `/account` still live outside it with the pilot shell; moving
  them is their owner's page work, and the shell needs nothing more for it: render
  `<PageLayout mode="desk" defaultPane="rail" paneKey="…" />` and a `Desk`.

## 8. Added for the pages (block 1 of phase B2)

### Today and Explore

| Part | Props | Notes |
|---|---|---|
| `GateList` | `label`, `gates: GateItem[]`, `level: 2 \| 3` | `GateItem`: `{ id, title, authority?, state: "ok" \| "old" \| "loading" \| "unavailable" \| "none", headline, tone?: "plain" \| "regulation" \| "closure", date?, facts?: { id, text, role?, old?, stamp? }[], relation?, actions?: { id, label, href, external? }[] }`. Gates stay separate; an old or missing gate is quieter and says so. A fact with `role: "advisory"` is set heavier |
| `Advisories` | `label`, `items: { id, text, source, href?, area? }[]`, `none?` | The first-viewport slot: render it before the dates at every width. With no items it shows `none` (say which source was read), or nothing when `none` is not given |
| `Kind` | adds `"advisory"` | An agency's warning in force. Drawn with the official ring (`markFor("advisory") === "official"`) |
| `Sounding` | `stale?` | A stale figure is a step smaller, regular weight, in `--ink-2`, with "(old)" beside it |
| `Readout` | adds `subtitle?`, `actions?`, and `state: "outside"` | `subtitle`: what was sampled ("Cell 3 km, C-HARM grid"). `outside`: the point is outside the product's footprint; its own mark (`outside-coverage`), not the no-view hatch. `actions`: two or three links, shown in every state |
| `MarginSticky` | div props | Stays at the foot of the margin column while it scrolls (a legend while a reading is open). Static on a phone |
| `ChartDock` | adds `align: "pane" \| "frame"`, `variant: "panel" \| "rail"` | `align="frame"` starts and ends the dock's content where the chart's frame does (`--frame-l`, `--frame-r`), so a time rail lines up with the chart's edges. `variant="rail"` is a thin strip |
| Phone sheet | `openMargin("peek")` | A detent below rest (about 30 % of the screen). The chart's free area follows it. "More" goes from peek to rest |

Chart API (`src/nx/chart`):

| Call | From | What it does |
|---|---|---|
| `keepChartPointInView(lon, lat, margin = 56)` | `control` | Pans just enough to bring a place clear of the phone sheet (measured at its present height), a panel and the frame's edge. Already clear: nothing moves |
| `zoomChart(±1)`, `refitChart()`, `fitChartBounds(bounds, maxZoom?)` | `control` | What the chart's own buttons do; a page may call them too |
| `scene.controls` | `scene` | The chart's buttons (zoom in, zoom out, back to the page's view) show wherever the reader may pan. `controls: false` hides them; `controls: { fit: { label, bounds } }` adds a "fit the data" button |
| `scene.view.tall` | `scene` | A second box, used when the free area is at least 1.25 times taller than wide (a desk page's pane). `tallBoundsAround(at, kmAlong, kmAcross?)` builds one. `useChartShape()` (`runtime`) gives `{ width, height, tall }` for a page that needs more than that |
| `useChartObstacle(id, ref)` | `cover` | Chart lettering keeps clear of this element (a locator inset) |
| `registerChartObstacle(id, (project) => Rect[])` | `cover` | Lettering keeps clear of something drawn in chart coordinates; `lineObstacle(a, b, pad)` makes the rects for a transect line. Returns the function that removes it |
| `#nx-chart-marks` | | Kept, unchanged |

Latitude labels on the neatline are written whole or not at all (never cut by the frame's corner or by the
resting sheet).

### My Coast

- `WaterfallChart` takes any number of steps. `orientation: "auto" | "columns" | "rows"`: rows (one bar per line)
  whenever a column would be under 88 px, so eight steps read at 390 px. Each step's `formula` is its worked sum.
- A one-variable sweep is a `Plot` on a linear axis:

```tsx
<Plot label="Net to boat as catch changes, hypothetical" data={sweep} x={{ type: "linear", domain: [0, 800], unit: "lb" }}
      y={{ unit: "$", format: (v) => usd(v) }} height={{ desktop: 320, phone: 220 }}>
  <GridY /> <AxisX title="Catch, lb" /> <AxisY title="Net to boat" />
  <Line data={sweep} kind="estimated" /> <Points data={[yours]} />
  <BreakEvenMarker at={breakEvenLb} label={`Break-even ${breakEvenLb} lb`} />
  <Crosshair /> <Tip />
</Plot>
```

  `BreakEvenMarker { label?, at? }` works in both: the zero rule in a waterfall; in a plot, the zero rule and a
  vertical rule at `at`. Give sweep points `kind: "estimated"` so they are never drawn as measurements.
- `EditableText { label, value, onChange, placeholder = "Untitled", maxLength = 60, size: "head" | "body" }`: a name
  edited in place. Enter or leaving keeps it, Escape restores it, an empty name is never kept.
- `Drawer` (the `Dialog` props without `placement`): from the right on a wide screen, from the bottom on a phone.
- `ButtonGroup` for Update / Save as new / Duplicate; `Segmented size="s"` for a compact two-option toggle;
  `DeskSticky` for the phone's sticky result; `Figure cents` for the hero.

### Research

- `OptionList { label, hideLabel?, options: { value, label, meta?, end?, disabled? }[], value, onChange }`: the
  variable rail. A radio group; a variable with nothing behind it stays listed, disabled, saying why in `meta`.
- `Inspector { title, subtitle?, onClose?, actions?, children }` with `Facts { items: { label, value }[] }`: the
  detail card for a pinned sample. A ruled card on a wide screen; on a phone it is pinned above the destinations,
  at most half the screen tall.
- `Brush`: when the frame is under 72 px wide the two handles stand outside it and face outward, so they never
  overlap; the frame between them still pans. `handle.current.jumpTo(t, width?)` as before.
- Plot mark tokens, set in each chart mode: `--plot-cross`, `--plot-focus`, `--plot-point`, `--plot-censored`,
  `--plot-zero`, `--plot-model`, `--plot-grid`, `--plot-gap`.

### Storage keys this system writes

`nx-pane-v1` (localStorage): `{ [paneKey]: "rail" | "context" | "split" | "chart" }`. Nothing else.

## 9. Ownership

`src/nx/ds`, `src/nx/shell`, `src/app/globals.css`, `src/app/system` and `scripts/design-lint.mjs` have one owner
(the art-direction agent). Pages compose these; they do not add `position: fixed` panels, raw d3 charts or their
own button styles.

## 10. Lint rules added in v13 (`npm run lint:design`)

The v12 rules are unchanged. The page rules below apply outside `src/nx/ds`, `src/nx/shell`, `src/nx/chart` and
`src/app/system`. Warnings are counted in the run's last line and listed with
`node scripts/design-lint.mjs --warnings`; they do not fail it. Any line can be excused with a reason:
`/* design-lint-allow <rule>: <why> */`.

| Rule | Level now | What it wants | Becomes an error |
|---|---|---|---|
| `margin-width` | error | No `var(--margin-w)` in a page's stylesheet. The workspace sizes the page | already |
| `fixed-position` | warning | No `position: fixed` in a page. Lay out in the `Desk`, the margin, a `ChartDock`, an `Inspector` or a `Drawer`. A real overlay (a coach mark) takes an allowance with its reason | when Today's tour carries its allowance |
| `raw-chart` | warning | No `d3-scale` or `d3-shape` import in a page. Draw with the plot kit (§5) | when Research's strips and Explore's transect are on the kit |
| `raw-button` | warning | No hand-styled `<button>` in a page (one finding per file, with the count). Use `Button`, `IconButton`, `Segmented`, `Disclosure`, `OptionList`, `RangeControl` | when the count reaches zero, or each remaining one has an allowance |
| `one-primary` | warning | At most one `<Button variant="primary">` per file | with `raw-button` |
| `small-text` | warning | At most six uses of 14 px text (`--t-14`, `--type-small`) in a page's stylesheet | with `raw-button` |

Counts when the rules were added: raw-button 14 files, small-text 3 stylesheets, raw-chart 3 imports, one-primary
1, fixed-position 3 (all in Today's tour).

## 11. The catalog and the kit's browser tests

`/system` has a plate for every v13 part (Desk, Controls, Figure, DataState and Availability, the plot kit,
WaterfallChart with SensitivityGrid and a sweep, GateList and Advisories) in each chart mode (`?mode=dusk`).
`tests/e2e/v13-system.spec.ts` drives them in a browser: the plot's keyboard, mouse and touch contract, the
figure's travel, the unset choice, the waterfall's worked sums and the grid's keyboard. Pages get that behaviour
by composing the parts; they do not need to re-test it.

For tests only, `window.__nxShell.openMargin(detent)` and `window.__nxChart.keep(lon, lat)` call the page-facing
requests as a page would.

## 12. Changes after the pages were built (block 4)

- **`DeskHeader` condenses without moving the page.** What the header loses in height it keeps as a margin under
  itself, and it is never the scroll anchor. A page no longer needs `overflow-anchor: none` for it. On a phone
  its `actions` take their own line under the title.
- **Data states.** A fifteenth kind, `rejected` ("Rejected by QC"): kept in the record, never drawn on the axis.
  The default sentence for `reported-zero` no longer says "none": it claims neither a limit nor an absence.
  `DataStateLegend` takes `words` (a sentence per kind) and `DataState` takes `word`, for a source's own terms.
- **Plot.** `PlotSync` accepts `pinned: { t }` with no plot id: every plot inside shows its sample there as
  pinned. With under 24 px above the plot, `AxisY`'s title takes the top tick label's place.
- **`OptionList`** options take `strip`: a full-width decorative strip under the row's words.
- **`Segmented size="s"`** segments are 44 px tall.
- **`SensitivityGrid`** previews only for a mouse (a touch chooses), has tighter cells in a narrow column and
  scrolls inside itself.
- **`WaterfallChart`** keeps every figure inside the drawing: the amounts' column is as wide as the widest
  figure, and a wide net figure over the last bar is set back from the edge.
- **`GateList`** links are each their own 44 px target in the flow; stacked links no longer share a hit area.
- **`MarginSticky edge="top"`** keeps a legend at the head of the margin while it scrolls to a reading under it.
  The margin's scroller has scroll padding, so something scrolled into view keeps a line of air above it.
- **Phone sheet.** Brought back to rest from a raised detent it shows the page from its first line. The handle's
  grab area reaches 12 px above the sheet's edge.
- **Menu.** The chart-mode choice is read each time the menu opens.

## 13. Changes from the independent review

- **Phone sheet (F-01).** The sheet's scroller ends where the visible sheet ends at every detent (above the
  destinations bar), so a page's last control is reached by scrolling without raising the sheet.
- **Chart buttons on a phone (F-06).** At 720 px and under the zoom buttons are not shown (a pinch zooms). What
  remains is one 44 px column in the chart's foot corner: back to the page's view and, when a scene names one,
  "fit the data" as an icon with its name.
- **`PageLayout chartLabel` (F-15).** The phone strip's bar and the collapsed rail say "Chart · …" only with
  what the page passes (`<PageLayout mode="desk" paneKey="research" chartLabel={station.name} />`). Without it
  they say "Chart": never a port's name over a chart that shows something else.
- **Names keep clear of a zone's boundary line** (a parallel drawn by the regulation layer).
- **`WaterfallChart` (F-03).** A step that cannot be drawn yet is a short open mark in its column; what each one
  waits for is one sentence under the chart (`data-testid="waterfall-waiting"`), never a label per bar.
- **Checkboxes (F-12).** A native `<input type="checkbox">` inside the shell is drawn in the system's terms in
  every chart mode: an empty ruled box, or an ink box with a paper tick.
- **`DeskHeader` gives its room in the same commit as it condenses** (from its two measured heights), so the
  page is never laid out with the room missing and no page needs `overflow-anchor` for it.
- **A sticky `DeskAside` and `DeskSticky` have paper under them** and stand above the column that scrolls beside
  them (F-17).
