# Direction B — motion

Motion here is instrument feedback, not decoration. It confirms a control moved, keeps you oriented when the view changes, and never touches the data. Distilled from Emil Kowalski's skills, Impeccable and Vaul (scratchpad/design/refs/REPORT.md).

## Curves and durations (tokens in b.css)
| Use | Duration | Curve |
|---|---|---|
| Press feedback (buttons, chips, segments) | 140 ms, `scale(.97)` | `--ease-out` (0.23, 1, 0.32, 1) |
| Segmented-control indicator, row highlight | 200 ms, transform and width only | `--ease-out` |
| Bottom sheet snap | 460 ms | `--ease-sheet` (0.32, 0.72, 0, 1) |
| Map fly-to to a place or station | 700 ms or less (`easeTo`), never zooms out and in | MapLibre default ease |
| Popover/readout open | 150 ms opacity + `scale(.96→1)` from its anchor | `--ease-out` |
| Exits | about 70 % of the entrance | `--ease-out` |
Never `ease-in`, bounce or elastic. Only transform and opacity animate.

## What moves
- **Segmented control**: one indicator slides under the pressed segment (`b.js segmented()`); interruptible because it is a CSS transition, not a keyframe.
- **Sheet (phone)**: three snaps (peek / half / full). Drag follows the finger 1:1 (no transition while dragging); a flick faster than 0.11 px/ms moves one snap in its direction; past the top it damps to ¼ speed. The panel's content scrolls only at the full snap, so a drag that starts on content never fights scrolling. Keyboard: the handle is a button; Enter toggles half/full.
- **Marker ↔ evidence**: hovering or tapping a station row draws a ring on its map marker and tints the row (200 ms); tapping a marker does the reverse. Tapping a row also eases the map to it (700 ms).
- **Readout**: appears at the tapped point, anchored to it.

## What never moves
- **Values.** No count-ups, no interpolation between two days, two hours or two runs. Switching a day or a dataset swaps the layer instantly (raster fade is set to 0); at most a 120 ms opacity crossfade of the whole layer, never a morph between values.
- **Notices, legends, warnings, the "not verified" mark**: they appear in place, at full strength, in the first frame.
- **Time scrubbing**: each step is an instant swap; holding an arrow key steps without animation.
- **Initial load**: nothing animates on first paint (no stagger, no hero motion), so there is no layout shift and nothing for hydration to race.
- Particles for currents are not used in this direction: static arrows show only the cells with radar coverage, so a gap stays a gap.

## Reduced motion
`prefers-reduced-motion: reduce` sets every transition to 0 ms (`b.css`), `easeTo` becomes `jumpTo` (`b.js reduced`), and the sheet snaps without travel. Colour and opacity changes that carry state remain.
