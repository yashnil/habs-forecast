# Direction A: motion

Motion here is rare and quiet. It confirms that something moved where the reader looks; it never decorates, and it never touches a value.

| Interaction | What moves | Duration and easing | Reduced motion |
|---|---|---|---|
| Forecast day (segmented control, map) | The dark indicator slides to the chosen day; the map swaps the day's cells instantly | 200 ms, `cubic-bezier(0.77, 0, 0.175, 1)` (on-screen move) | Indicator jumps; no slide |
| Layer row (map panel) | The chosen layer's legend and provenance open beneath it; the previous one closes | 200 ms, `cubic-bezier(0.23, 1, 0.32, 1)`, height via `grid-template-rows` | Opens and closes instantly |
| Phone sheet (map) | The sheet rises from peek to half on drag or tap; a flick above 0.11 px/ms snaps | 450 ms, `cubic-bezier(0.32, 0.72, 0, 1)`; drag follows the finger with no easing | Snaps without travel |
| Fly to a place (map) | Camera eases to the place; the place's notices and evidence update after it lands | at most 800 ms, ease-out | `jumpTo` |
| Marker to evidence (map and My Coast) | Selecting a station draws a thin rule from the marker to its row in the evidence list and highlights the row for 1 s | rule 160 ms ease-out; highlight fades over 600 ms | Highlight only, no drawn rule |
| Press feedback (buttons) | `scale(0.97)` | 140 ms ease-out | None |

Never animated:
- values or counts (no count-ups);
- chart marks between days;
- notices appearing;
- legend changes;
- time scrubbing (instant swap);
- anything on first load.

Implemented in the prototype:
- the day indicator (`.seg-ind`);
- the layer-row expansion (`.layer-more`);
- the press feedback on the menu button;
- the arrow nudge on "the 12-month record" link.

All of them sit behind `prefers-reduced-motion`.
