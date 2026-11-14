# Neritic v13: interaction clips

The six recordings the owner's brief asks for, on the production build with fixture data (clock 2026-10-13),
recorded by `coastwatch-web/scripts/v13-clips.mjs` with Playwright's `recordVideo` and re-encoded with ffmpeg
(VP9; desktop 1440 × 900 scaled to 960 px wide, phone 390 × 844 at 1x; 20 fps). Each file is named
`<n>-<what>-<width>.webm`.

| Clip | What it shows |
|---|---|
| `1-map-transitions` | Today → Explore → My Coast → Research → Today from the destinations, then Back: the chart panning and re-framing under each page, and the page changing (chart mode to desk mode and back). This is also the page-transition clip. |
| `2-port-transition` | The selected port changed from the bar: Moss Landing → Bodega Bay → Moss Landing. The chart travels, Today is rebuilt for the new port. |
| `3-research-select-brush` | Santa Cruz Wharf, pDA. Desktop: reading along the plot, a sample opened, water temperature chosen on the rail and back, the 1 y range, the window dragged on the brush. Phone: a tap on the plot opens the sample sheet; Next steps through samples. |
| `4-economics-sensitivity` | The owner's case typed in (400 lb at $9, 30 gal at $8.24, $100, 0 %, one crew on 20 % after costs, shared): the result and the waterfall arriving, then the catch × price grid and "One at a time" moving the result. Phone: the fixed result bar while typing, then Results. |
| `5-explore-time-and-read` | A cell read on chlorophyll, the days played, then temperature and C-HARM. |
| `6-loading-offline` | Today, then My Coast loaded cold at Slow 4G with the cache off (placeholders at the page's own geometry, then the records), then the network removed: `/` opens the copy this device kept; Alerts, which this device never saved, says so. |

To record them again: `npm run build`, start the app on fixture data, then
`node scripts/v13-clips.mjs http://localhost:<port>` (`CLIPS=3,4` records a subset).

These are evidence for review, not a verdict. What the review found in them is in [../review.md](../review.md).
