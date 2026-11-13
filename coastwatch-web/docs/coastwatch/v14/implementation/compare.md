# v14: the app beside the approved synthesis (assessment)

App at `feat/neritic-v14` ee6f7505a, 2026-11-13. Stills: `scripts/v14-reference-shots.mjs` (prototype, `reference/`) and
`scripts/v14-app-shots.mjs` (app, `actual-fixture/` + `actual-staging/`), device scale 1, reduced motion. Each composite is
reference | actual. Generated listings with the set-up log per state: [compare-fixture.md](compare-fixture.md),
[compare-staging.md](compare-staging.md). All 50 app states set up with every step passing, no page errors, no sideways scroll.

Values differ by design (fixture clock Oct 13; staging is live data); judged here: composition, typography, colour,
cartography and hierarchy. Fixture and staging agree on every judgement below (staging only changes the live values, e.g.
a gale warning instead of a small-craft advisory). Port control on Research says "Moss Landing" because the harness seeds
that home port; the prototype says "Near Santa Cruz". Not a defect.

Verdicts: **faithful** · **minor** (small differences, composition intact) · **significant** (named).

## Per state

| State | 1440×900 | 1280×720 | 1100×800 | 390×844 | 320×640 |
|---|---|---|---|---|---|
| trip-empty | minor (S3) | minor (S3) | minor (S3) | minor | minor |
| trip-done | minor (S3) | minor (S3) | minor (S3) | minor | faithful |
| today | faithful | **significant (S1)** | **significant (S1)** | minor (S4) | minor |
| research | minor (S6) | faithful | faithful | minor | minor |
| alerts | **significant (S2)** | **significant (S2)** | **significant (S2)** | minor | minor (S5) |

One line per state:

- **trip-empty 1440 / 1280 / 1100**: Floating chrome, layer chips, whale-zone plate, lit sea, equation sheet, the dotted empty slots and the net-to-boat card all match. Differences: the map frame (S3), and an extra row under the equation ("How crew is paid · After costs / Of gross / Day rate"), which the trip needs to work.
- **trip-done 1440 / 1280 / 1100**: The waterfall (column-end values, the break-even caliper, the net hatch), the what-if dumbbells and the $2,602.24 hero match closely. Same S3 frame, plus the payer chips (Shared / Boat) added to the extra row.
- **trip-empty / trip-done 390, 320**: Sheet order matches (net to boat hero, hypothetical trip, season warning, waterfall) and Save/Add fuel sits on the tabs. Differences:
  - the map strip is taller and frames Santa Cruz to Moss Landing (design issue 2, as asked);
  - a "A trip not saved yet · Not saved / Saved (0)" line sits between the hero and the trip header, pushing the season warning to the fold at 390×844.
- **today 1440**: Faithful. Ticket stub at Fort Bragg, the date as hero, tabs, whale timeline, buoy ticket, sea card with the NWS bar chart. The primary is "Follow Fort Bragg" (filled) because the port is not followed in this scenario.
- **today 1280 / 1100**: S1. The sea card overlaps the ticket.
- **today 390**: Matches the reference down to the domoic row. S4: the filled Follow primary is half under the destinations bar at the sheet's resting height.
- **today 320**: Matches. The sheet rests higher than the reference, so the map strip is shorter (about 170 px against about 340 px).
- **research 1440**: The sheet (station, chips, range, log plot with the selected May 27 flag, brush, sample card with the same-visit rows, Sample record) matches closely. S6: the CDPH test ticket is missing from the map, although Toxin tests is on.
- **research 1280 / 1100**: Faithful. The same-visit rows are compact (3-up) and the card fits without scrolling (design issue 3, as asked).
- **research 390 / 320**: Sheet content and order match. The map strip is taller and frames the station with its neighbours (issue 2), so at 390 the plot starts at the fold, and at 320 the sample value does. An extra UTC line and a "Details" link add height.
- **alerts 1440 / 1280 / 1100**: Was→now dumbbell, the strike-through "Jan 1 → Jan 15", the SAMPLE stamp and the device card bottom-left all match. Differences:
  - S2: the ticket runs off the bottom of the screen;
  - the Whale Zone 2 plate lacks its "○ Season delayed · fleet advisory" line (present on Today).
- **alerts 390**: Same order as the reference. The headline is the record's full sentence plus a "Before:" paragraph, so the ticket is longer; Open the record is still above the tabs.
- **alerts 320**: S5. The primary action is below the fold.

## Significant differences, ranked

1. **S1: Today at 1280×720 and 1100×800, the sea card collides with the ticket.**
   - Where: the bottom-left "Sea · NWS PZZ455" card spans to about x 1060 of 1280, under the ticket's left edge.
   - What it covers: the left of the ticket's "Why" label, the whale timeline's start ("Today" marker), the "Domoic acid test" label and the left end of the Follow button.
   - Expected: as in the reference, the sea card stops about 20 px short of the ticket. Narrow its forecast chart, or shorten the advisory text that widens it.
   - Live advisory text is long ("issued October 12 at 9:06PM PDT until …"); the reference used "through late tonight".
2. **S2: Alerts on desktop, the alert ticket carries the inbox and the settings, and runs off the screen.**
   - Where: below "Open the record · Today at Fort Bragg", the ticket also holds "1 alert here · Mark all read · Clear the list", then "Fort Bragg · 3 of 5 kinds…", "What alerts you" and the trigger switches.
   - Result: at 1440×900 the ticket reaches the bottom edge and is cut off; at 1280×720 and 1100×800 it is cut off after the inbox row.
   - Expected: the ticket ends at its two actions; the settings sit behind the device card's "What alerts you" link.
   - Secondary: the headline is the record's sentence over two lines, plus a "Before:" paragraph. The reference has a short title ("Commercial Dungeness delay moved") and the was/now row says the rest.
3. **S3: My Coast on desktop, the chart frame and the map objects around the port.**
   - Map frame: the reference frames Monterey Bay close, with Moss Landing mid-right. The app is zoomed out, with Moss Landing pushed down to the sheet's top edge at the bottom right.
   - CDPH ticket: drawn below the port, touching the sheet, instead of above-left of it.
   - Missing objects: the buoy tickets (46092, 46236; staging shows one only at 1100) and the NWS zone plate ("Monterey Bay · NWS PZZ535 · Small craft advisory · Gale warning Friday").
   - Expected: fit to the port's whale zone and the bay (as issue 2 does on the phone), with Sea now's buoy tickets and the zone plate.
4. **S4: Today at 390×844, the primary action rests under the destinations bar.**
   - At the sheet's resting height, "Follow Fort Bragg" is half hidden by the tab bar.
   - Expected: the primary row clears the tabs at rest, as "Following Fort Bragg / CDFW" does in the reference.
   - The live advisory block is taller than the example, which pushes it down.
5. **S5: Alerts at 320×640, the primary action is below the fold.**
   - The full headline and the Before paragraph push "Open the record" off the first screen.
   - Expected: it is on the first screen (the reference shows it with "Today at Fort Bragg" under it).
   - S2's short title would fix this too.
6. **S6: Research on desktop, no CDPH test ticket on the map while Toxin tests is on.**
   - The reference draws the "CDPH test · Monterey Bay · Sep 28 · Dungeness, highest 9.8 of 30 ppm" ticket by the stations.
   - The app shows only the station marks. My Coast does draw that ticket.

## Minor differences seen across pages (not ranked)

- **Land:** the app draws land with hillshade texture; the reference's land is flat pale `#EFF1E6`.
- **Extra regulation labels:** "ZONE 1/2/3" appear at the zone lines on the app.
- **Phone nav:** the brand pill has no "Layers" word button. It is an icon button plus a Menu button (the reference has "Layers" with its word at 390).
- **Alerts badge:** the "Alerts 1" count badge in the nav appears only on Alerts. The reference shows it on every page because its example has an unread alert; the app is right for its own state.
