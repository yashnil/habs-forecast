# Today at your port: design directions

The top half of `/today` (header, status strip, sea and weather) was drawn three ways as static pages over the real Neritic tokens (Ocean Instrument, `src/v4/theme.css`). Each was shot at 390 px (phone) and 1440 px (desktop). The content is the real Monterey Bay situation on 2026-11-08.

| | Phone | Desktop |
|---|---|---|
| **A. Instrument plates.** Ruled rows inside one plate, a coloured tick, and a status chip at the right. The sea shown as five day columns of bars. | `a-plates-phone.png` | `a-plates-desktop.png` |
| **B. Harbour board.** Solid red and amber tiles in a grid, large numerals, and a freshness pill for each source. | `b-board-phone.png` | `b-board-desktop.png` |
| **C. Almanac (chosen).** A hero with the port name, a one-sentence answer, and contour lines. The status set as sentences. The sea as shared time lanes. | `c-almanac-phone.png` | `c-almanac-desktop.png` |

## Choice: C, with one idea each from A and B

**Why C.** Fishermen ask for "a timely, accessible, and easily understood" signal (Pomeroy & Culver 2026). C is the only direction that answers before it shows numbers:
- The **lede sentence** says the most important official fact and today's wind in one breath, for example "Commercial Dungeness is delayed to at least Jan 1 for whale entanglements. Northwest 10–15 kt this morning."
- Below it, each **status row** reads as a sentence: fishery, then state, then cause.
- The **time lanes** for wind and seas take in five days at a glance. They also make room for the bloom timeline further down the page (days 0–3 C-HARM, then the 8-day chlorophyll outlook) in the same visual grammar.

C keeps the Neritic editorial voice: the Newsreader display serif for places and the lede, Plex for reading, and mono for values. It also sits calmly on the dark 4 a.m. ground.

**Why not B.** The big solid tiles are glanceable, but they read as a traffic light, which is exactly the "green = safe" frame we refuse. The neutral tiles also look empty and broken next to the loud ones. B is the loudest of the three, and the owner asked for calm.

**Why not A.** A is honest and tidy, but it is busy: chips, ticks and pills compete in every row. The meaning sits in grey 14 px text under a bold label.

**Taken from A:** a status *word* in each row (Delayed / Restricted / Watch / None on record). Together with an icon shape (filled triangle / half / ring), a row never relies on colour alone (R25).
**Taken from B:** freshness in plain words for each source ("NWS 3:15 AM · buoy 12 min ago · tests Sep 28"), shown once in the header.

## The rules the build follows
- **Status colours:**
  - red = official closure or delay (`--nx-down`);
  - amber = official restriction, or a process or signal to watch (`--nx-stale`);
  - the calmest state is **"No restriction on record"** in blue-grey (`--nx-offline`).

  There is no green anywhere in the status strip. All three tokens already carry 7:1 daylight values.
- **Family colours keep their meaning:**
  - measured tests are teal;
  - C-HARM is violet;
  - the chlorophyll outlook is violet, hatched and labelled "not a toxin forecast";
  - sea and weather are sky blue.
- **One idea per section.** Each section opens with a plain sentence and then shows its chart. Every chart has a text alternative and a source line with its date.
- **Field use:**
  - controls are 44 px minimum, and the primary actions are 56 px;
  - a one-tap **Sun** toggle in the header switches to the existing high-contrast daylight tokens (`html[data-daylight]`);
  - the dark tokens are the default for 4 a.m.
- **Lean:** no map library in `/today`. The charts are small hand-drawn SVGs, and the page is server-rendered from `buildPortToday`.

## As built (fixture data, recorded NOAA answers of 2026-10-13)
- **Phone:** `today-phone-dark.png`, `today-phone-day.png` (bright-sun mode), `today-phone-dark-full.png` (the whole page), `today-picker-phone.png` (no port).
- **Desktop:** `today-desktop-dark.png`, `today-desktop-day.png`, `today-desktop-day-full.png`.

On desktop the hero gains an "At a glance" column, and the sections sit in two columns: status beside sea, tests beside bloom, whale rules beside what changed. The DOM order stays the phone order, so headings and screen readers follow the same hierarchy.
