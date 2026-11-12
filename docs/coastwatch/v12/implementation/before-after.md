# v11 → v12: before and after

The same pages at the same sizes as v11's stills in [`../../v11/after/`](../../v11/after/) (1440 × 900 at 1x;
390 × 844 at 2x, so 780 × 1688), followed by v12 alone at 320 × 640 and 844 × 390.

**Read the pairs with two differences in mind.** v11's stills are real staging data from 2026-11-09; v12's are the
fixture dataset on the fixture clock (2026-10-13), so dates and values differ while the records' kinds and
states are the same (Moss Landing, Dungeness crab: season delayed, whale measures, a CDPH test, NWS sea, a past
C-HARM run). v11's default was dark: its plain stills pair with v12 in **Dusk**, its `-day` stills with v12 in
**Day**, which is v12's default; v12 Day versions of the dark-only pairs are added. Reproduce:
`node scripts/before-after-shots.mjs <baseUrl>` on a fixture build. v12 files are under
[`before-after/`](before-after/).

## Matched pairs

| Page | v11 (before) | v12 (after) |
|---|---|---|
| Today, desktop | [today-desktop](../../v11/after/today-desktop.png) | [today-desktop](before-after/today-desktop.webp) (Dusk) |
| Today, desktop, day | [today-desktop-day](../../v11/after/today-desktop-day.png) | [today-desktop-day](before-after/today-desktop-day.webp) |
| Today, phone | [today-phone](../../v11/after/today-phone.png) | [today-phone](before-after/today-phone.webp) (Dusk) |
| Today, phone, day | [today-phone-day](../../v11/after/today-phone-day.png) | [today-phone-day](before-after/today-phone-day.webp) |
| Explore, no place | [explore-desktop](../../v11/after/explore-desktop.png) · [phone](../../v11/after/explore-phone.png) | [explore-desktop](before-after/explore-desktop.webp) · [phone](before-after/explore-phone.webp) (Dusk) |
| Explore at the port | [explore-place-desktop](../../v11/after/explore-place-desktop.png) · [phone](../../v11/after/explore-place-phone.png) | [explore-place-desktop](before-after/explore-place-desktop.webp) · [phone](before-after/explore-place-phone.webp) (Dusk) · [day](before-after/explore-place-desktop-day.webp) |
| My Coast, no port | [coast-desktop](../../v11/after/coast-desktop.png) · [phone](../../v11/after/coast-phone.png) | [coast-desktop](before-after/coast-desktop.webp) · [phone](before-after/coast-phone.webp) (Dusk) |
| My Coast at the port | [coast-brief-desktop](../../v11/after/coast-brief-desktop.png) · [phone](../../v11/after/coast-brief-phone.png) | [coast-brief-desktop](before-after/coast-brief-desktop.webp) · [phone](before-after/coast-brief-phone.webp) (Dusk) |
| My Coast at the port, day | [coast-brief-desktop-day](../../v11/after/coast-brief-desktop-day.png) · [phone](../../v11/after/coast-brief-phone-day.png) | [coast-brief-desktop-day](before-after/coast-brief-desktop-day.webp) · [phone](before-after/coast-brief-phone-day.webp) |
| Research, Santa Cruz Wharf | [research-desktop](../../v11/after/research-desktop.png) · [phone](../../v11/after/research-phone.png) | [research-desktop](before-after/research-desktop.webp) · [phone](before-after/research-phone.webp) (Dusk) · [day](before-after/research-desktop-day.webp) |

**Best starting pair:** Today, desktop, day. v11 led with the port's name over a dark satellite field and a
fisheries list; v12 leads with the season statement in Literata, the next decision date, and two soundings
placed where they were measured on a nautical chart of the port's water.

## What changed between the pairs

- **One chart, persisting.** v11 put a satellite image behind panels on Today and My Coast; v12 is one chart
  (depth tints, isobaths, relief) shared by every page, with the page's view of it and a ruled margin.
- **Statement first.** Today and My Coast open on the official status and the next date, each value with a
  stamp (kind, source, date, "not verified"); v11 split them across a header, chips and a list.
- **Readings live where they were measured** (the test at its test area, the buoy at the buoy), not in KPI rows.
- **Explore** fills the sheet with the data's own extent, stops at the data's edge, and carries one legend and a
  short time rail; v11's dock and layer popover are gone.
- **Research** keeps the record as strips on one time axis beside the chart where the stations sample.
- **Colour** is ink and paper with one foreign hue (magenta for regulation); data ramps live only on the water.
- **Offline**: every page now says when it is a saved copy and what is and is not available (not visible in
  these online stills; see `test-migration.md` and `npm run test:offline`).

## v12 only: the smallest phone and a phone on its side

| Page | 320 × 640 | 844 × 390 |
|---|---|---|
| Home (first visit) | [home-320x640](before-after/home-320x640.webp) | [home-844x390](before-after/home-844x390.webp) |
| Today | [today-320x640](before-after/today-320x640.webp) | [today-844x390](before-after/today-844x390.webp) |
| Explore | [explore-320x640](before-after/explore-320x640.webp) | [explore-844x390](before-after/explore-844x390.webp) |
| My Coast | [coast-320x640](before-after/coast-320x640.webp) | [coast-844x390](before-after/coast-844x390.webp) |
| Research | [research-320x640](before-after/research-320x640.webp) | [research-844x390](before-after/research-844x390.webp) |
| Official notices | [notices-320x640](before-after/notices-320x640.webp) | [notices-844x390](before-after/notices-844x390.webp) |

Motion is in [`motion/`](motion/README.md); performance against v11 in [`performance.md`](performance.md).
