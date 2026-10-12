# 30 — Welcome by purpose (optional)

**Status:** draft PR for review, built on My Coast (#38). Web only. No data or notice record changes.

## What it does

The existing first-visit card on the Ocean Map becomes a two-step welcome. It is still one card, not a second popup.

**Step 1: what CoastWatch is, then "What brings you here? (optional)"**

| Purpose | Step 2 asks only | Where the visitor lands | What changes (defaults only) |
|---|---|---|---|
| Commercial fishing | the waters (port, CDPH site), and whether that port is their home port | My Coast for those waters | My Coast starts filtered to commercial fishing |
| Recreational fishing or shellfish harvest | fishing or shellfish, and the waters | My Coast for those waters | My Coast starts filtered to recreational fishing or shellfish harvest |
| Ports and coastal operations | the port (saved as home port) | My Coast for that port | — |
| Research and science | where to start: station measurements (Bloom), My Coast, or the map tour | the chosen page | My Coast puts measurements, forecast coverage and observations before official information |
| Just exploring | nothing | the existing nine-step guided tour | — |

**Controls:** Skip, the X and Back are always there. Skip or close on step 1 saves nothing, and looking at step 2 saves nothing until the visitor finishes it.

**Later visits:** the welcome does not return. A visitor who chose waters sees one small "Your waters: Crescent City →" link on the map. The tour is always available from the map's Tour control.

**Changing purpose:**
- "Set up for" on My Coast (at the end of the page) and on About.
- "Show the welcome again" on About clears the purpose and the chosen waters.

**Storage:** on this device only (`cw-purpose-v1`, `cw-my-waters`, the existing `cw-home-port` and the intro flag). No account and nothing identifying.

**Links take precedence:**
- Any address with a parameter suppresses the welcome.
- On My Coast, a link's own `for=` filter wins over the saved purpose's default.

**Never changes:** values, notices, verification status, caveats or which pages exist. A test checks that the notices listed for a place are identical with and without a purpose. Official information stays in My Coast's summary and the masthead pill for every purpose.

## Screenshots (production data)

| Step | Desktop | Phone |
|---|---|---|
| Welcome | [1440](welcome/d1440-1-welcome.jpg) | [390](welcome/m390-1-welcome.jpg) |
| Commercial: waters | [1440](welcome/d1440-2-commercial-waters.jpg) | [390](welcome/m390-2-commercial-waters.jpg) |
| Lands in My Coast, filtered | [1440](welcome/d1440-3-commercial-coast.jpg) | [390](welcome/m390-3-commercial-coast.jpg) |
| Later visit: "Your waters" link | [1440](welcome/d1440-4-return.jpg) | [390](welcome/m390-4-return.jpg) |
| Recreational: shellfish | [1440](welcome/d1440-5-shellfish.jpg) | [390](welcome/m390-5-shellfish.jpg) |
| Research: where to start | [1440](welcome/d1440-6-research.jpg) | [390](welcome/m390-6-research.jpg) |
| Research: My Coast, measurements first | [1440, full page](welcome/d1440-7-research-coast.jpg) | [390](welcome/m390-7-research-coast.jpg) |
| About: change or reset | [1440](welcome/d1440-8-about-switch.jpg) | [390](welcome/m390-8-about-switch.jpg) |
| Small screens: welcome | [1024](welcome/d1024-1-welcome.jpg) | [360](welcome/m360-1-welcome.jpg) · [320](welcome/m320-1-welcome.jpg) (the card scrolls on a 640 px-tall screen) · [360, step 2](welcome/m360-2-commercial-waters.jpg) |
| "Just exploring" opens the nine-step tour | [1024](welcome/d1024-9-tour.jpg) | [390](welcome/m390-9-tour.jpg) |

**Final review (2026-10-12, production data, after My Coast merged):** at 320, 360, 390, 1024 and 1440 px:
- the notices pill or tab stays uncovered and opens the notices while the welcome is showing, and the welcome is still there afterwards;
- the last choice is never covered;
- nothing scrolls sideways;
- "Just exploring" starts step 1 of 9;
- a deep link (`/?region=…&port=…`) shows no welcome;
- on My Coast, a link's `for=` wins over a saved purpose, the saved purpose's default applies without one, and "Set up for" is visible.

The phone's 390 px welcome and step 2 images above were retaken after the small-phone change.

## Tests

**Browser:** `welcome.spec.ts`, 10 tests. They cover:
- choose, Back, choose again, pick waters, then land filtered;
- the return link;
- Skip and X saving nothing;
- exploring starting the tour, and the tour reopening;
- research order with identical notices;
- link precedence over the welcome and the saved purpose;
- change and reset;
- the phone shellfish path;
- on 360 and 320 px phones, every choice on screen in front of the layer sheet (it scrolls on a 640 px-tall screen), with the notices still one tap away;
- axe on both steps on desktop and phone.

**Unit:** `purpose.test.ts`.

The full suite passes: 211 browser and 211 unit tests.

**Changed existing test:** the M8 first-visit test now checks that the card never overlaps the place chip, instead of sitting below it. On desktop the welcome sits in the free top right of the map.
