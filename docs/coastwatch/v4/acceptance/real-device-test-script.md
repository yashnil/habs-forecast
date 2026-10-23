# Real-device and people test script (v4)

Every v4 check so far has been scripted or headless. This script is for the first checks on real
phones and with real people. It takes about two hours across both parts. Record what happened, not
what should have happened.

## Before you start

- **Build:** the private preview named in PR #68 (Vercel login or the share link). Write down the
  deployment id and the date shown in the status centre ("Data published …").
- **Devices:** one iPhone (Safari, iOS 18 or later, any size; a small one if you can) and one Android
  phone (Chrome, Android 13 or later). Record model, OS version and browser version.
- **Network:** do part A on mobile data, not Wi-Fi, at least once.
- **Consent:** tell each participant this is a research preview, that notices in it are not verified,
  and that no decision about fishing or eating seafood should rest on it. Record no names; use P1, P2, ….

## A. Device checks (you, about 30 minutes per phone)

| # | Do | Pass when | Note |
|---|---|---|---|
| A1 | Open the preview cold (closed browser) | The first screen paints in under ~2 s; nothing jumps after you start reading | Time it with a stopwatch |
| A2 | Explore: pinch, pan, tap the water at Monterey | The map follows your finger; a tap opens one cell, not a scroll | Note stutter, double taps |
| A3 | Open the notices control with a harbour open | The panel is above everything; the agency's wording opens and scrolls | |
| A4 | My Coast → Crescent City → "Choose waters on the map" → tap the water | The new briefing opens at its top, named by coordinates | |
| A5 | Research → "Map tools" → Satellite → tap a pixel | The pixel's date and footprint show; an empty pixel says "no observation" | |
| A6 | My Coast Monterey → "Save for offline use" | Finishes and says "verified"; note the size and the time taken | |
| A7 | Airplane mode on, close the browser fully, reopen the saved briefing | It opens from the pack with the offline banner; Explore shows the saved map | Note anything blank |
| A8 | Still offline: open a place that was not saved | It says it is offline; nothing pretends to be live | |
| A9 | Airplane mode off, reload | The banner goes; the status centre is "Ready" | |
| A10 | Outdoors in sunlight, at arm's length | Notices, dates and "not verified" are readable | Photo of the screen |

## B. Sessions with people (moderated, about 20 minutes each)

About 5 fishers (commercial and recreational, shellfish harvesters if possible) and 5 researchers
(HAB scientists or students), each on their own phone where possible. One moderator, one note-taker.
Read each task as written. Do not point, hint or explain the interface. If someone is stuck for two
minutes, note it and move on.

**Fishers**

1. "You fish out of Monterey. Find what the latest observations say about the water there."
2. "Make Monterey your home port, then look at Crescent City, then get back to Monterey."
3. "Check whether any official notice covers the waters you would fish. Read what the agency actually said."
4. "You are taking out a boat tomorrow. Use the checklist the app gives you."
5. "Save what you need for a trip with no signal, then show me it works with the phone offline."

**Researchers**

1. "Find Santa Cruz Wharf's particulate domoic acid record and export it."
2. "How much of Monterey Bay did the satellite actually see in its latest view, and when?"
3. "Read the satellite chlorophyll and the HF radar current at one point in the bay, with their times."
4. "Draw a transect across the bay and tell me which pixels have no observation."
5. "Compare the currents with chlorophyll. Are they from the same time?"

**After each person, ask (open questions, write their words):**

- "What does this app tell you, in one sentence?"
- "Is there anything here you would act on? Why or why not?"
- "Was anything confusing, or did anything look more certain than it should?"
- Fishers only: "Does anything here say the water or the seafood is safe?" (The right answer is no;
  note any "yes".)

## What to record

For every task: completed (yes / with help / no), time, number of wrong turns, and the person's words
at the moment they found or missed the answer. Note any misreading of these:

- an observation read as a forecast;
- chlorophyll read as toxin;
- an unverified notice read as verified;
- a gap or "no observation" read as clean or calm;
- the logo or the name mistaken for an official agency.

## Results sheet (copy per person or device)

| Task | Done | Time | Wrong turns | Their words | Misreading |
|---|---|---|---|---|---|
| | | | | | |

Report the results as observed: who (P-number and role), on which device, what failed. These are the
first real-user data for Neritic; they are not a statistical study and should not be described as one.
