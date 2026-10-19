# Real-device acceptance checklist (iPhone and Android)

**Time:** about 30 minutes per phone.

**Setup:**
- Use the private preview named in PR #68.
- Write down the deployment id and the status centre's "Data published …" time.

**Recording:**
- Tick a box only when the check passed on the device in front of you.
- When something fails, write what you saw.
- People-based checks are in `real-device-test-script.md`.

| | iPhone (Safari) | Android (Chrome) |
|---|---|---|
| Model, OS, browser version | | |
| Network for steps 1–9 (mobile data, not Wi-Fi) | | |

## Touch and map responsiveness

- [ ] Cold start: open the preview from a closed browser. The first screen is readable in under ~2 s, and nothing jumps once you start reading.
- [ ] Pinch-zoom and pan Explore around Monterey Bay. The map follows the finger with no stutter longer than a moment.
- [ ] Tap the water once. One cell or point opens; the page does not scroll and nothing double-fires.
- [ ] Tap a harbour (Monterey). The sheet opens to half, and the layer chips are at its top.
- [ ] Tap "Satellite", then "Currents", from the chips. The map changes in under ~2 s each time, and the time row is in view.
- [ ] Every button and link is easy to hit with a thumb (no misses on the notices, the sheet, the dock).

## Reading in bright sunlight (outdoors, screen at arm's length, auto-brightness on)

- [ ] "Notices not verified" is readable on the map and on My Coast.
- [ ] The notice count ("N here" / "in CA") and the status dot are readable.
- [ ] A briefing's dates and "not reviewed by a person" are readable.
- [ ] The map's latitude limit and its label can be seen against the water.
- [ ] Take one photo of each screen for the record.

## Offline pack: save, persist, reopen

- [ ] My Coast, Monterey: "Save for offline use". It ends with "verified". Note its size and the time taken.
- [ ] Close the browser completely and reopen it. The pack is still listed as saved, with the same time.
- [ ] Airplane mode on. Close the browser completely again.
- [ ] Reopen the saved briefing:
  - it opens;
  - the offline banner shows the saved time;
  - the banner does not cover "Notices not verified".
- [ ] Open Explore from the briefing:
  - the saved map and forecast are drawn;
  - a harbour panel opens.
- [ ] Still offline, open a place that was not saved. It says it is offline, and nothing pretends to be live.
- [ ] Restart the phone, then repeat the airplane-mode reopen. The pack survived.
- [ ] Airplane mode off, reload. The banner goes and the status centre says what is current.
- [ ] Optional: fill the phone's storage, then try a new save. The app says storage is full and never says "saved".

## Result

| Device | Passed | Failed (what you saw) | Photos |
|---|---|---|---|
| iPhone | | | |
| Android | | | |

A phone passes only if every box is ticked or the failure is understood and accepted by the owner.
