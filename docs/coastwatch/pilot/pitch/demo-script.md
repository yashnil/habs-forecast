# Demo script: "Monday 5 a.m. at Moss Landing" (4 minutes)

Start on a phone-width browser, already loaded, wifi tested. Open with: **"This is a pilot demo, running on staging data. Some layers are labelled honestly as not verified."**

| # | Time | Click | Say |
|---|---|---|---|
| 1 | 0:00 | `/today?place=port-592` (Moss Landing) | "It's 5 a.m. I fish crab out of Moss Landing. This is the first thing I open." |
| 2 | 0:20 | Status strip | "Official status first. Commercial Dungeness: delayed to at least January 1 — whale entanglements, not toxin. Next CDFW assessment December 15–18. There is no green 'safe' — the best state is 'No restriction on record'." |
| 3 | 0:50 | Toxin tests | "Measured, not modelled: Monterey Bay crab tested clean on September 28, average 5.5 ppm against a 30 ppm limit. These are CDPH lab results." |
| 4 | 1:15 | Sea & weather | "This is why they come back daily: NWS zone forecast, buoy 46042, tides, water temperature." |
| 5 | 1:40 | Bloom timeline | "Days 0–3 are NOAA's C-HARM toxin-risk probability. Days 4–8 are a chlorophyll outlook — conditions, not a toxin forecast." |
| 6 | 2:05 | Whale rules | "The RAMP zone this port is in, the current action, the next assessment — transcribed from CDFW, marked Not verified until I check the sources." |
| 7 | 2:25 | What changed | "Since my last visit — only the differences." |
| 8 | 2:40 | Save offline | "No signal past the jetty, so the whole page saves to the phone." |
| 9 | 2:55 | Set an alert | "Alerts on status changes and new test results. Web push works now; SMS in carrier review." |
| 10 | 3:15 | Ask Neritic (demo mode) | Ask "Is the Monterey crab season open?" "Every answer cites the record and its age. Ask 'is it safe to eat' and it refuses and points to CDPH." |
| 11 | 3:45 | Explore / Research | "And one tap away, the map and the research — which is where I'd like your critique." |

Readiness labels, exact phrasing: **"Pilot demo"** (not "launched"); **"Not verified"** (notices not human-reviewed); **"SMS in carrier review"** (not live). If Ask Neritic is in canned demo mode, say **"demo mode"**.

## Likely questions, honest answers
- *How accurate is the 8-day outlook?* "Held-out RMSE 0.760 in ln chlorophyll vs 0.857 climatology, n = 185,364, 2019–2021. It's chlorophyll only, and doesn't cover the bays." (#74)
- *Does it forecast domoic acid?* "No. Toxin-risk probability is NOAA's C-HARM, days 0–3 only."
- *Who uses it?* "Nobody in production yet. Ten interviews are the next step."
- *Your PINN?* "It matches, does not beat, the ConvLSTM. My own audit suggests a simple baseline matches both and the physics term may be inert — that needs independent reproduction."
- *Why not green?* "Toxin is invisible and notices lag; a green light is the one claim that could hurt someone."
- *What if the data is stale?* "Every card shows its age; the timeline falls back when C-HARM is stale (it was about 6 days stale recently)." (plan §1.6)
- *Liability?* "Terms say not an official source; it links to CDPH, CDFW, NWS, USCG."

## No-wifi fallback
Integrator to capture phone-width PNGs (light + dark) of: (1) Today top with status strip; (2) toxin tests card; (3) sea & weather card; (4) bloom timeline; (5) whale rules card; (6) what changed; (7) save-offline confirmation; (8) alert setup; (9) an Ask Neritic cited answer and the "safe to eat" refusal; (10) Explore map; (11) Research page. Keep them in a local folder and a PDF; the app's offline field pack is the second fallback.
