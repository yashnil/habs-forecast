# Neritic v5: fishermen user-experience audit

The audit was written on 2026-10-20 by Agent D, on branch `wip/v5-D`. It walked the private preview `coastwatch-preview-ard1mcft8` and a local build at 390 px and 1440 px. Screenshots are in `scratchpad/v5-evidence/D/{before,after}/`.

The owner's question was: *"Where and what to catch, while maximizing economic output."* For a California fisherman, that comes down to five questions asked every morning:

- **What can I catch?**
- **Is it open here?**
- **Is the sea fishable?**
- **What is it worth?**
- **Where should I look?**

## 1. How California fishermen actually decide

The research was done on 2026-10-20. Sources are CDFW, FGC, PFMC, CDPH and NWS pages, plus CDFW *By The Numbers* 2023 and 2024.

### The order they decide in

1. **Legal status.** This covers season, zone, depth and quota.
   - The rules differ by CDFW groundfish management area: Northern, Mendocino, San Francisco, Central and Southern.
   - Commercial Dungeness crab has statutory dates in FGC §8276: Districts 6–9 run Dec 1 – Jul 15, and the rest of the state runs Nov 15 – Jun 30. In practice two things delay it:
     - the whale-entanglement RAMP: in 2025–26 commercial crab opened Jan 5 or Jan 15 by zone, under 15–40% gear reductions;
     - domoic acid testing: Humboldt opened late January.
   - Recreational Dungeness crab is "scheduled to reopen on November 7, 2026" (CDFW regional pages, read 2026-10-20).
   - Salmon is set each year by PFMC. Commercial salmon reopened in 2026 for the first time since 2022, and CDFW says "the 2026 commercial ocean salmon fishery has concluded." Sport salmon is open only from Point Reyes to Pigeon Point, through Oct 31.
   - Rockfish 2026 is open at all depths from Apr 1 to Dec 31 north of Point Conception. The Southern area has its own depth table.
   - Fishermen learn about in-season changes from the CDFW in-season page and from hotlines: salmon (707) 576-3429, and the domoic-acid closure line.
2. **Weather window.** This covers wind, swell, wave period and Small Craft Advisories.
   - Sources are NWS coastal waters zones (e.g. PZZ545) and NDBC buoys, often checked through Buoyweather or Windy.
   - Every skipper has personal limits for wind and seas.
3. **Ocean features.** Fishermen look at:
   - SST breaks: albacore anglers cite 58–64 °F water and 1 °F breaks, which is folk knowledge rather than science;
   - chlorophyll edges, fishing the clean-blue side;
   - upwelling, canyons and banks.
   - They read these from Hilton's, Terrafin, FishTrack and SatFish charts.
4. **Tide and current.** These matter most to shellfish harvesters, divers and halibut drifters.
5. **Price and market.** Ex-vessel value varies widely from year to year.
   - In 2024, market squid ($67.9M) passed Dungeness crab ($49.7M). In 2023 crab led ($54.1M), ahead of squid ($30.4M). Source: CDFW By The Numbers.
   - Port areas swing just as much: Eureka was $44.0M in 2023 and $20.0M in 2024.

### Tools they use today

| Need | Tools |
|---|---|
| Regulations and seasons | CDFW Ocean Sport Fishing map pages, CDFW groundfish summary, CDFW in-season page, PFMC |
| Weather | NWS marine forecasts, NDBC, Buoyweather, Windy |
| Where to fish | Fishbrain, FishTrack, Hilton's, Terrafin, SatFish |
| Landings and value | CDFW Marine Fisheries Data Explorer, PacFIN |

None of these tools combines the agency rule, the weather and the value for one harbour in one view. That gap is Neritic's opening, provided it quotes and never decides.

## 2. Walking the app as each persona (before)

The walk covered `/coast`, `/coast?place=port-452&for=commercial&species=dungeness` (Half Moon Bay crabber), `/coast?place=port-550` (Monterey), `/fisheries` and `/`.

| Persona | What they needed in 10 seconds | What they got |
|---|---|---|
| Commercial crabber, Half Moon Bay | Opener date; RAMP zone; seas tomorrow; price | A quote that the season is "closed", with no reopen date. Five HAB and toxin statements. Wind and seas as the 6th line. No value. |
| Charter / CPFV skipper, Monterey | Rockfish depth and season; salmon status; Small Craft Advisory | No finfish season information at all. Rockfish was in the species list, but the page showed only HAB evidence and a link. |
| Albacore / offshore | Water temperature, temperature breaks, chlorophyll edges, seas | Satellite chlorophyll (with "not toxin" copy) only. No sea temperature. Buoy water temperature was hidden inside the conditions claim. |
| Sport halibut / lobster (Southern California) | Season, bag limit, open/closed | Not answered. Lobster showed only a toxin statement. |
| Urchin diver | Swell and period; days open | Urchin was not in the species list. |
| Shellfish harvester | Quarantine; tides | Well served by notices and tides. This is the app's historic strength. |

## 3. Prioritised findings

### P0: the core question is unanswered

1. **No "is it open here" for most species.** Only Dungeness crab (a statewide quote) and toxin closures were covered. Rockfish, salmon, halibut, tunas, lobster, white seabass and rock crab had no season information.
   → Built: **Seasons in these waters**, from CDFW's regional page.
2. **Sea conditions were buried.** They appeared as one line, the 6th "matter". There was no tonight/tomorrow forecast, no prominent Small Craft Advisory, no measured water temperature on top, and no way to apply your own limits.
   → Built: **Sea conditions**, plus **your own limits**.
3. **The 10-second answer was the wrong one.** For a finfish trip, four of the six lead statements were about toxin in shellfish or crab.
   → Built: an **At a glance** block before "What matters now", with four answers: Season, Sea, Notices and Worth.

### P1: economic context and entry

4. **No value on My Coast.** The Fisheries page covered only six toxin-affected groups, statewide. Squid, urchin, halibut, sablefish and spot prawn were missing, and there were no port-area figures.
   → Built: **What the catch was worth**, from CDFW By The Numbers 2023 and 2024. It shows the port area and the species, with an average $/lb that is labelled as Neritic's arithmetic.
5. **Species list gaps.** Albacore/tunas and red sea urchin were missing.
   → Added both.
6. **Place-first only.** Fishermen think species-first ("I fish crab out of HMB").
   → Built: a **"What do you fish?"** chip row on the My Coast landing, carried into every harbour link.
7. **The Fisheries page species panel had no seasons or value.**
   → Built: CDFW lists by region for the chosen species, plus its dock value.

### P2: follow-ups, not built here

8. **There is no SST layer**, and no chlorophyll-front or gradient layer. This is the biggest "where to look" gap. **Request to Agent B** below.
9. **Commercial in-season actions** (RAMP zone status, depth limits, gear reduction) exist only as CDFW memos and press releases. Neritic links them but cannot parse them reliably. A future pipeline adapter could watch the in-season page. That belongs to Agent C and needs a human review flow.
10. **Jargon.** "C-HARM", "particulate domoic acid" and "Pseudo-nitzschia" still lead some statements. A plain-words pass on `lib/harborEvidence` claim titles is needed. That file is not owned by D.
11. **Registry staleness (for the lead and a human reviewer).** The CDFW in-season page lists "Monterey Bay Anchovy and Sardine Restrictions Lifted as Health Concern Alleviated". The registry still shows `cdfw-2026-anchovy-take-restriction-monterey-bay` as active. I did not change it: only `cwp review-official` by a person may.
12. **Sunrise/sunset and moon phase** for night squid boats and grunion. These are cheap to add later.
13. **Daily check-in.** The home port plus SinceLastCheck already exist. A shareable "morning link" per home port with species is the next step.

### What we will not do

- No "best spot" and no catch prediction.
- No go/no-go verdict from Neritic.
- No price forecast, and no claim that a fishery is "open" beyond quoting which CDFW list it is on and when.
- No toxin inference from chlorophyll.

## 4. What was built (after)

- **At a glance** (`brief-glance`): four tiles, each an anchor link to its evidence.
  - **Season**: CDFW's list and marker for the reader's species. Where CDFW's section gives a reopen date, its sentence is quoted. For rockfish, the groundfish-table row for the trip day is shown.
  - **Sea**: the NWS period for the trip day, plus any NWS alert, plus measured water temperature.
  - **Notices**: the count, marked not verified.
  - **Worth**: CDFW's dock value for the species, or for the port area.
- **Sea conditions** (`brief-sea`):
  - NWS alerts first, then up to four NWS periods with the highest wind and seas stated in each, and the full NWS text one tap away;
  - measured wind, waves and water, with station, distance and time;
  - tides predicted for the trip day;
  - **your own limits** (wind kt, seas ft), kept on this device and compared only with the forecaster's stated numbers.
- **Seasons in these waters** (`brief-seasons`):
  - CDFW's region, its boundaries and its "updated on" date;
  - Neritic's copy date and an unverified label, with a staleness warning after 14 days;
  - the reader's species first, quoted, then the commonly fished entries, then all entries;
  - a note that commercial rules differ, with links to CDFW commercial rules and in-season changes;
  - salmon hotlines.
- **What the catch was worth** (fold `brief-worth`): CDFW By The Numbers 2024 and 2023.
  - For the reader's species: value, pounds, rank and average $/lb.
  - For the port area: value, pounds, rank and share of the state.
  - Sources, "data as of" dates and a "history, not price" caveat are shown.
- **Landing**: "What do you fish?" species chips.
- **Fisheries page**: the CDFW list for the chosen species in all five regions, plus its dock value.

### Data provenance

| File | What it holds | How it was made |
|---|---|---|
| `src/content/seasons-cdfw.json` | All five CDFW regional pages | Generated by `docs/coastwatch/v5/fishermen-seasons-extract.py`, read 2026-10-20, with CDFW's own updated dates (Oct 1, 2, 9, 13, 2026) |
| `src/content/seasons.ts` | CDFW groundfish table ("Updated June 23, 2026"); CDFW salmon hotlines | Transcribed by hand |
| `src/content/landings.ts` | CDFW By The Numbers 2024 (data as of 02/14/2025) and 2023 (as of 2/9/2024) | Transcribed by hand; unit tests check that port-area rows sum to CDFW's totals |

## 5. Requests to other agents

- **Agent B (map):** add a sea-surface-temperature layer (observed; e.g. NOAA geo-polar blended or MUR via CoastWatch ERDDAP, °F legend) and, optionally, an SST/chlorophyll gradient ("fronts") overlay.
  - Label both as observation context, never a catch forecast.
  - My Coast would then link "Where to look" to `/?layer=sst&sel=harbor:<code>`.
- **Agent C (pipeline):** a watched source for the CDFW in-season page ("In-Season Ocean Fishing Regulation Changes") that produces *unreviewed* registry candidates for human review. Separately, the anchovy record above needs a person to review it.
