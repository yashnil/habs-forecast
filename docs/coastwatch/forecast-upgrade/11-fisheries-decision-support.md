# 11. Practical fisheries implications

## 11.1 What each source can and cannot support

| Source | Can support | Cannot support |
|---|---|---|
| CDPH/CDFW notices | What is officially restricted, where and for which species: the only legal status | Anything about areas *without* a notice. Absence is not "open" or "safe". |
| C-HARM (0–3 days) | Where conditions resemble past toxic-bloom conditions, as a probability for water | Seafood toxicity, closures, catch, or anything beyond 3 days |
| Satellite chlorophyll (300 m–4 km) | Where algal biomass is high and how it is distributed nearshore | Species, toxin, or whether a bloom is *Pseudo-nitzschia* |
| Pier measurements (CalHABMAP) | Measured pDA, cDA and *Pseudo-nitzschia* at a pier on a date | Conditions at other places or later dates |
| Currents and drift | Where surface water may move in 1–3 days | Bloom growth or toxin; where fish or shellfish are |
| Historical landings (FOSS, statewide) | Which species groups have mattered economically, and how much was landed in past years | Predicted catch, predicted losses, or port-level effects (port data unavailable) |

## 11.2 Decision-support outputs that are honest

1. **"What applies to my fishery today"**
   - **Inputs:** species × official notices, joined from the registry already built in M2.
   - **Output:** a per-species summary ("Dungeness crab: 1 CDFW notice applies statewide · Not verified"), linking the agency text.
   - **Status:** ships without new science.
2. **"Watch list near a port"**
   - **Inputs:** a C-HARM probability at or above a display threshold in nearby water, measured pDA above detection at the nearest pier in the last 14 days, and an official notice nearby.
   - **Output:** counts and dates only, stated as "conditions to watch", never as risk to seafood.
   - **Status:** ships after P1 and P2.
3. **"How current is what I'm seeing"**
   - **Inputs:** the age of the newest clear satellite view, the newest pier sample and the C-HARM issue date for a port.
   - **Output:** a freshness card per port.
   - **Status:** directly useful to a skipper deciding whether to check with CDPH.
4. **"Seasonal exposure context"**
   - **Inputs:** historical landings by species group (M3) plus the timing of past toxin closures from official records.
   - **Output:** "In 2015–16, the commercial Dungeness season was delayed by domoic acid (CDFW)", with dollar history.
   - **Status:** history, labelled as such.
5. **"Where water may move"** (experimental, after the §6 gate)
   - **Output:** a 72-hour surface-drift spread from a chosen harbour mouth or from an observed nearshore patch.
   - **Use:** useful for understanding transport.
   - **Label:** "not where a bloom will be".

## 11.3 Outputs to refuse

These can't be supported by evidence:
- Predicted catches, or "good fishing" areas.
- Maps of legally open areas. Agencies don't publish them as data, and absence of a notice is not openness.
- Expected economic damages or losses for an upcoming season.
- Port-level dollar exposure until CDFW permits port-area data (draft request in `docs/coastwatch/drafts/`).
- Any "safe to harvest" or green "all clear" state.
