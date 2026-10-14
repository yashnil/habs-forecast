# My Coast: the HAB briefing

My Coast (Coastal mode) used to be a four-chapter accordion. In practice it showed a map, the notices,
a years-old toxin value and "no forecast". It is now one page with nothing to open first:

1. **Your waters and species**: the kind of fishing, the species and an optional trip day. Choosing a
   species is encouraged but never assumed; without one, the page says it is a general overview.
2. **HAB conditions** (`components/coast/HabBriefing.tsx`, `lib/hab.ts`): a factual summary, then one
   card per kind of evidence. Each card states its value, date, age, distance and source.
3. **Official information and ocean conditions**: the official card and NOAA conditions, side by side.
4. **Before you go**: personalised steps. Each step says which waters and species it concerns, gives
   a direct source, and says how sure it is.

Everything comes from data CoastWatch already publishes. There are no new manifest or schema fields,
and the demo is untouched.

## What each piece of evidence is, and is not

| Card | Source | Rule | Never says |
|---|---|---|---|
| Pseudo-nitzschia cells, measured | CalHABMAP shore stations (SCCOOS ERDDAP), both size classes from one sample | Takes the nearest station within 60 km that sampled in the last 30 days. Otherwise takes the nearest within 60 km, with its age. Otherwise names the nearest station and says it is too far to describe these waters. When the evidence comes from farther away, the summary also gives the nearest station's own last sample and its age. | That cells mean toxin; that "none counted" means clean |
| Toxin in seawater, measured | CalHABMAP particulate domoic acid | Same rule. Zeros read "not quantified". | Anything about seafood |
| Chlorophyll from space, observed | OLCI 300 m (else VIIRS), latest clear pixel within about 5 km, with its date | A cloudy week reads "cloud, fog or no overpass, not low chlorophyll". | That chlorophyll confirms or rules out a harmful bloom |
| NOAA C-HARM, modelled | The waters' own 3 km cell when it has a value. Otherwise the real valued cells within 15 km (median, range, nearest distance). Otherwise the nearest cell within 80 km, named as describing that cell. | Toxin and bloom are separate probabilities. The day bars show the median as a bar and the range as a whisker. The port's nowcast history line breaks on missing runs. | That an offshore cell is the waters' forecast; that a probability is a measurement |

The answer line at the top follows the same rules. When CDFW publishes a season status for the
chosen species (for example Dungeness crab, "Season is closed"), that quoted status comes first, above
any "no notice in CoastWatch's list" line.

## The map follows the briefing

- **Overview:** your waters; the stations the briefing names, with threads (fainter beyond 25 km); every
  station within 80 km (filled if sampled in the last 30 days, a ring if older); the model cell used;
  the official limits.
- **A station card:** the camera travels to that station, which is drawn larger, with a single thread.
- **The model card's "Show the model's cells and its nearshore gaps":** the C-HARM grid at exact cell
  edges, the 15 km ring, and the no-value hatch, explained in the map key.
- **The satellite card:** the satellite chlorophyll field and a thread to the pixel used.
- **The conditions card's "Show the buoy and the tide station":** the NOAA buoy and tide station that the
  card reads, each labelled and threaded to your waters, in the conditions colour (context, not HAB
  evidence).
- **A station clicked on the map:** if the briefing names it, its card takes focus. Otherwise a small card
  shows its latest measurements, each with its own date, and links to its history on Bloom.

Only one data field is in view at a time. Nothing is interpolated between stations or cells.

## Species and activity drive the steps (`lib/nextSteps.ts`)

- **Disputed notices** (an agency release appears to lift a notice): a step with both the later release
  and the original notice. It appears only when the notice concerns the chosen species, and says no
  person has reviewed it.
- **CDFW statewide statements** for the species and activity are quoted, season status first.
- **A recent shore sample** (30 days or less, within 60 km) is a step that links to the station's
  history on Bloom.
- **Shellfish** (by activity or species): CDPH's advisory map and test results come right after the
  notices, and the tides lead the conditions.
- **Dungeness crab:** a link-only step for CDFW's whale and sea turtle (RAMP) actions, which CDFW
  publishes only as prose. CoastWatch infers nothing from it.
- **A species with a landings group:** a link to its Fisheries history, labelled "history only".
- **Trip day:** picks the model day, the NWS marine period for that day, and that day's tides.

## Other HAB feeds investigated (2026-10-13)

| Feed | What is available | Decision |
|---|---|---|
| CalOOS IFCB dashboard (`ifcb.caloos.org/api/list_datasets`) | 22 California datasets, including Santa Cruz Municipal Wharf, Stearns Wharf, Newport Beach Pier, Trinidad Pier, Del Mar mooring and the MBARI power buoy. The public API serves instrument bins and images; there is no validated, versioned Pseudo-nitzschia count series. | **Not integrated.** Classifier output is unvalidated and would read as a measurement. Revisit if CalOOS publishes validated class counts on ERDDAP. |
| CDPH shellfish biotoxin results | A PDF of the past 30 days, plus the advisory map; no machine-readable feed | **Linked, not ingested** (shellfish steps). Parsing a PDF into "results" would risk misstatement. |
| CDFW Whale Safe Fisheries (RAMP) | Prose page and Director's Declaration PDFs; the zone GIS has no status attribute | **Linked, not ingested** (Dungeness step). |
| CalHABMAP (SCCOOS ERDDAP) | Already ingested: pDA, both Pseudo-nitzschia classes, extracted chlorophyll, temperature | Used, now with 400-day timelines per station |
| NOAA C-HARM v3.1 | Already ingested | Used, with the 15 km neighbourhood rule from the port summaries |

## Known limitations

- Most California ports lie in C-HARM's nearshore gap. The briefing shows the offshore cells within
  15 km clearly separated, but there is no nearshore toxin model, and CoastWatch does not make one.
- CalHABMAP samples weekly at a handful of piers. The nearest recent sample can be 40 km or more away
  (Monterey → Santa Cruz Wharf), and the page says so.
- The satellite card often reports no clear view (coastal fog). That is stated as unknown, not low.
- Activity and species live in the link; they are not yet remembered between visits.
