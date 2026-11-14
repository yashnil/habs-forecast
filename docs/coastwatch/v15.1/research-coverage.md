# Research coverage audit (v15.1, Agent C)

Date: 2026-11-13. Data read: the staging channel (`coastwatch-data-staging/v1`, manifest generated
2026-11-09T09:57:26Z, pipeline `33411500b447`, `observations-d1fa96877f.json`) and the test fixture
(`coastwatch-web/tests/fixture-data/v1`, generated 2026-10-08). Pipeline code read only (`pipeline/coastwatch_pipeline/sources/observations.py`,
`port_intel.py`, `biotoxin.py`). Nothing was published, and no pipeline code was changed.

Why we did this: at Crescent City (port-201) Research opened Trinidad Pier, 77 km away, with pDA last measured on
Apr 17, 2026. That is correct, but the page looked sparse. This audit asks whether the sparseness comes from the
source or from Neritic.

## 1. What Research reads

| Dataset | What it is | Where it is published | Used by Research |
|---|---|---|---|
| CalHABMAP shore stations (`observations-*.json`) | Water samples from 17 piers on SCCOOS ERDDAP: pDA, dDA, tDA, Pseudo-nitzschia seriata and delicatissima groups, extracted chlorophyll, water temperature | production and staging | yes: the record, the stations, the coverage |
| C-HARM history per station (`station.charm.history`) | Modelled probabilities (median of cells within 15 km) | production and staging, inside observations | yes: "Show the model", labelled as modelled |
| C-HARM per port (`port-intel-*.json`) | The current run only. History is kept for 30 days (`port_intel.HISTORY_DAYS`), for chlorophyll and not for C-HARM at the port | production and staging | no (Today and My Coast use it) |
| CDPH crab and seafood domoic acid (`biotoxin/index.json`) | Agency tissue tests (viscera or meat), measured in ppm | **staging only** (production returns 404) | no (the pilot's Today reads it) |

## 2. Station coverage (staging, 2026-11-09)

Here "n" counts samples that carry a value: quantified results plus reported zeros. Samples rejected by QC, and
visits that did not measure the variable, are not counted. Cadence is the median gap between those samples over
the last 365 days. "–" means the station has never published the variable.

| Station | Nearest port (km) | Visits (first to last) | pDA | tDA | PN seriata / delicatissima | Chlorophyll | Temperature |
|---|---|---|---|---|---|---|---|
| Trinidad Pier | Trinidad 0.6 | 2017-08-31 to 2026-04-17 | 108, last 2026-04-17, every 19 d | 106, last 2026-04-17 | – | – | – |
| Humboldt Bay | Eureka 3.6 | 2017-09-05 to 2026-04-07 | 88, last 2026-04-07, every 45 d | 87 | – | – | – |
| Humboldt Bay, South Bay | Eureka 9.7 | 2017-08-16 to 2026-03-30 | 104, last 2026-03-30, every 17 d | 103 | – | – | – |
| Bodega Marine Lab | Bodega Bay 2.7 | 2020-01-20 to 2026-04-13 | – | – | 324, last 2026-04-13, weekly | – | 309 |
| Bodega Marine Lab buoy | Bodega Bay 3.8 | 2020-02-20 to 2026-01-28 | – | – | 53, every 30 d | – | 1 (2025-07-25) |
| Tomales Bay mouth / mid-channel / inner | Bodega Bay 13 to 29 | 2020 to 2026-03-03 | – | – | 57 to 65 each, every 32 d | – | – |
| Santa Cruz Wharf | Santa Cruz 2.2 | 2014-01-08 to 2026-10-28 | 625, weekly | – | seriata 572, weekly; **delicatissima –** | 633 | 634 |
| Monterey Wharf | Monterey 0.6 | 2014-01-08 to 2026-11-04 | 165, **last 2022-08-31** | – | 473 / 475, last 2025-08-27 | 420, last 2024-09-11 | 527, weekly |
| Morro Bay, front and back bay | Morro Bay 1 and 4 | 2023-01-03 to 2026-10-13 | – | – | about 195, weekly | about 195 | 196 |
| Cal Poly Pier | Avila 1.4 | 2014-01-05 to 2026-11-01 | 653, weekly | – | 657, weekly | 661 | 624 |
| Stearns Wharf | Santa Barbara 0.4 | 2014-01-05 to 2026-10-26 | 636, weekly | – | 637, weekly | 638 | 627 |
| Santa Monica Pier | San Pedro 36 | 2014-01-06 to 2026-11-03 | 658, weekly | – | 663, weekly | 566 | 659 |
| Newport Beach Pier | Newport Beach 1 | 2014-01-06 to 2026-08-18 | 604, last 2026-08-10 | – | 595, last 2026-06-22 | 603 | 603 |
| Scripps Pier | San Diego 20 | 2014-01-06 to 2026-10-05 | 657, last 2026-08-10 | – | 637, last 2026-09-14 | 664 | 666 |

No station has ever published dDA. All 17 stations requested returned data (`status: updated`), so no
"source unavailable" state occurs today.

The C-HARM history at each station has 142 model days, from 2026-05-07 to 2026-11-02. The pipeline keeps 180
calendar days (`CHARM_HISTORY_DAYS`), and days without a model run are absent.

## 3. Ports: the nearest station for each variable (staging)

| Port | pDA | Pseudo-nitzschia | Chlorophyll | Temperature |
|---|---|---|---|---|
| Crescent City | Trinidad Pier 77 km S (2026-04-17) | Bodega Marine Lab 394 km | Santa Cruz Wharf 565 km | Bodega Marine Lab 394 km |
| Shelter Cove | Humboldt Bay, South Bay 79 km | Bodega Marine Lab buoy 208 km | Santa Cruz Wharf 385 km | Bodega buoy 208 km (one 2025 sample) |
| Fort Bragg | Humboldt Bay, South Bay 149 km | Bodega buoy 139 km | Santa Cruz Wharf 316 km | Bodega buoy 139 km |
| Point Arena | Humboldt Bay, South Bay 207 km | Bodega buoy 85 km | Santa Cruz Wharf 262 km | Bodega buoy 85 km |
| Bodega Bay | Santa Cruz Wharf 178 km | Bodega Marine Lab 3 km | Santa Cruz Wharf 178 km | Bodega Marine Lab 3 km |
| San Francisco | Santa Cruz Wharf 101 km | Inner Tomales Bay 52 km | Santa Cruz Wharf 101 km | Tomales mid-channel 62 km (2021) |
| Princeton / Half Moon Bay | Santa Cruz Wharf 73 km | Santa Cruz Wharf 73 km | Santa Cruz Wharf 73 km | Santa Cruz Wharf 73 km |
| Moss Landing | Monterey Wharf 24 km (2022) and Santa Cruz Wharf 25 km (current) | Monterey Wharf 24 km (2025); Santa Cruz seriata 25 km (current) | Monterey Wharf 24 km (2024); Santa Cruz 25 km (current) | Monterey Wharf 24 km (current) |
| Ventura, Port Hueneme | Stearns Wharf 39 and 53 km | same | same | same |
| San Pedro, Dana Point, Oceanside | Santa Monica 36 km, Newport 27 km, Scripps 38 km | same | same | same |
| Other ports | a station within 25 km | | | |

Four ports have no station of any kind within 60 km: Crescent City, Shelter Cove, Fort Bragg and Point Arena.
North of Bodega Bay, no station counts Pseudo-nitzschia or measures chlorophyll. Between Humboldt Bay and Santa Cruz
(about 470 km of coast), no station measures pDA.

## 4. Each gap: source or pipeline?

| Gap | Kind | Evidence | Fix |
|---|---|---|---|
| No station near Crescent City, Shelter Cove, Fort Bragg or Point Arena | **Source gap (no sampling)** | The pipeline reads all 17 CalHABMAP datasets on SCCOOS ERDDAP (`STATIONS`, checked 2026-10-08). None lies between Trinidad (41.05°N) and Bodega (38.32°N), or north of Trinidad. | None in Neritic. Section 5 lists other sources that could be ingested. |
| North-coast piers measure pDA and tDA only; Bodega and Tomales measure cells only | **Source gap (variable not collected)** | The request asks every station for all seven columns (`COLUMNS`), and the source returns NaN for the columns those labs do not report. | None. |
| Humboldt Bay every 45 days; Trinidad and Humboldt South Bay every 17 to 19 days; Tomales and the Bodega buoy every 30 to 32 days | **Source gap (sparse cadence)** | The whole record is requested (`orderBy("time")`, no thinning). | None. The page now shows this cadence. |
| North-coast records end in March or April 2026 | **Source gap (seasonal or lagging sampling)** | Earlier years also pause in winter and spring and resume later. The pipeline ran on 2026-11-09, so nothing newer existed at the source then. | None. The page says "old but valid". |
| Monterey Wharf pDA ends 2022-08-31, chlorophyll 2024-09-11, cells 2025-08-27 (temperature continues weekly) | **Source gap (variable no longer collected)** | Checked at SCCOOS ERDDAP on 2026-11-12 (see the note in `availability.ts`). | None. |
| Santa Cruz Wharf: no delicatissima group | **Source gap (not collected)** | The column is NaN at the source. | None. |
| Records start in 2014 on staging, but at the source they reach back to 2005 or 2008 at 6 stations (Monterey from 2005-06-03; Santa Cruz, Cal Poly, Stearns, Santa Monica, Newport and Scripps from 2008) | **Pipeline limit (window cut-off): already fixed in code, not yet published** | Staging `window_start` is "2014-01-01". Commit `00a59922b` (2026-11-12, on `feat/neritic-v15`) sets `WINDOW_START = "2005-01-01"`. The fixture already holds the 2005 copy (Monterey has 664 visits from 2005; on staging it has 537 from 2014). | **Needs a staging run** of the existing code. The data contract does not change. **Lead / owner decision.** |
| C-HARM history at a station covers 180 days only | **Pipeline limit (window cut-off)** | `CHARM_HISTORY_DAYS = 180`. C-HARM v3.1 nowcasts on ERDDAP go further back (not checked here). | Optional: raise it (for example to 730) in a separate pipeline PR. That makes observations-*.json larger (about 3 × 4 bytes per day per station, before JSON overhead) and needs a staging publish. **Flag for lead.** Not implemented. |
| No C-HARM history at a port without a station (for example Crescent City). Only the current run is in port-intel | **Pipeline limit (ingestion missing)** | `port_intel` keeps a 30-day history for chlorophyll and none for C-HARM at the port. | Optional: add `charm.history` at each port in port-intel. That is a new field, and **the deployed web and the public demo reject unknown manifest and record fields** (manifest schema compatibility rule), so it needs a schema bump and app support first. Not implemented. |
| CDPH crab tests at George Reef and Klamath River (near Crescent City) inform no place | **Pipeline limit (deliberately conservative mapping)** + **staging only** | `biotoxin.SITES` gives George Reef and Klamath River `[]` places, and `biotoxin/index.json` returns 404 on production. | Only if the owner wants it: map George Reef and Klamath River to port-201, and publish the biotoxin sidecar to production. These are agency **tissue** tests, not seawater, so they would need their own row ("seafood tested by CDPH"), not a strip on the water record. Not implemented. |
| dDA in the schema, but no station publishes it | **Source gap (variable not collected)** | Every station returns NaN. | None. Research no longer lists dDA among the variables. |

None of these gaps is caused by Neritic dropping, thinning or interpolating values. Every value with a sample is
kept. A reported 0 is kept as "reported 0", and a sample rejected by QC is kept as "rejected".

## 5. Sources not ingested (candidates, not checked live here)

These would be new pipelines, each needing its own audit, licence check and staging publish. They are not part
of v15.1.

- **CDPH Marine Biotoxin Monitoring (shellfish: mussels).** Weekly PSP and domoic acid in shellfish tissue,
  including Del Norte and Humboldt counties. These are tissue tests, a different matrix from seawater.
- **CDPH Phytoplankton Monitoring Program (volunteers).** Relative abundance of Pseudo-nitzschia along the north
  coast. The results are qualitative, not cells/L, so they could never be drawn on the cells/L axis.
- **NOAA Phytoplankton Monitoring Network.** Volunteer sites. Coverage of the north coast was not checked.
- **IOOS / CeNCOOS moored or glider HAB sensors**, where any exist north of Bodega. Not checked.

## 6. What changed in the app (v15.1)

These changes are in `coastwatch-web/src/nx/research/coverage.ts`, `Nearby.tsx` and `Research.tsx`:

- **Near <place>.** For the place in the link, or else the reader's port, the page lists every station within
  60 km, or the three nearest if fewer are that close. Each station shows its distance and direction, and one row
  per variable it measures. Each row draws every sample with a value as a tick on one shared time axis and gives
  the latest date with its state. A dotted mark lists "Not collected here: …".
- **States in words and marks:**
  - Regular samples (●): a value within 45 days, weekly or every two weeks.
  - Sparse cadence (●, "every 30 days"): a value within 45 days, but further apart than 21 days, or fewer than 12
    in the year.
  - Old but valid (◐): a valid record of its day that says nothing about now.
  - Not collected (dotted): the station has never published the variable.
  - Source unavailable: the station's last request failed.
  - Outside this window: the record has samples, but none in this range.
  - Modelled, not sampled (dashed): C-HARM.
- **Variables no listed station measures** each get one sentence, for example "The nearest station measuring
  Pseudo-nitzschia (seriata) is Bodega Marine Lab, 394 km south; nothing nearer samples it." The sentence links
  to that record and never presents it as this water's record.
- **A place no station is near opens on this view**, not on a distant pier's chart.
- **One click** on a latest date opens that station and variable at its newest sample. A row opens the station's
  record, and the whole record is one range button away. The place stays in the address (`place=`), so the
  geographic context survives a change of station or variable. The page says how far the open station is from the
  place, and in which direction.
- **"Statewide ocean map"** links to `/explore?place=<portKey>` (Agent A's Explore).
- The C-HARM history stays behind "Show the model", marked as modelled, with its radius and its own scale.
