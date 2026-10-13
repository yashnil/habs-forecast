# Notice review packet: CDPH SN26-021 lifts SN26-018 (Monterey County bivalves)

**Status: awaiting a person.** Nothing in this folder has been applied to the registry. CoastWatch has not marked any notice lifted or verified. Applying the change and recording a review are for a person, after reading the sources.

## What happened

| Date | Source | What it says |
|---|---|---|
| 2026-09-14 | [CDPH SN26-018](https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-018.aspx) | "CDPH is advising the public not to eat sport-harvested mussels, clams, scallops, or oysters, gathered from Monterey County…" |
| **2026-10-12** | [**CDPH SN26-021**](https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-021.aspx) | "CDPH has removed the warning that advised against eating recreationally harvested bivalve shellfish from the waters of Monterey County. The annual mussel quarantine for recreationally harvested mussels along the entire California coast remains in effect." |

The registry (`data/curated/official_notices.json`) still lists `cdph-2026-sn26-018-monterey-bivalves` as `active`. Production published it that way until this PR's change.

## What this PR changes (no record is edited)

- **Pipeline (`official.reported_lifts`):** an *unreviewed* CDPH release whose title says it lifts something is compared with each active CDPH record. A match needs all three of:
  - the same county, from the record's structured `counties` field;
  - the same kind of seafood;
  - a later release number than the record's own.
- **What a match does:** it is published as an "Agency source discrepancy" in the dataset's existing `conflicts` field. The published schema is unchanged, so the public research demo is unaffected; the older web shows its generic "records contradict each other" reason.
- **Live check, 2026-10-13 05:12Z (`cwp watch-official`):** exactly one discrepancy, SN26-021 against SN26-018.
- **Web:** a disputed record is never shown as in effect.
  - It is not counted in the masthead, the drawer heading or the place counts.
  - It appears in a separate "Reported lifted by the agency · awaiting review" group with a link to the release: in the drawer, on My Coast, in the map's port panel, in the Bloom strip and in the Fisheries notes.
- **When it clears:** the state disappears by itself once a person updates the record and reviews the release.

## Proposed registry change (for a person to approve)

[`proposed-record.json`](proposed-record.json) replaces the SN26-018 record:
- `status: "lifted"` and `lifted_date: "2026-10-12"`;
- SN26-021 added as a source;
- the lifting sentence quoted in `uncertainties`.

Checked against a copy of the registry: `validate_registry` reports no errors and no conflicts, and no discrepancy remains for SN26-021.

**To apply after checking:**
1. Read both releases (links above) and the CDPH advisory map.
2. Replace the record in `data/curated/official_notices.json` with `proposed-record.json`. Edit it if you read the sources differently.
3. Run `cwp review-official --reviewer "<your name>" --status pending_human_review --confirm`. This records SN26-020 and SN26-021 as reviewed and stores the current page fingerprints. Use `--status human_verified` only after checking *every* record.

## Evidence (fetched 2026-10-13 05:27Z; raw-page SHA-256 in each file)

| File | Source | Raw SHA-256 |
|---|---|---|
| [evidence-SN26-021.txt](evidence-SN26-021.txt) | CDPH SN26-021 | `8ff1ece969ac391de709946e708dce82a1fd7f3e702bc083056828ea534e4f2a` |
| [evidence-SN26-018.txt](evidence-SN26-018.txt) | CDPH SN26-018 | `354b095621a3f8d5921f0e6a3184003cf02f0e6a7d6d0d3543914b01e0e34b73` |
| [evidence-Shellfish-Advisories.txt](evidence-Shellfish-Advisories.txt) | CDPH release list | `e9bce66903159b05cac636eaf97cf10050d1f1da622151bf93f3b36f9dd2eb12` |
| [evidence-cdfw.txt](evidence-cdfw.txt) | CDFW Health Advisories page | `26e1768b9aa46dc50e6c971cc44616774d2cd48c079d0e4efd715bf8c334540f` |

## Other findings from the 2026-10-12/13 audit

| Item | Finding | Action |
|---|---|---|
| CDPH SN26-020 (Del Norte razor clams, 10-08/09) | In the registry, but its release id is not yet recorded as reviewed, so the watcher keeps listing it | Reviewer: recorded by the next `review-official` |
| CDFW Health Advisories page | Changed since the last review; its statements match the registry. The newest item is the Del Norte razor clam closure (10/9/2026). The page's news list still shows CDPH's September Monterey bivalve warning and does not mention the lift | No record change; review fingerprints |
| Mussel quarantine | "will continue through at least October 31" (SN26-021 repeats it) | A person decides after 2026-11-01 (an expected end date is never treated as a lift) |
| Rock crab commercial closure 40°N | Start date still not published | Unchanged |
| C-HARM | Latest run issued 2026-10-11; on 10-13 UTC the app labels it stale. NOAA skipped 09-17, 10-05 and 10-08 in the last month | Shown as stale (correct) |
| **C-HARM input satellite** | NASA reports Suomi-NPP data, including VIIRS, ends **2026-11-02** (notice of 2026-10-05). C-HARM v3.1 names S-NPP VIIRS ocean colour as its input | Plan for a possible C-HARM outage or version change; gaps must never read as "no risk" |
| CalHABMAP North Coast and North-Central pier series (Trinidad, Humboldt, Bodega, Tomales) | Latest published samples are Jan–Apr 2026 | Shown with their age; confirm whether sampling stopped |
