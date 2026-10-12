# Proposal: station observation CSV export

**Status:** proposal only; nothing has been implemented. It addresses the M8 outstanding concern "researchers can see every value but cannot download it" (doc 25 §8, journey 3).

## What a researcher needs

The values for one station, with dates, units and qualifiers, plus a citation and licence they can trust. In practice this means either the original data or an unchanged copy that says exactly where it came from.

## What already exists

- Every station in the published observations file (`observations-*.json`, about 950 kB, already loaded by Bloom Intelligence) carries:
  - `request_url`, the exact SCCOOS ERDDAP query the pipeline used. It is already a CSV request, for example `…/tabledap/HABs-SantaCruzWharf.csv?time,latitude,…,Temp&time>=2014-01-01…`.
  - `retrieved_at`, plus a dataset-level provenance block (citation, licence, institution).
- That URL returns the CSV directly: 200, 59 kB for Santa Cruz Wharf since 2014, in 0.2 s, tested 2026-10-12. ERDDAP sends a units row and a download filename.
- Licence (as recorded by the pipeline from the dataset): "may be used and redistributed for free but is not intended for legal use". CoastWatch cites CalHABMAP and SCCOOS wherever it shows the data, and an export would too.

## Options

| | Option | What the user gets | Cost | Risks |
|---|---|---|---|---|
| **1** | **"Download from source (CSV)" link** in the values table and provenance line, pointing to the station's `request_url` | The authoritative CalHABMAP data, unchanged, with ERDDAP's own units row | A link and a test; no new data, no schema change | Upstream may be briefly down (SCCOOS reloads). It shows raw upstream values (NaN, raw zeros), not CoastWatch's qualifiers. |
| **2** | **"Download what is shown (CSV)"**, built in the browser from the data already loaded | The values exactly as CoastWatch shows them, with qualifiers, plus a commented header: source, citation, licence, request URL, retrieved time, CoastWatch version, and "not for legal use" | About 1 small component and a pure function with unit tests; no new requests, no pipeline change | Must never drop a qualifier or turn null into 0; it must say it is a copy and link to the source. |
| 3 | The pipeline publishes per-station CSV files | Same as 2, but cacheable | New artifacts, data-branch publishing and manifest or schema questions | The deployed demo rejects unknown manifest fields; not worth it for 17 small series. |

## Recommendation

**Do 1 now, and 2 after the scientific review.** Option 1 is properly sourced by construction, needs no data change and closes the journey-3 gap. Option 2 adds value, because qualifiers like "reported 0 (not quantified)" matter. But its column definitions should reflect any wording changes from the review. Don't do 3.

### Option 2's file, as proposed

```
# CoastWatch copy of CalHABMAP shore-station data — Santa Cruz Wharf (HABs-SantaCruzWharf)
# Source: California Harmful Algal Bloom Monitoring and Alert Program (CalHABMAP), via SCCOOS ERDDAP
# Original data: <request_url>
# Retrieved: <retrieved_at>   CoastWatch pipeline: <pipeline_version>
# Licence: <provenance.license>
# Values are unchanged from the source. Empty = not measured in this sample.
# qualifier "reported_zero": reported 0, not quantified (no detection limit published); not "absent".
# This describes the water sampled at this pier on these dates, not nearby waters or seafood.
time_utc,depth_m,variable,value,units,qualifier
2026-09-30T15:30:00Z,,pDA,0.12,ng/mL,
…
```

The long format (one row per value) keeps fractions and size groups separate and cannot imply that they add up. A wide format is easy to add if reviewers prefer it.

### Tests (when implemented)
- The file's values equal the published series exactly (null stays empty, 0 stays 0 with its qualifier, flagged values keep their flag).
- The header contains the source URL, the citation and the licence.
- A page-level safety test: no "safe", "open" or forecast wording in the file.

### Decision needed
Approve option 1 as a small follow-up PR, and/or option 2 after the review.
