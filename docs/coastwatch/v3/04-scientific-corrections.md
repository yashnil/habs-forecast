# v3 · scientific corrections (2026-10-18)

The scientific review of this PR, posted on #66 from the v4 acceptance pass, listed seven items to fix
before merge. Each one was checked against its original source, fixed in the logic or wording rather
than by loosening a test, held by a regression test, and recorded here.

## Sources checked

| Source | What it settles | How it was checked |
|---|---|---|
| Pipeline, `pipeline/coastwatch_pipeline/sources/observations.py` | A reported 0 is "at or below the laboratory's limit, not that toxin was absent" (pDA; no limit is published). For cells, "cells may still be present below the counting limit". `WINDOW_START = "2014-01-01"` | read in this branch |
| SCCOOS ERDDAP `HABs-<station>` | First upstream samples: Monterey Wharf 2005-06-03, Scripps Pier 2008-06-30, Cal Poly Pier 2008-08-15, Santa Cruz Wharf 2011-10-05, Trinidad Pier 2017-08-31 | `tabledap/HABs-<key>.csv?time&orderBy("time")`, 2026-10-18 |
| CeNCOOS ERDDAP `info/<dataset>/index.csv` | `creator_institution` of each shore dataset; `publisher_name` CeNCOOS; licence "reference … CeNCOOS and NOAA" | read 2026-10-18 |
| CDPH "Domoic Acid" FAQ, CDPH Marine Biotoxin Monitoring Program, CDFW Ocean Fishing Health Advisories | Every species quote is verbatim on the live page | fetched 2026-10-18 (HTTP 200); quotes matched after whitespace normalisation |
| CDPH SN26-018 (Sept 14, 2026) | "The annual mussel quarantine … will continue through at least October 31." | fetched 2026-10-18 (HTTP 200) |

## Resolution

| # | Finding | Fix | Regression tests |
|---|---|---|---|
| 1.1 | Reported zeros were ranked as "lower" than a positive value, so a rank could be an artefact of censoring (Trinidad Pier 0.0057 ng/mL: 23 of the 24 "lower" samples were zeros) | `history.ts`: zeros are a censored category. A positive value is ranked only against the season's quantified samples, and the zeros are counted and said beside the rank. With too few quantified samples (< 12 from 3 years) the counts are said and no percentage is given. "The highest" needs at least one earlier quantified sample. The record node's units, support count, quality flags, lineage and caveats say the same | `history.test.ts`: Trinidad exact counts and sentence (no %); Santa Cruz 16 quantified / 126 zeros; counts form; the synthetic ranks. `evidence-v3.spec.ts`: page wording |
| 1.2 | "Since the record began in 2014" and "its record starts in 2014" described Neritic's window, not the station's record | "The highest in Neritic's copy of its record, which starts in YYYY". The too-few sentence says the same. The lineage reads "Neritic's copy of <station>'s CalHABMAP series … (Neritic keeps samples from 2014 on; the station's own series may begin earlier)". Doc 01 corrected | `history.test.ts`: the copy wording, and no "record began" / "its record starts" / "since the record" |
| 2.1 | The source was named "CeNCOOS, <station>", crediting the aggregator only | `shoreStations.ts`: `operator` (current name) and `operatorAsPublished` (the dataset's `creator_institution`). The source is "<station> (<operator>), via CeNCOOS ERDDAP"; the lineage adds the operator and "CeNCOOS is a regional association of NOAA's U.S. IOOS (credit: the operator, CeNCOOS and NOAA, per the dataset licence)". The data are never labelled NOAA's. The silent-station date is now in Pacific time, not UTC (a review note) | `shoreStations.test.ts`: all five operators as published, node source and lineage, the conditions label. `evidence-v3.spec.ts`: `/api/conditions` and `/api/evidence` |
| 3.1 | "CDPH quarantines sport-harvested mussels from May 1 to October 31 every year" would read as "over" after Oct 31, against CDPH's "through at least October 31" | The mussel species line drops the dated quarantine fact. The registry record (status, expected end, SN26-018 caveat, review state) speaks for the quarantine; the species line keeps CDPH's monitoring sentence | `speciesEvidence.test.ts`: no species fact carries an end date, and no mussel fact is about the quarantine. Checker rule (end date in an agency-page line), with a negative test. `evidence-v3.spec.ts`: the mussel page has no "October 31" |
| 3.2 | The species line was Neritic's synthesis styled as an official notice, with no word that it is an unreviewed copy | A new claim kind `agency_page` with its own hollow mark; the line is never styled as a notice. Its headline "the agencies' pages name …" attributes the synthesis to the pages. A visible provenance line reads "Neritic's copy of the agencies' pages, read October 16, 2026; not reviewed by a person. The agency pages are the authority." | `speciesEvidence.test.ts`: every non-gap species line is `agency_page` with that provenance. Checker rule (missing provenance), with negative tests. `evidence-v3.spec.ts`: `data-kind`, the mark, the provenance on the page |
| 3.3 | "What CDPH says" showed Neritic's paraphrases as the agency's words, and labelled a CDFW sentence as CDPH's (mussels) | `Claim.more` (paraphrases) is replaced by `agencyWords`: every fact is the agency's own quote, with its own agency, page label and link. The disclosure reads "What the agencies' pages say" | `speciesEvidence.test.ts`: each quote is verbatim on its page under its own agency; mussels quote CDFW under CDFW. `evidence-v3.spec.ts`: both crab quotes with their CDPH links; the mussel quote under CDFW |
| 3.4 | "Neritic's shore readings and NOAA's model are about domoic acid in the water" presented cell counts and chlorophyll as toxin | "Neritic's shore samples (Pseudo-nitzschia cells and particulate domoic acid in seawater) and NOAA's model describe the water, not the <animal>." | `speciesEvidence.test.ts`: no species line says readings are about toxin. Checker rule (readings, cells or chlorophyll "about" toxin), with negative tests on the old sentence and a chlorophyll variant |

The consistency sweep (every fixture harbour × activity × species × model, satellite and conditions
state, at three dates) still finds zero violations with the three new rules in place.

## Effect on what the fixture pages say

| Station / variable | Before | After |
|---|---|---|
| Santa Cruz Wharf pn, 4,900 cells/L | higher than 98% of the Aug–Oct samples | higher than 81% of the 16 quantified Aug–Oct samples …; another 126 had no cells counted (not necessarily none present) and are not ranked |
| Santa Cruz Wharf pDA, 0.202 ng/mL | higher than 94% | higher than 84% of the 57 quantified …; another 98 were reported 0 (not quantified, not absent) and are not ranked |
| Trinidad Pier pDA, 0.0057 ng/mL | higher than 86% | of the 28 Mar–May samples there in 2019–2024, 23 were reported 0 (not quantified, not absent), and 1 of the 5 quantified was lower |
| Newport Beach pn, 15,593 | higher than 64% | higher than 47% of the 96 quantified …; another 46 had no cells counted … |
| Scripps Pier pn, 2,872 | higher than 29% | higher than 26% of the 140 quantified …; another 6 … |
| Bodega Marine Lab buoy pn, 418 | higher than all 13 | of the 13 Dec–Feb samples …, 4 had no cells counted …, and all 9 quantified were lower |

## Not changed here (review notes, not fix-before-merge)

- `flag_high` values stay in the rank and in "last as high or higher". Excluding them, or saying "including
  values flagged for review", is a separate decision.
- Year spans still read "in 2014–2025" though a season can miss years; the node's lineage gives the
  number of years.
- Same-day duplicate visits still count as separate samples, and samples after 17:00 PDT carry the next
  UTC date.
- The record guard compares dates, not values (`r.date === e.date`).
- "QARTOD: passed" is the CeNCOOS aggregate flag. Which tests ran is not read, and fluorometer
  biofouling is not caught by automated tests.
- The species facts are static with `READ = "2026-10-16"`. No watcher re-reads the three pages; the
  verbatim test compares against saved excerpts, and the live check of 2026-10-18 was manual.
- The toxin list in a species headline includes statewide notices naming the species.
- Agency statement and species-fact nodes keep the class `measurement` (an existing v2 convention).
  The evidence list therefore draws them with the measured mark; the page line itself now has the
  agency-page mark.
- v4 (#68) is not changed. Its rehearsal branch's integration commit renders `Claim.more`, which no
  longer exists. That commit must render `agencyWords` and `provenance` when #68 is rebased onto main
  after this PR merges.
