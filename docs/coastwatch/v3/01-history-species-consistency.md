# v3 · station record, species lines, consistency rules

## 1. A measurement against its own station's record (`src/lib/history.ts`)

Neritic's copy of the CalHABMAP series starts at the observations window, 2014-01-01 (the pipeline's
`WINDOW_START`). Several piers' own series begin earlier upstream (SCCOOS ERDDAP, read 2026-10-18:
Monterey Wharf 2005, Scripps Pier and Cal Poly Pier 2008, Santa Cruz Wharf 2011), so the long-running
piers' samples here start in 2014 because Neritic's copy does, not because the station's record does.
Trinidad and Humboldt start in 2017, and Bodega, Tomales and Morro Bay in 2020–23. The published
observations hold the series, so the comparison is computed server-side in `slimStations` and travels
as `StationLite.record`; no pipeline or manifest field.

Said, on the HAB line, only when the line's sample is the station's newest:

> Compared with earlier years at Santa Cruz Wharf: higher than 81% of the 16 quantified Aug–Oct
> samples there in 2014–2025; another 126 had no cells counted (not necessarily none present) and are
> not ranked. Last as high or higher: September 2 (15,900 cells/L). Previous sample September 23: 100 cells/L.

(Corrected 2026-10-18, see `04-scientific-corrections.md`. The first version said "higher than 98% of
the Aug–Oct samples": 126 of its 139 "lower" samples were zero counts.)

Rules: same season = the sample's month and its neighbours, earlier years only. **Reported zeros are
censored, not small numbers** (the pipeline: "0 means the value was at or below the laboratory's
limit, not that toxin was absent"; for cells, "cells may still be present below the counting limit"),
so a positive value is ranked only against the season's quantified samples, and the zeros are counted
and said beside the rank. A rank needs ≥ 12 quantified samples from ≥ 3 years. With enough samples but
too few quantified, the counts are said instead ("of the 28 Mar–May samples there in 2019–2024, 23
were reported 0 (not quantified, not absent), and 1 of the 5 quantified was lower"); with too few
samples, "too few samples … to compare by season". A zero sample is said as "no cells counted" / "not
quantified (reported 0)" with the share of the season's samples that were also 0. "The highest" is
said as "the highest in Neritic's copy of its record, which starts in YYYY", and only when an earlier
sample was quantified. QC-rejected values are dropped;
Pseudo-nitzschia is the seriata + delicatissima sum, as the briefing reports it; percentages never
round to 0 % or 100 % unless exact. Never "unusual", "risk", "safe": it is the station's own past,
not other waters, seafood or the future. The sentence is its own node (`record:…`, class
`record_statistic`) and its own field (`Claim.history`), not part of the claim's `text`, which the
HAB summary paragraph repeats.

## 2. The chosen species (`src/content/speciesEvidence.ts`)

For a chosen species, one line: which toxins the agencies' pages name for it, each agency sentence
quoted under its own agency and page, and what Neritic's readings are evidence of. The line is an
**agency page** line (`kind: "agency_page"`, a hollow official mark), not an official notice, and it
says so on the page: "Neritic's copy of the agencies' pages, read October 16, 2026; not reviewed by a
person. The agency pages are the authority." Pages read 2026-10-16 (excerpts kept in
`coastwatch-web/tests/unit/fixtures/agency-pages/`; a test checks every quote is verbatim; all quotes
re-checked live on 2026-10-18):

| Species | Toxins | Agency sentence (quoted) |
|---|---|---|
| Dungeness crab | domoic acid | CDPH pre-season testing; viscera holds much more than meat |
| Rock crab | domoic acid | viscera |
| Spiny lobster, anchovy/sardine | domoic acid | CDPH *may* increase sampling when blooms are seen |
| Mussels | PSP + domoic acid | routine shellfish monitoring (CDFW's page). The annual quarantine is an official notice, not a species fact: the registry record carries its status and CDPH's "will continue through at least October 31" (SN26-018) |
| Clams/oysters/scallops, razor clams | PSP + domoic acid | routine shellfish monitoring |
| Salmon, groundfish, halibut, squid | none named | a gap: "Neritic's sources name no biotoxin test specific to …" |

The toxins listed also include any `toxins` of the notices here that name the species, so a new
agency notice extends the line without a code change. For PSP the line says plainly that Neritic
carries no seawater reading of the algae behind it (CalHABMAP's Alexandrium counts are a candidate
source; see `03-alexandrium-cdip-feasibility.md`).

**Owner's decision (2026-10-16), as built in `pageClaims`:** at most six "Right now" lines, ranked
by relevance; a line is added for what it says, not because the feature exists.
- A species line that is only a gap (salmon, groundfish, halibut, squid) stays in the API and adds
  no page line.
- Past six lines, satellite chlorophyll gives way first, then the model, then the sea. Official
  words, the measurement and the species' toxins always stay.
- The species line shows its headline, what the readings can say and its provenance. Every agency
  sentence sits behind "What the agencies' pages say" (closed by default), each quoted with its own
  agency and page; no paraphrase is shown under an agency's name.

Measured at 390 × 844 (fixture clock): the first screen still holds the official line and the HAB
line. The species line is 300 → 213 px tall for Dungeness crab and 362 → 319 px for mussels.
Salmon goes from five lines back to four.

## 3. Consistency rules (`src/lib/evidenceCheck.ts`)

`checkGraph(graph)` returns violations of seven rules: **cites** (reading claims cite existing
nodes), **numbers** (headline numbers are in the cited nodes; any other number is somewhere in the
graph or is a stated rule), **dates** (headline dates are cited nodes' dates), **kind words**
(chlorophyll with "not toxin", a model as a probability and not a measurement, on the page *and* in
the API text separately; a measurement never a forecast; seawater never seafood; readings, cells and
chlorophyll never "about" toxin; an agency-page line carries its "Neritic's copy … not reviewed by a
person" provenance and no calendar end date), **safety**,
**reach** (context/far evidence said as such) and **once** (no sentence twice among page claims).

Run over every harbour of the fixture ports × activity × species × six model, six satellite and two
conditions states at three dates (≈ 1,900 graphs): zero violations. It found three real defects,
fixed in the graph rather than by loosening a rule:

- the tide heights on the conditions line came from no node: tide predictions are now a node
  (`tides:…`, published forecast);
- the HAB line named a closer, silent station ("Monterey Wharf, 0.6 km, is closer…") with no node:
  it is now `silent:<station>`, with its last sample date, cited by the line;
- the record sentence's "last as high" and "previous" values were not in its node.
