# v3 · station record, species lines, consistency rules

## 1. A measurement against its own station's record (`src/lib/history.ts`)

CalHABMAP piers have weekly samples since 2014 (Santa Cruz, Monterey, Cal Poly, Stearns, Santa
Monica, Newport, Scripps), since 2017 (Trinidad, Humboldt) or 2020–23 (Bodega, Tomales, Morro Bay).
The published observations already hold the full series, so the comparison is computed server-side
in `slimStations` and travels as `StationLite.record`; no pipeline or manifest field.

Said, on the HAB line, only when the line's sample is the station's newest:

> Compared with earlier years at Santa Cruz Wharf: higher than 98% of the Aug–Oct samples there in
> 2014–2025. Last as high or higher: September 2 (15,900 cells/L). Previous sample September 23: 100 cells/L.

Rules: same season = the sample's month and its neighbours, earlier years only; a rank needs ≥ 12
such samples from ≥ 3 years, otherwise "too few samples … to compare by season"; reported zeros
count as 0 and are said as "none counted" / "not quantified"; QC-rejected values are dropped;
Pseudo-nitzschia is the seriata + delicatissima sum, as the briefing reports it; percentages never
round to 0 % or 100 % unless exact. Never "unusual", "risk", "safe": it is the station's own past,
not other waters, seafood or the future. The sentence is its own node (`record:…`, class
`record_statistic`) and its own field (`Claim.history`), not part of the claim's `text`, which the
HAB summary paragraph repeats.

## 2. The chosen species (`src/content/speciesEvidence.ts`)

For a chosen species, one line: which toxins the agencies watch for in it, the agency's sentence
quoted, and what Neritic's readings are evidence of. Pages read 2026-10-16 (excerpts kept in
`coastwatch-web/tests/unit/fixtures/agency-pages/`; a test checks every quote is verbatim):

| Species | Toxins | Agency sentence (quoted) |
|---|---|---|
| Dungeness crab | domoic acid | CDPH pre-season testing; viscera holds much more than meat |
| Rock crab | domoic acid | viscera |
| Spiny lobster, anchovy/sardine | domoic acid | CDPH *may* increase sampling when blooms are seen |
| Mussels | PSP + domoic acid | annual quarantine May 1 – Oct 31; routine shellfish monitoring |
| Clams/oysters/scallops, razor clams | PSP + domoic acid | routine shellfish monitoring |
| Salmon, groundfish, halibut, squid | none named | a gap: "Neritic's sources name no biotoxin test specific to …" |

The toxins listed also include any `toxins` of the notices here that name the species, so a new
agency notice extends the line without a code change. For PSP the line says plainly that Neritic
carries no seawater reading of the algae behind it (CalHABMAP's Alexandrium counts are a candidate
source, not yet read). Owner decision: with a species chosen, "Right now" can show six lines.

## 3. Consistency rules (`src/lib/evidenceCheck.ts`)

`checkGraph(graph)` returns violations of seven rules: **cites** (reading claims cite existing
nodes), **numbers** (headline numbers are in the cited nodes; any other number is somewhere in the
graph or is a stated rule), **dates** (headline dates are cited nodes' dates), **kind words**
(chlorophyll with "not toxin", a model as a probability and not a measurement, on the page *and* in
the API text separately; a measurement never a forecast; seawater never seafood), **safety**,
**reach** (context/far evidence said as such) and **once** (no sentence twice among page claims).

Run over every harbour of the fixture ports × activity × species × six model, six satellite and two
conditions states at three dates (≈ 1,900 graphs): zero violations. It found three real defects,
fixed in the graph rather than by loosening a rule:

- the tide heights on the conditions line came from no node: tide predictions are now a node
  (`tides:…`, published forecast);
- the HAB line named a closer, silent station ("Monterey Wharf, 0.6 km, is closer…") with no node:
  it is now `silent:<station>`, with its last sample date, cited by the line;
- the record sentence's "last as high" and "previous" values were not in its node.
