# v4 on main with Harbor Intelligence v3 (#66), 2026-10-19

## What happened, in order

1. Before merging #66, checked:
   - head `9bf320d95` (local and PR identical);
   - every CI check green (pipeline, web-static, both browser shards, the web gate; the branch has no
     required checks configured);
   - the seven scientific corrections and their regression tests present (`40d3cf9`, `9bf320d`).
2. Rollback references, all kept:
   - `archive/main-before-66` (= `4d2e3e1c`, also `demo-release/2026-10-16`);
   - `archive/neritic-v4-before-rebase-on-66` (= `2b490f9`);
   - the `feat/harbor-intelligence-v3` branch is not deleted.
3. #66 squash-merged into main as `b78c69f`. Main's tree is byte-identical to the tested v3 head.
4. `feat/neritic-v4` rebased onto `b78c69f` (approved by the owner). The 33 commits applied without a
   textual conflict.
5. One integration commit (`7e38b09`): the corrected species line and station record in v4's claim
   rows, and the small-sample disclosure.

## The species line's schema change, resolved without weakening it

#66 removed `Claim.more` (Neritic's paraphrases shown as the agency's words), added `Claim.agencyWords`,
`Claim.provenance` and the claim kind `agency_page`.

v4's claim rows (`ClaimRow`, used by the briefing and Explore's harbour panel) looked a kind up in a
string map and fell back to the grey "Not known" gap style. The typechecker could not see this.
Untreated, the species line would have looked like a gap, without its provenance or its agency words.
The integration renders:

- the `agency_page` kind: the official colour, drawn hollow (never a notice's filled mark), labelled
  "Agency page · Neritic's copy";
- the provenance line: "Neritic's copy of the agencies' pages, read October 16, 2026; not reviewed by
  a person. The agency pages are the authority.";
- every agency sentence quoted under its own agency and page ("What the agencies' pages say");
- the station-record sentence (`history`).

Nothing renders `Claim.more`.

#66's two page tests are ported to v4's claim ids (`claim-*`). Every corrected assertion is kept: the
kind, the provenance, both crab quotes with their CDPH links, the mussel line's CDFW sentence under
CDFW, no "October 31", and quantified ranks with the zeros said apart. The only change is that the
apostrophe accepts v4's typographic ’.

## Owner's check 7: small samples and below-detection-limit handling

Every record rank now says, in the same sentence:

- the sample size it rests on: "of the 16 quantified Aug–Oct samples";
- when that size is small (under 30), that it is small and what one sample is worth: "(a small sample:
  each of the 16 is 6 points of the rank)";
- how values below the detection limit were handled: "another 126 had no cells counted (not
  necessarily none present) and are not ranked", or for toxin "were reported 0 (not quantified, not
  absent) and are not ranked";
- with too few quantified samples, no percentage at all, only counts.

Held by `tests/unit/history.test.ts` ("discloses a small sample and the below-detection-limit handling
in the same sentence") and the page test's pattern.

## Unchanged

- Official notices: unverified, with their review state (no record marked verified, lifted or expired).
- Every scientific caveat.
- The C2 logo: still an unapproved candidate, not installed.
- The public research preview: not deployed.
