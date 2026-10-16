# Official notices: a guide to the human review (prepared 2026-10-16)

For the owner, or any person checking the official-notice registry
(`data/curated/official_notices.json`) against CDFW and CDPH. Nothing here has been changed in the
registry. Its review status is `pending_human_review`: the records are an AI transcription of
2026-10-08 and later, and no person has checked them. Neritic shows every notice as **Not
verified** until a person does. Claude must not mark anything verified, lifted or expired; only the
person doing this review may.

The notes under "What the source appears to say" are Claude's reading of the pages on 2026-10-16.
They are a starting point, not a finding: read each source yourself.

## Two facts about the tool before you start

1. **`cwp review-official` covers the whole registry, not one record.** `--status human_verified
   --confirm` states that you checked *every* record (11 now) against its sources. There is no
   per-record "verified" flag.
2. **It also marks every CDPH release currently listed as reviewed, even with
   `--status pending_human_review`.** It stores the watched pages' fingerprints and adds every
   release id on CDPH's list (now SN26-021 and SN26-020) to `cdph_release_ids_reviewed`. That clears
   the "Agency pages have changed since this list was last reviewed" warning and the automatic
   "SN26-021 appears to lift SN26-018" flag. **Run it only when every record is right.** If any
   question below stays open, don't run it at all. The open item then stays visible as unreviewed
   or disputed, which is the honest state.

## What changed since the last review

The watcher (data run of 2026-10-16 11:27 UTC) reports two changed sources:

- the CDPH shellfish advisories list: new releases **SN26-020** and **SN26-021**;
- the CDFW Health Advisories page: its news list and razor-clam section changed.

### SN26-018 (Monterey County bivalves), possibly lifted by SN26-021

| | |
|---|---|
| Registry record | `cdph-2026-sn26-018-monterey-bivalves`: `active`, effective 2026-09-14; mussels, clams, scallops, oysters; Monterey County |
| Sources | https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-018.aspx · https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-021.aspx · CDPH interactive advisory map (linked from both) |
| What the source appears to say | SN26-021, dated **October 12, 2026**, "CDPH Lifts Warnings about Sport-Harvested Bivalve Shellfish from Monterey County": the September 14 notification is lifted; "while this warning for clams, scallops and oysters in Monterey County has been lifted, the annual mussel quarantine remains in effect" (through at least October 31). CDFW's news list also shows the lift (10/12/2026). |
| What Neritic shows now | the record as **disputed**: "CDPH appears to have lifted … Confirm with CDPH which applies before relying on either". |
| If you confirm the lift | set the record's `status` to `"lifted"` and `lifted_date` to `"2026-10-12"`, and add SN26-021's URL to its `sources`. Leave `cdph-2026-annual-mussel-quarantine` active: it still covers mussels in Monterey County. Validation rejects `lifted` without a `lifted_date`. |
| If you cannot confirm it | for example, if the CDPH interactive map still shows Monterey under a warning: change nothing and don't run `review-official`. The page keeps the record disputed. |

### SN26-020 (Del Norte County razor clams)

| | |
|---|---|
| Registry records | `cdph-2026-sn26-020-razor-clam-del-norte` (CDPH advisory) and `cdfw-2026-razor-clam-del-norte` (CDFW fishery closure): both `active`, effective 2026-10-09; both already transcribed, never checked by a person |
| Sources | https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-020.aspx · https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories |
| What the source appears to say | CDPH advises against eating sport-harvested razor clams from Del Norte County (only razor clams), "in addition to the razor clam advisory for Humboldt County". CDFW: "The recreational razor clam fishery closed in Del Norte County on October 9, 2026, and remains closed in Humboldt County". |
| Check | the CDPH release date. The release page reads **October 9, 2026**, but CDFW's news list dates the CDPH release **10/8/2026**; the record says 2026-10-09. Also check the CDFW record's open uncertainty: the OEHHA memo or CDFW news release for the closure hasn't been located. |
| If confirmed | correct `effective_date` if needed, and resolve or keep the uncertainty in words. Nothing else changes. |

### The other records

`human_verified` covers these too: the annual mussel quarantine (to 2026-10-31), the anchovy
advisory (SN26-019) and CDFW take restriction (Pigeon Point to Point Lobos), the Humboldt razor
clam advisory and closure, the commercial rock-crab closure north of 40° N, the rock-crab viscera
advisory, and the Northern Channel Islands bivalve advisory. Open each record's `sources` and
compare `official_text`, the area, the dates and the species.

## Steps

1. Read every source above, and each other record's sources if you intend to verify the whole
   registry.
2. Make the record edits you confirmed on a branch from `main`, after #65 has merged (#65 changes one
   line of this file). Write `summary` in your own words, and keep `official_text` verbatim.
3. Check that the registry still validates: `cd pipeline && uv run cwp watch-official`. It is
   read-only and always exits 0, so read its output: `registry_errors` must be `[]`, and
   `conflicts` should no longer list SN26-021 if you recorded the lift.
4. Only if **every** record is confirmed:
   `cd pipeline && uv run cwp review-official --reviewer "<your name>" --confirm`.
   Otherwise skip this step: the registry stays `pending_human_review` and Neritic keeps saying
   Not verified.
5. Open a PR. Once it merges into `main`, the next scheduled data run (every 6 h, at minute 17)
   publishes the registry to both the public site and any preview.

Claude can prepare step 2's edits as a draft for you to check, but only on your instruction for each
record, and the commit and the `review-official` run must be yours.
