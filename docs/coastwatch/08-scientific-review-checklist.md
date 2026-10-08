# 08 — Scientific review checklist (C-HARM layer, Milestone 1)

For a reviewer familiar with harmful algal blooms and domoic acid on the US West Coast. About 20 minutes. Open the app (or the screenshots in [`m1/`](m1/)) and the C-HARM ERDDAP pages linked in the app. Mark each item **OK**, **Change** (with wording), or **Unsure**.

## A. Variables and thresholds

| # | Check | What the app says | Source of truth |
|---|---|---|---|
| A1 | `pseudo_nitzschia` is described correctly | "Probability that Pseudo-nitzschia exceeds 10,000 cells per litre" | ERDDAP long_name: "Probability of Pseudo-nitzschia > 10,000 cells/L" |
| A2 | `particulate_domoic` is described correctly | "Probability that particulate domoic acid exceeds 500 ng per litre"; explainer: "domoic acid in the plankton … per litre of seawater" | "Probability of Particulate Domoic Acid > 500 nanograms/L" |
| A3 | `cellular_domoic` is described correctly | "Probability that cellular domoic acid exceeds 10 pg per cell"; explainer: "toxin per Pseudo-nitzschia cell" | "Probability of Cellular Domoic Acid > 10 picograms/cell" |
| A4 | Short labels are not misleading | Buttons: "Bloom", "Particulate DA", "Cellular DA"; heading "Bloom and domoic acid forecast" | — |
| A5 | Probabilities are shown as 0–100% on a fixed scale; no rescaling per map | Legend 0, 25, 50, 75, 100% | — |
| A6 | Is it acceptable to call a ≥ 10,000 cells/L event a "bloom"? | Uses C-HARM's own definition | Anderson et al. 2016 |

## B. Time

| # | Check | What the app says |
|---|---|---|
| B1 | Lead semantics | "Nowcast" = valid day before the run; "+1/+2/+3 days" = valid days after the nowcast |
| B2 | Issue date inference is reasonable | "Issued … (inferred)": nowcast valid day + 1, because C-HARM publishes only valid days |
| B3 | Freshness thresholds suit a daily product with frequent gaps | Current ≤ 1 day since issue; stale 2–7 days; historical > 7 days |
| B4 | Missing leads | Shown as "not issued"; never filled from an older run |

## C. Interpretation and caveats (all shown without expanding anything)

| # | Check | Text in the app |
|---|---|---|
| C1 | Not a seafood toxin measurement or closure decision | "A forecast probability, not a measurement of toxin in seafood and not a closure decision." / "…CDFW and CDPH decide closures and advisories." |
| C2 | Low ≠ safe | "A low probability does not mean an area is safe." |
| C3 | Skill statement is honest | "Published skill assessment covers C-HARM v1 (Anderson et al. 2016); no published skill assessment for v3.1 was found." — Is there a v3 / v3.1 assessment we should cite instead (e.g., work using WCOFS)? |
| C4 | Known weaknesses | "…salinity errors can affect domoic acid predictions and … bloom predictions include many false positives" (from the SCCOOS Dec 2022 bulletin). Still accurate? |
| C5 | Nearshore masking | "Toxin probabilities are not provided for many cells within about 3–6 km of shore, including piers and harbours." The inspector then shows the nearest forecast cell with its distance. Is "nearest cell" appropriate, or should nearshore points show nothing? |
| C6 | Badge wording | C-HARM is badged "Agency forecast" (not "official"), and closures are a separate "Official closures and health advisories" card stating they are not tracked yet. Is the distinction clear? |

## D. Values

| # | Check |
|---|---|
| D1 | Pick 2–3 points you know (e.g., Monterey Bay mid-bay 36.80 N 121.95 W). Compare the inspector with ERDDAP for the same valid day (link under "Source and provenance"). The automated check matched 156/156 published values against ERDDAP on 2026-10-08 (`evidence/m1-charm-verification-2026-10-08.json`). |
| D2 | Does the spatial pattern look like the ERDDAP preview for the same day (no shift, flip, or offset)? |

## E. Anything missing or overstated

| # | Question |
|---|---|
| E1 | Any statement that implies toxicity of seafood, safety, or fishing productivity? (Automated checks forbid phrases like "safe to eat", "all clear", "go fishing", "best place to fish".) |
| E2 | Should the CalHABMAP shore-station observations or the California HAB Bulletin be linked next to the forecast in M1? |
| E3 | Satellite chlorophyll copy: "Chlorophyll measures algae biomass. It does not measure toxins and does not predict where fish are." Adequate? |
| E4 | Preferred citation for C-HARM v3.1 and contact for a skill statement. |

Reviewer: ____________________  Date: __________  Overall: ☐ Approve  ☐ Approve with changes  ☐ Do not publish yet
