# Brief: Prof. Oliver Fringer, Thu 2026-11-12

Sources: [`../plan.md`](../plan.md) §1.8 and §3F (research pass, 2026-11-08). Items marked **(unverified)** were not confirmed against a primary source.

## Who he is
- Stanford Civil & Environmental Engineering; affiliated with the Doerr School / oceans programs. Developer of **SUNTANS**, an unstructured-grid nonhydrostatic coastal ocean model used heavily for San Francisco Bay.
- PI, with **David Senn (SFEI)**, of a Woods Institute **"Big Ideas for Oceans"** project on Bay Area HAB forecasting using machine learning plus model fusion, motivated by the **2022 Bay *Heterosigma* bloom** (plan §1.8).
- Student **Sarah Chang** works on ML for HABs. That she is the student attending is an **inference (unverified)**.

## Where Neritic fits
His project produces Bay HAB knowledge; Neritic is a **last-mile dissemination layer** (official status, tests, conditions, alerts to fishermen's phones) and can offer a **shared validation baseline** (climatology + AR(1) on the same held-out period). We are complementary, not competing: Neritic does not make its own toxin forecast (plan §4).

## Do not claim
- That the PINN beats anything. Safe line: "the PINN matches, does not beat, the ConvLSTM" (CLAUDE.md). The audit findings are *possible issues requiring independent reproduction* (AUDIT.md, standing note).
- That chlorophyll is toxin, or that observations are forecasts. The 8-day outlook is chlorophyll only; toxin-risk probability (C-HARM) is NOAA's and stops at day 3.
- That notices are verified (they are "Not verified" unless the owner runs `cwp review-official`).
- Willingness-to-pay, "$5–15k per trip", ">$100M" losses (plan §1.7). Bay coverage: the 8-day outlook **does not cover the bays** (#74).
- That fishermen asked for Neritic specifically. Zero interviews done yet.

## Asks, ranked
1. **Methods critique and a shared baseline.** Will he (or his student) look at the clim+AR(1) baseline and the PINN audit, and compare against their Bay ML models on one protocol?
2. **Faculty PI or mentor on a Big Ideas for Oceans proposal.** Up to $150k, "expected to reopen in November", needs a faculty PI (plan §1.8; reopening date **unverified**). Owner offers to write the first draft.
3. **Data and model access.** SUNTANS / SFEI Bay hindcasts for validation; permission to surface their validated Bay HAB likelihood maps in Neritic with attribution.
4. **Introductions:** David Senn (SFEI); CeNCOOS; Raphael Kudela (UCSC, C-HARM); California Sea Grant extension (fishermen interviews); Moss Landing Marine Labs / Hopkins Marine Station.
5. **A defined student role and venue** (directed study or RA) for the corrected PINN re-analysis, including where an honest negative result could be written up.

## Questions to ask him
- What does the 2022 *Heterosigma* event teach about lead time that matters to users?
- Which Bay observations (SFEI moorings, USGS cruises) would he trust as ground truth for a shared baseline?
- Is an 8-day chlorophyll outlook useful at all, or is it noise next to physics-driven models?
- Would Stanford host an independent review of the notices and science text?
- What would make a student project here publishable, and what would make him decline?
- Is there a timeline for his project's maps being public?
