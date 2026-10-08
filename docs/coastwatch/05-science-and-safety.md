# 05 — Scientific and safety constraints

These rules are product requirements, not style guidance. Each has a test in `06-development-plan.md` §5. A pull request that violates one is not mergeable.

---

## 1. Information hierarchy (strict, top wins)

| Rank | Class (`product_class`) | Examples | May it be overridden? |
|---|---|---|---|
| 1 | `official_regulatory` | CDFW closures, season delays, Director's declarations; CDPH quarantines and health advisories; OEHHA consumption advisories; MPA boundaries | Never, by anything below |
| 2 | `official_forecast` | C-HARM bloom / domoic-acid probabilities; NWS marine forecasts and warnings | Only by rank 1 |
| 3 | `observation` | VIIRS/PACE chlorophyll; HABMAP shore-station cell counts and toxin samples | Only by ranks 1–2 |
| 4 | `experimental_model` | This project's ConvLSTM/PINN chlorophyll outlooks | Only by ranks 1–3; never sole basis of any statement |
| 5 | `historical_context` | Landings, revenue, past closure history | Descriptive only; never a prediction |

**Rules derived from the hierarchy**

1. **R1 — Regulatory override.** If any rank-1 record applies to a port, zone, or species, it is shown first, in full, and no lower-rank layer may produce a contrary-sounding summary for the same place/species (e.g., no "low bloom probability" headline above an active crab delay).
2. **R2 — Unknown ≠ open.** If the regulatory feed for a place is stale, failed, or not covered, the UI shows "Status not verified — check CDFW / CDPH" with links. It never shows "no advisories."
3. **R3 — No safety labels.** The product never labels any area, species, or date "safe," "clear," "go," or green-for-go. Low values are described as "lower estimated probability" with the product name and threshold.
4. **R4 — No harvest/consumption advice.** We restate official advisories verbatim (with link); we do not interpret toxin risk for consumption.

## 2. Chlorophyll

5. **R5 — Chlorophyll is biomass, not toxin and not fish.** Every chlorophyll layer and summary carries the sentence: "Chlorophyll measures algae biomass. It does not measure toxins and does not predict where fish are."
6. **R6 — No within-map percentiles as risk.** Tiers relative to the current map extent (the old `regional.py` tertiles) are prohibited. Allowed comparisons:
   - Absolute value with the product's own units and legend (mg m⁻³, log scale).
   - Anomaly vs a **fixed, documented climatology** (per pixel, per calendar period, stated baseline years and sensor), labeled "above/below the usual for this time of year."
7. **R7 — One product per legend.** Do not composite products with different algorithms (VIIRS, PACE, MODIS) into one visual unless a documented merged product is used.
8. **R8 — Missing data is visible.** Cloud gaps render as a distinct "no data" pattern, never as low values, and summaries report % valid pixels.

## 3. Official HAB forecasts (C-HARM)

9. **R9 — Quote the definition.** C-HARM layers are labeled with the event they predict and its threshold (e.g., "Probability that *Pseudo-nitzschia* ≥ 10,000 cells/L"), the issue time, and the valid time/lead.
10. **R10 — Probability is not certainty or toxin in seafood.** C-HARM predicts water-column conditions; shellfish/crab toxin levels are determined by CDPH/CDFW sampling. Copy must say so.
11. **R11 — Respect the product's own caveats** (coverage, skill documentation, nearshore gaps) and link to the producer's page.

## 4. Experimental models (this project)

12. **R12 — Visually and verbally distinct.** Experimental layers use a distinct legend frame ("Experimental — research model"), are off by default, are never on the same toggle group as official forecasts, and are never summarized in the port status card.
13. **R13 — Model card required.** A model may appear only with a public model card: training period, inputs, sensor, skill vs persistence on held-out data (same-pipeline numbers only), known failure modes, and the date it was last validated against current inputs.
14. **R14 — No current-condition claims from hindcasts.** The 2003–2021 predicted fields may be shown only on a historical/showcase page clearly dated.
15. **R15 — Operational gating.** The experimental live layer ships only after the retrained model beats persistence on a post-2022 held-out period using operational inputs (see `06-development-plan.md` §2, P3).
16. **R16 — Uncertainty or nothing.** Experimental forecasts display spread (ensemble/MC-dropout) or a skill map; a point forecast with no uncertainty is not shown.

## 5. Economics

17. **R17 — Historical exposure ≠ predicted loss.** "Historical economic exposure" = observed past landings value from HAB-sensitive species at a port (source, years, inflation basis, suppression noted). It is labeled as history. Any modeled expected revenue loss is a separate, later product with its own methodology page and uncertainty; the two are never summed or shown as one number.
18. **R18 — Respect confidentiality suppression.** Never back-calculate suppressed (fewer-than-three-participants) values from totals.
19. **R19 — No individual-level inference.** No vessel- or dealer-level economics.

## 6. Provenance, time, and uncertainty

20. **R20 — Every layer and number carries:** source name + link, product version, `issued_at`, `valid_time` (or `valid_start`/`valid_end`), `retrieved_at`, spatial resolution, and a freshness state (`current` / `stale` / `failed` / `not_covered`).
21. **R21 — Freshness thresholds are per source** (defined in `02-data-sources.md`), and stale data is labeled in place, not just in a footer.
22. **R22 — Times are shown in Pacific local time with UTC on hover**; composites show their full window.
23. **R23 — Last-known-good, labeled.** On ingestion failure, serve the last good artifact with a "stale since …" label; never silently substitute synthetic or fallback values.

## 7. Language and access

24. **R24 — Plain language first, technical detail on demand.** Headlines in short sentences; units and methods one click away.
25. **R25 — No color-only encoding.** Status uses icon + text + color; palettes are colorblind-safe and legible in sunlight (high contrast mode).
26. **R26 — Disclaimers are not a substitute for correct design.** The banner "not a regulatory product" does not license any violation above.

## 8. Standing disclaimer (short form)

> Coastwatch brings together public data from state and federal agencies and research models. It is not an official source. Closures, advisories, and seasons from CDFW, CDPH and OEHHA always take precedence. Chlorophyll does not measure toxins. Forecasts are probabilities, not guarantees.
