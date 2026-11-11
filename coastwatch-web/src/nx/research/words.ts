/**
 * The few sentences Research quotes exactly (truth rules T10, T34; E-science-and-perf.md finding 0.3).
 * Quoted only on the Research surface; no other model figure is quoted anywhere.
 */

/** T10: near every C-HARM figure. */
export const MODEL_WORDS = "Probability of conditions in the water, not toxin in seafood.";

/** T34, exactly. */
export const CHARM_ASSESSMENT = "Published assessment covers v1 (Anderson 2016); none found for v3.1. Producer notes many false positives.";

/** T34 nearshore, with the AUC and BSS allowed on Research only (E finding 0.3, same-day v3.1 nowcast at the nearest cell). */
export const CHARM_SHORE =
  "At the shore it has no demonstrated skill: against CalHABMAP piers, the same-day v3.1 nowcast at the nearest cell scored AUC 0.30 (CI 0.23–0.39) and BSS −11.5 for P(pDA > 500 ng/L), n = 1,058 with 46 events. A high reading here is near-constant background, not an alarm.";

/** T11: what the model strip is a number of. */
export const modelPlace = (km: number) => `The median of the model cells within ${km} km of the station each day: not a value at the pier.`;
