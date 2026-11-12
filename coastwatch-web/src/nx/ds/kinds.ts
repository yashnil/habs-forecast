/**
 * The kinds of reading (truth rule T1) and the five semantic marks (spec section 4) they are drawn with.
 * Shape first, colour second: a greyscale print still separates measured from model (T2).
 */

/** Kind words, exactly as T1 lists them. */
export type Kind = "measured" | "analysis" | "estimated" | "model" | "outlook" | "prediction" | "official";

/** The five marks. `official` always travels with the words "not verified" (T38). */
export type MarkKind = "measured" | "model" | "official" | "noview" | "stale";

export const KIND_WORD: Record<Kind, string> = {
  measured: "measured",
  analysis: "analysis",
  estimated: "estimated",
  model: "model",
  outlook: "outlook",
  prediction: "prediction",
  official: "official",
};

/** Observed things are solid; anything computed about the water is hollow or hatched. */
export function markFor(kind: Kind): Extract<MarkKind, "measured" | "model" | "official"> {
  switch (kind) {
    case "measured":
      return "measured";
    case "official":
      return "official";
    default:
      return "model";
  }
}

/** Sentence-case kind label for the start of a stamp: "Measured", "Official". */
export function kindLabel(kind: Kind): string {
  const w = KIND_WORD[kind];
  return w.charAt(0).toUpperCase() + w.slice(1);
}

/** The words each mark carries in a legend. One legend per view replaces per-figure caveats. */
export const MARK_WORDS: Record<MarkKind, string> = {
  measured: "Measured",
  model: "Model or prediction",
  official: "Official, not verified",
  noview: "No view: no data, not a low value",
  stale: "Older than its update cycle",
};

/** Verification is words, never an icon alone (T38). Neritic never writes "verified" itself. */
export const NOT_VERIFIED = "not verified";
