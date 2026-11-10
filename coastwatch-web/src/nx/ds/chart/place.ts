/**
 * Placement shared by everything set on the chart. The chart host computes screen positions, collisions
 * and priorities; these components only draw at the place they are given.
 */
export type Anchor = "ne" | "nw" | "se" | "sw" | "e" | "w" | "n" | "s";

/** Where the text block sits relative to its anchor point, as a CSS translate. */
export function anchorTranslate(a: Anchor): string {
  switch (a) {
    case "ne":
      return "translate(0, -100%)";
    case "nw":
      return "translate(-100%, -100%)";
    case "se":
      return "translate(0, 0)";
    case "sw":
      return "translate(-100%, 0)";
    case "e":
      return "translate(0, -50%)";
    case "w":
      return "translate(-100%, -50%)";
    case "n":
      return "translate(-50%, -100%)";
    case "s":
      return "translate(-50%, 0)";
  }
}

export const anchorAlign = (a: Anchor): "left" | "right" | "center" => (a === "nw" || a === "sw" || a === "w" ? "right" : a === "n" || a === "s" ? "center" : "left");

/** A small gap between a leader's end and the type, in the direction of the anchor. */
export function anchorGap(a: Anchor, gap = 4): { x: number; y: number } {
  const x = a.includes("e") ? gap : a.includes("w") ? -gap : 0;
  const y = a.startsWith("n") ? -gap : a.startsWith("s") ? gap : 0;
  return { x, y };
}

/** Lower priority numbers are kept first; the host drops from the end when space is short. */
export function byPriority<T extends { priority?: number }>(items: T[]): T[] {
  return [...items].sort((a, b) => (a.priority ?? 5) - (b.priority ?? 5));
}
