/**
 * Collision-aware placement for everything written on the chart: soundings, notes, port names, water names.
 * Pure and deterministic, so it is unit tested and runs per frame while the camera moves.
 *
 * Each item has a point on screen (where it was measured or where the place is) and a measured box. Items
 * are placed by priority, highest first; each tries its anchors in order (the anchor it had last time first,
 * so labels do not jump while panning) and takes the first spot that stays inside the free area and clear of
 * every obstacle and every box already placed. An item with no spot is dropped (README §9 ChartNote: notes
 * drop by priority when space is short), unless it is `keep` (the reader's own port), which takes its first
 * anchor regardless. An `insist` item (a page's main figure on a small screen) is not dropped for another
 * place's dot: when no clear spot is left it takes the first spot inside the area that is clear of everything
 * but those dots.
 */
export type Anchor = "right" | "left" | "top" | "bottom" | "top-right" | "bottom-right" | "top-left" | "bottom-left" | "center";
export type Rect = { x0: number; y0: number; x1: number; y1: number };

export type PlaceItem = {
  id: string;
  /** the point, in screen px */
  x: number;
  y: number;
  width: number;
  height: number;
  /** higher is placed first */
  priority: number;
  anchors?: Anchor[];
  /** the gap between the point and the box, px */
  offset?: number;
  /** a mark drawn at the point (a port's circled dot): reserved for everyone, radius px */
  pointRadius?: number;
  /** never dropped */
  keep?: boolean;
  /** with its point in view, may cover lower places' dots rather than be dropped */
  insist?: boolean;
};

export type Placed = { id: string; left: number; top: number; anchor: Anchor; visible: boolean };

export const DEFAULT_ANCHORS: Anchor[] = ["right", "left", "top-right", "bottom-right", "top-left", "bottom-left", "top", "bottom"];
const GAP = 4;

export function boxAt(item: PlaceItem, anchor: Anchor): Rect {
  const o = item.offset ?? 8;
  const { x, y, width: w, height: h } = item;
  const d = o * Math.SQRT1_2;
  switch (anchor) {
    case "right":
      return { x0: x + o, y0: y - h / 2, x1: x + o + w, y1: y + h / 2 };
    case "left":
      return { x0: x - o - w, y0: y - h / 2, x1: x - o, y1: y + h / 2 };
    case "top":
      return { x0: x - w / 2, y0: y - o - h, x1: x + w / 2, y1: y - o };
    case "bottom":
      return { x0: x - w / 2, y0: y + o, x1: x + w / 2, y1: y + o + h };
    case "top-right":
      return { x0: x + d, y0: y - d - h, x1: x + d + w, y1: y - d };
    case "bottom-right":
      return { x0: x + d, y0: y + d, x1: x + d + w, y1: y + d + h };
    case "top-left":
      return { x0: x - d - w, y0: y - d - h, x1: x - d, y1: y - d };
    case "bottom-left":
      return { x0: x - d - w, y0: y + d, x1: x - d, y1: y + d + h };
    case "center":
      return { x0: x - w / 2, y0: y - h / 2, x1: x + w / 2, y1: y + h / 2 };
  }
}

export const overlaps = (a: Rect, b: Rect, gap = GAP) => a.x0 < b.x1 + gap && b.x0 < a.x1 + gap && a.y0 < b.y1 + gap && b.y0 < a.y1 + gap;
export const inside = (a: Rect, area: Rect) => a.x0 >= area.x0 && a.y0 >= area.y0 && a.x1 <= area.x1 && a.y1 <= area.y1;

export function placeAll(items: PlaceItem[], area: Rect, obstacles: Rect[] = [], previous: Map<string, Anchor> = new Map()): Placed[] {
  const taken: Rect[] = [...obstacles];
  // every point mark is reserved before any box is placed, so a name never covers another port's dot
  const dots = new Set<Rect>();
  for (const it of items) {
    if (!it.pointRadius || !pointIn(it, area)) continue;
    const dot = { x0: it.x - it.pointRadius, y0: it.y - it.pointRadius, x1: it.x + it.pointRadius, y1: it.y + it.pointRadius };
    dots.add(dot);
    taken.push(dot);
  }
  const order = [...items].sort((a, b) => b.priority - a.priority || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  const out = new Map<string, Placed>();
  for (const it of order) {
    const anchors = it.anchors ?? DEFAULT_ANCHORS;
    const prev = previous.get(it.id);
    const tries = prev && anchors.includes(prev) ? [prev, ...anchors.filter((a) => a !== prev)] : anchors;
    let placed: Placed | null = null;
    if (pointIn(it, area) || it.keep) {
      for (const a of tries) {
        const r = boxAt(it, a);
        if (!inside(r, area)) continue;
        if (taken.some((t) => overlaps(r, t) && !isOwnPoint(t, it))) continue;
        taken.push(r);
        placed = { id: it.id, left: r.x0, top: r.y0, anchor: a, visible: true };
        break;
      }
      if (!placed && it.insist && pointIn(it, area)) {
        for (const a of tries) {
          const r = boxAt(it, a);
          if (!inside(r, area) || taken.some((t) => !dots.has(t) && overlaps(r, t))) continue;
          taken.push(r);
          placed = { id: it.id, left: r.x0, top: r.y0, anchor: a, visible: true };
          break;
        }
      }
      // kept even without a free spot, but never written over a panel when its point lies under one
      if (!placed && it.keep && pointIn(it, area)) {
        const r = boxAt(it, tries[0]);
        taken.push(r);
        placed = { id: it.id, left: r.x0, top: r.y0, anchor: tries[0], visible: true };
      }
    }
    out.set(it.id, placed ?? { id: it.id, left: 0, top: 0, anchor: tries[0], visible: false });
  }
  return items.map((it) => out.get(it.id)!);
}

const pointIn = (it: PlaceItem, a: Rect) => it.x >= a.x0 && it.x <= a.x1 && it.y >= a.y0 && it.y <= a.y1;
const isOwnPoint = (t: Rect, it: PlaceItem) => !!it.pointRadius && t.x0 === it.x - it.pointRadius && t.y0 === it.y - it.pointRadius;
