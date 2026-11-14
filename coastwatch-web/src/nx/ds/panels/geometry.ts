/**
 * The panels' geometry (v15.1): pure functions, no DOM. A panel is a box in the stage's coordinates (px from the
 * stage's top-left); the area is the part of the stage a panel may stand in (inside the gutters, under the bar).
 * Nothing here knows about tickets or docks.
 */

export type Box = { x: number; y: number; w: number; h: number };
export type Area = { left: number; top: number; right: number; bottom: number };
export type Size = { w: number; h: number };
export type Limits = { minW: number; maxW: number; minH: number; maxH: number };

/**
 * Where a moved panel stands, held against its nearest edges so it keeps its corner when the screen changes size: a
 * panel left near the right edge stays near the right edge on a wider screen.
 */
export type Placement = { ax: "l" | "r"; dx: number; ay: "t" | "b"; dy: number };

export const areaW = (a: Area) => Math.max(0, a.right - a.left);
export const areaH = (a: Area) => Math.max(0, a.bottom - a.top);

/** A size within the limits and never larger than the area. */
export function clampSize(size: Size, lim: Limits, area: Area): Size {
  const maxW = Math.min(lim.maxW, areaW(area));
  const maxH = Math.min(lim.maxH, areaH(area));
  return {
    w: Math.round(Math.max(Math.min(lim.minW, maxW), Math.min(maxW, size.w))),
    h: Math.round(Math.max(Math.min(lim.minH, maxH), Math.min(maxH, size.h))),
  };
}

/**
 * A box kept on screen: wholly inside the area when it fits; a box larger than the area keeps its top-left corner
 * (its header and controls) inside. Never moves a box that is already inside.
 */
export function clampBox(b: Box, area: Area): Box {
  const x = b.w >= areaW(area) ? area.left : Math.min(Math.max(b.x, area.left), area.right - b.w);
  const y = b.h >= areaH(area) ? area.top : Math.min(Math.max(b.y, area.top), area.bottom - b.h);
  return { x: Math.round(x), y: Math.round(y), w: b.w, h: b.h };
}

/** The box held against its nearest horizontal and vertical edges of the area. */
export function toPlacement(b: Box, area: Area): Placement {
  const l = b.x - area.left, r = area.right - (b.x + b.w);
  const t = b.y - area.top, btm = area.bottom - (b.y + b.h);
  return {
    ax: r < l ? "r" : "l",
    dx: Math.round(Math.max(0, r < l ? r : l)),
    ay: btm < t ? "b" : "t",
    dy: Math.round(Math.max(0, btm < t ? btm : t)),
  };
}

/** The box a placement means for a panel of this size in this area, kept on screen. */
export function fromPlacement(p: Placement, size: Size, area: Area): Box {
  const x = p.ax === "r" ? area.right - p.dx - size.w : area.left + p.dx;
  const y = p.ay === "b" ? area.bottom - p.dy - size.h : area.top + p.dy;
  return clampBox({ x, y, w: size.w, h: size.h }, area);
}

export const validPlacement = (p: unknown): p is Placement => {
  if (!p || typeof p !== "object") return false;
  const q = p as Record<string, unknown>;
  return (q.ax === "l" || q.ax === "r") && (q.ay === "t" || q.ay === "b") && Number.isFinite(q.dx) && Number.isFinite(q.dy) && (q.dx as number) >= 0 && (q.dy as number) >= 0;
};

/** A key's move: arrows by `step` px (a grip), clamped to the area. null for any other key. */
export function nudge(b: Box, key: string, step: number, area: Area): Box | null {
  const d: Record<string, [number, number]> = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
  const m = d[key];
  if (!m) return null;
  return clampBox({ ...b, x: b.x + m[0], y: b.y + m[1] }, area);
}

/** A guide shown while dragging: a vertical (`x`) or horizontal (`y`) line where the panel's edge meets another edge. */
export type Guide = { axis: "x" | "y"; at: number; from: number; to: number; kind: "area" | "panel" };

/**
 * The guides for a box being dragged: each of its edges within `tol` px of the area's edge or of another panel's edge
 * (its near or far side) shows a line there. Guides only show; nothing pulls the panel to them.
 */
export function guidesFor(b: Box, others: Box[], area: Area, tol = 6): Guide[] {
  const out: Guide[] = [];
  const xs = [b.x, b.x + b.w], ys = [b.y, b.y + b.h];
  const seen = new Set<string>();
  const add = (g: Guide) => {
    const k = `${g.axis}${Math.round(g.at)}`;
    if (seen.has(k)) return;
    seen.add(k);
    out.push(g);
  };
  for (const at of [area.left, area.right]) if (xs.some((x) => Math.abs(x - at) <= tol)) add({ axis: "x", at, from: area.top, to: area.bottom, kind: "area" });
  for (const at of [area.top, area.bottom]) if (ys.some((y) => Math.abs(y - at) <= tol)) add({ axis: "y", at, from: area.left, to: area.right, kind: "area" });
  for (const o of others) {
    for (const at of [o.x, o.x + o.w])
      if (xs.some((x) => Math.abs(x - at) <= tol)) add({ axis: "x", at, from: Math.min(b.y, o.y), to: Math.max(b.y + b.h, o.y + o.h), kind: "panel" });
    for (const at of [o.y, o.y + o.h])
      if (ys.some((y) => Math.abs(y - at) <= tol)) add({ axis: "y", at, from: Math.min(b.x, o.x), to: Math.max(b.x + b.w, o.x + o.w), kind: "panel" });
  }
  return out;
}

export type Side = "left" | "right" | "top" | "bottom" | "none";

/**
 * Which edge of the chart a panel standing here covers, for the camera (src/nx/chart/cover.ts): a tall panel the
 * nearer side, a wide one the nearer of top and bottom, if it stands within a quarter of the screen of that edge;
 * a panel floating in the middle is only an obstacle for the chart's names ("none").
 */
export function coverSideFor(b: Box, vp: Size): Side {
  const l = b.x, r = vp.w - (b.x + b.w), t = b.y, btm = vp.h - (b.y + b.h);
  const tall = b.h / Math.max(1, vp.h) >= b.w / Math.max(1, vp.w);
  if (tall) {
    const near = Math.min(l, r);
    if (near > vp.w * 0.25) return "none";
    return r <= l ? "right" : "left";
  }
  const near = Math.min(t, btm);
  if (near > vp.h * 0.25) return "none";
  return btm <= t ? "bottom" : "top";
}

/**
 * The panels' default, non-overlapping layout: a panel at its resting place (a dock along the foot) gives up width so
 * it stops `gap` px short of any resting panel beside it that it would otherwise run into (the ticket), never below
 * `minW`. Returns the width it may take.
 */
export function fitBeside(b: Box, others: Box[], gap: number, minW: number): number {
  let w = b.w;
  for (const o of others) {
    const vertical = o.y < b.y + b.h && o.y + o.h > b.y;
    if (!vertical) continue;
    // the other panel stands to its right, within its reach
    if (o.x > b.x && o.x < b.x + w + gap) w = Math.min(w, o.x - gap - b.x);
  }
  return Math.round(Math.max(Math.min(minW, b.w), w));
}

const overlap = (a: Box, b: Box) => a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
export const boxesOverlap = overlap;

/** True when the box overlaps any of the others. */
export const hitsAny = (b: Box, others: Box[]) => others.some((o) => overlap(b, o));

/**
 * The box moved the least distance that clears every obstacle (the bar, the layer chips, the zoom buttons; on a
 * viewport change, the other panels), kept inside the area. A box too tall to fit under an obstacle may give up
 * height down to `minH` (its content scrolls). Returns the box unchanged when nothing is in the way; when no free
 * place exists it returns the nearest try, and the caller decides (`hitsAny`).
 */
export function avoid(b: Box, obstacles: Box[], area: Area, minH = 200, gap = 8): Box {
  const start = clampBox(b, area);
  let cur = start;
  const cost = (c: Box) => Math.abs(c.x - start.x) + Math.abs(c.y - start.y) + (start.h - c.h) * 0.75;
  for (let i = 0; i < 4; i++) {
    const hit = obstacles.filter((o) => overlap(cur, o));
    if (!hit.length) return cur;
    const tries: Box[] = [];
    for (const o of hit) {
      const below = o.y + o.h + gap;
      tries.push({ ...cur, y: below }, { ...cur, y: o.y - gap - cur.h }, { ...cur, x: o.x + o.w + gap }, { ...cur, x: o.x - gap - cur.w });
      // under the obstacle, shorter: a tall panel keeps its column and scrolls
      const room = area.bottom - below;
      if (room >= Math.min(minH, cur.h) && room < cur.h) tries.push({ ...cur, y: below, h: Math.round(room) });
    }
    const clamped = tries.map((t) => clampBox(t, area));
    const free = clamped.filter((t) => !hitsAny(t, obstacles));
    const pool = free.length ? free : clamped;
    pool.sort((p, q) => cost(p) - cost(q));
    cur = pool[0];
    if (free.length) return cur;
  }
  return cur;
}
