// Neritic logo directions, round 2 (bathymetric identity). Emits SVGs to ./out and the comparison page ./board.html.
// Run from coastwatch-web:  node ../docs/coastwatch/v2/logo/gen.mjs   (then node ../docs/coastwatch/v2/logo/shots.mjs)
// Fonts: OFL faces fetched into ./fonts at runtime (see README); opentype.js from coastwatch-web/node_modules (npm i --no-save opentype.js).
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import * as opentype from "../../../../coastwatch-web/node_modules/opentype.js/dist/opentype.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = join(HERE, "out");
mkdirSync(OUT, { recursive: true });
const f2 = (n) => (Math.round(n * 100) / 100).toString();

// ---------------------------------------------------------------- palette
export const PAL = {
  light: { k: "l", ink: "#0b2238", sea: "#1d7894", amber: "#e5902a", bg: "#ffffff", paper: "#ffffff" },
  dark:  { k: "d", ink: "#eef3f8", sea: "#5fc0d8", amber: "#f2ae4a", bg: "#0b2238", paper: "#0b2238" },
  mono:  { k: "m", ink: "#000000", sea: "#000000", amber: "#000000", bg: "none", mono: true },
  monoW: { k: "w", ink: "#ffffff", sea: "#ffffff", amber: "#ffffff", bg: "none", mono: true },
};
const svg48 = (body, s = 48) => `<svg xmlns="http://www.w3.org/2000/svg" width="${s}" height="${s}" viewBox="0 0 48 48">${body}</svg>`;

// ---------------------------------------------------------------- geometry helpers
const bz = (p0, p1, p2, p3, t) => { const u = 1 - t; return [0, 1].map((k) => u * u * u * p0[k] + 3 * u * u * t * p1[k] + 3 * u * t * t * p2[k] + t * t * t * p3[k]); };
function minDist(A, B) { let m = 1e9; for (const a of A) for (const b of B) { const d = Math.hypot(a[0] - b[0], a[1] - b[1]); if (d < m) m = d; } return m; }
const KAPPA = 0.5523;
const pt = (q) => `${f2(q[0])} ${f2(q[1])}`;
const unit = (a, b) => { const dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy) || 1; return [dx / L, dy / L]; };
const nrm = (d) => [d[1], -d[0]];
/** offset one path element (L: 2 points, C: 4 points) by dist along its side-A normal; cubics by Tiller–Hanson */
function offsetElem(e, dist) {
  const mv = (q, n) => [q[0] + n[0] * dist, q[1] + n[1] * dist];
  if (e.t === "L") { const n = nrm(unit(e.p[0], e.p[1])); return { t: "L", p: e.p.map((q) => mv(q, n)) }; }
  const P = e.p; const edges = [];
  for (let i = 0; i < 3; i++) { const n = nrm(unit(P[i], P[i + 1])); edges.push([mv(P[i], n), mv(P[i + 1], n)]); }
  const isect = (e1, e2) => { const [a, b] = e1, [c, d] = e2; const r = [b[0] - a[0], b[1] - a[1]], q = [d[0] - c[0], d[1] - c[1]]; const den = r[0] * q[1] - r[1] * q[0]; if (Math.abs(den) < 1e-9) return b; const t = ((c[0] - a[0]) * q[1] - (c[1] - a[1]) * q[0]) / den; return [a[0] + t * r[0], a[1] + t * r[1]]; };
  return { t: "C", p: [edges[0][0], isect(edges[0], edges[1]), isect(edges[1], edges[2]), edges[2][1]] };
}
const fwd = (els) => els.map((e) => e.t === "L" ? `L${pt(e.p[1])}` : `C${pt(e.p[1])} ${pt(e.p[2])} ${pt(e.p[3])}`).join("");
const rev = (els) => [...els].reverse().map((e) => e.t === "L" ? `L${pt(e.p[0])}` : `C${pt(e.p[2])} ${pt(e.p[1])} ${pt(e.p[0])}`).join("");
const samples = (els, n = 40) => els.flatMap((e) => e.t === "L" ? Array.from({ length: n }, (_, i) => [e.p[0][0] + (e.p[1][0] - e.p[0][0]) * i / n, e.p[0][1] + (e.p[1][1] - e.p[0][1]) * i / n]) : Array.from({ length: n }, (_, i) => bz(...e.p, i / n)));
/** filled outline of an open stroke whose first and last elements are straight lines crossing y = top and y = bot (cut flat there) */
function strokeOutline(els, sw, top = 0, bot = 48) {
  const cut = (p, q, y) => { const t = (y - p[1]) / (q[1] - p[1]); return [p[0] + t * (q[0] - p[0]), y]; };
  const A = els.map((e) => offsetElem(e, sw / 2)), B = els.map((e) => offsetElem(e, -sw / 2));
  for (const S of [A, B]) { S[0].p[0] = cut(S[0].p[0], S[0].p[1], top); const l = S[S.length - 1]; l.p[1] = cut(l.p[0], l.p[1], bot); }
  return `M${pt(A[0].p[0])}${fwd(A)}L${pt(B[B.length - 1].p[1])}${rev(B)}Z`;
}
/** Catmull-Rom chain through points → cubic elements */
function through(P, tension = 0.5) {
  const els = []; const Q = [P[0], ...P, P[P.length - 1]];
  for (let i = 1; i < Q.length - 2; i++) { const [p0, p1, p2, p3] = [Q[i - 1], Q[i], Q[i + 1], Q[i + 2]]; els.push({ t: "C", p: [p1, [p1[0] + (p2[0] - p0[0]) * tension / 3, p1[1] + (p2[1] - p0[1]) * tension / 3], [p2[0] - (p3[0] - p1[0]) * tension / 3, p2[1] - (p3[1] - p1[1]) * tension / 3], p2] }); }
  return els;
}
const roundRect = (r, S = 48) => `M0 ${r}A${r} ${r} 0 0 1 ${r} 0H${S - r}A${r} ${r} 0 0 1 ${S} ${r}V${S - r}A${r} ${r} 0 0 1 ${S - r} ${S}H${r}A${r} ${r} 0 0 1 0 ${S - r}Z`;

// ---------------------------------------------------------------- 1b · Coast-cut contours
// Plan view, north up. The east third of a tile is land, a solid field with a slightly irregular near-vertical coastline.
// Three open isobaths run parallel to the coast and all hook sharply inshore into one canyon head (Monterey Canyon):
// they crowd at the head and spread apart to the south-west. The lines are clipped flat by the tile's top and bottom
// edges, never closing. Every version is built from filled outlines, so mono and the navy knock-out are single paths.
export const COASTCUT = {
  rx: 6,
  coast: [[36.2, 0], [35.3, 9], [36.9, 24], [35.2, 39], [36.3, 48]],
  // the deep isobath: a straight north arm, a large-radius sweep shoreward (R1), a tight semicircle around the canyon head
  // (r), the sweep back, and a straight south arm. The shallower isobaths are its exact parallels at `off` (left of travel
  // = east/outside), which keeps the gap uniform where the lines crowd at the head; their south arms are then splayed to
  // the exits xb so the lines spread apart to the south-west. R1 must exceed the largest offset or the parallels degenerate.
  base: { xn: 8.6, xt: 22.6, r: 2.8, R1: 11 },
  iso: [ // deep → shallow (nearest the coast, thinnest); off = centreline offset = half widths + 2.6 u clear
    { sw: 3.5, off: 0, xb: 7.6, role: "deep" },
    { sw: 2.5, off: 5.6, xb: 16.0, role: "mid" },
    { sw: 2.0, off: 10.45, xb: 23.5, role: "shallow" },
  ],
  favIso: [{ sw: 3.4, off: 5.2, xb: 16.0, role: "mid" }],
  baseEls() {
    const { xn, xt, r, R1 } = this.base; const k = KAPPA; const yA = 24 - r - R1, yB = 24 + r + R1;
    return [
      { t: "L", p: [[xn, -2], [xn, yA]] },
      { t: "C", p: [[xn, yA], [xn, yA + k * R1], [xn + R1 - k * R1, yA + R1], [xn + R1, yA + R1]] },
      { t: "L", p: [[xn + R1, 24 - r], [xt - r, 24 - r]] },
      { t: "C", p: [[xt - r, 24 - r], [xt - r + k * r, 24 - r], [xt, 24 - k * r], [xt, 24]] },
      { t: "C", p: [[xt, 24], [xt, 24 + k * r], [xt - r + k * r, 24 + r], [xt - r, 24 + r]] },
      { t: "L", p: [[xt - r, 24 + r], [xn + R1, 24 + r]] },
      { t: "C", p: [[xn + R1, 24 + r], [xn + R1 - k * R1, 24 + r], [xn, yB - k * R1], [xn, yB]] },
      { t: "L", p: [[xn, yB], [xn, 50]] },
    ].filter((e) => e.t !== "L" || Math.hypot(e.p[1][0] - e.p[0][0], e.p[1][1] - e.p[0][1]) > 0.05);
  },
  hooks(iso = this.iso) {
    const E = this.baseEls();
    return iso.map((l) => {
      const O = E.map((e) => offsetElem(e, l.off));
      const last = O[O.length - 1]; const S0 = last.p[0]; const d = unit(S0, [l.xb, 48]); last.p[1] = [S0[0] + d[0] * ((50 - S0[1]) / d[1]), 50]; // splayed south arm
      const tipEl = O.find((e, i) => E[i].p[E[i].p.length - 1][1] === 24);
      return { ...l, els: O, xn: O[0].p[0][0], xt: tipEl.p[3][0], h: this.base.r + l.off };
    });
  },
  landPath() { const r = this.rx; const els = through(this.coast); return `M${pt(this.coast[0])}${fwd(els)}H${48 - r}A${r} ${r} 0 0 0 48 ${48 - r}V${r}A${r} ${r} 0 0 0 ${48 - r} 0Z`; },
  isoPaths(iso = this.iso) { return this.hooks(iso).map((l) => strokeOutline(l.els, l.sw)); },
  body(c, iso = this.iso) {
    const land = this.landPath(), lines = this.isoPaths(iso);
    if (c.mono) return `<path d="${land}${lines.join("")}" fill="${c.ink}"/>`;
    return `<path d="${land}" fill="${c.ink}"/>\n<path d="${lines.join("")}" fill="${c.sea}"/>`;
  },
  /** navy tile with the land and the isobaths knocked out (one even-odd path) */
  tileBody(c, iso = this.iso) { return `<path d="${roundRect(this.rx)}${this.landPath()}${this.isoPaths(iso).join("")}" fill="${c.mono ? c.ink : c.k === "d" ? c.sea : c.ink}" fill-rule="evenodd"/>`; },
  /** 16 px: the coast and one contour; knocked out of the tile on light grounds, plain on dark */
  faviconBody(c) { return c.k === "d" || c.k === "w" ? this.body(c, this.favIso) : this.tileBody(c, this.favIso); },
  check() {
    const L = this.hooks(); const H = L.map((l) => samples(l.els, 80).filter((q) => q[1] >= 0 && q[1] <= 48)); const coast = samples(through(this.coast), 60); const out = [];
    const d0 = minDist(H[2], coast) - L[2].sw / 2; out.push(`shallow line to coast: clear ${d0.toFixed(2)} u`);
    for (let i = 0; i < H.length - 1; i++) { const d = minDist(H[i], H[i + 1]) - L[i].sw / 2 - L[i + 1].sw / 2; out.push(`gap ${L[i].role}-${L[i + 1].role}: clear ${d.toFixed(2)} u (${(d / 1.5).toFixed(2)} px @32)`); }
    out.push(`north arms x ${L.map((l) => l.xn.toFixed(1)).join(" / ")}; head tips ${L.map((l) => l.xt.toFixed(1)).join(" / ")}; hairpin radii ${L.map((l) => l.h.toFixed(1)).join(" / ")}; bottom exits ${L.map((l) => l.xb).join(" / ")} u`);
    return out;
  },
  crop: [0, 0, 48, 48],
};

// ---------------------------------------------------------------- 2 · Shelf (negative space)  and  2b · Shelf with isobath
// Section view, coast to the east. A solid disc cut once by the seafloor profile: shelf (shallow, flat, right), the break,
// the slope and the deep floor (left). Above the cut is the water, below it the ground. Lines and circular fillets, so
// the gap is a true offset of constant width and both pieces are closed filled paths. 2b adds one thin isobath (a 1.2 u
// knock-out line) parallel to the cut on the water side, 4 u above the gap.
function shelfPieces({ r = 21, cx = 24, cy = 24, yd = 31.2, ys = 17.4, xs0 = 15.6, xs1 = 23.6, gap = 4.0, filA = 5.4, filB = 2.6, iso = null }) {
  const g = gap / 2;
  const ang = Math.atan2(yd - ys, xs1 - xs0); const t = Math.tan(ang / 2);
  function edge(o) { // o = +g for the ground edge (below the centreline), negative values for edges on the water side
    const y1 = yd + o, y2 = ys + o;
    const nx = Math.sin(ang), ny = Math.cos(ang);
    const sx = xs0 + o * nx, sy = yd + o * ny, dx = Math.cos(ang), dy = -Math.sin(ang);
    const xa = sx + ((y1 - sy) / dy) * dx, xb = sx + ((y2 - sy) / dy) * dx;
    const RA = Math.max(filA + o, 0.05), RB = filB - o; const la = RA * t, lb = RB * t;
    return { d: `H${f2(xa - la)}A${f2(RA)} ${f2(RA)} 0 0 0 ${f2(xa + la * dx)} ${f2(y1 + la * dy)}L${f2(xb - lb * dx)} ${f2(y2 - lb * dy)}A${f2(RB)} ${f2(RB)} 0 0 1 ${f2(xb + lb)} ${f2(y2)}`, y1, y2 };
  }
  const cxAt = (y) => Math.sqrt(r * r - (y - cy) * (y - cy));
  const span = (yR, yL, over) => { const aR = Math.atan2(yR - cy, cxAt(yR)), aL = Math.atan2(yL - cy, -cxAt(yL)); let d = over ? aR - aL : aL - aR; d = ((d % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI); return d > Math.PI ? 1 : 0; };
  const above = (o) => { const e = edge(o); return `M${f2(cx - cxAt(e.y1))} ${f2(e.y1)}${e.d}H${f2(cx + cxAt(e.y2))}A${r} ${r} 0 ${span(e.y2, e.y1, true)} 0 ${f2(cx - cxAt(e.y1))} ${f2(e.y1)}Z`; };
  const lo = edge(g);
  const ground = `M${f2(cx - cxAt(lo.y1))} ${f2(lo.y1)}${lo.d}H${f2(cx + cxAt(lo.y2))}A${r} ${r} 0 ${span(lo.y2, lo.y1, false)} 1 ${f2(cx - cxAt(lo.y1))} ${f2(lo.y1)}Z`;
  const out = { water: above(-g), ground, ang: ang * 180 / Math.PI };
  if (iso) { const o1 = -g - iso.off + iso.w / 2, o2 = -g - iso.off - iso.w / 2; out.strip = above(-g) + above(o1); out.upper = above(o2); out.isoEdges = [o1, o2]; }
  return out;
}
export const SHELF = {
  params: { r: 21, cx: 24, cy: 24, yd: 31.2, ys: 17.4, xs0: 15.6, xs1: 23.6, gap: 4.0, filA: 5.4, filB: 2.6 },
  pieces() { return shelfPieces(this.params); },
  body(c) {
    const { water, ground } = this.pieces();
    if (c.mono) return `<path d="${water}${ground}" fill="${c.ink}"/>`;
    return `<path d="${water}" fill="${c.sea}"/>\n<path d="${ground}" fill="${c.ink}"/>`;
  },
  crop: [3, 3, 42, 42],
};
export const SHELF_ISO = {
  params: { ...SHELF.params, filA: 7.0, iso: { off: 4.0, w: 1.4 } }, // foot fillet opened so the isobath's offset of it keeps a radius
  pieces() { return shelfPieces(this.params); },
  body(c) {
    const { water, ground, strip, upper } = this.pieces();
    if (c.mono) return `<path d="${ground}${strip}${upper}" fill="${c.ink}" fill-rule="evenodd"/>`; // one path: the isobath is a hole
    return `<path d="${strip}" fill="${c.sea}" fill-rule="evenodd"/>\n<path d="${upper}" fill="${c.sea}"/>\n<path d="${ground}" fill="${c.ink}"/>`;
  },
  faviconBody(c) { return SHELF.body(c); }, // at 16 px the isobath is dropped
  crop: [3, 3, 42, 42],
};

// ---------------------------------------------------------------- type
const fontCache = {};
const font = (name) => (fontCache[name] ??= opentype.parse(readFileSync(join(HERE, "fonts", name + ".ttf")).buffer.slice(0)));
function contoursOf(cmds) { const out = []; let cur = []; for (const c of cmds) { if (c.type === "M" && cur.length) { out.push(cur); cur = []; } cur.push(c); } if (cur.length) out.push(cur); return out; }
function bbox(cs) { let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9; for (const c of cs) for (const [x, y] of [[c.x, c.y], [c.x1, c.y1], [c.x2, c.y2]]) if (x !== undefined) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); } return { x0, y0, x1, y1 }; }
const pathD = (cs) => cs.map((c) => c.type === "Z" ? "Z" : c.type === "Q" ? `Q${f2(c.x1)} ${f2(c.y1)} ${f2(c.x)} ${f2(c.y)}` : c.type === "C" ? `C${f2(c.x1)} ${f2(c.y1)} ${f2(c.x2)} ${f2(c.y2)} ${f2(c.x)} ${f2(c.y)}` : `${c.type}${f2(c.x)} ${f2(c.y)}`).join("");

/** Hand-built N (y up, baseline 0): stems of width `stem`, diagonal of perpendicular thickness tN·stem with flat cuts at
 *  cap and baseline, and the step: the diagonal's upper edge leaves the left stem `ledge` below the cap, mirrored at the
 *  foot of the right stem, so each join has a short vertical wall instead of a dark wedge. */
export function shelfN({ cap, stem }, { widthRatio = 1.0, tN = 1.02, ledge = 0 } = {}) {
  const W = cap * widthRatio, s = stem, inner = W - 2 * s;
  const want = tN * s; let y1 = cap * 0.4;
  for (let it = 0; it < 80; it++) { const theta = Math.atan2(cap - ledge - y1, inner); const hw = (y1 - ledge) * inner / (cap - ledge - y1); y1 += (want - hw * Math.sin(theta)) * 0.3; }
  const y2 = cap - y1;
  const pts = [[0, 0], [s, 0], [s, y2], [W - s, ledge], [W - s, 0], [W, 0], [W, cap], [W - s, cap], [W - s, y1], [s, cap - ledge], [s, cap], [0, cap]];
  return { pts, W, d: "M" + pts.map(([x, y]) => `${f2(x)} ${f2(-y)}`).join("L") + "Z" };
}

/** Outline `text` at `size` px (baseline 0, y up negative). i-dots are removed and returned as circles. */
function setText(fnt, text, size, { track = 0, pairs = {}, customN, dotScale = 1, dotLift = 0 } = {}) {
  const glyphs = [...text].map((ch) => fnt.charToGlyph(ch)); const sc = size / fnt.unitsPerEm; let x = 0; let body = ""; const dots = []; const boxes = [];
  const xh = fnt.tables.os2.sxHeight * sc, cap = fnt.tables.os2.sCapHeight * sc;
  const iCs = contoursOf(fnt.charToGlyph("i").getPath(0, 0, size).commands).map((c) => ({ c, b: bbox(c) })).sort((a, b) => (b.b.y1 - b.b.y0) - (a.b.y1 - a.b.y0));
  const stem = iCs[0].b.x1 - iCs[0].b.x0;
  glyphs.forEach((g, i) => {
    const ch = text[i]; let adv = g.advanceWidth * sc;
    if (ch === "N" && customN) {
      const lsb = g.leftSideBearing * sc; const n = customN({ cap, stem, xh, size });
      body += "M" + n.pts.map(([px, py]) => `${f2(px + x + lsb)} ${f2(-py)}`).join("L") + "Z";
      boxes.push({ x0: x + lsb, x1: x + lsb + n.W, y0: -cap, y1: 0 }); adv = n.W + 2 * lsb;
    } else {
      const p = g.getPath(x, 0, size);
      for (const c of contoursOf(p.commands)) { const b = bbox(c); if (ch === "i" && b.y1 < -xh * 1.02) dots.push({ cx: (b.x0 + b.x1) / 2, cy: (b.y0 + b.y1) / 2 - dotLift * size, r: ((b.x1 - b.x0) / 2) * dotScale }); else body += pathD(c); }
      boxes.push(bbox(p.commands));
    }
    adv += track * size;
    if (i < glyphs.length - 1) { if (!(ch === "N" && customN)) adv += fnt.getKerningValue(g, glyphs[i + 1]) * sc; adv += (pairs[ch + text[i + 1]] ?? 0) * size; }
    x += adv;
  });
  const minX = Math.min(...boxes.map((b) => b.x0)), maxX = Math.max(...boxes.map((b) => b.x1));
  return { body, dots, xh, cap, stem, minX, maxX, width: maxX - minX };
}

const WM = {
  "1b": { font: "inter-600", label: "Inter 600, tracked −1.5 %; i-dots redrawn as circles", track: -0.015, pairs: { Ne: -0.004, ri: -0.004, ti: -0.002, ic: -0.004 }, dotScale: 1.1, dotLift: 0.01 },
  "2b": { font: "sourceserif4-600", label: "Source Serif 4 600, as set; i-dots redrawn as circles", track: 0.002, pairs: { Ne: -0.006 }, dotScale: 1.06 },
  1: { font: "inter-600", label: "Inter 600, tracked −1.5 %; i-dots redrawn as circles", track: -0.015, pairs: { Ne: -0.004, ri: -0.004, ti: -0.002, ic: -0.004 }, dotScale: 1.1, dotLift: 0.01 },
  2: { font: "sourceserif4-600", label: "Source Serif 4 600, as set; i-dots redrawn as circles", track: 0.002, pairs: { Ne: -0.006 }, dotScale: 1.06 },
  3: { font: "sora-600", label: "Sora 600, hand-cut N (stepped joins), circular i-dots 1.2 × stem", track: -0.02, pairs: { Ne: -0.008, er: -0.002, ri: -0.004, ti: 0, ic: -0.006 }, dotScale: 1.2, dotLift: 0.012, customN: (m) => shelfN(m, { widthRatio: 1.0, tN: 1.02, ledge: m.cap * 0.1 }) },
  R: { font: "sora-500", label: "Sora 500 (reference)", track: -0.012, pairs: { Ne: -0.006 }, dotScale: 1.04 },
};
const TEXT = "Neritic", U = 100;
const setWM = (k) => setText(font(WM[k].font), TEXT, U, WM[k]);
const dotsSvg = (o, c) => o.dots.map((d) => `<circle cx="${f2(d.cx)}" cy="${f2(d.cy)}" r="${f2(d.r)}" fill="${c.ink}"/>`).join("");

/** standalone wordmark; h = x-height in px. extra(o, c) may append elements in wordmark units. */
function wordSvg(k, c, h, extra) {
  const o = setWM(k); const s = h / o.xh; const pad = 2; const top = o.cap * 1.12; const below = extra ? o.xh * 0.3 + o.xh * 0.13 + 3 : 3; const vbH = top + below;
  const padR = pad;
  return `<svg xmlns="http://www.w3.org/2000/svg" height="${f2(vbH * s)}" width="${f2((o.width + pad + padR) * s)}" viewBox="${f2(o.minX - pad)} ${f2(-top)} ${f2(o.width + pad + padR)} ${f2(vbH)}"><path d="${o.body}" fill="${c.ink}"/>${dotsSvg(o, c)}${extra ? extra(o, c) : ""}</svg>`;
}

/** lockup: symbol (cropped to its ink box) at K x-heights, its bottom on the baseline, gap in x-heights */
function lockupSvg(k, c, xh, { body, crop, K, gap, lift = 0 }) {
  const o = setWM(k); const s = xh / o.xh; const [cx0, cy0, cw, ch] = crop;
  const symH = xh * K; const symW = symH * cw / ch; const g = xh * gap;
  const top = o.cap * 1.12 * s; const H = top + 3 * s; const W = symW + g + (o.width + 4) * s;
  const symY = top - symH - lift * xh;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${f2(W)}" height="${f2(H)}" viewBox="0 0 ${f2(W)} ${f2(H)}">` +
    `<svg x="0" y="${f2(symY)}" width="${f2(symW)}" height="${f2(symH)}" viewBox="${cx0} ${cy0} ${cw} ${ch}">${body(c)}</svg>` +
    `<g transform="translate(${f2(symW + g - (o.minX - 2) * s)} ${f2(top)}) scale(${f2(s)})"><path d="${o.body}" fill="${c.ink}"/>${dotsSvg(o, c)}</g></svg>`;
}

// ---------------------------------------------------------------- 3 · Wordmark-led: N monogram tile + tick candidates
export const MONO_N = {
  cap: 24, rx: 11,
  n() { const o = setWM(3); return shelfN({ cap: this.cap, stem: this.cap * o.stem / o.cap }, { widthRatio: 1.0, tN: 1.02, ledge: this.cap * 0.1 }); },
  nPath() { const n = this.n(); const ox = 24 - n.W / 2, oy = 24 + this.cap / 2; return "M" + n.pts.map(([x, y]) => `${f2(x + ox)} ${f2(oy - y)}`).join("L") + "Z"; },
  tile() { const r = this.rx; return `M0 ${r}A${r} ${r} 0 0 1 ${r} 0H${48 - r}A${r} ${r} 0 0 1 48 ${r}V${48 - r}A${r} ${r} 0 0 1 ${48 - r} 48H${r}A${r} ${r} 0 0 1 0 ${48 - r}Z`; },
  body(c) { const fill = c.mono ? c.ink : c.k === "d" ? c.sea : c.ink; return `<path d="${this.tile()}${this.nPath()}" fill="${fill}" fill-rule="evenodd"/>`; }, // N knocked out
  crop: [0, 0, 48, 48],
};
// tick candidates (wordmark units: cap ≈ 73, xh ≈ 53.4, stem ≈ 13.9 per 100 u em)
const TICKS = {
  c: { label: "c · the contour tick: one rule under the word with a single dip, the canyon, under the first i", kind: "under" },
};
function tickAfter(o, c) { // closed loop: the innermost canyon-head contour, after the word, centred on the x-height
  const x0 = o.maxX + 0.42 * o.xh, h = o.xh * 0.62, w = h * 1.55, cy = -o.xh / 2; const sw = o.stem * 0.72;
  const d = `M${f2(x0)} ${f2(cy)}C${f2(x0)} ${f2(cy - h / 2)} ${f2(x0 + w * 0.3)} ${f2(cy - h / 2)} ${f2(x0 + w * 0.62)} ${f2(cy - h / 2)}C${f2(x0 + w * 0.88)} ${f2(cy - h / 2)} ${f2(x0 + w)} ${f2(cy - h * 0.16)} ${f2(x0 + w)} ${f2(cy)}C${f2(x0 + w)} ${f2(cy + h * 0.16)} ${f2(x0 + w * 0.88)} ${f2(cy + h / 2)} ${f2(x0 + w * 0.62)} ${f2(cy + h / 2)}C${f2(x0 + w * 0.3)} ${f2(cy + h / 2)} ${f2(x0)} ${f2(cy + h / 2)} ${f2(x0)} ${f2(cy)}Z`;
  return `<path d="${d}" fill="none" stroke="${c.ink}" stroke-width="${f2(sw)}"/>`;
}
function tickUnder(o, c) { // one rule under the word with a single dip: the canyon, under the first i
  const y = o.xh * 0.3, sw = o.stem * 0.4, xi = o.dots[0]?.cx ?? o.minX + o.width * 0.5, dep = o.xh * 0.13, hw = o.xh * 0.22;
  const d = `M${f2(o.minX)} ${f2(y)}H${f2(xi - hw)}C${f2(xi - hw * 0.45)} ${f2(y)} ${f2(xi - hw * 0.45)} ${f2(y + dep)} ${f2(xi)} ${f2(y + dep)}C${f2(xi + hw * 0.45)} ${f2(y + dep)} ${f2(xi + hw * 0.45)} ${f2(y)} ${f2(xi + hw)} ${f2(y)}H${f2(o.maxX)}`;
  return `<path d="${d}" fill="none" stroke="${c.ink}" stroke-width="${f2(sw)}" stroke-linecap="round"/>`;
}
function tickLockup(t, c, xh) {
  if (t.kind === "before") return lockupSvg(3, c, xh, { body: t.body, crop: t.crop, K: t.K, gap: t.gap });
  return wordSvg(3, c, xh, t.kind === "after" ? tickAfter : tickUnder);
}

// ---------------------------------------------------------------- reference: rejected Depth N
export const DEPTHN = {
  body(c) { return `<defs><clipPath id="bn${c.k}"><rect x="0" y="8.6" width="48" height="30.9"/></clipPath></defs><path clip-path="url(#bn${c.k})" d="M10.6 44V8.6L22.6 12.9C25.9 14.1 27.6 16 28.9 19.2L37.4 39.5V4" stroke="${c.ink}" stroke-width="5.2" stroke-linejoin="miter" stroke-miterlimit="12" fill="none"/><circle cx="27.4" cy="7.2" r="3" fill="${c.amber}"/>`; },
  crop: [8, 4.2, 32, 35.3],
};

// ---------------------------------------------------------------- directions
export const DIRS = {
  "1b": { name: "Coast-cut contours", sub: "land as a solid field east of an irregular coast; three open isobaths hook inshore into one canyon head, clipped by the tile", geo: COASTCUT, K: 1.5, gap: 0.5, iconFull: true },
  2: { name: "Shelf", sub: "a disc cut once by the shelf profile: water above, ground below", geo: SHELF, K: 1.9, gap: 0.5, iconBox: 300 },
  "2b": { name: "Shelf with isobath", sub: "the Shelf disc with one 1.4 u isobath knocked out 4 u above the cut; dropped at 16 px", geo: SHELF_ISO, K: 1.9, gap: 0.5, iconBox: 300 },
  3: { name: "Wordmark-led", sub: "Sora with a hand-cut N; the N tile as favicon; a contour tick only in large lockups", geo: MONO_N, K: 1.37, gap: 0.5, wordOnlyHeader: true, iconInset: true },
  R: { name: "Depth N (rejected, reference)", sub: "previous recommendation, for comparison only", geo: DEPTHN, K: 1.62, gap: 0.6, iconBox: 300 },
};
for (const d of Object.values(DIRS)) { d.body = (c) => d.geo.body(c); d.crop = d.geo.crop; }
const ORDER = ["1b", "2", "2b", "3", "R"]; // board and file order (object keys would sort the numeric ones first)
const ENTRIES = ORDER.map((k) => [k, DIRS[k]]);

/** app icon 512, maskable: full-bleed navy, symbol inside the safe zone (inner 80 % circle) */
function appIcon(dir) {
  const c = PAL.dark; const S = 512;
  if (dir.iconInset) return `<svg xmlns="http://www.w3.org/2000/svg" width="${S}" height="${S}" viewBox="0 0 48 48"><rect width="48" height="48" fill="${c.bg}"/><svg x="8.5" y="8.5" width="31" height="31" viewBox="0 0 48 48"><path d="${MONO_N.nPath()}" fill="${c.ink}"/></svg></svg>`;
  if (dir.iconFull) return `<svg xmlns="http://www.w3.org/2000/svg" width="${S}" height="${S}" viewBox="0 0 48 48"><rect width="48" height="48" fill="${c.bg}"/>${dir.body(c)}</svg>`; // the tile is the icon: land and lines full-bleed
  const box = dir.iconBox; const off = (S - box) / 2;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${S}" height="${S}" viewBox="0 0 ${S} ${S}"><rect width="${S}" height="${S}" fill="${c.bg}"/><svg x="${off}" y="${off}" width="${box}" height="${box}" viewBox="0 0 48 48">${dir.body(c)}</svg></svg>`;
}
const lockup = (k, c, xh, header = false) => {
  const d = DIRS[k];
  if (d.wordOnlyHeader && header) return wordSvg(k, c, xh);
  return lockupSvg(k, c, xh, { body: header && d.headerBody ? d.headerBody : d.body, crop: d.crop, K: d.K, gap: d.gap });
};

// ---------------------------------------------------------------- emit files
const files = [];
const put = (name, s) => { writeFileSync(join(OUT, name), s); files.push(name); };
for (const [k, dir] of ENTRIES) {
  put(`${k}-symbol.svg`, svg48(dir.body(PAL.light)));
  put(`${k}-symbol-mono.svg`, svg48(dir.body(PAL.mono)));
  put(`${k}-symbol-dark.svg`, svg48(dir.body(PAL.dark)));
  put(`${k}-wordmark.svg`, lockup(k, PAL.light, 40));
  put(`${k}-wordmark-dark.svg`, lockup(k, PAL.dark, 40));
  put(`${k}-wordmark-only.svg`, wordSvg(k, PAL.light, 40));
  put(`${k}-icon-512.svg`, appIcon(dir));
  if (dir.geo.faviconBody) { put(`${k}-favicon.svg`, svg48(dir.geo.faviconBody(PAL.light))); put(`${k}-favicon-dark.svg`, svg48(dir.geo.faviconBody(PAL.dark))); }
  if (dir.geo.tileBody) { put(`${k}-symbol-tile.svg`, svg48(dir.geo.tileBody(PAL.light))); put(`${k}-symbol-tile-mono.svg`, svg48(dir.geo.tileBody(PAL.mono))); }
}
put("3-wordmark-tick.svg", tickLockup(TICKS.c, PAL.light, 40)); put("3-wordmark-tick-dark.svg", tickLockup(TICKS.c, PAL.dark, 40));
console.log("coast-cut:\n  " + COASTCUT.check().join("\n  "));
console.log(`shelf: slope ${SHELF.pieces().ang.toFixed(1)}°, gap ${SHELF.params.gap} u, fillets foot ${SHELF.params.filA} / break ${SHELF.params.filB} u; 2b foot fillet ${SHELF_ISO.params.filA}, isobath edges at ${SHELF_ISO.pieces().isoEdges.map((v) => v.toFixed(1)).join(" / ")} u from the centreline`);
{ const o = setWM(3); const n = MONO_N.n(); console.log(`wordmark 3: stem ${o.stem.toFixed(1)} cap ${o.cap.toFixed(1)} xh ${o.xh.toFixed(1)} width ${o.width.toFixed(0)} (per 100 u em); tile N: cap ${MONO_N.cap} stem ${(MONO_N.cap * o.stem / o.cap).toFixed(2)} width ${n.W.toFixed(2)}`); }
console.log("wrote", files.length, "files to out/ and board.html");

// ---------------------------------------------------------------- board
const sym = (dir, c, s) => svg48(dir.body(c), s);
const fav = (dir, c, s) => dir.geo.faviconBody && s <= 16 ? svg48(dir.geo.faviconBody(c), s) : sym(dir, c, s);
const css = `*{box-sizing:border-box}body{margin:0;background:#e9ebee;font:13px/1.4 system-ui,-apple-system,sans-serif;color:#33414f;padding:24px}
h1{font:600 18px system-ui;margin:0 0 14px;color:#0b2238}
.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:14px}
.col{background:#fff;border:1px solid #dfe3e8;display:flex;flex-direction:column}
.col h2{font:600 15px system-ui;margin:0;padding:12px 14px 2px;color:#0b2238}.col p{margin:0;padding:0 14px 10px;color:#5d6b79;font-size:12px;min-height:62px}
.cell{border-top:1px solid #e6e8eb;min-height:120px;display:grid;place-items:center;position:relative;padding:12px}
.cell i{position:absolute;left:8px;bottom:5px;font:10px/1 ui-monospace,monospace;color:#7b8794;font-style:normal}
.dark{background:#0b2238}.dark i{color:#7f93ab}.navy{background:#0b2238}.grey{background:#e9ebee}
.mast{height:56px;width:100%;display:flex;align-items:center;padding:0 16px;gap:14px}.mast .nav{margin-left:auto;font-size:12px}
.sizes{display:flex;gap:14px;align-items:flex-end}.px{image-rendering:pixelated}
.row{display:flex;gap:16px;align-items:center;justify-content:center}
.icon{width:96px;height:96px;border-radius:22px;overflow:hidden;box-shadow:0 0 0 1px #e3e6ea}.icon svg{width:100%;height:100%;display:block}
.circ{border-radius:50%}`;
let html = `<!doctype html><meta charset="utf-8"><title>Neritic logo board</title><style>${css}</style><body><h1>Neritic · logo directions, round 2 (bathymetric), iteration 5</h1><div class="grid">`;
for (const [k, dir] of ENTRIES) {
  const tile = dir.geo.tileBody ? (c) => dir.geo.tileBody(c) : null;
  html += `<section class="col" id="d${k}"><h2>${k === "R" ? "" : k + " · "}${dir.name}</h2><p>${dir.sub}</p>
  <div class="cell" style="min-height:170px">${sym(dir, PAL.light, 128)}<i>symbol 128</i></div>
  <div class="cell dark" style="min-height:170px">${sym(dir, PAL.dark, 128)}<i>symbol 128, dark</i></div>
  ${tile ? `<div class="cell grey"><div class="row">${svg48(tile(PAL.light), 96)}${svg48(tile(PAL.light), 32)}<span class="px" style="line-height:0">${svg48(dir.geo.faviconBody(PAL.light), 16)}</span></div><i>knock-out from a navy tile · 96 / 32 / 16 (favicon cut)</i></div>` : ""}
  <div class="cell"><div class="row">${sym(dir, PAL.mono, 72)}<div class="navy" style="padding:10px;border-radius:6px">${sym(dir, PAL.monoW, 52)}</div></div><i>monochrome</i></div>
  <div class="cell"><div class="sizes"><span class="px" style="display:inline-block;line-height:0">${fav(dir, PAL.light, 16)}</span><span style="line-height:0">${sym(dir, PAL.light, 32)}</span><span style="line-height:0">${sym(dir, PAL.light, 48)}</span><span class="px" style="display:inline-block;line-height:0;transform:scale(4);transform-origin:bottom left;margin:0 54px 0 6px">${fav(dir, PAL.light, 16)}</span></div><i>16 · 32 · 48 · 16 at 4×</i></div>
  <div class="cell dark"><div class="sizes"><span class="px" style="display:inline-block;line-height:0">${fav(dir, PAL.dark, 16)}</span><span style="line-height:0">${sym(dir, PAL.dark, 32)}</span><span style="line-height:0">${sym(dir, PAL.dark, 48)}</span><span style="display:inline-flex;background:#202124;padding:6px 10px;border-radius:8px 8px 0 0;gap:7px;align-items:center;color:#e8eaed;font-size:11px;line-height:0">${fav(dir, PAL.dark, 16)}<span style="line-height:1">Neritic</span></span></div><i>16 · 32 · 48 dark · tab</i></div>
  <div class="cell" style="padding:0;min-height:0"><div class="mast" style="background:#fff">${lockup(k, PAL.light, 9.6, true)}<span class="nav" style="color:#5d6b79">My Coast&nbsp;&nbsp;Map&nbsp;&nbsp;Blooms</span></div></div>
  <div class="cell dark" style="padding:0;min-height:0"><div class="mast">${lockup(k, PAL.dark, 9.6, true)}<span class="nav" style="color:#a9b8c9">My Coast&nbsp;&nbsp;Map&nbsp;&nbsp;Blooms</span></div></div>
  <div class="cell" style="min-height:110px">${lockup(k, PAL.light, 20)}<i>lockup · ${WM[k].label}</i></div>
  <div class="cell dark" style="min-height:110px">${lockup(k, PAL.dark, 20)}<i>lockup, dark</i></div>
  ${k === "3" ? `<div class="cell" style="min-height:110px">${tickLockup(TICKS.c, PAL.light, 20)}<i>${TICKS.c.label}</i></div>` : ""}
  <div class="cell"><div class="row"><div class="icon">${appIcon(dir)}</div><div class="icon circ">${appIcon(dir)}</div></div><i>app icon 512 maskable (square · circle mask)</i></div>
  </section>`;
}
html += `</div></body>`;
writeFileSync(join(HERE, "board.html"), html);
