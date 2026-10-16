// Wordmark outlines: glyphs from a font, then modified (i-dots removed and redrawn, custom tracking).
import opentype from "opentype.js";
import { readFileSync } from "node:fs";
const F = (n) => opentype.parse(readFileSync(new URL(`./fonts/f${n}.ttf`, import.meta.url)).buffer.slice(0));
export const FONTS = { sora500: F(3), sora400: F(2), sora600: F(4), spectral500: F(6), spectral400: F(5), spectral600: F(7), schib500: F(9), schib400: F(8), schib600: F(10) };
function contours(cmds) { const out = []; let cur = []; for (const c of cmds) { if (c.type === "M" && cur.length) { out.push(cur); cur = []; } cur.push(c); } if (cur.length) out.push(cur); return out; }
function bbox(cs) { let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9; for (const c of cs) for (const [x, y] of [[c.x, c.y], [c.x1, c.y1], [c.x2, c.y2]]) if (x !== undefined) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); } return { x0, y0, x1, y1 }; }
const d = (cs) => cs.map((c) => c.type === "Z" ? "Z" : c.type === "Q" ? `Q${c.x1.toFixed(2)} ${c.y1.toFixed(2)} ${c.x.toFixed(2)} ${c.y.toFixed(2)}` : c.type === "C" ? `C${c.x1.toFixed(2)} ${c.y1.toFixed(2)} ${c.x2.toFixed(2)} ${c.y2.toFixed(2)} ${c.x.toFixed(2)} ${c.y.toFixed(2)}` : `${c.type}${c.x.toFixed(2)} ${c.y.toFixed(2)}`).join("");
/** text set at `size` px with `track` (em) extra spacing and optional per-pair adjustments; returns body path, dot centres, metrics */
export function wordmark(font, text, size, { track = 0, pairs = {}, dropDots = true } = {}) {
  const glyphs = font.stringToGlyphs(text); const scale = size / font.unitsPerEm; let x = 0; let body = ""; const dots = []; const boxes = [];
  glyphs.forEach((g, i) => {
    const p = g.getPath(x, 0, size); const cs = contours(p.commands);
    for (const c of cs) { const b = bbox(c); const isDot = dropDots && text[i] === "i" && b.y1 < -(font.tables.os2.sxHeight * scale) * 1.02; if (isDot) dots.push({ cx: (b.x0 + b.x1) / 2, cy: (b.y0 + b.y1) / 2, r: (b.x1 - b.x0) / 2 }); else body += d(c); }
    boxes.push(bbox(p.commands));
    let adv = g.advanceWidth * scale + track * size;
    if (i < glyphs.length - 1) { adv += font.getKerningValue(g, glyphs[i + 1]) * scale; adv += (pairs[text[i] + text[i + 1]] ?? 0) * size; }
    x += adv;
  });
  const xh = font.tables.os2.sxHeight * scale, cap = font.tables.os2.sCapHeight * scale, asc = font.ascender * scale;
  const minX = Math.min(...boxes.map((b) => b.x0)), maxX = Math.max(...boxes.map((b) => b.x1));
  return { body, dots, xh, cap, asc, minX, maxX, width: maxX - minX, boxes };
}
