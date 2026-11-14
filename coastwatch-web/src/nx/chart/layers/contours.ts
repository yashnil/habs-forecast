/**
 * Isotherms traced on the CPU (marching squares) for a model temperature grid, as GeoJSON lines for the chart.
 * PURE. A cell with any corner without a value (land, the near-shore band, beyond the grid) emits nothing, so no
 * segment ever runs along a mask or a grid edge; lines shorter than `minKm` (tiny closed rings, specks) are dropped.
 */
import type { FieldGrid } from "./grid";

export type Isoline = { level: number; major: boolean; coords: [number, number][]; km: number };

const KM_PER_DEG = 111.32;
const lenKm = (c: [number, number][]) => {
  let s = 0;
  for (let i = 1; i < c.length; i++) {
    const [x0, y0] = c[i - 1], [x1, y1] = c[i];
    const k = Math.cos((((y0 + y1) / 2) * Math.PI) / 180);
    s += Math.hypot((x1 - x0) * k, y1 - y0) * KM_PER_DEG;
  }
  return s;
};

/**
 * Lines of equal value every `step` (in the units `toValue` returns, e.g. °F), heavier every `index` steps.
 * `toValue(code)` decodes a published code; `nodata` codes are missing.
 */
export function traceIsolines(g: FieldGrid, codes: Uint16Array, o: { step: number; index: number; toValue: (code: number) => number; minKm?: number }): Isoline[] {
  const { width: W, height: H, nodata } = g;
  const val = new Float64Array(W * H);
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < W * H; i++) {
    if (codes[i] === nodata) {
      val[i] = NaN;
      continue;
    }
    const v = o.toValue(codes[i]);
    val[i] = v;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo)) return [];
  const lonOf = (c: number) => g.lon_first + c * g.lon_step;
  const latOf = (r: number) => g.lat_first + r * g.lat_step;
  const out: Isoline[] = [];
  for (let level = Math.ceil(lo / o.step) * o.step; level <= hi; level += o.step) {
    // segments of this level, each end keyed for stitching
    const segs: Array<[[number, number], [number, number]]> = [];
    for (let r = 0; r + 1 < H; r++) {
      for (let c = 0; c + 1 < W; c++) {
        const a = val[r * W + c], b = val[r * W + c + 1], d = val[(r + 1) * W + c], e = val[(r + 1) * W + c + 1];
        // a missing corner: no segment at all (never along a mask or the grid's edge)
        if (Number.isNaN(a) || Number.isNaN(b) || Number.isNaN(d) || Number.isNaN(e)) continue;
        const idx = (a >= level ? 8 : 0) | (b >= level ? 4 : 0) | (e >= level ? 2 : 0) | (d >= level ? 1 : 0);
        if (idx === 0 || idx === 15) continue;
        const t = (p: number, q: number) => (p === q ? 0.5 : (level - p) / (q - p));
        const top: [number, number] = [lonOf(c + t(a, b)), latOf(r)];
        const right: [number, number] = [lonOf(c + 1), latOf(r + t(b, e))];
        const bottom: [number, number] = [lonOf(c + t(d, e)), latOf(r + 1)];
        const left: [number, number] = [lonOf(c), latOf(r + t(a, d))];
        switch (idx) {
          case 1: case 14: segs.push([left, bottom]); break;
          case 2: case 13: segs.push([bottom, right]); break;
          case 3: case 12: segs.push([left, right]); break;
          case 4: case 11: segs.push([top, right]); break;
          case 6: case 9: segs.push([top, bottom]); break;
          case 7: case 8: segs.push([left, top]); break;
          case 5: segs.push([left, top], [bottom, right]); break;
          case 10: segs.push([left, bottom], [top, right]); break;
        }
      }
    }
    for (const line of stitch(segs)) {
      const km = lenKm(line);
      // a closed ring (an eddy speck on a 4 km grid) needs more length than an open line to be kept
      const closed = line.length > 3 && key(line[0]) === key(line[line.length - 1]);
      if (km < (o.minKm ?? 0) * (closed ? 2.5 : 1)) continue;
      out.push({ level, major: Math.round(level / o.step) % o.index === 0, coords: line, km });
    }
  }
  return out;
}

const key = (p: [number, number]) => `${p[0].toFixed(6)},${p[1].toFixed(6)}`;

/** Join segments that share ends into polylines. */
function stitch(segs: Array<[[number, number], [number, number]]>): [number, number][][] {
  const ends = new Map<string, number[]>();
  segs.forEach(([p, q], i) => {
    for (const k of [key(p), key(q)]) (ends.get(k) ?? ends.set(k, []).get(k)!).push(i);
  });
  const used = new Uint8Array(segs.length);
  const lines: [number, number][][] = [];
  const grow = (line: [number, number][], forward: boolean) => {
    for (;;) {
      const tip = forward ? line[line.length - 1] : line[0];
      const next = (ends.get(key(tip)) ?? []).find((j) => !used[j]);
      if (next === undefined) return;
      used[next] = 1;
      const [p, q] = segs[next];
      const other = key(p) === key(tip) ? q : p;
      if (forward) line.push(other);
      else line.unshift(other);
    }
  };
  for (let i = 0; i < segs.length; i++) {
    if (used[i]) continue;
    used[i] = 1;
    const line: [number, number][] = [segs[i][0], segs[i][1]];
    grow(line, true);
    grow(line, false);
    lines.push(line);
  }
  return lines;
}

/**
 * Label points along the major lines: up to `perLevel` per level, one every `spacingKm` along its longest lines,
 * taken in turn across the levels (so every level is named, not one level many times), at most `max`. The chart's
 * collision pass keeps those with room.
 */
export function isolineLabels(lines: Isoline[], max = 24, spacingKm = 45, perLevel = 4): Array<{ id: string; lon: number; lat: number; text: string }> {
  const byLevel = new Map<number, Array<{ id: string; lon: number; lat: number; text: string }>>();
  for (const l of [...lines].filter((x) => x.major).sort((a, b) => b.km - a.km)) {
    const list = byLevel.get(l.level) ?? byLevel.set(l.level, []).get(l.level)!;
    let walked = 0, next = Math.min(spacingKm / 3, l.km / 2), n = 0;
    for (let i = 1; i < l.coords.length && list.length < perLevel; i++) {
      const [x0, y0] = l.coords[i - 1], [x1, y1] = l.coords[i];
      walked += Math.hypot((x1 - x0) * Math.cos((((y0 + y1) / 2) * Math.PI) / 180), y1 - y0) * KM_PER_DEG;
      if (walked >= next) {
        list.push({ id: `isot-${l.level}-${Math.round(x1 * 100)}-${Math.round(y1 * 100)}-${n++}`, lon: x1, lat: y1, text: `${Math.round(l.level)}°` });
        next += spacingKm;
      }
    }
  }
  const out: Array<{ id: string; lon: number; lat: number; text: string }> = [];
  for (let k = 0; out.length < max; k++) {
    let any = false;
    for (const list of byLevel.values()) if (list[k] && out.length < max) {
      out.push(list[k]);
      any = true;
    }
    if (!any) break;
  }
  return out;
}
