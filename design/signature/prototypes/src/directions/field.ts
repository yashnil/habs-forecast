import type { Grid } from "../data/store";
import type { PaletteStop } from "../data/types";
import { km } from "../lib/geo";
import { classOf, fieldExtent, type FieldOptions } from "../map/field";

/**
 * Same exact-cell renderer as map/field.ts, but each valued cell is mixed toward a caller-chosen
 * base colour (the variant's sea), so light basemaps get light veils. Values, classes and cell
 * edges are untouched; only the drawing strength differs.
 */
const PX = 4;
const mercY = (lat: number) => Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360));
const latFromY = (y: number) => (360 / Math.PI) * Math.atan(Math.exp(y)) - 90;
const hex = (c: string) => { const n = parseInt(c.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; };

export function renderFieldOn(canvas: HTMLCanvasElement, grid: Grid, stops: PaletteStop[], opt: FieldOptions, base: [number, number, number]) {
  const s = grid.spec;
  const ext = fieldExtent(grid);
  const W = s.width * PX;
  const yN = mercY(ext.north), yS = mercY(ext.south);
  const H = Math.round(((yN - yS) / (((ext.east - ext.west) * Math.PI) / 180)) * W);
  canvas.width = W; canvas.height = H;
  const g = canvas.getContext("2d")!;
  const img = g.createImageData(W, H);
  const cols = stops.map((st) => hex(st.color));
  const mix = new Float32Array(s.width * s.height);
  for (let r = 0; r < s.height; r++) for (let c = 0; c < s.width; c++) {
    let a = opt.mode === "edges" ? 0.24 : opt.veil ?? 0.6;
    if (opt.mode === "lens" && opt.lens) {
      const d = km([opt.lens.lon, opt.lens.lat], [s.lon_first + c * s.lon_step, s.lat_first + r * s.lat_step]);
      a = d <= opt.lens.km ? 1 : opt.outside ?? 0.16;
    }
    mix[r * s.width + c] = a;
  }
  const d = img.data;
  for (let y = 0; y < H; y++) {
    const lat = latFromY(yN - ((y + 0.5) / H) * (yN - yS));
    const row = Math.round((lat - s.lat_first) / s.lat_step);
    if (row < 0 || row >= s.height) continue;
    for (let x = 0; x < W; x++) {
      const i = row * s.width + Math.floor(x / PX);
      const v = grid.values[i];
      if (Number.isNaN(v) || mix[i] === 0) continue; // outside: 0 → not drawn (Living Coast lens)
      const k = cols[classOf(v)], a = mix[i], o = (y * W + x) * 4;
      d[o] = base[0] + (k[0] - base[0]) * a; d[o + 1] = base[1] + (k[1] - base[1]) * a; d[o + 2] = base[2] + (k[2] - base[2]) * a; d[o + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
  return ext.corners;
}
