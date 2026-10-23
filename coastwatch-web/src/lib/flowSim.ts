import type { CurrentField } from "@/lib/currents";
import { SPEED_CLASSES } from "@/lib/basemap";

/**
 * The particle simulation behind the animated currents (components/map/FlowLayer draws it). Pure, so
 * the honesty rules are tested without a GPU:
 * - ONE observed hour, frozen: the field never changes while particles move (a new hour is a cut).
 * - Nearest cell, no interpolation: a particle takes the u/v of the published 2 km cell it is in.
 * - A particle dies the moment it would enter a cell without an observation (a gap, land, the edge of
 *   coverage) and is reborn only in an observed cell in view: particles never cross or fill a gap, and
 *   a drawn particle is always inside an observed cell.
 * - Screen speed is proportional to current speed and the same at every zoom (PX_PER_MS px/s per m/s):
 *   a scaled symbol of direction and RELATIVE speed, never the water's real speed or a path. Lives are
 *   short (1.2-2.4 s) and trails about 0.5 s, so no streak spans more than a few cells.
 * Positions are Web Mercator, 0..1 (georeferenced: they move with the map while it pans).
 */
export const PX_PER_MS = 30;
export const HIST = 10; // trail samples per particle
export const HIST_MS = 55; // ms between trail samples (a trail spans about 0.5 s)
/** Floats per drawn segment: ax ay bx by alpha r g b. */
export const SEG = 8;

/** Sequential, lightness-monotone speed ramp for the optional colour buckets (one colour per SPEED_CLASSES bucket; never a rainbow). */
export const SPEED_RAMP = ["#4f7a8c", "#5fa3b5", "#8cc9c4", "#c6e6b4", "#fbf3c4"] as const;
/** The particles' colour when they are not coloured by speed. */
export const FLOW_INK = "#edf5fa";

export const mercX = (lon: number) => (lon + 180) / 360;
export const mercY = (lat: number) => {
  const s = Math.sin((lat * Math.PI) / 180);
  return 0.5 - (0.25 * Math.log((1 + s) / (1 - s))) / Math.PI;
};
export const lonOfX = (x: number) => x * 360 - 180;
export const latOfY = (y: number) => (Math.atan(Math.sinh(Math.PI * (1 - 2 * y))) * 180) / Math.PI;

/** The speed bucket of a speed (m/s): the index of the SPEED_CLASSES class it falls in. */
export function speedBucket(speed: number): number {
  let b = 0;
  for (let i = 1; i < SPEED_CLASSES.length; i++) if (speed >= SPEED_CLASSES[i]) b = i;
  return b;
}

const rgb = (hex: string): [number, number, number] => [parseInt(hex.slice(1, 3), 16) / 255, parseInt(hex.slice(3, 5), 16) / 255, parseInt(hex.slice(5, 7), 16) / 255];
const RAMP_RGB = SPEED_RAMP.map(rgb);
const INK_RGB = rgb(FLOW_INK);

export type View = { west: number; south: number; east: number; north: number };

export class FlowSim {
  readonly max: number;
  count = 0;
  field: CurrentField | null = null;
  /** observed cells (index r * width + c), all and in view */
  valid: Int32Array = new Int32Array(0);
  visible: Int32Array = new Int32Array(0);
  private mask: Uint8Array = new Uint8Array(0);
  private speed: Float32Array = new Float32Array(0);
  readonly px: Float64Array;
  readonly py: Float64Array;
  private age: Float32Array;
  private life: Float32Array;
  private spd: Float32Array;
  private hist: Float32Array;
  private hn: Uint8Array;
  private rand: () => number;
  /** the smaller side of a cell, in Mercator units (the sub-step bound) */
  private cell = 1;

  constructor(max: number, rand: () => number = Math.random) {
    this.max = max;
    this.px = new Float64Array(max);
    this.py = new Float64Array(max);
    this.age = new Float32Array(max);
    this.life = new Float32Array(max);
    this.spd = new Float32Array(max);
    this.hist = new Float32Array(max * HIST * 2);
    this.hn = new Uint8Array(max);
    this.rand = rand;
  }

  /** A new hour (or the 24 h mean): every particle is reborn in it; nothing carries over from the last field. */
  setField(f: CurrentField) {
    this.field = f;
    const n = f.u.length;
    this.mask = new Uint8Array(n);
    this.speed = new Float32Array(n);
    const v: number[] = [];
    for (let k = 0; k < n; k++) {
      if (Number.isFinite(f.u[k]) && Number.isFinite(f.v[k])) {
        this.mask[k] = 1;
        this.speed[k] = Math.hypot(f.u[k], f.v[k]);
        v.push(k);
      }
    }
    this.valid = Int32Array.from(v);
    this.cell = Math.min(Math.abs(f.g.lon_step), Math.abs(f.g.lat_step)) / 360;
    this.visible = this.valid;
    this.life.fill(0);
    this.hn.fill(0);
  }

  /** The observed cell at a Mercator position, or -1 (outside the grid or no observation). */
  cellAt(x: number, y: number): number {
    const g = this.field?.g;
    if (!g) return -1;
    const r = Math.floor((latOfY(y) - g.lat_first) / g.lat_step + 0.5);
    const c = Math.floor((lonOfX(x) - g.lon_first) / g.lon_step + 0.5);
    if (r < 0 || c < 0 || r >= g.height || c >= g.width) return -1;
    const k = r * g.width + c;
    return this.mask[k] ? k : -1;
  }

  /** The observed cells whose centre is in view: where particles are born. */
  setView(b: View) {
    const g = this.field?.g;
    if (!g) return;
    const out: number[] = [];
    for (const k of this.valid) {
      const lat = g.lat_first + Math.floor(k / g.width) * g.lat_step;
      const lon = g.lon_first + (k % g.width) * g.lon_step;
      if (lat >= b.south && lat <= b.north && lon >= b.west && lon <= b.east) out.push(k);
    }
    this.visible = Int32Array.from(out);
  }

  setCount(n: number) {
    const c = Math.max(0, Math.min(this.max, Math.round(n)));
    for (let i = this.count; i < c; i++) this.life[i] = 0;
    this.count = c;
  }

  private spawn(i: number, randomAge: boolean) {
    const g = this.field?.g;
    const vis = this.visible;
    if (!g || !vis.length) {
      this.life[i] = 0;
      this.hn[i] = 0;
      return;
    }
    const k = vis[Math.floor(this.rand() * vis.length)];
    const r = Math.floor(k / g.width);
    const c = k % g.width;
    // anywhere inside the cell's own footprint (never on its edge, so it rounds back to this cell)
    this.px[i] = mercX(g.lon_first + (c + this.rand() * 0.98 - 0.49) * g.lon_step);
    this.py[i] = mercY(g.lat_first + (r + this.rand() * 0.98 - 0.49) * g.lat_step);
    this.life[i] = 1.2 + this.rand() * 1.2;
    this.age[i] = randomAge ? this.rand() * this.life[i] : 0;
    this.hn[i] = 0;
    this.spd[i] = this.speed[k];
  }

  respawnAll() {
    for (let i = 0; i < this.count; i++) this.spawn(i, true);
  }

  /**
   * Advance every particle by `dt` seconds at a zoom where the world is `worldPx` pixels wide
   * (512 * 2^zoom), sampling the trail when asked.
   */
  step(dt: number, worldPx: number, sampleHist: boolean) {
    const f = this.field;
    if (!f) return;
    const s = (PX_PER_MS * dt) / worldPx;
    for (let i = 0; i < this.count; i++) {
      if (this.life[i] === 0) {
        this.spawn(i, false);
        continue;
      }
      this.age[i] += dt;
      const k = this.cellAt(this.px[i], this.py[i]);
      if (k < 0 || this.age[i] > this.life[i]) {
        this.spawn(i, false);
        continue;
      }
      if (sampleHist) {
        const n = this.hn[i];
        const o = i * HIST * 2;
        if (n < HIST) {
          this.hist[o + n * 2] = this.px[i];
          this.hist[o + n * 2 + 1] = this.py[i];
          this.hn[i] = n + 1;
        } else {
          this.hist.copyWithin(o, o + 2, o + HIST * 2);
          this.hist[o + (HIST - 1) * 2] = this.px[i];
          this.hist[o + (HIST - 1) * 2 + 1] = this.py[i];
        }
      }
      // each cell's own vector, in sub-steps of under half a cell (zoomed out a cell is under a pixel, and a
      // step must never hop over a gap): entering a cell without an observation ends the particle there
      let kk = k;
      let x = this.px[i];
      let y = this.py[i];
      const n = Math.max(1, Math.ceil(Math.max(Math.abs(f.u[k]), Math.abs(f.v[k])) * s / (this.cell * 0.5)));
      let dead = false;
      for (let j = 0; j < n; j++) {
        x += (f.u[kk] * s) / n;
        y -= (f.v[kk] * s) / n;
        kk = this.cellAt(x, y);
        if (kk < 0) {
          dead = true;
          break;
        }
      }
      if (dead) {
        this.spawn(i, false);
        continue;
      }
      this.px[i] = x;
      this.py[i] = y;
      this.spd[i] = this.speed[kk];
    }
  }

  /** Fill the trails once without drawing: the still frame (reduced motion, paused). */
  settle(worldPx: number) {
    this.respawnAll();
    for (let f = 0; f < HIST + 1; f++) this.step(HIST_MS / 1000, worldPx, true);
  }

  /**
   * The drawn segments into `out` (SEG floats each): trail samples, then the live head. Alpha rises toward
   * the head and with speed, and fades in and out over the particle's life. Returns the segment count.
   */
  build(out: Float32Array, o: { colour: boolean; alpha: number }): number {
    let n = 0;
    for (let i = 0; i < this.count; i++) {
      const m = this.hn[i];
      if (this.life[i] === 0 || m === 0) continue;
      const a0 = Math.min(1, this.age[i] / 0.25, (this.life[i] - this.age[i]) / 0.35);
      const base = Math.max(0, a0) * Math.min(0.95, 0.4 + this.spd[i] * 1.4) * o.alpha;
      const [cr, cg, cb] = o.colour ? RAMP_RGB[speedBucket(this.spd[i])] : INK_RGB;
      const h = i * HIST * 2;
      for (let j = 0; j < m; j++) {
        if ((n + 1) * SEG > out.length) return n;
        const q = n * SEG;
        out[q] = this.hist[h + j * 2];
        out[q + 1] = this.hist[h + j * 2 + 1];
        out[q + 2] = j + 1 < m ? this.hist[h + (j + 1) * 2] : this.px[i];
        out[q + 3] = j + 1 < m ? this.hist[h + (j + 1) * 2 + 1] : this.py[i];
        out[q + 4] = base * ((j + 1) / m);
        out[q + 5] = cr;
        out[q + 6] = cg;
        out[q + 7] = cb;
        n++;
      }
    }
    return n;
  }

  /** Live particle heads as [lon, lat] (the e2e debug hook and tests). */
  heads(): [number, number][] {
    const out: [number, number][] = [];
    for (let i = 0; i < this.count; i++) if (this.life[i] > 0) out.push([lonOfX(this.px[i]), latOfY(this.py[i])]);
    return out;
  }

  /** Every trail point and head of live particles as [lon, lat]. */
  points(): [number, number][] {
    const out = this.heads();
    for (let i = 0; i < this.count; i++) {
      if (this.life[i] === 0) continue;
      for (let j = 0; j < this.hn[i]; j++) out.push([lonOfX(this.hist[(i * HIST + j) * 2]), latOfY(this.hist[(i * HIST + j) * 2 + 1])]);
    }
    return out;
  }
}
