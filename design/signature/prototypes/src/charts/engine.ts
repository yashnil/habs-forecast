import { fmtTick, fmtTickValue, fmtValue, linear, log, time } from "./scale";
import { DAY, GAP_DAYS, type Quantity, type Sample } from "./season";
import type { OfficialRecord } from "../data/types";
import { shortNotice } from "./season";
import { NOW } from "../lib/time";

/**
 * Imperative SVG engine for the season chart. React owns the page; this owns the marks, so pointer
 * inspection and 60 fps transitions never re-render React.
 *
 * Transition rules (scientific honesty):
 *  - Changing the window interpolates the *axis domain*. Every mark is placed from the
 *    interpolated axis each frame, so samples slide to where they belong; no value changes.
 *  - Changing the quantity moves each sample vertically from its old position to its new one at
 *    the same x. A sample with no value for the new quantity fades out where it was; it never
 *    travels to zero. Lines are always straight segments between real samples, re-drawn from the
 *    current positions; segments are only drawn between samples ≤ 21 days apart.
 *  - Reduced motion: no movement; the plot cross-fades in 120 ms.
 */
const SVGNS = "http://www.w3.org/2000/svg";
const el = <K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}, parent?: Element) => {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  parent?.appendChild(e);
  return e;
};
const ease = (t: number) => 1 - Math.pow(1 - t, 3) * (1 - t * 0.2); // ≈ cubic-bezier(.2,0,0,1)
const lerp = (a: number, b: number, k: number) => a + (b - a) * k;

export interface ChartState {
  samples: Sample[];
  quantity: Quantity;
  xDomain: [number, number];
  notices: OfficialRecord[];
  model: { date: string; value: number }[] | null;
  highlight: { samples?: string[]; rules?: string[]; tailGap?: boolean } | null;
}
interface View { x0: number; x1: number; y0: number; y1: number; q: Quantity }

export class SeasonEngine {
  private svg: SVGSVGElement;
  private w = 720;
  private h = 300;
  private m = { top: 50, right: 18, bottom: 40, left: 54 };
  private g: Record<string, SVGGElement> = {};
  private dots = new Map<string, SVGCircleElement>();
  private lineA!: SVGPathElement;
  private lineB!: SVGPathElement;
  private state: ChartState | null = null;
  private view: View | null = null;
  private raf = 0;
  private cursor = -1;
  private visible: Sample[] = [];
  private readout: HTMLDivElement;
  private live: HTMLElement;
  private reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  constructor(private host: HTMLDivElement, readout: HTMLDivElement, live: HTMLElement) {
    this.readout = readout;
    this.live = live;
    this.svg = el("svg", { class: "sc-svg", "aria-hidden": "true" }) as SVGSVGElement;
    host.prepend(this.svg);
    const defs = el("defs", {}, this.svg);
    const pat = el("pattern", { id: "sc-hatch", width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, defs);
    el("rect", { width: 1.6, height: 6, fill: "var(--cw-gap-hatch)" }, pat);
    const clip = el("clipPath", { id: "sc-clip" }, defs);
    this.clipRect = el("rect", {}, clip) as SVGRectElement;
    for (const name of ["gaps", "model", "grid", "xaxis", "yaxis", "yaxisOld", "raxis", "floor", "rules", "plot", "cursor"]) {
      this.g[name] = el("g", { class: `sc-${name}` }, this.svg) as SVGGElement;
    }
    this.g.plot.setAttribute("clip-path", "url(#sc-clip)");
    this.g.gaps.setAttribute("clip-path", "url(#sc-clip)");
    this.g.model.setAttribute("clip-path", "url(#sc-clip)");
    this.lineA = el("path", { class: "sc-line" }, this.g.plot) as SVGPathElement;
    this.lineB = el("path", { class: "sc-line" }, this.g.plot) as SVGPathElement;
    this.dotLayer = el("g", {}, this.g.plot) as SVGGElement;
    this.cursorLine = el("line", { class: "sc-cursor-line" }, this.g.cursor) as SVGLineElement;
    this.cursorRing = el("circle", { class: "sc-cursor-ring", r: 6.5 }, this.g.cursor) as SVGCircleElement;
    this.g.cursor.style.opacity = "0";

    host.addEventListener("pointermove", this.onPointer);
    host.addEventListener("pointerdown", this.onPointer);
    host.addEventListener("pointerleave", () => { if (document.activeElement !== host) this.setCursor(-1); });
    host.addEventListener("keydown", this.onKey);
    host.addEventListener("blur", () => this.setCursor(-1));
  }
  private clipRect: SVGRectElement;
  private dotLayer: SVGGElement;
  private cursorLine: SVGLineElement;
  private cursorRing: SVGCircleElement;

  destroy() { cancelAnimationFrame(this.raf); this.svg.remove(); }

  resize(width: number) {
    this.w = Math.max(280, width);
    this.h = this.w < 520 ? 250 : 300;
    this.m.left = this.w < 520 ? 44 : 54;
    if (this.state && this.view) this.draw(this.view, this.view, 1);
  }

  /* -------------------------------------------------------------- scales */
  private yDomainFor(q: Quantity, s: Sample[], x: [number, number]): [number, number] {
    const vals = s.filter((p) => p.t >= x[0] && p.t <= x[1]).map((p) => p.v[q.id]).filter((v): v is number => v != null && (q.scale === "linear" || v > 0));
    if (!vals.length) return q.scale === "log" ? [0.01, 1] : [0, 1];
    const mn = Math.min(...vals), mx = Math.max(...vals);
    if (q.scale === "log") return [10 ** Math.floor(Math.log10(mn)), 10 ** Math.ceil(Math.log10(mx) + 1e-9)];
    const pad = (mx - mn) * 0.12 || 1;
    const lo = q.id === "chl" ? 0 : Math.floor(mn - pad), hi = Math.ceil(mx + pad);
    return [lo, hi];
  }
  private plotBox() {
    const r = this.state?.model && this.state.quantity.model ? 46 : this.m.right;
    return { x0: this.m.left, x1: this.w - r, y0: this.m.top, y1: this.h - this.m.bottom };
  }
  private floorLane(q: Quantity) { return q.scale === "log" ? 22 : 0; }
  private yScale(v: View) {
    const b = this.plotBox();
    const lane = this.floorLane(v.q);
    const range: [number, number] = [b.y1 - lane - 4, b.y0];
    return v.q.scale === "log" ? log([v.y0, v.y1], range) : linear([v.y0, v.y1], range);
  }
  private yOf(v: View, s: Sample): number | null {
    const x = s.v[v.q.id];
    if (x == null) return null;
    if (v.q.scale === "log" && x <= 0) return this.plotBox().y1 - 6; // the "reported 0" lane
    return this.yScale(v)(x);
  }

  /* -------------------------------------------------------------- update */
  set(next: ChartState, animate = true) {
    const prev = this.state;
    this.state = next;
    const yd = this.yDomainFor(next.quantity, next.samples, next.xDomain);
    const target: View = { x0: next.xDomain[0], x1: next.xDomain[1], y0: yd[0], y1: yd[1], q: next.quantity };
    const from = this.view ?? target;
    this.view = target;
    this.ensureDots(next.samples);
    this.drawStatic();
    cancelAnimationFrame(this.raf);
    if (!prev || !animate) return this.draw(target, target, 1);
    if (this.reduced) {
      this.draw(target, target, 1);
      this.svg.animate([{ opacity: 0.35 }, { opacity: 1 }], { duration: 120, easing: "linear" });
      return;
    }
    const t0 = performance.now(), dur = 460;
    const step = (now: number) => {
      const k = Math.min(1, (now - t0) / dur);
      this.draw(from, target, ease(k));
      if (k < 1) this.raf = requestAnimationFrame(step);
    };
    this.raf = requestAnimationFrame(step);
  }

  private ensureDots(samples: Sample[]) {
    if (this.dots.size) return;
    for (const s of samples) {
      const c = el("circle", { r: 3.2, class: "sc-dot" }, this.dotLayer) as SVGCircleElement;
      this.dots.set(s.key, c);
    }
  }

  /** Static-per-state parts: highlight classes. */
  private drawStatic() {
    const h = this.state!.highlight;
    const hs = new Set(h?.samples ?? []);
    this.svg.classList.toggle("sc-has-highlight", !!(h && (hs.size || h.rules?.length || h.tailGap)));
    for (const [k, c] of this.dots) c.classList.toggle("is-hl", hs.has(k));
  }

  private draw(A: View, B: View, k: number) {
    const st = this.state!;
    const b = this.plotBox();
    this.svg.setAttribute("viewBox", `0 0 ${this.w} ${this.h}`);
    this.svg.setAttribute("width", String(this.w));
    this.svg.setAttribute("height", String(this.h));
    this.clipRect.setAttribute("x", String(b.x0 - 6));
    this.clipRect.setAttribute("y", String(0));
    this.clipRect.setAttribute("width", String(b.x1 - b.x0 + 12));
    this.clipRect.setAttribute("height", String(this.h));
    const sameQ = A.q.id === B.q.id;
    const xD: [number, number] = [lerp(A.x0, B.x0, k), lerp(A.x1, B.x1, k)];
    const X = time(xD, [b.x0, b.x1]);
    let Vk: View = B;
    if (sameQ) {
      const li = (a: number, c: number) => (B.q.scale === "log" ? 10 ** lerp(Math.log10(a), Math.log10(c), k) : lerp(a, c, k));
      Vk = { ...B, y0: li(A.y0, B.y0), y1: li(A.y1, B.y1) };
    }
    const spanDays = (xD[1] - xD[0]) / DAY;

    // x axis
    this.g.xaxis.replaceChildren();
    el("line", { x1: b.x0, x2: b.x1, y1: b.y1, y2: b.y1, class: "sc-axis-line" }, this.g.xaxis);
    for (const t of X.ticks(this.w < 520 ? 4 : 7)) {
      const x = X(t);
      if (x < b.x0 - 1 || x > b.x1 + 1) continue;
      el("line", { x1: x, x2: x, y1: b.y1, y2: b.y1 + 4, class: "sc-tick" }, this.g.xaxis);
      const tx = el("text", { x, y: b.y1 + 18, class: "sc-tick-label", "text-anchor": "middle" }, this.g.xaxis);
      tx.textContent = fmtTick(t, spanDays);
    }
    // y axes (cross-fade on quantity change)
    this.drawY(this.g.yaxis, sameQ ? Vk : B, sameQ ? 1 : k);
    if (!sameQ) this.drawY(this.g.yaxisOld, A, 1 - k);
    else this.g.yaxisOld.replaceChildren();
    // reported-zero lane
    this.g.floor.replaceChildren();
    if (B.q.scale === "log" || (!sameQ && A.q.scale === "log")) {
      const op = sameQ ? 1 : B.q.scale === "log" && A.q.scale === "log" ? 1 : B.q.scale === "log" ? k : 1 - k;
      const y = b.y1 - 6;
      const lane = el("g", { opacity: op }, this.g.floor);
      el("line", { x1: b.x0, x2: b.x1, y1: y, y2: y, class: "sc-floor-line" }, lane);
      const t = el("text", { x: b.x0 - 6, y: y + 4, class: "sc-tick-label", "text-anchor": "end" }, lane);
      t.textContent = "0";
    }

    // gaps: intervals with no sample visit > GAP_DAYS, and the tail since the last sample
    this.g.gaps.replaceChildren();
    const s = st.samples;
    const tailGap = st.highlight?.tailGap;
    for (let i = 1; i <= s.length; i++) {
      const a = s[i - 1].t, c = i < s.length ? s[i].t : NOW.getTime();
      const tail = i === s.length;
      if ((c - a) / DAY <= (tail ? 7 : GAP_DAYS)) continue;
      const xa = X(a), xc = X(c);
      if (xc < b.x0 || xa > b.x1) continue;
      const r = el("rect", { x: xa + 3, y: b.y0, width: Math.max(0, xc - xa - 6), height: b.y1 - b.y0, fill: "url(#sc-hatch)", class: tail ? "sc-gap sc-gap-tail" + (tailGap ? " is-hl" : "") : "sc-gap" }, this.g.gaps);
      if (tail) {
        r.setAttribute("data-tail", "1");
        if (xc - xa > 36) {
          const t = el("text", { x: xc - 4, y: b.y0 + 12, class: "sc-gap-label", "text-anchor": "end" }, this.g.gaps);
          t.textContent = xc - xa > 110 ? "No sample since" : "None since";
        }
      }
    }

    // model: stepped daily columns on a 0–100 % right axis (a field, not a line)
    this.g.model.replaceChildren();
    this.g.raxis.replaceChildren();
    if (st.model && B.q.model) {
      const R = linear([0, 1], [b.y1, b.y0]);
      for (const p of st.model) {
        const t = Date.parse(p.date + "T00:00:00Z");
        const xa = X(t), xc = X(t + DAY);
        if (xc < b.x0 || xa > b.x1) continue;
        el("rect", { x: xa, width: Math.max(0.6, xc - xa - (xc - xa > 3 ? 0.5 : 0)), y: R(p.value), height: b.y1 - R(p.value), class: "sc-model" }, this.g.model);
      }
      for (const v of [0, 0.5, 1]) {
        const tx = el("text", { x: b.x1 + 6, y: R(v) + 4, class: "sc-tick-label sc-model-tick" }, this.g.raxis);
        tx.textContent = `${v * 100}%`;
      }
    }

    // official rules
    this.g.rules.replaceChildren();
    const hlRules = new Set(st.highlight?.rules ?? []);
    // labels take the first row where they don't collide with a label already placed
    const rows: [number, number][][] = [[], [], []];
    const rules = [...st.notices].sort((p, q) => p.effective_date!.localeCompare(q.effective_date!));
    for (const r of rules) {
      const x = X(Date.parse(r.effective_date! + "T12:00:00Z"));
      if (x < b.x0 || x > b.x1) continue;
      const text = this.w < 520 ? r.agency : shortNotice(r);
      const w = text.length * 6.3;
      const anchorEnd = x + 4 + w > b.x1 + this.m.right;
      const span: [number, number] = anchorEnd ? [x - 4 - w, x - 4] : [x + 4, x + 4 + w];
      let row = rows.findIndex((rw) => rw.every(([a, c]) => span[1] < a - 6 || span[0] > c + 6));
      if (row < 0) row = rows.length - 1;
      rows[row].push(span);
      const gr = el("g", { class: "sc-rule" + (hlRules.has(r.id) ? " is-hl" : "") }, this.g.rules);
      el("line", { x1: x, x2: x, y1: b.y0 - 26 + row * 12 + 2, y2: b.y1, class: "sc-rule-line" }, gr);
      const tx = el("text", { x: x + (anchorEnd ? -4 : 4), y: b.y0 - 22 + row * 12 + 6, class: "sc-rule-label", "text-anchor": anchorEnd ? "end" : "start" }, gr);
      tx.textContent = text;
    }

    // samples and lines
    const visible: Sample[] = [];
    const ptsA: ([number, number] | null)[] = [];
    const ptsB: ([number, number] | null)[] = [];
    for (const p of s) {
      const c = this.dots.get(p.key)!;
      const x = X(p.t);
      const inX = x >= b.x0 - 8 && x <= b.x1 + 8;
      if (inX && p.t >= xD[0] - DAY && p.t <= xD[1]) visible.push(p);
      let y: number | null, op: number;
      if (sameQ) {
        y = this.yOf(Vk, p);
        op = y == null ? 0 : 1;
        ptsB.push(y == null || !this.lineable(B.q, p) ? null : [x, y]);
      } else {
        const ya = this.yOf(A, p), yb = this.yOf(B, p);
        if (ya != null && yb != null) { y = lerp(ya, yb, k); op = 1; }
        else if (ya != null) { y = ya; op = 1 - k; }
        else if (yb != null) { y = yb; op = k; }
        else { y = null; op = 0; }
        const yy = y ?? 0;
        ptsA.push(ya != null && this.lineable(A.q, p) ? [x, yb != null ? yy : ya] : null);
        ptsB.push(yb != null && this.lineable(B.q, p) ? [x, ya != null ? yy : yb] : null);
      }
      if (!inX || y == null || op <= 0.001) { c.style.display = "none"; continue; }
      c.style.display = "";
      c.setAttribute("cx", x.toFixed(1));
      c.setAttribute("cy", y.toFixed(1));
      c.style.opacity = String(op);
      const zero = B.q.scale === "log" && p.v[B.q.id] === 0;
      c.classList.toggle("is-zero", zero);
    }
    this.visible = visible;
    const path = (pts: ([number, number] | null)[]) => {
      let d = "", prevT = -Infinity, open = false;
      pts.forEach((pt, i) => {
        const t = s[i].t;
        if (!pt) { open = false; return; }
        if (open && (t - prevT) / DAY <= GAP_DAYS) d += `L${pt[0].toFixed(1)},${pt[1].toFixed(1)}`;
        else d += `M${pt[0].toFixed(1)},${pt[1].toFixed(1)}`;
        open = true;
        prevT = t;
      });
      return d;
    };
    this.lineB.setAttribute("d", path(ptsB));
    this.lineB.style.opacity = sameQ ? "1" : String(k);
    this.lineA.setAttribute("d", sameQ ? "" : path(ptsA));
    this.lineA.style.opacity = sameQ ? "0" : String(1 - k);
    if (this.cursor >= 0) this.placeCursor();
  }

  private lineable(q: Quantity, p: Sample) {
    const v = p.v[q.id];
    return v != null && (q.scale === "linear" || v > 0);
  }

  private drawY(g: SVGGElement, v: View, op: number) {
    const b = this.plotBox();
    const Y = this.yScale(v);
    g.replaceChildren();
    g.setAttribute("opacity", String(op));
    for (const t of Y.ticks(this.h < 280 ? 4 : 5)) {
      const y = Y(t);
      if (y < b.y0 - 2 || y > b.y1) continue;
      el("line", { x1: b.x0, x2: b.x1, y1: y, y2: y, class: "sc-grid" }, g);
      const tx = el("text", { x: b.x0 - 6, y: y + 4, class: "sc-tick-label", "text-anchor": "end" }, g);
      tx.textContent = fmtTickValue(t);
    }
  }

  /* -------------------------------------------------------------- inspection */
  private onPointer = (e: PointerEvent) => {
    if (!this.visible.length) return;
    const r = this.svg.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * this.w;
    const b = this.plotBox();
    if (px < b.x0 - 10 || px > b.x1 + 10) return this.setCursor(-1);
    const X = time([this.view!.x0, this.view!.x1], [b.x0, b.x1]);
    let best = -1, bd = Infinity;
    this.visible.forEach((s, i) => { const d = Math.abs(X(s.t) - px); if (d < bd) { bd = d; best = i; } });
    this.setCursor(best);
  };
  private onKey = (e: KeyboardEvent) => {
    const n = this.visible.length;
    if (!n) return;
    let i = this.cursor;
    if (e.key === "ArrowRight") i = i < 0 ? n - 1 : Math.min(n - 1, i + 1);
    else if (e.key === "ArrowLeft") i = i < 0 ? n - 1 : Math.max(0, i - 1);
    else if (e.key === "Home") i = 0;
    else if (e.key === "End") i = n - 1;
    else if (e.key === "Escape") i = -1;
    else return;
    e.preventDefault();
    this.setCursor(i, true);
  };
  private setCursor(i: number, announce = false) {
    this.cursor = i;
    if (i < 0) { this.g.cursor.style.opacity = "0"; this.readout.hidden = true; return; }
    this.placeCursor();
    if (announce) this.live.textContent = this.readout.innerText.replace(/\n+/g, ". ");
  }
  private placeCursor() {
    const s = this.visible[this.cursor];
    if (!s || !this.view) return;
    const b = this.plotBox();
    const X = time([this.view.x0, this.view.x1], [b.x0, b.x1]);
    const x = X(s.t);
    const y = this.yOf(this.view, s);
    this.cursorLine.setAttribute("x1", String(x));
    this.cursorLine.setAttribute("x2", String(x));
    this.cursorLine.setAttribute("y1", String(b.y0));
    this.cursorLine.setAttribute("y2", String(b.y1));
    this.cursorRing.style.display = y == null ? "none" : "";
    if (y != null) { this.cursorRing.setAttribute("cx", String(x)); this.cursorRing.setAttribute("cy", String(y)); }
    this.g.cursor.style.opacity = "1";
    const q = this.view.q;
    const v = s.v[q.id];
    const date = new Date(s.t).toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric", year: "numeric", timeZone: "America/Los_Angeles" });
    const val = v == null ? "No value (not measured)" : v === 0 && q.scale === "log" ? "Reported 0 (not quantified)" : `${fmtValue(v)} ${q.units}`;
    this.readout.innerHTML = `<span class="sc-ro-date">${date}</span><span class="sc-ro-val num">${val}</span><span class="sc-ro-q">${q.short}</span>`;
    this.readout.hidden = false;
    const hostW = this.host.clientWidth;
    const scale = hostW / this.w;
    const left = x * scale;
    const roW = this.readout.offsetWidth || 180;
    this.readout.style.transform = `translate(${Math.round(left + 12 + roW > hostW ? left - roW - 12 : left + 12)}px, ${Math.round((b.y0 + 4) * scale)}px)`;
  }
}

export function summarySentence(st: ChartState): string {
  const q = st.quantity;
  const vis = st.samples.filter((s) => s.t >= st.xDomain[0] && s.t <= st.xDomain[1]);
  const vals = vis.map((s) => s.v[q.id]).filter((v): v is number => v != null);
  const fmt = (t: number) => new Date(t).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
  if (!vals.length) return `${q.short} at this pier, ${fmt(st.xDomain[0])} to ${fmt(st.xDomain[1])}: no values in this period.`;
  const mx = Math.max(...vals);
  const at = vis.find((s) => s.v[q.id] === mx)!;
  const zeros = q.scale === "log" ? vals.filter((v) => v === 0).length : 0;
  return `${q.short} at this pier, ${fmt(st.xDomain[0])} to ${fmt(st.xDomain[1])}: ${vals.length} samples, highest ${fmtValue(mx)} ${q.units} on ${fmt(at.t)}${zeros ? `, ${zeros} reported 0 (not quantified)` : ""}.${st.notices.length ? ` Vertical lines mark when official notices took effect.` : ""}`;
}
