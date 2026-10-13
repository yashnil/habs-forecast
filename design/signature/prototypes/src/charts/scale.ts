/**
 * Tiny local scales so the prototype has no chart dependency. coastwatch-web should use
 * d3-scale (scaleUtc, scaleLog, scaleLinear) and d3-shape instead (~14 kB gzip, see
 * research/02-library-research.md §8); the behaviour here mirrors theirs closely enough to judge
 * the design.
 */
export interface Scale { (v: number): number; domain: [number, number]; range: [number, number]; ticks: (n?: number) => number[]; kind: "linear" | "log" | "time" }

export function linear(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain, [r0, r1] = range;
  const f = ((v: number) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0)) as Scale;
  f.domain = domain; f.range = range; f.kind = "linear";
  f.ticks = (n = 5) => {
    const span = d1 - d0;
    if (!(span > 0)) return [d0];
    const raw = span / n, mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) ?? mag * 10;
    const out: number[] = [];
    for (let v = Math.ceil(d0 / step) * step; v <= d1 + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  };
  return f;
}

export function log(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain.map((v) => Math.log10(v)) as [number, number];
  const [r0, r1] = range;
  const f = ((v: number) => r0 + ((Math.log10(v) - d0) / (d1 - d0 || 1)) * (r1 - r0)) as Scale;
  f.domain = domain; f.range = range; f.kind = "log";
  f.ticks = () => {
    const out: number[] = [];
    for (let e = Math.ceil(d0 - 1e-9); e <= Math.floor(d1 + 1e-9); e++) out.push(10 ** e);
    return out;
  };
  return f;
}

const DAY = 86_400_000;
export function time(domain: [number, number], range: [number, number]): Scale {
  const base = linear(domain, range);
  const f = ((v: number) => base(v)) as Scale;
  f.domain = domain; f.range = range; f.kind = "time";
  f.ticks = (n = 6) => {
    const [t0, t1] = domain;
    const days = (t1 - t0) / DAY;
    const out: number[] = [];
    const d = new Date(t0);
    if (days > 900) {
      // years
      const step = days > 3000 ? 2 : 1;
      for (let y = d.getUTCFullYear() + 1; ; y += step) { const t = Date.UTC(y, 0, 1); if (t > t1) break; out.push(t); }
    } else {
      const months = days / 30.4;
      const step = months > 10 ? 2 : 1;
      for (let m = d.getUTCMonth() + 1, y = d.getUTCFullYear(); ; m += step) {
        const t = Date.UTC(y + Math.floor(m / 12), m % 12, 1);
        if (t > t1) break;
        out.push(t);
      }
    }
    return out.length > n * 2 ? out.filter((_, i) => i % 2 === 0) : out;
  };
  return f;
}

export function fmtTick(t: number, spanDays: number) {
  const d = new Date(t);
  if (spanDays > 900) return String(d.getUTCFullYear());
  const m = d.toLocaleDateString("en-US", { month: "short", timeZone: "UTC" });
  return d.getUTCMonth() === 0 ? `${m} ${d.getUTCFullYear()}` : m;
}

export function fmtValue(v: number) {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 10000) return Math.round(v).toLocaleString("en-US");
  if (a >= 100) return Math.round(v).toLocaleString("en-US");
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toPrecision(2);
}
export function fmtTickValue(v: number) {
  if (v >= 1000) return v >= 1e6 ? `${v / 1e6}M` : `${(v / 1000).toLocaleString("en-US")}k`;
  return String(+v.toPrecision(3));
}
