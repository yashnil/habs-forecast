"use client";

import { useId } from "react";
import type { BiotoxinSet } from "@/pilot/contracts";
import type { BloomDay, OutlookRead } from "@/pilot/facts";
import { dayWords } from "@/pilot/facts";
import type { Tide } from "@/lib/conditions";
import { minutesOf, tideDay } from "@/lib/tideCurve";
import { ChartFrame, ChartArea, ChartBars, ChartLine, ChartMarks, DirectLabel, NowLine, TimeAxis, scaleLinear } from "@/v4/charts/kit";
import { useWidth } from "@/v4/charts/useWidth";
import type { TestSiteView } from "@/pilot/facts";
import type { LanePeriod } from "./marine";
import { windWords } from "./marine";

/**
 * Today's small instruments: hand-drawn SVG, no chart library. Each chart is an image with a text
 * alternative (`<title>` and `<desc>`) that says everything the drawing says. Measured values are
 * dots at their own dates. Nothing is interpolated between them, and nothing is extrapolated past
 * the newest.
 */

const fmt = (v: number) => (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1).replace(/\.0$/, "") : v.toFixed(1));

/** CDPH sample sets over time: every individual value as a dot, the average as a tick, against the action level. */
export function ToxinChart({ sets, action, label, now }: { sets: BiotoxinSet[]; action: number | null; label: string; now: string }) {
  const id = useId();
  const shown = [...sets].sort((a, b) => a.date.localeCompare(b.date)).slice(-6);
  const vals = shown.flatMap((s) => s.values_ppm.filter((v): v is number => v != null));
  const maxV = Math.max(0, ...vals);
  const W = 320;
  const H = 136;
  const L = 34;
  const R = 10;
  const T = 14;
  const B = 24;
  const log = action != null && maxV / action > 6;
  const top = log ? 10 ** Math.ceil(Math.log10(Math.max(maxV, (action ?? 1) * 2))) : Math.max((action ?? 0) * 1.35, maxV * 1.12, 1);
  const y = (v: number) => (log ? T + (H - T - B) * (1 - Math.log10(Math.max(v, 1)) / Math.log10(top)) : T + (H - T - B) * (1 - v / top));
  const step = (W - L - R) / Math.max(shown.length, 1);
  const x = (i: number) => L + step * (i + 0.5);
  const ticks = log ? [1, 10, 100, 1000, 10000, 100000].filter((t) => t <= top) : action != null ? [0, action] : [0, Math.round(top / 2), Math.round(top)];
  const tickLabel = (t: number) => (t >= 1000 ? `${t / 1000}k` : String(t));
  const desc = shown.map((s) => `${dayWords(s.date, now)}: ${s.values_ppm.length} samples, ${s.values_ppm.map((v) => (v == null ? "ND" : fmt(v))).join(", ")} ppm${s.avg_ppm != null ? ` (average ${fmt(s.avg_ppm)})` : ""}`).join("; ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" className="today-chart block" role="img" aria-labelledby={`${id}t ${id}d`} data-testid="toxin-chart">
      <title id={`${id}t`}>{`${label}: domoic acid in each sample, by collection date${action != null ? `, against the ${action} ppm action level` : ""}${log ? " (log scale)" : ""}`}</title>
      <desc id={`${id}d`}>{desc}</desc>
      {ticks.map((t) => (
        <g key={t}>
          <line className="grid" x1={L} x2={W - R} y1={y(t)} y2={y(t)} />
          <text x={L - 6} y={y(t) + 3.5} textAnchor="end">
            {tickLabel(t)}
          </text>
        </g>
      ))}
      {action != null && (
        <g>
          <line className="limit" x1={L} x2={W - R} y1={y(action)} y2={y(action)} />
          <text className="limit-t" x={W - R} y={y(action) - 4} textAnchor="end">
            limit {action} ppm
          </text>
        </g>
      )}
      {shown.map((s, i) => {
        const n = s.values_ppm.length;
        return (
          <g key={s.date}>
            {s.values_ppm.map((v, j) => {
              const cx = x(i) + (j - (n - 1) / 2) * Math.min(6, (step - 10) / Math.max(n, 1));
              return v == null ? <circle key={j} className="dot nd" cx={cx} cy={y(log ? 1 : 0)} r={2.6} /> : <circle key={j} className={`dot${action != null && v >= action ? " over" : ""}`} cx={cx} cy={y(v)} r={3} />;
            })}
            {s.avg_ppm != null && <line className="avg" x1={x(i) - Math.min(step * 0.32, 26)} x2={x(i) + Math.min(step * 0.32, 26)} y1={y(s.avg_ppm)} y2={y(s.avg_ppm)} />}
            <text x={x(i)} y={H - 6} textAnchor="middle">
              {dayWords(s.date, now).replace(/, \d{4}$/, "")}
            </text>
          </g>
        );
      })}
      {log && (
        <text x={L} y={10}>
          ppm, log scale
        </text>
      )}
      {!log && (
        <text x={L} y={10}>
          ppm
        </text>
      )}
    </svg>
  );
}

const addDays = (d: string, n: number) => new Date(Date.parse(`${d}T12:00:00Z`) + n * 86_400_000).toISOString().slice(0, 10);
const diffDays = (a: string, b: string) => Math.round((Date.parse(`${b}T12:00:00Z`) - Date.parse(`${a}T12:00:00Z`)) / 86_400_000);

/** C-HARM's daily probabilities (days 0–3 of its run) on their own dates, with today marked. */
export function BloomTimeline({ days, today, old, now }: { days: BloomDay[]; today: string; old: boolean; now: string }) {
  const id = useId();
  const dated = days.filter((d) => d.date);
  const start = [dated[0]?.date, today].filter((x): x is string => !!x).sort()[0];
  const end = [dated.at(-1)?.date, addDays(start, 3), today].filter((x): x is string => !!x).sort().at(-1)!;
  const n = diffDays(start, end) + 1;
  const W = 340;
  const L = 4;
  const R = 4;
  const col = (W - L - R) / n;
  const xOf = (d: string) => L + diffDays(start, d) * col;
  const T1 = 38;
  const B1 = 104;
  const H = 128;
  const yP = (p: number) => B1 - (B1 - T1) * Math.max(0, Math.min(1, p));
  const labelEvery = n > 10 ? 2 : 1;
  const desc = dated.map((d) => `${dayWords(d.date!, now)}: ${d.pda != null ? `${Math.round(d.pda * 100)}% particulate domoic acid` : "no domoic acid value"}${d.pn != null ? `, ${Math.round(d.pn * 100)}% Pseudo-nitzschia bloom` : ""}`).join("; ");
  // a bar never wider than a day on a short axis would make it look like a block
  const bw = Math.min(col, 44);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" className="today-chart block" role="img" aria-labelledby={`${id}t ${id}d`} data-testid="bloom-timeline">
      <title id={`${id}t`}>C-HARM: probability of bloom and toxin conditions in the water, by day</title>
      <desc id={`${id}d`}>{desc}</desc>
      <text x={L} y={12} style={{ letterSpacing: "0.08em" }}>
        C-HARM · PROBABILITY OF CONDITIONS
      </text>
      <line className="grid" x1={L} x2={W - R} y1={B1} y2={B1} />
      <line className="grid" x1={L} x2={W - R} y1={yP(0.5)} y2={yP(0.5)} strokeDasharray="2 3" />
      {dated.map((d) => {
        const x0 = xOf(d.date!) + (col - bw) / 2;
        const g = old ? "old" : "";
        return (
          <g key={d.lead} className={g}>
            {d.pn != null && <rect className="pn" x={x0 + bw * 0.12} width={bw * 0.76} y={yP(d.pn)} height={B1 - yP(d.pn)} rx={1.5} />}
            {d.pda != null && <rect className="pda" x={x0 + bw * 0.32} width={bw * 0.36} y={yP(d.pda)} height={B1 - yP(d.pda)} rx={1.5} />}
            {d.pda != null && (
              <text className="val" x={x0 + bw / 2} y={yP(Math.max(d.pda, d.pn ?? 0)) - 4} textAnchor="middle">
                {Math.round(d.pda * 100)}%
              </text>
            )}
            {d.pda == null && d.pn == null && (
              <text x={x0 + bw / 2} y={B1 - 6} textAnchor="middle">
                —
              </text>
            )}
          </g>
        );
      })}
      {today >= start && today <= end && <line className="now" x1={xOf(today) + col / 2} x2={xOf(today) + col / 2} y1={T1 - 4} y2={B1 + 4} />}
      {Array.from({ length: n }, (_, i) => addDays(start, i)).map((d, i) =>
        d === today || (i % labelEvery === 0 && Math.abs(diffDays(today, d)) > (col < 34 ? 1 : 0)) ? (
          <text key={d} x={xOf(d) + col / 2} y={H - 6} textAnchor="middle" style={d === today ? { fill: "var(--nx-ink)", fontWeight: 600 } : undefined}>
            {d === today ? "today" : dayWords(d, now).replace(/^\w+ /, "")}
          </text>
        ) : null,
      )}
    </svg>
  );
}

/**
 * The 8-day chlorophyll outlook against usual, on one scale. The centre tick is the usual value for
 * this time of year; each step left or right is a halving or a doubling (a log scale, because the
 * outlook's error is a factor, not an amount). The dot is the outlook, the shaded band its typical
 * error either way, and the bracket the span Today calls "about usual". The scale ends at a quarter
 * and at four times usual; an outlook beyond an end is drawn at the end with an arrow.
 */
export function OutlookScale({ read, errorFactor, signal }: { read: OutlookRead; errorFactor: number | null; signal: number }) {
  const id = useId();
  const W = 340;
  const L = 12;
  const R = 12;
  const SPAN = 2; // doublings shown either side of usual
  const Y = 48;
  const x = (ratio: number) => L + ((Math.log2(ratio) + SPAN) / (2 * SPAN)) * (W - L - R);
  const clampR = (r: number) => Math.max(2 ** -SPAN, Math.min(2 ** SPAN, r));
  const mx = x(clampR(read.ratio));
  const off = read.ratio > 2 ** SPAN ? 1 : read.ratio < 2 ** -SPAN ? -1 : 0;
  const band = errorFactor ? ([x(clampR(read.ratio / errorFactor)), x(clampR(read.ratio * errorFactor))] as const) : null;
  const times = `${read.ratio >= 9.95 ? read.ratio.toFixed(0) : read.ratio.toFixed(1)}×`;
  const label = `outlook · ${times} usual`;
  // the label sits over the dot, kept inside the drawing
  const lx = Math.max(L + 44, Math.min(W - R - 44, mx));
  const efWords = errorFactor ? String(Math.round(errorFactor * 2) / 2) : null;
  const desc = `The outlook is ${times} the usual chlorophyll for this time of year: ${read.phrase}.${efWords ? ` Shaded band: its typical error, a factor of ${efWords} either way, from ${(read.ratio / errorFactor!).toFixed(1)}× to ${(read.ratio * errorFactor!).toFixed(1)}× usual.` : ""} Between ${(1 / signal).toFixed(2)}× and ${signal}× usual is called about usual.`;
  return (
    <svg viewBox={`0 0 ${W} 92`} width="100%" className="today-chart today-outlook block" role="img" aria-labelledby={`${id}t ${id}d`} data-testid="outlook-scale" data-level={read.level}>
      <title id={`${id}t`}>8-day chlorophyll outlook against the usual level for this time of year (not toxin)</title>
      <desc id={`${id}d`}>{desc}</desc>
      <text x={L} y={11} className={read.level === "lower" ? "on" : undefined}>
        ← lower than usual
      </text>
      <text x={W - R} y={11} textAnchor="end" className={read.level === "higher" ? "on" : undefined}>
        higher than usual →
      </text>
      {band && <rect className="band" x={band[0]} width={Math.max(2, band[1] - band[0])} y={Y - 9} height={18} rx={2} />}
      <line className="axis" x1={L} x2={W - R} y1={Y} y2={Y} />
      {[0.25, 0.5, 2, 4].map((r) => (
        <line key={r} className="axis" x1={x(r)} x2={x(r)} y1={Y - 4} y2={Y + 4} />
      ))}
      <line className="usual" x1={x(1)} x2={x(1)} y1={Y - 13} y2={Y + 13} />
      {/* the span called "about usual" */}
      <path className="bracket" d={`M${x(1 / signal)} ${Y + 14} v4 H${x(signal)} v-4`} />
      <text x={lx} y={Y - 17} textAnchor="middle" className="val">
        {label}
      </text>
      {off === 0 ? <circle className="mark" cx={mx} cy={Y} r={5.5} /> : <path className="mark" d={off > 0 ? `M${mx - 9} ${Y - 6} L${mx + 1} ${Y} L${mx - 9} ${Y + 6} Z` : `M${mx + 9} ${Y - 6} L${mx - 1} ${Y} L${mx + 9} ${Y + 6} Z`} />}
      <text x={x(0.5)} y={84} textAnchor="middle">
        half
      </text>
      <text x={x(1)} y={84} textAnchor="middle" className={read.level === "usual" ? "on" : undefined}>
        about usual
      </text>
      <text x={x(2)} y={84} textAnchor="middle">
        double
      </text>
    </svg>
  );
}

/** The six RAMP zones, north to south, with the port's zone marked (a diagram, not a map). */
export function ZoneLadder({ zone, severity }: { zone: string; severity: "red" | "amber" | "neutral" | "info" }) {
  const zones = ["1", "2", "3", "4", "5", "6"];
  const H = 156;
  const h = (H - 10) / zones.length;
  return (
    <svg viewBox={`0 0 92 ${H}`} width="92" height={H} className="today-chart block shrink-0" aria-hidden="true" data-testid="zone-ladder">
      {zones.map((z, i) => (
        <g key={z}>
          <rect className={`zone${z === zone ? ` here ${severity === "red" ? "" : severity}` : ""}`} x={18} y={5 + i * h} width={44} height={h - 3} rx={2} />
          <text x={40} y={5 + i * h + h / 2 + 2} textAnchor="middle" className={z === zone ? "val" : undefined}>
            {z}
          </text>
          {z === zone && <path d={`M66 ${5 + i * h + h / 2 - 1} l8 -5 v10 z`} fill="var(--nx-ink)" />}
        </g>
      ))}
      <text x={2} y={12}>
        N
      </text>
      <text x={2} y={H - 4}>
        S
      </text>
    </svg>
  );
}

/* ---- chapter graphics, on the shared kit: one per chapter, drawn in once ---- */

const clock12 = (m: number) => {
  const d = ((m % 1440) + 1440) % 1440;
  const h = Math.floor(d / 60);
  return d === 0 ? "12a" : d === 720 ? "noon" : `${h % 12 || 12}${h < 12 ? "a" : "p"}`;
};
const WEEKDAY = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

/**
 * The next 36 hours on one axis (past left of now, future right): NWS's wind as a stepped range per forecast
 * period, seas in numbers, the predicted tide under it, a now line through both. A period's words are its own
 * ("W 10–15 kt, then NW 5–10 kt"); the bar is the range NWS states, never a joined number.
 */
export function SeaStrip({ lanes, tides, day, nowMin }: { lanes: LanePeriod[]; tides: Tide[]; day: string; nowMin: number }) {
  const [ref, w] = useWidth<HTMLDivElement>(640);
  const H = 232;
  const domain = [nowMin - 90, nowMin + 36 * 60] as const;
  const x = scaleLinear(domain, [0, w]);
  const WIND_TOP = 40;
  const WIND_BASE = 92;
  const SEAS_Y = 110;
  const TIDE_TOP = 150;
  const TIDE_BASE = 206;
  const spans = lanes
    .filter((p) => p.date)
    .map((p) => {
      const start = minutesOf(`${p.date} ${p.night ? "18:00" : "06:00"}`, day);
      const end = start + 720;
      return { p, from: Math.max(start, domain[0]), to: Math.min(end, domain[1]), start, end };
    })
    .filter((s) => s.to > s.from);
  const maxWind = Math.max(25, ...spans.map((s) => s.p.windHi ?? 0));
  const wy = scaleLinear([0, maxWind], [WIND_BASE, WIND_TOP]);
  const seasOf = (p: LanePeriod) => (p.seasHi != null ? `${p.seasLo != null && p.seasLo !== p.seasHi ? `${p.seasLo}–${p.seasHi}` : p.seasHi} ft` : "—");
  // the tide as one curve through its predicted turns, across the days the axis touches
  const pts: { m: number; ft: number }[] = [];
  const turns: { m: number; ft: number; type: string; at: string }[] = [];
  for (let k = 0; k < 3; k++) {
    const d = new Date(Date.parse(`${day}T12:00:00Z`) + k * 86_400_000).toISOString().slice(0, 10);
    const td = tideDay(tides, d, 15);
    if (!td) continue;
    for (const pt of td.points) if (pt.min + k * 1440 >= domain[0] && pt.min + k * 1440 <= domain[1]) pts.push({ m: pt.min + k * 1440, ft: pt.ft });
    for (const t of td.turns) if (t.min + k * 1440 > domain[0] + 30 && t.min + k * 1440 < domain[1] - 20) turns.push({ m: t.min + k * 1440, ft: t.ft, type: t.type, at: t.at });
  }
  const lo = Math.min(0, ...pts.map((p) => p.ft));
  const hi = Math.max(1, ...pts.map((p) => p.ft));
  const ty = scaleLinear([lo, hi], [TIDE_BASE, TIDE_TOP]);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${x(p.m).toFixed(1)} ${ty(p.ft).toFixed(1)}`).join(" ");
  const area = pts.length ? `${line} L${x(pts[pts.length - 1].m).toFixed(1)} ${TIDE_BASE} L${x(pts[0].m).toFixed(1)} ${TIDE_BASE} Z` : "";
  const ticks = [];
  for (let m = Math.ceil(domain[0] / 360) * 360; m <= domain[1]; m += 360) {
    const label = m % 1440 === 0 ? WEEKDAY[new Date(Date.parse(`${day}T12:00:00Z`) + (m / 1440) * 86_400_000).getUTCDay()] : clock12(m);
    if (Math.abs(x(m) - x(nowMin)) > 28) ticks.push({ t: m, label });
  }
  ticks.push({ t: nowMin, label: "now" });
  const desc = `${spans.map((s) => `${s.p.name}: ${windWords(s.p) ?? "no wind stated"}, seas ${seasOf(s.p)}`).join(". ")}. Predicted tide: ${turns.map((t) => `${t.type === "H" ? "high" : "low"} ${t.ft.toFixed(1)} ft`).join(", ")}.`;
  return (
    <div ref={ref} data-testid="sea-lanes">
      <ChartFrame width={w} height={H} title="Wind and seas from the NWS forecast, with the predicted tide, for the next 36 hours" desc={desc} testid="sea-strip">
        {spans.map((s) => {
          const x0 = x(s.from) + 1.5;
          const x1 = x(s.to) - 1.5;
          // a period already under way: its words start right of the now line, not on it
          const tx = s.from < nowMin ? x(nowMin) + 6 : x0 + 2;
          const hiV = s.p.windHi;
          const loV = s.p.winds.reduce((m, v) => (v.lo != null && v.lo < m ? v.lo : m), s.p.windLo ?? hiV ?? 0);
          const parts = (windWords(s.p) ?? "").split(/, (?=then)/);
          return (
            <g key={`${s.p.date}${s.p.night}`}>
              {hiV != null && (
                <>
                  <rect className="nx-draw-bar" x={x0} y={wy(hiV)} width={Math.max(1, x1 - x0)} height={WIND_BASE - wy(hiV)} rx={2} style={{ fill: "var(--nx-k-conditions)", fillOpacity: 0.22 }} />
                  <rect className="nx-draw-bar" x={x0} y={wy(loV)} width={Math.max(1, x1 - x0)} height={WIND_BASE - wy(loV)} rx={2} style={{ fill: "var(--nx-k-conditions)", fillOpacity: 0.8 }} />
                </>
              )}
              <ChartMarks>
                {x1 - tx > 44 && (
                  <text className="today-strip-t" x={tx} y={14}>
                    <tspan fontWeight={500}>{s.p.name.replace(/ Night$/, " night")}</tspan>
                    {parts[0] && x1 - tx > 76 ? <tspan x={tx} dy={14}>{x1 - tx > 190 ? parts.join(", ") : parts[0]}</tspan> : null}
                  </text>
                )}
                <text className="today-strip-v" x={tx} y={SEAS_Y}>
                  {x1 - tx > 76 ? `seas ${seasOf(s.p)}` : ""}
                </text>
              </ChartMarks>
            </g>
          );
        })}
        <g data-testid="tide-curve">
          {pts.length > 1 && (
            <>
              <ChartArea d={area} family="conditions" opacity={0.12} />
              <ChartLine d={line} family="conditions" />
            </>
          )}
          <ChartMarks>
            <text className="today-strip-t" x={x(nowMin) + 6} y={TIDE_TOP - 18}>
              Tide, predicted
            </text>
            {turns.map((t) => {
              const tx = Math.min(w - 26, Math.max(26, x(t.m)));
              return (
                <g key={t.at}>
                  <circle cx={x(t.m)} cy={ty(t.ft)} r={2.6} fill="var(--nx-k-conditions)" />
                  <DirectLabel x={tx} y={ty(t.ft) - 7} anchor="middle" family="ink">
                    {t.ft.toFixed(1)}
                  </DirectLabel>
                </g>
              );
            })}
          </ChartMarks>
        </g>
        <NowLine x={x(nowMin)} y0={2} y1={TIDE_BASE} label="" />
        <TimeAxis x={x} domain={domain} y={TIDE_BASE} ticks={ticks} />
      </ChartFrame>
    </div>
  );
}

/**
 * The latest set per species as dots on one shared 0-to-limit strip: a dot's place is its share of that
 * species' action level, so a crab test and a shellfish test read against the same line. Measured values only.
 */
export function ToxinStrip({ sites, now }: { sites: TestSiteView[]; now: string }) {
  const [ref, w] = useWidth<HTMLDivElement>(640);
  const rows = sites.filter((t) => t.latest);
  const ROW = 56;
  const L = 0;
  const R = 8;
  const TOP = 18;
  const H = TOP + rows.length * ROW + 24;
  // every row runs 0 to its own limit; a value over its limit is drawn past the limit line, to scale
  const over = Math.max(1, ...rows.map((t) => Math.max(0, ...t.latest!.values_ppm.filter((v): v is number => v != null)) / (t.action ?? Infinity)));
  const xr = scaleLinear([0, Math.max(1.15, over * 1.06)], [L, w - R]);
  const lim = xr(1);
  const desc = rows.map((t) => `${t.site.species}, ${t.site.site_label}, ${dayWords(t.latest!.date, now)}${t.stale ? " (old)" : ""}: ${t.latest!.values_ppm.map((v) => (v == null ? "not detected" : `${fmt(v)} ppm`)).join(", ")}${t.action != null ? `, against the ${t.action} ppm action level` : ""}`).join("; ");
  return (
    <div ref={ref}>
      <ChartFrame width={w} height={H} title="Latest domoic acid samples for each species, against each species' action level" desc={desc} testid="toxin-strip">
        <line x1={lim} x2={lim} y1={TOP - 6} y2={H - 20} stroke="var(--nx-k-official)" strokeWidth={1.25} strokeDasharray="4 3" />
        {rows.map((t, i) => {
          const y = TOP + i * ROW + 34;
          const a = t.action ?? Math.max(1, ...t.latest!.values_ppm.filter((v): v is number => v != null));
          const n = t.latest!.values_ppm.length;
          return (
            <g key={t.site.id}>
              <line x1={L} x2={w - R} y1={y} y2={y} stroke="var(--nx-line-strong)" strokeWidth={1} />
              <text className="today-strip-t" x={L} y={y - 18}>
                <tspan fontWeight={500}>{t.site.species}</tspan>
                <tspan className={t.stale ? "today-old" : undefined}> · {dayWords(t.latest!.date, now).replace(/, \d{4}$/, "")}{t.stale ? " (old)" : ""}</tspan>
                {t.latest!.avg_ppm != null ? <tspan> · avg {fmt(t.latest!.avg_ppm)} of {t.action ?? "?"} ppm</tspan> : null}
              </text>
              <ChartMarks>
                {t.latest!.values_ppm.map((v, j) =>
                  v == null ? <circle key={j} cx={xr(0) + 5 + j * 7} cy={y} r={3.2} fill="none" stroke="var(--nx-k-measured)" strokeWidth={1.25} /> : <circle key={j} cx={xr(v / a) + (n > 1 ? (j - (n - 1) / 2) * 0.8 : 0)} cy={y} r={4} fill={t.action != null && v >= t.action ? "var(--nx-k-official)" : "var(--nx-k-measured)"} fillOpacity={0.85} />,
                )}
                {t.latest!.avg_ppm != null && <line x1={xr(t.latest!.avg_ppm / a)} x2={xr(t.latest!.avg_ppm / a)} y1={y - 9} y2={y + 9} stroke="var(--nx-ink)" strokeWidth={1.75} />}
              </ChartMarks>
            </g>
          );
        })}
        <text className="nx-k-axis-t" x={L} y={H - 4} fill="var(--nx-ink-3)" fontSize={11} fontFamily="var(--nx-font-data)">
          0
        </text>
        <DirectLabel x={lim} y={H - 4} family="official" anchor="middle">
          action limit
        </DirectLabel>
      </ChartFrame>
    </div>
  );
}

/**
 * The two bloom steps on one time axis, past left of now: C-HARM's daily probabilities in the water (violet
 * bars, muted and named "past run" when every day has gone by) and the Neritic 8-day chlorophyll outlook as one
 * span. Never added together, and the outlook is one 8-day average: drawn as a span, not a curve.
 */
export function BloomAxis({ days, today, past, outlook }: { days: BloomDay[]; today: string; past: boolean; outlook: { from: string; to: string; phrase: string; stale: boolean } | null }) {
  const [ref, w] = useWidth<HTMLDivElement>(640);
  const dated = days.filter((d) => d.date);
  const start = [dated[0]?.date, today].filter((x): x is string => !!x).sort()[0];
  const ends = [dated.at(-1)?.date, outlook?.to, addDays(today, 3)].filter((x): x is string => !!x).sort();
  const end = ends[ends.length - 1];
  const n = diffDays(start, end) + 1;
  const H = 158;
  const x = scaleLinear([0, n], [0, w]);
  const dx = (d: string) => diffDays(start, d);
  const T = 34;
  const B = 92;
  const CAP = (p: number) => B - (B - T) * Math.max(0, Math.min(1, p));
  const barW = Math.min(x(1) * 0.6, 26);
  const bars = dated.map((d) => ({ d, cx: x(dx(d.date!) + 0.5) }));
  const opacity = past ? 0.42 : 1;
  const OB = 108;
  const OH = 20;
  const ox0 = outlook ? x(dx(outlook.from)) : 0;
  const ox1 = outlook ? x(dx(outlook.to) + 1) : 0;
  const tickEvery = n > 20 ? 7 : n > 11 ? 3 : 2;
  const ticks = Array.from({ length: n }, (_, i) => i).filter((i) => i % tickEvery === 0 && Math.abs(i - dx(today)) > 1).map((i) => {
    const iso = addDays(start, i);
    const label = new Date(`${iso}T12:00:00Z`).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
    return { t: i + 0.5, label };
  });
  ticks.push({ t: dx(today) + 0.5, label: "today" });
  const desc = [dated.length ? `C-HARM ${past ? "past run" : "next days"}: ${dated.map((d) => `${d.date}: ${d.pda != null ? `${Math.round(d.pda * 100)}%` : "no value"}`).join(", ")}` : null, outlook ? `Chlorophyll outlook from ${outlook.from} to ${outlook.to}: ${outlook.phrase}. Chlorophyll, not toxin.` : null].filter(Boolean).join(". ");
  return (
    <div ref={ref}>
      <ChartFrame width={w} height={H} title="Bloom signals on one time axis, with today marked" desc={desc} testid="bloom-timeline">
        {dated.length > 0 && (
          <text className="today-strip-t" x={0} y={16}>
            <tspan fontWeight={500}>NOAA C-HARM</tspan>
            <tspan> · {past ? "past run" : "next 3 days"}</tspan>
          </text>
        )}
        <g style={{ opacity }}>
          <ChartBars family="model" baseline={B} width={barW} bars={bars.filter((b) => b.d.pda != null).map((b) => ({ x: b.cx, y: CAP(b.d.pda!) }))} />
          <ChartMarks>
            {bars.map((b) =>
              b.d.pda != null ? (
                <text key={b.d.lead} className="today-strip-v" x={b.cx} y={CAP(b.d.pda) - 4} textAnchor="middle">
                  {Math.round(b.d.pda * 100)}%
                </text>
              ) : null,
            )}
          </ChartMarks>
        </g>
        {outlook && (
          <g>
            <rect className="nx-draw-bar" x={ox0} y={OB} width={Math.max(2, ox1 - ox0)} height={OH} rx={4} style={{ fill: "var(--nx-k-model)", fillOpacity: outlook.stale ? 0.2 : 0.3 }} />
            <text className="today-strip-v" x={Math.min(ox0 + 8, w - 200)} y={OB + 14}>
              {outlook.stale ? "outlook window has passed" : ox1 - ox0 < 300 ? `${outlook.phrase} · not toxin` : `${outlook.phrase} · chlorophyll, not toxin`}
            </text>
          </g>
        )}
        <NowLine x={x(dx(today) + 0.5)} y0={22} y1={OB + OH + 4} label="" />
        <TimeAxis x={x} domain={[0, n]} y={OB + OH + 4} ticks={ticks} />
      </ChartFrame>
    </div>
  );
}
