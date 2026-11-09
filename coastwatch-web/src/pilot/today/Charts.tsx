import { useId } from "react";
import type { BiotoxinSet } from "@/pilot/contracts";
import type { BloomDay, OutlookInput } from "@/pilot/facts";
import { dayWords } from "@/pilot/facts";
import type { Tide } from "@/lib/conditions";
import { tideDay } from "@/lib/tideCurve";

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

/**
 * One timeline, two lanes that are never combined: C-HARM's daily probabilities (days 0–3 of its run)
 * and the 8-day chlorophyll outlook (one mean over its window, hatched: an indicator, not toxin).
 */
export function BloomTimeline({ days, outlook, today, old, now }: { days: BloomDay[]; outlook: OutlookInput | null; today: string; old: boolean; now: string }) {
  const id = useId();
  const dated = days.filter((d) => d.date);
  const start = [dated[0]?.date, outlook?.valid_from, today].filter((x): x is string => !!x).sort()[0];
  const end = [dated.at(-1)?.date, outlook?.valid_to, addDays(start, 3)].filter((x): x is string => !!x).sort().at(-1)!;
  const n = diffDays(start, end) + 1;
  const W = 340;
  const L = 4;
  const R = 4;
  const col = (W - L - R) / n;
  const xOf = (d: string) => L + diffDays(start, d) * col;
  const T1 = 38;
  const B1 = 104;
  const T2 = 132;
  const B2 = 160;
  const H = 184;
  const yP = (p: number) => B1 - (B1 - T1) * Math.max(0, Math.min(1, p));
  const labelEvery = n > 10 ? 2 : 1;
  const descParts = [
    ...dated.map((d) => `${dayWords(d.date!, now)}: ${d.pda != null ? `${Math.round(d.pda * 100)}% particulate domoic acid` : "no domoic acid value"}${d.pn != null ? `, ${Math.round(d.pn * 100)}% Pseudo-nitzschia bloom` : ""}`),
    outlook ? `Chlorophyll outlook ${dayWords(outlook.valid_from, now)} to ${dayWords(outlook.valid_to, now)}: ${outlook.value != null ? `about ${fmt(outlook.value)} mg per cubic metre, one 8-day mean` : "no value here"}` : "No chlorophyll outlook",
  ];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" className="today-chart block" role="img" aria-labelledby={`${id}t ${id}d`} data-testid="bloom-timeline">
      <title id={`${id}t`}>Bloom outlook timeline: C-HARM probabilities by day, then the 8-day chlorophyll outlook</title>
      <desc id={`${id}d`}>{descParts.join("; ")}</desc>
      <defs>
        <pattern id={`${id}h`} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line className="today-hatch-line" x1="0" y1="0" x2="0" y2="6" />
        </pattern>
      </defs>
      <text x={L} y={12} style={{ letterSpacing: "0.08em" }}>
        C-HARM · PROBABILITY OF CONDITIONS
      </text>
      <line className="grid" x1={L} x2={W - R} y1={B1} y2={B1} />
      <line className="grid" x1={L} x2={W - R} y1={yP(0.5)} y2={yP(0.5)} strokeDasharray="2 3" />
      <text x={W - R} y={T1 - 8} textAnchor="end">
        dashed line: 50%
      </text>
      {dated.map((d) => {
        const x0 = xOf(d.date!);
        const g = old ? "old" : "";
        return (
          <g key={d.lead} className={g}>
            {d.pn != null && <rect className="pn" x={x0 + col * 0.12} width={col * 0.76} y={yP(d.pn)} height={B1 - yP(d.pn)} rx={1.5} />}
            {d.pda != null && <rect className="pda" x={x0 + col * 0.32} width={col * 0.36} y={yP(d.pda)} height={B1 - yP(d.pda)} rx={1.5} />}
            {d.pda != null && (
              <text className="val" x={x0 + col / 2} y={yP(Math.max(d.pda, d.pn ?? 0)) - 4} textAnchor="middle">
                {Math.round(d.pda * 100)}%
              </text>
            )}
            {d.pda == null && d.pn == null && (
              <text x={x0 + col / 2} y={B1 - 6} textAnchor="middle">
                —
              </text>
            )}
          </g>
        );
      })}
      <text x={L} y={T2 - 8} style={{ letterSpacing: "0.08em" }}>
        CHLOROPHYLL OUTLOOK · NOT TOXIN
      </text>
      {outlook ? (
        <g>
          <rect x={xOf(outlook.valid_from)} width={(diffDays(outlook.valid_from, outlook.valid_to) + 1) * col} y={T2} height={B2 - T2} fill={`url(#${id}h)`} stroke="var(--nx-model-line)" rx={2} />
          <text className="val" x={xOf(outlook.valid_from) + 6} y={T2 + 18}>
            {outlook.value != null ? `≈ ${fmt(outlook.value)} mg/m³ · 8-day mean` : "no value here"}
          </text>
        </g>
      ) : (
        <text x={L} y={T2 + 18}>
          not available
        </text>
      )}
      {today >= start && today <= end && <line className="now" x1={xOf(today) + col / 2} x2={xOf(today) + col / 2} y1={T1 - 4} y2={B2 + 4} />}
      {Array.from({ length: n }, (_, i) => addDays(start, i)).map((d, i) =>
        d === today || (i % labelEvery === 0 && Math.abs(diffDays(today, d)) > 1) ? (
          <text key={d} x={xOf(d) + col / 2} y={H - 6} textAnchor="middle" style={d === today ? { fill: "var(--nx-ink)", fontWeight: 600 } : undefined}>
            {d === today ? "today" : dayWords(d, now).replace(/^\w+ /, "")}
          </text>
        ) : null,
      )}
    </svg>
  );
}

/** Today's tide drawn through NOAA's predicted highs and lows (lib/tideCurve), with the time now. */
export function TideCurve({ tides, day, nowMin }: { tides: Tide[]; day: string; nowMin: number | null }) {
  const id = useId();
  const d = tideDay(tides, day, 15);
  if (!d) return null;
  const W = 340;
  const H = 104;
  const T = 18;
  const B = 32;
  const span = Math.max(0.5, d.hi - d.lo);
  const x = (m: number) => (m / 1440) * W;
  const y = (ft: number) => T + (H - T - B) * (1 - (ft - d.lo) / span);
  const line = d.points.map((p, i) => `${i ? "L" : "M"}${x(p.min).toFixed(1)} ${y(p.ft).toFixed(1)}`).join(" ");
  const clock = (m: number) => `${Math.floor(m / 60) % 12 || 12}${m % 60 ? `:${String(m % 60).padStart(2, "0")}` : ""}${m < 720 ? "a" : "p"}`;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" className="today-chart block" role="img" aria-labelledby={`${id}t`} data-testid="tide-curve">
      <title id={`${id}t`}>{`Predicted tide today: ${d.turns.map((t) => `${t.type === "H" ? "high" : "low"} ${t.ft.toFixed(1)} ft at ${clock(t.min)}`).join(", ")}`}</title>
      <path className="tide-fill" d={`${line} L${W} ${H - B} L0 ${H - B} Z`} />
      <line className="grid" x1={0} x2={W} y1={H - 16} y2={H - 16} />
      <path className="tide" d={line} />
      {[0, 360, 720, 1080].map((m) => (
        <text key={m} x={x(m) + 2} y={H - 4}>
          {m === 0 ? "12a" : m === 720 ? "12p" : clock(m)}
        </text>
      ))}
      {d.turns.map((t) => (
        <g key={t.at}>
          <circle cx={x(t.min)} cy={y(t.ft)} r={2.6} fill="var(--nx-conditions)" />
          <text className="val" x={Math.min(W - 30, Math.max(14, x(t.min)))} y={t.type === "H" ? y(t.ft) - 6 : y(t.ft) + 13} textAnchor="middle">
            {t.ft.toFixed(1)}
          </text>
        </g>
      ))}
      {nowMin != null && <line className="now" x1={x(nowMin)} x2={x(nowMin)} y1={4} y2={H - B} />}
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
