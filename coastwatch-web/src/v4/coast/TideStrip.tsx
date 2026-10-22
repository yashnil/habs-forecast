"use client";

import { useId } from "react";
import type { Tide } from "@/lib/conditions";
import { dtf } from "@/lib/time";
import { LABEL_CHAR_W, nowLine, placeLabel, tideDay, tideNow } from "@/lib/tideCurve";
import { useWidth } from "@/v4/charts/useWidth";

const PT_HM = dtf("en-US", { timeZone: "America/Los_Angeles", hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
/** Minutes after the Pacific midnight of an instant. */
const pacificMinutes = (d: Date) => {
  const [h, m] = PT_HM.format(d).split(":").map(Number);
  return h * 60 + m;
};
const clock = (min: number) => {
  const h = Math.floor(min / 60) % 24;
  const m = Math.round(min % 60);
  return `${h % 12 || 12}:${String(m).padStart(2, "0")} ${h < 12 ? "AM" : "PM"}`;
};

/**
 * The trip day's tide (v8; every tide app's first picture): NOAA CO-OPS's predicted highs and lows for
 * the station, drawn as one curve across the day with each turn labelled, and, on today, a line at now
 * with "rising" or "falling". Predictions, said so, in feet above MLLW; the stretch before the first or
 * after the last prediction in the data is dashed.
 */
export function TideStrip({ tides, day, today, now, station }: { tides: Tide[]; day: string; today: string | null; now: Date | null; station: string | null }) {
  const id = useId();
  const [ref, w] = useWidth<HTMLDivElement>();
  const d = tideDay(tides, day);
  if (!d) return null;
  // shorter on a phone, so the official position under the conditions stays on the first screen
  const H = w && w < 420 ? 62 : 76;
  const top = 18;
  const bottom = 18;
  const W = Math.max(240, w || 640);
  const span = Math.max(0.5, d.hi - d.lo);
  const x = (min: number) => (min / 1440) * W;
  const y = (ft: number) => top + (1 - (ft - d.lo) / span) * (H - top - bottom);
  const line = (pts: typeof d.points) => pts.map((p, i) => `${i ? "L" : "M"}${x(p.min).toFixed(1)},${y(p.ft).toFixed(1)}`).join(" ");
  const inData = d.points.filter((p) => p.min >= d.edgeBefore && p.min <= d.edgeAfter);
  const before = d.points.filter((p) => p.min <= d.edgeBefore);
  const after = d.points.filter((p) => p.min >= d.edgeAfter);
  const area = `${line(d.points)} L${W},${H} L0,${H} Z`;
  const isToday = !!now && day === today;
  const nowMin = isToday && now ? pacificMinutes(now) : null;
  const state = nowMin != null ? tideNow(tides, day, nowMin) : null;
  const nowFt = nowMin != null ? d.points.reduce((a, p) => (Math.abs(p.min - nowMin) < Math.abs(a.min - nowMin) ? p : a)).ft : null;
  const summary = [
    state ? `${state.rising ? "Rising" : "Falling"} now, ${state.next?.type === "H" ? "high" : "low"} ${state.next?.ft.toFixed(1)} ft at ${clock(((state.next?.min ?? 0) % 1440 + 1440) % 1440)}` : null,
    d.turns.map((t) => `${t.type === "H" ? "high" : "low"} ${t.ft.toFixed(1)} ft at ${clock(t.min)}`).join(", "),
  ]
    .filter(Boolean)
    .join(". ");
  const labels = d.turns.map((t) => {
    const tx = x(t.min);
    const label = `${t.ft.toFixed(1)} ft · ${clock(t.min)}`;
    const width = label.length * LABEL_CHAR_W;
    const { lx, anchor } = placeLabel(tx, width, W, nowMin != null ? x(nowMin) : null);
    const ly = t.type === "H" ? y(t.ft) - 7 : y(t.ft) + 15;
    const l = anchor === "start" ? lx : anchor === "end" ? lx - width : lx - width / 2;
    return { t, tx, label, lx, anchor, ly, box: { l, r: l + width, top: ly - 10, bottom: ly + 3 } };
  });
  return (
    <figure className="flex flex-col gap-1.5" data-testid="tide-strip">
      <figcaption className="flex flex-wrap items-baseline justify-between gap-x-3 text-[13px] leading-5">
        <span className="font-semibold text-n-ink">
          Tide
          {/* the station is named in the Sea note and the figure's label; a phone keeps the caption on one line */}
          {station && <span className="max-sm:hidden"> at {station}</span>}
          {state && (
            <span className="font-normal text-n-ink-2" data-testid="tide-now">
              {" "}
              · {state.rising ? "rising" : "falling"} now
            </span>
          )}
        </span>
        <span className="text-n-ink-3">
          NOAA predictions<span className="max-sm:hidden"> · ft above MLLW</span>
        </span>
      </figcaption>
      <div ref={ref} className="relative w-full">
        <svg role="img" aria-label={`Predicted tide for the day${station ? ` at ${station}` : ""}, in feet above MLLW: ${summary}.`} viewBox={`0 0 ${W} ${H}`} width="100%" height={H} className="block overflow-visible">
          <defs>
            <linearGradient id={`${id}-fill`} x1="0" x2="0" y1="0" y2="1">
              <stop offset="0" stopColor="var(--nx-conditions)" stopOpacity="0.22" />
              <stop offset="1" stopColor="var(--nx-conditions)" stopOpacity="0.02" />
            </linearGradient>
          </defs>
          {[6, 12, 18].map((h) => (
            <line key={h} x1={x(h * 60)} x2={x(h * 60)} y1={top - 6} y2={H} stroke="var(--nx-line)" />
          ))}
          <path d={area} fill={`url(#${id}-fill)`} />
          {before.length > 1 && <path d={line(before)} fill="none" stroke="var(--nx-conditions)" strokeWidth="1.6" strokeDasharray="3 3" opacity="0.6" />}
          {after.length > 1 && <path d={line(after)} fill="none" stroke="var(--nx-conditions)" strokeWidth="1.6" strokeDasharray="3 3" opacity="0.6" />}
          <path d={line(inData)} fill="none" stroke="var(--nx-conditions)" strokeWidth="2" strokeLinejoin="round" />
          {/* the now line stops short of a label it could not be moved off (a phone's narrow strip), its dot over the labels */}
          {nowMin != null && nowFt != null && nowLine(x(nowMin), top - 10, H, labels).map(([a, b]) => <line key={a} x1={x(nowMin)} x2={x(nowMin)} y1={a} y2={b} stroke="var(--nx-ink)" strokeWidth="1.2" />)}
          {labels.map(({ t, tx, label, lx, anchor, ly }) => {
            return (
            <g key={t.at}>
              <circle cx={tx} cy={y(t.ft)} r="3" fill="var(--nx-surface)" stroke="var(--nx-conditions)" strokeWidth="1.6" />
              {/* a halo in the card's colour keeps the label clear of the hour lines it crosses */}
              <text x={lx} y={ly} textAnchor={anchor} fontSize="11.5" fill="var(--nx-ink-2)" stroke="var(--nx-surface)" strokeWidth="3" paintOrder="stroke" strokeLinejoin="round" className="tabular-nums" data-testid="tide-turn-label">
                {label}
              </text>
            </g>
            );
          })}
          {nowMin != null && nowFt != null && (
            <circle data-testid="tide-now-mark" cx={x(nowMin)} cy={y(nowFt)} r="4.5" fill="var(--nx-ink)" stroke="var(--nx-surface)" strokeWidth="2" />
          )}
        </svg>
        <div aria-hidden className="mt-0.5 flex justify-between text-[11px] tabular-nums text-n-ink-3">
          <span>12 AM</span>
          <span>6 AM</span>
          <span>Noon</span>
          <span>6 PM</span>
          <span>12 AM</span>
        </div>
      </div>
    </figure>
  );
}
