"use client";

import { useMemo, type ReactNode } from "react";
import { compass, stationName, type Conditions, type Tide } from "@/lib/conditions";
import { datePeriods, marineMaxima, overLimits, periodFor, type DatedPeriod, type SeaLimits } from "@/lib/trip";
import { fmtPct } from "@/lib/hab";
import { dtf } from "@/lib/time";
import { FORECAST_COPY } from "@/content/copy";
import type { OutlookDayLite } from "@/lib/harborEvidence";
import { seasValue, windValue } from "@/components/coast/FishingBrief";
import { clockLabel, kmLabel } from "@/v4/ui/time";
import { useWidth } from "@/v4/charts/useWidth";
import { Band, ChartFrame, ChartLine, ChartMarks, NowLine, scaleLinear } from "@/v4/charts/kit";
import { Pop } from "@/v4/ui/Pop";
import { Icon } from "@/v4/ui/Icon";

/**
 * The page's hero (v11): the next 60 hours on one clock, drawn with the chart kit. Wind and seas are the
 * NWS periods as blocks against the reader's own limits (a block over a limit is hatched and says so in
 * words); an advisory is a band across its stated hours; the tide is NOAA's predicted curve; the latest
 * measurements sit at their own times, past left of now; NOAA's model is context, said so, and a run whose
 * days are past is a gap, never coverage of the trip. Every lane keeps its own source. No go or no-go.
 */

const PT_PARTS = dtf("en-CA", { timeZone: "America/Los_Angeles", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
const dayIndex = (date: string, today: string) => Math.round((Date.parse(`${date}T00:00:00Z`) - Date.parse(`${today}T00:00:00Z`)) / 86_400_000);
/** Minutes after the Pacific midnight that starts `today`, for an instant. */
export function instantMin(iso: string, today: string): number | null {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  const p = Object.fromEntries(PT_PARTS.formatToParts(d).map((x) => [x.type, x.value]));
  return dayIndex(`${p.year}-${p.month}-${p.day}`, today) * 1440 + Number(p.hour) * 60 + Number(p.minute);
}
/** The same for a Pacific wall-clock time ("2026-10-22T18:00" or CO-OPS's "2026-10-22 18:00"). */
export function localMin(s: string, today: string): number {
  const [date, hm] = s.replace(" ", "T").split("T");
  const [h, m] = (hm ?? "00:00").split(":").map(Number);
  return dayIndex(date, today) * 1440 + h * 60 + (m || 0);
}
export const clock = (min: number) => {
  const t = ((min % 1440) + 1440) % 1440;
  const h = Math.floor(t / 60);
  const m = Math.round(t % 60);
  return `${h % 12 || 12}:${String(m).padStart(2, "0")} ${h < 12 ? "AM" : "PM"}`;
};
const clockShort = (min: number) => clock(min).replace(":00", "").replace(" AM", "a").replace(" PM", "p");
const WEEKDAY = dtf("en-US", { weekday: "short", timeZone: "UTC" });
const dayName = (today: string, k: number) => {
  const d = new Date(Date.parse(`${today}T12:00:00Z`) + k * 86_400_000);
  return `${WEEKDAY.format(d)} ${d.getUTCDate()}`;
};

const C_TO_F = (c: number) => (c * 9) / 5 + 32;

/** A dated label for a period: its NWS name live; once saved, its own date ("Mon, Oct 12, night"). */
export function periodName(p: DatedPeriod, saved: boolean): string {
  if (!saved) return p.name;
  if (!p.date) return "Undated period";
  const d = new Date(`${p.date}T12:00:00Z`);
  return `${dtf("en-US", { weekday: "short", month: "short", day: "numeric", timeZone: "UTC" }).format(d)}${p.night ? ", night" : ""}`;
}

type Seg = { p: DatedPeriod; from: number; to: number; wind: string | null; seas: string | null; windKt: number | null; seasFt: number | null; over: { wind: boolean; seas: boolean }; name: string };

/** Water temperature with where and when it was measured: the harbour's own gauge first, else the buoy. */
export function waterReading(c: Conditions | null): { f: number; at: string; where: string } | null {
  if (c?.harbor?.status === "ok") return { f: Math.round(C_TO_F(c.harbor.data.waterC.value)), at: c.harbor.data.waterC.at, where: c.harbor.station ? `${stationName(c.harbor.station.name)} (CO-OPS), ${kmLabel(c.harbor.station.km)} away` : "the harbour (CO-OPS)" };
  if (c?.buoy.status === "ok" && c.buoy.data.waterC) return { f: Math.round(C_TO_F(c.buoy.data.waterC.value)), at: c.buoy.data.waterC.at, where: c.buoy.station ? `buoy ${c.buoy.station.id} (NDBC), ${kmLabel(c.buoy.station.km)} away` : "the buoy (NDBC)" };
  return null;
}

export type Cond = { data: Conditions | null; loading: boolean; error: string | null };

export type TimeViewProps = {
  cond: Cond;
  now: Date | null;
  today: string | null;
  tripDate: string | null;
  limits: SeaLimits;
  onLimits: (l: SeaLimits) => void;
  /** "as saved Oct 19, 3:07 PM PDT" when the page is a saved field-pack copy */
  asSaved: string | null;
  outlook: OutlookDayLite[] | null | "error";
  runIssued: string | null;
};

type Data = { nowMin: number; start: number; end: number; segs: Seg[]; nowIn: boolean; title: string } | null;

export function TimeView(props: TimeViewProps) {
  const { cond, now, today, tripDate, asSaved } = props;
  const c = cond.data;
  const [ref, w] = useWidth<HTMLDivElement>(0, 0);
  const period = periodFor(c, tripDate, today);
  // a reading from today says its time; an older one (a saved copy) says its date too
  const when = (iso: string) => {
    const full = clockLabel(iso);
    if (!now || asSaved) return full;
    const [d, ...rest] = full.split(", ");
    return d === clockLabel(now.toISOString()).split(", ")[0] && rest.length ? rest.join(", ") : full;
  };
  const data: Data = useMemo(() => {
    if (!c || !today || !now) return null;
    const nowMin = instantMin(now.toISOString(), today) ?? 0;
    const tripK = tripDate ? dayIndex(tripDate, today) : 0;
    const start = tripK > 0 ? tripK * 1440 - 360 : nowMin - 360;
    const end = start + 66 * 60;
    const periods = c.marine.status === "ok" ? datePeriods(c.marine.data.periods, c.marine.data.updated ?? c.marine.fetchedAt ?? null) : [];
    const issued = c.marine.status === "ok" && c.marine.data.updated ? instantMin(c.marine.data.updated, today) : null;
    const segs: Seg[] = [];
    periods.forEach((p, i) => {
      if (!p.start) return;
      let from = localMin(p.start, today);
      if (i === 0 && issued !== null) from = Math.min(from, issued);
      const nx = periods.slice(i + 1).find((q) => q.start);
      const to = nx?.start ? localMin(nx.start, today) : from + 720;
      if (to <= start || from >= end) return;
      const m = marineMaxima(p.text);
      segs.push({ p, from: Math.max(from, start), to: Math.min(to, end), wind: windValue(p.text, m.windKt), seas: seasValue(p.text, m.seasFt), windKt: m.windKt, seasFt: m.seasFt, over: overLimits(m, props.limits), name: periodName(p, !!asSaved) });
    });
    const nowIn = nowMin >= start && nowMin <= end && !asSaved;
    const title = start + 360 <= nowMin + 1 ? "The next 60 hours" : `60 hours from ${dayName(today, Math.round((start + 360) / 1440))}`;
    return { nowMin, start, end, segs, nowIn, title };
  }, [c, today, now, tripDate, props.limits, asSaved]);

  return (
    <div className="flex flex-col gap-3">
      <div ref={ref} className="coast-lanes-frame" data-testid="time-lanes">
        {w > 0 ? <Lanes {...props} data={data} when={when} width={w} /> : null}
      </div>
      <LimitsLine {...props} data={data} period={period} />
    </div>
  );
}

const compactRange = (lo: string, hi: string) => `${/^\d+%$/.test(lo) && /%$/.test(hi) ? lo.slice(0, -1) : lo}–${hi}`;

function Lanes({ cond, today, asSaved, limits, outlook, runIssued, data, when, width: W }: TimeViewProps & { data: Data; when: (iso: string) => string; width: number }) {
  const c = cond.data;
  if (!data || !today)
    return (
      <div className="flex h-[360px] items-center text-[15px] text-n-ink-2" role="status">
        {cond.error ? "Conditions are unavailable right now." : "Reading NWS, NDBC and CO-OPS…"}
      </div>
    );
  const { nowMin, start, end, segs, nowIn, title } = data;
  const phone = W < 480;
  const alerts = c?.alerts.status === "ok" ? c.alerts.data.slice(0, 2) : [];
  const X = scaleLinear([start, end], [0, W]);
  const clamp = (m: number) => Math.min(Math.max(m, start), end);
  const fromOf = (a: number) => (nowIn ? Math.max(a, nowMin) : a);

  // the vertical rhythm of the lanes
  const axisY = 40;
  const laneH = phone ? 54 : 58;
  const headH = 18;
  const windY = axisY + 2;
  const seasY = windY + headH + laneH + 6;
  const bandY = seasY + headH + laneH + 6;
  const tideY = bandY + (alerts.length ? headH + 14 + 10 : 0) + 2;
  const tideH = phone ? 56 : 62;
  const measY = tideY + headH + tideH + 8;
  const modelY = measY + headH + 22 + 4;
  const H = modelY + headH + 30 + 2;

  const dayKs: number[] = [];
  for (let k = Math.ceil(start / 1440); k * 1440 < end; k++) dayKs.push(k);
  const ticks: number[] = [];
  for (let t = Math.ceil(start / 360) * 360; t < end; t += 360) if (t % 1440 !== 0 && !(nowIn && Math.abs(t - nowMin) < 100) && !(phone && t % 720 !== 0)) ticks.push(t);

  // the highest wind, seas or limit sets each lane's scale (a block never overflows its lane)
  const maxW = Math.max(limits.windKt ?? 0, 20, ...segs.map((s) => s.windKt ?? 0));
  const maxS = Math.max(limits.seasFt ?? 0, 8, ...segs.map((s) => s.seasFt ?? 0));

  const laneBlock = (kind: "wind" | "seas", y0: number) => {
    const isWind = kind === "wind";
    const limit = isWind ? limits.windKt : limits.seasFt;
    const unit = isWind ? "kt" : "ft";
    const top = y0 + headH;
    const base = top + laneH;
    const Y = scaleLinear([0, isWind ? maxW : maxS], [base, top + 2]);
    return (
      <g data-testid={`lane-${kind}`}>
        <text x={0} y={y0 + 12} className="coast-lane-head">
          {isWind ? "Wind" : "Seas"}
        </text>
        <text x={isWind ? 34 : 33} y={y0 + 12} className="coast-lane-sub">
          NWS forecast, {unit}
        </text>
        <g>
          {segs.length ? (
            segs.map((s) => {
              const a = fromOf(s.from);
              if (a >= s.to) return null;
              const v = isWind ? s.windKt : s.seasFt;
              const over = isWind ? s.over.wind : s.over.seas;
              const x0 = X(a) + 1.5;
              const x1 = X(s.to) - 1.5;
              const wid = Math.max(0, x1 - x0);
              const yt = v != null ? Y(v) : top + laneH * 0.6;
              const full = isWind ? s.wind : s.seas;
              // a narrow block says the number alone; narrower still, its words are in the list below the chart
              const digits = full?.replace(/^[A-Z]+\s*/, "") ?? null;
              const label = !full ? null : wid >= full.length * 8 + 6 ? full : digits && wid >= digits.length * 8 + 6 ? digits : "";
              return (
                <g key={`${s.p.name}${s.p.date}`} data-over={over || undefined} data-testid={over ? "lane-over" : undefined}>
                  <title>{`${s.name}: ${s.p.text}`}</title>
                  <rect x={x0} y={v != null ? yt : top + laneH * 0.6} width={wid} height={v != null ? base - yt : laneH * 0.4} className={`coast-lane-block nx-draw-bar ${over ? "is-over" : ""}`} />
                  {over && <rect x={x0} y={yt} width={wid} height={base - yt} fill="url(#coast-hatch-over)" />}
                  <rect x={x0} y={v != null ? yt : top + laneH * 0.6} width={wid} height={2} className={over ? "coast-lane-cap-over" : "coast-lane-cap"} />
                  {label !== "" && (
                    <text x={x0 + 4} y={yt - 5} className={`coast-lane-value ${over ? "is-over" : ""}`}>
                      {label ?? <tspan className="coast-lane-sub">not stated</tspan>}
                    </text>
                  )}
                </g>
              );
            })
          ) : (
            <text x={0} y={top + laneH / 2} className="coast-lane-empty">
              {cond.loading ? "Reading…" : c?.marine.status === "ok" ? "No NWS period in this window" : "No NWS zone forecast read here"}
            </text>
          )}
        </g>
        {limit != null && (
          <Band line x0={0} x1={W} y0={Y(limit)} y1={Y(limit)} family="ink" />
        )}
        {limit != null && (
          <text x={W} y={Y(limit) - 4} textAnchor="end" className="coast-lane-limit">
            your {limit} {unit}
          </text>
        )}
      </g>
    );
  };

  // the tide: NOAA's turns, a cosine between them (drawn, said so in the notes), each turn labelled
  const tide = (() => {
    if (c?.tides.status !== "ok") return { el: <text x={0} y={tideY + headH + 40} className="coast-lane-empty">{cond.loading ? "Reading…" : "No tide predictions read for these waters"}</text>, ok: false };
    const turns = (c.tides.data as Tide[]).map((t) => ({ m: localMin(t.at, today), ft: t.ft, type: t.type })).sort((a, b) => a.m - b.m);
    const inWin = turns.filter((t) => t.m >= start - 720 && t.m <= end + 720);
    if (inWin.length < 2) return { el: <text x={0} y={tideY + headH + 40} className="coast-lane-empty">Too few predictions in this window</text>, ok: false };
    const lo = Math.min(...inWin.map((t) => t.ft));
    const hi = Math.max(...inWin.map((t) => t.ft));
    const yy = scaleLinear([lo, hi], [tideY + headH + tideH - 16, tideY + headH + 16]);
    const pts: [number, number][] = [];
    for (let i = 0; i < inWin.length - 1; i++) {
      const a = inWin[i];
      const b = inWin[i + 1];
      for (let m = Math.max(a.m, start); m <= Math.min(b.m, end); m += 10) {
        const f = (1 - Math.cos((Math.PI * (m - a.m)) / (b.m - a.m))) / 2;
        pts.push([X(m), yy(a.ft + (b.ft - a.ft) * f)]);
      }
    }
    const d = pts.map(([px, py], i) => `${i ? "L" : "M"}${px.toFixed(1)},${py.toFixed(1)}`).join(" ");
    const shown = inWin.filter((t) => t.m >= start && t.m <= end);
    const summary = shown.map((t) => `${t.type === "H" ? "high" : "low"} ${t.ft.toFixed(1)} ft at ${clock(t.m)}`).join(", ");
    // labels placed left to right, a turn too near the last label is dropped from the drawing (never from the words)
    let lastRight = -1;
    const labels = shown.flatMap((t) => {
      const px = X(t.m);
      const txt = `${t.ft.toFixed(1)} ${clockShort(t.m)}`;
      const lw = txt.length * 6.1;
      const left = Math.min(Math.max(px - lw / 2, 0), W - lw);
      if (left < lastRight + 6) return [];
      lastRight = left + lw;
      return [{ t, left, txt }];
    });
    return {
      ok: true,
      el: (
        <g data-testid="tide-strip">
          <svg role="img" aria-label={`Predicted tide for the next 60 hours, NOAA predictions, in feet above MLLW: ${summary}.`} x={0} y={0} width={W} height={H} overflow="visible">
            <ChartLine d={d} family="conditions" />
            <path d={`${d} L${pts.at(-1)![0]},${tideY + headH + tideH} L${pts[0][0]},${tideY + headH + tideH} Z`} className="nx-draw-area" style={{ fill: "var(--nx-k-conditions)", fillOpacity: 0.1 }} />
            <ChartMarks>
              {labels.map(({ t, left, txt }) => (
                <text key={t.m} x={left} y={t.type === "H" ? yy(t.ft) - 6 : yy(t.ft) + 15} className="coast-tide-label" data-testid="tide-turn-label">
                  {txt}
                </text>
              ))}
            </ChartMarks>
          </svg>
        </g>
      ),
    };
  })();

  // what was measured, at its own time
  const b = c?.buoy.status === "ok" ? c.buoy : null;
  const water = waterReading(c);
  const measured: { at: string; text: string }[] = [];
  if (b?.data.waveFt) measured.push({ at: b.data.waveFt.at, text: `waves ${b.data.waveFt.value.toFixed(1)} ft${b.data.periodS ? ` at ${Math.round(b.data.periodS.value)} s` : ""}` });
  if (b?.data.windKt) measured.push({ at: b.data.windKt.at, text: `wind ${b.data.windDirDeg ? `${compass(b.data.windDirDeg.value)} ` : ""}${Math.round(b.data.windKt.value)} kt${b.data.gustKt ? `, gusts ${Math.round(b.data.gustKt.value)}` : ""}` });
  if (water) measured.push({ at: water.at, text: `water ${water.f} °F at ${water.where}` });
  const measText = measured.length ? `${b?.station ? `Buoy ${b.station.id}, ${kmLabel(b.station.km)}: ` : ""}${measured.map((it) => `${it.text} (${when(it.at)})`).join(" · ")}` : cond.loading ? "Reading…" : "Nothing measured near these waters just now";

  // the model lane: one cell per valid day the run has for the window, never a day already past
  const model = (() => {
    if (outlook === null || outlook === "error") return <text x={0} y={modelY + headH + 20} className="coast-lane-empty">{outlook === null ? "Reading NOAA's model…" : "C-HARM could not be read"}</text>;
    const allPast = outlook.length > 0 && outlook.every((d) => !!d.valid && d.valid < today);
    type Cell = { a: number; b: number; date: string; o: OutlookDayLite | null };
    const cells: Cell[] = [];
    for (let k = Math.floor(start / 1440); k * 1440 < end; k++) {
      const date = new Date(Date.parse(`${today}T00:00:00Z`) + k * 86_400_000).toISOString().slice(0, 10);
      const o = date >= today ? (outlook.find((d) => d.valid === date) ?? null) : null;
      cells.push({ a: k * 1440, b: k * 1440 + 1440, date, o: o?.pda ? o : null });
    }
    // contiguous days without a run are one gap, said once
    const spans: Cell[] = [];
    for (const c0 of cells) {
      const last = spans.at(-1);
      if (!c0.o && last && !last.o) last.b = c0.b;
      else spans.push({ ...c0 });
    }
    const top = modelY + headH + 2;
    const h = 26;
    return spans.map((c0) => {
      const a = fromOf(clamp(c0.a));
      const z = clamp(c0.b);
      if (z <= a) return null;
      const x0 = X(a) + 1;
      const wid = Math.max(0, X(z) - x0 - 1);
      if (!c0.o)
        return (
          <g key={c0.a} data-testid="model-gap">
            <title>{allPast ? "The latest NOAA run's days are past: it does not cover this trip" : "No NOAA C-HARM run for these hours"}</title>
            <rect x={x0} y={top} width={wid} height={h} fill="url(#coast-hatch-gap)" className="coast-model-gap" />
            <text x={x0 + 8} y={top + 17} className="coast-lane-sub">
              {wid >= 330 && allPast ? `past run${runIssued ? ` (issued ${runIssued})` : ""}: no run for these hours, a gap, not a low value` : wid >= 180 ? "no run: a gap, not a low value" : wid >= 90 ? "no run: a gap" : ""}
            </text>
          </g>
        );
      const nb = c0.o.pda!;
      const range = nb.min != null && nb.max != null;
      const value = nb.here != null ? fmtPct(nb.here) : range ? compactRange(fmtPct(nb.min), fmtPct(nb.max)) : null;
      const words = nb.here != null ? `${fmtPct(nb.here)} here` : range ? `${fmtPct(nb.min)}–${fmtPct(nb.max)} offshore` : "no value for these waters";
      const title = `C-HARM, valid ${c0.date}: ${words}, chance particulate domoic acid exceeds 500 ng/L`;
      return (
        <g key={c0.a} data-testid="model-cell">
          <title>{title}</title>
          <rect x={x0} y={top} width={wid} height={h} className="coast-model-cell" />
          <text x={x0 + 8} y={top + 17} className="coast-model-text" data-testid="model-cell-text">
            {wid >= 240 ? `${words} · pDA > 500 ng/L` : wid >= 130 ? words : wid >= 46 ? (value ?? words) : ""}
          </text>
        </g>
      );
    });
  })();

  const nowX = X(nowMin);
  return (
    <section aria-labelledby="lanes-h" className="relative" data-testid="lanes">
      <div className="mb-1 flex items-baseline justify-between gap-4">
        <h2 id="lanes-h" className="nx-title">
          {title}
          {asSaved ? <span className="ml-2 font-ui text-[13px] font-normal tracking-normal text-n-ink-2">{asSaved}</span> : null}
        </h2>
        <Pop
          label="How to read the lanes"
          testId="lanes-note"
          triggerClass="nx-focus inline-flex min-h-11 min-w-11 items-center justify-center text-n-ink-2 hover:text-n-ink"
          trigger={
            <>
              <Icon name="info" className="h-[18px] w-[18px]" />
              <span className="sr-only">How to read the lanes</span>
            </>
          }
        >
          <p className="text-[15px] font-semibold text-n-ink">How to read the lanes</p>
          <p className="mt-1 text-[14px] leading-[20px] text-n-ink-2">
            Every lane keeps its own source and time. Shaded: before now. Hatched block: over your own limit. Model lane: {FORECAST_COPY.notA} {FORECAST_COPY.lowNotSafe}
            {runIssued ? ` Run issued ${runIssued} (inferred).` : ""} Tide: times are NOAA&rsquo;s predictions, the curve between them is drawn.
          </p>
        </Pop>
      </div>
      <div className="relative">
        <ChartFrame width={W} height={H} title="The next 60 hours" desc={`Wind and seas from NWS periods against your limits, tide from NOAA predictions, latest measurements and NOAA's model as context, each on one time axis.`} className="coast-lanes-svg" testid="lanes-chart">
          <defs>
            <pattern id="coast-hatch-over" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <line x1="0" y1="0" x2="0" y2="7" stroke="var(--nx-down)" strokeOpacity="0.55" strokeWidth="1.6" />
            </pattern>
            <pattern id="coast-hatch-gap" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <line x1="0" y1="0" x2="0" y2="6" stroke="var(--nx-ink-3)" strokeOpacity="0.4" strokeWidth="1" />
            </pattern>
          </defs>
          {/* the shaded hours before now, under everything else */}
          {nowIn && <rect x={0} y={axisY - 2} width={nowX} height={H - axisY + 2} className="coast-lanes-past" />}
          {/* day boundaries and the clock */}
          <g className="nx-k-axis" aria-hidden>
            {dayKs.map((k) => (
              <g key={k}>
                <line className="nx-k-grid" x1={X(k * 1440)} x2={X(k * 1440)} y1={14} y2={H} />
                <text x={X(k * 1440) + 4} y={26} className="coast-lane-day">
                  {dayName(today, k)}
                </text>
              </g>
            ))}
            {ticks.map((t) => (
              <text key={t} x={X(t)} y={38} textAnchor="middle">
                {t % 1440 === 720 ? "noon" : clockShort(t)}
              </text>
            ))}
          </g>
          {nowIn && (
            <g aria-hidden data-testid="lanes-now">
              <NowLine x={nowX} y0={14} y1={H} label={`now ${clock(nowMin)}`} />
            </g>
          )}
          {laneBlock("wind", windY)}
          {laneBlock("seas", seasY)}
          {alerts.length > 0 && (
            <g data-testid="lane-alert">
              <text x={0} y={bandY + 12} className="coast-lane-head">
                Advisory
              </text>
              <text x={62} y={bandY + 12} className="coast-lane-sub">
                NWS
              </text>
              {alerts.map((a, i) => {
                const from = (a.onset ? instantMin(a.onset, today) : null) ?? start;
                const to = (a.ends ? instantMin(a.ends, today) : null) ?? end;
                const x0 = X(clamp(from));
                const wid = Math.max(0, X(clamp(to)) - x0);
                return (
                  <g key={a.url + i}>
                    <title>{a.headline}</title>
                    <rect x={x0} y={bandY + headH} width={wid} height={18} rx={3} className="coast-alert" />
                    {wid >= 110 && (
                      <text x={x0 + 8} y={bandY + headH + 13} className="coast-alert-text">
                        {a.event}
                        {a.ends ? "" : " · no end time given"}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
          )}
          <g data-testid="lane-tide">
            <text x={0} y={tideY + 12} className="coast-lane-head">
              Tide
            </text>
            <text x={30} y={tideY + 12} className="coast-lane-sub">
              NOAA predictions, ft
            </text>
            {tide.el}
          </g>
          <g data-testid="lane-measured">
            <text x={0} y={measY + 12} className="coast-lane-head">
              Measured
            </text>
            <text x={65} y={measY + 12} className="coast-lane-sub">
              latest readings
            </text>
            {measured.map((it) => {
              const m = instantMin(it.at, today);
              return m === null ? null : <rect key={it.text} x={X(clamp(m)) - 4} y={measY + headH + 4} width={8} height={8} transform={`rotate(45 ${X(clamp(m))} ${measY + headH + 8})`} className="coast-measured-mark" />;
            })}
          </g>
          <g data-testid="lane-model">
            <text x={0} y={modelY + 12} className="coast-lane-head-model">
              Model
            </text>
            <text x={45} y={modelY + 12} className="coast-lane-sub">
              C-HARM · context
            </text>
            {model}
          </g>
        </ChartFrame>
        {/* the words start after the now line, never under it */}
        <p className="coast-measured-words" title={measText} style={{ left: nowIn ? nowX + 10 : 0, top: measY + headH + 2 }} data-testid="measured-text">
          {measText}
        </p>
      </div>
      {/* the same reading as words, for a screen reader and for anyone reading the page without the chart */}
      <ol className="sr-only" data-testid="lane-periods">
        {segs.map((s) => (
          <li key={`${s.p.name}${s.p.date}`} data-testid="lane-period">
            {s.name}: wind {s.wind ?? "not stated"} kt, seas {s.seas ?? "not stated"} ft{(s.over.wind || s.over.seas) ? `, over your ${[s.over.wind && `${limits.windKt} kt`, s.over.seas && `${limits.seasFt} ft`].filter(Boolean).join(" and ")}` : ""}.
          </li>
        ))}
      </ol>
    </section>
  );
}

function LimitsLine({ limits, onLimits, data, period }: TimeViewProps & { data: Data; period: DatedPeriod | null }) {
  const num = (v: string) => (v.trim() === "" ? null : Number(v) > 0 ? Number(v) : null);
  const ahead = data ? data.segs.filter((s) => s.to > data.nowMin) : [];
  const maxW = Math.max(-1, ...ahead.map((s) => s.windKt ?? -1));
  const maxS = Math.max(-1, ...ahead.map((s) => s.seasFt ?? -1));
  const overW = limits.windKt != null && maxW > limits.windKt ? ahead.filter((s) => s.over.wind).map((s) => s.name) : [];
  const overS = limits.seasFt != null && maxS > limits.seasFt ? ahead.filter((s) => s.over.seas).map((s) => s.name) : [];
  const has = limits.windKt != null || limits.seasFt != null;
  const box = "nx-focus min-h-11 w-14 border-b border-n-line-strong bg-transparent text-right font-ui text-[20px] font-light tabular-nums text-n-ink outline-none focus:border-n-ink";
  const line: ReactNode = !has ? (
    "Set your own limits to compare them with the highest numbers in each NWS period."
  ) : overW.length || overS.length ? (
    <span className="text-n-ink" data-testid="glance-over">
      NWS states more than your own {[overW.length && `${limits.windKt} kt`, overS.length && `${limits.seasFt} ft`].filter(Boolean).join(" and ")} limit
      {` (${[...new Set([...overW, ...overS])].slice(0, 3).join(", ")})`}.
    </span>
  ) : period || ahead.length ? (
    "Within your limits in every NWS period shown."
  ) : (
    "No NWS period to compare yet."
  );
  return (
    <div className="coast-limits" data-testid="glance-limits">
      <fieldset className="flex flex-wrap items-center gap-x-6 gap-y-1">
        <legend className="sr-only">Your own limits, kept on this device</legend>
        <span className="nx-caps">Your limits</span>
        <label className="flex items-baseline gap-2 text-[14px] text-n-ink-2">
          Wind above
          <input type="text" inputMode="numeric" defaultValue={limits.windKt ?? ""} key={`w${limits.windKt == null}`} onChange={(e) => onLimits({ ...limits, windKt: num(e.target.value) })} className={box} data-testid="glance-limit-wind" aria-label="Your wind limit, knots" />
          kt
        </label>
        <label className="flex items-baseline gap-2 text-[14px] text-n-ink-2">
          Seas above
          <input type="text" inputMode="decimal" defaultValue={limits.seasFt ?? ""} key={`s${limits.seasFt == null}`} onChange={(e) => onLimits({ ...limits, seasFt: num(e.target.value) })} className={box} data-testid="glance-limit-seas" aria-label="Your seas limit, feet" />
          ft
        </label>
      </fieldset>
      <p className="max-w-[46ch] text-[14px] leading-[20px] text-n-ink-2" aria-live="polite">
        {line} <span className="text-n-ink-2">Never a go or no-go from Neritic.</span>
      </p>
    </div>
  );
}
