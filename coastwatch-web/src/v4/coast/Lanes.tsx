"use client";

import { useMemo, type ReactNode } from "react";
import { compass, stationName, type Conditions, type Tide } from "@/lib/conditions";
import { datePeriods, marineMaxima, overLimits, periodFor, windChangeNote, windSteps, type DatedPeriod, type SeaLimits } from "@/lib/trip";
import { fmtPct } from "@/lib/hab";
import { dtf } from "@/lib/time";
import { FORECAST_COPY } from "@/content/copy";
import type { OutlookDayLite } from "@/lib/harborEvidence";
import { seasValue, windValue } from "@/components/coast/FishingBrief";
import { TideStrip } from "./TideStrip";
import { clockLabel, kmLabel } from "@/v4/ui/time";
import { Icon } from "@/v4/ui/Icon";
import { useWidth } from "@/v4/charts/useWidth";
import { Caps, Reading } from "./ui/Instrument";

/**
 * The decision view's time (v10, from direction C's shared time lanes in direction A's dark style):
 * the trip period's four readings first (NWS forecast, then measured, each said so), then the next 60
 * hours as lanes on one clock: the NWS periods for wind and seas against the reader's own limits, any NWS
 * advisory, the predicted tide, the latest measurements at their own times, and NOAA's model as context.
 * Every lane keeps its own source and time; a gap is drawn as a gap, never as a low value. On a phone
 * the lanes become a list of periods in reading order. Heights are fixed, so late data never moves the
 * page (the briefing's CLS budget).
 */

type Cond = { data: Conditions | null; loading: boolean; error: string | null };

const PT_PARTS = dtf("en-CA", { timeZone: "America/Los_Angeles", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
const dayIndex = (date: string, today: string) => Math.round((Date.parse(`${date}T00:00:00Z`) - Date.parse(`${today}T00:00:00Z`)) / 86_400_000);
/** Minutes after the Pacific midnight that starts `today`, for an instant. */
function instantMin(iso: string, today: string): number | null {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  const p = Object.fromEntries(PT_PARTS.formatToParts(d).map((x) => [x.type, x.value]));
  return dayIndex(`${p.year}-${p.month}-${p.day}`, today) * 1440 + Number(p.hour) * 60 + Number(p.minute);
}
/** The same for a Pacific wall-clock time ("2026-10-22T18:00" or CO-OPS's "2026-10-22 18:00"). */
function localMin(s: string, today: string): number {
  const [date, hm] = s.replace(" ", "T").split("T");
  const [h, m] = (hm ?? "00:00").split(":").map(Number);
  return dayIndex(date, today) * 1440 + h * 60 + (m || 0);
}
const clock = (min: number) => {
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
function periodName(p: DatedPeriod, saved: boolean): string {
  if (!saved) return p.name;
  if (!p.date) return "Undated period";
  const d = new Date(`${p.date}T12:00:00Z`);
  return `${dtf("en-US", { weekday: "short", month: "short", day: "numeric", timeZone: "UTC" }).format(d)}${p.night ? ", night" : ""}`;
}

type Seg = { p: DatedPeriod; from: number; to: number; wind: string | null; seas: string | null; windKt: number | null; seasFt: number | null; over: { wind: boolean; seas: boolean }; name: string };

/** Water temperature with where and when it was measured: the harbour's own gauge first, else the buoy. */
function waterReading(c: Conditions | null): { f: number; at: string; where: string } | null {
  if (c?.harbor?.status === "ok") return { f: Math.round(C_TO_F(c.harbor.data.waterC.value)), at: c.harbor.data.waterC.at, where: c.harbor.station ? `${stationName(c.harbor.station.name)} (CO-OPS), ${kmLabel(c.harbor.station.km)} away` : "the harbour (CO-OPS)" };
  if (c?.buoy.status === "ok" && c.buoy.data.waterC) return { f: Math.round(C_TO_F(c.buoy.data.waterC.value)), at: c.buoy.data.waterC.at, where: c.buoy.station ? `buoy ${c.buoy.station.id} (NDBC), ${kmLabel(c.buoy.station.km)} away` : "the buoy (NDBC)" };
  return null;
}

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

export function TimeView(props: TimeViewProps) {
  const { cond, now, today, tripDate, asSaved } = props;
  const c = cond.data;
  const [ref, w] = useWidth<HTMLDivElement>(0, 0);
  const wide = w >= 720;
  const period = periodFor(c, tripDate, today);
  // a reading from today says its time; an older one (a saved copy) says its date too
  const when = (iso: string) => {
    const full = clockLabel(iso);
    if (!now || asSaved) return full;
    const [d, ...rest] = full.split(", ");
    return d === clockLabel(now.toISOString()).split(", ")[0] && rest.length ? rest.join(", ") : full;
  };
  const data = useMemo(() => {
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
    return { nowMin, start, end, segs };
  }, [c, today, now, tripDate, props.limits, asSaved]);

  return (
    <div ref={ref} className="flex flex-col gap-5">
      <Soundings {...props} period={period} when={when} wide={wide} />
      <LimitsLine {...props} data={data} period={period} />
      {/* the lanes (or the phone's list) have a fixed height: data that arrives late fills it, never moves it */}
      <div className={wide ? "min-h-[372px]" : "min-h-[300px]"} data-testid="time-lanes" data-layout={w ? (wide ? "lanes" : "list") : "pending"}>
        {w === 0 ? null : wide ? <Lanes {...props} data={data} when={when} /> : <PeriodList {...props} data={data} when={when} />}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- the trip period's readings

function Soundings({ cond, tripDate, today, asSaved, period, when, wide }: TimeViewProps & { period: DatedPeriod | null; when: (iso: string) => string; wide: boolean }) {
  const c = cond.data;
  const max = period ? marineMaxima(period.text) : null;
  const wind = period ? windValue(period.text, max?.windKt ?? null) : null;
  const seas = period ? seasValue(period.text, max?.seasFt ?? null) : null;
  const gust = period?.text.match(/\bgusts?\s+(?:up\s+)?to\s+(\d+)\s*kt/i)?.[1];
  const windNote = period ? (windChangeNote(windSteps(period.text)) ?? (gust ? `gusts to ${gust} kt` : undefined)) : undefined;
  const buoy = c?.buoy.status === "ok" ? c.buoy : null;
  const wave = buoy?.data.waveFt ?? null;
  const water = waterReading(c);
  const fc = period ? "NWS forecast" : cond.loading ? "reading NWS…" : "no NWS period";
  const tideOk = c?.tides.status === "ok";
  return (
    <div className="flex flex-col gap-3">
      <dl className="grid grid-cols-2 gap-x-5 gap-y-5 border-t border-n-line-strong pt-4 md:grid-cols-4 md:gap-x-0 md:[&>div+div]:border-l md:[&>div+div]:border-n-line md:[&>div+div]:pl-5 md:[&>div]:pr-4">
        <Sounding caps={`Wind · ${fc}`} testId="glance-wind" value={wind} unit="kt" note={windNote && wide ? <>then {windNote.replace(/^then /, "")}</> : undefined} />
        <Sounding caps={`Seas · ${fc}`} testId="glance-seas" value={seas} unit="ft" note={period && max?.seasFt != null ? `up to ${max.seasFt} ft in the NWS text` : undefined} />
        <Sounding
          caps={wave ? "Waves · measured" : cond.loading ? "Waves · reading…" : "Waves · none nearby"}
          testId="glance-waves"
          value={wave ? wave.value.toFixed(1) : null}
          unit={wave ? `ft${buoy?.data.periodS ? ` · ${Math.round(buoy.data.periodS.value)} s` : ""}` : undefined}
          note={wave && buoy ? `${when(wave.at)} · buoy ${buoy.station ? `${buoy.station.id}, ${kmLabel(buoy.station.km)} out` : ""} · not the harbour` : c && c.buoy.status !== "ok" ? c.buoy.reason : undefined}
        />
        <Sounding caps={water ? "Water · measured" : cond.loading ? "Water · reading…" : "Water · none nearby"} testId="glance-water-t" value={water ? String(water.f) : null} unit="°F" note={water ? <span data-testid="glance-water">measured {when(water.at)} at {water.where}</span> : c?.harbor && c.harbor.status !== "ok" ? c.harbor.reason : c && c.buoy.status !== "ok" ? c.buoy.reason : undefined} />
      </dl>
      {windNote && !wide && (
        <p className="-mt-1 text-[13px] leading-snug text-n-ink-2" data-testid="glance-wind-changes">
          Wind {windNote}
        </p>
      )}
      {/* the alert, then the source of the forecast; every NWS period and the forecaster's text are in the Sea note */}
      <p className="flex flex-wrap items-baseline gap-x-3 gap-y-1 text-[13px] leading-snug text-n-ink-2">
        {c?.alerts.status === "ok" && c.alerts.data.length > 0 && (
          <span className="inline-flex items-baseline gap-1.5 text-n-conditions" data-testid="glance-alert">
            <Icon name="alert" className="h-3.5 w-3.5 self-center" /> NWS {c.alerts.data[0].event}
            {c.alerts.data[0].ends ? <span className="text-n-ink-2"> until {clockLabel(c.alerts.data[0].ends)}</span> : null}
          </span>
        )}
        <a href="#sea-h" className="nx-focus rounded-[2px] text-n-ink underline decoration-n-line-strong underline-offset-4 hover:decoration-n-ink" data-testid="glance-sea-link">
          {period ? `Marine forecast (NWS), ${periodName(period, !!asSaved)}` : cond.loading ? "Reading the NWS forecast…" : c?.marine.status === "ok" ? (asSaved ? "No NWS period saved for this day" : "No NWS period for this day yet") : "NWS zone forecast not available here"}
        </a>
        {c?.marine.status === "ok" && <span>{c.marine.data.zone}{c.marine.data.updated ? `, issued ${when(c.marine.data.updated)}` : ""}</span>}
        {!tideOk && !cond.loading && <span>No tide predictions read for these waters.</span>}
        {tripDate && today && tripDate !== today && <span>Readings are the latest measured, not the trip day&rsquo;s.</span>}
      </p>
    </div>
  );
}

function Sounding({ caps, value, unit, note, testId }: { caps: string; value: string | null; unit?: string; note?: ReactNode; testId: string }) {
  return (
    <div className="flex min-w-0 flex-col gap-1.5" data-testid={testId}>
      <dt>
        <Caps family="conditions">{caps}</Caps>
      </dt>
      <dd className="flex min-w-0 flex-col gap-1">
        <Reading value={value ?? "—"} unit={value ? unit : undefined} muted={!value} className="text-[32px] md:text-[36px]" />
        {note && <span className="text-[12.5px] leading-snug text-n-ink-2">{note}</span>}
      </dd>
    </div>
  );
}

// ---------------------------------------------------------------- the lanes (wide screens)

type Data = { nowMin: number; start: number; end: number; segs: Seg[] } | null;

function Lanes({ cond, today, asSaved, limits, outlook, runIssued, data, when }: TimeViewProps & { data: Data; when: (iso: string) => string }) {
  const c = cond.data;
  if (!data || !today)
    return (
      <div className="flex h-[372px] items-center text-[14px] text-n-ink-2" role="status">
        {cond.error ? "Conditions are unavailable right now." : "Reading NWS, NDBC and CO-OPS…"}
      </div>
    );
  const { nowMin, start, end, segs } = data;
  const span = end - start;
  const x = (m: number) => `${((Math.min(Math.max(m, start), end) - start) / span) * 100}%`;
  const wd = (a: number, b: number) => `${((Math.min(b, end) - Math.max(a, start)) / span) * 100}%`;
  const days: number[] = [];
  for (let k = Math.ceil(start / 1440); k * 1440 < end; k++) days.push(k);
  const ticks: number[] = [];
  const nowIn = nowMin >= start && nowMin <= end && !asSaved;
  for (let t = Math.ceil(start / 360) * 360; t < end; t += 360) if (t % 1440 !== 0 && !(nowIn && Math.abs(t - nowMin) < 100)) ticks.push(t);
  // a forecast cell starts at now: the hours before it are the measured past
  const fromOf = (s: Seg) => (nowIn ? Math.max(s.from, nowMin) : s.from);
  const alerts = c?.alerts.status === "ok" ? c.alerts.data : [];
  const title = start + 360 <= nowMin + 1 ? "The next 60 hours" : `60 hours from ${dayName(today, Math.round((start + 360) / 1440))}`;
  return (
    <section aria-labelledby="lanes-h" className="flex flex-col gap-2 overflow-x-clip" data-testid="lanes">
      <div className="flex items-baseline justify-between gap-4">
        <h3 id="lanes-h" className="font-serif font-medium tracking-[-0.015em] leading-[1.1] text-[19px] text-n-ink">
          {title}
        </h3>
        <span className="text-[12.5px] text-n-ink-2">every lane keeps its own source and time · shaded: before now</span>
      </div>
      <div className="relative mt-4 grid grid-cols-[124px_minmax(0,1fr)]">
        {/* the clock */}
        <div />
        <div className="relative h-7 border-b border-n-line" aria-hidden>
          {days.map((k) => (
            <span key={k} className="absolute top-0 flex h-full flex-col" style={{ left: x(k * 1440) }}>
              <span className="absolute -left-px top-0 h-full w-px bg-n-line-strong" />
              <span className="whitespace-nowrap pl-1.5 font-data text-[12px] text-n-ink">{dayName(today, k)}</span>
            </span>
          ))}
          {ticks.map((t) => (
            <span key={t} className="absolute bottom-0 -translate-x-1/2 whitespace-nowrap pb-0.5 font-data text-[10.5px] text-n-ink-3" style={{ left: x(t) }}>
              {t % 1440 === 720 ? "noon" : clockShort(t)}
            </span>
          ))}
        </div>
        <Lane label="Wind" sub="NWS forecast, kt" family="conditions" testId="lane-wind">
          {segs.length ? segs.map((s) => <PeriodCell key={`w${s.p.name}${s.p.date}`} s={s} left={x(fromOf(s))} width={wd(fromOf(s), s.to)} value={s.wind} over={s.over.wind} limit={limits.windKt} unit="kt" />) : <Empty>{cond.loading ? "Reading…" : c?.marine.status === "ok" ? "No NWS period in this window" : "No NWS zone forecast read here"}</Empty>}
        </Lane>
        <Lane label="Seas" sub="NWS forecast, ft" family="conditions" testId="lane-seas">
          {segs.map((s) => (
            <PeriodCell key={`s${s.p.name}${s.p.date}`} s={s} left={x(fromOf(s))} width={wd(fromOf(s), s.to)} value={s.seas} over={s.over.seas} limit={limits.seasFt} unit="ft" showName={false} />
          ))}
        </Lane>
        {alerts.length > 0 && (
          <Lane label="Advisory" sub="NWS" family="conditions" testId="lane-alert" h="h-8">
            {alerts.slice(0, 2).map((a, i) => {
              const from = (a.onset ? instantMin(a.onset, today) : null) ?? start;
              const to = (a.ends ? instantMin(a.ends, today) : null) ?? end;
              return (
                <span key={a.url + i} className="absolute inset-y-1.5 flex items-center overflow-hidden rounded-[2px] border border-n-conditions/60 bg-n-conditions-bg px-2" style={{ left: x(from), width: wd(from, to) }} title={a.headline}>
                  <span className="truncate text-[12px] text-n-ink">
                    {a.event}
                    {a.ends ? "" : " · no end time given"}
                  </span>
                </span>
              );
            })}
          </Lane>
        )}
        <Lane label="Tide" sub="NOAA predictions, ft" family="conditions" testId="lane-tide" h="h-[84px]">
          {c?.tides.status === "ok" ? <TideLane tides={c.tides.data} today={today} start={start} end={end} nowMin={nowIn ? nowMin : null} /> : <Empty>{cond.loading ? "Reading…" : "No tide predictions read for these waters"}</Empty>}
        </Lane>
        <Lane label="Measured" sub="latest readings" family="conditions" testId="lane-measured" h="h-12">
          <MeasuredLane c={c} today={today} x={x} when={when} loading={cond.loading} />
        </Lane>
        <Lane label="Model" sub="C-HARM, context" family="model" testId="lane-model" h="h-10">
          <ModelLane outlook={outlook} today={today} start={start} end={end} x={x} wd={wd} />
        </Lane>
        {/* now, and the shaded hours before it */}
        {nowIn && (
          <div aria-hidden className="pointer-events-none absolute bottom-0 left-[124px] right-0 top-0">
            <span className="absolute inset-y-0 left-0 bg-[rgba(5,9,13,0.45)]" style={{ width: x(nowMin) }} />
            <span className="absolute inset-y-0 w-px bg-n-ink" style={{ left: x(nowMin) }} />
            <span className="absolute -top-5 -translate-x-1/2 whitespace-nowrap rounded-[2px] bg-n-ink px-1.5 py-px font-data text-[11px] text-n-bg" style={{ left: x(nowMin) }}>
              now {clock(nowMin)}
            </span>
          </div>
        )}
      </div>
      <p className="nx-margin pl-[124px]">
        Model lane: {FORECAST_COPY.notA} {FORECAST_COPY.lowNotSafe}
        {runIssued ? ` Run issued ${runIssued} (inferred).` : ""} Tide: times are NOAA&rsquo;s predictions, the curve between them is drawn.
      </p>
    </section>
  );
}

function Lane({ label, sub, family, children, testId, h = "h-[52px]" }: { label: string; sub: string; family: "conditions" | "model"; children: ReactNode; testId: string; h?: string }) {
  return (
    <>
      <div className={`flex flex-col justify-center border-b border-n-line pr-3 ${h}`}>
        <Caps family={family}>{label}</Caps>
        <span className="truncate text-[11.5px] leading-tight text-n-ink-2">{sub}</span>
      </div>
      <div className={`relative border-b border-n-line ${h}`} data-testid={testId}>
        {children}
      </div>
    </>
  );
}

function Empty({ children }: { children: ReactNode }) {
  return <span className="absolute inset-0 flex items-center pl-2 text-[12.5px] text-n-ink-2">{children}</span>;
}

function PeriodCell({ s, left, width, value, over, limit, unit, showName = true }: { s: Seg; left: string; width: string; value: string | null; over: boolean; limit: number | null; unit: string; showName?: boolean }) {
  return (
    <span className={`absolute inset-y-1 flex min-w-0 flex-col justify-center overflow-hidden border-l border-n-line-strong px-2 ${over ? "bg-[repeating-linear-gradient(135deg,transparent_0_6px,rgba(240,138,116,0.16)_6px_7px)] shadow-[inset_0_-2px_0_var(--nx-down)]" : ""}`} style={{ left, width }} title={`${s.name}: ${s.p.text}`} data-over={over || undefined}>
      <span className="truncate font-data text-[15px] leading-tight text-n-ink">
        {value ?? <span className="text-n-ink-3">not stated</span>}
        {value && <span className="ml-0.5 text-[10.5px] text-n-ink-2">{unit}</span>}
      </span>
      <span className="truncate text-[10.5px] leading-tight text-n-ink-2">{over && limit != null ? `over your ${limit} ${unit}` : showName ? s.name : ""}</span>
    </span>
  );
}

/** NOAA's predicted highs and lows as one curve over the window (cosine between turns), each turn labelled. */
function TideLane({ tides, today, start, end, nowMin }: { tides: Tide[]; today: string; start: number; end: number; nowMin: number | null }) {
  const turns = tides.map((t) => ({ m: localMin(t.at, today), ft: t.ft, type: t.type })).sort((a, b) => a.m - b.m);
  const inWin = turns.filter((t) => t.m >= start - 720 && t.m <= end + 720);
  if (inWin.length < 2) return <Empty>Too few predictions in this window</Empty>;
  const lo = Math.min(...inWin.map((t) => t.ft));
  const hi = Math.max(...inWin.map((t) => t.ft));
  const W = 1000;
  const H = 84;
  const top = 18;
  const bot = 20;
  const X = (m: number) => ((m - start) / (end - start)) * W;
  const Y = (ft: number) => top + (1 - (ft - lo) / Math.max(0.5, hi - lo)) * (H - top - bot);
  const pts: [number, number][] = [];
  for (let i = 0; i < inWin.length - 1; i++) {
    const a = inWin[i];
    const b = inWin[i + 1];
    for (let m = Math.max(a.m, start); m <= Math.min(b.m, end); m += 10) {
      const f = (1 - Math.cos((Math.PI * (m - a.m)) / (b.m - a.m))) / 2;
      pts.push([X(m), Y(a.ft + (b.ft - a.ft) * f)]);
    }
  }
  const d = pts.map(([px, py], i) => `${i ? "L" : "M"}${px.toFixed(1)},${py.toFixed(1)}`).join(" ");
  const shown = inWin.filter((t) => t.m >= start && t.m <= end);
  const summary = shown.map((t) => `${t.type === "H" ? "high" : "low"} ${t.ft.toFixed(1)} ft at ${clock(t.m)}`).join(", ");
  return (
    <figure className="absolute inset-0" data-testid="tide-strip">
      <svg role="img" aria-label={`Predicted tide for the next 60 hours, NOAA predictions, in feet above MLLW: ${summary}.`} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="absolute inset-0 h-full w-full">
        <path d={`${d} L${pts.at(-1)![0]},${H} L${pts[0][0]},${H} Z`} fill="var(--nx-conditions)" opacity="0.08" />
        <path d={d} fill="none" stroke="var(--nx-conditions)" strokeWidth="1.6" vectorEffect="non-scaling-stroke" />
      </svg>
      {shown.map((t) => (
        <span key={t.m} className="absolute -translate-x-1/2 whitespace-nowrap font-data text-[10.5px] leading-none text-n-ink-2" style={{ left: `${(X(t.m) / W) * 100}%`, top: t.type === "H" ? `${((Y(t.ft) - 13) / H) * 100}%` : `${((Y(t.ft) + 5) / H) * 100}%` }} data-testid="tide-turn-label">
          <span className="text-n-ink">{t.ft.toFixed(1)}</span> {clockShort(t.m)}
        </span>
      ))}
      {nowMin !== null && <span className="sr-only">Now {clock(nowMin)}.</span>}
    </figure>
  );
}

function MeasuredLane({ c, today, x, when, loading }: { c: Conditions | null; today: string; x: (m: number) => string; when: (iso: string) => string; loading: boolean }) {
  const b = c?.buoy.status === "ok" ? c.buoy : null;
  const water = waterReading(c);
  const items: { at: string; text: string }[] = [];
  if (b?.data.waveFt) items.push({ at: b.data.waveFt.at, text: `waves ${b.data.waveFt.value.toFixed(1)} ft${b.data.periodS ? ` at ${Math.round(b.data.periodS.value)} s` : ""}` });
  if (b?.data.windKt) items.push({ at: b.data.windKt.at, text: `wind ${b.data.windDirDeg ? `${compass(b.data.windDirDeg.value)} ` : ""}${Math.round(b.data.windKt.value)} kt${b.data.gustKt ? `, gusts ${Math.round(b.data.gustKt.value)}` : ""}` });
  if (water) items.push({ at: water.at, text: `water ${water.f} °F` });
  if (!items.length) return <Empty>{loading ? "Reading…" : "Nothing measured near these waters just now"}</Empty>;
  return (
    <>
      {items.map((it) => {
        const m = instantMin(it.at, today);
        return m === null ? null : <span key={it.text} aria-hidden className="absolute top-1 h-2.5 w-2.5 -translate-x-1/2 rotate-45 border border-n-conditions bg-n-bg" style={{ left: x(m) }} />;
      })}
      <span className="absolute inset-x-0 bottom-1 truncate pl-2 text-[12px] text-n-ink-2">
        {b?.station ? `Buoy ${b.station.id}, ${kmLabel(b.station.km)}: ` : ""}
        {items.map((it) => `${it.text} (${when(it.at)})`).join(" · ")}
      </span>
    </>
  );
}

function ModelLane({ outlook, today, start, end, x, wd }: { outlook: OutlookDayLite[] | null | "error"; today: string; start: number; end: number; x: (m: number) => string; wd: (a: number, b: number) => string }) {
  if (outlook === null) return <Empty>Reading NOAA&rsquo;s model…</Empty>;
  if (outlook === "error") return <Empty>C-HARM could not be read</Empty>;
  const days: number[] = [];
  for (let k = Math.floor(start / 1440); k * 1440 < end; k++) days.push(k);
  return (
    <>
      {days.map((k) => {
        const date = new Date(Date.parse(`${today}T00:00:00Z`) + k * 86_400_000).toISOString().slice(0, 10);
        const o = outlook.find((d) => d.valid === date);
        const a = k * 1440;
        const b = a + 1440;
        if (!o || !o.pda)
          return (
            <span key={k} className="absolute inset-y-1 flex items-center overflow-hidden border-l border-n-line bg-[repeating-linear-gradient(135deg,transparent_0_5px,rgba(142,162,175,0.18)_5px_6px)] px-2" style={{ left: x(a), width: wd(a, b) }}>
              <span className="truncate text-[11.5px] text-n-ink-2">no run: a gap, not a low value</span>
            </span>
          );
        const nb = o.pda;
        const text = nb.here != null ? `${fmtPct(nb.here)} here` : nb.min != null && nb.max != null ? `${fmtPct(nb.min)}–${fmtPct(nb.max)} offshore` : "no value for these waters";
        return (
          <span key={k} className="absolute inset-y-1 flex items-center overflow-hidden border-l border-n-model-line bg-n-model-bg px-2" style={{ left: x(a), width: wd(a, b) }} title={`C-HARM, valid ${date}: chance particulate domoic acid exceeds 500 ng/L`}>
            <span className="truncate text-[12px] text-n-ink">
              <span className="font-data text-n-model">{text}</span> <span className="text-n-ink-2">· pDA &gt; 500 ng/L</span>
            </span>
          </span>
        );
      })}
    </>
  );
}

// ---------------------------------------------------------------- the phone's list

function PeriodList({ cond, tripDate, today, now, asSaved, outlook, limits, data }: TimeViewProps & { data: Data; when: (iso: string) => string }) {
  const c = cond.data;
  const day = tripDate ?? today;
  return (
    <section aria-labelledby="lanes-h" className="flex flex-col gap-4" data-testid="lanes">
      <h3 id="lanes-h" className="font-serif font-medium tracking-[-0.015em] leading-[1.1] text-[19px] text-n-ink">
        {tripDate && today && tripDate !== today ? "From the trip day" : "The next 60 hours"}
      </h3>
      <ol className="flex flex-col border-t border-n-line" data-testid="lane-periods">
        {!data || !data.segs.length ? (
          <li className="py-3 text-[14px] text-n-ink-2">{cond.loading ? "Reading the NWS forecast…" : c?.marine.status === "ok" ? "No NWS period in this window." : "No NWS zone forecast read here."}</li>
        ) : (
          data.segs.slice(0, 5).map((s) => (
            <li key={`${s.p.name}${s.p.date}`} className={`grid h-[52px] grid-cols-[minmax(0,1fr)_auto_auto] items-center gap-x-4 border-b border-n-line ${s.over.wind || s.over.seas ? "shadow-[inset_2px_0_0_var(--nx-down)] pl-2" : ""}`} data-testid="lane-period">
              <span className="min-w-0">
                <span className="block truncate text-[14px] text-n-ink">{s.name}</span>
                {(s.over.wind || s.over.seas) && <span className="block truncate text-[11.5px] text-n-ink-2">over your {[s.over.wind && `${limits.windKt} kt`, s.over.seas && `${limits.seasFt} ft`].filter(Boolean).join(" and ")}</span>}
              </span>
              <span className="font-data text-[15px] text-n-ink">
                {s.wind ?? "—"}
                <span className="ml-0.5 text-[10.5px] text-n-ink-2">kt</span>
              </span>
              <span className="w-[4.5rem] text-right font-data text-[15px] text-n-ink">
                {s.seas ?? "—"}
                <span className="ml-0.5 text-[10.5px] text-n-ink-2">ft</span>
              </span>
            </li>
          ))
        )}
      </ol>
      {c?.tides.status === "ok" && day && <TideStrip tides={c.tides.data} day={day} today={today} now={asSaved ? null : now} station={c.tides.station ? stationName(c.tides.station.name) : null} />}
      <p className="text-[13px] leading-snug text-n-ink-2">
        <Caps family="model">Model · context only</Caps>{" "}
        {outlook === null
          ? "Reading NOAA's model…"
          : outlook === "error"
            ? "C-HARM could not be read."
            : (() => {
                const o = outlook.find((d) => d.valid === day);
                if (!o?.pda) return "No C-HARM run covers the trip day: a gap, not a low value.";
                const nb = o.pda;
                return nb.here != null ? `C-HARM: ${fmtPct(nb.here)} chance particulate domoic acid exceeds 500 ng/L in these waters' cell. ${FORECAST_COPY.lowNotSafe}` : nb.min != null ? `C-HARM has no value for these waters; offshore cells read ${fmtPct(nb.min)}–${fmtPct(nb.max!)}, context only. ${FORECAST_COPY.lowNotSafe}` : "C-HARM has no value for these waters.";
              })()}
      </p>
    </section>
  );
}

// ---------------------------------------------------------------- the reader's own limits

function LimitsLine({ limits, onLimits, data, period }: TimeViewProps & { data: Data; period: DatedPeriod | null }) {
  const num = (v: string) => (v.trim() === "" ? null : Number(v) > 0 ? Number(v) : null);
  const ahead = data ? data.segs.filter((s) => s.to > data.nowMin) : [];
  const maxW = Math.max(-1, ...ahead.map((s) => s.windKt ?? -1));
  const maxS = Math.max(-1, ...ahead.map((s) => s.seasFt ?? -1));
  const overW = limits.windKt != null && maxW > limits.windKt ? ahead.filter((s) => s.over.wind).map((s) => s.name) : [];
  const overS = limits.seasFt != null && maxS > limits.seasFt ? ahead.filter((s) => s.over.seas).map((s) => s.name) : [];
  const has = limits.windKt != null || limits.seasFt != null;
  const box = "nx-focus min-h-11 w-16 border-b border-n-line-strong bg-transparent text-right font-data text-[18px] font-light text-n-ink outline-none focus:border-n-accent";
  return (
    <div className="flex flex-col gap-2 border-y border-n-line py-3 md:flex-row md:items-center md:justify-between md:gap-8" data-testid="glance-limits">
      <fieldset className="flex flex-wrap items-center gap-x-5 gap-y-1">
        <legend className="sr-only">Your own limits, kept on this device</legend>
        <Caps className="w-full sm:w-auto">Your limits</Caps>
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
      <p className="max-w-[56ch] text-[13.5px] leading-snug text-n-ink-2 md:text-right" aria-live="polite">
        {!has ? (
          "Set your own limits to compare them with the highest numbers in each NWS period."
        ) : overW.length || overS.length ? (
          <span className="text-n-ink" data-testid="glance-over">
            NWS states more than your own {[overW.length && `${limits.windKt} kt`, overS.length && `${limits.seasFt} ft`].filter(Boolean).join(" and ")} limit
            {` (${[...new Set([...overW, ...overS])].slice(0, 3).join(", ")})`}.
          </span>
        ) : (
          <span>{period || ahead.length ? "Within your limits in every NWS period shown." : "No NWS period to compare yet."}</span>
        )}{" "}
        Never a go or no-go from Neritic.
      </p>
    </div>
  );
}
