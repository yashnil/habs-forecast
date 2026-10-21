"use client";

import { useEffect, useId, useRef, useState } from "react";
import { motion } from "motion/react";
import { SPRING } from "@/v4/motion";
import type { Manifest } from "@/generated/schema";
import { charmRun, satelliteDays, satelliteLatest, type SatProduct } from "@/lib/layers";
import { currentsHourly, currentsMean, defaultHour, observedCells, usableHours } from "@/lib/currents";
import { dtf, pacificToday } from "@/lib/time";
import { Icon } from "@/v4/ui/Icon";
import { clockLabel, dayLabel } from "@/v4/ui/time";
import { runDays, type LayerChoice } from "./layers";
import { derivedOutlook, type DerivedIndex, type OutlookLayer } from "@/lib/derived";

const WEEKDAY = dtf("en-US", { weekday: "short", timeZone: "UTC" });
/** "Thu" for a date; an unreadable date gives "Invalid Date", as toLocaleDateString did */
const weekdayOf = (d: string) => {
  const t = new Date(`${d}T12:00:00Z`);
  return Number.isNaN(t.getTime()) ? "Invalid Date" : WEEKDAY.format(t);
};

/**
 * Time for the layer on the map, and only for a layer that has time. Every step is a frame that
 * was actually published, labelled with its own date (forecast days are marked as forecasts);
 * there is no free date picker and nothing is interpolated between frames. ← → step, Home/End jump.
 */
type Step = { key: string; label: string; top?: string; sub: string; forecast?: boolean; today?: boolean; incomplete?: boolean; choice: LayerChoice; disabled?: boolean };

export function timeSteps(m: Manifest, c: LayerChoice, now: Date | null, derived: DerivedIndex | null = null): { title: string; short?: string; steps: Step[]; current: string; latest?: LayerChoice } | null {
  if (c.kind === "outlook") {
    // one frame: the outlook is a single mean over its window, never a day-by-day series
    const o = derivedOutlook(derived);
    if (!o) return null;
    return {
      title: `Outlook issued ${dayLabel(o.issued)}: one mean over ${dayLabel(o.valid_from)}–${dayLabel(o.valid_to)}, not day by day`,
      current: "outlook",
      steps: [{ key: "outlook", label: `${dayLabel(o.valid_from)}–${dayLabel(o.valid_to)}`, top: "Next 8 days", sub: "outlook", forecast: true, choice: c }],
    };
  }
  if (c.kind === "forecast") {
    const run = charmRun(m);
    const today = now ? pacificToday(now) : null;
    const days = runDays(m, now);
    return {
      title: run ? `C-HARM run issued ${dayLabel(run.issued_date)}${run.issued_date_derived ? " (inferred)" : ""}: nowcast, then forecasts` : "C-HARM",
      short: run ? `C-HARM run ${dayLabel(run.issued_date)}${run.issued_date_derived ? " (inferred)" : ""} · nowcast, then forecasts` : "C-HARM",
      current: `lead${c.lead}`,
      steps: days.map((d) => ({
        key: `lead${d.lead}`,
        label: d.date ? dayLabel(d.date) : "—",
        top: d.date ? WEEKDAY.format(new Date(`${d.date}T12:00:00Z`)) : undefined,
        sub: d.lead === 0 ? "nowcast" : "forecast",
        forecast: d.lead > 0,
        today: !!d.date && d.date === today,
        disabled: !d.issued,
        choice: { ...c, lead: d.lead },
      })),
    };
  }
  if (c.kind === "satellite") {
    if (c.product === "multi") return null;
    const days = satelliteDays(m, c.product as SatProduct);
    const latest = satelliteLatest(m, c.product as SatProduct);
    return {
      title: `${c.product === "olci300" ? "Sentinel-3 OLCI" : "VIIRS"}: daily overpasses, or the latest clear view of each pixel`,
      short: "Daily overpasses, or the latest clear view",
      current: c.day ?? "latest",
      latest: { ...c, day: null },
      steps: [
        ...days.slice(-7).map((d) => ({ key: d.time.observed_date ?? d.layer_id, label: dayLabel(d.time.observed_date ?? ""), top: weekdayOf((d.time.observed_date ?? "").slice(0, 10)), sub: "overpass", choice: { ...c, day: d.time.observed_date ?? null } as LayerChoice })),
        ...(latest ? [{ key: "latest", label: "Latest clear", sub: "per-pixel dates", choice: { ...c, day: null } as LayerChoice }] : []),
      ],
    };
  }
  if (c.kind === "currents") {
    const hours = currentsHourly(m).slice(-12);
    const mean = currentsMean(m);
    const usable = usableHours(m);
    return {
      title: "HF radar: hourly observations",
      short: "Hourly radar observations",
      current: c.mean ? "mean" : (c.hour ?? defaultHour(m)?.layer_id.split("_").pop() ?? ""),
      latest: { ...c, hour: null, mean: false },
      steps: [
        ...hours.map((h) => {
          const id = h.layer_id.split("_").pop()!;
          const t = h.time.valid_time ?? "";
          // an hour whose coverage collapsed stays selectable, labelled with how little it saw
          const n = usable.get(h.layer_id) === false ? observedCells(h) : null;
          return { key: id, label: t ? clockLabel(t).replace(/^\w+ \d+, /, "") : id, sub: usable.get(h.layer_id) === false ? `incomplete${n != null ? ` (${n.toLocaleString("en-US")} cells)` : ""}` : t ? dayLabel(t) : "", incomplete: usable.get(h.layer_id) === false, choice: { ...c, hour: id, mean: false } as LayerChoice };
        }),
        ...(mean ? [{ key: "mean", label: "24 h mean", sub: "averaged", choice: { ...c, hour: null, mean: true } as LayerChoice }] : []),
      ],
    };
  }
  return null;
}

/** Day as a number of whole days after `from` (calendar dates). */
const daysAfter = (from: string, d: string) => Math.round((Date.parse(`${d.slice(0, 10)}T12:00:00Z`) - Date.parse(`${from.slice(0, 10)}T12:00:00Z`)) / 86_400_000);

/**
 * The outlook's horizon, drawn to scale: the day it was issued (and today), then the window its one
 * mean covers. Nothing is selectable inside the window: there is one value per pixel for all of it.
 */
export function OutlookHorizon({ layer, now, compact = false }: { layer: OutlookLayer; now: Date | null; compact?: boolean }) {
  const span = Math.max(1, daysAfter(layer.issued, layer.valid_to) + 1);
  const at = (d: string) => `${(Math.min(span, Math.max(0, daysAfter(layer.issued, d))) / span) * 100}%`;
  const today = now ? pacificToday(now) : null;
  const w0 = daysAfter(layer.issued, layer.valid_from);
  const w1 = daysAfter(layer.issued, layer.valid_to) + 1;
  return (
    <div role="group" aria-label={`Outlook horizon: issued ${dayLabel(layer.issued)}, a mean over ${dayLabel(layer.valid_from)} to ${dayLabel(layer.valid_to)}`} data-testid="timebar" className="flex min-w-0 flex-col gap-2">
      {!compact && <p className="truncate text-[13px] text-n-ink-2">Outlook: one 8-day mean, not a day-by-day forecast</p>}
      <div className="relative h-[52px]" data-testid="outlook-horizon">
        <div className="absolute inset-x-0 top-[22px] h-[3px] rounded-full bg-n-surface-3" aria-hidden />
        <div
          className="absolute top-[14px] h-[19px] rounded-md border border-n-model/60"
          style={{ left: `${(w0 / span) * 100}%`, width: `${((w1 - w0) / span) * 100}%`, backgroundImage: "repeating-linear-gradient(135deg, transparent 0 6px, rgba(194,168,245,0.28) 6px 8px)" }}
          aria-hidden
        />
        <span className="absolute top-[12px] h-[23px] w-[2px] -translate-x-1/2 rounded-full bg-n-ink" style={{ left: at(layer.issued) }} aria-hidden />
        {today && today >= layer.issued && today <= layer.valid_to && <span className="absolute top-[16px] h-[15px] w-[2px] -translate-x-1/2 bg-n-accent" style={{ left: `${((daysAfter(layer.issued, today) + 0.5) / span) * 100}%` }} aria-hidden data-testid="outlook-today" />}
        <span className="absolute left-0 top-[36px] whitespace-nowrap text-[12px] text-n-ink-2" data-testid="outlook-issued">
          Issued {dayLabel(layer.issued)}
        </span>
        <span className="absolute top-0 whitespace-nowrap text-[12px] font-semibold text-n-ink" style={{ left: at(layer.valid_from) }} data-testid="outlook-window">
          Mean {dayLabel(layer.valid_from)} → {dayLabel(layer.valid_to)}
        </span>
        <span className="absolute right-0 top-[36px] whitespace-nowrap text-[12px] text-n-ink-2">valid to {dayLabel(layer.valid_to)}</span>
      </div>
    </div>
  );
}

/**
 * `dense` (v7 dock): one short caption ("Run Oct 20 (inferred) · nowcast, then forecasts") and day steps of
 * two lines (weekday and date, then nowcast / forecast), 44 px on a desktop and 56 px on a phone.
 */
export function Timebar({ manifest, choice, onChoice, now, compact = false, dense = false, caption = true, derived = null }: { manifest: Manifest; choice: LayerChoice; onChoice: (c: LayerChoice) => void; now: Date | null; compact?: boolean; dense?: boolean; caption?: boolean; derived?: DerivedIndex | null }) {
  const t = timeSteps(manifest, choice, now, derived);
  const row = useRef<HTMLDivElement | null>(null);
  // one selected pill that slides between frames (the map itself still cuts from frame to frame);
  // a held arrow key jumps it, so fast stepping never queues motion
  const pill = `time-pill-${useId()}`;
  const [instant, setInstant] = useState(false);
  const current = t?.current;
  // the selected frame is always in view (a narrow timebar scrolls)
  useEffect(() => {
    const el = row.current?.querySelector<HTMLElement>('[aria-checked="true"]');
    if (el && row.current) row.current.scrollLeft = Math.max(0, el.offsetLeft - row.current.clientWidth / 2 + el.clientWidth / 2);
  }, [current]);
  if (!t || !t.steps.length) return null;
  const outlook = choice.kind === "outlook" ? derivedOutlook(derived) : null;
  if (outlook) return <OutlookHorizon layer={outlook} now={now} compact={compact || dense} />;
  const i = Math.max(0, t.steps.findIndex((s) => s.key === t.current));
  // dense: a short run (the forecast's four days) shows every step, so no Earlier / Later buttons (← → still step)
  const arrows = !dense || t.steps.length > 5;
  const go = (j: number) => {
    const s = t.steps[Math.max(0, Math.min(t.steps.length - 1, j))];
    if (s && !s.disabled) onChoice(s.choice);
  };
  return (
    <div
      role="group"
      aria-label="Time"
      data-testid="timebar"
      onKeyDown={(e) => {
        setInstant(e.repeat);
        if (e.key === "ArrowRight") go(i + 1);
        else if (e.key === "ArrowLeft") go(i - 1);
        else if (e.key === "Home") go(0);
        else if (e.key === "End") go(t.steps.length - 1);
        else return;
        e.preventDefault();
      }}
      className={`flex min-w-0 flex-col ${dense ? "gap-1.5" : "gap-2"}`}
    >
      {dense ? (
        <p className={caption ? "truncate text-[12.5px] leading-4 text-n-ink-3" : "sr-only"} title={t.title}>
          {t.short ?? t.title}
        </p>
      ) : (
        !compact && <p className="truncate text-[13px] text-n-ink-2">{t.title}</p>
      )}
      <div className="flex min-w-0 items-stretch gap-1.5">
        {arrows && (
        <button type="button" aria-label="Earlier" onClick={() => go(i - 1)} disabled={i === 0} className={`nx-press nx-focus grid shrink-0 place-items-center rounded-lg text-n-ink-2 hover:text-n-ink disabled:opacity-35 max-md:min-h-[56px] md:min-h-11 ${dense ? "w-11 max-md:w-11" : "w-11 border border-n-line-strong"}`}>
          <Icon name="back" className="h-5 w-5" />
        </button>
        )}
        <motion.div layoutScroll ref={row} className="relative flex min-w-0 flex-1 gap-1 overflow-x-auto [scrollbar-width:none]" role="radiogroup" aria-label={t.title}>
          {t.steps.map((s) => {
            const on = s.key === t.current;
            return (
              <button
                key={s.key}
                type="button"
                role="radio"
                aria-checked={on}
                tabIndex={on ? 0 : -1}
                disabled={s.disabled}
                onClick={() => {
                  setInstant(false);
                  onChoice(s.choice);
                }}
                data-testid={`time-${s.key}`}
                data-incomplete={s.incomplete ? "true" : undefined}
                className={`nx-focus relative isolate flex min-w-fit flex-1 shrink-0 flex-col items-center justify-center rounded-lg px-2 text-center disabled:opacity-35 max-md:min-h-[56px] ${dense ? "basis-[64px] py-0.5 md:min-h-11" : "basis-[84px] py-1 md:min-h-12"} ${
                  on ? "text-n-bg" : "nx-row bg-n-surface-2 text-n-ink hover:bg-n-surface-3"
                }`}
              >
                {on && <motion.span aria-hidden layoutId={pill} transition={instant ? { duration: 0 } : SPRING} className="absolute inset-0 -z-10 rounded-lg bg-n-ink" />}
                {dense ? (
                  <span className="whitespace-nowrap text-[13.5px] font-semibold leading-tight">
                    {s.today && <span aria-hidden className={`mr-1 inline-block h-1.5 w-1.5 -translate-y-0.5 rounded-full ${on ? "bg-n-bg" : "bg-n-accent"}`} />}
                    {s.top ? (
                      <>
                        <span className={`mr-1 font-normal ${on ? "opacity-80" : "text-n-ink-2"}`}>{s.top}</span>
                        {/* the weekday and the day of the month; the full date is the step's name */}
                        <span aria-hidden>{s.label.split(" ").pop()}</span>
                        <span className="sr-only">{s.label}</span>
                      </>
                    ) : (
                      s.label
                    )}
                  </span>
                ) : (
                  <>
                    {s.top && <span className={`text-[11.5px] leading-tight ${on ? "opacity-80" : "text-n-ink-3"}`}>{s.top}</span>}
                    <span className="whitespace-nowrap text-[14px] font-semibold leading-tight">{s.label}</span>
                  </>
                )}
                <span className={`whitespace-nowrap text-[11.5px] leading-tight ${on ? "opacity-80" : "text-n-ink-3"}`}>
                  {/* dense: today is the dot before the date (and said to a screen reader) */}
                  {s.today ? (dense ? <span className="sr-only">today · </span> : "today · ") : ""}
                  {s.sub}
                </span>
              </button>
            );
          })}
        </motion.div>
        {arrows && (
        <button type="button" aria-label="Later" onClick={() => go(i + 1)} disabled={i === t.steps.length - 1} className={`nx-press nx-focus grid shrink-0 place-items-center rounded-lg text-n-ink-2 hover:text-n-ink disabled:opacity-35 max-md:min-h-[56px] md:min-h-11 ${dense ? "w-11" : "w-11 border border-n-line-strong"}`}>
          <Icon name="chevronRight" className="h-5 w-5" />
        </button>
        )}
      </div>
    </div>
  );
}
