"use client";

import { useEffect, useRef } from "react";
import type { Manifest } from "@/generated/schema";
import { charmRun, satelliteDays, satelliteLatest, type SatProduct } from "@/lib/layers";
import { currentsHourly, currentsMean } from "@/lib/currents";
import { pacificToday } from "@/lib/time";
import { Icon } from "@/v4/ui/Icon";
import { clockLabel, dayLabel } from "@/v4/ui/time";
import { runDays, type LayerChoice } from "./layers";

/**
 * Time for the layer on the map, and only for a layer that has time. Every step is a frame that
 * was actually published, labelled with its own date (forecast days are marked as forecasts);
 * there is no free date picker and nothing is interpolated between frames. ← → step, Home/End jump.
 */
type Step = { key: string; label: string; top?: string; sub: string; forecast?: boolean; today?: boolean; choice: LayerChoice; disabled?: boolean };

export function timeSteps(m: Manifest, c: LayerChoice, now: Date | null): { title: string; steps: Step[]; current: string; latest?: LayerChoice } | null {
  if (c.kind === "forecast") {
    const run = charmRun(m);
    const today = now ? pacificToday(now) : null;
    const days = runDays(m, now);
    return {
      title: run ? `C-HARM run issued ${dayLabel(run.issued_date)}${run.issued_date_derived ? " (inferred)" : ""}: nowcast, then forecasts` : "C-HARM",
      current: `lead${c.lead}`,
      steps: days.map((d) => ({
        key: `lead${d.lead}`,
        label: d.date ? dayLabel(d.date) : "—",
        top: d.date ? new Date(`${d.date}T12:00:00Z`).toLocaleDateString("en-US", { weekday: "short", timeZone: "UTC" }) : undefined,
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
      current: c.day ?? "latest",
      latest: { ...c, day: null },
      steps: [
        ...days.slice(-7).map((d) => ({ key: d.time.observed_date ?? d.layer_id, label: dayLabel(d.time.observed_date ?? ""), top: new Date(`${(d.time.observed_date ?? "").slice(0, 10)}T12:00:00Z`).toLocaleDateString("en-US", { weekday: "short", timeZone: "UTC" }), sub: "overpass", choice: { ...c, day: d.time.observed_date ?? null } as LayerChoice })),
        ...(latest ? [{ key: "latest", label: "Latest clear", sub: "per-pixel dates", choice: { ...c, day: null } as LayerChoice }] : []),
      ],
    };
  }
  if (c.kind === "currents") {
    const hours = currentsHourly(m).slice(-12);
    const mean = currentsMean(m);
    return {
      title: "HF radar: hourly observations",
      current: c.mean ? "mean" : (c.hour ?? hours[hours.length - 1]?.layer_id.split("_").pop() ?? ""),
      latest: { ...c, hour: null, mean: false },
      steps: [
        ...hours.map((h) => {
          const id = h.layer_id.split("_").pop()!;
          const t = h.time.valid_time ?? "";
          return { key: id, label: t ? clockLabel(t).replace(/^\w+ \d+, /, "") : id, sub: t ? dayLabel(t) : "", choice: { ...c, hour: id, mean: false } as LayerChoice };
        }),
        ...(mean ? [{ key: "mean", label: "24 h mean", sub: "averaged", choice: { ...c, hour: null, mean: true } as LayerChoice }] : []),
      ],
    };
  }
  return null;
}

export function Timebar({ manifest, choice, onChoice, now, compact = false }: { manifest: Manifest; choice: LayerChoice; onChoice: (c: LayerChoice) => void; now: Date | null; compact?: boolean }) {
  const t = timeSteps(manifest, choice, now);
  const row = useRef<HTMLDivElement | null>(null);
  const current = t?.current;
  // the selected frame is always in view (a narrow timebar scrolls)
  useEffect(() => {
    const el = row.current?.querySelector<HTMLElement>('[aria-checked="true"]');
    if (el && row.current) row.current.scrollLeft = Math.max(0, el.offsetLeft - row.current.clientWidth / 2 + el.clientWidth / 2);
  }, [current]);
  if (!t || !t.steps.length) return null;
  const i = Math.max(0, t.steps.findIndex((s) => s.key === t.current));
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
        if (e.key === "ArrowRight") go(i + 1);
        else if (e.key === "ArrowLeft") go(i - 1);
        else if (e.key === "Home") go(0);
        else if (e.key === "End") go(t.steps.length - 1);
        else return;
        e.preventDefault();
      }}
      className="flex min-w-0 flex-col gap-2"
    >
      {!compact && <p className="truncate text-[13px] text-n-ink-2">{t.title}</p>}
      <div className="flex min-w-0 items-stretch gap-1.5">
        <button type="button" aria-label="Earlier" onClick={() => go(i - 1)} disabled={i === 0} className="nx-press nx-focus grid w-11 shrink-0 place-items-center rounded-lg border border-n-line-strong text-n-ink-2 disabled:opacity-35 max-md:min-h-[56px] md:min-h-11">
          <Icon name="back" className="h-5 w-5" />
        </button>
        <div ref={row} className="relative flex min-w-0 flex-1 gap-1 overflow-x-auto [scrollbar-width:none]" role="radiogroup" aria-label={t.title}>
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
                onClick={() => onChoice(s.choice)}
                data-testid={`time-${s.key}`}
                className={`nx-focus relative flex min-w-[84px] flex-1 shrink-0 basis-[84px] flex-col items-center justify-center rounded-lg px-2 py-1 text-center disabled:opacity-35 max-md:min-h-[56px] md:min-h-12 ${
                  on ? "bg-n-ink text-n-bg" : "bg-n-surface-2 text-n-ink hover:bg-n-surface-3"
                }`}
                style={s.forecast && !on ? { backgroundImage: "repeating-linear-gradient(135deg, transparent 0 6px, rgba(194,168,245,0.10) 6px 8px)" } : undefined}
              >
                {s.top && <span className={`text-[11.5px] leading-tight ${on ? "opacity-80" : "text-n-ink-3"}`}>{s.top}</span>}
                <span className="whitespace-nowrap text-[14px] font-semibold leading-tight">{s.label}</span>
                <span className={`whitespace-nowrap text-[11.5px] ${on ? "opacity-80" : "text-n-ink-3"}`}>
                  {s.today ? "today · " : ""}
                  {s.sub}
                </span>
              </button>
            );
          })}
        </div>
        <button type="button" aria-label="Later" onClick={() => go(i + 1)} disabled={i === t.steps.length - 1} className="nx-press nx-focus grid w-11 shrink-0 place-items-center rounded-lg border border-n-line-strong text-n-ink-2 disabled:opacity-35 max-md:min-h-[56px] md:min-h-11">
          <Icon name="chevronRight" className="h-5 w-5" />
        </button>
      </div>
    </div>
  );
}
