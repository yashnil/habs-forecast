"use client";

import { useEffect, useRef, useState } from "react";
import type { Manifest } from "@/generated/schema";
import { charmRun, satelliteDays, satelliteLatest, type CharmVariable, type SatProduct } from "@/lib/layers";
import { currentsHourly, currentsMean, defaultHour, observedCells, usableHours } from "@/lib/currents";
import { dtf, pacificToday } from "@/lib/time";
import { Icon } from "@/v4/ui/Icon";
import { clockLabel, dayLabel } from "@/v4/ui/time";
import { runDays, type LayerChoice } from "./layers";
import { derivedOutlook, derivedSst, type DerivedIndex } from "@/lib/derived";

/** One frame every PLAY_MS while playing: long enough for the map to draw and the eye to compare. */
const PLAY_MS = 1200;
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

export function timeSteps(m: Manifest, c: LayerChoice, now: Date | null, derived: DerivedIndex | null = null, region: string | null = null): { title: string; short?: string; steps: Step[]; current: string; latest?: LayerChoice } | null {
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
      short: run ? `NOAA model run ${dayLabel(run.issued_date)}${run.issued_date_derived ? " (inferred)" : ""} · nowcast, then forecasts` : "C-HARM",
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
      current: c.mean ? "mean" : (c.hour ?? defaultHour(m, region)?.layer_id.split("_").pop() ?? ""),
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


// ------------------------------------------------------------------------------------- the time rail

const DAY_MS = 86_400_000;
const HOUR_MS = 3_600_000;
/** A calendar date as a whole day number (days since 1970-01-01). */
const dayNum = (d: string) => Math.floor(Date.parse(`${d.slice(0, 10)}T12:00:00Z`) / DAY_MS);
const dateOf = (n: number) => new Date(n * DAY_MS + DAY_MS / 2).toISOString().slice(0, 10);
const PT_CLOCK = dtf("en-US", { timeZone: "America/Los_Angeles", hour: "numeric", minute: "numeric", hourCycle: "h23" });
const PT_HOUR = dtf("en-US", { timeZone: "America/Los_Angeles", hour: "numeric" });
const PT_HOUR24 = dtf("en-US", { timeZone: "America/Los_Angeles", hour: "numeric", hourCycle: "h23" });
const PT_NOW = dtf("en-US", { timeZone: "America/Los_Angeles", weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short" });
/** Now on the Pacific calendar, in days: the whole part is today's date, the fraction the time of day. */
function nowDays(now: Date): number {
  const p = PT_CLOCK.formatToParts(now);
  const h = Number(p.find((x) => x.type === "hour")?.value ?? 0) % 24;
  const mi = Number(p.find((x) => x.type === "minute")?.value ?? 0);
  return dayNum(pacificToday(now)) + (h + mi / 60) / 24;
}

/** The four families a frame can belong to on the rail, each in its data family's colour (system.md rule 3). */
type Family = "observed" | "model" | "outlook" | "analysis" | "radar";
const FAMILY_TONE: Record<Family, string> = {
  observed: "var(--nx-measured)",
  model: "var(--nx-model)",
  outlook: "var(--nx-model)",
  analysis: "var(--nx-conditions)",
  radar: "var(--nx-conditions)",
};
type Mark = {
  key: string;
  family: Family;
  /** 0: the cell row; 1: the span row under it (the latest clear view, the outlook's one window) */
  row: 0 | 1;
  t0: number;
  t1: number;
  /** a frame of the layer on the map: one of the rail's radios */
  step?: Step;
  /** a frame of another layer: a pointer shortcut that switches the map to it */
  ctx?: LayerChoice;
  /** the layer on the map without frames of its own (sea temperature): lit, not a radio */
  on?: boolean;
  nowcast?: boolean;
  incomplete?: boolean;
  /** two families on one day share the cell row: this one's slot (of `of`) */
  slot?: { i: number; of: number };
  title: string;
};
type Group = { key: string; family: Family; t0: number; t1?: number; label: React.ReactNode; text: string; active: boolean };
type RailModel = { unit: "day" | "hour"; lo: number; hi: number; marks: Mark[]; groups: Group[]; nowAt: number | null; nowAfter: number | null; mean: Step | null };

const span = (a: string, b: string) => {
  const x = dayLabel(a);
  const y = dayLabel(b);
  if (a === b) return x;
  return x.split(" ")[0] === y.split(" ")[0] ? `${x}–${y.split(" ")[1]}` : `${x}–${y}`;
};

/**
 * Everything the rail draws, on one time axis: the satellite's overpasses and its latest clear view
 * (observed), the C-HARM run's days (model), the outlook's one window (outlook, never day by day) and
 * the sea-temperature analysis date, or, for the currents, the radar's real hours. The layer on the map
 * contributes its frames as radios; the other layers are drawn muted, as context and shortcuts.
 */
function railModel(m: Manifest, choice: LayerChoice, t: ReturnType<typeof timeSteps>, now: Date | null, derived: DerivedIndex | null, variable: CharmVariable, lead: number, only = false): RailModel | null {
  if (choice.kind === "currents") {
    if (!t) return null;
    const hours = currentsHourly(m);
    const marks: Mark[] = [];
    for (const s of t.steps) {
      if (s.key === "mean") continue;
      const l = hours.find((h) => h.layer_id.endsWith(s.key));
      const vt = l?.time.valid_time;
      if (!vt) continue;
      const h0 = Math.floor(Date.parse(vt) / HOUR_MS);
      marks.push({ key: `hr-${s.key}`, family: "radar", row: 0, t0: h0, t1: h0 + 1, step: s, incomplete: s.incomplete, title: `${clockLabel(vt)}${s.incomplete ? `, ${s.sub}` : ""}` });
    }
    if (!marks.length) return null;
    const lo = Math.min(...marks.map((x) => x.t0));
    const last = Math.max(...marks.map((x) => x.t1));
    const nowH = now ? now.getTime() / HOUR_MS : null;
    // now is drawn where it falls when it is within a day of the newest hour; further on, the rail ends at
    // the newest hour and says how far behind it is (the radar's real lag, never closed by invented hours)
    const near = nowH != null && nowH <= last + 24;
    const hi = near ? Math.max(last, nowH + 0.5) : last;
    const label = <>HF radar · observed hourly, 2 km cells</>;
    return {
      unit: "hour",
      lo,
      hi,
      marks,
      groups: [{ key: "radar", family: "radar", t0: lo, label, text: "HF radar · observed hourly, 2 km cells", active: true }],
      nowAt: near ? nowH : null,
      nowAfter: nowH != null && !near ? nowH - last : null,
      mean: t.steps.find((s) => s.key === "mean") ?? null,
    };
  }

  const marks: Mark[] = [];
  const groups: Group[] = [];
  // the satellite: each overpass, and the latest clear view as a span over the days its pixels come from
  const prod: SatProduct = choice.kind === "satellite" && choice.product !== "multi" ? choice.product : satelliteLatest(m, "olci300") ? "olci300" : "viirs750";
  const satRadio = choice.kind === "satellite" && !!t;
  const satSteps = satRadio ? t!.steps : (timeSteps(m, { kind: "satellite", product: prod, day: null }, now)?.steps ?? []);
  const latest = satelliteLatest(m, choice.kind === "satellite" && choice.product === "multi" ? "multi" : prod);
  const la = latest?.composite?.oldest_observed_date ?? latest?.time.observed_date ?? null;
  const lb = latest?.composite?.newest_observed_date ?? latest?.time.observed_date ?? null;
  for (const s of satSteps) {
    const own = satRadio ? { step: s } : { ctx: s.choice };
    if (s.key === "latest") {
      if (la && lb) marks.push({ key: "sat-latest", family: "observed", row: 1, t0: dayNum(la), t1: dayNum(lb) + 1, ...own, title: `Satellite chlorophyll, latest clear view of each pixel, ${span(la, lb)}` });
    } else if (/^\d{4}-\d{2}-\d{2}/.test(s.key)) marks.push({ key: `sat-${s.key}`, family: "observed", row: 0, t0: dayNum(s.key), t1: dayNum(s.key) + 1, ...own, title: `Satellite overpass of ${dayLabel(s.key)}` });
  }
  if (choice.kind === "satellite" && choice.product === "multi" && la && lb) marks.push({ key: "sat-multi", family: "observed", row: 1, t0: dayNum(la), t1: dayNum(lb) + 1, on: true, title: `Multi-sensor view, pixels observed ${span(la, lb)}` });
  const satMarks = marks.filter((x) => x.family === "observed");
  if (satMarks.length) {
    const text = `Satellite · observed${la && lb ? `, latest clear view ${span(la, lb)}` : ""}`;
    groups.push({ key: "observed", family: "observed", t0: Math.min(...satMarks.map((x) => x.t0)), label: text, text, active: choice.kind === "satellite" });
  }

  // the C-HARM run: the nowcast and its forecast days, each valid for its own date
  const run = charmRun(m);
  const days = runDays(m, now);
  const fcRadio = choice.kind === "forecast" && !!t;
  const fcSteps = fcRadio ? t!.steps : run ? (timeSteps(m, { kind: "forecast", variable, lead: Math.max(0, lead) }, now)?.steps ?? []) : [];
  for (const s of fcSteps) {
    if (s.disabled) continue;
    const L = s.choice.kind === "forecast" ? s.choice.lead : -1;
    const d = days.find((x) => x.lead === L)?.date;
    if (!d) continue;
    marks.push({ key: `fc-${L}`, family: "model", row: 0, t0: dayNum(d), t1: dayNum(d) + 1, nowcast: L === 0, ...(fcRadio ? { step: s } : { ctx: s.choice }), title: `C-HARM ${L === 0 ? "nowcast" : `forecast day ${L}`}, valid ${dayLabel(d)}` });
  }
  const fcMarks = marks.filter((x) => x.family === "model");
  if (run && fcMarks.length) {
    const ds = days.filter((x) => x.issued && x.date).map((x) => x.date!);
    const text = `C-HARM model · valid ${span(ds[0], ds[ds.length - 1])}, run ${dayLabel(run.issued_date)}${run.issued_date_derived ? " (inferred)" : ""}`;
    groups.push({ key: "model", family: "model", t0: Math.min(...fcMarks.map((x) => x.t0)), label: text, text, active: choice.kind === "forecast" });
  }

  // the outlook: one mean over its window, a single span (never cells: there is no day-by-day value)
  const o = derivedOutlook(derived);
  if (o) {
    const s = choice.kind === "outlook" ? t?.steps[0] : undefined;
    marks.push({ key: "outlook", family: "outlook", row: 1, t0: dayNum(o.valid_from), t1: dayNum(o.valid_to) + 1, ...(s ? { step: s } : { ctx: { kind: "outlook" } as LayerChoice }), title: `Outlook, one mean over ${span(o.valid_from, o.valid_to)}, issued ${dayLabel(o.issued)}` });
    const n = dayNum(o.valid_to) - dayNum(o.valid_from) + 1;
    const active = choice.kind === "outlook";
    const label = active ? (
      <>
        Outlook, not observed · <span data-testid="outlook-issued">Issued {dayLabel(o.issued)}</span> · <span data-testid="outlook-window">Mean {dayLabel(o.valid_from)} → {dayLabel(o.valid_to)}</span>
      </>
    ) : (
      `Outlook: one ${n}-day mean, ${span(o.valid_from, o.valid_to)}`
    );
    groups.push({ key: "outlook", family: "outlook", t0: dayNum(o.valid_from), label, text: `Outlook: one ${n}-day mean, ${span(o.valid_from, o.valid_to)}`, active });
  }

  // sea temperature: an analysis for its date (one frame, nothing to step through)
  const sst = derivedSst(derived);
  const sd = sst?.observed_dates[sst.observed_dates.length - 1];
  if (sst && sd) {
    const active = choice.kind === "sst";
    marks.push({ key: "sst", family: "analysis", row: 0, t0: dayNum(sd), t1: dayNum(sd) + 1, ...(active ? { on: true } : { ctx: { kind: "sst" } as LayerChoice }), title: `Sea temperature, analysis for ${dayLabel(sd)}` });
    if (active) groups.push({ key: "analysis", family: "analysis", t0: dayNum(sd), label: `Sea temperature · analysis for ${dayLabel(sd)}`, text: `Sea temperature · analysis for ${dayLabel(sd)}`, active });
  }

  const today = now ? nowDays(now) : null;
  // a phone draws the layer on the map alone (its frames, each a full-size target), with now
  if (only) {
    const own = marks.filter((x) => x.step || x.on);
    if (!own.length) return null;
    marks.splice(0, marks.length, ...own);
    groups.splice(0, groups.length, ...groups.filter((g) => g.active));
  }
  if (!marks.length && today == null) return null;
  let lo = marks.length ? Math.min(...marks.map((x) => x.t0)) : Math.floor(today!) - 6;
  let hi = marks.length ? Math.max(...marks.map((x) => x.t1)) : Math.floor(today!) + 2;
  if (today != null) {
    lo = Math.max(Math.min(lo, Math.floor(today)), Math.floor(today) - 14);
    hi = Math.max(hi, Math.floor(today) + 1);
  }
  const kept = marks.filter((x) => x.t1 > lo).map((x) => ({ ...x, t0: Math.max(x.t0, lo) }));
  // two families on the same day share its cell: each takes a slot, so neither is drawn over the other
  const byDay = new Map<number, Mark[]>();
  for (const x of kept) if (x.row === 0) byDay.set(x.t0, [...(byDay.get(x.t0) ?? []), x]);
  for (const list of byDay.values()) if (list.length > 1) list.forEach((x, i) => (x.slot = { i, of: list.length }));
  const fam = (g: Group) => kept.filter((x) => x.family === g.family);
  const placed = groups.map((g) => ({ ...g, t0: Math.max(g.t0, lo), t1: Math.max(...fam(g).map((x) => x.t1), g.t0 + 1) })).sort((a, b) => a.t0 - b.t0);
  // a label never starts inside the frames of the family named before it (the outlook's window begins
  // inside the C-HARM run's days): it waits for that family's last frame
  for (let k = 1; k < placed.length; k++) if (placed[k].t0 < placed[k - 1].t1! && placed[k - 1].t1! < hi) placed[k].t0 = placed[k - 1].t1!;
  return { unit: "day", lo, hi, marks: kept, groups: placed, nowAt: today, nowAfter: null, mean: null };
}

/**
 * v10 time rail (direction A): the time instrument across the bottom of Explore. One axis of real time
 * with a hard "now" line (the future faintly hatched); the satellite's observed days in teal, the C-HARM
 * run's days in violet, the outlook as one dashed span; the currents on an axis of real hours.
 *
 * Only frames that were published are drawn and selectable (radios): ← → step, Home/End jump, space plays.
 * A finger or pointer can scrub along the rail: it snaps to the nearest real frame (a cut, never a blend).
 * The other layers' frames are drawn muted as context; a tap on one switches the map to that layer.
 *
 * v10 playback honesty: `shown` is the frame whose data is on the map, `pending` the frame asked for whose
 * data has not arrived. The checked frame (and every label) stays on the shown frame until the new one's
 * data is in; the pending frame says it is loading after 400 ms; playback waits for it and asks for the
 * next real frame ahead (`onPrefetch`), so each step cuts to data already loaded.
 */
export function TimeRail({
  manifest,
  choice,
  onChoice,
  now,
  derived = null,
  region = null,
  shown = null,
  pending = null,
  onPrefetch,
  compact = false,
  caption = null,
  variable = "particulate_domoic",
  lead = 1,
}: {
  manifest: Manifest;
  choice: LayerChoice;
  onChoice: (c: LayerChoice) => void;
  now: Date | null;
  derived?: DerivedIndex | null;
  region?: string | null;
  shown?: string | null;
  pending?: string | null;
  onPrefetch?: (c: LayerChoice) => void;
  /** phone: no lane labels over the cells (the active one is said above), sparse day labels, taller targets */
  compact?: boolean;
  /** the layer's own date line and anything wrong with it (Provenance), under the rail */
  caption?: React.ReactNode;
  variable?: CharmVariable;
  lead?: number;
}) {
  const t0 = timeSteps(manifest, choice, now, derived, region);
  const t = t0 && t0.steps.length ? (shown && pending && t0.steps.some((s) => s.key === shown) ? { ...t0, current: shown } : t0) : null;
  const rail = railModel(manifest, choice, t, now, derived, variable, lead, compact);
  const [playing, setPlaying] = useState(false);
  const live = useRef({ t, onChoice, pending, onPrefetch });
  live.current = { t, onChoice, pending, onPrefetch };
  useEffect(() => {
    if (!playing) return;
    const ahead = (k: number) => {
      const { t: tt, onPrefetch: pre } = live.current;
      const usable = tt?.steps.filter((s) => !s.disabled) ?? [];
      const at = usable.findIndex((s) => s.key === tt?.current);
      if (pre && usable.length > 1) pre(usable[(at + k) % usable.length].choice);
    };
    ahead(1);
    const tick = window.setInterval(() => {
      const { t: tt, onChoice: pick, pending: wait } = live.current;
      if (!tt || document.hidden) return setPlaying(false);
      // the frame asked for has not arrived: hold the one shown (and its label), never skip ahead of the data
      if (wait) return;
      const usable = tt.steps.filter((s) => !s.disabled);
      if (usable.length < 2) return setPlaying(false);
      const at = usable.findIndex((s) => s.key === tt.current);
      pick(usable[(at + 1) % usable.length].choice);
      ahead(2);
    }, PLAY_MS);
    return () => window.clearInterval(tick);
  }, [playing]);
  // a new layer kind stops playback (its frames are another series)
  const kind = choice.kind;
  useEffect(() => setPlaying(false), [kind]);
  const track = useRef<HTMLDivElement | null>(null);
  const [w, setW] = useState(0);
  const hasRail = !!rail;
  useEffect(() => {
    const el = track.current;
    if (!el) return;
    const f = () => setW(el.clientWidth);
    f();
    const ro = new ResizeObserver(f);
    ro.observe(el);
    return () => ro.disconnect();
  }, [hasRail]);
  const scroller = useRef<HTMLDivElement | null>(null);
  // the checked frame is always in view (a phone's rail scrolls)
  const checkedKey = t?.current;
  useEffect(() => {
    const sc = scroller.current;
    const el = sc?.querySelector<HTMLElement>('[aria-checked="true"]');
    if (sc && el && sc.scrollWidth > sc.clientWidth) sc.scrollLeft = Math.max(0, el.offsetLeft - sc.clientWidth / 2 + el.clientWidth / 2);
  }, [checkedKey, w]);
  const scrubbing = useRef(false);
  if (!rail) return null;

  const { lo, hi, marks, groups, unit } = rail;
  const X = (u: number) => ((u - lo) / (hi - lo)) * 100;
  const steps = t?.steps ?? [];
  const i = Math.max(0, steps.findIndex((s) => s.key === t?.current));
  const go = (j: number) => {
    setPlaying(false);
    const s = steps[Math.max(0, Math.min(steps.length - 1, j))];
    if (s && !s.disabled) onChoice(s.choice);
  };
  const playable = steps.filter((s) => !s.disabled).length > 1;
  // a pointer on the rail: the nearest real frame of the layer on the map, in the row under the pointer
  const radios = marks.filter((x) => x.step && !x.step.disabled);
  const scrubTo = (clientX: number, clientY: number) => {
    const el = track.current;
    if (!el || !radios.length) return;
    const r = el.getBoundingClientRect();
    const u = lo + ((clientX - r.left) / r.width) * (hi - lo);
    const row = clientY - r.top > cellH ? 1 : 0;
    const pool = radios.some((x) => x.row === row) ? radios.filter((x) => x.row === row) : radios;
    let best = pool[0];
    for (const x of pool) if (Math.abs((x.t0 + x.t1) / 2 - u) < Math.abs((best.t0 + best.t1) / 2 - u)) best = x;
    if (best.step && best.step.key !== t?.current) {
      setPlaying(false);
      onChoice(best.step.choice);
    }
  };
  const cellH = compact ? 56 : 28;
  const spanH = compact ? 12 : 18;
  const pxPer = w / Math.max(1, hi - lo);
  const visH = compact ? 22 : 16;
  // a phone: the spans (the latest clear view, the outlook's window) are chips beside the rail, each a full
  // target; the rail scrolls sideways when its frames would be narrower than a finger
  const chips = compact ? marks.filter((x) => x.step && x.row === 1) : [];
  const minPx = compact ? Math.ceil((hi - lo) * 46) : 0;
  const nowIso = now ? PT_NOW.format(now) : null;

  // the axis: days (weekday and date, then sparser as the rail narrows) or hours (with the date at midnight)
  const ticks: { u: number; text: string; now?: boolean }[] = [];
  if (unit === "day") {
    const today = now ? Math.floor(nowDays(now)) : null;
    const every = pxPer >= 22 ? 1 : 2;
    for (let d = Math.ceil(lo); d < hi; d++) {
      const date = dateOf(d);
      const isToday = d === today;
      if (!isToday && (d - Math.ceil(lo)) % every) continue;
      ticks.push({ u: d + 0.5, now: isToday, text: pxPer >= 50 || (isToday && pxPer >= 34) ? `${weekdayOf(date)} ${date.slice(8).replace(/^0/, "")}` : date.slice(8).replace(/^0/, "") });
    }
  } else {
    const every = pxPer >= 46 ? 1 : pxPer >= 24 ? 2 : pxPer >= 14 ? 3 : 6;
    for (let h = Math.ceil(lo); h < hi; h++) {
      const at = new Date(h * HOUR_MS);
      const h24 = Number(PT_HOUR24.format(at)) % 24;
      if (h24 % every) continue;
      ticks.push({ u: h + 0.5, text: h24 === 0 ? dayLabel(pacificToday(at)) : PT_HOUR.format(at).replace(" ", "").toLowerCase() });
    }
  }
  const activeGroup = groups.find((g) => g.active) ?? null;
  const MEAN_W = 76;

  return (
    <div
      role="group"
      aria-label="Time"
      data-testid={t ? "timebar" : "time-rail"}
      data-unit={unit}
      onKeyDown={(e) => {
        if (!t) return;
        if (e.key === "ArrowRight") go(i + 1);
        else if (e.key === "ArrowLeft") go(i - 1);
        else if (e.key === "Home") go(0);
        else if (e.key === "End") go(steps.length - 1);
        else if (e.key === " " && playable && !(e.target instanceof HTMLButtonElement && e.target.getAttribute("role") !== "radio")) setPlaying((v) => !v);
        else return;
        e.preventDefault();
      }}
      className="nx-rail flex min-w-0 flex-col gap-1"
    >
      {compact && activeGroup && (
        <p className="flex min-w-0 items-baseline justify-between gap-3 text-[12px] leading-4">
          <span className="truncate" style={{ color: FAMILY_TONE[activeGroup.family] }} title={activeGroup.text}>
            {activeGroup.label}
          </span>
          {rail.nowAfter != null && (
            <span className="nx-data shrink-0 text-[11px] text-n-ink-2" data-testid="rail-now-after">
              now +{Math.round(rail.nowAfter)} h →
            </span>
          )}
        </p>
      )}
      <div className="flex min-w-0 items-center gap-3">
        {playable ? (
          <button
            type="button"
            aria-label={playing ? "Pause" : "Play the frames in turn"}
            aria-pressed={playing}
            title={playing ? "Pause (space)" : "Play (space)"}
            onClick={() => setPlaying((v) => !v)}
            data-testid="time-play"
            className={`nx-press nx-focus grid shrink-0 place-items-center rounded-full border ${compact ? "h-14 w-14" : "h-11 w-11"} ${playing ? "border-n-ink bg-n-ink text-n-bg" : "border-n-line-strong text-n-ink hover:border-n-ink-2"}`}
          >
            <Icon name={playing ? "pause" : "play"} className="h-4 w-4" />
          </button>
        ) : (
          <span aria-hidden className={`grid shrink-0 place-items-center rounded-full border border-n-line text-n-ink-3 opacity-60 ${compact ? "h-14 w-14" : "h-11 w-11"}`}>
            <Icon name="clock" className="h-4 w-4" />
          </span>
        )}
        <div className="relative min-w-0 flex-1">
          {/* the families on the rail, each named at the start of its frames */}
          {!compact && (
            <div className="flex gap-2" aria-hidden={t ? undefined : true}>
              <div className="relative h-[18px] min-w-0 flex-1">
                {groups.map((g, k) => {
                  const next = groups[k + 1];
                  const left = Math.min(X(g.t0), 100);
                  const room = (next ? X(next.t0) : 100) - left;
                  const end = !next && left > 62;
                  return (
                    <span
                      key={g.key}
                      className={`absolute top-0 truncate text-[11.5px] leading-[16px] ${g.active ? "" : "opacity-70"}`}
                      style={{ color: FAMILY_TONE[g.family], ...(end ? { right: 0, maxWidth: `${100 - left + 30}%` } : { left: `${left}%`, maxWidth: `calc(${room}% - 10px)` }) }}
                      title={g.text}
                      data-testid={`rail-label-${g.key}`}
                    >
                      {g.label}
                    </span>
                  );
                })}
                {rail.nowAfter != null && (
                  <span className="nx-data absolute right-0 top-0 whitespace-nowrap text-[11px] leading-[16px] text-n-ink-2" data-testid="rail-now-after" title="Now is this long after the newest radar hour: the radar's real lag, never filled">
                    now is {Math.round(rail.nowAfter)} h later →
                  </span>
                )}
              </div>
              {chips.length + (rail.mean ? 1 : 0) > 0 && <span className="shrink-0" style={{ width: (chips.length + (rail.mean ? 1 : 0)) * (MEAN_W + 8) - 8 }} />}
            </div>
          )}
          <div role={t ? "radiogroup" : undefined} aria-label={t?.title} className="flex min-w-0 items-start gap-2" data-testid={choice.kind === "outlook" ? "outlook-horizon" : undefined}>
            <div ref={scroller} className={`min-w-0 flex-1 ${compact ? "overflow-x-auto overscroll-x-contain [scrollbar-width:none]" : ""}`}>
            <div className="relative" style={{ width: compact ? `max(100%, ${minPx}px)` : "100%" }}>
            <div
              ref={track}
              className={`nx-rail-track relative ${compact ? "touch-pan-x" : "touch-pan-y"}`}
              style={{ height: cellH + spanH }}
              onPointerDown={(e) => {
                // a finger on a phone scrolls the rail and taps a frame; a pointer (or a finger on a wide screen) scrubs
                if (compact && e.pointerType !== "mouse") return;
                if (e.button !== 0 || (e.target as HTMLElement).closest("[data-ctx]")) return;
                scrubbing.current = true;
                scrubTo(e.clientX, e.clientY);
              }}
              onPointerMove={(e) => scrubbing.current && e.buttons && scrubTo(e.clientX, e.clientY)}
              onPointerUp={() => (scrubbing.current = false)}
              onPointerCancel={() => (scrubbing.current = false)}
              onPointerLeave={() => (scrubbing.current = false)}
            >
              {/* the axis' day (or hour) boundaries */}
              <span aria-hidden className="absolute inset-x-0 border-t border-n-line" style={{ top: cellH / 2 }} />
              {unit === "day" &&
                Array.from({ length: Math.max(0, Math.ceil(hi) - Math.ceil(lo) + 1) }, (_, k) => Math.ceil(lo) + k).map((d) => (
                  <span key={d} aria-hidden className="absolute w-px bg-n-line-strong" style={{ left: `${X(d)}%`, top: cellH / 2 - 4, height: 8 }} />
                ))}
              {/* the future, faintly hatched: nothing past now is observed */}
              {rail.nowAt != null && <span aria-hidden className="nx-rail-future absolute inset-y-0 right-0" style={{ left: `${X(rail.nowAt)}%` }} />}
              {marks.map((mk) => {
                const left = X(mk.t0);
                const width = X(mk.t1) - left;
                const tone = FAMILY_TONE[mk.family];
                const top = mk.row === 0 ? 0 : cellH;
                const h = mk.row === 0 ? cellH : spanH;
                const on = !!(mk.step && mk.step.key === t?.current) || !!mk.on;
                const loading = !!mk.step && !!pending && mk.step.key === pending && mk.step.key !== t?.current;
                const slot = mk.slot;
                const vis =
                  mk.row === 0 ? (
                    <span
                      aria-hidden
                      className={`nx-rail-cell absolute ${mk.nowcast === false || (!mk.nowcast && mk.family === "model") ? "nx-rail-fc" : ""} ${mk.incomplete ? "nx-rail-incomplete" : ""}`}
                      style={{
                        left: 1,
                        right: 1,
                        top: (cellH - visH) / 2 + (slot ? (visH / slot.of) * slot.i : 0),
                        height: slot ? visH / slot.of - 1 : visH,
                        ["--tone" as string]: tone,
                        opacity: mk.step || mk.on ? (on ? 1 : 0.62) : 0.32,
                        boxShadow: on ? "0 0 0 2px var(--nx-bg), 0 0 0 3.5px var(--nx-ink)" : undefined,
                      }}
                    />
                  ) : (
                    <span
                      aria-hidden
                      className={`nx-rail-span absolute ${mk.family === "outlook" ? "nx-rail-dashed" : ""}`}
                      style={{ left: 1, right: 1, top: spanH / 2 - 3, height: 6, ["--tone" as string]: tone, opacity: mk.step || mk.on ? (on ? 1 : 0.7) : 0.4, boxShadow: on ? "0 0 0 1.5px var(--nx-bg), 0 0 0 3px var(--nx-ink)" : undefined }}
                    />
                  );
                const box: React.CSSProperties = { left: `${left}%`, width: `${width}%`, top, height: h };
                if (mk.step && chips.includes(mk)) return <span key={mk.key} className="absolute" style={box} title={mk.title}>{vis}</span>;
                if (mk.step) {
                  const s = mk.step;
                  return (
                    <button
                      key={mk.key}
                      type="button"
                      role="radio"
                      aria-checked={on}
                      aria-busy={loading || undefined}
                      data-loading={loading ? "true" : undefined}
                      data-incomplete={s.incomplete ? "true" : undefined}
                      tabIndex={on ? 0 : -1}
                      disabled={s.disabled}
                      title={mk.title}
                      onClick={() => {
                        setPlaying(false);
                        if (s.key !== t?.current) onChoice(s.choice);
                      }}
                      data-testid={`time-${s.key}`}
                      className={`nx-focus absolute ${mk.family === "outlook" && on ? "" : ""}`}
                      style={box}
                      {...(mk.family === "outlook" ? { "data-outlook": "true" } : {})}
                    >
                      {vis}
                      <span className="absolute inset-0 overflow-hidden">
                        <span className="sr-only">
                          {s.top ? `${s.top} ` : ""}
                          {s.label}
                          {s.today ? " · today" : ""} · {s.sub}
                        </span>
                      </span>
                      {/* the frame asked for, still loading: said after 400 ms (a fast frame never flashes it) */}
                      {loading && (
                        <span className="cw-late absolute inset-x-1 bottom-0 h-0.5 overflow-hidden rounded-full bg-n-surface-3" data-testid="time-loading">
                          <span className="sr-only">loading</span>
                          <span aria-hidden className="block h-full w-1/2 animate-pulse rounded-full bg-n-accent" />
                        </span>
                      )}
                    </button>
                  );
                }
                if (mk.ctx) {
                  const c = mk.ctx;
                  return (
                    <button key={mk.key} type="button" tabIndex={-1} aria-hidden data-ctx title={`${mk.title}: show this layer`} onClick={() => onChoice(c.kind === "forecast" ? { ...c, variable } : c)} className="absolute cursor-pointer" style={box} data-testid={`rail-ctx-${mk.key}`}>
                      {vis}
                    </button>
                  );
                }
                return (
                  <span key={mk.key} className="absolute" style={box} title={mk.title} data-testid={`rail-on-${mk.key}`}>
                    {vis}
                  </span>
                );
              })}
              {/* the nowcast and the forecast days are set apart */}
              {choice.kind === "forecast" &&
                (() => {
                  const f1 = marks.find((x) => x.step && x.family === "model" && !x.nowcast);
                  const n0 = marks.find((x) => x.step && x.family === "model" && x.nowcast);
                  return f1 && n0 ? <span aria-hidden data-testid="time-forecast-break" className="absolute w-[2px] bg-n-bg" style={{ left: `calc(${X(f1.t0)}% - 1px)`, top: 2, height: cellH - 4 }} /> : null;
                })()}
              {rail.nowAt != null && (
                <span aria-hidden data-testid="rail-now" className="nx-rail-now pointer-events-none absolute w-[1.5px]" style={{ left: `${X(rail.nowAt)}%`, top: compact ? 0 : -20, bottom: -18 }} />
              )}
            </div>
            <div className="relative h-[16px]" aria-hidden>
              {ticks.map((k) => (
                <span key={k.u} className={`nx-data absolute top-0.5 -translate-x-1/2 whitespace-nowrap text-[10.5px] leading-[14px] ${k.now ? "text-n-ink" : "text-n-ink-3"}`} style={{ left: `${X(k.u)}%` }}>
                  {k.text}
                </span>
              ))}
            </div>
            </div>
            </div>
            {[...chips.map((x) => ({ s: x.step!, text: x.family === "outlook" ? "8-day mean" : "Latest clear" })), ...(rail.mean ? [{ s: rail.mean, text: "24 h mean" }] : [])].map(({ s, text }) => {
              const on = s.key === t?.current;
              return (
                <button
                  key={s.key}
                  type="button"
                  role="radio"
                  aria-checked={on}
                  tabIndex={on ? 0 : -1}
                  onClick={() => {
                    setPlaying(false);
                    if (!on) onChoice(s.choice);
                  }}
                  data-testid={`time-${s.key}`}
                  className={`nx-focus relative shrink-0 rounded-[3px] border px-2 text-[12px] leading-tight ${on ? "border-n-ink text-n-ink" : "border-n-line-strong text-n-ink-2 hover:text-n-ink"}`}
                  style={{ width: MEAN_W, minHeight: compact ? 56 : undefined }}
                >
                  {text}
                  <span className="absolute inset-0 overflow-hidden">
                    <span className="sr-only">
                      {" "}
                      · {s.label} · {s.sub}
                    </span>
                  </span>
                </button>
              );
            })}
          </div>
          {nowIso && <span className="sr-only">Now: {nowIso}</span>}
        </div>
      </div>
      {caption && <div className={`min-w-0 ${compact ? "" : "pl-14"}`}>{caption}</div>}
    </div>
  );
}
