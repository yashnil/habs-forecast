"use client";

import { useEffect, useRef, useState } from "react";
import type { Manifest, Palette } from "@/generated/schema";
import { AGE_COLOURS, SENSOR_COLOURS, displayPalette, gradientCss } from "@/lib/palette";
import { CHL_PALETTE, SST_PALETTE, cToF, derivedOutlook, derivedSst, type DerivedIndex } from "@/lib/derived";
import { FLOW_SKY, SPEED_FULL, SPEED_RAMP } from "@/lib/flowSim";
import { SPEED_CLASSES, ARROW_LENGTHS } from "@/lib/basemap";
import { charmRun } from "@/lib/layers";
import { forecastFreshness } from "@/lib/opening";
import { pacificToday } from "@/lib/time";
import { layerFreshness } from "@/v4/readiness/shown";
import type { Read } from "@/components/map/HoverReadout";
import { clockLabel, dayLabel } from "@/v4/ui/time";
import { Icon } from "@/v4/ui/Icon";
import { windowLabel } from "./LayerPanel";
import { runDays, SAT_WORDS, type LayerChoice, type Resolved } from "./layers";

/**
 * The field dock's head: the legend is the map's title (design.md §7.3, research pattern 2). One line says
 * what the colours are, what kind of thing they are, for when, and what they are not; under it the ramp, with
 * a marker that moves along it for the point under the cursor or the point selected. The ramp is the published
 * palette; the marker is the published value at that point, or the word "no view" (a gap is never read as low).
 */

/** What the head says. Each piece is its own element, so tests and readers can find the kind and the date. */
export type DockWords = { title: string; kind: string; tone: string; when: string | null; old: string | null; caveat: string | null };

const satKind = "observed";

/** The words for the layer on the map, frame by frame: the date is the frame's own, and its age says so in place. */
export function dockWords(m: Manifest, c: LayerChoice, r: Resolved, now: Date | null, derived: DerivedIndex | null): DockWords {
  // short on the dock: the amber date and "(old)" say it in place; the age and the sentence are in the layer list
  const oldOf = (f: { state: string; ageDays: number | null } | null) => (f && (f.state === "stale" || f.state === "historical") ? "old" : null);
  if (c.kind === "forecast") {
    const run = charmRun(m);
    const days = runDays(m, now);
    const d = days.find((x) => x.lead === c.lead)?.date ?? null;
    const today = now ? pacificToday(now) : null;
    const allPast = !!today && days.filter((x) => x.issued && x.date).every((x) => x.date! < today);
    const fresh = now ? forecastFreshness(m, now) : null;
    return {
      title: "Bloom & toxin probability",
      kind: allPast ? "model, past run" : c.lead === 0 ? "model nowcast" : "model forecast",
      tone: "text-n-model",
      when: d ? dayLabel(d) : run ? `run ${dayLabel(run.issued_date)}` : null,
      old: oldOf(fresh),
      caveat: "probability of conditions, not toxin in seafood",
    };
  }
  if (c.kind === "satellite") {
    const l = r.satLayer;
    const d = c.day ?? l?.composite?.newest_observed_date ?? l?.time.observed_date ?? null;
    const view = c.view === "age" ? "age of each pixel" : c.view === "sensor" ? "which sensor" : null;
    return { title: view ? `Chlorophyll, ${view}` : "Chlorophyll", kind: satKind, tone: "text-n-measured", when: d ? dayLabel(d) : null, old: oldOf(layerFreshness(l ?? null, now)), caveat: "not toxin" };
  }
  if (c.kind === "outlook") {
    const o = r.outlookLayer ?? derivedOutlook(derived);
    const past = !!o && !!now && o.valid_to < pacificToday(now);
    return { title: "Chlorophyll outlook", kind: past ? "outlook, past run" : "outlook, not observed", tone: "text-n-model", when: o ? windowLabel(o.valid_from, o.valid_to) : null, old: past ? "past" : null, caveat: "not toxin" };
  }
  if (c.kind === "sst") {
    const t = r.sstLayer ?? derivedSst(derived);
    const d = t?.observed_dates[t.observed_dates.length - 1];
    return { title: "Sea temperature", kind: "analysis", tone: "text-n-conditions", when: d ? dayLabel(d) : null, old: null, caveat: null };
  }
  if (c.kind === "currents") {
    const l = r.curLayer;
    const t = l?.time.valid_time;
    return { title: c.chl ? "Surface currents, with chlorophyll" : "Surface currents", kind: "observed", tone: "text-n-conditions", when: c.mean ? "24 h mean" : t ? clockLabel(t) : null, old: oldOf(layerFreshness(l ?? null, now)), caveat: c.chl ? "chlorophyll is not toxin" : "not a forecast" };
  }
  return { title: "Seafloor", kind: "reference", tone: "text-n-ink-2", when: null, old: null, caveat: "not data" };
}

export type RampModel = {
  /** the bar's CSS background; null for the arrows' own scale */
  css: string | null;
  labels: string[];
  /** a value to a share of the bar, 0..1; null when the bar is classes with no position for a value */
  at: ((v: number) => number) | null;
  gap: "no view" | "no value" | null;
};

const clamp01 = (x: number) => Math.max(0, Math.min(1, x));
/** A value's place on a published palette, by the palette's own scale. */
const position = (p: Palette) => (v: number) => {
  const x = p.scale === "log10" ? Math.log10(Math.max(v, 1e-12)) : v;
  return clamp01((x - p.domain[0]) / (p.domain[1] - p.domain[0]));
};
const stepped = (colours: readonly string[]) => `linear-gradient(90deg, ${colours.map((c, i) => `${c} ${(i / colours.length) * 100}% ${((i + 1) / colours.length) * 100}%`).join(", ")})`;
const DEPTH = "linear-gradient(90deg, #3e8fb0 0%, #225e84 25%, #1a486c 43%, #133555 68%, #0e2540 100%)";

/** The ramp for the layer on the map: its published colours, its own ticks, and how a value finds its place on it. */
export function rampOf(c: LayerChoice, r: Resolved, flow: { colour: boolean } | null): RampModel | null {
  if (c.kind === "forecast") {
    const p = r.forecastLayer?.palette;
    return p ? { css: gradientCss(displayPalette(p)), labels: ["0%", "25%", "50%", "75%", "100%"], at: (v) => clamp01(v / 100), gap: "no value" } : null;
  }
  if (c.kind === "satellite") {
    if (c.view === "age") return { css: stepped(AGE_COLOURS), labels: ["today", "7 days old"], at: null, gap: "no view" };
    if (c.view === "sensor") return { css: stepped(SENSOR_COLOURS), labels: [SAT_WORDS.olci300.native, SAT_WORDS.viirs750.native], at: null, gap: "no view" };
    const p = r.satLayer?.palette;
    return p ? { css: gradientCss(p), labels: ["0.05", "0.3", "1.6", "9", "50 mg/m³"], at: position(p), gap: "no view" } : null;
  }
  if (c.kind === "outlook") {
    const p = r.outlook?.palette ?? CHL_PALETTE;
    return { css: gradientCss(p), labels: ["0.05", "0.3", "1.6", "9", "50 mg/m³"], at: position(p), gap: "no value" };
  }
  if (c.kind === "sst") {
    const [lo, hi] = SST_PALETTE.domain;
    return { css: gradientCss(SST_PALETTE), labels: [`≤${Math.round(cToF(lo))}°F`, `${Math.round(cToF((lo + hi) / 2))}°F`, `≥${Math.round(cToF(hi))}°F`], at: position(SST_PALETTE), gap: "no value" };
  }
  if (c.kind === "currents") {
    if (!flow) return { css: null, labels: [], at: null, gap: null };
    if (flow.colour) return { css: stepped(SPEED_RAMP), labels: [...SPEED_CLASSES.map(String).slice(0, -1), `≥${SPEED_CLASSES[SPEED_CLASSES.length - 1]} m/s`], at: null, gap: null };
    return { css: `linear-gradient(90deg, ${FLOW_SKY.map(([t, col]) => `${col} ${t * 100}%`).join(", ")})`, labels: ["0", "0.2", "0.4", `≥${SPEED_FULL} m/s`], at: (v) => clamp01(v / SPEED_FULL), gap: null };
  }
  return { css: DEPTH, labels: ["0", "200", "1,000", "3,000 m"], at: null, gap: null };
}

/** One published read, shown on the ramp: the value's place and its words, or the honest absence. */
const placeOf = (read: Read | null, ramp: RampModel | null): { t: number | null; text: string } | null => {
  if (!read || !ramp) return null;
  if (read.value == null) return { t: null, text: ramp.gap ? `${ramp.gap} here` : "no value here" };
  const n = Number.parseFloat(read.value.replace(/[^\d.-]/g, ""));
  return { t: ramp.at && Number.isFinite(n) ? ramp.at(n) : null, text: `${read.value}${read.unit ? ` ${read.unit}` : ""}` };
};

function DockRamp({ ramp, sink, selected }: { ramp: RampModel; sink: React.RefObject<((r: Read | null) => void) | null>; selected: Read | null }) {
  const [hover, setHover] = useState<Read | null>(null);
  useEffect(() => {
    sink.current = setHover;
    return () => {
      sink.current = null;
    };
  }, [sink]);
  const place = placeOf(hover ?? selected, ramp);
  const pct = place?.t != null ? place.t * 100 : null;
  return (
    <div className="flex min-w-0 items-end gap-3">
      <div className="min-w-0 flex-1">
        {ramp.css ? (
          <div className="relative h-[8px] rounded-[2px]" style={{ background: ramp.css }} data-testid="dock-ramp" aria-hidden>
            {pct != null && <span className="nx-ramp-mark" style={{ left: `${pct}%` }} data-testid="dock-marker" />}
          </div>
        ) : (
          <div className="flex items-end gap-3" data-testid="dock-ramp" aria-hidden>
            {SPEED_CLASSES.map((v, i) => (
              <span key={v} className="flex items-center">
                <svg width={ARROW_LENGTHS[i] + 4} height="8">
                  <line x1={2} y1={4} x2={ARROW_LENGTHS[i] + 2} y2={4} stroke="currentColor" strokeWidth={1.7} strokeLinecap="round" className="text-n-ink" />
                </svg>
              </span>
            ))}
          </div>
        )}
        <div className="relative mt-[3px] h-[14px]">
          <div className={`nx-data flex justify-between text-[11px] leading-[14px] text-n-ink-2 transition-opacity duration-100 ${place ? "opacity-0" : ""}`} aria-hidden>
            {ramp.labels.map((l, i) => (
              <span key={i}>{l}</span>
            ))}
          </div>
          {place && (
            <span className="absolute top-0 whitespace-nowrap text-[12.5px] font-medium leading-[14px] text-n-ink" style={{ left: `clamp(0px, calc(${pct ?? 0}% - 40px), calc(100% - 110px))` }} data-testid="dock-readout">
              {place.text}
            </span>
          )}
        </div>
      </div>
      {ramp.gap && (
        <span className="flex shrink-0 items-center gap-1.5 pb-[3px] text-[12.5px] leading-none text-n-ink-2" data-testid="no-view-key" title={ramp.gap === "no view" ? "No view: cloud, fog or no pass. A gap is not clear water and not low chlorophyll." : "No value here: not modelled, or no data. Never zero."}>
          <span aria-hidden className="nx-sw-gap h-[9px] w-[16px]" />
          {ramp.gap}
          <span className="sr-only">: a gap is never a low value</span>
        </span>
      )}
    </div>
  );
}

/**
 * The legend-as-title. `phone`: the sheet's own edge (no chevron: the sheet's handle is beside it).
 * `open`: the layer list is showing. `extra`: a control that sits beside the title (the currents' pause).
 */
export function DockHead({ words, ramp, sink, selected, open, onToggle, capsuleRef, phone = false, testids = true, extra = null, describe }: { words: DockWords; ramp: RampModel | null; sink: React.RefObject<((r: Read | null) => void) | null>; selected: Read | null; open: boolean; onToggle: () => void; capsuleRef: React.Ref<HTMLButtonElement>; phone?: boolean; testids?: boolean; extra?: React.ReactNode; describe: string }) {
  return (
    <div className="flex min-w-0 flex-col" data-testid={testids ? "dock-head" : undefined}>
      <div className="flex min-w-0 items-start gap-1">
        <button ref={capsuleRef} type="button" aria-expanded={open} aria-controls="layer-deck" aria-label={`Map layer: ${describe}. ${open ? "Close the layer list" : "Change the layer"}`} onClick={onToggle} className={`nx-press nx-focus flex min-h-11 min-w-0 flex-1 items-center gap-2 rounded-[10px] text-left ${phone ? "-ml-1 pl-1" : "-mx-1 px-1 hover:bg-n-surface-2"} -mb-1.5 -mt-1 py-1`} data-testid="layer-capsule">
          <span className="min-w-0 flex-1 text-[14.5px] leading-[20px] text-n-ink-2" data-testid="dock-title">
            <span className="font-medium text-n-ink">{words.title}</span>
            <span> · </span>
            <span className={`${words.tone} font-medium`} data-testid="dock-kind">{words.kind}</span>
            {words.when && (
              <>
                <span> · </span>
                <span className={words.old ? "font-medium text-n-stale" : ""} data-testid="dock-date" data-stale={words.old ? "true" : undefined}>
                  {words.old && <Icon name="clock" className="mr-0.5 inline h-3.5 w-3.5 -translate-y-px" />}
                  {words.when}
                  {words.old ? ` (${words.old})` : ""}
                </span>
              </>
            )}
            {words.caveat && (
              <>
                <span> · </span>
                <span className={words.caveat === "not toxin" ? "whitespace-nowrap" : ""} data-testid="dock-caveat">{words.caveat}</span>
              </>
            )}
          </span>
          {!phone && <Icon name="layers" className={`h-[18px] w-[18px] shrink-0 ${open ? "text-n-ink" : "text-n-ink-2"}`} />}
        </button>
        {extra}
      </div>
      {ramp && (
        <div className="mt-1">
          <DockRamp ramp={ramp} sink={sink} selected={selected} />
        </div>
      )}
    </div>
  );
}

/** A small ramp for a chip: decorative; the words beside it say what it is. */
export function MiniRamp({ ramp }: { ramp: RampModel | null }) {
  if (!ramp?.css) return null;
  return <span aria-hidden className="h-[6px] w-9 shrink-0 rounded-[1px]" style={{ background: ramp.css }} data-testid="capsule-ramp" />;
}

/** The ref a hover readout writes into; a move re-renders only the marker. */
export function useDockSink() {
  return useRef<((r: Read | null) => void) | null>(null);
}
