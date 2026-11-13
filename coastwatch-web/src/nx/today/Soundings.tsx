"use client";

import { useEffect, useState } from "react";
import { ChartMarks } from "@/nx/chart/Labels";
import { onFrame } from "@/nx/chart/runtime";
import { LimitGauge, Micro, WindArrow } from "@/nx/ds/synth/Grammar";
import { useMedia } from "@/nx/ds/useMedia";
import type { SoundingDatum } from "./model";
import s from "./today.module.css";

export const COMPACT = "(max-width: 640px), (max-height: 480px)";

/**
 * The readings set where they were measured (v14 synthesis, SYSTEM.md "Map layers"): the buoy belongs to Sea now and
 * the CDPH crab test to Toxin tests, each drawn only while its chip is on. The reading lives once, at its point: a
 * small card on the water with its time, its value and (the buoy) the wind; it opens where it comes from. They wait
 * for the chart to settle on Today's view, then register in place. The host places them clear of the port and the
 * names, with a leader to the point.
 */
export function useChartSettled(sceneId: string): boolean {
  const [settled, setSettled] = useState(false);
  useEffect(() => {
    setSettled(false);
    let t = 0;
    const arm = () => {
      window.clearTimeout(t);
      t = window.setTimeout(() => {
        if (document.documentElement.dataset.nxScene === sceneId) setSettled(true);
        else arm();
      }, 240);
    };
    const off = onFrame(arm);
    arm();
    return () => {
      off();
      window.clearTimeout(t);
    };
  }, [sceneId]);
  return settled;
}

/** The reading's time in the card's head: the buoy's clock time ("10:30 PM"), the test's day ("Sep 28"). */
const when = (d: SoundingDatum) => {
  const c = (d.caption[0] ?? "").replace(/^(NDBC|CDPH), /, "");
  return d.id === "buoy" ? c.replace(/^[A-Z][a-z]{2} \d{1,2}, /, "").replace(/ PT$/, "") : c.split(",")[0];
};

export type ReadingExtra = { windDeg: number | null; wind: string | null };

/** The sea dock's buoy mark lights the buoy's card on the chart while it is hovered or focused (ds SeaBars "nx-buoy-hl"). */
function useBuoyHighlight(on: boolean): boolean {
  const [hl, setHl] = useState(false);
  useEffect(() => {
    if (!on) return;
    const h = (e: Event) => setHl(!!(e as CustomEvent<boolean>).detail);
    document.addEventListener("nx-buoy-hl", h);
    return () => document.removeEventListener("nx-buoy-hl", h);
  }, [on]);
  return hl;
}

function Card({ d, onOpen, small, extra, tour }: { d: SoundingDatum; onOpen: () => void; small: boolean; extra?: ReadingExtra; tour?: boolean }) {
  const hl = useBuoyHighlight(d.id === "buoy");
  const test = d.id === "test";
  const value = test && d.unit ? Number(d.value) : NaN;
  const limit = test && d.unit ? Number(/of ([\d.]+) ppm/.exec(d.unit)?.[1]) : NaN;
  return (
    <div className={s.readingWrap} data-nx-sounding={d.id} data-tour={tour ? "sounding" : undefined}>
      {/* design-lint-allow raw-button: a reading on the chart is a card that opens its provenance; no ds control is a chart card */}
      <button type="button" className={s.reading} data-stale={d.stale ? "1" : undefined} data-hl={hl ? "1" : undefined} aria-label={d.spoken} onClick={onOpen}>
        <Micro glyph={d.stale ? "old" : "measured"} className={s.readingHead}>
          {test ? "CDPH crab test" : d.label.replace(/^(Waves|Wind), buoy /, "Buoy ")} · {when(d)}
        </Micro>
        <span className={s.readingBig}>
          {d.value}
          {d.unit ? <small>{d.unit}</small> : null}
        </span>
        {test && Number.isFinite(value) && Number.isFinite(limit) ? <LimitGauge value={value} limit={limit} label={`${d.value} against an action level of ${limit} ppm`} /> : null}
        {test && !small ? <span className={s.readingSub}>{d.label}</span> : null}
        {!test && !small && extra?.wind ? (
          <span className={s.readingSub}>
            {extra.windDeg != null ? <WindArrow fromDeg={extra.windDeg} /> : null}
            {extra.wind}
          </span>
        ) : null}
      </button>
    </div>
  );
}

export function TodaySoundings({ items, sceneId, onOpen, extra }: { items: SoundingDatum[]; sceneId: string; onOpen: (id: SoundingDatum["id"]) => void; extra?: ReadingExtra }) {
  const settled = useChartSettled(sceneId);
  const phone = useMedia(COMPACT);
  const narrow = useMedia("(max-width: 359px)");
  if (!settled || !items.length) return null;
  // a phone keeps the buoy (Sea now is on by default); the test lives in the ticket's Status on a phone
  const shown = items.filter((d) => !phone || d.id === "buoy");
  if (narrow) return null;
  return (
    <ChartMarks
      marks={shown.map((d) => ({
        id: `today-${d.id}`,
        lon: d.lon,
        lat: d.lat,
        layer: d.id === "test" ? "toxin" : "sea",
        priority: d.id === "test" ? 90 : 85,
        insist: d.id === "buoy" && phone,
        leader: true,
        offset: phone ? 18 : 26,
        anchors: phone ? ["left", "top-left", "bottom-left", "top", "bottom"] : d.id === "buoy" ? ["left", "bottom-left", "top-left", "bottom", "right"] : ["top-left", "left", "top-right", "right", "bottom-left"],
        content: <Card d={d} small={phone} extra={d.id === "buoy" ? extra : undefined} tour={d.id === (shown.some((x) => x.id === "buoy") ? "buoy" : "test")} onOpen={() => onOpen(d.id)} />,
      }))}
    />
  );
}
