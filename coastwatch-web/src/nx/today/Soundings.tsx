"use client";

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { ChartMarks } from "@/nx/chart/Labels";
import { onFrame } from "@/nx/chart/runtime";
import { Sounding } from "@/nx/ds/chart/Sounding";
import { useMedia, usePhone } from "@/nx/ds/useMedia";
import type { SoundingDatum } from "./model";

/**
 * At most two figures set where they were measured (README principle 2): the test area and the buoy. They
 * wait for the chart to settle on Today's view, then each registers in place (ds Sounding, `register`).
 * The host (src/nx/chart) places them clear of the port and the names, with a leader to the point.
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

/** The ds Sounding draws absolutely; the host measures boxes, so the box takes the type's own size. */
function Box({ d, onOpen, phone, narrow }: { d: SoundingDatum; onOpen: () => void; phone: boolean; narrow: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: phone ? 200 : 236, h: phone ? 84 : 104 });
  useLayoutEffect(() => {
    const block = ref.current?.querySelector<HTMLElement>("[data-priority] > :last-child");
    if (!block) return;
    const measure = () => setSize((s) => (s.w === block.offsetWidth && s.h === block.offsetHeight ? s : { w: block.offsetWidth, h: block.offsetHeight }));
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(block);
    return () => ro.disconnect();
  }, []);
  return (
    <div ref={ref} className="nx-today-sounding" style={{ position: "relative", width: size.w, height: size.h, whiteSpace: "normal" }} data-nx-sounding={d.id} data-tour={d.id === "test" ? "sounding" : undefined}>
      <Sounding
        x={0}
        y={0}
        value={d.value}
        unit={d.unit ?? undefined}
        label={phone ? undefined : d.label}
        caption={
          phone ? (
            d.short
          ) : (
            <>
              {d.caption.map((c, i) => (
                <span key={i} style={{ display: "block" }}>
                  {c}
                </span>
              ))}
            </>
          )
        }
        kind="measured"
        anchor="se"
        leader={null}
        size={phone ? "md" : "lg"}
        stale={d.stale}
        onActivate={onOpen}
        ariaLabel={d.spoken}
        width={narrow ? 150 : phone ? 176 : 256}
      />
    </div>
  );
}

export function TodaySoundings({ items, sceneId, onOpen }: { items: SoundingDatum[]; sceneId: string; onOpen: (id: SoundingDatum["id"]) => void }) {
  const settled = useChartSettled(sceneId);
  const phone = usePhone();
  const narrow = useMedia("(max-width: 360px)");
  if (!settled || !items.length) return null;
  return (
    <ChartMarks
      marks={items.map((d) => ({
        id: `today-${d.id}`,
        lon: d.lon,
        lat: d.lat,
        priority: d.id === "test" ? 90 : 85,
        leader: true,
        offset: phone ? 30 : 26,
        anchors: phone
          ? d.id === "test"
            ? ["top-right", "bottom-right", "top", "right", "top-left", "left"]
            : ["bottom-right", "bottom", "bottom-left", "right", "left", "top-right"]
          : d.id === "buoy"
            ? ["bottom-right", "top-right", "right", "bottom-left", "left", "top-left"]
            : ["top-right", "top-left", "right", "left", "bottom-right", "bottom-left"],
        content: <Box d={d} phone={phone} narrow={narrow} onOpen={() => onOpen(d.id)} />,
      }))}
    />
  );
}
