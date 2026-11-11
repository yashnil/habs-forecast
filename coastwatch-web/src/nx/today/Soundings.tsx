"use client";

import { useEffect, useState } from "react";
import { ChartMarks } from "@/nx/chart/Labels";
import { onFrame } from "@/nx/chart/runtime";
import { Sounding } from "@/nx/ds/chart/Sounding";
import { useMedia } from "@/nx/ds/useMedia";

export const COMPACT = "(max-width: 720px), (max-height: 480px)";
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

function Figure({ d, onOpen, phone, narrow }: { d: SoundingDatum; onOpen: () => void; phone: boolean; narrow: boolean }) {
  return (
    <div className="nx-today-sounding" data-nx-sounding={d.id} data-tour={d.id === "test" ? "sounding" : undefined}>
      <Sounding
        x={0}
        y={0}
        placement="box"
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
        size={phone ? "md" : "lg"}
        stale={d.stale}
        onActivate={onOpen}
        ariaLabel={d.spoken}
        width={narrow ? 128 : phone ? 136 : 256}
      />
    </div>
  );
}

export function TodaySoundings({ items, sceneId, onOpen }: { items: SoundingDatum[]; sceneId: string; onOpen: (id: SoundingDatum["id"]) => void }) {
  const settled = useChartSettled(sceneId);
  // phones, and short landscape screens: a figure and one line
  const phone = useMedia(COMPACT);
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
        offset: phone ? 22 : 26,
        // on a phone the water lies west of the points (camera.ts phoneBounds): the type goes there first
        anchors: phone
          ? d.id === "test"
            ? ["top-left", "bottom-left", "left", "top", "bottom", "top-right", "right"]
            : ["bottom-left", "bottom", "left", "bottom-right", "right", "top-left"]
          : d.id === "buoy"
            ? ["bottom-right", "top-right", "right", "bottom-left", "left", "top-left"]
            : ["top-right", "top-left", "right", "left", "bottom-right", "bottom-left"],
        content: <Figure d={d} phone={phone} narrow={narrow} onOpen={() => onOpen(d.id)} />,
      }))}
    />
  );
}
