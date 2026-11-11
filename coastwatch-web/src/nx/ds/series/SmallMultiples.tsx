"use client";
import { useMemo, useState } from "react";
import { KIND_WORD, type Kind } from "../kinds";
import { cx } from "../util";
import { Brush } from "./Brush";
import { SeriesChart } from "./SeriesChart";
import { classify, ms as toMs, readoutText, nearestIndex, type SeriesPoint } from "./series";
import s from "./series.module.css";

export interface Strip {
  id: string;
  label: string;
  unit: string;
  points: SeriesPoint[];
  scale?: "log" | "linear";
  kind?: Kind;
  marks?: "points" | "line" | "both";
}

export interface SmallMultiplesProps {
  strips: Strip[];
  /** The brush overview draws this strip's record. Defaults to the first. */
  overview?: string;
  /** Initial window, ms; defaults to the last three years of the record. */
  initial?: [number, number];
  /**
   * The window, controlled (ms): the page holds it (URL state, presets, export) and `onWindow` asks to change
   * it. Without it the strips keep their own window from `initial`.
   */
  value?: [number, number];
  /** Called as the window changes (to keep URL state or export the window). */
  onWindow?: (w: [number, number]) => void;
  /** What the window filters, for the brush's words. */
  label?: string;
  stripHeight?: number;
  className?: string;
}

/**
 * A station's record as strips on one time axis with one brush (spec section 9). The crosshair is shared:
 * reading a date in one strip reads it in all, each with its own words.
 */
export function SmallMultiples({ strips, overview, initial, value, onWindow, label = "every strip above", stripHeight = 110, className }: SmallMultiplesProps) {
  const extent = useMemo<[number, number]>(() => {
    const ts = strips.flatMap((st) => st.points.filter((p) => classify(p) !== "missing").map((p) => toMs(p.t)));
    return ts.length ? [Math.min(...ts), Math.max(...ts)] : [0, 1];
  }, [strips]);
  const [own, setOwn] = useState<[number, number]>(initial ?? [Math.max(extent[0], extent[1] - 3 * 365 * 86_400_000), extent[1]]);
  const win = value ?? own;
  const [hoverT, setHoverT] = useState<number | null>(null);
  const ov = strips.find((st) => st.id === overview) ?? strips[0];
  const change = (w: [number, number]) => {
    if (!value) setOwn(w);
    onWindow?.(w);
  };
  return (
    <div className={cx(s.multiples, className)} role="group" aria-label={`${strips.length} records on one time axis`}>
      {strips.map((st, i) => {
        const inWin = st.points.filter((p) => classify(p) !== "missing" && toMs(p.t) >= win[0] && toMs(p.t) <= win[1]);
        const idx = hoverT === null ? -1 : nearestIndex(inWin, hoverT);
        const words = idx >= 0 ? readoutText(inWin[idx], st.unit, KIND_WORD[st.kind ?? "measured"]) : null;
        return (
          <SeriesChart
            key={st.id}
            points={st.points}
            label={st.label}
            unit={st.unit}
            scale={st.scale}
            kind={st.kind}
            marks={st.marks}
            xDomain={win}
            height={stripHeight}
            hoverT={hoverT}
            onHoverT={setHoverT}
            showReadout={false}
            xAxis={i === strips.length - 1}
            title={
              <>
                <span>{st.label}</span>
                <small>{st.unit}</small>
                <span className={s.stripReadout} aria-live="polite">
                  {words ?? ""}
                </span>
              </>
            }
          />
        );
      })}
      <Brush extent={extent} value={win} onChange={change} points={ov.points} scale={ov.scale} label={label} />
    </div>
  );
}
