import { useEffect, useId, useRef } from "react";
import { type ChartState, SeasonEngine, summarySentence } from "./engine";

/** React shell around the imperative season chart (engine.ts). */
export function SeasonChart({ state, label }: { state: ChartState; label: string }) {
  const host = useRef<HTMLDivElement>(null);
  const readout = useRef<HTMLDivElement>(null);
  const live = useRef<HTMLParagraphElement>(null);
  const engine = useRef<SeasonEngine | null>(null);
  const first = useRef(true);
  const descId = useId();

  useEffect(() => {
    const e = new SeasonEngine(host.current!, readout.current!, live.current!);
    engine.current = e;
    const ro = new ResizeObserver(([entry]) => e.resize(Math.round(entry.contentRect.width)));
    ro.observe(host.current!);
    e.resize(host.current!.clientWidth);
    return () => { ro.disconnect(); e.destroy(); engine.current = null; first.current = true; };
  }, []);

  useEffect(() => {
    engine.current?.set(state, !first.current);
    first.current = false;
  }, [state]);

  return (
    <div className="sc">
      <div
        ref={host}
        className="sc-host"
        tabIndex={0}
        role="img"
        aria-label={`${label}. Use the left and right arrow keys to read each sample.`}
        aria-describedby={descId}
      >
        <div ref={readout} className="sc-readout" hidden aria-hidden="true" />
      </div>
      <p id={descId} className="sr-only">{summarySentence(state)}</p>
      <p ref={live} className="sr-only" aria-live="polite" />
    </div>
  );
}
