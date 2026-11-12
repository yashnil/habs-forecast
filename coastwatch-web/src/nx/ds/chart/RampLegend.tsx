import type { ReactNode } from "react";
import { sig } from "../format";
import { hatchSegments } from "../Mark";
import { cx } from "../util";
import s from "./legend.module.css";

export interface RampLegendProps {
  /** The active layer's name: the legend is its title. */
  title: string;
  unit: string;
  /** Custom-property names of the ramp's steps, low to high: ["--chl-0", …, "--chl-7"]. */
  steps: string[];
  /** Data domain the steps span. */
  domain: [number, number];
  scale?: "log" | "linear";
  /** Labelled ticks (T21: 0.1, 1, 10 for chlorophyll). */
  ticks: number[];
  tickFormat?: (v: number) => string;
  /** Kind, source, dates and the mandatory words ("algae, not toxin", T5). */
  caption?: ReactNode;
  /** Shows the "no view" hatch beside the ramp (T19). Pass a string to change the words. */
  noView?: boolean | string;
  /** Drawn at the opacity the data uses on the chart (T23). */
  opacity?: number;
  className?: string;
}

/** Position of a value along the ramp, 0 to 1. */
export function rampPosition(v: number, [lo, hi]: [number, number], scale: "log" | "linear" = "log"): number {
  const f = scale === "log" ? (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) : (v - lo) / (hi - lo);
  return Math.min(1, Math.max(0, f));
}

const NOVIEW = hatchSegments(0, 0, 40, 16, 4);

/** Legend-as-title for the active layer: stepped ramp, log ticks, and the "no view" hatch. */
export function RampLegend({ title, unit, steps, domain, scale = "log", ticks, tickFormat = (v) => sig(v, 1), caption, noView = true, opacity = 1, className }: RampLegendProps) {
  const noViewWords = typeof noView === "string" ? noView : "No view";
  return (
    <figure className={cx(s.legend, className)} aria-label={`${title}, ${unit}. Scale from ${tickFormat(domain[0])} to ${tickFormat(domain[1])}${scale === "log" ? ", logarithmic" : ""}.`}>
      <figcaption className={s.head}>
        <span className={s.title}>{title}</span>
        <span className={s.unit}>{unit}</span>
      </figcaption>
      <div className={s.scaleRow}>
        <div className={s.rampCol}>
          <div className={s.bar} style={{ opacity }} aria-hidden>
            {steps.map((v) => (
              <span key={v} className={s.step} style={{ background: `var(${v})` }} />
            ))}
          </div>
          <div className={s.ticks} aria-hidden>
            {ticks.map((t) => (
              <span key={t} className={s.tick} style={{ left: `${rampPosition(t, domain, scale) * 100}%` }}>
                {tickFormat(t)}
              </span>
            ))}
          </div>
        </div>
        {noView ? (
          <div className={s.noViewCol}>
            <svg className={s.noView} viewBox="0 0 40 16" preserveAspectRatio="none" aria-hidden focusable="false">
              <rect x="0.5" y="0.5" width="39" height="15" className={s.noViewFrame} />
              <path d={NOVIEW} className={s.noViewHatch} />
            </svg>
            <span className={s.noViewLabel}>{noViewWords}</span>
          </div>
        ) : null}
      </div>
      {/* which end is "more" depends on the mode's table (tokens.css): darker on Day's paper, lighter on the dark
          grounds of Dusk and Night. Only the active mode's words show (and are read). */}
      <p className={s.direction} data-testid="ramp-direction">
        <span className={s.saysDarker}>Darker is more.</span>
        <span className={s.saysLighter}>Lighter is more (dark chart).</span>
      </p>
      {caption ? <p className={s.caption}>{caption}</p> : null}
    </figure>
  );
}
