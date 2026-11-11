"use client";

import { RampLegend } from "@/nx/ds";
import type { CurrentFrame, ExploreFrame, ExploreProduct } from "./model";
import { CHL_DOMAIN, CHL_STEPS, CHL_TICKS, currentsCaption, legendCaption, PROB_STEPS } from "./words";
import s from "./explore.module.css";

/**
 * Legend-as-title (README §9 RampLegend): the active layer's name is the title, its scale drawn as the data is
 * drawn (T23), its caption the kind, source, dates and mandatory words. Chlorophyll and the outlook use the
 * ds RampLegend, and so does C-HARM: its ten display classes (cw-probability-classes-v1) on the quiet --prob
 * ramp at the opacity the chart draws them, with the model's hatch keyed beside it. The temperature's isotherms
 * have no colour ramp (T22), so its legend draws its own marks in the same form.
 */
function HatchSwatch({ spacing, past }: { spacing: number; past?: boolean }) {
  const lines = [];
  for (let x = -24; x < 48; x += spacing) lines.push(<line key={x} x1={x} y1={24} x2={x + 24} y2={0} />);
  return (
    <svg viewBox="0 0 40 16" className={s.swatch} aria-hidden="true" preserveAspectRatio="xMidYMid slice">
      <rect x="0.5" y="0.5" width="39" height="15" className={s.swatchFrame} />
      <g className={past ? s.swatchHatchPast : s.swatchHatch}>{lines}</g>
    </svg>
  );
}

export function ExploreLegend({ product, frame, currents }: { product: ExploreProduct; frame: ExploreFrame; currents: CurrentFrame | null }) {
  const caption = legendCaption(product, frame);
  let body;
  if (product.id === "chl") {
    body = <RampLegend title="Chlorophyll" unit="mg/m³" steps={CHL_STEPS} domain={CHL_DOMAIN} scale="log" ticks={CHL_TICKS} caption={caption} noView className={s.legend} />;
  } else if (product.id === "outlook") {
    body = (
      <>
        <RampLegend title="Chlorophyll outlook" unit="mg/m³" steps={CHL_STEPS} domain={CHL_DOMAIN} scale="log" ticks={CHL_TICKS} noView={false} className={s.legend} />
        <ul className={s.keys} aria-label="Marks">
          <li>
            <HatchSwatch spacing={8} />
            <span>Hatched: outlook, a model</span>
          </li>
          <li>
            <svg viewBox="0 0 40 16" className={s.swatch} aria-hidden="true">
              <rect x="0.5" y="0.5" width="39" height="15" className={s.swatchFrame} />
              {[6, 14, 22, 30].flatMap((x) => [4, 11].map((y) => <circle key={`${x}-${y}`} cx={x + (y > 6 ? 4 : 0)} cy={y} r="1.2" className={s.swatchDot} />))}
            </svg>
            <span>No signal</span>
          </li>
        </ul>
        <p className={s.caption}>{caption}</p>
      </>
    );
  } else if (product.id === "charm") {
    body = (
      <>
        <RampLegend
          title="C-HARM, P(pDA > 500 ng/L)"
          unit="probability, ten display classes"
          steps={PROB_STEPS}
          domain={[0, 1]}
          scale="linear"
          ticks={[0, 0.5, 1]}
          tickFormat={(v) => `${Math.round(v * 100)} %`}
          noView={false}
          opacity={product.pastRun ? 0.42 : 0.72}
          className={s.legend}
        />
        <ul className={s.keys} aria-label="Marks">
          <li>
            <HatchSwatch spacing={8} past={product.pastRun} />
            <span>Hatched: a model{product.pastRun ? ", past run drawn lighter" : ""}</span>
          </li>
        </ul>
        <p className={s.caption}>{caption}</p>
      </>
    );
  } else {
    body = (
      <figure className={s.figure} aria-label="Sea temperature, isotherms every degree Fahrenheit">
        <figcaption className={s.figHead}>
          <span className={s.figTitle}>Sea temperature</span>
          <span className={s.figUnit}>°F</span>
        </figcaption>
        <div className={s.isoKey} aria-hidden="true">
          <svg viewBox="0 0 120 24" className={s.isoSvg}>
            <path d="M2,18 C30,4 60,22 118,8" className={s.isoMajor} />
            <path d="M2,22 C34,12 64,26 118,15" className={s.isoMinor} />
          </svg>
          <span>every 1 °F, heavier every 2 °F</span>
        </div>
        <p className={s.caption}>{caption}</p>
      </figure>
    );
  }
  return (
    <div className={s.legendBlock} data-testid="nx-legend" data-product={product.id}>
      {body}
      {currents ? (
        <p className={s.currentsKey} data-testid="nx-currents-key">
          <svg viewBox="0 0 40 16" className={s.swatch} aria-hidden="true">
            <path d="M3,12 C14,11 24,7 37,4" className={s.streakSwatch} />
          </svg>
          <span>{currentsCaption(currents)}</span>
        </p>
      ) : null}
    </div>
  );
}
