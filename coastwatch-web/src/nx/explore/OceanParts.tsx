"use client";

import type { ReactNode } from "react";
import { RampLegend } from "@/nx/ds";
import type { ChlEstimateIndex, WcofsIndex, WcofsStep } from "@/lib/ocean/contract";
import { chlEstimateLabel, wcofsLabel } from "@/lib/ocean/contract";
import { chlEstimateStateWords, stepWhen, wcofsStateWords } from "./ocean";
import { CHL_DOMAIN, CHL_STEPS, CHL_TICKS } from "./words";
import s from "./explore.module.css";

/**
 * Round 2 (D, E): the model and estimate rows of the time rail and their legends. WCOFS nowcast steps sit on Analysis /
 * Estimates, its forecast steps on Forecasts, the DINEOF estimate on Analysis / Estimates with its one day; each row says
 * its own last valid point. A step that could not be read is marked, never filled in.
 */
export function StepRow({ label, steps, selected, failed, onStep, testid }: { label: string; steps: WcofsStep[]; selected: string | null; failed: Set<string>; onStep: (id: string) => void; testid: string }) {
  if (!steps.length) return null;
  const last = steps[steps.length - 1];
  return (
    <div className={s.oceanRow} data-testid={testid}>
      <span className={s.railLabel}>{label}</span>
      <div className={s.oceanSteps} role="radiogroup" aria-label={`${label} steps`}>
        {steps.map((st) => (
          // design-lint-allow raw-button: a time step on a strip (a radio group drawn as ticks), like the rail's frames
          <button
            key={st.id}
            type="button"
            role="radio"
            aria-checked={st.id === selected}
            aria-label={`${st.kind === "nowcast" ? "Nowcast" : "Forecast"}, valid ${stepWhen(st)}${failed.has(st.id) ? ", could not be read" : ""}`}
            className={s.oceanStep}
            data-failed={failed.has(st.id) ? "1" : undefined}
            onClick={() => onStep(st.id)}
            data-testid={`nx-step-${st.id}`}
          />
        ))}
      </div>
      <span className={s.oceanLast}>{last.kind === "nowcast" ? `to ${stepWhen(last)}` : `valid to ${stepWhen(last)}`}</span>
    </div>
  );
}

export function EstimateRow({ ix, loaded, on, onChoose, offline = false }: { ix: ChlEstimateIndex | null; loaded: boolean; on: boolean; onChoose: () => void; offline?: boolean }) {
  return (
    <div className={s.oceanRow} data-testid="nx-rail-chl-estimate">
      <span className={s.railLabel}>Chl estimate</span>
      {ix?.grids && ix.date ? (
        // design-lint-allow raw-button: the estimate's one day on the rail (a single radio), like the rail's frames
        <button type="button" role="radio" aria-checked={on} className={s.oceanDay} onClick={onChoose} data-testid="nx-rail-chl-estimate-day">
          {chlEstimateStateWords(ix, loaded)}
        </button>
      ) : (
        <span className={s.oceanNote}>{chlEstimateStateWords(ix, loaded, offline)}</span>
      )}
    </div>
  );
}

export function WcofsNote({ ix, loaded, offline = false }: { ix: WcofsIndex | null; loaded: boolean; offline?: boolean }) {
  return (
    <p className={s.oceanNote} data-testid="nx-wcofs-state">
      {wcofsStateWords(ix, loaded, offline)}
    </p>
  );
}

/** The model temperature's legend: isotherms (no colour ramp, as for every temperature on the chart), the step's label. */
export function SstModelLegend({ ix, step, failed, about }: { ix: WcofsIndex; step: WcofsStep | null; failed: boolean; about?: ReactNode }) {
  return (
    <div className={s.legendBlock} data-testid="nx-legend" data-product="sst-model">
      <figure className={s.figure} aria-label="Model sea temperature, isotherms every degree Fahrenheit">
        <figcaption className={s.figHead}>
          <span className={s.figTitle}>Sea temperature (model)</span>
          <span className={s.figUnit}>°F</span>
        </figcaption>
        <div className={s.isoKey} aria-hidden="true">
          <svg viewBox="0 0 120 24" className={s.isoSvg}>
            <path d="M2,18 C30,4 60,22 118,8" className={s.isoMajorWarm} />
            <path d="M2,22 C34,12 64,26 118,15" className={s.isoMinorWarm} />
          </svg>
          <span>every 1 °F, heavier and labelled every 2 °F</span>
        </div>
      </figure>
      <p className={s.caption} data-testid="nx-model-label">
        {failed ? `This step could not be read from NOAA WCOFS (unavailable). Nothing is drawn in its place.` : step && ix.cycle ? wcofsLabel(ix.cycle.id, step) : wcofsStateWords(ix, true)}
      </p>
      <details className={s.about} data-testid="nx-about">
        <summary>About this layer</summary>
        <div className={s.aboutBody}>
          <p>Not drawn within {ix.near_shore_km} km of the shore, where the model&rsquo;s 4 km grid is coarsest.</p>
          {ix.caveats.map((c) => (
            <p key={c}>{c}</p>
          ))}
          {about}
        </div>
      </details>
    </div>
  );
}

/** The gap-filled estimate's legend: the chlorophyll ramp under the estimate's hatch, its day and staleness. */
export function ChlEstimateLegend({ ix, about }: { ix: ChlEstimateIndex; about?: ReactNode }) {
  return (
    <div className={s.legendBlock} data-testid="nx-legend" data-product="chl-estimate">
      <RampLegend title="Chlorophyll estimate" unit="mg/m³" steps={CHL_STEPS} domain={CHL_DOMAIN} scale="log" ticks={CHL_TICKS} noView={false} className={s.legend} />
      <ul className={s.keys} aria-label="Marks">
        <li>
          <svg viewBox="0 0 40 16" className={s.swatch} aria-hidden="true">
            <rect x="0.5" y="0.5" width="39" height="15" className={s.swatchFrame} />
            <g className={s.swatchHatch}>
              {[-24, -16, -8, 0, 8, 16, 24, 32, 40].map((x) => (
                <line key={x} x1={x} y1={24} x2={x + 24} y2={0} />
              ))}
            </g>
          </svg>
          <span>Hatched: an estimate, not observed</span>
        </li>
      </ul>
      <p className={s.caption} data-testid="nx-model-label">
        {ix.date ? chlEstimateLabel(ix.date) : chlEstimateStateWords(ix, true)}
        {ix.status === "stale" ? ` · stale: ${ix.age_days} days old` : ""}
      </p>
      <details className={s.about} data-testid="nx-about">
        <summary>About this layer</summary>
        <div className={s.aboutBody}>
          <p>Not drawn within {ix.near_shore_km} km of the shore. The estimate reads about three times lower than the satellites near shore.</p>
          {ix.reason ? <p>{ix.reason}</p> : null}
          {ix.caveats.map((c) => (
            <p key={c}>{c}</p>
          ))}
          {about}
        </div>
      </details>
    </div>
  );
}
