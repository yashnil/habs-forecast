"use client";

import { Button, ButtonGroup } from "@/nx/ds";
import { day, latLon } from "@/nx/ds/format";
import type { SampleRecord } from "./availability";
import s from "./v13.module.css";

const km = (v: number) => (v < 1 ? `${Math.round(v * 1000)} m` : v < 10 ? `${v.toFixed(1)} km` : `${Math.round(v)} km`);
const checkName = (n: string) => ({ units: "Units", duplicates: "Duplicates", time: "Sample times" })[n] ?? n.charAt(0).toUpperCase() + n.slice(1).replace(/_/g, " ");

/**
 * One sample, inspected: the number exactly as published, its unit and state in words, when (UTC and the
 * station's own clock), where, how deep, what else that visit measured, the checks, its own row at the source
 * and how to cite it. Previous and Next step through this variable's samples (not the station's visits). With
 * nothing selected it says how to select.
 */
export function Inspector({ record, onPrev, onNext, onClose, position }: { record: SampleRecord | null; onPrev: (() => void) | null; onNext: (() => void) | null; onClose: () => void; position: string | null }) {
  if (!record) {
    return (
      <section className={s.inspector} aria-label="Sample" data-testid="research-inspector" data-empty="">
        <h2 className={s.h2}>Sample</h2>
        <p className={s.quiet}>Select a sample to read it exactly: click or tap a mark on the chart, or focus the chart, step with the arrow keys and press Enter.</p>
        <p className={s.vh} role="status" aria-live="polite" data-testid="research-inspector-live" />
      </section>
    );
  }
  const r = record;
  const passed = r.qc.filter((q) => q.passed).length;
  return (
    <section className={s.inspector} aria-label={`Sample of ${r.utc}`} data-testid="research-inspector" data-index={r.index} data-state={r.state}>
      <header className={s.inspHead}>
        <div>
          <h2 className={s.h2}>Sample · {day(r.time, { year: true })}</h2>
          {position ? <p className={s.quiet}>{position}</p> : null}
        </div>
        <ButtonGroup align="end">
          <Button variant="secondary" onClick={onPrev ?? undefined} disabled={!onPrev} data-testid="inspector-prev">
            Previous
          </Button>
          <Button variant="secondary" onClick={onNext ?? undefined} disabled={!onNext} data-testid="inspector-next">
            Next
          </Button>
          <Button variant="quiet" onClick={onClose} data-testid="inspector-close">
            Close
          </Button>
        </ButtonGroup>
      </header>
      <p className={s.inspValue} data-testid="inspector-value">
        {r.state === "rejected" ? (
          <span className={s.inspFigure}>No value</span>
        ) : (
          <>
            <span className={s.inspFigure}>{r.valueText}</span> <span className={s.inspUnit}>{r.variable.unit}</span>
          </>
        )}
      </p>
      <p className={s.inspState} data-testid="inspector-state">
        {r.variable.label}. {r.stateWords}.
      </p>
      <dl className={s.inspFacts}>
        <div>
          <dt>Sampled, UTC</dt>
          <dd data-testid="inspector-utc">{r.utc}</dd>
        </div>
        <div>
          <dt>Sampled, Pacific</dt>
          <dd data-testid="inspector-local">{r.local}</dd>
        </div>
        <div>
          <dt>Depth</dt>
          <dd>{r.depthM == null ? "Not recorded" : `${r.depthM} m`}</dd>
        </div>
        <div>
          <dt>Location</dt>
          <dd>
            {r.station.name}
            {r.station.locationCode ? ` (${r.station.locationCode})` : ""}
            {r.station.lat != null && r.station.lon != null ? `, ${latLon(r.station.lat, r.station.lon)}` : ""}
            {r.station.nearestPort && r.station.nearestPortKm != null ? `, ${km(r.station.nearestPortKm)} from ${r.station.nearestPort}` : ""}
          </dd>
        </div>
        <div>
          <dt>Sample ID</dt>
          <dd data-testid="inspector-sampleid">
            At the source: Neritic&rsquo;s copy does not hold it.{" "}
            {r.sourceRowUrl ? (
              <a href={r.sourceRowUrl} target="_blank" rel="noreferrer">
                Open this sample&rsquo;s row
              </a>
            ) : null}
          </dd>
        </div>
        <div>
          <dt>Source column</dt>
          <dd>
            {r.variable.sourceColumn}, {r.station.id}
            {r.qualifier ? `; Neritic qualifier ${r.qualifier}` : ""}
          </dd>
        </div>
        {r.variable.detectionLimitNote ? (
          <div>
            <dt>Limit</dt>
            <dd>{r.variable.detectionLimitNote}</dd>
          </div>
        ) : null}
        {r.variable.method ? (
          <div>
            <dt>Method</dt>
            <dd>{r.variable.method}</dd>
          </div>
        ) : null}
      </dl>
      <h3 className={s.h3}>The same visit also recorded</h3>
      <ul className={s.inspOthers} data-testid="inspector-others">
        {r.others.map((o) => (
          <li key={o.v} data-state={o.state}>
            <span>{o.label}</span>
            <span className={s.num}>{o.words}</span>
          </li>
        ))}
      </ul>
      <h3 className={s.h3}>
        Checks: {passed} of {r.qc.length} passed
      </h3>
      <ul className={s.inspChecks} data-testid="inspector-qc">
        {r.qc.map((q) => (
          <li key={q.name}>
            <b>{q.passed ? "Passed" : "Failed"}: {checkName(q.name)}.</b> {q.detail}
          </li>
        ))}
      </ul>
      <p className={s.small}>Checks are made on the station&rsquo;s whole file: the source publishes no flag for a single sample.</p>
      <p className={s.inspLinks}>
        {r.sourceRowUrl ? (
          <a href={r.sourceRowUrl} target="_blank" rel="noreferrer" data-testid="inspector-source-row">
            This sample at SCCOOS ERDDAP
          </a>
        ) : null}
        <a href={r.sourceUrl} target="_blank" rel="noreferrer">
          The station&rsquo;s dataset
        </a>
      </p>
      <p className={s.small} data-testid="inspector-citation">
        {r.citation}
        {r.retrievedAt ? ` Retrieved ${day(r.retrievedAt, { year: true })}.` : ""}
      </p>
      <p className={s.vh} role="status" aria-live="polite" data-testid="research-inspector-live">
        Sample of {r.utc}: {r.words}. {r.stateWords}.
      </p>
    </section>
  );
}
