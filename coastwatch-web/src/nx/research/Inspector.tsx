"use client";

import { Button, ButtonGroup, Facts, Inspector as Card } from "@/nx/ds";
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
export function Inspector({ record, onPrev, onNext, onClose, position, className }: { record: SampleRecord | null; onPrev: (() => void) | null; onNext: (() => void) | null; onClose: () => void; position: string | null; className?: string }) {
  if (!record) {
    return (
      <p className={s.inspHint} data-testid="research-inspector" data-empty="">
        Select a sample to read it exactly: click or tap a mark on the plot, or focus the plot, step with the arrow keys and press Enter.
      </p>
    );
  }
  const r = record;
  const passed = r.qc.filter((q) => q.passed).length;
  const link = { target: "_blank", rel: "noreferrer" } as const;
  return (
    <Card
      title={`Sample · ${day(r.time, { year: true })}`}
      subtitle={position}
      onClose={onClose}
      className={className}
      data-testid="research-inspector"
    >
      <div data-testid="inspector-body" data-index={r.index} data-state={r.state}>
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
        <ButtonGroup className={s.inspStep}>
          <Button variant="secondary" onClick={onPrev ?? undefined} disabled={!onPrev} data-testid="inspector-prev">
            Previous
          </Button>
          <Button variant="secondary" onClick={onNext ?? undefined} disabled={!onNext} data-testid="inspector-next">
            Next
          </Button>
        </ButtonGroup>
        <Facts
          items={[
            { label: "Sampled, UTC", value: <span data-testid="inspector-utc">{r.utc}</span> },
            { label: "Sampled, Pacific", value: <span data-testid="inspector-local">{r.local}</span> },
            { label: "Depth", value: r.depthM == null ? "Not recorded" : `${r.depthM} m` },
            {
              label: "Location",
              value: `${r.station.name}${r.station.locationCode ? ` (${r.station.locationCode})` : ""}${r.station.lat != null && r.station.lon != null ? `, ${latLon(r.station.lat, r.station.lon)}` : ""}${r.station.nearestPort && r.station.nearestPortKm != null ? `, ${km(r.station.nearestPortKm)} from ${r.station.nearestPort}` : ""}`,
            },
            {
              label: "Sample ID",
              value: (
                <span data-testid="inspector-sampleid">
                  At the source: Neritic&rsquo;s copy does not hold it.{" "}
                  {r.sourceRowUrl ? (
                    <a href={r.sourceRowUrl} {...link} className={s.inline}>
                      Open this sample&rsquo;s row
                    </a>
                  ) : null}
                </span>
              ),
            },
            { label: "Source column", value: `${r.variable.sourceColumn}, ${r.station.id}${r.qualifier ? `; Neritic qualifier ${r.qualifier}` : ""}` },
            ...(r.variable.detectionLimitNote ? [{ label: "Limit", value: r.variable.detectionLimitNote }] : []),
            ...(r.variable.method ? [{ label: "Method", value: r.variable.method }] : []),
          ]}
        />
        <h4 className={s.h3}>The same visit also recorded</h4>
        <ul className={s.inspOthers} data-testid="inspector-others">
          {r.others.map((o) => (
            <li key={o.v} data-state={o.state}>
              <span>{o.label}</span>
              <span className={s.num}>{o.words}</span>
            </li>
          ))}
        </ul>
        <h4 className={s.h3}>
          Checks: {passed} of {r.qc.length} passed
        </h4>
        <ul className={s.inspChecks} data-testid="inspector-qc">
          {r.qc.map((q) => (
            <li key={q.name}>
              <b>
                {q.passed ? "Passed" : "Failed"}: {checkName(q.name)}.
              </b>{" "}
              {q.detail}
            </li>
          ))}
        </ul>
        <p className={s.small}>Checks are made on the station&rsquo;s whole file: the source publishes no flag for a single sample.</p>
        {/* in the body, not the card's foot: on a phone the pinned card is half the screen and the facts need the room */}
        <p className={s.inspLinks}>
        {r.sourceRowUrl ? (
          <a href={r.sourceRowUrl} {...link} data-testid="inspector-source-row">
            This sample at SCCOOS ERDDAP
          </a>
        ) : null}
        <a href={r.sourceUrl} {...link}>
          The station&rsquo;s dataset
        </a>
        </p>
        <p className={s.small} data-testid="inspector-citation">
          {r.citation}
          {r.retrievedAt ? ` Retrieved ${day(r.retrievedAt, { year: true })}.` : ""}
        </p>
        <p className={s.vh} data-testid="research-inspector-live">
          Sample of {r.utc}: {r.words}. {r.stateWords}.
        </p>
      </div>
    </Card>
  );
}
