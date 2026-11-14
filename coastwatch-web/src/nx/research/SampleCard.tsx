"use client";

import { Button, Glyph, Micro, RollingNumber } from "@/nx/ds";
import { day, latLon } from "@/nx/ds/format";
import type { ObsVariableId } from "@/lib/observations";
import type { SampleRecord } from "./availability";
import { figure, type RecPoint, type Strip } from "./model";
import s from "./rx.module.css";

const km = (v: number) => (v < 1 ? `${Math.round(v * 1000)} m` : v < 10 ? `${v.toFixed(1)} km` : `${Math.round(v)} km`);
const checkName = (n: string) => ({ units: "Units", duplicates: "Duplicates", time: "Sample times" })[n] ?? n.charAt(0).toUpperCase() + n.slice(1).replace(/_/g, " ");

/** What a visit's other measurements are called on the card (the prototype's `VISIT_NAMES`): short, the unit beside the value. */
const VISIT_NAME: Record<ObsVariableId, string> = {
  pDA: "Domoic acid (pDA)",
  dDA: "Dissolved domoic acid",
  tDA: "Total domoic acid",
  pn_seriata: "Pseudo-nitzschia, seriata",
  pn_delicatissima: "Pseudo-nitzschia, delicatissima",
  chl_extracted: "Chlorophyll-a",
  temp: "Water",
};

/** "May 27, 2026, 7:45 AM PDT" → ["May 27, 2026", "7:45 AM PDT"]. */
function splitLocal(local: string): [string, string] {
  const m = /^([A-Z][a-z]{2} \d{1,2}, \d{4}),? (.*)$/.exec(local);
  return m ? [m[1], m[2]] : [local, ""];
}

/** The sample against the model's reference level, in the prototype's words (never a safety limit). */
export function refMultiple(strip: Pick<Strip, "threshold">, p: { cls: RecPoint["cls"]; v: number | null }): string | null {
  const thr = strip.threshold;
  if (!thr || p.cls !== "value" || p.v == null) return null;
  if (p.v >= thr.value * 1.5) return `${Math.round(p.v / thr.value).toLocaleString("en-US")}× the model’s reference level`;
  if (p.v >= thr.value) return "At the model’s reference level";
  return "Below the model’s reference level";
}

export type SampleCardProps = {
  record: SampleRecord | null;
  point: RecPoint | null;
  strip: Strip | null;
  /** "Sample 603 of 625" */
  n: number | null;
  total: number;
  outside: boolean;
  prev: RecPoint | null;
  next: RecPoint | null;
  onStep: (dir: -1 | 1) => void;
  recordOpen: boolean;
  onRecord: () => void;
  /** phone: bring the sample's details into view in the sheet */
  onDetails: () => void;
};

/**
 * The sample card in the pass's stub (v14 synthesis, Research): which sample of how many, its value rolling to the page's
 * display precision, when (Pacific and UTC), the number exactly as published, its multiple of the model's reference
 * level, what the same visit measured, and the record (the one filled action) and the source row. Previous and next
 * step through this variable's samples. With nothing selected it says how to pick one.
 */
export function SampleCard({ record: r, point, strip, n, total, outside, prev, next, onStep, recordOpen, onRecord, onDetails }: SampleCardProps) {
  if (!r || !point || !strip) {
    return (
      <div className={s.card} data-testid="research-inspector" data-empty="">
        <div className={s.lead}>
          <Micro glyph="none">No sample selected</Micro>
          <p className={s.pick}>Pick a sample</p>
          <p className={s.pickHow}>Click a mark on the plot, or focus the plot and step with the arrow keys.</p>
        </div>
      </div>
    );
  }
  const zero = r.state === "reported_zero";
  const rejected = r.state === "rejected";
  const [date, time] = splitLocal(r.local);
  const ratio = refMultiple(strip, point);
  const others = r.others.filter((o) => o.state !== "absent");
  const step = (o: RecPoint | null, dir: -1 | 1) => (
    <button
      type="button"
      className={s.stepBtn}
      aria-disabled={o ? undefined : "true"}
      aria-label={o ? `${dir < 0 ? "Previous" : "Next"} sample, ${day(o.t, { year: true })}` : `No ${dir < 0 ? "earlier" : "later"} sample`}
      title={o ? day(o.t, { year: true }) : undefined}
      onClick={() => o && onStep(dir)}
      data-testid={dir < 0 ? "inspector-prev" : "inspector-next"}
    >
      <span aria-hidden="true">{dir < 0 ? "‹" : "›"}</span>
    </button>
  );
  return (
    <div className={s.card} data-testid="research-inspector">
      <div data-testid="inspector-body" data-index={r.index} data-state={r.state} className={s.cardBody}>
        <div className={s.lead}>
          <Micro glyph={zero ? "zero" : rejected ? "none" : "measured"} data-testid="inspector-position">
            Sample {n?.toLocaleString("en-US")} of {total.toLocaleString("en-US")}
            {outside ? " · outside this range" : ""}
          </Micro>
          <div className={s.steps}>
            {step(prev, -1)}
            {step(next, 1)}
          </div>
          {zero || rejected ? (
            <p className={s.words}>{zero ? "Reported 0" : "No value"}</p>
          ) : (
            <p className={s.fig}>
              <RollingNumber value={r.value} format={(v) => figure(strip, v)} srSuffix={r.variable.unit} pace="slow" data-testid="inspector-figure" />
              <small>{r.variable.unit}</small>
            </p>
          )}
          <p className={s.when}>
            <span data-testid="inspector-local">
              {date} · {time}
            </span>{" "}
            <span className={s.utc} data-testid="inspector-utc">
              {r.utc}
            </span>
          </p>
          <button type="button" className={s.detailsBtn} onClick={onDetails} data-testid="sample-details">
            Details
          </button>
        </div>
        <div className={s.detail} id="rx-sample-detail">
          <p className={s.exact}>
            {zero ? (
              <>
                At or below the lab’s limit, not absent (published as <span data-testid="inspector-value">0 {r.variable.unit}</span>)
              </>
            ) : rejected ? (
              <>Rejected by QC: the source published a negative value, so no value is shown or counted</>
            ) : (
              <>
                Measured, as published: <span data-testid="inspector-value">{`${r.valueText} ${r.variable.unit}`}</span>
                {r.state === "flagged_high" ? " (flagged high, kept as published)" : ""}
              </>
            )}
          </p>
          {ratio ? (
            <p className={s.ratio} data-testid="inspector-ratio">
              <Glyph kind="predicted" />
              {ratio}
            </p>
          ) : null}
          {others.length ? (
            <div className={s.visit} data-testid="inspector-others" data-count={others.length}>
              <Micro as="p" className={s.visitHead}>
                Same visit
              </Micro>
              <dl className={s.visitList}>
                {others.map((o) => (
                  <div key={o.v} data-state={o.state}>
                    <dt title={VISIT_NAME[o.v]}>{VISIT_NAME[o.v]}</dt>
                    <dd>
                      {o.state === "zero" ? (
                        <small>reported 0, not quantified</small>
                      ) : o.state === "rejected" ? (
                        <small>rejected by QC</small>
                      ) : (
                        <>
                          {figure({ kind: "measured", v: o.v }, o.value as number)} <small>{o.unit}</small>
                        </>
                      )}
                    </dd>
                  </div>
                ))}
              </dl>
            </div>
          ) : null}
          <div className={s.acts}>
            <Button variant="primary" glyph={recordOpen ? "×" : "→"} onClick={onRecord} aria-expanded={recordOpen} aria-controls="rx-record" data-testid="research-open-record" className={s.recordBtn}>
              {recordOpen ? "Close the record" : "Sample record"}
            </Button>
            {r.sourceRowUrl ? (
              <Button variant="quiet" href={r.sourceRowUrl} external data-testid="inspector-source-row">
                Source
              </Button>
            ) : null}
          </div>
        </div>
        <p className={s.vh} data-testid="research-inspector-live">
          Sample of {r.utc}: {r.words}. {r.stateWords}.
        </p>
      </div>
    </div>
  );
}

/**
 * The sample's record (the prototype's `.rs-record`), opened from the card over the pass's main part (in the sheet on a
 * phone): the value as published, when, depth, where, the source column, the checks, the method, the limit and how to
 * cite it, with the source row and the dataset one step away.
 */
export function SampleRecordPanel({ record: r, n, total, short, onClose }: { record: SampleRecord; n: number | null; total: number; short: string; onClose: () => void }) {
  const zero = r.state === "reported_zero";
  const passed = r.qc.filter((q) => q.passed).length;
  const link = { target: "_blank", rel: "noreferrer" } as const;
  const where = `${r.station.name}${r.station.locationCode ? ` (${r.station.locationCode})` : ""}${r.station.lat != null && r.station.lon != null ? `, ${latLon(r.station.lat, r.station.lon)}` : ""}${r.station.nearestPort && r.station.nearestPortKm != null ? `; ${km(r.station.nearestPortKm)} from ${r.station.nearestPort}` : ""}`;
  return (
    <div className={s.recHead}>
      <Micro glyph={zero ? "zero" : r.state === "rejected" ? "none" : "measured"}>
        Sample record · {n?.toLocaleString("en-US")} of {total.toLocaleString("en-US")} {short} samples
      </Micro>
      <Button variant="quiet" onClick={onClose} data-testid="research-record-close">
        Close
      </Button>
      <dl className={s.rr} data-testid="research-sample-record-facts">
        <div>
          <Micro as="dt">Value, as published</Micro>
          <dd data-testid="inspector-state">
            {r.state === "rejected" ? "No value. " : zero ? `0 ${r.variable.unit}. ` : `${r.valueText} ${r.variable.unit}. `}
            {r.variable.label}. {r.stateWords}.
          </dd>
        </div>
        <div>
          <Micro as="dt">Sampled</Micro>
          <dd>
            {r.utc}
            <br />
            {r.local}
          </dd>
        </div>
        <div>
          <Micro as="dt">Depth</Micro>
          <dd>{r.depthM == null ? "Not recorded" : `${r.depthM} m`}</dd>
        </div>
        <div>
          <Micro as="dt">Location</Micro>
          <dd>{where}</dd>
        </div>
        <div>
          <Micro as="dt">Source column</Micro>
          <dd>
            {r.variable.sourceColumn}, {r.station.id}
            {r.qualifier ? `; Neritic qualifier ${r.qualifier}` : ""}
          </dd>
        </div>
        <div>
          <Micro as="dt">Sample ID</Micro>
          <dd data-testid="inspector-sampleid">
            At the source: Neritic’s copy does not hold it.{" "}
            {r.sourceRowUrl ? (
              <a href={r.sourceRowUrl} {...link}>
                Open this sample’s row
              </a>
            ) : null}
          </dd>
        </div>
        <div className={s.wide}>
          <Micro as="dt">
            QC: {passed} of {r.qc.length} passed
          </Micro>
          <dd>
            <ul className={s.qc} data-testid="inspector-qc">
              {r.qc.map((q) => (
                <li key={q.name}>
                  <b>
                    {q.passed ? "Passed" : "Failed"}: {checkName(q.name)}.
                  </b>{" "}
                  {q.detail}
                </li>
              ))}
            </ul>
            Checks are made on the station’s whole file: the source publishes no flag for a single sample.
          </dd>
        </div>
        {r.variable.method ? (
          <div className={s.wide}>
            <Micro as="dt">Method</Micro>
            <dd>{r.variable.method}</dd>
          </div>
        ) : null}
        {r.variable.detectionLimitNote ? (
          <div className={s.wide}>
            <Micro as="dt">Limit</Micro>
            <dd>{r.variable.detectionLimitNote}</dd>
          </div>
        ) : null}
        <div className={s.wide}>
          <Micro as="dt">Citation</Micro>
          <dd data-testid="inspector-citation">
            {r.citation}
            {r.retrievedAt ? ` Retrieved ${day(r.retrievedAt, { year: true })}.` : ""}
          </dd>
        </div>
      </dl>
      <p className={s.recLinks}>
        {r.sourceRowUrl ? (
          <Button variant="quiet" href={r.sourceRowUrl} external>
            Source row at SCCOOS ERDDAP
          </Button>
        ) : null}
        <Button variant="quiet" href={r.sourceUrl} external>
          Dataset
        </Button>
      </p>
    </div>
  );
}
