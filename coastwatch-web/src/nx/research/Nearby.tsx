"use client";

import Link from "next/link";
import { Glyph, Micro, type GlyphKind } from "@/nx/ds";
import { day } from "@/nx/ds/format";
import type { ObsVariableId } from "@/lib/observations";
import { RECENT_DAYS, STATE_WORDS, valueTimes, whereWords, reachWords, type CoverageState, type Located, type NearbyStation, type VariableReach, type ModelCoverage } from "./coverage";
import { DAY, NEAR_KM } from "./model";
import n from "./nearby.module.css";

export const NEARBY_SHORT: Record<ObsVariableId, string> = {
  pDA: "pDA",
  dDA: "dDA",
  tDA: "tDA",
  pn_seriata: "Pseudo-nitzschia (seriata)",
  pn_delicatissima: "Pseudo-nitzschia (delicatissima)",
  chl_extracted: "chlorophyll",
  temp: "water temperature",
};
const ROW: Record<ObsVariableId, string> = {
  pDA: "pDA",
  dDA: "dDA",
  tDA: "tDA",
  pn_seriata: "Seriata",
  pn_delicatissima: "Delicatissima",
  chl_extracted: "Chlorophyll",
  temp: "Temperature",
};

const GLYPH: Record<CoverageState, GlyphKind> = { recent: "measured", sparse: "measured", old: "old", notCollected: "none", unavailable: "none" };

/**
 * "Near Crescent City": the shore stations around a place, nearest first, each with its distance and direction and
 * one row per variable it has measured: every sample with a value as a tick on one shared time axis (so a weekly
 * record and a sparse one are told apart at a glance), and the newest sample's date with its state. A row opens
 * that station on that variable; its date jumps to that latest sample. Variables no listed station measures get
 * one sentence each, naming the nearest station that does, with its distance: never as this water's record.
 */
export function Nearby({
  placeName,
  rows,
  reach,
  model,
  extent,
  now,
  selected,
  focus,
  exploreHref,
  drawTicks,
  onOpen,
  onLatest,
  onClose,
}: {
  placeName: string;
  rows: NearbyStation<Located>[];
  reach: VariableReach<Located>[];
  model: ModelCoverage | null;
  /** the shared time axis: [first sample of any listed row, now] */
  extent: [number, number];
  now: number;
  selected: string;
  focus: ObsVariableId;
  exploreHref: string;
  drawTicks: boolean;
  onOpen: (station: string, v: ObsVariableId) => void;
  onLatest: (station: string, v: ObsVariableId) => void;
  onClose?: () => void;
}) {
  const near = rows.filter((r) => r.near);
  const listed = new Set(rows.flatMap((r) => r.measures));
  const elsewhere = reach.filter((r) => !listed.has(r.v));
  const span = Math.max(DAY, extent[1] - extent[0]);
  const pct = (t: number) => Math.min(100, Math.max(0, ((t - extent[0]) / span) * 100));
  const y0 = new Date(extent[0]).getUTCFullYear() + 1;
  const y1 = new Date(extent[1]).getUTCFullYear();
  const years: number[] = [];
  const stepY = y1 - y0 > 8 ? 4 : 2;
  for (let y = y0; y <= y1; y++) if ((y1 - y) % stepY === 0) years.push(y);
  const first = rows[0];

  return (
    <div className={n.wrap} data-testid="research-nearby" data-near={near.length}>
      <div className={n.head}>
        <Micro as="h2" className={n.kicker}>
          Near {placeName} · shore stations
        </Micro>
        <div className={n.links}>
          <Link href={exploreHref} className={n.link} data-testid="research-statewide">
            Statewide ocean map
          </Link>
          {onClose ? (
            <button type="button" className={n.link} onClick={onClose} data-testid="research-nearby-close">
              Back to the record
            </button>
          ) : null}
        </div>
      </div>
      <p className={n.lead} data-testid="research-nearby-lead">
        {near.length
          ? `${near.length === 1 ? "One station samples" : `${near.length} stations sample`} within ${NEAR_KM} km of ${placeName}.`
          : first
            ? `Nothing samples within ${NEAR_KM} km of ${placeName}. The nearest station is ${first.station.name}, ${whereWords(first)}: its record is of that water.`
            : `No shore station is in Neritic's copy.`}
      </p>

      <div className={n.axis} aria-hidden="true">
        <span />
        <span className={n.axisTrack}>
          {years.map((y) => (
            <i key={y} style={{ left: `${pct(Date.UTC(y, 0, 1))}%` }}>
              {y}
            </i>
          ))}
        </span>
        <span className={n.axisEnd}>Latest sample</span>
      </div>

      <ol className={n.stations} data-testid="research-nearby-stations">
        {rows.map((r) => {
          const measured = r.vars.filter((c) => c.n > 0);
          const not = r.vars.filter((c) => c.state === "notCollected");
          return (
            <li key={r.station.station_id} className={n.station} data-testid={`nearby-${r.station.station_id}`} data-near={r.near || undefined} aria-current={r.station.station_id === selected ? "true" : undefined}>
              <div className={n.stationHead}>
                <b>{r.station.name}</b>
                <span data-testid={`nearby-where-${r.station.station_id}`}>
                  {whereWords(r)}
                  {r.near ? "" : " · beyond 60 km"}
                </span>
              </div>
              {r.station.status === "failed" && !measured.length ? (
                <p className={n.not} data-testid={`nearby-unavailable-${r.station.station_id}`}>
                  <Glyph kind="none" /> {STATE_WORDS.unavailable}: the station&rsquo;s last request failed.
                </p>
              ) : null}
              <ul className={n.vars}>
                {measured.map((c) => {
                  const ts = drawTicks ? valueTimes(r.station, c.v) : [];
                  const on = r.station.station_id === selected && c.v === focus;
                  const cad = c.cadenceDays == null ? null : c.cadenceDays <= 8 ? "weekly" : `${Math.round(c.cadenceDays)} d apart`;
                  return (
                    <li key={c.v} className={n.var} data-state={c.state} data-testid={`nearby-${r.station.station_id}-${c.v}`}>
                      <button type="button" className={n.open} aria-current={on ? "true" : undefined} onClick={() => onOpen(r.station.station_id, c.v)} aria-label={`Open ${ROW[c.v]} at ${r.station.name}, ${c.n} samples since ${c.first ? day(c.first, { year: true }) : ""}`}>
                        <span className={n.name}>{ROW[c.v]}</span>
                        <span className={n.track}>
                          {ts.map((t, i) => (
                            <i key={i} style={{ left: `${pct(t)}%` }} />
                          ))}
                        </span>
                      </button>
                      <button
                        type="button"
                        className={n.latest}
                        onClick={() => onLatest(r.station.station_id, c.v)}
                        data-testid={`nearby-latest-${r.station.station_id}-${c.v}`}
                        aria-label={`Latest ${ROW[c.v]} sample at ${r.station.name}: ${c.last ? day(c.last, { year: true }) : ""}`}
                      >
                        <Glyph kind={GLYPH[c.state]} />
                        <span>
                          {c.last ? day(c.last, { year: true }) : ""}
                          <small>
                            {STATE_WORDS[c.state]}
                            {cad ? ` · ${cad}` : ""}
                          </small>
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
              {not.length ? (
                <p className={n.not} data-testid={`nearby-not-${r.station.station_id}`}>
                  <Glyph kind="none" /> Not collected here: {not.map((c) => NEARBY_SHORT[c.v]).join(", ")}
                </p>
              ) : null}
            </li>
          );
        })}
      </ol>

      {elsewhere.length ? (
        <ul className={n.elsewhere} data-testid="research-nearby-elsewhere">
          {elsewhere.map((r) => (
            <li key={r.v} data-testid={`nearby-reach-${r.v}`}>
              <span>{reachWords(r, NEARBY_SHORT[r.v], null)}</span>
              {r.nearest ? (
                <button type="button" className={n.link} onClick={() => onOpen(r.nearest!.station.station_id, r.v)}>
                  Open its record
                </button>
              ) : null}
            </li>
          ))}
        </ul>
      ) : null}

      {model && model.days ? (
        <p className={n.model} data-testid="research-nearby-model">
          <Glyph kind="predicted" /> <b>Modelled, not sampled:</b> C-HARM bloom probabilities are kept for {model.days} days at each station (since {day(model.first!, { year: true })}), the median of model cells within {model.radiusKm} km. Open a record and
          turn on &ldquo;Show the model&rdquo; to read it beside the samples.
        </p>
      ) : null}
      <p className={n.legend} aria-hidden="true">
        <span>
          <Glyph kind="measured" /> sample within {RECENT_DAYS} days
        </span>
        <span>
          <Glyph kind="old" /> older: valid for its day, says nothing about now
        </span>
        <span>
          <Glyph kind="none" /> not collected
        </span>
        <span>Each tick is one sample with a value; no tick is no sample, never zero.</span>
      </p>
      <span hidden data-testid="research-nearby-now" data-now={now} />
    </div>
  );
}
