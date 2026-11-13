"use client";

import { Button, day, dayTime, Disclosure, Glyph, Micro, StampLine, type GlyphKind } from "@/nx/ds";
import { SOURCE_KEYS, SOURCE_LABEL } from "../snapshot";
import type { EvidenceItem, Snapshot } from "../types";
import s from "./evidence.module.css";

/**
 * "What Neritic showed you on <date>" (v15 §7): a planning snapshot (or a recheck) exactly as it was kept: its
 * evidence in the records' own words with their dates and sources, the forecast's issuance, the readings with their
 * ages, the reader's assumptions and what the evidence does not cover. Historical context, not an endorsement.
 * Superseded planning snapshots stay inspectable. Evidence first; the rest on demand.
 */
export type SnapshotViewProps = {
  snapshot: Snapshot;
  now: string;
  /** earlier planning snapshots kept after a refresh, oldest first */
  superseded?: Snapshot[];
  onOpen?: (id: string) => void;
  /** the evidence key to bring forward (from a change's "Evidence") */
  focusKey?: string | null;
};

const STATE_WORDS = { ok: "read", absent: "nothing applies here", failed: "could not be read", notChecked: "not checked" } as const;

function glyphOf(e: EvidenceItem): GlyphKind {
  if (e.kind === "season" || e.kind === "whale") return e.verified ? "measured" : "unverified-coral";
  if (e.kind === "advisory") return "unverified";
  return e.stale ? "old" : "measured";
}

export function SnapshotView({ snapshot: x, now, superseded = [], onOpen, focusKey = null }: SnapshotViewProps) {
  const title = x.kind === "planning" ? `What Neritic showed you on ${day(x.takenAt, { ref: now })}` : `What a recheck read on ${day(x.takenAt, { ref: now })}`;
  const official = x.evidence.filter((e) => e.kind !== "observation");
  const obs = x.evidence.filter((e) => e.kind === "observation");
  const f = x.forecast;
  return (
    <article className={s.snap} aria-label={title} data-testid="trip-snapshot" data-kind={x.kind} data-mode={x.mode}>
      <header className={s.snapHead}>
        <Micro>{x.kind === "planning" ? "Planning snapshot" : x.mode === "saved" ? "Recheck against a saved copy" : "Recheck"}</Micro>
        <h2 className={s.snapTitle}>{title}</h2>
        <p className={s.sub}>
          {x.mode === "saved" ? `From the copy saved on this device${x.savedFrom ? ` ${dayTime(x.savedFrom, { ref: now })}` : ""}; nothing was fetched.` : `Read ${dayTime(x.takenAt, { ref: now })}. Kept as it was; never overwritten.`}
        </p>
      </header>

      {official.length ? (
        <ul className={s.ev} aria-label="Official records and tests">
          {official.map((e) => (
            <li key={e.key} className={s.evItem} data-key={e.key} data-focus={focusKey === e.key || undefined} tabIndex={focusKey === e.key ? -1 : undefined}>
              <p className={s.evTitle}>
                <Glyph kind={glyphOf(e)} />
                {e.title}
              </p>
              <p className={s.evText}>{e.text}</p>
              {e.conflict ? <p className={s.evConflict}>Sources differ: {e.conflict}</p> : null}
              <StampLine glyph={glyphOf(e)}>
                {[e.source, e.scope, e.asOf ? `${e.kind === "cdphTest" ? "sampled" : e.kind === "advisory" ? "issued" : "checked"} ${/^\d{4}-\d{2}-\d{2}$/.test(e.asOf) ? day(e.asOf, { ref: now }) : dayTime(e.asOf, { ref: now })}` : null, e.kind === "season" || e.kind === "whale" ? (e.verified ? "verified by a person" : "not verified") : null, e.stale ? "old" : null]
                  .filter(Boolean)
                  .join(" · ")}
                {e.url ? (
                  <>
                    {" · "}
                    <a className={s.link} href={e.url} target="_blank" rel="noreferrer noopener">
                      Source ↗
                    </a>
                  </>
                ) : null}
              </StampLine>
            </li>
          ))}
        </ul>
      ) : (
        <p className={s.empty}>No official record or test was in this snapshot.</p>
      )}

      <div>
        <Disclosure summary="Marine forecast" value={f ? `${f.zone ?? "zone"}${f.issued ? ` · issued ${dayTime(f.issued, { ref: now })}` : ""}` : "not in this snapshot"}>
          {f && f.periods.length ? (
            <ul className={s.periods}>
              {f.periods.map((p) => (
                <li key={p.key}>
                  <b>{p.name}</b> · {p.text}
                </li>
              ))}
            </ul>
          ) : (
            <p className={s.empty}>{x.sources.nwsForecast.detail ?? "No forecast periods were kept."}</p>
          )}
          {f?.url ? (
            <a className={s.link} href={f.url} target="_blank" rel="noreferrer noopener">
              NWS zone forecast ↗
            </a>
          ) : null}
        </Disclosure>
        <Disclosure summary="Readings and tides" value={obs.length ? `${obs.length} ${obs.length === 1 ? "station" : "stations"}` : "none kept"}>
          <ul className={s.periods}>
            {obs.map((e) => (
              <li key={e.key}>
                <b>{e.title}</b> · {e.text}
                {e.asOf ? ` (measured ${dayTime(e.asOf, { ref: now })}${e.stale ? ", old when kept" : ""})` : ""}
              </li>
            ))}
            {x.tides?.events.length ? (
              <li>
                <b>Tides{x.tides.station ? `, ${x.tides.station}` : ""}</b> · {x.tides.events.slice(0, 4).map((t) => `${t.type === "H" ? "high" : "low"} ${t.ft} ft ${t.at.slice(11, 16)}`).join(", ")} (CO-OPS predictions)
              </li>
            ) : null}
          </ul>
        </Disclosure>
        <Disclosure summary="Sources" value={`${SOURCE_KEYS.filter((k) => x.sources[k].state === "ok" || x.sources[k].state === "absent").length} of ${SOURCE_KEYS.length} answered`}>
          <dl className={s.sources}>
            {SOURCE_KEYS.map((k) => (
              <div key={k} style={{ display: "contents" }}>
                <dt>{SOURCE_LABEL[k]}</dt>
                <dd data-state={x.sources[k].state} title={x.sources[k].detail ?? undefined}>
                  {STATE_WORDS[x.sources[k].state]}
                </dd>
              </div>
            ))}
          </dl>
        </Disclosure>
        <Disclosure summary="Your assumptions" value={x.assumptions.depart ? day(x.assumptions.depart.date, { ref: now }) : "no date set"}>
          <ul className={s.plain}>
            <li>Departure: {x.assumptions.depart ? `${day(x.assumptions.depart.date, { ref: now })}${x.assumptions.depart.time ? `, ${x.assumptions.depart.time}` : ", some time that day"}` : "not set"}</li>
            <li>Wind limit: {x.assumptions.limits.windKt != null ? `${x.assumptions.limits.windKt} kt` : "none set"}</li>
            <li>Sea limit: {x.assumptions.limits.seasFt != null ? `${x.assumptions.limits.seasFt} ft` : "none set"}</li>
            <li>Economics: {x.assumptions.economics ? "kept with this snapshot" : "none modelled"}</li>
          </ul>
        </Disclosure>
        <Disclosure summary="What this does not cover" value={`${x.limitations.length}`}>
          <ul className={s.plain}>
            {x.limitations.map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
        </Disclosure>
      </div>

      {superseded.length && onOpen ? (
        <div className={s.history}>
          <Micro as="h3">Earlier plans, kept</Micro>
          <div className={s.versions}>
            {superseded.map((p) => (
              <Button key={p.id} variant="secondary" glyph={null} onClick={() => onOpen(p.id)}>
                {dayTime(p.takenAt, { ref: now })}
              </Button>
            ))}
          </div>
        </div>
      ) : null}
    </article>
  );
}
