"use client";

import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { Button, Glyph, Micro, PageLayout } from "@/nx/ds";
import { age, day, dayTime, time } from "@/nx/ds/format";
import { keepTripsOffline } from "@/nx/offline/trips";
import { usePorts } from "@/nx/shell/port";
import { diffSnapshots } from "../diff";
import type { Snapshot, Trip } from "../types";
import { briefing, type Briefing as Model, type BriefPeriod } from "./briefingModel";
import { allSnapshots, readTrip, tripSnapshots } from "./localTrips";
import { knownSnapshot } from "../known";
import { TripChart } from "./TripChart";
import s from "./briefing.module.css";

/**
 * The departure briefing (v15): one printable page of a trip, read entirely from this device's storage, so it opens
 * and prints with no network once the trips page has been opened online (public/sw.js, src/nx/offline/trips.ts).
 *
 * Port, fishery and date; the reader's own limits; what the evidence rests on (the latest recheck with its time and
 * whether every source answered, or the planning snapshot with its age); the departure periods against those limits;
 * the official items with their verification state and links; the economics headline. A saved copy is said to be one;
 * nothing is called current, safe or allowed.
 */
export type BriefingProps = {
  tripId: string;
  /** for tests and the catalog: the trip and snapshots instead of storage */
  data?: { trip: Trip; planning: Snapshot | null; latest: Snapshot | null; all?: Snapshot[] } | null;
  /** where "Back to the trip" goes (default: the trip's workspace) */
  backHref?: string;
};

type Loaded = { state: "loading" } | { state: "missing" | "unreadable" | "newer" } | { state: "ok"; trip: Trip; planning: Snapshot | null; latest: Snapshot | null; all?: Snapshot[] };

export function Briefing({ tripId, data, backHref }: BriefingProps) {
  const [loaded, setLoaded] = useState<Loaded>(data ? { state: "ok", ...data } : { state: "loading" });
  const [now, setNow] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setNow(new Date().toISOString());
    if (data) return;
    const read = () => {
      const r = readTrip(tripId);
      if (r.state !== "ok") return setLoaded({ state: r.state });
      setLoaded({ state: "ok", trip: r.value, ...tripSnapshots(r.value), all: allSnapshots(r.value) });
    };
    read();
    // another tab saved a recheck or an edit: the briefing follows it
    const on = (e: StorageEvent) => (e.key === null || e.key.startsWith("nx-trips-v1:")) && read();
    window.addEventListener("storage", on);
    return () => window.removeEventListener("storage", on);
  }, [tripId, data]);
  useEffect(() => {
    setMounted(true);
    void keepTripsOffline();
  }, []);

  const ports = usePorts();
  const model = useMemo(() => {
    if (loaded.state !== "ok") return null;
    const { trip, planning, latest, all } = loaded;
    const changes = planning && latest && now ? diffSnapshots(planning, latest, now) : [];
    const port = ports.find((p) => p.key === (trip.departKey ?? trip.placeKey)) ?? null;
    // the chart draws what the device knows (a failed or offline read never removes a known buoy or test site)
    const snaps = all ?? [planning, latest].filter((x): x is Snapshot => !!x);
    const shown = snaps.length ? knownSnapshot(trip, snaps, now ?? trip.updatedAt, `${trip.id}-known`).snapshot : null;
    return { m: briefing(trip, { planning, latest, all: snaps }, { portName: port?.name ?? null, changes }), port, trip, shown };
  }, [loaded, now, ports]);

  if (loaded.state === "loading")
    return (
      <>
        <PageLayout mode="desk" paneKey="trips-briefing" />
        <section className={s.page} aria-busy="true" />
      </>
    );
  if (!model) {
    const words = loaded.state === "newer" ? "This trip was saved by a newer version of Neritic. Open it there." : loaded.state === "unreadable" ? "This trip's record on this device cannot be read. It is kept as it is." : "No trip with this id is saved on this device.";
    return (
      <section className={s.page} data-testid="trip-briefing" data-state={loaded.state}>
        <PageLayout mode="desk" paneKey="trips-briefing" />
        <h1 className={s.title}>Departure briefing</h1>
        <p className={s.note}>{words}</p>
        <Button variant="secondary" href="/coast/trips">
          Your trips
        </Button>
      </section>
    );
  }
  return (
    <>
      {/* a page to read and print: the desk layout (a wide panel, the chart beside it), not the narrow chart card */}
      <PageLayout mode="desk" paneKey="trips-briefing" chartLabel={model.port?.name} />
      {/* the chart beside the briefing frames the trip's own water, with the same marks as the workspace */}
      {model.port ? <TripChart trip={model.trip} snapshot={model.shown} port={model.port} variant="scene" /> : null}
      <div className={s.wrap}>
        <Body {...model} now={now} backHref={backHref} />
      </div>
      {/* the print copy: outside the app's panels (which clip and scroll), shown only on paper */}
      {mounted ? createPortal(<div className={s.printRoot} data-print-root="" aria-hidden="true"><Body {...model} now={now} backHref={backHref} print /></div>, document.body) : null}
    </>
  );
}

type BodyProps = { m: Model; port: { key: string; name: string; lon: number; lat: number } | null; trip: Trip; shown: Snapshot | null; now: string | null; backHref?: string; print?: boolean };

function Body({ m, port, trip, shown, now, backHref, print }: BodyProps) {
  // the print copy carries no ids or test ids: one page, one set of names
  const id = (x: string) => (print ? undefined : x);
  const tid = (x: string) => (print ? undefined : x);
  return (
    <article className={s.page} data-testid={tid("trip-briefing")} aria-labelledby={id("briefing-title")}>
      <header className={s.head}>
        <Micro>Departure briefing</Micro>
        <h1 id={id("briefing-title")} className={s.title}>
          {port?.name ?? m.title}
        </h1>
        <p className={s.sub}>
          {m.fishery}
          {m.activity === "recreational" ? "" : ", commercial"}
          {m.area ? ` · ${m.area}` : ""}
        </p>
        <p className={s.date} data-testid={tid("briefing-date")}>
          {m.depart ? `${weekdayDate(m.depart.date)}${m.depart.time ? `, ${m.depart.time}` : ""}` : "No date set"}
        </p>
        <dl className={s.limits} aria-label="Your limits">
          <div>
            <dt>Your wind limit</dt>
            <dd>{m.limits.windKt != null ? `${m.limits.windKt} kt` : "None set"}</dd>
          </div>
          <div>
            <dt>Your seas limit</dt>
            <dd>{m.limits.seasFt != null ? `${m.limits.seasFt} ft` : "None set"}</dd>
          </div>
        </dl>
      </header>

      <Basis m={m} now={now} print={print} />

      <section className={s.periods} aria-labelledby={id("briefing-periods")}>
        <h2 id={id("briefing-periods")} className={s.h2}>
          Departure periods
        </h2>
        {m.periods.length ? (
          <ol className={s.strip}>
            {m.periods.map((p) => (
              <Period key={p.key} p={p} print={print} />
            ))}
          </ol>
        ) : (
          <p className={s.slot}>
            <Glyph kind="none" /> {m.periodsWhy}
          </p>
        )}
        {m.forecast ? (
          <p className={s.prov}>
            <Glyph kind="predicted" /> NWS {m.forecast.zone ?? "zone forecast"}
            {m.forecast.issued ? `, issued ${dayTime(m.forecast.issued)}` : ""}. Period times derived from NWS period names. Your limits are your own, not a judgement of the sea.
            {m.forecast.held.latest ? null : <HeldAge held={m.forecast.held} now={now} print={print} />}
          </p>
        ) : null}
      </section>

      {m.changes.length ? (
        <section className={s.changes} aria-labelledby={id("briefing-changes")}>
          <h2 id={id("briefing-changes")} className={s.h2}>
            Since you planned
          </h2>
          <ul className={s.list}>
            {m.changes.slice(0, 6).map((c) => (
              <li key={c.id}>
                <span className={s.rowTitle}>{c.title}</span>
                {c.url ? (
                  <a className={s.link} href={c.url} target="_blank" rel="noreferrer">
                    Source
                  </a>
                ) : null}
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <section className={s.official} aria-labelledby={id("briefing-official")}>
        <h2 id={id("briefing-official")} className={s.h2}>
          Official and measured records
        </h2>
        {m.official.length ? (
          <ul className={s.list}>
            {m.official.map((o) => (
              <li key={o.key} data-testid={id("briefing-official")}>
                <Glyph kind={o.source === "CDFW" ? (o.verified ? "measured" : "unverified-coral") : o.kind === "cdphTest" ? (o.stale ? "old" : "measured") : "unverified"} />
                <div>
                  <p className={s.rowTitle}>
                    {o.title.startsWith(o.source) ? o.title : `${o.source}: ${o.title}`}
                  </p>
                  <p className={s.rowText}>{o.text}</p>
                  <p className={s.prov}>
                    {o.asOf ? `${o.kind === "cdphTest" ? "Collected" : "As of"} ${day(o.asOf)}` : "No date given"}
                    {o.held.latest ? null : <HeldAge held={o.held} now={now} print={print} />}
                    {o.source === "CDFW" ? (o.verified ? " · checked by a person" : " · not verified by Neritic") : ""}
                    {o.conflict ? ` · sources differ: ${o.conflict}` : ""}
                    {o.url ? (
                      <>
                        {" · "}
                        <a className={s.link} href={o.url} target="_blank" rel="noreferrer">
                          {o.source} page
                        </a>
                      </>
                    ) : null}
                  </p>
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <p className={s.slot}>
            <Glyph kind="none" /> No official records were saved with this trip.
          </p>
        )}
      </section>

      <aside className={s.side}>
        {port ? <TripChart trip={trip} snapshot={shown} port={port} variant="plate" /> : null}
        <Economics m={m} print={print} />
      </aside>

      <footer className={s.foot}>
        <p className={s.prov}>Kept on this device. Call CDFW (831) 649-2883 and CDPH 1 (800) 553-4133 to confirm closures before you leave port.</p>
        <div className={s.actions}>
          <Button variant="secondary" glyph={null} onClick={() => window.print()}>
            Print
          </Button>
          <Button variant="quiet" href={backHref ?? `/coast/trips?id=${encodeURIComponent(trip.id)}`}>
            Back to the trip
          </Button>
        </div>
      </footer>
    </article>
  );
}

/** A known value that is not from the newest check: when it was read, and how long ago. */
function HeldAge({ held, now, print }: { held: { readAt: string }; now: string | null; print?: boolean }) {
  return (
    <span className={s.held} data-testid={print ? undefined : "briefing-known-age"}>
      {` · last read ${dayTime(held.readAt)}${now ? `, ${age(held.readAt, now)}` : ""}; the latest check did not read it`}
    </span>
  );
}

const weekdayDate = (ymd: string) => new Date(`${ymd}T12:00:00Z`).toLocaleDateString("en-US", { weekday: "long", month: "short", day: "numeric", timeZone: "UTC" });

function Basis({ m, now, print }: { m: Model; now: string | null; print?: boolean }) {
  const tid = (x: string) => (print ? undefined : x);
  const b = m.basis;
  if (!b)
    return (
      <p className={s.basis} data-testid={tid("briefing-basis")} data-complete="false">
        <Glyph kind="none" /> No evidence saved with this trip yet.
      </p>
    );
  const saved = b.mode === "saved";
  const lead = b.kind === "recheck" ? (saved ? "Compared with the copy saved on this device" : "Rechecked") : "Planning snapshot";
  return (
    <div className={s.basis} data-testid={tid("briefing-basis")} data-complete={b.complete ? "true" : "false"} data-kind={b.kind} data-mode={b.mode}>
      <p className={s.basisLine}>
        <span className={s.basisLead}>{lead}</span> {day(b.takenAt)}, {time(b.takenAt)}
        {now ? ` · ${age(saved && b.savedFrom ? b.savedFrom : b.takenAt, now)}` : ""}
        {saved && b.savedFrom ? ` · copy from ${dayTime(b.savedFrom)}; no agency was contacted` : ""}
        {" · "}
        <strong>{b.complete ? "Every source answered" : "Incomplete"}</strong>
      </p>
      {b.unchecked.length ? (
        <p className={s.prov} data-testid={tid("briefing-unchecked")}>
          Could not check: {b.unchecked.map((u) => u.label).join(", ")}.
        </p>
      ) : null}
    </div>
  );
}

function Period({ p, print }: { p: BriefPeriod; print?: boolean }) {
  const over = p.windCell.state === "over" || p.seasCell.state === "over";
  const cells = [p.windCell, p.seasCell].filter((c) => c.state === "over" || c.state === "unstated");
  return (
    <li className={s.period} data-over={over || undefined} data-testid={print ? undefined : "briefing-period"}>
      <p className={s.periodName}>{p.name}</p>
      <p className={s.fig}>{p.wind ?? "Wind not stated"}</p>
      <p className={s.fig}>{p.seas ? `Seas ${p.seas}` : "Seas not stated"}</p>
      {/* only what needs the reader's eye: a limit crossed, or a value the forecast does not state */}
      {cells.length ? (
        <ul className={s.cells}>
          {cells.map((c, i) => (
            <li key={i} data-state={c.state}>
              {c.text}
            </li>
          ))}
        </ul>
      ) : null}
    </li>
  );
}

function Economics({ m, print }: { m: Model; print?: boolean }) {
  const e = m.economics;
  if (!e) return <p className={s.slot}>No costs modelled for this trip.</p>;
  return (
    <div className={s.econ} data-testid={print ? undefined : "briefing-economics"}>
      <Micro glyph="hypothetical">Your plan, not a prediction</Micro>
      {e.audience === "commercial" ? (
        <>
          <p className={s.econFig}>
            {e.net ?? "…"} <small>net to boat</small>
          </p>
          <p className={s.prov}>
            {e.gross ? `${e.gross} gross` : "Gross waits for catch and price"}
            {e.breakEven ? ` · break-even ${e.breakEven}` : ""}
          </p>
        </>
      ) : (
        <>
          <p className={s.econFig}>
            {e.tripCost ?? "…"} <small>trip cost</small>
          </p>
          {e.perPerson ? <p className={s.prov}>{e.perPerson} each</p> : null}
        </>
      )}
    </div>
  );
}
