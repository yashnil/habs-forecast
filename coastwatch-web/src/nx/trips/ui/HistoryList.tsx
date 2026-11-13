"use client";

import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useEffect, useId, useMemo, useRef, useState } from "react";
import { Button, IconButton, Micro, Popover, day, register, registerFrom, registerTo } from "@/nx/ds";
import { tripFishery } from "@/nx/coast/fisheries";
import { money } from "@/nx/coast/workspace";
import { duplicateTrip, groupTrips, ownRecords, renameTrip, searchTrips, MIN_FOR_COMPARISON, type GroupId } from "../history";
import { planVsActual } from "../outcome";
import type { Result, TripsData, TripsStore } from "../store";
import type { Trip } from "../types";
import { setPendingUndo } from "../undo";
import { notifyTrips, tripsStore } from "../useTripsData";
import { IMPORT_WORDS, ImportTrips } from "./ImportTrips";
import { TripsUndo } from "./TripsUndo";
import s from "./economics.module.css";

export type HistoryListProps = {
  /** from useTripsData(); null while the device's storage has not been read */
  data: TripsData | null;
  now: Date;
  portName: (key: string) => string | null;
  /** open a trip's workspace */
  onOpen: (id: string) => void;
  /** defaults to this tab's store; tests and previews pass their own */
  store?: TripsStore;
  /** after any change (defaults to notifying every view of this tab) */
  onChanged?: () => void;
};

/** the one figure a row shows: the plan, the outcome, or none (a cancelled trip's plan is no longer news) */
const FIGURE: Record<GroupId, "planned" | "actual" | null> = { upcoming: "planned", pastDue: "planned", drafts: "planned", completed: "actual", cancelled: null };

function download(name: string, type: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** The row's name as a text field, focused and selected at once: Enter or leaving keeps it, Escape keeps the old name. */
function RenameField({ name, onCommit, onCancel }: { name: string; onCommit: (v: string) => void; onCancel: () => void }) {
  const ref = useRef<HTMLInputElement>(null);
  const [v, setV] = useState(name);
  const done = useRef(false);
  useEffect(() => {
    // after the row menu has closed and handed focus back to its trigger
    const id = window.setTimeout(() => {
      ref.current?.focus();
      ref.current?.select();
    }, 0);
    return () => window.clearTimeout(id);
  }, []);
  const commit = () => {
    if (done.current) return;
    done.current = true;
    if (v.trim() && v.trim() !== name) onCommit(v);
    else onCancel();
  };
  return (
    <input
      ref={ref}
      className={s.hlRename}
      aria-label="Trip name"
      value={v}
      maxLength={90}
      onChange={(e) => setV(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          commit();
        } else if (e.key === "Escape") {
          e.preventDefault();
          done.current = true;
          onCancel();
        }
      }}
      data-testid="history-rename"
    />
  );
}

/**
 * The reader's trips (v15): one search line, then the trips grouped as upcoming, planned with the date passed,
 * drafts, completed and cancelled. Each row is the trip's name, where and when, and one figure (planned net to boat,
 * or the actual for a completed trip); its menu renames (straight into a focused field), duplicates or deletes (with an Undo that
 * outlives the list). Export (JSON or CSV) and import (checked first, then an explicit consent with the real counts)
 * sit at the foot. Comparisons of the reader's own records show
 * only from three completed trips with the figures recorded.
 */
export function HistoryList({ data, now, portName, onOpen, store: given, onChanged }: HistoryListProps) {
  const store = given ?? tripsStore();
  const changed = onChanged ?? notifyTrips;
  const reduce = useReducedMotion();
  const [q, setQ] = useState("");
  const [notice, setNotice] = useState<string | null>(null);
  const [menu, setMenu] = useState<string | null>(null);
  const [renaming, setRenaming] = useState<string | null>(null);
  const searchId = useId();

  const trips = useMemo(() => data?.trips ?? [], [data]);
  const shown = useMemo(() => searchTrips(trips, q, portName), [trips, q, portName]);
  const groups = useMemo(() => groupTrips(shown, now), [shown, now]);
  const own = useMemo(() => ownRecords(trips), [trips]);
  const status = data?.status;
  const t = reduce ? { duration: 0 } : register;

  const fail = (r: Result) => (r.ok ? null : setNotice(IMPORT_WORDS[r.reason] ?? "That did not work."));

  const remove = (trip: Trip) => {
    setMenu(null);
    const r = store.remove(trip.id);
    if (!r.ok) return fail(r);
    setPendingUndo({ name: trip.name, run: r.undo });
    changed();
  };
  const duplicate = (trip: Trip) => {
    setMenu(null);
    const r = duplicateTrip(store, trip.id, now);
    fail(r);
    changed();
  };
  const rename = (trip: Trip, name: string) => {
    setRenaming(null);
    fail(renameTrip(store, trip.id, name));
    changed();
  };

  const figure = (trip: Trip, kind: "planned" | "actual" | null) => {
    if (!kind) return null;
    const pv = planVsActual(trip);
    const f = kind === "actual" && pv.actual.result.value !== null ? pv.actual.result : pv.planned.result;
    if (f.value === null) return null;
    const rec = pv.measure === "tripCost";
    return { v: money(f.value), words: `${kind === "actual" && f === pv.actual.result ? "actual" : "planned"} ${rec ? "trip cost" : "net to boat"}` };
  };

  const when = (trip: Trip) => (trip.state === "completed" && trip.outcome?.departedOn ? day(trip.outcome.departedOn, { ref: now }) : trip.depart ? day(trip.depart.date, { ref: now }) : null);

  if (!data) return <div className={s.hlLoading} aria-busy="true" />;

  return (
    <section className={s.hl} aria-label="Your trips" data-testid="history">
      {status && !status.available ? <p className={s.hlNotice}>This browser is not keeping data for Neritic (a private window, or storage blocked). Trips cannot be saved here.</p> : null}
      {status?.readOnly ? <p className={s.hlNotice}>A newer version of Neritic saved trips on this device. They are shown as they are and nothing is changed here.</p> : null}
      {status?.corrupt.length ? (
        <p className={s.hlNotice} data-testid="history-corrupt">
          {status.corrupt.length === 1 ? "One saved item" : `${status.corrupt.length} saved items`} could not be read. {status.corrupt.length === 1 ? "It is" : "They are"} kept untouched on this device.
        </p>
      ) : null}
      {status?.evicted ? (
        <div className={s.hlNotice} data-testid="history-evicted">
          <p>{status.evicted.all ? `This browser cleared the trips saved on this device (${status.evicted.had} before).` : `${status.evicted.ids.length === 1 ? "One trip" : `${status.evicted.ids.length} trips`} saved on this device ${status.evicted.ids.length === 1 ? "is" : "are"} gone: the browser cleared ${status.evicted.ids.length === 1 ? "it" : "them"}.`} An export you downloaded can bring them back.</p>
          <Button
            variant="quiet"
            onClick={() => {
              store.acknowledgeEviction();
              changed();
            }}
          >
            Understood
          </Button>
        </div>
      ) : null}

      {trips.length > 3 ? (
        <div className={s.hlSearch}>
          <label htmlFor={searchId} className={s.vh}>
            Search your trips by port or fishery
          </label>
          <input id={searchId} type="search" placeholder="Search by port or fishery" value={q} onChange={(e) => setQ(e.target.value)} data-testid="history-search" />
        </div>
      ) : null}

      {groups.length === 0 && q ? <p className={s.empty}>No trip matches “{q}”.</p> : null}

      {groups.map((g) => (
        <div key={g.id} className={s.hlGroup} data-testid={`history-group-${g.id}`}>
          <Micro as="h3" className={s.hlGroupHead}>
            {g.label} <span className={s.hlCount}>{g.trips.length}</span>
          </Micro>
          <ul className={s.hlList}>
            <AnimatePresence initial={false}>
              {g.trips.map((trip) => {
                const f = figure(trip, FIGURE[g.id]);
                const port = portName(trip.departKey ?? trip.placeKey);
                const meta = [port, tripFishery(trip.fishery)?.label ?? trip.fishery, when(trip)].filter(Boolean).join(" · ");
                return (
                  <motion.li key={trip.id} className={s.hlRow} layout={!reduce} initial={registerFrom} animate={registerTo} exit={{ opacity: 0 }} transition={t} data-testid="history-row" data-state={trip.state}>
                    {renaming === trip.id ? (
                      <RenameField name={trip.name} onCommit={(v) => rename(trip, v)} onCancel={() => setRenaming(null)} />
                    ) : (
                      <a
                        className={s.hlMain}
                        href={`?id=${encodeURIComponent(trip.id)}`}
                        onClick={(e) => {
                          if (e.metaKey || e.ctrlKey || e.shiftKey) return;
                          e.preventDefault();
                          onOpen(trip.id);
                        }}
                      >
                        <span className={s.hlName}>{trip.name}</span>
                        <span className={s.hlMeta}>{meta}</span>
                      </a>
                    )}
                    {f ? (
                      <span className={s.hlFig} title={f.words}>
                        <span className={s.vh}>{f.words}: </span>
                        {f.v}
                      </span>
                    ) : (
                      <span className={s.hlFig} />
                    )}
                    <Popover
                      open={menu === trip.id}
                      onOpenChange={(o) => setMenu(o ? trip.id : null)}
                      label={`${trip.name}: rename, duplicate or delete`}
                      align="end"
                      trigger={<IconButton icon="menu" size={44} label={`More for ${trip.name}`} />}
                    >
                      <div className={s.hlMenu}>
                        <Button
                          variant="quiet"
                          onClick={() => {
                            setMenu(null);
                            setRenaming(trip.id);
                          }}
                        >
                          Rename
                        </Button>
                        <Button variant="quiet" onClick={() => duplicate(trip)}>
                          Duplicate as a draft
                        </Button>
                        <Button variant="quiet" onClick={() => remove(trip)} data-testid="history-delete">
                          Delete
                        </Button>
                      </div>
                    </Popover>
                  </motion.li>
                );
              })}
            </AnimatePresence>
          </ul>
        </div>
      ))}

      {own.enough ? (
        <div className={s.own} data-testid="history-own">
          <Micro>Your own records</Micro>
          {own.net ? <p>{own.net.words}</p> : null}
          {own.catch ? <p>{own.catch.words}</p> : null}
          <p className={s.note}>Your past trips only, as you recorded them. Not a forecast.</p>
        </div>
      ) : own.completed > 0 ? (
        <p className={s.note} data-testid="history-own-wait">
          Comparisons of your plans with what happened appear once {MIN_FOR_COMPARISON} completed trips have their figures recorded.
        </p>
      ) : null}

      <TripsUndo onChanged={changed} />
      {notice ? (
        <p className={s.hlNotice} role="alert">
          {notice}
        </p>
      ) : null}

      <div className={s.hlFoot}>
        <span className={s.hlFootWords}>Kept on this device only.</span>
        {trips.length ? (
          <>
            <Button variant="quiet" onClick={() => download(`neritic-trips-${day(now, { year: true }).replace(/[ ,]+/g, "-")}.json`, "application/json", store.exportAll())} data-testid="history-export">
              Export
            </Button>
            <Button variant="quiet" onClick={() => download(`neritic-trips-${day(now, { year: true }).replace(/[ ,]+/g, "-")}.csv`, "text/csv", store.exportCsv(portName))} data-testid="history-export-csv">
              CSV
            </Button>
          </>
        ) : null}
        <ImportTrips store={store} onDone={() => changed()} />
      </div>

    </section>
  );
}
