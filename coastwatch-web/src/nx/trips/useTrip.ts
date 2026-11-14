"use client";

import { startTransition, useCallback, useMemo, useState } from "react";
import { usePorts } from "@/nx/shell/port";
import { tripFishery } from "@/nx/coast/fisheries";
import { diffSnapshots } from "./diff";
import { forecastFromSea } from "./forecast";
import { cancel, complete, edit, nextAction, plan, recordOutcome, reopen, type NameWords, type Result, type TripPatch } from "./lifecycle";
import { fieldPackCopy, recheck as runRecheck, refreshPlanning as runRefresh, summarize, type RecheckDeps } from "./recheck";
import { knownView, type KnownView } from "./known";
// recheck.ts reads the field pack's cache names with a dynamic import; importing the module here keeps it in this page's
// bundle, so an offline recheck never has to fetch a chunk (a failed chunk load makes Next reload the document)
import { FIELD_CACHE } from "@/nx/offline/pack";
import type { Outcome, Recheck, Snapshot, Trip } from "./types";
import { notifyTrips, tripsStore, useTripsData } from "./useTripsData";

/**
 * One trip, headless (v15): the trip as stored, its planning snapshot and newest recheck, what changed since planning,
 * the one next action, and every transition (each one written to the store at once). `ready` is false until the
 * device's storage has been read (never during server render).
 */

/** A stored recheck snapshot read back as the comparison it was (so a recheck survives a reload). */
/** A stored recheck read back against the planning snapshot: the same summary recheck() gave when it ran (one rule). */
export function recheckOf(planning: Snapshot, next: Snapshot, view?: KnownView): Recheck {
  return summarize(planning, next, diffSnapshots(planning, next, next.serverNow ?? next.takenAt), next.takenAt, view);
}

/** the field pack's cache, named so the import above is kept */
export const PACK_CACHE = FIELD_CACHE;

export function useTrip(id: string | null) {
  const data = useTripsData();
  const ports = usePorts();
  const ready = data !== null;
  const trip = useMemo(() => (data && id ? (data.trips.find((t) => t.id === id) ?? null) : null), [data, id]);
  const planning = useMemo(() => (trip?.snapshots.planning ? tripsStore().getSnapshot(trip.snapshots.planning) : null), [trip]);
  const latest = useMemo(() => {
    const last = trip?.snapshots.rechecks.at(-1) ?? null;
    return last ? tripsStore().getSnapshot(last) : null;
  }, [trip]);
  // what the device knows per source, across the planning snapshot and every kept recheck: a failed read never blanks it
  const known = useMemo(() => {
    const ids = trip ? [trip.snapshots.planning, ...trip.snapshots.rechecks].filter((x): x is string => !!x) : [];
    return knownView(ids.map((i) => tripsStore().getSnapshot(i)).filter((x): x is Snapshot => !!x));
  }, [trip]);
  // a recheck run in this view (an offline comparison is not stored as a snapshot), else the newest stored one
  const [session, setSession] = useState<{ tripId: string; r: Recheck; savedFrom: string | null } | null>(null);
  const recheck = useMemo<Recheck | null>(() => {
    if (session && session.tripId === trip?.id) return session.r;
    return planning && latest ? recheckOf(planning, latest, known) : null;
  }, [session, trip, planning, latest, known]);
  const [busy, setBusy] = useState(false);
  const [reviewed, setReviewed] = useState(false);
  const [said, setSaid] = useState<string | null>(null);

  const words: NameWords = useMemo(() => ({ port: ports.find((p) => p.key === trip?.placeKey)?.name ?? null, fishery: trip ? (tripFishery(trip.fishery)?.label ?? null) : null }), [ports, trip]);

  const commit = useCallback((r: Result): boolean => {
    if (!r.ok) {
      setSaid(r.words);
      return false;
    }
    const put = tripsStore().put(r.trip);
    if (!put.ok) {
      setSaid(put.reason === "quota" ? "This device's storage is full: the change was not saved." : put.reason === "readOnly" ? "These trips were saved by a newer Neritic: read only here." : "This browser does not let Neritic keep anything.");
      return false;
    }
    setSaid(null);
    notifyTrips();
    return true;
  }, []);

  const now = () => new Date();
  // the clock the evidence was read on: the server's, as its newest answer carried it (a fixture pins it), kept as an
  // offset from the device's so time still moves; the device's own when no answer carried one. In production they agree.
  const skew = useMemo(() => {
    const s = (latest?.mode === "live" && latest.serverNow ? latest : null) ?? (planning?.serverNow ? planning : null);
    return s?.serverNow ? Date.parse(s.serverNow) - Date.parse(s.takenAt) : 0;
  }, [latest, planning]);
  const clock = useCallback(() => new Date(Date.now() + skew), [skew]);
  const deps = useCallback((): RecheckDeps => {
    const st = tripsStore();
    return { store: { getSnapshot: (i) => st.getSnapshot(i), putSnapshot: (x) => st.putSnapshot(x), savedCopy: (t) => fieldPackCopy(t) }, forecastFromSea: (x) => forecastFromSea(x), now: () => new Date() };
  }, []);
  const doRecheck = useCallback(async () => {
    if (!trip) return;
    setBusy(true);
    try {
      const r = await runRecheck(trip, deps());
      // a transition: the workspace stays on screen (with the feed's checking state) while the new answer renders
      startTransition(() => {
        setSession({ tripId: trip.id, r, savedFrom: r.snapshot.savedFrom });
        setReviewed(false);
        notifyTrips();
      });
    } catch (e) {
      setSaid((e as Error).message);
    } finally {
      setBusy(false);
    }
  }, [trip, deps]);
  const doRefresh = useCallback(async () => {
    if (!trip) return;
    setBusy(true);
    try {
      const r = await runRefresh(trip, deps());
      if (!r.ok) setSaid(r.reason);
      else
        startTransition(() => {
          setSession(null);
          notifyTrips();
        });
    } finally {
      setBusy(false);
    }
  }, [trip, deps]);

  const next = trip ? nextAction(trip, { changes: recheck?.changed ?? null, recheck, now: clock(), reviewed, forecast: known.forecast ? (known.forecast.latest ? "latest" : "older") : "none" }) : null;

  return {
    ready,
    trip,
    planning,
    latest,
    /** per source, the newest value this device holds (D's knownView) */
    known,
    recheck,
    next,
    busy,
    said,
    words,
    /** the evidence clock (see skew) */
    clock,
    markReviewed: () => setReviewed(true),
    recheckNow: doRecheck,
    refreshPlanning: doRefresh,
    savedFrom: session && session.tripId === trip?.id ? session.savedFrom : null,
    plan: () => trip && commit(plan(trip, now())),
    complete: () => trip && commit(complete(trip, now())),
    cancel: () => trip && commit(cancel(trip, now())),
    reopen: () => trip && commit(reopen(trip, now())),
    edit: (patch: TripPatch, o: { plan?: boolean } = {}) => {
      if (!trip) return false;
      const r = edit(trip, patch, now(), words);
      if (r.ok && o.plan && r.trip.state === "draft") {
        const p = plan(r.trip, now());
        if (p.ok) return commit(p);
      }
      return commit(r);
    },
    rename: (name: string) => trip && commit(edit(trip, { name }, now(), words)),
    recordOutcome: (o: Omit<Outcome, "recordedAt" | "updatedAt">) => trip && commit(recordOutcome(trip, o, now())),
  };
}

export type TripModel = ReturnType<typeof useTrip>;

/** Every trip on this device, newest change first; null before hydration. */
export function useTrips(): Trip[] | null {
  const data = useTripsData();
  return useMemo(() => (data ? [...data.trips].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt)) : null), [data]);
}
