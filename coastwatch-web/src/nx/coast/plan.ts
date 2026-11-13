"use client";

import { useCallback, useMemo, useSyncExternalStore } from "react";
import { departure, draftTrip, limitValue, plan as planTrip } from "@/nx/trips/lifecycle";
import type { Trip } from "@/nx/trips/types";
import { forecastFromSea } from "@/nx/trips/forecast";
import { takeSnapshot, type Answer } from "@/nx/trips/snapshot";
import type { TripsStore } from "@/nx/trips/store";
import type { CoastGates, CoastSea } from "./brief";
import { audienceOf, type CoastFishery } from "./fisheries";
import { readRecord, subscribeCoast, writeRecordField, type Loadable } from "./useCoast";
import { hasEntry, toDraft, type Workspace } from "./workspace";

/**
 * The planner's own fields (v15): the intended date (and time, if the reader has one), their personal limits and a
 * few words for the area. Kept on this device in My Coast's record (nx-coast-v1, beside the fishery) so moving
 * between the brief and the worksheet, or reloading, keeps them. Blank is "not set", never 0; the limits are the
 * reader's own and never a safe limit.
 */
export type PlanDraft = { date: string; time: string; windKt: string; seasFt: string; area: string };
export const EMPTY_PLAN: PlanDraft = { date: "", time: "", windKt: "", seasFt: "", area: "" };

const str = (v: unknown) => (typeof v === "string" ? v.slice(0, 120) : "");
export function parsePlan(v: unknown): PlanDraft {
  const o = v && typeof v === "object" ? (v as Record<string, unknown>) : {};
  return { date: str(o.date), time: str(o.time), windKt: str(o.windKt), seasFt: str(o.seasFt), area: str(o.area) };
}

let memo: PlanDraft | null = null;
const readRaw = () => JSON.stringify(memo ?? parsePlan(readRecord().plan));

/** The planner's fields and a setter (one or more fields at a time). EMPTY_PLAN before hydration. */
export function usePlan(): [PlanDraft, (patch: Partial<PlanDraft>) => void] {
  const raw = useSyncExternalStore(subscribeCoast, readRaw, () => JSON.stringify(EMPTY_PLAN));
  const plan = useMemo(() => parsePlan(JSON.parse(raw)), [raw]);
  const set = useCallback((patch: Partial<PlanDraft>) => {
    const next = { ...parsePlan(JSON.parse(readRaw())), ...patch };
    memo = next;
    writeRecordField("plan", next);
  }, []);
  return [plan, set];
}

export const planReady = (p: PlanDraft) => /^\d{4}-\d{2}-\d{2}$/.test(p.date);

export type SaveTripInput = {
  plan: PlanDraft;
  placeKey: string;
  portName: string | null;
  fishery: CoastFishery;
  /** the worksheet, when the reader modelled costs (an empty one is "no economics") */
  ws: Workspace | null;
  /** what the page already loaded: the planning snapshot is taken from these (no second fetch) */
  gates: Loadable<CoastGates>;
  sea: Loadable<CoastSea>;
  now: Date;
  store: TripsStore;
};

const FAIL: Record<string, string> = {
  quota: "This device's storage is full: the trip was not saved. Nothing else was changed.",
  readOnly: "Trips here were saved by a newer Neritic: read only on this version.",
  unavailable: "This browser does not let Neritic keep anything: the trip was not saved.",
  invalid: "The trip could not be saved: something in it was not valid.",
};

export type SaveTripOutcome = { ok: true; trip: Trip } | { ok: false; words: string };

/**
 * Save Trip: a new trip from the planner, Planned when it has a date (Draft otherwise), with the planning snapshot of
 * what the page showed at that moment. Saving with no economics is fine: conditions and official information only.
 */
export function saveTripFromPlanner(i: SaveTripInput): SaveTripOutcome {
  if (!i.placeKey) return { ok: false, words: "Choose a port first." };
  const economics = i.ws && hasEntry(i.ws) ? toDraft(i.ws, i.fishery) : null;
  const draft = draftTrip(
    {
      id: i.store.newId(),
      activity: audienceOf(i.fishery) === "recreational" ? "recreational" : "commercial",
      fishery: i.fishery.id,
      placeKey: i.placeKey,
      depart: departure(i.plan.date, i.plan.time),
      area: i.plan.area,
      limits: { windKt: limitValue(i.plan.windKt), seasFt: limitValue(i.plan.seasFt) },
      economics,
      words: { port: i.portName, fishery: i.fishery.label },
    },
    i.now,
  );
  const planned = planTrip(draft, i.now);
  const trip = planned.ok ? planned.trip : draft;
  const answer = <T,>(l: Loadable<T>): Answer<T> => (l.state === "ok" ? l.data : l.state === "error" ? "failed" : null);
  const sea = answer(i.sea);
  const snap = takeSnapshot({
    trip,
    kind: "planning",
    gates: answer(i.gates),
    sea,
    forecast: i.sea.state === "ok" ? forecastFromSea(i.sea.data) : null,
    now: i.now.toISOString(),
    mode: "live",
    savedFrom: null,
  });
  // the trip first, then its planning snapshot attached to it (the store records the id)
  const put = i.store.put(trip);
  if (!put.ok) return { ok: false, words: FAIL[put.reason] ?? FAIL.unavailable };
  const kept = i.store.putSnapshot(snap);
  const withSnap = kept.ok && kept.trip ? kept.trip : put.trip;
  return { ok: true, trip: withSnap };
}
