"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ScenarioRecord } from "@/lib/tripEconomicsStore";
import { tripFishery, type CoastFishery } from "./fisheries";
import { capacity, deleteTrip, duplicateTrip, loadWorking, renameTrip, saveNew, saveWorking, tripState, updateTrip, type DeleteOutcome, type SaveOutcome } from "./scenarios";
import { notifyTrip, tripStore, useTripData } from "./useTripStore";
import { emptyWorkspace, evaluateWorkspace, fromDraft, switchFishery, type Scale, type Workspace } from "./workspace";

type Held = { key: string; fishery: CoastFishery | null; ws: Workspace; openId: string | null };

/**
 * The trip workspace, headless (v13): the reader's inputs for this port and fishery, their evaluation, and
 * what Save, Update, Duplicate, Rename, Delete and Undo do. No layout. Every edit is autosaved on this device
 * with the id of the trip it was opened from, so leaving the page, reloading or restarting the browser brings
 * the same worksheet back, and Update still means update.
 *
 * `ready` is false until the device's storage has been read (never during server render): the page draws the
 * blank worksheet's shape until then, not somebody's numbers.
 */
export function useTripWorkspace(o: { placeKey: string; fishery: CoastFishery | null; onFishery?: (id: string) => void }) {
  const { placeKey, fishery, onFishery } = o;
  const data = useTripData();
  const key = `${placeKey}|${fishery?.id ?? ""}`;
  const [held, setHeld] = useState<Held | null>(null);
  /** a saved trip being opened brings its own fishery: its numbers are not replaced when the fishery follows */
  const opening = useRef<Held | null>(null);
  const [cleared, setCleared] = useState<string[]>([]);
  const [undo, setUndo] = useState<{ record: ScenarioRecord; run: () => boolean } | null>(null);
  const ready = data !== null;

  /** the held worksheet as last set, read where an effect must not depend on it */
  const last = useRef<Held | null>(null);
  const hold = useCallback((next: Held) => {
    last.current = next;
    setHeld(next);
  }, []);

  useEffect(() => {
    if (!ready) return;
    const prev = last.current;
    if (prev?.key === key) return;
    if (opening.current?.key === key) {
      const next = opening.current;
      opening.current = null;
      hold(next);
      return;
    }
    const w = loadWorking(tripStore().load(), placeKey, fishery);
    if (w.from === "working" || !prev || prev.key.split("|")[0] !== placeKey) {
      setCleared([]);
      hold({ key, fishery, ws: w.ws, openId: w.openId });
      return;
    }
    // the same port, another fishery, nothing kept for it: the reader's numbers come along, less what no longer means the same
    const moved = switchFishery(prev.ws, prev.fishery, fishery);
    setCleared(moved.cleared);
    saveWorking(tripStore(), placeKey, fishery, moved.workspace, null);
    hold({ key, fishery, ws: moved.workspace, openId: null });
  }, [ready, key, placeKey, fishery, hold]);

  const current = held?.key === key ? held : null;
  const ws = useMemo(() => current?.ws ?? emptyWorkspace(fishery), [current, fishery]);
  const openId = current?.openId ?? null;

  const put = useCallback(
    (next: Workspace, id: string | null) => {
      hold({ key, fishery, ws: next, openId: id });
      saveWorking(tripStore(), placeKey, fishery, next, id);
      notifyTrip();
    },
    [key, fishery, placeKey, hold],
  );

  const edit = useCallback(
    (change: (ws: Workspace) => Workspace) => {
      setCleared([]);
      put(change(ws), openId);
    },
    [put, ws, openId],
  );

  const result = useMemo(() => evaluateWorkspace(ws, fishery), [ws, fishery]);
  const preview = useCallback((scale: Scale) => evaluateWorkspace(ws, fishery, scale), [ws, fishery]);
  const state = useMemo(() => tripState(data, ws, fishery, openId), [data, ws, fishery, openId]);

  const after = (r: SaveOutcome): SaveOutcome => {
    if (r.ok) put(ws, r.id);
    return r;
  };

  return {
    ready,
    ws,
    result,
    preview,
    edit,
    /** what a change of fishery dropped ("price", "catch"), for the page to say once */
    cleared,
    state,
    openId,
    capacity: capacity(data),
    data,
    saveAsNew: (opts: { name?: string; allowIdentical?: boolean } = {}) => after(saveNew(tripStore(), { ws, fishery, placeKey }, { now: new Date(), ...opts })),
    /** only when a saved trip is open */
    update: (): SaveOutcome => (openId ? after(updateTrip(tripStore(), openId, { ws, fishery, placeKey })) : { ok: false, reason: "gone" }),
    duplicate: (id: string): SaveOutcome => {
      const r = duplicateTrip(tripStore(), id);
      notifyTrip();
      return r;
    },
    rename: (id: string, name: string): SaveOutcome => {
      const r = renameTrip(tripStore(), id, name);
      notifyTrip();
      return r;
    },
    remove: (id: string): DeleteOutcome => {
      const r = deleteTrip(tripStore(), id);
      if (r.ok) {
        setUndo({ record: r.record, run: r.undo });
        // the open trip was deleted: what is on the worksheet is unsaved work again
        if (openId === id) put(ws, null);
        else notifyTrip();
      }
      return r;
    },
    /** the trip last deleted in this view, until it is put back or another is deleted */
    undo: undo
      ? {
          name: undo.record.name,
          run: () => {
            const ok = undo.run();
            if (ok) setUndo(null);
            notifyTrip();
            return ok;
          },
        }
      : null,
    /** Open a saved trip here. A trip from another fishery switches the fishery; one from another port stays a copy to save here. */
    open: (id: string): boolean => {
      const rec = data?.scenarios.find((s) => s.id === id);
      if (!rec) return false;
      const f = tripFishery(rec.species) ?? fishery;
      const next: Held = { key: `${placeKey}|${f?.id ?? ""}`, fishery: f, ws: fromDraft(rec.draft, rec.species || null), openId: rec.placeKey === placeKey ? rec.id : null };
      saveWorking(tripStore(), placeKey, f, next.ws, next.openId);
      setCleared([]);
      if (f && f.id !== fishery?.id && onFishery) {
        opening.current = next;
        onFishery(f.id);
      } else hold(next);
      notifyTrip();
      return true;
    },
    clear: () => {
      setCleared([]);
      put(emptyWorkspace(fishery), null);
    },
  };
}

export type TripWorkspace = ReturnType<typeof useTripWorkspace>;
