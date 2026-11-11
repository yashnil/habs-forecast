"use client";

import { useEffect, useSyncExternalStore } from "react";
import { readRecord, subscribeRecord, verifyRecord, type FieldRecord } from "./pack";

function subscribeOnline(l: () => void) {
  window.addEventListener("online", l);
  window.addEventListener("offline", l);
  return () => {
    window.removeEventListener("online", l);
    window.removeEventListener("offline", l);
  };
}

/** Whether the browser says it is online; true on the server and during hydration. */
export function useOnline(): boolean {
  return useSyncExternalStore(
    subscribeOnline,
    () => navigator.onLine !== false,
    () => true,
  );
}

type SavedMark = { at: string | null; url: string | null };
const noop = () => () => undefined;
let markMemo: SavedMark | null | undefined;
const readMark = (): SavedMark | null => {
  if (markMemo === undefined) markMemo = (window as unknown as { __NERITIC_SAVED__?: SavedMark }).__NERITIC_SAVED__ ?? null;
  return markMemo;
};

/** The service worker's mark on a page it answered from a saved copy (public/sw.js), or null. */
export function useSavedMark(): SavedMark | null {
  return useSyncExternalStore(noop, readMark, () => null);
}

/** The field pack on this device (checked once against the cache), or null; undefined before hydration. */
export function useFieldRecord(): FieldRecord | null | undefined {
  const r = useSyncExternalStore(subscribeRecord, readRecord, () => undefined);
  useEffect(() => {
    void verifyRecord();
  }, []);
  return r;
}
