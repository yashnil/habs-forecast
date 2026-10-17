"use client";

import { useCallback, useEffect, useState } from "react";
import { packWasSaved, readRecord, type PackRecord } from "./packs";

/** Registers the service worker (once) and reads a pack's record; re-reads it on demand. */
export function useOfflinePack(id: string | null) {
  const [record, setRecord] = useState<PackRecord | null | undefined>(undefined);
  const [online, setOnline] = useState(true);
  const [sw, setSw] = useState<"unsupported" | "registering" | "ready" | "failed">("registering");
  // saved on this device once, but the copy is gone: the browser evicted it (low storage, unused site)
  const [evicted, setEvicted] = useState<string | null>(null);
  const refresh = useCallback(async () => {
    if (!id) return setRecord(null);
    const r = await readRecord(id);
    setRecord(r);
    setEvicted(r ? null : packWasSaved(id));
  }, [id]);
  useEffect(() => {
    const on = () => setOnline(navigator.onLine);
    on();
    window.addEventListener("online", on);
    window.addEventListener("offline", on);
    if (!("serviceWorker" in navigator)) setSw("unsupported");
    else
      navigator.serviceWorker
        .register("/sw.js", { scope: "/" })
        .then(() => navigator.serviceWorker.ready)
        .then(() => setSw("ready"))
        .catch(() => setSw("failed"));
    return () => {
      window.removeEventListener("online", on);
      window.removeEventListener("offline", on);
    };
  }, []);
  useEffect(() => {
    void refresh();
  }, [refresh]);
  return { record, online, sw, refresh, evicted };
}

/** How old the saved data is, in the words the banner and the status centre use. */
export function packAge(r: PackRecord, now: Date): { hours: number; outdated: boolean } {
  const h = (now.getTime() - Date.parse(r.savedAt)) / 3_600_000;
  // C-HARM runs daily and observations change within hours: after a day the copy is outdated
  return { hours: h, outdated: h > 24 };
}
