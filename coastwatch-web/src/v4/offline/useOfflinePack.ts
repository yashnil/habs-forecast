"use client";

import { useCallback, useEffect, useState } from "react";
import { packState, packWasSaved, type PackProblem, type PackRecord } from "./packs";
import { useRun } from "./runStore";

export const SW_URL = `/sw.js${process.env.NEXT_PUBLIC_NERITIC_BUILD ? `?b=${process.env.NEXT_PUBLIC_NERITIC_BUILD}` : ""}`;

/** Registers the service worker (once) and reads a pack's record; re-reads it on demand. */
export function useOfflinePack(id: string | null) {
  const [record, setRecord] = useState<PackRecord | null | undefined>(undefined);
  // what is wrong with the saved copy, if anything: an older format, files gone, a damaged record, unfinished
  const [problem, setProblem] = useState<PackProblem | null>(null);
  const [online, setOnline] = useState(true);
  const [sw, setSw] = useState<"unsupported" | "registering" | "ready" | "failed">("registering");
  // saved on this device once, but the copy is gone: the browser evicted it (low storage, unused site)
  const [evicted, setEvicted] = useState<string | null>(null);
  const refresh = useCallback(async () => {
    if (!id) return setRecord(null);
    const s = await packState(id);
    setProblem(s.problem);
    setRecord(s.record);
    setEvicted(s.record || s.problem ? null : packWasSaved(id));
  }, [id]);
  useEffect(() => {
    const on = () => setOnline(navigator.onLine);
    on();
    window.addEventListener("online", on);
    window.addEventListener("offline", on);
    if (!("serviceWorker" in navigator)) setSw("unsupported");
    else
      navigator.serviceWorker
        // one worker URL per build: a deploy installs a worker that drops the old build's code cache
        .register(SW_URL, { scope: "/" })
        .then(() => navigator.serviceWorker.ready)
        .then(() => setSw("ready"))
        .catch(() => setSw("failed"));
    return () => {
      window.removeEventListener("online", on);
      window.removeEventListener("offline", on);
    };
  }, []);
  // a download finished or the pack was removed (from any view): read the record again
  const version = useRun(id).version;
  useEffect(() => {
    void refresh();
  }, [refresh, version]);
  return { record, problem, online, sw, refresh, evicted };
}

/** How old the saved data is, in the words the banner and the status centre use. */
export function packAge(r: PackRecord, now: Date): { hours: number; outdated: boolean } {
  const h = (now.getTime() - Date.parse(r.savedAt)) / 3_600_000;
  // C-HARM runs daily and observations change within hours: after a day the copy is outdated
  return { hours: h, outdated: h > 24 };
}
