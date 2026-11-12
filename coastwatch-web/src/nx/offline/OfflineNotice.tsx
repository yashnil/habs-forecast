"use client";

import { usePathname } from "next/navigation";
import { Offline } from "@/nx/ds/States";
import { packWords } from "./pack";
import { useFieldRecord, useOnline, useSavedMark } from "./useOffline";
import s from "./offline.module.css";

const pageOf = (path: string) => (path.startsWith("/notices") ? "notices" : path.startsWith("/today") ? "today" : path.startsWith("/coast") ? "coast" : path.startsWith("/explore") ? "explore" : path.startsWith("/research") ? "research" : "home");

/**
 * The offline notice, at the head of every page's margin (README §9 States: Offline): shown when the browser is
 * offline or the page is a saved copy. It says when the copy was saved, what this device holds for offline use,
 * and what waits for a connection. A connection lost while reading says that what is on screen stays.
 */
export function OfflineNotice() {
  const online = useOnline();
  const mark = useSavedMark();
  const rec = useFieldRecord();
  const pathname = usePathname();
  if (online && !mark) return null;
  const page = pageOf(pathname);
  const w = packWords(rec ?? null, page);
  const savedAt = mark?.at ?? null;
  return (
    <div className={s.notice} data-testid="nx-offline-notice" data-saved={savedAt ? "1" : "0"}>
      {savedAt ? (
        <Offline savedAt={savedAt} heading={online ? "A saved copy" : "Offline"}>
          {w.saved ? `${w.saved} ` : ""}
          {w.notSaved}
        </Offline>
      ) : (
        <Offline savedAt={null}>
          What is on screen stays; nothing updates until you&rsquo;re back online. {w.saved ? `${w.saved} ` : "Menu, Offline saves your port while you have a connection. "}
          {w.notSaved}
        </Offline>
      )}
    </div>
  );
}
