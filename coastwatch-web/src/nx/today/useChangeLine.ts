"use client";

import { useEffect, useRef, useState } from "react";
import type { PortFact } from "@/pilot/contracts";
import { snapFacts } from "@/pilot/facts";
import { readSeen, storageWorks, writeSeen, type Seen } from "@/pilot/today/prefs";
import { changeLine, type ChangeLine } from "./change";

/**
 * Which earlier visit Today compares with (journeys.md Journey 2), decided once per visit.
 *
 * Two records per port, both on this device:
 *   nx-today-seen-v1:<port>   (localStorage)   what Today showed the last time it was opened at the port;
 *   nx-today-visit-v1:<port>  (sessionStorage) the earlier visit THIS visit is comparing with, and when
 *                                              the page was last opened in this tab.
 *
 * Opening Today at a port: if this tab already has a visit under way for it (opened less than 30 minutes
 * ago), the same earlier visit is used again, so a page that mounts twice, a reload or a trip to another tab
 * and back all show one line and never compare the page with itself. Otherwise the stored `seen` record
 * becomes this visit's earlier visit. Either way what the page shows now is written as `seen` for next time.
 *
 * No earlier visit: no line. A port opened for the first time never says "since".
 */
export const VISIT_PREFIX = "nx-today-visit-v1:";
export const VISIT_IDLE_MS = 30 * 60_000;

type Visit = { prev: Seen | null; touched: number };

/** where a visit is kept; injectable so the rule is tested without a browser */
export type VisitStore = {
  readVisit: (port: string) => Visit | null;
  writeVisit: (port: string, v: Visit) => void;
  readSeen: (port: string) => Seen | null;
  writeSeen: (port: string, s: Seen) => void;
};

// a tab whose sessionStorage is blocked still compares once per page load
const memory = new Map<string, Visit>();

const isSeen = (v: unknown): v is Seen => !!v && typeof v === "object" && typeof (v as Seen).at === "string" && Array.isArray((v as Seen).facts);

export const browserStore: VisitStore = {
  readVisit(port) {
    try {
      const raw = window.sessionStorage.getItem(VISIT_PREFIX + port);
      if (raw) {
        const v = JSON.parse(raw) as Visit;
        if (v && typeof v.touched === "number" && (v.prev === null || isSeen(v.prev))) return v;
      }
    } catch {
      // blocked or unreadable: this page load's memory
    }
    return memory.get(port) ?? null;
  },
  writeVisit(port, v) {
    memory.set(port, v);
    try {
      window.sessionStorage.setItem(VISIT_PREFIX + port, JSON.stringify(v));
    } catch {
      // kept in memory for this page load
    }
  },
  readSeen,
  writeSeen: (port, s) => void writeSeen(port, s),
};

/**
 * Open Today at `port` showing `facts`: the line for this visit, or null when there was no earlier visit.
 * `pageNow` is the page's clock (the server's, which the dates on the page use); `deviceNow` only measures
 * how long ago this tab last opened the port.
 */
export function openVisit(port: string, facts: PortFact[], pageNow: string, deviceNow: number, store: VisitStore = browserStore): ChangeLine | null {
  const visit = store.readVisit(port);
  const idle = visit ? deviceNow - visit.touched : Infinity;
  const under = visit !== null && idle >= 0 && idle < VISIT_IDLE_MS;
  const prev = under ? visit.prev : store.readSeen(port);
  store.writeVisit(port, { prev, touched: deviceNow });
  store.writeSeen(port, { at: pageNow, facts: snapFacts(facts) });
  return changeLine(prev, facts, pageNow);
}

/**
 * The change line for Today at `key`: undefined until the page's facts are ready, null when there is nothing
 * to say (no earlier visit, or this browser keeps nothing), else the line. Computed once per port per mount.
 */
export function useChangeLine(key: string, ready: boolean, facts: PortFact[], now: string): ChangeLine | null | undefined {
  const [line, setLine] = useState<{ key: string; line: ChangeLine | null } | undefined>(undefined);
  const done = useRef<string | null>(null);
  useEffect(() => {
    if (!ready || done.current === key) return;
    done.current = key;
    setLine({ key, line: storageWorks() ? openVisit(key, facts, now, Date.now()) : null });
  }, [ready, key, facts, now]);
  // a line computed for another port is never shown at this one
  return line && line.key === key ? line.line : undefined;
}
