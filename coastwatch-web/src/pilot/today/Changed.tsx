"use client";

import { stableFormat } from "@/lib/time";
import { useEffect, useRef, useState } from "react";
import type { PortFact } from "@/pilot/contracts";
import { diffFacts, snapFacts, type FactChange } from "@/pilot/facts";
import { Fold } from "@/v4/motion";
import { readSeen, storageWorks, writeSeen } from "./prefs";
import { StatusMark } from "./ui";

const PT = new Intl.DateTimeFormat("en-US", { timeZone: "America/Los_Angeles", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
const WORD: Record<FactChange["change"], string> = { new: "New", changed: "Changed", gone: "No longer listed" };

/**
 * "Since your last visit": one line, only when something changed (keyed by PortFact.id, on this device).
 * The comparison runs once per visit, after the sea has loaded, then this visit becomes the one the next
 * compares with. The hourly sea, tide and buoy readings are not listed. A first visit, or a browser that
 * keeps no visits, shows nothing: the line never claims "nothing changed".
 */
export function WhatChanged({ place, facts, ready, now }: { place: string; facts: PortFact[]; ready: boolean; now: string }) {
  const [diff, setDiff] = useState<{ since: string; changes: FactChange[] } | null>(null);
  const done = useRef<string | null>(null);
  useEffect(() => {
    if (!ready || done.current === place) return;
    done.current = place;
    if (!storageWorks()) return;
    const prev = readSeen(place);
    writeSeen(place, { at: now, facts: snapFacts(facts) });
    if (prev) setDiff({ since: prev.at, changes: diffFacts(prev.facts, facts) });
  }, [ready, place, facts, now]);
  if (!diff || !diff.changes.length) return null;
  return (
    <div className="today-changed" data-testid="today-changed">
      <Fold
        testid="changed-fold"
        title={
          <span data-testid="changed-summary">
            {diff.changes.length} change{diff.changes.length > 1 ? "s" : ""} in Neritic&rsquo;s records since your last visit, {stableFormat(PT, new Date(diff.since))}
          </span>
        }
      >
        <ul data-testid="changed-list">
          {diff.changes.map((c) => {
            const f = c.after ?? c.before!;
            return (
              <li key={c.id} className="flex gap-3 border-t border-n-line py-3 first:border-t-0">
                <StatusMark severity={f.severity} className="mt-0.5" />
                <div className="min-w-0">
                  <p className="nx-caps">{WORD[c.change]}</p>
                  <p className="mt-0.5 text-[15px] leading-snug text-n-ink">{f.headline}</p>
                  {c.change === "changed" && c.before && c.before.headline !== f.headline && <p className="mt-1 text-[13.5px] leading-snug text-n-ink-3">Before: {c.before.headline}</p>}
                </div>
              </li>
            );
          })}
        </ul>
      </Fold>
    </div>
  );
}
