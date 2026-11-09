"use client";

import { useEffect, useRef, useState } from "react";
import type { PortFact } from "@/pilot/contracts";
import { diffFacts, snapFacts, type FactChange } from "@/pilot/facts";
import { readSeen, storageWorks, writeSeen } from "./prefs";
import { Section, StatusMark } from "./ui";

const PT = new Intl.DateTimeFormat("en-US", { timeZone: "America/Los_Angeles", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
const WORD: Record<FactChange["change"], string> = { new: "New", changed: "Changed", gone: "No longer listed" };

type State = { phase: "idle" } | { phase: "off" } | { phase: "first" } | { phase: "diff"; since: string; changes: FactChange[] };

/**
 * What changed since the last visit to this port, on this device (keyed by PortFact.id, like
 * lib/changes for My Coast). The comparison runs once per visit, after the sea has loaded (an NWS
 * advisory counts). Then this visit becomes the one the next visit is compared with. The hourly sea,
 * tide and buoy readings are not listed. With storage blocked it says so: it never claims "nothing changed".
 */
export function WhatChanged({ place, facts, ready, now }: { place: string; facts: PortFact[]; ready: boolean; now: string }) {
  const [state, setState] = useState<State>({ phase: "idle" });
  const done = useRef<string | null>(null);
  useEffect(() => {
    if (!ready || done.current === place) return;
    done.current = place;
    if (!storageWorks()) return setState({ phase: "off" });
    const prev = readSeen(place);
    writeSeen(place, { at: now, facts: snapFacts(facts) });
    if (!prev) return setState({ phase: "first" });
    setState({ phase: "diff", since: prev.at, changes: diffFacts(prev.facts, facts) });
  }, [ready, place, facts, now]);
  return (
    <Section id="changed" area="changed" title="What changed since your last visit">
      {state.phase === "idle" && <p className="today-src mt-3">Comparing with your last visit…</p>}
      {state.phase === "off" && <p className="today-say">This browser is not keeping visits (private window or storage blocked), so changes cannot be compared.</p>}
      {state.phase === "first" && (
        <p className="today-say" data-testid="changed-first">
          First visit to this port on this device. Next time, anything new or different in the status, tests, whale rules or bloom outlook is listed here.
        </p>
      )}
      {state.phase === "diff" && (
        <>
          <p className="today-say" data-testid="changed-summary">
            {state.changes.length
              ? `${state.changes.length} change${state.changes.length > 1 ? "s" : ""} in Neritic's records since ${PT.format(new Date(state.since))}.`
              : `No change in Neritic's records since your last visit (${PT.format(new Date(state.since))}). The agencies may still have acted; Neritic checks them once a day.`}
          </p>
          {state.changes.length > 0 && (
            <ul className="mt-2" data-testid="changed-list">
              {state.changes.map((c) => {
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
          )}
        </>
      )}
    </Section>
  );
}
