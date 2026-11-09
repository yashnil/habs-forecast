"use client";

import { useEffect, useRef, useState } from "react";
import { Icon } from "@/v4/ui/Icon";
import { OVERALL_WORDS, STATE_WORDS, sortedSources, useReadinessSnapshot, type Overall, type SourceState } from "./store";
import { relTime } from "@/v4/ui/time";

export const STATE_TONE: Record<SourceState, string> = {
  waiting: "text-n-ink-3",
  loading: "text-n-partial",
  ready: "text-n-ink-2",
  partial: "text-n-stale",
  stale: "text-n-stale",
  cached: "text-n-stale",
  unavailable: "text-n-down",
  downloading: "text-n-partial",
  incomplete: "text-n-down",
};
const STATE_ICON: Record<SourceState, string> = {
  waiting: "clock",
  loading: "clock",
  ready: "check",
  partial: "alert",
  stale: "clock",
  cached: "keep",
  unavailable: "alert",
  downloading: "download",
  incomplete: "alert",
};

/** The source list with each one's state, time and actions (status centre panel and the start card). */
export function SourceList({ compact = false, now }: { compact?: boolean; now: Date | null }) {
  const snap = useReadinessSnapshot();
  const list = sortedSources(snap);
  return (
    <ul className="divide-y divide-n-line" data-testid="source-list">
      {list.map((e) => (
        <li key={e.id} className={`flex items-start gap-3 ${compact ? "py-2" : "py-3"}`} data-testid={`source-${e.id}`} data-state={e.state}>
          <span className={`mt-0.5 ${STATE_TONE[e.state]}`}>
            <Icon name={STATE_ICON[e.state]} className="h-[18px] w-[18px]" />
          </span>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-baseline justify-between gap-x-3">
              <span className="text-[15px] font-semibold text-n-ink">{e.label}</span>
              <span className={`text-[13px] font-semibold ${STATE_TONE[e.state]}`}>{STATE_WORDS[e.state]}</span>
            </div>
            {(e.detail || e.asOf) && (
              <p className="mt-0.5 text-[13.5px] leading-snug text-n-ink-2">
                {e.detail}
                {e.asOf && now ? <span className="text-n-ink-3">{e.detail ? " · " : ""}{relTime(e.asOf, now)}</span> : null}
              </p>
            )}
            {!!e.actions?.length && (
              <div className="mt-2 flex flex-wrap gap-2">
                {e.actions.map((a) => (
                  <button key={a.label} type="button" onClick={a.run} className="nx-btn nx-btn-secondary nx-press nx-focus">
                    {a.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        </li>
      ))}
    </ul>
  );
}

/** Announces only changes of the overall state, after it has held for a moment (no chatter). */
export function useCalmAnnouncement(state: Overall, online: boolean) {
  const [said, setSaid] = useState("");
  const first = useRef(true);
  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    const t = setTimeout(() => setSaid(online ? `Neritic: ${OVERALL_WORDS[state]}` : "Neritic is offline. Showing saved data where available."), 900);
    return () => clearTimeout(t);
  }, [state, online]);
  return said;
}
