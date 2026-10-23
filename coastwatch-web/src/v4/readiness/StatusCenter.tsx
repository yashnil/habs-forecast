"use client";

import Link from "next/link";
import { useEffect, useId, useRef, useState } from "react";
import { Icon } from "@/v4/ui/Icon";
import { exitProps, usePresence } from "@/v4/motion";
import { OVERALL_WORDS, STATE_WORDS, overall, sortedSources, useReadinessSnapshot, type Overall, type SourceState } from "./store";
import { relTime } from "@/v4/ui/time";

const OVERALL_ICON: Record<Overall, string> = {
  initializing: "clock",
  partial: "clock",
  ready: "check",
  stale: "clock",
  offline: "offline",
  unavailable: "alert",
  downloading: "download",
  incomplete: "alert",
  outdated: "clock",
};
const OVERALL_TONE: Record<Overall, string> = {
  initializing: "text-n-ink-2",
  partial: "text-n-partial",
  ready: "text-n-ok",
  stale: "text-n-stale",
  offline: "text-n-offline",
  unavailable: "text-n-down",
  downloading: "text-n-partial",
  incomplete: "text-n-down",
  outdated: "text-n-stale",
};
export const STATE_TONE: Record<SourceState, string> = {
  waiting: "text-n-ink-3",
  loading: "text-n-partial",
  ready: "text-n-ok",
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
                  <button key={a.label} type="button" onClick={a.run} className="nx-press nx-focus min-h-11 rounded-lg border border-n-line-strong px-3 text-[14px] font-semibold text-n-ink hover:bg-n-surface-2">
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
function useCalmAnnouncement(state: Overall, online: boolean) {
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

export function StatusCenter({ now, align = "right", compact = false }: { now: Date | null; align?: "left" | "right"; compact?: boolean }) {
  const snap = useReadinessSnapshot();
  const state = overall(snap);
  const [open, setOpen] = useState(false);
  const pop = usePresence(open);
  const id = useId();
  const box = useRef<HTMLDivElement | null>(null);
  const said = useCalmAnnouncement(state, snap.online);
  useEffect(() => {
    if (!open) return;
    const key = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      setOpen(false);
    };
    const click = (e: MouseEvent) => box.current && !box.current.contains(e.target as Node) && setOpen(false);
    window.addEventListener("keydown", key);
    window.addEventListener("mousedown", click);
    return () => {
      window.removeEventListener("keydown", key);
      window.removeEventListener("mousedown", click);
    };
  }, [open]);
  const busy = state === "initializing" || state === "partial" || state === "downloading";
  return (
    <div ref={box} className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-controls={id}
        data-testid={compact ? "status-center-compact" : "status-center"}
        data-state={state}
        aria-label={compact ? `Status: ${OVERALL_WORDS[state]}` : undefined}
        className={`nx-press nx-focus flex h-11 items-center gap-2 rounded-full border border-n-line-strong bg-n-surface text-[14px] font-semibold text-n-ink ${compact ? "w-11 justify-center" : "px-3.5"}`}
      >
        <span className={`relative grid place-items-center ${OVERALL_TONE[state]}`}>
          <Icon name={OVERALL_ICON[state]} className="h-[18px] w-[18px]" />
        </span>
        {!compact && <span className="whitespace-nowrap">{OVERALL_WORDS[state]}</span>}
        {busy && <span className="sr-only">, details available</span>}
      </button>
      {/* polite and only on settled changes of the overall state */}
      <span className="sr-only" role="status" aria-live="polite">
        {said}
      </span>
      {pop.mounted && (
        <div
          id={id}
          role="dialog"
          aria-label="Data and connection status"
          // on a phone the panel spans the screen under the top bar (as the notices panel does): hung from
          // the button it ran off the left edge; elsewhere it hangs from the button
          className={`nx-pop fixed inset-x-3 top-[64px] z-50 max-h-[calc(100dvh-80px)] overflow-y-auto overscroll-contain rounded-[18px] border border-n-line bg-n-surface p-4 text-n-ink shadow-[var(--nx-shadow)] md:absolute md:inset-x-auto md:top-[52px] md:max-h-[min(72vh,680px)] md:w-[380px] ${align === "right" ? "md:right-0" : "md:left-0 nx-origin-left"}`}
          {...exitProps(pop.state, "status-panel")}
        >
          <div className="flex items-center justify-between">
            <p className="text-[16px] font-semibold">{OVERALL_WORDS[state]}</p>
            <span className={`flex items-center gap-1.5 text-[13px] font-semibold ${snap.online ? "text-n-ok" : "text-n-offline"}`}>
              <Icon name={snap.online ? "online" : "offline"} className="h-4 w-4" />
              {snap.online ? "Online" : "Offline"}
            </span>
          </div>
          <p className="mt-1 text-[13.5px] text-n-ink-2">Each source keeps its own time. Nothing older is shown as current.</p>
          <div className="mt-2">
            <SourceList now={now} compact />
          </div>
          <Link href="/sources" className="nx-focus mt-3 flex min-h-11 items-center gap-2 text-[14px] font-semibold text-n-accent">
            Full data status <Icon name="chevronRight" className="h-4 w-4" />
          </Link>
        </div>
      )}
    </div>
  );
}
