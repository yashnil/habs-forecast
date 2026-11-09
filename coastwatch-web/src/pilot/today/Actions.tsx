"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import type { Readiness } from "@/pilot/contracts";
import { READINESS_WORDS } from "@/pilot/readiness";
import { Icon } from "@/v4/ui/Icon";
import { Feedback } from "@/pilot/account/Feedback";
import { readSaved } from "./prefs";
import { saveTodayOffline } from "./offline";
import { Chapter } from "./ui";

const PT = new Intl.DateTimeFormat("en-US", { timeZone: "America/Los_Angeles", month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short" });

type SaveState = { phase: "idle" } | { phase: "saving" } | { phase: "saved"; at: string } | { phase: "failed"; why: string };

/** What an alert can reach today, said plainly from readiness (never claimed). */
export function alertWords(r: { alerts: Readiness; sms: Readiness }): string {
  const sms = r.sms === "live" ? "" : " SMS in carrier review — preview only.";
  if (r.alerts === "not_configured") return `Not yet connected: accounts, push and text. Alerts work in the app on this device.`;
  return `${READINESS_WORDS[r.alerts]}: in the app, and by push where it is turned on.${sms}`;
}

/**
 * Keep, alert, ask: the last chapter. Three actions, each with one honest line about what it reaches;
 * the one "was this useful?" on the page closes it.
 */
export function Actions({
  place,
  label,
  conditionsUrl,
  readiness,
  pilot,
  onAsk,
  feedbackFact,
}: {
  place: string;
  label: string;
  conditionsUrl: string;
  readiness: { alerts: Readiness; ask: Readiness; sms: Readiness };
  /** one of the pilot ports: alerts and Ask Neritic cover those only */
  pilot: boolean;
  onAsk: () => void;
  /** the fact "Was this useful?" is about (the briefing's lead) */
  feedbackFact: string | null;
}) {
  const [save, setSave] = useState<SaveState>({ phase: "idle" });
  useEffect(() => {
    const s = readSaved().find((x) => x.place === place);
    if (s) setSave({ phase: "saved", at: s.at });
  }, [place]);
  const q = `place=${encodeURIComponent(place)}`;
  return (
    <Chapter id="actions" area="actions" title="Keep, alert, ask" testid="today-actions">
      <div className="today-actions">
        <div>
          <button
            type="button"
            className="today-act nx-focus nx-press"
            disabled={save.phase === "saving"}
            onClick={async () => {
              setSave({ phase: "saving" });
              const r = await saveTodayOffline(place, label, conditionsUrl);
              setSave(r.ok ? { phase: "saved", at: r.at } : { phase: "failed", why: r.why });
            }}
            data-testid="save-offline"
          >
            <Icon name={save.phase === "saved" ? "check" : "download"} className="h-5 w-5" />
            {save.phase === "saving" ? "Saving…" : save.phase === "saved" ? "Saved · update" : "Keep it offline"}
          </button>
          <p className="today-note mt-1.5" role="status" aria-live="polite" data-testid="save-status">
            {save.phase === "saved" && `Saved ${PT.format(new Date(save.at))}. This page opens without a connection, as it was then.`}
            {save.phase === "failed" && `Could not save for offline: ${save.why}`}
            {save.phase === "idle" && "Keeps this page and its sea on this device, for use without signal."}
          </p>
        </div>
        <div>
          <Link href={`/account?${q}`} className="today-act nx-focus nx-press" data-testid="action-alert">
            <Icon name="bell" className="h-5 w-5" /> Set an alert
          </Link>
          <p className="today-note mt-1.5" data-testid="readiness-alerts">
            {pilot ? alertWords(readiness) : "Alerts cover the pilot ports for now: Moss Landing, Monterey, Santa Cruz and Half Moon Bay."}
          </p>
        </div>
        <div>
          <a
            href={`/ask?${q}`}
            onClick={(e) => {
              if (!pilot) return;
              e.preventDefault();
              onAsk();
            }}
            aria-haspopup={pilot ? "dialog" : undefined}
            className="today-act nx-focus nx-press"
            data-testid="action-ask"
          >
            <Icon name="ask" className="h-5 w-5" /> Ask Neritic
          </a>
          <p className="today-note mt-1.5" data-testid="readiness-ask">
            {pilot ? `${READINESS_WORDS[readiness.ask]}: answers from the facts on this page, each with its source.` : "Ask Neritic answers for the pilot ports for now."}
          </p>
        </div>
      </div>
      <div className="mt-6 flex flex-wrap items-center gap-x-6 gap-y-2">
        <Link href="/sources" className="nx-link nx-focus inline-flex min-h-11 items-center text-[15px]">
          Sources and data status
        </Link>
        {feedbackFact && <Feedback factId={feedbackFact} placeId={place} compact />}
      </div>
    </Chapter>
  );
}
