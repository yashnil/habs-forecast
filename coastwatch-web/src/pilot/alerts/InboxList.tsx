"use client";

import type { InboxItem } from "./inbox";
import { Feedback } from "@/pilot/account/Feedback";
import { TRIGGER_LABELS } from "./diff";

const DOT: Record<InboxItem["severity"], string> = {
  red: "bg-n-down",
  amber: "bg-n-stale",
  neutral: "bg-n-ink-3",
  info: "bg-n-ink-3",
};

const when = (iso: string) => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZone: "America/Los_Angeles" });
};

/** The alert inbox: newest first, each item with its trigger, source and date. Never says "safe". */
export function InboxList({ items, compact = false, empty }: { items: InboxItem[]; compact?: boolean; empty?: React.ReactNode }) {
  if (!items.length)
    return (
      <p className="py-3 text-[14.5px] leading-relaxed text-n-ink-2" data-testid="inbox-empty">
        {empty ?? "No alerts yet. When something you follow changes at a saved port, it is listed here."}
      </p>
    );
  return (
    <ul className="divide-y divide-n-line" data-testid="inbox-list">
      {items.map((i) => (
        <li key={i.key} className="flex gap-3 py-3" data-testid="inbox-item" data-trigger={i.trigger} data-read={i.read_at ? "1" : "0"}>
          <span aria-hidden className={`mt-[7px] h-2 w-2 shrink-0 rounded-full ${DOT[i.severity]}`} />
          <div className="min-w-0 flex-1">
            <p className={`text-[15px] leading-snug text-n-ink ${i.read_at ? "" : "font-semibold"}`}>
              {!i.read_at && <span className="sr-only">Unread: </span>}
              {i.title}
            </p>
            {!compact && <p className="mt-1 text-[14px] leading-relaxed text-n-ink-2">{i.body}</p>}
            <p className="mt-1 text-[12.5px] text-n-ink-2">
              {TRIGGER_LABELS[i.trigger]?.label ?? i.trigger} · {when(i.created_at)}
              {i.origin === "account" ? " · from your account" : " · on this device"}
              {i.sample && (
                <span className="text-n-stale" data-testid="inbox-sample">
                  {" "}
                  · Sample data (test fixture), not live
                </span>
              )}
              {i.source?.url && !compact && (
                <>
                  {" · "}
                  <a href={i.source.url} target="_blank" rel="noreferrer" className="nx-focus text-n-accent underline underline-offset-2">
                    Source
                  </a>
                </>
              )}
            </p>
            {!compact && (
              <div className="mt-1">
                <Feedback factId={i.fact_id} placeId={i.place_id} />
              </div>
            )}
          </div>
        </li>
      ))}
    </ul>
  );
}
