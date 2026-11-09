"use client";

import Link from "next/link";
import { useEffect, useId, useRef, useState } from "react";
import type { PortToday } from "@/pilot/contracts";
import { usePilotAccount } from "@/pilot/account/context";
import { useAccount } from "@/pilot/account/hooks";
import { InboxList } from "./InboxList";
import { runLocalCheck } from "./inbox";
import { useInbox } from "./useInbox";

/**
 * The alerts bell for the top bar: unread count, and the newest alerts in a small panel with a link
 * to the full inbox (/account#inbox). Works with no account (the device inbox).
 *
 * Mounted in the top bar (src/v4/shell/TopBar.tsx), next to the ⋯ menu, on every page. Today runs the
 * device's alert check itself (<AlertCheck>); `todays` lets another page do it through the bell.
 */
export function AlertsBell({ todays }: { todays?: PortToday[] }) {
  const { session } = usePilotAccount();
  const { account, hydrated } = useAccount();
  const { items, unread, markAllRead } = useInbox(session);
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement | null>(null);
  const panelId = useId();

  const facts = (todays ?? []).map((t) => `${t.place_id}@${t.generated_at}`).join(",");
  useEffect(() => {
    if (hydrated && todays?.length) runLocalCheck(account, todays);
    // the check is idempotent; run it when the facts arrive or the saved ports change
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hydrated, facts, account.saved_places.join(","), JSON.stringify(account.alerts)]);

  useEffect(() => {
    if (!open) return;
    const k = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    const c = (e: MouseEvent) => box.current && !box.current.contains(e.target as Node) && setOpen(false);
    window.addEventListener("keydown", k);
    window.addEventListener("mousedown", c);
    return () => {
      window.removeEventListener("keydown", k);
      window.removeEventListener("mousedown", c);
    };
  }, [open]);

  const label = unread ? `Alerts, ${unread} unread` : "Alerts";
  return (
    <div ref={box} className="relative">
      <button
        type="button"
        aria-label={label}
        aria-expanded={open}
        aria-controls={open ? panelId : undefined}
        onClick={() => setOpen((o) => !o)}
        className="nx-press nx-focus nx-tint relative grid h-11 w-11 place-items-center rounded-xl text-n-ink-2"
        data-testid="alerts-bell"
      >
        <svg aria-hidden viewBox="0 0 24 24" className="h-[22px] w-[22px]" fill="none" stroke="currentColor" strokeWidth={1.7} strokeLinecap="round" strokeLinejoin="round">
          <path d="M6 9a6 6 0 1 1 12 0c0 4.5 1.5 6.5 2 7H4c.5-.5 2-2.5 2-7Z" />
          <path d="M10 19.5a2 2 0 0 0 4 0" />
        </svg>
        {unread > 0 && (
          <span aria-hidden className="absolute right-1 top-1 grid h-[18px] min-w-[18px] place-items-center rounded-full bg-n-down px-1 text-[11px] font-bold leading-none text-white" data-testid="alerts-unread">
            {unread > 9 ? "9+" : unread}
          </span>
        )}
      </button>
      {open && (
        <div id={panelId} role="dialog" aria-label="Alerts" className="nx-pop absolute right-0 top-[52px] z-50 w-[min(360px,calc(100vw-24px))] rounded-2xl bg-n-surface p-3 shadow-[var(--nx-shadow)] max-sm:-right-11" data-testid="alerts-panel">
          <div className="flex items-center justify-between gap-2 px-1">
            <h2 className="text-[15px] font-semibold text-n-ink">Alerts</h2>
            {unread > 0 && (
              <button type="button" onClick={markAllRead} className="nx-focus min-h-11 rounded-lg px-2 text-[13.5px] font-medium text-n-ink-2 underline decoration-n-line-strong underline-offset-4">
                Mark all read
              </button>
            )}
          </div>
          <div className="max-h-[50vh] overflow-y-auto px-1">
            <InboxList items={items.slice(0, 5)} compact empty={account.saved_places.length ? undefined : "Save a port to get alerts when its official status, toxin tests or whale rules change."} />
          </div>
          <Link href="/account#inbox" onClick={() => setOpen(false)} className="nx-focus mt-1 flex min-h-11 items-center justify-center rounded-lg text-[14px] font-medium text-n-ink underline decoration-n-line-strong underline-offset-4 hover:bg-n-surface-2" data-testid="alerts-open-inbox">
            Open inbox and alert settings
          </Link>
        </div>
      )}
    </div>
  );
}
