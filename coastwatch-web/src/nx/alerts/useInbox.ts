"use client";

import { useMemo, useSyncExternalStore } from "react";
import { readRaw, subscribe } from "@/nx/personal/store";
import { FOLLOW_KEY, readFollows, type Follow } from "./follow";
import { CHECK_KEY, INBOX_KEY, clearInbox, markRead, readInbox, readStamp, unreadCount, type CheckStamp, type InboxItem } from "./inbox";

/**
 * The alerts, the follows and the last check, as React reads them. Each hook re-renders when its key changes
 * in this tab or another; on the server and during hydration each returns `undefined` (the first client
 * render must match the server's HTML), so a count is never flashed and then corrected.
 */
const raw = (key: string) => () => readRaw(key) ?? "";
const server = () => undefined;

export type Inbox = { items: InboxItem[]; unread: number; markRead: (keys: string[] | "all") => void; clear: () => void };

export function useInbox(): Inbox | undefined {
  const v = useSyncExternalStore(subscribe, raw(INBOX_KEY), server);
  return useMemo(() => {
    if (v === undefined) return undefined;
    const items = readInbox();
    return { items, unread: unreadCount(items), markRead: (keys) => markRead(keys), clear: clearInbox };
  }, [v]);
}

/** Unread alerts on this device, for the bar's Alerts link: undefined until hydrated, then 0 or more. */
export function useUnread(): number | undefined {
  return useInbox()?.unread;
}

export function useFollows(): Follow[] | undefined {
  const v = useSyncExternalStore(subscribe, raw(FOLLOW_KEY), server);
  return useMemo(() => (v === undefined ? undefined : readFollows()), [v]);
}

/** When this device last checked, and which ports it read; null when it never has. */
export function useLastCheck(): CheckStamp | null | undefined {
  const v = useSyncExternalStore(subscribe, raw(CHECK_KEY), server);
  return useMemo(() => (v === undefined ? undefined : readStamp()), [v]);
}
