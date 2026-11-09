"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import type { Session } from "@/pilot/account/hooks";
import { useDeviceInbox } from "@/pilot/account/hooks";
import { fetchInbox, markAccountRead } from "@/pilot/account/sync";
import { markRead, mergeInbox, type InboxItem } from "./inbox";

/** The device inbox, plus the account's server-written alerts when signed in (deduplicated by key). */
export function useInbox(session: Session): { items: InboxItem[]; unread: number; markAllRead: () => void; refresh: () => void } {
  const device = useDeviceInbox();
  const [remote, setRemote] = useState<InboxItem[]>([]);
  const [tick, setTick] = useState(0);
  const { client, status } = session;
  useEffect(() => {
    if (!client || status !== "signed_in") return;
    let live = true;
    fetchInbox(client).then((r) => live && setRemote(r));
    return () => {
      live = false;
    };
  }, [client, status, tick]);
  const items = useMemo(() => mergeInbox(status === "signed_in" ? remote : [], device), [remote, device, status]);
  const unread = items.filter((i) => !i.read_at).length;
  const markAllRead = useCallback(() => {
    markRead("all");
    if (client && status === "signed_in") {
      const ids = remote.filter((i) => !i.read_at && i.account_id).map((i) => i.account_id!);
      void markAccountRead(client, ids).then(() => setTick((t) => t + 1));
    }
  }, [client, status, remote]);
  return { items, unread, markAllRead, refresh: () => setTick((t) => t + 1) };
}
