"use client";

import { useEffect, useMemo, useState, useSyncExternalStore } from "react";
import type { SupabaseClient, User } from "@supabase/supabase-js";
import { readInbox, type InboxItem } from "@/pilot/alerts/inbox";
import { EMPTY_ACCOUNT, readAccount, type LocalAccount } from "./local";
import { persistent, storeVersion, subscribe } from "./storage";
import { browserClient, type SupabaseConfig } from "./supabase";

/** -1 on the server and during hydration: callers render their empty state, then the device's. */
function useStoreVersion(): number {
  return useSyncExternalStore(subscribe, storeVersion, () => -1);
}

export function useAccount(): { account: LocalAccount; hydrated: boolean; persistent: boolean } {
  const v = useStoreVersion();
  return useMemo(() => (v < 0 ? { account: EMPTY_ACCOUNT, hydrated: false, persistent: true } : { account: readAccount(), hydrated: true, persistent: persistent() }), [v]);
}

export function useDeviceInbox(): InboxItem[] {
  const v = useStoreVersion();
  return useMemo(() => (v < 0 ? [] : readInbox()), [v]);
}

export type Session = { client: SupabaseClient | null; user: User | null; status: "off" | "loading" | "signed_out" | "signed_in" };

/** The Supabase session when accounts are configured; status "off" when they are not. */
export function useSession(cfg: SupabaseConfig): Session {
  const client = useMemo(() => browserClient(cfg), [cfg]);
  const [state, setState] = useState<Omit<Session, "client">>({ user: null, status: client ? "loading" : "off" });
  useEffect(() => {
    if (!client) return;
    let live = true;
    client.auth
      .getUser()
      .then(({ data }) => live && setState({ user: data.user, status: data.user ? "signed_in" : "signed_out" }))
      .catch(() => live && setState({ user: null, status: "signed_out" }));
    const { data } = client.auth.onAuthStateChange((_e, s) => setState({ user: s?.user ?? null, status: s?.user ? "signed_in" : "signed_out" }));
    return () => {
      live = false;
      data.subscription.unsubscribe();
    };
  }, [client]);
  return { client, ...state };
}
