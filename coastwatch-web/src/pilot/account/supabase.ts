"use client";

import type { SupabaseClient } from "@supabase/supabase-js";

/**
 * The browser Supabase client (anon key + the signed-in person's session cookie, via @supabase/ssr).
 * The config comes from the server page at request time (so adding env vars needs no rebuild); null
 * means accounts are not configured and everything stays on this device.
 *
 * The library is loaded only when accounts are configured, and only then: the alerts bell sits in the
 * top bar of every page (Explore, My Coast, Research), and none of them should carry Supabase's code
 * for a session that does not exist.
 */
export type SupabaseConfig = { url: string; anonKey: string } | null;

let cached: { key: string; client: Promise<SupabaseClient | null> } | null = null;

export function browserClient(cfg: SupabaseConfig): Promise<SupabaseClient | null> {
  if (!cfg) return Promise.resolve(null);
  const key = `${cfg.url}|${cfg.anonKey}`;
  if (cached?.key === key) return cached.client;
  const client = import("@supabase/ssr").then(
    ({ createBrowserClient }) => {
      try {
        return createBrowserClient(cfg.url, cfg.anonKey) as SupabaseClient;
      } catch {
        return null;
      }
    },
    () => null,
  );
  cached = { key, client };
  return client;
}
