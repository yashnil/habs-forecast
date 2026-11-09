"use client";

import { createBrowserClient } from "@supabase/ssr";
import type { SupabaseClient } from "@supabase/supabase-js";

/**
 * The browser Supabase client (anon key + the signed-in person's session cookie, via @supabase/ssr).
 * The config comes from the server page at request time (so adding env vars needs no rebuild); null
 * means accounts are not configured and everything stays on this device.
 */
export type SupabaseConfig = { url: string; anonKey: string } | null;

let cached: { key: string; client: SupabaseClient } | null = null;

export function browserClient(cfg: SupabaseConfig): SupabaseClient | null {
  if (!cfg) return null;
  const key = `${cfg.url}|${cfg.anonKey}`;
  if (cached?.key === key) return cached.client;
  try {
    const client = createBrowserClient(cfg.url, cfg.anonKey);
    cached = { key, client };
    return client;
  } catch {
    return null;
  }
}
