import type { PilotFeature, Readiness } from "@/pilot/contracts";
import { readiness } from "@/pilot/readiness";

/**
 * Server side: what the account features may do in this deployment, decided from env at request time
 * (adding the env vars makes them live without a rebuild). Only public values leave the server: the
 * Supabase URL and anon key (protected by row-level security) and the VAPID public key.
 */
export type PilotAccountConfig = {
  supabase: { url: string; anonKey: string } | null;
  vapidPublicKey: string | null;
  readiness: Record<Exclude<PilotFeature, "ask">, Readiness>;
};

export function pilotAccountConfig(env: Record<string, string | undefined> = process.env): PilotAccountConfig {
  const accounts = readiness("accounts", env);
  const push = readiness("web_push", env);
  return {
    supabase: accounts !== "not_configured" ? { url: env.NEXT_PUBLIC_SUPABASE_URL!, anonKey: env.NEXT_PUBLIC_SUPABASE_ANON_KEY! } : null,
    // web push needs an account to deliver to (the server sends it); without accounts it stays off
    vapidPublicKey: push === "live" && accounts !== "not_configured" ? env.NEXT_PUBLIC_VAPID_PUBLIC_KEY! : null,
    readiness: { accounts, alerts: readiness("alerts", env), web_push: accounts === "not_configured" ? "not_configured" : push, sms: readiness("sms", env) },
  };
}

export const NOT_CONFIGURED: PilotAccountConfig = {
  supabase: null,
  vapidPublicKey: null,
  readiness: { accounts: "not_configured", alerts: "not_configured", web_push: "not_configured", sms: "not_configured" },
};
