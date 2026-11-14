import type { PilotFeature, Readiness } from "./contracts";

/**
 * Honest readiness labels for pilot features, decided from configuration only (never claimed).
 * Server-side: reads env. The web shows the label next to every pilot feature that is not "live".
 *  - accounts / alerts: NEXT_PUBLIC_SUPABASE_URL + NEXT_PUBLIC_SUPABASE_ANON_KEY (local stack = "demo")
 *  - web_push: NEXT_PUBLIC_VAPID_PUBLIC_KEY (+ server VAPID_PRIVATE_KEY)
 *  - ask: ANTHROPIC_API_KEY ("demo" = deterministic, grounded answers without a model)
 *  - sms: TWILIO_* + NERITIC_SMS_MODE=live (trial accounts / verified numbers only = "demo")
 */
export function readiness(feature: PilotFeature, env: Record<string, string | undefined> = process.env): Readiness {
  switch (feature) {
    case "accounts":
    case "alerts": {
      const url = env.NEXT_PUBLIC_SUPABASE_URL;
      if (!url || !env.NEXT_PUBLIC_SUPABASE_ANON_KEY) return "not_configured";
      return /127\.0\.0\.1|localhost/.test(url) ? "demo" : "live";
    }
    case "web_push":
      return env.NEXT_PUBLIC_VAPID_PUBLIC_KEY && env.VAPID_PRIVATE_KEY ? "live" : "not_configured";
    case "ask":
      return env.ANTHROPIC_API_KEY ? "live" : "demo";
    case "sms":
      if (!env.TWILIO_ACCOUNT_SID || !env.TWILIO_AUTH_TOKEN) return "not_configured";
      return env.NERITIC_SMS_MODE === "live" ? "live" : "demo";
  }
}

export const READINESS_WORDS: Record<Readiness, string> = {
  live: "Live",
  demo: "Pilot demo",
  not_configured: "Not yet connected",
};
