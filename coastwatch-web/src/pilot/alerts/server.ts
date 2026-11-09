import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import webpush from "web-push";
import type { AlertItem } from "./diff";
import type { FactStateRow, FanoutStore, PushSender, PushTarget, Subscription } from "./fanout";

/**
 * Server-only pieces of the alert fan-out: the service-role Supabase store and the web-push sender.
 * Imported only by route handlers (src/app/api/alerts/**). The service role key and the VAPID
 * private key never reach the browser (they are not NEXT_PUBLIC_*).
 */
type Env = Record<string, string | undefined>;

export function serviceClient(env: Env = process.env): SupabaseClient | null {
  const url = env.NEXT_PUBLIC_SUPABASE_URL;
  const key = env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !key) return null;
  return createClient(url, key, { auth: { persistSession: false, autoRefreshToken: false } });
}

function check<T>(r: { data: T; error: { message: string } | null }, what: string): T {
  if (r.error) throw new Error(`${what}: ${r.error.message}`);
  return r.data;
}

export class SupabaseFanoutStore implements FanoutStore {
  constructor(private db: SupabaseClient) {}

  async activeSubscriptions(): Promise<Subscription[]> {
    return check(await this.db.from("alert_subscriptions").select("id, user_id, place_id, triggers, channel").eq("active", true), "subscriptions") as Subscription[];
  }

  async factState(ids: string[]): Promise<FactStateRow[]> {
    if (!ids.length) return [];
    const rows: FactStateRow[] = [];
    for (let i = 0; i < ids.length; i += 200) {
      const part = check(await this.db.from("alert_fact_state").select("subscription_id, fact_id, fact_version, fact, seen_at").in("subscription_id", ids.slice(i, i + 200)), "fact state");
      rows.push(...(part as FactStateRow[]));
    }
    return rows;
  }

  async claim(sub: Subscription, a: AlertItem): Promise<boolean> {
    const r = await this.db
      .from("notifications_sent")
      .upsert({ subscription_id: sub.id, fact_id: a.fact_id, fact_version: a.fact_version, user_id: sub.user_id, channel: sub.channel }, { onConflict: "subscription_id,fact_id,fact_version", ignoreDuplicates: true })
      .select("subscription_id");
    return (check(r, "claim") ?? []).length > 0;
  }

  async insertAlert(sub: Subscription, a: AlertItem): Promise<void> {
    check(
      await this.db.from("alerts").insert({
        user_id: sub.user_id,
        subscription_id: sub.id,
        place_id: a.place_id,
        fact_id: a.fact_id,
        fact_version: a.fact_version,
        trigger: a.trigger,
        severity: a.severity,
        title: a.title.slice(0, 300),
        body: a.body.slice(0, 2000),
        basis: a.basis,
        verified: a.verified,
        source_name: a.source?.name ?? null,
        source_url: a.source?.url ?? null,
        as_of: a.as_of,
      }),
      "alert",
    );
  }

  async saveFactState(rows: (FactStateRow & { user_id: string })[]): Promise<void> {
    if (rows.length) check(await this.db.from("alert_fact_state").upsert(rows, { onConflict: "subscription_id,fact_id" }), "save state");
  }

  async pushTargets(userId: string): Promise<PushTarget[]> {
    return check(await this.db.from("push_subscriptions").select("endpoint, p256dh, auth").eq("user_id", userId), "push targets") as PushTarget[];
  }

  async forgetPushTarget(endpoint: string): Promise<void> {
    check(await this.db.from("push_subscriptions").delete().eq("endpoint", endpoint), "forget push");
  }
}

/** A web-push sender, or null when VAPID keys are not configured (readiness "web_push" ≠ live). */
export function pushSender(env: Env = process.env): PushSender | null {
  const pub = env.NEXT_PUBLIC_VAPID_PUBLIC_KEY;
  const priv = env.VAPID_PRIVATE_KEY;
  if (!pub || !priv) return null;
  webpush.setVapidDetails(env.VAPID_SUBJECT ?? "mailto:owner@example.invalid", pub, priv);
  return async (t, payload) => {
    try {
      await webpush.sendNotification({ endpoint: t.endpoint, keys: { p256dh: t.p256dh, auth: t.auth } }, JSON.stringify(payload), { TTL: 6 * 3600, urgency: "normal" });
      return "sent";
    } catch (e) {
      const code = (e as { statusCode?: number }).statusCode;
      return code === 404 || code === 410 ? "gone" : "failed";
    }
  };
}
